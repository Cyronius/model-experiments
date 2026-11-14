# Whole-network distillation of a projected student: forward KL to the teacher on fixed corpus text, with checkpoints by token count (2026-10-21).
import argparse
import math
import random
import signal
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from common import _wikitext, build_data, load_model, save_json, _ultrachat_text, set_threads

ap = argparse.ArgumentParser()
ap.add_argument("--teacher", default="Qwen/Qwen3.5-2B")
ap.add_argument("--student", required=True, help="HF folder of the projected student")
ap.add_argument("--out", required=True)
ap.add_argument("--tokens", type=float, default=500e6)
ap.add_argument("--checkpoints", default="5e6,20e6,50e6,100e6,200e6,350e6,500e6")
ap.add_argument("--eval-every", type=float, default=2.5e6)
ap.add_argument("--rows-per-step", type=int, default=64)
ap.add_argument("--micro", type=int, default=8)
ap.add_argument("--chunk", type=int, default=1024, help="tokens per KL chunk (full-vocab fp32 logits)")
ap.add_argument("--lr", type=float, default=2e-5)
ap.add_argument("--warmup", type=int, default=100)
ap.add_argument("--T", type=int, default=512)
ap.add_argument("--corpus", choices=["full", "tiny"], default="full")
ap.add_argument("--rows-cache", default="")
ap.add_argument("--start-tokens", type=int, default=0, help="Already-trained tokens when continuing a weights-only checkpoint")
ap.add_argument("--resume-state", default="", help="resume_latest.pt containing full model, optimizer and progress")
ap.add_argument("--history", default="", help="Previous history.json for a weights-only continuation")
ap.add_argument("--save-every", type=float, default=25e6, help="Save resumable training state every this many tokens")
ap.add_argument("--threads", type=int, default=12)
args = ap.parse_args()
batch_tokens = args.rows_per_step * args.T
resume = torch.load(args.resume_state, map_location="cpu", weights_only=False) if args.resume_state else None
start_tokens = resume["tokens"] if resume else args.start_tokens
if start_tokens < 0 or start_tokens % batch_tokens:
    ap.error("starting token count must be a nonnegative multiple of rows-per-step * T")
if args.tokens <= start_tokens or args.eval_every <= 0 or args.save_every <= 0:
    ap.error("target tokens must exceed starting tokens; eval/save intervals must be positive")
if resume:
    for field in ("teacher", "T", "rows_per_step", "tokens", "lr", "warmup", "corpus"):
        if resume["args"][field] != vars(args)[field]:
            ap.error(f"full-state resume requires the original {field}")
set_threads(args.threads)
dev = "cuda" if torch.cuda.is_available() else "cpu"
amp = dev == "cuda"
T = args.T
t0 = time.time()
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)


def log(msg):
    print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)


teacher, tok = load_model(args.teacher, dtype=torch.bfloat16 if amp else torch.float32)
student, _ = load_model(args.student)
if resume:
    student.load_state_dict(resume["student"])
for p in student.parameters():
    p.requires_grad_(True)
student.train()
log(f"teacher {args.teacher}, student {args.student}: {sum(p.numel() for p in student.parameters()) / 1e9:.3f}B parameters, device {dev}")

# ---------- data ----------
n_rows = int(args.tokens / T) + args.rows_per_step
split = build_data(tok, T=T, n_fit=1500, n_eval=8, n_eval_copy=16, rich=True)
eval_ids = torch.tensor([r[1] for r in split["eval"]])


def chat_rows(n_tokens_target):
    import pandas as pd
    from huggingface_hub import snapshot_download
    path = snapshot_download("HuggingFaceH4/ultrachat_200k", repo_type="dataset", allow_patterns=["data/train_sft-*.parquet"])
    rows, total = [], 0

    def consume(texts):
        nonlocal total
        # Batch encoding preserves conversation order and token boundaries while
        # allowing the fast tokenizer to use the pod's CPU cores.
        for ids in tok(texts, add_special_tokens=False).input_ids:
            rows.extend(ids[i:i + T] for i in range(0, len(ids) - T + 1, T))
            total += len(ids)
            if total >= n_tokens_target:
                return True
        return False

    for f in sorted(Path(path).rglob("train_sft-*.parquet")):
        conversations = pd.read_parquet(f).messages
        log(f"tokenizing {len(conversations)} conversations from {f.name}")
        texts = []
        for index, conv in enumerate(conversations, 1):
            text = tok.apply_chat_template([{"role": m["role"], "content": m["content"]} for m in conv], tokenize=False,
                                           enable_thinking=False)
            texts.append(text)
            if len(texts) == 128:
                if consume(texts):
                    return rows
                texts = []
            if index % 4096 == 0:
                log(f"corpus prep: {index}/{len(conversations)} conversations; {total / 1e6:.1f}M chat tokens")
        if texts:
            if consume(texts):
                return rows
    return rows


def build_rows():
    if args.rows_cache and Path(args.rows_cache).exists():
        return torch.load(args.rows_cache)
    rng = random.Random(0)
    small = [r[1] for r in split["fit"] if r[0] != "copy"]
    wiki = tok(_wikitext("train"), add_special_tokens=False).input_ids
    small += [wiki[i * T:(i + 1) * T] for i in range(len(wiki) // T)]
    if args.corpus == "tiny":
        rows = small[:n_rows]
    else:
        need = int(n_rows * 0.9)
        chat = chat_rows(need * T)
        log(f"unique chat rows {len(chat)}, needed {need} (repeats after one pass)")
        chat = (chat * (need // len(chat) + 1))[:need]
        rng.shuffle(small)
        rows = chat + [small[i % len(small)] for i in range(n_rows - need)]
    rng.shuffle(rows)
    t = torch.tensor(rows[:n_rows])
    if args.rows_cache:
        torch.save(t, args.rows_cache)
    return t


rows = build_rows()
n_rows = len(rows)
steps_total = math.ceil(args.tokens / batch_tokens)
if n_rows < steps_total * args.rows_per_step:
    raise ValueError("cached corpus is shorter than the requested training budget")
log(f"corpus: {n_rows} rows of {T} tokens ({n_rows * T / 1e6:.1f}M); eval rows {len(eval_ids)}")

# ---------- loss ----------
Ht = teacher.model
Hs = student.model


def kl_chunk(hs, ht):
    """KL(teacher || student) summed over a chunk of positions; heads applied here so full-vocab logits stay chunk-sized."""
    with torch.no_grad():
        lt = F.log_softmax(teacher.lm_head(ht).float(), -1)
    ls = F.log_softmax(student.lm_head(hs).float(), -1)
    return (lt.exp() * (lt - ls)).sum()


def batch_loss(ids):
    ids = ids.to(dev)
    with torch.no_grad(), torch.autocast(dev, dtype=torch.bfloat16, enabled=amp):
        ht = Ht(input_ids=ids).last_hidden_state[:, :-1]
    with torch.autocast(dev, dtype=torch.bfloat16, enabled=amp):
        hs = Hs(input_ids=ids).last_hidden_state[:, :-1]
    hs, ht = hs.reshape(-1, hs.shape[-1]), ht.reshape(-1, ht.shape[-1])
    total = 0.0
    for i in range(0, hs.shape[0], args.chunk):
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=amp):
            total = total + checkpoint(kl_chunk, hs[i:i + args.chunk], ht[i:i + args.chunk], use_reentrant=False)
    return total / hs.shape[0]


@torch.no_grad()
def eval_kl():
    student.eval()
    vals = []
    for i in range(0, len(eval_ids), args.micro):
        vals.append(batch_loss(eval_ids[i:i + args.micro]).item())
    student.train()
    return sum(vals) / len(vals)


# ---------- train ----------
opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95))
start_step = start_tokens // batch_tokens
if resume:
    opt.load_state_dict(resume["optimizer"])
    hist = resume["history"]
elif args.history:
    import json
    hist = json.loads(Path(args.history).read_text())
    for key in ("eval", "train"):
        hist[key] = [r for r in hist[key] if r["tokens"] <= start_tokens]
else:
    hist = {"args": vars(args), "eval": [], "train": []}
if start_tokens:
    hist.setdefault("resumes", []).append({"tokens": start_tokens, "args": vars(args),
                                         "optimizer_restored": bool(resume)})
initial_kl = eval_kl()
hist["eval"].append({"tokens": start_tokens, "kl": initial_kl})
log(f"start: held-out KL {initial_kl:.4f}; continuing at {start_tokens:,} tokens toward {args.tokens:,.0f}")
if start_tokens and not resume:
    log("WEIGHTS-ONLY continuation: AdamW state is fresh; global LR schedule and corpus offset are preserved")
marks = sorted(float(x) for x in args.checkpoints.split(",") if start_tokens < float(x) <= args.tokens)
next_eval = (start_tokens // args.eval_every + 1) * args.eval_every
next_save = (start_tokens // args.save_every + 1) * args.save_every
mi = 0
cursor = start_tokens // T
out.mkdir(parents=True, exist_ok=True)
save_json(hist, out / "history.json")
stop_requested = False


def request_stop(signum, frame):
    global stop_requested
    stop_requested = True


signal.signal(signal.SIGTERM, request_stop)
signal.signal(signal.SIGINT, request_stop)


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    return value


def save_resume(tokens, step):
    # Replace only after the entire state has been written successfully.
    state = {"student": cpu_tree(student.state_dict()), "optimizer": cpu_tree(opt.state_dict()),
             "tokens": tokens, "step": step, "cursor": cursor, "args": vars(args), "history": hist,
             "torch_rng": torch.get_rng_state(), "python_rng": random.getstate(),
             "cuda_rng": torch.cuda.get_rng_state_all() if amp else None}
    temporary = out / "resume_latest.pt.tmp"
    torch.save(state, temporary)
    temporary.replace(out / "resume_latest.pt")
    save_json({"tokens": tokens, "step": step, "cursor": cursor}, out / "resume_latest.json")
    save_json(hist, out / "history.json")
    log(f"saved FULL RESUME state at {tokens:,} tokens")


def save_model(tokens, name):
    d = out / name
    # Export bf16 without rounding the live fp32 training parameters in place.
    weights = {k: v.detach().to(device="cpu", dtype=torch.bfloat16) if v.is_floating_point()
               else v.detach().cpu() for k, v in student.state_dict().items()}
    student.save_pretrained(d, state_dict=weights)
    tok.save_pretrained(d)
    save_json({"tokens": tokens, "source_start_tokens": start_tokens}, d / "training_progress.json")
    log(f"saved {d}")


if resume:
    torch.set_rng_state(resume["torch_rng"])
    random.setstate(resume["python_rng"])
    if amp and resume["cuda_rng"] is not None:
        torch.cuda.set_rng_state_all(resume["cuda_rng"])
    del resume
t_step = time.time()
tokens = start_tokens
for step in range(start_step, steps_total):
    lr = args.lr * min(1.0, (step + 1) / args.warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / steps_total)))
    for g in opt.param_groups:
        g["lr"] = lr
    batch = rows[cursor:cursor + args.rows_per_step]
    cursor += args.rows_per_step
    loss_sum = 0.0
    for i in range(0, len(batch), args.micro):
        loss = batch_loss(batch[i:i + args.micro]) * (len(batch[i:i + args.micro]) / len(batch))
        loss.backward()
        loss_sum += loss.item()
    torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
    opt.step()
    opt.zero_grad(set_to_none=True)
    tokens = (step + 1) * batch_tokens
    hist["train"].append({"tokens": tokens, "loss": loss_sum, "lr": lr})
    if step == start_step:
        log(f"first update complete: step {step + 1}, {tokens:,} tokens, train KL {loss_sum:.4f}, lr {lr:.2e}")
    if (step + 1) % 20 == 0:
        rate = 20 * args.rows_per_step * T / (time.time() - t_step)
        t_step = time.time()
        log(f"step {step + 1}/{steps_total}  tokens {tokens / 1e6:.2f}M  train KL {loss_sum:.4f}  lr {lr:.2e}  {rate:.0f} tok/s")
    if tokens >= next_eval:
        next_eval += args.eval_every
        hist["eval"].append({"tokens": tokens, "kl": eval_kl()})
        log(f"held-out KL {hist['eval'][-1]['kl']:.4f} at {tokens / 1e6:.1f}M tokens")
        save_json(hist, out / "history.json")
    hit_mark = mi < len(marks) and tokens >= marks[mi]
    if hit_mark:
        save_model(tokens, f"ckpt_{int(marks[mi] / 1e6)}M")
        mi += 1
    if step == start_step or tokens >= next_save or hit_mark or stop_requested:
        save_resume(tokens, step + 1)
        next_save = (tokens // args.save_every + 1) * args.save_every
    if stop_requested:
        save_model(tokens, f"interrupted_{tokens}")
        log(f"INTERRUPTED safely at {tokens:,} tokens; resume from {out / 'resume_latest.pt'}")
        raise SystemExit(75)
save_model(tokens, f"ckpt_{int(args.tokens / 1e6)}M")
hist["eval"].append({"tokens": tokens, "kl": eval_kl()})
save_resume(tokens, steps_total)
save_json(hist, out / "history.json")
log(f"done: final held-out KL {hist['eval'][-1]['kl']:.4f}")
