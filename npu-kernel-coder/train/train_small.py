"""Full fine-tune of a small Qwen3.5 (0.8B) on tasks.py's npuasm skill tasks.

    ./run_small.sh NAME [flags]                    (dibs gpu hold, preemptible; log runs/NAME/out.log)
    PYTHONPATH=.fla .venv/bin/python train_small.py MODEL_DIR data/tasks/train.jsonl data/tasks/val.jsonl --out runs/NAME

All weights train (fp32 master weights and AdamW state, bf16 autocast), gradient checkpointing,
loss on the answer tokens only (chunked through lm_head: the vocabulary is 248,320). Examples are
batched by length under a token budget (`--batch-tokens`, padded), shuffled by batch. Linear warmup,
cosine to a tenth of `--lr`, clipping at 1.

Shares the GPU like train.py: under `dibs run --preemptible --requeue`, when another session waits
it saves model + optimizer + position and exits 75; dibs reruns it and it resumes. A full save is
~10 GB, so it saves every `--save-min` minutes and on yield, not every step.

Writes to --out: log.jsonl, state/ (resume point), model/ (final weights, HF format), summary.json.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path

YIELD_STATUS = 75
PAD_TO = 128  # batch lengths round up to this, so kernels see few distinct shapes


def causal_conv(x, weight, bias=None, activation=None, seq_idx=None):
    """causal_conv1d_fn's contract: x [B, C, T], weight [C, K]; depthwise, left-padded."""
    import torch.nn.functional as F
    assert seq_idx is None
    K, T = weight.shape[1], x.shape[-1]
    xp = F.pad(x, (K - 1, 0))
    y = xp[..., 0:T] * weight[:, 0, None]
    for k in range(1, K):
        y = y + xp[..., k:k + T] * weight[:, k, None]
    if bias is not None:
        y = y + bias[:, None]
    return F.silu(y) if activation in ("silu", "swish") else y


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", type=Path)
    ap.add_argument("train", type=Path)
    ap.add_argument("val", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch-tokens", type=int, default=16384, help="padded tokens per micro-batch")
    ap.add_argument("--accum", type=int, default=4, help="micro-batches per optimizer step")
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--max-tokens", type=int, default=12288)
    ap.add_argument("--val-per-kind", type=int, default=24)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--save-min", type=float, default=30)
    ap.add_argument("--max-steps", type=int, default=0, help="stop after this many steps (throughput pilot)")
    ap.add_argument("--time-budget-h", type=float, default=0,
                    help="training hours; the schedule runs on max(step, time) progress, so it ends in time")
    ap.add_argument("--require-fla", action="store_true", help="fail unless fla's GatedDeltaNet kernels load")
    ap.add_argument("--seed", type=int, default=20261111)
    ap.add_argument("--mem-floor-gb", type=float, default=12.0)
    ap.add_argument("--mem-cap-gb", type=float, default=40.0)
    a = ap.parse_args()

    import memguard
    memguard.start(a.mem_floor_gb, status=76)
    if os.environ.get("DIBS_YIELD_FILE"):
        Path(os.environ["DIBS_YIELD_FILE"]).unlink(missing_ok=True)
    import torch
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out = a.out
    out.mkdir(parents=True, exist_ok=True)
    log_f = (out / "log.jsonl").open("a")

    def log(**rec):
        rec["time"] = round(time.time(), 1)
        log_f.write(json.dumps(rec) + "\n")
        log_f.flush()
        print(json.dumps(rec), flush=True)

    total_mem = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, a.mem_cap_gb * 2**30 / total_mem))
    tok = AutoTokenizer.from_pretrained(a.model)

    def encode(e):
        msgs, kw = e["messages"], dict(tokenize=False, enable_thinking=False)
        full = tok.apply_chat_template(msgs, **kw)
        prefix = tok.apply_chat_template(msgs[:-1], add_generation_prompt=True, **kw)
        assert full.startswith(prefix), e["id"]
        ids = tok(full, add_special_tokens=False)["input_ids"]
        return ids, len(ids) - len(tok(prefix, add_special_tokens=False)["input_ids"])

    def read(path):
        rows = [json.loads(l) for l in path.open()]
        return [r for r in rows if r["meta"]["tokens"] <= a.max_tokens]

    t0 = time.time()
    train = read(a.train)
    val_all = read(a.val)
    by_kind = collections.defaultdict(list)
    for e in val_all:
        by_kind[e["kind"]].append(e)
    rv = random.Random(a.seed)
    val = [e for k in sorted(by_kind) for e in rv.sample(by_kind[k], min(a.val_per_kind, len(by_kind[k])))]
    enc_train = [encode(e) for e in train]
    enc_val = [encode(e) for e in val]
    log(event="encoded", seconds=round(time.time() - t0, 1), train=len(train), val=len(val))

    def batches(epoch_seed):
        idx = sorted(range(len(train)), key=lambda i: (len(enc_train[i][0]), i))
        out_b, cur, cur_max = [], [], 0
        for i in idx:
            n = -(-len(enc_train[i][0]) // PAD_TO) * PAD_TO
            if cur and max(cur_max, n) * (len(cur) + 1) > a.batch_tokens:
                out_b.append(cur)
                cur, cur_max = [], 0
            cur.append(i)
            cur_max = max(cur_max, n)
        if cur:
            out_b.append(cur)
        random.Random(epoch_seed).shuffle(out_b)
        return out_b

    n_batches = len(batches(0))
    total_steps = math.ceil(n_batches * a.epochs / a.accum)
    if a.max_steps:
        total_steps = min(total_steps, a.max_steps)
    warmup = max(1, round(a.warmup * total_steps))

    def progress():
        p = st["step"] / total_steps
        if a.time_budget_h:
            p = max(p, st.get("train_seconds", 0) / (3600 * a.time_budget_h))
        return p

    def lr_at(step):
        if step < warmup:
            return a.lr * (step + 1) / warmup
        # Past warmup, decay on progress (steps or the time budget, whichever is further along).
        w = warmup / total_steps
        p = min(1.0, max(0.0, (progress() - w) / max(1e-9, 1 - w)))
        return a.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))

    state_dir, new_dir, old_dir = out / "state", out / "state.new", out / "state.old"
    if not state_dir.exists() and old_dir.exists():
        old_dir.rename(state_dir)
    src = state_dir / "model" if state_dir.exists() else a.model
    if a.require_fla:
        import transformers.models.qwen3_5.modeling_qwen3_5 as qm
        assert qm.chunk_gated_delta_rule is not None, "fla kernels missing (torch fallback is ~50x slower)"
    model = AutoModelForCausalLM.from_pretrained(src, dtype=torch.float32).cuda()
    # Without the causal-conv1d package the GatedDeltaNet layers fall back to nn.Conv1d, which on
    # ROCm goes to MIOpen: a kernel search and compile for every new sequence length, and a slow
    # depthwise backward. K shifted multiply-adds compute the same thing.
    patched = 0
    for mod in model.modules():
        if hasattr(mod, "causal_conv1d_fn") and mod.causal_conv1d_fn is None:
            mod.causal_conv1d_fn = causal_conv
            patched += 1
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0, betas=(0.9, 0.95))
    st = {"batch": 0, "step": 0, "micro": 0, "evaluated": -1, "tokens": 0}
    if state_dir.exists():
        st = json.loads((state_dir / "state.json").read_text())
        opt.load_state_dict(torch.load(state_dir / "opt.pt", map_location="cuda"))
        log(event="resume", **st)
    else:
        log(event="start", args={k: str(v) for k, v in vars(a).items()}, train=len(train), val=len(val),
            batches=n_batches, steps=total_steps, warmup=warmup, params=sum(p.numel() for p in params),
            conv_patched=patched)
    log(event="loaded", seconds=round(time.time() - t0, 1), gb=round(torch.cuda.memory_allocated() / 2**30, 1))
    last_save = time.time()

    def save():
        nonlocal last_save
        if new_dir.exists():
            shutil.rmtree(new_dir)
        new_dir.mkdir()
        model.save_pretrained(new_dir / "model", safe_serialization=True)
        torch.save(opt.state_dict(), new_dir / "opt.pt")
        (new_dir / "state.json").write_text(json.dumps(st) + "\n")
        if state_dir.exists():
            state_dir.rename(old_dir)
        new_dir.rename(state_dir)
        if old_dir.exists():
            shutil.rmtree(old_dir)
        last_save = time.time()

    def yield_requested():
        f = os.environ.get("DIBS_YIELD_FILE")
        return bool(f) and os.path.exists(f)

    def hand_over():
        save()
        log(event="yield", **st)
        log_f.close()
        os._exit(YIELD_STATUS)

    def loss_of(items):
        """items: [(ids, n_answer)] -> (sum of answer-token losses, answer tokens)."""
        T = -(-max(len(ids) for ids, _ in items) // PAD_TO) * PAD_TO
        inp = torch.full((len(items), T), tok.pad_token_id or 0, dtype=torch.long)
        tgt = torch.full((len(items), T), -100, dtype=torch.long)
        att = torch.zeros((len(items), T), dtype=torch.long)
        for b, (ids, n) in enumerate(items):
            inp[b, :len(ids)] = torch.tensor(ids)
            att[b, :len(ids)] = 1
            tgt[b, len(ids) - n - 1: len(ids) - 1] = torch.tensor(ids[len(ids) - n:])
        inp, tgt, att = inp.cuda(), tgt.cuda(), att.cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hid = model.model(input_ids=inp, attention_mask=att).last_hidden_state
        sel = tgt != -100
        h, t = hid[sel], tgt[sel]

        def piece(hh, tt):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = model.lm_head(hh)
            return F.cross_entropy(lg.float(), tt, reduction="sum")

        total = h.new_zeros((), dtype=torch.float32)
        for i in range(0, len(t), 2048):
            total = total + checkpoint(piece, h[i:i + 2048], t[i:i + 2048], use_reentrant=False)
        return total, len(t)

    def evaluate():
        model.eval()
        sums, cnt = collections.Counter(), collections.Counter()
        t = time.time()
        with torch.no_grad():
            for e, it in zip(val, enc_val):
                s, n = loss_of([it])
                sums[e["kind"]] += s.item()
                cnt[e["kind"]] += n
        model.train()
        per = {k: round(sums[k] / cnt[k], 4) for k in sorted(cnt)}
        return {"mean": round(sum(per.values()) / len(per), 4), "by_kind": per, "seconds": round(time.time() - t, 1)}

    model.train()
    epoch_batches = batches(a.seed)
    mark = time.time()
    while st["step"] < total_steps and progress() < 1:
        if st["micro"] == 0 and st["evaluated"] < st["step"] and st["step"] % a.eval_every == 0:
            ev = evaluate()
            st["evaluated"] = st["step"]
            log(event="eval", step=st["step"], **ev)
        ep, pos = divmod(st["batch"], n_batches)
        if pos == 0 and st["batch"]:
            pass
        bl = batches(a.seed + ep)[pos] if ep else epoch_batches[pos]
        t = time.time()
        s, n = loss_of([enc_train[i] for i in bl])
        (s / n / a.accum).backward()
        torch.cuda.synchronize()
        st["batch"] += 1
        st["micro"] += 1
        st["tokens"] += sum(len(enc_train[i][0]) for i in bl)
        st["train_seconds"] = st.get("train_seconds", 0) + time.time() - mark  # wall, evals included
        mark = time.time()
        if st["micro"] == a.accum:
            lr = lr_at(st["step"])
            for g in opt.param_groups:
                g["lr"] = lr
            norm = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
            opt.step()
            opt.zero_grad(set_to_none=True)
            st["step"] += 1
            st["micro"] = 0
            log(event="step", step=st["step"], lr=round(lr, 8), grad_norm=round(norm, 3), loss=round(s.item() / n, 4),
                batch=len(bl), tokens=st["tokens"], seconds=round(time.time() - t, 2),
                peak_gb=round(torch.cuda.max_memory_allocated() / 2**30, 1))
            torch.cuda.reset_peak_memory_stats()
            if yield_requested():
                hand_over()
            if time.time() - last_save > 60 * a.save_min:
                save()

    ev = evaluate()
    log(event="eval", step=st["step"], **ev)
    if a.save_min < 1e6:
        save()
    model.to(torch.bfloat16).save_pretrained(out / "model", safe_serialization=True)  # released dtype
    tok.save_pretrained(out / "model")
    evals = [json.loads(l) for l in (out / "log.jsonl").open()]
    (out / "summary.json").write_text(json.dumps({
        "steps": st["step"], "tokens": st["tokens"], "train_examples": len(train),
        "val_loss": [(r["step"], r["mean"], r["by_kind"]) for r in evals if r.get("event") == "eval"]}, indent=1) + "\n")
    log(event="done", steps=st["step"])


if __name__ == "__main__":
    sys.exit(main())
