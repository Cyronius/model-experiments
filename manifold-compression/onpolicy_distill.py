# On-policy distillation: the student writes its own chat answers, the teacher scores every token, the student moves toward the teacher on its own text (2026-10-19).
import argparse
import json
import random
import re
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

from common import KLAccumulator, Runner, add_session_args, kl_per_token, load_model, save_json, session_from

ap = argparse.ArgumentParser()
add_session_args(ap, bs=8)
ap.add_argument("--student", required=True)
ap.add_argument("--tokens", type=float, default=5e6, help="completion tokens to train on")
ap.add_argument("--rollouts", type=int, default=128, help="student answers per step")
ap.add_argument("--max-new", type=int, default=256)
ap.add_argument("--teacher-max-new", type=int, default=768, help="long enough that most teacher answers end, so the student sees how answers stop")
ap.add_argument("--micro", type=int, default=4)
ap.add_argument("--lr", type=float, default=1e-5)
ap.add_argument("--think-frac", type=float, default=0.3, help="share of prompts answered with thinking on")
ap.add_argument("--teacher-frac", type=float, default=0.5, help="share of each step's answers written by the teacher (trained with forward KL)")
ap.add_argument("--score-first", type=int, default=96, help="score only this many leading tokens of the student's own answers")
ap.add_argument("--checkpoints", default="0.5e6,1e6,2e6,5e6")
ap.add_argument("--n-check", type=int, default=64, help="held-out prompts in each health check")
ap.add_argument("--check-max", type=int, default=400)
ap.add_argument("--out", required=True)
ap.add_argument("--tag", default="")
args = ap.parse_args()

S = session_from(args)
S.eval_batches()
teacher, tok, dev = S.model.to(torch.bfloat16), S.tok, S.device
tok.truncation_side = "left"
student, _ = load_model(args.student)
for p in student.parameters():
    p.requires_grad_(True)
student.train()
sr = Runner(student)
t0 = time.time()
STOP = [tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id]


def log(msg):
    print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)


def ultrachat(split):
    from huggingface_hub import snapshot_download
    # One shard is plenty of prompts; the local-first helper would stop at a cache that only holds the test split.
    path = snapshot_download("HuggingFaceH4/ultrachat_200k", repo_type="dataset", allow_patterns=[f"data/{split}-00000-*.parquet"])
    df = pd.concat(pd.read_parquet(f) for f in sorted(Path(path).rglob(f"{split}-00000-*.parquet")))
    return [conv[0]["content"][:1500] for conv in df.messages]


# Training prompts from the train split; health checks from test prompts the calibration set never used (it took the first 192).
train_prompts = ultrachat("train_sft")
random.Random(0).shuffle(train_prompts)
check_prompts = ultrachat("test_sft")[1000:1000 + args.n_check]


def render(p, think):
    return tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True, enable_thinking=think)


def repeat_share(text, n=4):
    w = re.findall(r"\w+", text.lower())
    g = [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
    return 1 - len(set(g)) / len(g) if g else 0.0


@torch.no_grad()
def generate(prompts, think, sample, max_new, model=None):
    model = model or student
    tok.padding_side = "left"
    enc = tok([render(p, think) for p in prompts], return_tensors="pt", padding=True, truncation=True, max_length=512).to(dev)
    student.eval()
    g = model.generate(**enc, max_new_tokens=max_new, do_sample=sample, temperature=1.0 if sample else None, top_p=None,
                         top_k=None, eos_token_id=STOP, pad_token_id=tok.pad_token_id)
    student.train()
    return enc, g


@torch.no_grad()
def health():
    """Greedy answers to held-out prompts: how often they run to the cap and how often they repeat themselves."""
    capped = looped = 0
    samples = []
    for i in range(0, len(check_prompts), 32):
        enc, g = generate(check_prompts[i:i + 32], False, False, args.check_max)
        for row in g[:, enc.input_ids.shape[1]:]:
            ended = any(t in STOP for t in row.tolist())
            text = tok.decode(row, skip_special_tokens=True)
            capped += not ended
            looped += repeat_share(text) > 0.15
            samples.append(text)
    acc = KLAccumulator()
    for b in S.eval_batches():
        emb, ctx = sr.prepare(b["ids"])
        logits, _ = sr.run(emb, 0, ctx, positions=b["pos"], keep_first=False)
        acc.add(kl_per_token(b["ref_lp"].to(dev), logits), b["doms"], b["focus"])
    n = len(check_prompts)
    return {"cap_rate": capped / n, "loop_rate": looped / n, "kl": acc.report()["all"]["mean"], "sample": samples[0][:600]}


opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=0.0)
checkpoints = sorted(float(x) for x in args.checkpoints.split(","))
seen, step, cursor = 0, 0, 0
hist = {"args": vars(args), "health": [{"tokens": 0, **health()}], "steps": []}
log(f"start: {hist['health'][0]}")
while seen < args.tokens:
    batch = train_prompts[cursor:cursor + args.rollouts]
    cursor += args.rollouts
    think = random.random() < args.think_frac
    n_t = int(len(batch) * args.teacher_frac)
    # Teacher-written answers show how a good answer goes and ends; the student's own answers show where it drifts.
    # A teacher reading a looping prefix predicts more loop, so student answers are scored only on their opening tokens.
    parts = []
    if n_t:
        parts.append((generate(batch[:n_t], think, True, args.teacher_max_new, teacher), "forward", args.teacher_max_new))
    if n_t < len(batch):
        parts.append((generate(batch[n_t:], think, True, args.max_new), "reverse", args.score_first))
    opt.zero_grad()
    tot, n_all = 0.0, 0.0
    staged = []
    for (enc, g), kind, limit in parts:
        P = enc.input_ids.shape[1]
        attn = torch.cat([enc.attention_mask, torch.ones_like(g[:, P:])], 1)
        comp = g[:, P:]
        is_stop = torch.zeros_like(comp, dtype=torch.bool)
        for t in STOP:
            is_stop |= comp == t
        after = (is_stop.cumsum(1) - is_stop.long()) > 0
        valid = (~after).float()
        attn[:, P:] = valid.long()
        valid[:, limit:] = 0
        staged.append((g, attn, valid, P, kind))
        n_all += valid.sum().item()
    for g, attn, valid, P, kind in staged:
        keep = g.shape[1] - P + 1
        for i in range(0, len(g), args.micro):
            ids, am, v = g[i:i + args.micro], attn[i:i + args.micro], valid[i:i + args.micro]
            with torch.no_grad():
                t_lp = F.log_softmax(teacher(input_ids=ids, attention_mask=am, logits_to_keep=keep).logits[:, :-1].float(), -1)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                s_logits = student(input_ids=ids, attention_mask=am, logits_to_keep=keep).logits[:, :-1]
            s_lp = F.log_softmax(s_logits.float(), -1)
            if kind == "reverse":
                kl = (s_lp.exp() * (s_lp - t_lp)).sum(-1)
            else:
                kl = (t_lp.exp() * (t_lp - s_lp)).sum(-1)
            loss = (kl * v).sum() / n_all
            loss.backward()
            tot += loss.item()
            del t_lp, s_logits, s_lp, kl
    n_valid = torch.tensor(n_all)
    torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
    opt.step()
    seen += int(n_valid.item())
    step += 1
    lens = {kind: (g.shape[1] - P) for g, _, _, P, kind in staged}
    hist["steps"].append({"step": step, "tokens": seen, "kl": tot, "think": think, "gen_len": lens})
    if step % 5 == 0:
        log(f"step {step}: {seen / 1e6:.2f}M tokens, KL {tot:.4f}, generated lengths {lens}")
    while checkpoints and seen >= checkpoints[0]:
        c = checkpoints.pop(0)
        h = {"tokens": seen, **health()}
        hist["health"].append(h)
        log(f"CHECK {seen / 1e6:.2f}M: cap {h['cap_rate']:.2f}  loop {h['loop_rate']:.2f}  KL {h['kl']:.4f}")
        d = f"{args.out}-{c / 1e6:g}M"
        student.save_pretrained(d, state_dict={k: v.to(torch.bfloat16) for k, v in student.state_dict().items()})
        tok.save_pretrained(d)
        save_json(hist, S.results / f"onpolicy{args.tag}.json")
save_json(hist, S.results / f"onpolicy{args.tag}.json")
log("DISTILL DONE")
