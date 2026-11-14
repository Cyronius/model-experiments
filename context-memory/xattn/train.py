#!/usr/bin/env python3
"""Train and evaluate gated cross-attention memory on a frozen Qwen3-4B.

Stages:
  train     train the memory adapter and cross-attention blocks (resumable)
  evaluate  held-out SQuAD v2 QA, reconstruction/continuation NLL, XDNA2 pilot
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import platform
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from common import (HERE, REPO, chat_ids, file_sha256, repo_path, rules, score_qa, target_ids,
                    user_prompt, work_dir)
from model import MemoryModel, init_adapter_from_lclm

KINDS = ("qa", "qa_removed", "recon", "continue")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ---------------------------------------------------------------- loading

def load_model(device, ckpt=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    R = rules()
    path = repo_path(R["decoder"]["path"])
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    lm = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, attn_implementation="sdpa",
                                              local_files_only=True).to(device)
    model = MemoryModel(lm, 1024, R["xattn"])
    init_adapter_from_lclm(model, repo_path(R["encoder"]["checkpoint"]) / "adapter" / "adapter.safetensors")
    model.adapter.to(device)
    model.blocks.to(device)
    if ckpt is not None:
        model.load_trainable(torch.load(ckpt, map_location="cpu", weights_only=True)["model"])
    return model, tok


class Bank:
    """Encoded passages for one split; memory for an example = concatenated passages."""

    def __init__(self, split):
        from safetensors.torch import load_file
        d = work_dir() / "memory"
        meta = json.loads((d / f"{split}_index.json").read_text())
        self.index = meta["index"]
        self.manifest = meta["manifest"]
        self.memory = load_file(d / f"{split}.safetensors")["memory"]
        self.max_slot = rules()["xattn"]["max_slot"]
        texts = json.loads((work_dir() / "examples" / f"passages_{split}.json").read_text())
        self.text = {p: v.get("text") for p, v in texts.items()}

    def get(self, pids):
        chunks, slots = [], []
        for p in pids:
            off, n, _ = self.index[p]
            chunks.append(self.memory[off:off + n])
            slots.append(torch.arange(n) % self.max_slot)
        return torch.cat(chunks), torch.cat(slots)

    def notes(self, pids):
        return "\n\n".join(self.text[p] for p in pids)


def read_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines()]


def encode_example(tok, ex, cache):
    key = (ex["kind"], ex["id"])
    if key not in cache:
        prompt = chat_ids(tok, user_prompt(ex["kind"], ex.get("question")))
        target = ex["target"] if "target" in ex else (ex["answers"][0] if ex["answers"] else "UNKNOWN")
        tgt = target_ids(tok, target)
        cache[key] = (prompt, tgt)
    return cache[key]


def collate(rows, pad_id, device):
    """rows: (prompt_ids, target_ids). Right padding; labels -100 except targets."""
    T = max(len(p) + len(t) for p, t in rows)
    ids = torch.full((len(rows), T), pad_id, dtype=torch.long)
    att = torch.zeros((len(rows), T), dtype=torch.long)
    lab = torch.full((len(rows), T), -100, dtype=torch.long)
    for i, (p, t) in enumerate(rows):
        n = len(p) + len(t)
        ids[i, :n] = torch.tensor(p + t)
        att[i, :n] = 1
        lab[i, len(p):n] = torch.tensor(t)
    return ids.to(device), att.to(device), lab.to(device)


def per_example_loss(model, ids, att, lab):
    """Mean token CE per example (so short QA answers weigh as much as long targets)."""
    h = model.lm.model(input_ids=ids, attention_mask=att, use_cache=False).last_hidden_state
    labels = lab[:, 1:]
    sel = labels != -100
    logits = model.lm.lm_head(h[:, :-1][sel]).float()
    tgt = labels[sel]
    ce = F.cross_entropy(logits, tgt, reduction="none")
    correct = (logits.argmax(-1) == tgt).float()
    rows = sel.nonzero()[:, 0]
    n = torch.zeros(ids.shape[0], device=ids.device).index_add_(0, rows, torch.ones_like(ce))
    loss = torch.zeros(ids.shape[0], device=ids.device).index_add_(0, rows, ce) / n
    acc = torch.zeros(ids.shape[0], device=ids.device).index_add_(0, rows, correct) / n
    return loss, acc, n


# ---------------------------------------------------------------- training

class Sampler:
    def __init__(self, examples, mix, seed):
        self.rng = random.Random(seed)
        self.by_kind = {k: [e for e in examples if e["kind"] == k] for k in KINDS}
        self.pos = {k: 0 for k in KINDS}
        self.order = {k: list(range(len(v))) for k, v in self.by_kind.items()}
        for k in KINDS:
            self.rng.shuffle(self.order[k])
        self.kinds, self.weights = list(mix), [mix[k] for k in mix]

    def next(self):
        k = self.rng.choices(self.kinds, self.weights)[0]
        if self.pos[k] == len(self.order[k]):
            self.rng.shuffle(self.order[k])
            self.pos[k] = 0
        ex = self.by_kind[k][self.order[k][self.pos[k]]]
        self.pos[k] += 1
        return ex

    def state(self):
        return {"rng": self.rng.getstate(), "pos": self.pos, "order": self.order}

    def load(self, s):
        self.rng.setstate(s["rng"])
        self.pos, self.order = s["pos"], s["order"]


def lr_at(step, T):
    if step < T["warmup_steps"]:
        return T["lr"] * (step + 1) / T["warmup_steps"]
    p = (step - T["warmup_steps"]) / max(1, T["max_steps"] - T["warmup_steps"])
    lo = T["lr"] * T["min_lr_ratio"]
    return lo + 0.5 * (T["lr"] - lo) * (1 + math.cos(math.pi * min(1.0, p)))


def save_ckpt(path, model, opt, step, sampler, extra):
    tmp = path.with_suffix(".tmp")
    torch.save({"model": model.trainable_state(), "opt": opt.state_dict() if opt else None,
                "step": step, "sampler": sampler.state() if sampler else None, **extra}, tmp)
    os.replace(tmp, path)


def train(args):
    R = rules()
    T = dict(R["train"])
    if args.max_steps:
        T["max_steps"] = args.max_steps
    if args.step_tokens:  # debugging only; recorded in the manifest
        T["step_tokens"], T["micro_tokens"] = args.step_tokens, min(T["micro_tokens"], args.step_tokens)
    if not T["max_steps"]:
        raise ValueError("Set --max-steps (recorded in the run manifest)")
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    ck_dir = work_dir() / "ckpt" / out.name
    ck_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(R["seed"])
    device = torch.device(args.device)
    model, tok = load_model(device)
    model.lm.config.use_cache = False
    model.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    params = model.trainable()
    for p in params:
        p.data = p.data.float()
    n_train = sum(p.numel() for p in params)
    opt = torch.optim.AdamW(params, lr=T["lr"], betas=(0.9, 0.95), weight_decay=T["weight_decay"])
    bank = Bank(args.split)
    examples = [e for f in args.examples for e in read_jsonl(work_dir() / "examples" / f)]
    if args.limit_examples:
        examples = examples[:args.limit_examples]
    sampler = Sampler(examples, R["data"]["mix"], R["seed"])
    step = 0
    last = ck_dir / "last.pt"
    if last.exists():
        ck = torch.load(last, map_location="cpu", weights_only=False)
        model.load_trainable(ck["model"])
        opt.load_state_dict(ck["opt"])
        sampler.load(ck["sampler"])
        step = ck["step"]
        log("RESUMED at step", step)
    manifest = {
        "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "rules": R, "train": T,
        "trainable_params": n_train, "device": str(device), "torch": torch.__version__,
        "python": platform.python_version(), "examples": {f: file_sha256(work_dir() / "examples" / f) for f in args.examples},
        "train_memory": bank.manifest, "paid_compute_cost_usd": 0,
    }
    (out / f"train_manifest_step{step}.json").write_text(json.dumps(manifest, indent=2) + "\n")
    log(f"trainable params {n_train/1e6:.1f}M; max_steps {T['max_steps']}")
    cache, skipped = {}, 0
    t_ck = time.time()
    probe_items = read_jsonl(work_dir() / "examples" / "eval_qa.jsonl")[:T["probe_items"]]
    eval_bank = Bank("eval") if T["probe_every_steps"] else None
    hist = open(out / "train_log.jsonl", "a")
    while step < T["max_steps"]:
        t0 = time.time()
        for g in opt.param_groups:
            g["lr"] = lr_at(step, T)
        tokens, n_ex, stats = 0, 0, {k: [0.0, 0.0, 0] for k in KINDS}
        # Draw a step's examples, then sort by length so micro-batches pad little.
        drawn = []
        while tokens < T["step_tokens"]:
            ex = sampler.next()
            p, t = encode_example(tok, ex, cache)
            drawn.append((ex, p, t))
            tokens += len(p) + len(t)
        drawn.sort(key=lambda r: len(r[1]) + len(r[2]))
        micro, batch = [], []
        for r in drawn:
            if batch and (len(r[1]) + len(r[2])) * (len(batch) + 1) > T["micro_tokens"]:
                micro.append(batch)
                batch = []
            batch.append(r)
        micro.append(batch)
        n_total = len(drawn)
        padded = 0
        for batch in micro:
            mem = [bank.get(ex["passages"]) for ex, _, _ in batch]
            ids, att, lab = collate([(p, t) for _, p, t in batch], tok.pad_token_id, device)
            padded += ids.numel()
            with torch.autocast(device.type, dtype=torch.bfloat16):
                model.set_memory([m for m, _ in mem], [s for _, s in mem])
                loss, acc, _ = per_example_loss(model, ids, att, lab)
                (loss.sum() / n_total).backward()
            if args.debug_nan:
                bad = [n for n, p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
                if bad or not torch.isfinite(loss).all():
                    log("NONFINITE", json.dumps({"params": bad[:6], "loss": loss.tolist(),
                        "rows": [(ex["kind"], ex["id"], len(p) + len(t), int(m.shape[0]))
                                 for (ex, p, t), (m, _) in zip(batch, mem)]}))
                    raise SystemExit(3)
            for (ex, _, _), l, a in zip(batch, loss.tolist(), acc.tolist()):
                s = stats[ex["kind"]]
                s[0] += l; s[1] += a; s[2] += 1
            n_ex += len(batch)
        model.set_memory(None)
        gnorm = torch.nn.utils.clip_grad_norm_(params, T["grad_clip"]).item()
        if math.isfinite(gnorm):
            opt.step()
            skipped = 0
        else:
            skipped += 1
            log("SKIPPED STEP: non-finite gradient", step + 1)
            if skipped >= 3:
                raise SystemExit("Three consecutive non-finite gradients; stopping")
        opt.zero_grad(set_to_none=True)
        step += 1
        dt_s = time.time() - t0
        row = {"step": step, "lr": opt.param_groups[0]["lr"], "tokens": tokens, "padded_tokens": padded,
               "examples": n_ex, "skipped": not math.isfinite(gnorm),
               "seconds": dt_s, "tok_per_s": tokens / dt_s, "grad_norm": gnorm,
               **{f"loss_{k}": s[0] / s[2] for k, s in stats.items() if s[2]},
               **{f"acc_{k}": s[1] / s[2] for k, s in stats.items() if s[2]}}
        if device.type == "cuda":
            row["max_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
        hist.write(json.dumps(row) + "\n"); hist.flush()
        if step % args.log_every == 0 or step <= 3:
            log("STEP", json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}))
        if T["probe_every_steps"] and step % T["probe_every_steps"] == 0:
            model.eval()
            r = qa_generate(model, tok, eval_bank, probe_items, "memory", device, R["eval"]["max_new_tokens"], 16)
            model.train()
            f1s = [x["f1"] for x in r if x["kind"] == "qa"]
            pr = {"step": step, "probe_qa_f1": sum(f1s) / len(f1s), "probe_items": len(r)}
            hist.write(json.dumps(pr) + "\n"); hist.flush()
            log("PROBE", pr)
        yield_file = os.environ.get("DIBS_YIELD_FILE")
        if yield_file and os.path.exists(yield_file) and step < T["max_steps"]:
            save_ckpt(last, model, opt, step, sampler, {})
            log("YIELD: checkpointed at step", step, "- handing the GPU over")
            sys.exit(75)
        if time.time() - t_ck > T["checkpoint_minutes"] * 60 or step == T["max_steps"]:
            save_ckpt(last, model, opt, step, sampler, {})
            t_ck = time.time()
            log("CHECKPOINT", step)
    save_ckpt(ck_dir / "final.pt", model, None, step, None, {"rules": R, "train": T})
    log("DONE", step)


# ---------------------------------------------------------------- evaluation

@torch.no_grad()
def generate(model, tok, prompts, banks, device, max_new):
    """Greedy, left-padded batch generation. banks None = memory off."""
    T = max(len(p) for p in prompts)
    ids = torch.full((len(prompts), T), tok.pad_token_id, dtype=torch.long)
    att = torch.zeros((len(prompts), T), dtype=torch.long)
    for i, p in enumerate(prompts):
        ids[i, T - len(p):] = torch.tensor(p)
        att[i, T - len(p):] = 1
    with torch.autocast(device.type, dtype=torch.bfloat16):
        model.set_memory(None if banks is None else [b for b, _ in banks],
                         None if banks is None else [s for _, s in banks])
        out = model.lm.generate(input_ids=ids.to(device), attention_mask=att.to(device), max_new_tokens=max_new,
                                do_sample=False, temperature=None, top_p=None, top_k=None,
                                pad_token_id=tok.pad_token_id, use_cache=True)
    model.set_memory(None)
    return [tok.decode(o[T:], skip_special_tokens=True).strip() for o in out]


def qa_generate(model, tok, bank, items, condition, device, max_new, batch):
    model.lm.config.use_cache = True
    rows = []
    for i in range(0, len(items), batch):
        chunk = items[i:i + batch]
        if condition == "full_text":
            prompts = [chat_ids(tok, user_prompt("qa", ex["question"], bank.notes(ex["passages"]))) for ex in chunk]
            banks = None
        else:
            prompts = [chat_ids(tok, user_prompt("qa", ex["question"])) for ex in chunk]
            key = "wrong_passages" if condition == "wrong_memory" else "passages"
            banks = None if condition == "no_memory" else [bank.get(ex[key]) for ex in chunk]
        for ex, ans in zip(chunk, generate(model, tok, prompts, banks, device, max_new)):
            rows.append({"id": ex["id"], "kind": ex["kind"], "condition": condition, "answer": ans,
                         "gold": ex["answers"], **score_qa(ans, ex["answers"])})
    return rows


@torch.no_grad()
def teacher_forced(model, tok, bank, items, condition, device, batch=4):
    rows = []
    for i in range(0, len(items), batch):
        chunk = items[i:i + batch]
        pairs, banks = [], []
        for ex in chunk:
            notes = bank.notes(ex["passages"]) if condition == "full_text" else None
            pairs.append((chat_ids(tok, user_prompt(ex["kind"], notes=notes)), target_ids(tok, ex["target"])))
            if condition in ("memory", "wrong_memory"):
                banks.append(bank.get(ex["wrong_passages"] if condition == "wrong_memory" else ex["passages"]))
        ids, att, lab = collate(pairs, tok.pad_token_id, device)
        with torch.autocast(device.type, dtype=torch.bfloat16):
            model.set_memory([b for b, _ in banks] if banks else None, [s for _, s in banks] if banks else None)
            loss, acc, n = per_example_loss(model, ids, att, lab)
        model.set_memory(None)
        for ex, l, a, k in zip(chunk, loss.tolist(), acc.tolist(), n.tolist()):
            rows.append({"id": ex["id"], "kind": ex["kind"], "condition": condition, "nll": l, "acc": a,
                         "tokens": int(k), "source": ex["passages"][0].split("-")[0]})
    return rows


def xdna(model, tok, bank, device, max_new):
    sys.path.insert(0, str(REPO / "context-memory"))
    from run import score_answer
    cases = json.loads((REPO / "context-memory" / "cases.json").read_text())
    review = json.loads((REPO / "context-memory" / "review.json").read_text())
    pids = {b: sorted((p for p in bank.index if p.startswith(f"xdna-{b}-")), key=lambda p: int(p.split("-")[-1]))
            for b in ("original", "counterfactual")}
    rows = []
    for condition in ("memory", "no_memory", "counterfactual"):
        selected = [c for c in cases if condition != "counterfactual" or c["category"] == "canary"]
        b = "counterfactual" if condition == "counterfactual" else "original"
        mem = None if condition == "no_memory" else bank.get(pids[b])
        prompts = [chat_ids(tok, user_prompt("xdna", c["question"])) for c in selected]
        answers = []
        for c, p in zip(selected, prompts):
            answers += generate(model, tok, [p], None if mem is None else [mem], device, max_new)
        for c, a in zip(selected, answers):
            ok, hits = score_answer(c, condition, a)
            rows.append({"id": c["id"], "category": c["category"], "condition": condition,
                         "answer": a, "correct": ok, "hits": hits})
    return rows, {"memory_slots": int(bank.get(pids["original"])[0].shape[0]),
                  "source_tokens": sum(bank.index[p][2] for p in pids["original"]),
                  "pilot_review_note": review["method"]}


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def evaluate(args):
    R = rules()
    E = R["eval"]
    device = torch.device(args.device)
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    model, tok = load_model(device, args.checkpoint)
    model.eval()
    bank = Bank("eval")
    qa = read_jsonl(work_dir() / "examples" / "eval_qa.jsonl")
    recon = read_jsonl(work_dir() / "examples" / "eval_recon.jsonl")
    cont = read_jsonl(work_dir() / "examples" / "eval_continue.jsonl")
    if args.limit:
        qa, recon, cont = qa[:args.limit], recon[:args.limit], cont[:args.limit]
    env = {"date_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "torch": torch.__version__,
           "device": str(device), "checkpoint": str(args.checkpoint),
           "checkpoint_sha256": file_sha256(args.checkpoint) if args.checkpoint else None,
           "eval_qa_sha256": file_sha256(work_dir() / "examples" / "eval_qa.jsonl"),
           "eval_memory": bank.manifest, "greedy": True, "paid_compute_cost_usd": 0}
    (out / "eval_environment.json").write_text(json.dumps(env, indent=2) + "\n")
    summary = {"checkpoint": str(args.checkpoint)}
    with (out / "qa_answers.jsonl").open("w") as f:
        qa_rows = []
        for c in args.conditions:
            t0 = time.time()
            rows = qa_generate(model, tok, bank, qa, c, device, E["max_new_tokens"], 8 if c == "full_text" else 16)
            for r in rows:
                f.write(json.dumps(r) + "\n")
            qa_rows += rows
            log("QA", c, f"{time.time() - t0:.0f}s")
    for c in args.conditions:
        rs = [r for r in qa_rows if r["condition"] == c]
        summary[f"qa/{c}"] = {
            "answerable_f1": mean(r["f1"] for r in rs if r["kind"] == "qa"),
            "answerable_em": mean(r["em"] for r in rs if r["kind"] == "qa"),
            "answerable_abstained": mean(r["abstained"] for r in rs if r["kind"] == "qa"),
            "unanswerable_abstain": mean(r["abstained"] for r in rs if r["kind"] == "qa_unanswerable"),
            "removed_abstain": mean(r["abstained"] for r in rs if r["kind"] == "qa_removed"),
            "n": len(rs)}
        log("QA", c, json.dumps(summary[f"qa/{c}"]))
    tf_rows = []
    for c in ("memory", "no_memory", "wrong_memory"):
        tf_rows += teacher_forced(model, tok, bank, recon, c, device)
        tf_rows += teacher_forced(model, tok, bank, cont, c, device)
    tf_rows += teacher_forced(model, tok, bank, cont, "full_text", device)
    with (out / "teacher_forced.jsonl").open("w") as f:
        for r in tf_rows:
            f.write(json.dumps(r) + "\n")
    for r in tf_rows:
        key = f"{r['kind']}-{r['source']}/{r['condition']}"
        summary.setdefault(key, {"nll": [], "acc": []})
        summary[key]["nll"].append(r["nll"]); summary[key]["acc"].append(r["acc"])
    for k in [k for k in summary if k.startswith(("recon", "continue"))]:
        summary[k] = {"nll": mean(summary[k]["nll"]), "acc": mean(summary[k]["acc"]), "n": len(summary[k]["acc"])}
        log("TF", k, json.dumps(summary[k]))
    x_rows, x_meta = xdna(model, tok, bank, device, E["xdna_max_new_tokens"])
    with (out / "xdna_answers.jsonl").open("w") as f:
        for r in x_rows:
            f.write(json.dumps(r) + "\n")
    for c in ("memory", "no_memory", "counterfactual"):
        rs = [r for r in x_rows if r["condition"] == c]
        summary[f"xdna/{c}"] = {cat: f"{sum(r['correct'] for r in rs if r['category'] == cat)}/"
                                     f"{sum(1 for r in rs if r['category'] == cat)}"
                                for cat in ("fact", "canary", "absent") if any(r["category"] == cat for r in rs)}
        log("XDNA", c, summary[f"xdna/{c}"])
    summary["xdna_meta"] = x_meta
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "eval.done").touch()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="stage", required=True)
    t = sub.add_parser("train")
    t.add_argument("--output", type=Path, required=True)
    t.add_argument("--device", default="cuda")
    t.add_argument("--max-steps", type=int)
    t.add_argument("--limit-examples", type=int)
    t.add_argument("--log-every", type=int, default=10)
    t.add_argument("--step-tokens", type=int, help="debugging only")
    t.add_argument("--debug-nan", action="store_true")
    t.add_argument("--split", default="train", help="memory bank split (eval only for debugging)")
    t.add_argument("--examples", nargs="+", default=["train.jsonl"])
    e = sub.add_parser("evaluate")
    e.add_argument("--output", type=Path, required=True)
    e.add_argument("--checkpoint", type=Path)
    e.add_argument("--device", default="cuda")
    e.add_argument("--limit", type=int)
    e.add_argument("--conditions", nargs="+", default=rules()["eval"]["conditions"])
    args = ap.parse_args()
    {"train": train, "evaluate": evaluate}[args.stage](args)


if __name__ == "__main__":
    main()
