"""LoRA SFT of Qwen3.6-35B-A3B (4-bit routed experts, q4moe.py) on make_sft.py's data.

    ./run_train.sh NAME [train.py flags]          (dibs gpu hold, preemptible; log runs/NAME/out.log)
    .venv/bin/python train.py MODEL_DIR data/train.jsonl data/val.jsonl --out runs/NAME [...]

Same setup as lora_check.py: LoRA (fp32 weights) on full and linear attention and the shared-expert
MLPs, router and routed experts frozen, gradient checkpointing, loss on the answer tokens only.
One example per forward (no padding); `--accum` examples per optimizer step, each weighted
equally. AdamW, linear warmup then cosine decay to a tenth of `--lr`, gradient clipping at 1.

An epoch is ~16 GPU hours, so the run shares the GPU: under `dibs run --preemptible --requeue`
it checks $DIBS_YIELD_FILE after every example, and when another session is waiting it saves its
full state (adapter, optimizer, the gradients accumulated so far, position in the epoch) and
exits 75, which hands the GPU over; dibs reruns it afterwards and it resumes. State is also
saved after every optimizer step, so a crash loses at most one step.

Writes to --out: log.jsonl (one line per example, optimizer step and evaluation), state/
(resume point), adapter/ (the final LoRA, peft format) and summary.json.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path

YIELD_STATUS = 75  # dibs --requeue: hand over and rerun


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", type=Path)
    ap.add_argument("train", type=Path)
    ap.add_argument("val", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--accum", type=int, default=8, help="examples per optimizer step")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=float, default=0.05, help="share of optimizer steps")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=32768, help="skip longer examples")
    ap.add_argument("--eval-every", type=int, default=10, help="optimizer steps between validation passes")
    ap.add_argument("--seed", type=int, default=20261108)
    ap.add_argument("--mem-floor-gb", type=float, default=12.0)
    ap.add_argument("--mem-cap-gb", type=float, default=60.0)
    ap.add_argument("--yield-every", type=float, default=0, metavar="MIN",
                    help="also hand the GPU over at the first optimizer-step boundary after MIN minutes "
                         "(low priority: lets waiters in even when dibs doesn't ask; 0 = only when asked)")
    a = ap.parse_args()

    import memguard
    memguard.start(a.mem_floor_gb, status=76)  # not 75: running out of RAM must not requeue
    if os.environ.get("DIBS_YIELD_FILE"):  # left from the turn we handed over; that waiter has had the GPU
        Path(os.environ["DIBS_YIELD_FILE"]).unlink(missing_ok=True)
    import torch
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
    from transformers import AutoTokenizer
    import q4moe
    from sft_loss import answer_loss

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
        prefix = tok.apply_chat_template(msgs[:1], add_generation_prompt=True, **kw)
        assert full.startswith(prefix), e["id"]
        ids = tok(full, add_special_tokens=False)["input_ids"]
        return ids, len(ids) - len(tok(prefix, add_special_tokens=False)["input_ids"])

    def read(path):
        rows = [json.loads(l) for l in path.open()]
        keep = [r for r in rows if r["meta"]["tokens"] <= a.max_tokens]
        return keep, len(rows) - len(keep)

    train, skipped = read(a.train)
    val, _ = read(a.val)
    steps_per_epoch = math.ceil(len(train) / a.accum)
    total_steps = steps_per_epoch * a.epochs
    warmup = max(1, round(a.warmup * total_steps))

    def lr_at(step):  # step counts from 0
        if step < warmup:
            return a.lr * (step + 1) / warmup
        p = (step - warmup) / max(1, total_steps - warmup)
        return a.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))

    t0 = time.time()
    t_start = t0
    model = q4moe.load(a.model)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    target = (r".*\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)"
              r"|linear_attn\.(in_proj_qkv|in_proj_z|out_proj)"
              r"|shared_expert\.(gate_proj|up_proj|down_proj))")
    model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.0,
                                             target_modules=target, task_type="CAUSAL_LM"),
                           autocast_adapter_dtype=True)
    params = [p for p in model.parameters() if p.requires_grad]
    assert all(p.dtype == torch.float32 for p in params), "LoRA weights should be fp32"
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)

    state_dir, new_dir, old_dir = out / "state", out / "state.new", out / "state.old"
    if not state_dir.exists() and old_dir.exists():
        old_dir.rename(state_dir)
    st = {"epoch": 0, "pos": 0, "step": 0, "micro": 0, "evaluated": -1, "skipped": skipped}
    if state_dir.exists():
        st = json.loads((state_dir / "state.json").read_text())
        set_peft_model_state_dict(model, torch.load(state_dir / "adapter.pt", map_location="cuda"))
        opt.load_state_dict(torch.load(state_dir / "opt.pt", map_location="cuda"))
        if (state_dir / "grads.pt").exists():
            for n, p, g in zip(names, params, torch.load(state_dir / "grads.pt", map_location="cuda")):
                p.grad = g
        log(event="resume", **st)
    else:
        log(event="start", args={k: str(v) for k, v in vars(a).items()}, train=len(train), val=len(val),
            skipped_long=skipped, steps=total_steps, warmup=warmup,
            trainable=sum(p.numel() for p in params))
    log(event="loaded", seconds=round(time.time() - t0, 1), gb=round(torch.cuda.memory_allocated() / 2**30, 1))

    def save(with_grads: bool):
        if new_dir.exists():
            shutil.rmtree(new_dir)
        new_dir.mkdir()
        torch.save(get_peft_model_state_dict(model), new_dir / "adapter.pt")
        torch.save(opt.state_dict(), new_dir / "opt.pt")
        if with_grads:
            torch.save([p.grad for p in params], new_dir / "grads.pt")
        (new_dir / "state.json").write_text(json.dumps(st) + "\n")
        if state_dir.exists():
            state_dir.rename(old_dir)
        new_dir.rename(state_dir)
        if old_dir.exists():
            shutil.rmtree(old_dir)

    def yield_requested():
        f = os.environ.get("DIBS_YIELD_FILE")
        if not f:
            return False
        if os.path.exists(f):
            return True
        # dibs 0.4.0 has been seen granting the GPU to a waiter without the yield file ever
        # appearing (2026-11-09, sft1), so also yield when the GPU hold is no longer ours.
        import subprocess
        try:
            status = subprocess.run(["dibs", "status"], capture_output=True, text=True, timeout=30).stdout
        except Exception:
            return False
        gpu = next((l for l in status.splitlines() if l.split()[:1] == ["gpu"]), "")
        return bool(gpu) and "free" not in gpu.split()[1:2] and os.environ.get("NKC_LABEL", "LoRA SFT") not in gpu

    def hand_over():
        save(with_grads=True)
        log(event="yield", **st)
        log_f.close()
        os._exit(YIELD_STATUS)

    def evaluate():
        model.eval()
        losses, t = {}, time.time()
        with torch.no_grad():
            for e in val:
                if yield_requested():
                    model.train()
                    return None
                ids, n_ans = encode(e)
                losses[e["id"]] = round(answer_loss(model, torch.tensor([ids], device="cuda"), n_ans).item(), 4)
        model.train()
        return {"mean": round(sum(losses.values()) / len(losses), 4), "by_id": losses, "seconds": round(time.time() - t, 1)}

    model.train()
    while st["epoch"] < a.epochs:
        order = list(range(len(train)))
        random.Random(a.seed * 1000 + st["epoch"]).shuffle(order)
        while st["pos"] < len(order):
            if st["micro"] == 0 and st["evaluated"] < st["step"] and st["step"] % a.eval_every == 0:
                ev = evaluate()
                if ev is None:
                    hand_over()
                st["evaluated"] = st["step"]
                log(event="eval", step=st["step"], **ev)
                save(with_grads=False)
            group = min(a.accum, len(order) - (st["pos"] - st["micro"]))
            e = train[order[st["pos"]]]
            ids, n_ans = encode(e)
            torch.cuda.reset_peak_memory_stats()
            t = time.time()
            loss = answer_loss(model, torch.tensor([ids], device="cuda"), n_ans)
            (loss / group).backward()
            torch.cuda.synchronize()
            log(event="example", epoch=st["epoch"], pos=st["pos"], step=st["step"], id=e["id"], tokens=len(ids),
                answer_tokens=n_ans, loss=round(loss.item(), 4), seconds=round(time.time() - t, 1),
                peak_gb=round(torch.cuda.max_memory_allocated() / 2**30, 1))
            st["pos"] += 1
            st["micro"] += 1
            if st["micro"] == group:
                lr = lr_at(st["step"])
                for g in opt.param_groups:
                    g["lr"] = lr
                norm = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
                opt.step()
                opt.zero_grad(set_to_none=True)
                st["step"] += 1
                st["micro"] = 0
                log(event="step", step=st["step"], lr=lr, grad_norm=round(norm, 4))
                save(with_grads=False)
            elif yield_requested():
                hand_over()
            if st["micro"] == 0 and (yield_requested() or
                                     (a.yield_every and time.time() - t_start > 60 * a.yield_every
                                      and st["pos"] < len(order))):
                hand_over()
        st["epoch"] += 1
        st["pos"] = 0
        save(with_grads=False)

    if st["evaluated"] < st["step"]:
        ev = evaluate()
        if ev is None:
            hand_over()
        st["evaluated"] = st["step"]
        log(event="eval", step=st["step"], **ev)
        save(with_grads=False)
    model.save_pretrained(out / "adapter")
    evals = [json.loads(l) for l in (out / "log.jsonl").open()]
    evals = [r for r in evals if r.get("event") == "eval"]
    (out / "summary.json").write_text(json.dumps({
        "steps": st["step"], "epochs": a.epochs, "train_examples": len(train), "skipped_long": st["skipped"],
        "val_loss": [(r["step"], r["mean"]) for r in evals]}, indent=1) + "\n")
    log(event="done", steps=st["step"])


if __name__ == "__main__":
    sys.exit(main())
