"""Train, evaluate, and export the first Q4 Nanbeige microdrafter pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from drafter import Config, Drafter
from q4_table import pack_q4, packed_bytes


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def data(path: Path, device: torch.device):
    with np.load(path) as archive:
        hidden = torch.from_numpy(archive["hidden"].astype(np.float32)).to(device)
        seed = torch.from_numpy(archive["seed"].astype(np.int64)).to(device)
        future = torch.from_numpy(archive["future"].astype(np.int64)).to(device)
    if hidden.ndim != 2 or hidden.shape[1] != Config.hidden or future.shape != (len(seed), 6):
        raise ValueError(f"wrong data shape in {path}")
    if len(seed) == 0 or seed.min() < 0 or future.min() < 0:
        raise ValueError(f"empty or invalid data in {path}")
    return hidden, torch.cat((seed[:, None], future), dim=1)


def forward_loss(model: Drafter, hidden: torch.Tensor, tokens: torch.Tensor,
                 position_weights: torch.Tensor | None = None) -> torch.Tensor:
    visible = torch.rand(tokens.shape, device=tokens.device) < 0.2
    visible[:, 0] = True
    # Half the minibatches exercise the actual first-pass path.
    if torch.rand(()) < 0.5:
        visible[:, 1:] = False
    visible[:, -1] = False
    logits = model(hidden, tokens, visible)
    labels = tokens[:, 1:].masked_fill(visible[:, 1:], -100)
    losses = F.cross_entropy(logits.reshape(-1, model.config.vocab), labels.reshape(-1),
                             reduction="none").reshape_as(labels)
    if position_weights is None:
        return losses.sum() / (labels != -100).sum()
    weights = position_weights[None, :] * (labels != -100)
    return (losses * weights).sum() / weights.sum()


@torch.no_grad()
def evaluate(model: Drafter, dataset, batch: int = 32, passes: int = 1) -> dict:
    hidden, tokens = dataset
    model.eval()
    total, matched, exact = 0, 0, 0
    position = np.zeros(6, dtype=np.int64)
    for start in range(0, len(tokens), batch):
        stop = start + batch
        draft = model.draft(hidden[start:stop], tokens[start:stop, 0], passes=passes)
        hits = draft[:, 1:].eq(tokens[start:stop, 1:])
        prefix = hits.long().cumprod(dim=1)
        position += hits.sum(dim=0).cpu().numpy()
        matched += int(prefix.sum())
        exact += int(prefix[:, -1].sum())
        total += len(draft)
    return {"blocks": total, "mean_accepted_future": matched / total,
            "mean_accepted_including_gpu_seed": 1 + matched / total,
            "passes": passes,
            "six_token_exact_rate": exact / total,
            "position_accuracy": (position / total).tolist()}


def export(model: Drafter, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    table = model.factors.detach().float().cpu().numpy()
    packed, scale = pack_q4(table)
    packed.tofile(output / "vocab.q4")
    scale.tofile(output / "vocab.scale.f16")
    body = {key: value.detach().cpu().half() for key, value in model.state_dict().items()
            if key != "factors"}
    torch.save(body, output / "body.f16.pt")
    layout = {}
    raw = bytearray()
    for key, value in sorted(body.items()):
        raw.extend(bytes((-len(raw)) % 64))
        offset = len(raw)
        blob = value.contiguous().numpy().astype("<f2", copy=False).tobytes()
        raw.extend(blob)
        layout[key] = {"shape": list(value.shape), "offset": offset, "bytes": len(blob),
                       "dtype": "fp16-le"}
    (output / "body.f16.bin").write_bytes(raw)
    (output / "body.layout.json").write_text(json.dumps(layout, indent=2) + "\n")
    body_bytes = sum(t.numel() * 2 for t in body.values())
    table_bytes = packed.nbytes + scale.nbytes
    if table_bytes != packed_bytes(model.config.vocab, model.config.rank):
        raise AssertionError("Q4 export size mismatch")
    report = {**model.manifest(), "vocab_table_bytes": table_bytes,
              "body_fp16_tensor_bytes": body_bytes,
              "body_fp16_aligned_bytes": len(raw),
              "total_tensor_bytes": table_bytes + body_bytes,
              "total_tensor_mib": (table_bytes + body_bytes) / 1048576,
              "total_aligned_bytes": table_bytes + len(raw),
              "note": "Tensor payload only; excludes NPU activations, code, alignment, staging and runtime buffers."}
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def run(args):
    if any(weight <= 0 for weight in args.position_weights):
        raise ValueError("position weights must be positive")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    train_set = data(args.train, device)
    eval_set = data(args.eval, device)
    if args.zero_hidden:
        train_set[0].zero_()
        eval_set[0].zero_()
    model = Drafter().to(device)
    if args.init_factors:
        factors = np.load(args.init_factors)
        if factors.shape != (model.config.vocab, model.config.rank):
            raise ValueError("initial token factors have the wrong shape")
        with torch.no_grad():
            model.factors.copy_(torch.from_numpy(factors.astype(np.float32)).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = {"train": str(args.train), "train_sha256": digest(args.train),
                "eval": str(args.eval), "eval_sha256": digest(args.eval),
                "seed": args.seed, "batch": args.batch, "steps": args.steps,
                "learning_rate": args.lr, "device": str(device),
                "position_weights": args.position_weights,
                "zero_hidden": args.zero_hidden,
                "init_factors": str(args.init_factors) if args.init_factors else None,
                "init_factors_sha256": digest(args.init_factors) if args.init_factors else None,
                "target": "Nanbeige4.3-33B-Q4pure-NPU2/model.gguf",
                "model": model.manifest()}
    (args.output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    initial = evaluate(model, eval_set)
    initial_two = evaluate(model, eval_set, passes=2)
    print("initial", json.dumps(initial), flush=True)
    n = len(train_set[1])
    train_probe = (train_set[0][:min(128, n)], train_set[1][:min(128, n)])
    started = time.monotonic()
    best = -1.0
    position_weights = torch.tensor(args.position_weights, device=device)
    for step in range(1, args.steps + 1):
        model.train()
        indices = torch.randint(n, (args.batch,), device=device)
        loss = forward_loss(model, train_set[0][indices], train_set[1][indices], position_weights)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            score = evaluate(model, eval_set)
            two_pass = evaluate(model, eval_set, passes=2)
            train_score = evaluate(model, train_probe)
            record = {"step": step, "loss": float(loss.detach()),
                      "elapsed_seconds": time.monotonic() - started,
                      **score, "two_pass": two_pass,
                      "train_probe_accepted_future": train_score["mean_accepted_future"]}
            with (args.output / "history.jsonl").open("a") as log:
                log.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
            if score["mean_accepted_future"] > best:
                best = score["mean_accepted_future"]
                torch.save({"config": model.manifest(), "model": model.state_dict(),
                            "step": step, "score": score}, args.output / "best.pt")
                export(model, args.output / "q4")
    (args.output / "result.json").write_text(json.dumps({"initial": initial,
        "initial_two_pass": initial_two,
        "best_accepted_future": best, "train_blocks": n,
        "eval_blocks": len(eval_set[1]), "elapsed_seconds": time.monotonic() - started}, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--eval", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--init-factors", type=Path)
    parser.add_argument("--zero-hidden", action="store_true")
    parser.add_argument("--position-weights", nargs=6, type=float, default=[1, 1, 1, 1, 1, 1])
    run(parser.parse_args())


if __name__ == "__main__":
    main()
