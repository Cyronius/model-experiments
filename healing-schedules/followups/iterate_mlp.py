"""Continue the saved Qwen3.5-2B pilot; run from the repository root."""
import json
import math
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

import adaptive_mlp_pilot as pilot

OUT = pilot.OUT / "iterative"
BUDGET = math.log(1.05)


def capture_inputs(model, rows, indices):
    parts = {i: [] for i in indices}

    def hook_for(i):
        def save(module, args):
            parts[i].append(args[0].flatten(0, 1).detach())
        return save

    hooks = [model.model.layers[i].mlp.down_proj.register_forward_pre_hook(hook_for(i)) for i in parts]
    try:
        for batch in rows.split(2):
            model.model(batch, use_cache=False)
    finally:
        for hook in hooks:
            hook.remove()
    return {i: torch.cat(v) for i, v in parts.items()}


def resize(mlp, keep):
    for name in ("gate_proj", "up_proj", "down_proj"):
        weight = getattr(mlp, name).weight.index_select(int(name == "down_proj"), keep)
        layer = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False, device="meta")
        layer.weight = torch.nn.Parameter(weight, requires_grad=False)
        setattr(mlp, name, layer)
    mlp.intermediate_size = len(keep)


def restore_edit(model, edit):
    mlp = model.model.layers[edit["first_layer"]].mlp
    resize(mlp, edit["kept_indices"].to(pilot.DEVICE))
    mlp.down_proj.weight.copy_(edit["healed_down_weight"])


def persist(report):
    tmp = OUT / "results.tmp"
    tmp.write_text(json.dumps(report, indent=2) + "\n")
    tmp.replace(OUT / "results.json")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(pilot.CHECKPOINT, local_files_only=True)
    train = pilot.samples(tok, "train", 40, 716)
    val = pilot.samples(tok, "valid", 80, 1717)
    rank, guard = val[:16], val[16:]
    test = pilot.samples(tok, "test", 128, 2718)
    model = pilot.load_model()
    count = sum(p.numel() for p in model.parameters())
    result_path = OUT / "results.json"
    if result_path.exists():
        report = json.loads(result_path.read_text())
    else:
        report = dict(checkpoint=pilot.CHECKPOINT.name, parameters=count, budget=BUDGET,
                      baseline=pilot.evaluate(model, guard), rounds=[], cut_size=1536)
        persist(report)
    first = torch.load(pilot.OUT / "surgery.pt", map_location="cpu")
    restore_edit(model, first)
    if "first" not in report:
        report["first"] = pilot.evaluate(model, guard)
        persist(report)
    for r in report["rounds"]:
        if r["accepted"]:
            restore_edit(model, torch.load(OUT / r["file"], map_location="cpu"))
    while report["cut_size"] >= 384:
        size = report["cut_size"]
        indices = [i for i, block in enumerate(model.model.layers) if block.mlp.intermediate_size - size >= 1536]
        inputs = capture_inputs(model, train[:8], indices)
        scores, removed = {}, {}
        for i in indices:
            mlp = model.model.layers[i].mlp
            energy = inputs.pop(i).square().mean(0) * mlp.down_proj.weight.square().sum(0)
            removed[i] = energy.argsort(stable=True)[:size]
            scores[i] = float(np.mean(pilot.masked(model, i, removed[i], rank)))
        i = min(scores, key=scores.get)
        mlp = model.model.layers[i].mlp
        original = (mlp.gate_proj, mlp.up_proj, mlp.down_proj, mlp.intermediate_size)
        x = capture_inputs(model, train[8:], [i])[i]
        drop = removed[i]
        keep_mask = torch.ones(mlp.intermediate_size, device=x.device, dtype=torch.bool)
        keep_mask[drop] = False
        keep = torch.where(keep_mask)[0]
        missing = x[:, drop] @ mlp.down_proj.weight[:, drop].T
        features = x[:, keep]
        resize(mlp, keep)
        before = pilot.evaluate(model, guard)
        gram = features @ features.T
        gram.diagonal().add_(0.1 * gram.diag().mean())
        mlp.down_proj.weight.add_((features.T @ torch.linalg.solve(gram, missing)).T)
        after = pilot.evaluate(model, guard)
        delta = float(np.mean(after) - np.mean(report["baseline"]))
        accepted = delta <= BUDGET
        record = dict(layer=i, size=size, rank_scores=scores, before=before, after=after,
                      delta=delta, accepted=accepted)
        if accepted:
            filename = f"edit-{len(report['rounds']) + 1:02}.pt"
            torch.save(dict(first_layer=i, kept_indices=keep.cpu(), healed_down_weight=mlp.down_proj.weight.cpu()), OUT / filename)
            record["file"] = filename
        else:
            mlp.gate_proj, mlp.up_proj, mlp.down_proj, mlp.intermediate_size = original
            report["cut_size"] //= 2
        report["rounds"].append(record)
        persist(report)
        print(f"Round {len(report['rounds'])}: layer {i}, cut {size}, guard PPL +{math.expm1(delta):.2%}, accepted={accepted}", flush=True)

    # Replay the trajectory once, evaluating test only after freezing all decisions.
    if "final_test" in report:
        return
    del model
    model = pilot.load_model()
    tests = {"baseline": pilot.evaluate(model, test)}
    restore_edit(model, first)
    tests["pilot"] = pilot.evaluate(model, test)
    milestones = {}
    history = [("pilot", np.mean(report["first"]) - np.mean(report["baseline"]))]
    history.extend((r["file"], r["delta"]) for r in report["rounds"] if r["accepted"])
    for pct in (1, 3, 5):
        eligible = [name for name, delta in history if delta <= math.log1p(pct / 100)]
        milestones[str(pct)] = eligible[-1] if eligible else "baseline"
    for r in report["rounds"]:
        if r["accepted"]:
            restore_edit(model, torch.load(OUT / r["file"], map_location="cpu"))
            if r["file"] in milestones.values():
                tests[r["file"]] = pilot.evaluate(model, test)
    report.update(final_test=tests, milestones=milestones, final_parameters=sum(p.numel() for p in model.parameters()))
    persist(report)
    print("complete", milestones, report["final_parameters"], flush=True)


if __name__ == "__main__":
    main()
