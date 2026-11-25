"""Single-cut Qwen3.5-2B sensitivity and local teacher-reconstruction pilot."""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoTokenizer, Qwen3_5ForCausalLM, Qwen3_5TextConfig

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = next((Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3.5-2B/snapshots").glob("*/config.json")).parent
CORPUS = ROOT / "target/qwen38-calibration/corpus/wikitext-2-raw"
OUT = ROOT / "target/adaptive-mlp-2b"
LAYERS = [3, 6, 10, 13, 17, 21]
REMOVE = 1536
LENGTH = 96
DEVICE = sys.argv[1] if len(sys.argv) > 1 else "cuda"
torch.set_num_threads(8)
torch.set_grad_enabled(False)


def load_model():
    config = Qwen3_5TextConfig.from_pretrained(CHECKPOINT, local_files_only=True)
    prefix = "model.language_model."
    with safe_open(next(CHECKPOINT.glob("*.safetensors")), framework="pt") as reader:
        state = {"model." + k[len(prefix):]: reader.get_tensor(k) for k in reader.keys() if k.startswith(prefix)}
    state["lm_head.weight"] = state["model.embed_tokens.weight"]
    # Instantiate normally so nonpersistent RoPE buffers have real values.
    model = Qwen3_5ForCausalLM(config).eval()
    model.load_state_dict(state, strict=True, assign=True)
    model.tie_weights()
    return model.to(device=DEVICE, dtype=torch.float32)


def samples(tokenizer, split, count, seed):
    paragraphs = [x.strip() for x in (CORPUS / f"wiki.{split}.raw").read_text().splitlines() if len(x.strip()) >= 600]
    rng = np.random.default_rng(seed)
    rows = []
    for index in rng.permutation(len(paragraphs)):
        ids = tokenizer.encode(paragraphs[index], add_special_tokens=False)
        if len(ids) >= LENGTH:
            start = int(rng.integers(len(ids) - LENGTH + 1))
            rows.append(ids[start:start + LENGTH])
        if len(rows) == count:
            break
    assert len(rows) == count
    return torch.tensor(rows, device=DEVICE)


def evaluate(model, rows):
    losses = []
    for batch in rows.split(2):
        logits = model(batch, use_cache=False).logits
        loss = F.cross_entropy(logits[:, :-1].flatten(0, 1), batch[:, 1:].flatten(), reduction="none")
        losses.extend(loss.reshape(len(batch), -1).mean(1).tolist())
    return losses


def masked(model, layer, removed, rows):
    def mask(module, args):
        return (args[0].index_fill(-1, removed, 0),)
    hook = model.model.layers[layer].mlp.down_proj.register_forward_pre_hook(mask)
    try:
        return evaluate(model, rows)
    finally:
        hook.remove()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, local_files_only=True)
    train = samples(tokenizer, "train", 40, 716)
    valid = samples(tokenizer, "valid", 16, 717)
    test = samples(tokenizer, "test", 32, 718)
    model = load_model()
    mlps = {i: model.model.layers[i].mlp for i in LAYERS}
    captured = {i: [] for i in LAYERS}

    def capture(layer):
        def hook(module, args):
            captured[layer].append(args[0].flatten(0, 1).cpu())
        return hook

    hooks = [mlp.down_proj.register_forward_pre_hook(capture(i)) for i, mlp in mlps.items()]
    for batch in train.split(2):
        model.model(batch, use_cache=False)
    for hook in hooks:
        hook.remove()
    activations = {i: torch.cat(parts).to(DEVICE) for i, parts in captured.items()}
    del captured
    removed = {}
    for i, mlp in mlps.items():
        energy = activations[i][:8 * LENGTH].square().mean(0) * mlp.down_proj.weight.square().sum(0)
        removed[i] = energy.argsort(stable=True)[:REMOVE]

    result = {"checkpoint": CHECKPOINT.name, "removed_channels_per_cut": REMOVE, "validation": {}, "test": {}}

    def rank(stage, layers):
        base = evaluate(model, valid)
        scores = {i: masked(model, i, removed[i], valid) for i in layers}
        order = sorted(scores, key=lambda i: np.mean(scores[i]))
        result["validation"][stage] = dict(base=base, masked=scores, order=order)
        print(stage, np.mean(base), [(i, round(np.mean(scores[i]) - np.mean(base), 6)) for i in order], flush=True)
        return order

    original = rank("original", LAYERS)
    first, stale = original[:2]
    result["test"]["original"] = evaluate(model, test)
    mlp = mlps[first]
    x = activations[first][8 * LENGTH:]
    del activations
    drop = removed[first]
    keep_mask = torch.ones(mlp.intermediate_size, device=DEVICE, dtype=torch.bool)
    keep_mask[drop] = False
    keep = torch.where(keep_mask)[0]
    rows = valid[:2]
    hook = mlp.down_proj.register_forward_pre_hook(lambda module, args: (args[0].index_fill(-1, drop, 0),))
    expected = model(rows, use_cache=False).logits
    hook.remove()
    residual = x[:, drop] @ mlp.down_proj.weight[:, drop].T
    for name in ("gate_proj", "up_proj", "down_proj"):
        proj = getattr(mlp, name)
        weight = proj.weight.index_select(1 if name == "down_proj" else 0, keep)
        resized = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False, device="meta")
        resized.weight = torch.nn.Parameter(weight, requires_grad=False)
        setattr(mlp, name, resized)
    mlp.intermediate_size = len(keep)
    actual = model(rows, use_cache=False).logits
    error = (actual - expected).abs()
    result["equivalence"] = {"max_abs": error.max().item(), "rms": error.square().mean().sqrt().item()}
    print("equivalence", result["equivalence"], flush=True)
    torch.testing.assert_close(actual, expected, atol=0.002, rtol=0.0001)
    del actual, expected, error
    rank("pruned", original[1:])
    result["test"]["pruned"] = evaluate(model, test)

    # Fit the missing teacher output from surviving channels with dual ridge regression.
    features = x[:, keep]
    gram = features @ features.T
    ridge = 0.1 * gram.diag().mean()
    gram.diagonal().add_(ridge)
    correction = features.T @ torch.linalg.solve(gram, residual)
    mlp.down_proj.weight.add_(correction.T)
    fresh = rank("healed", original[1:])[0]
    result["test"]["healed"] = evaluate(model, test)
    for i in sorted({fresh, stale}):
        result["test"][f"second_layer_{i}"] = masked(model, i, removed[i], test)
    result.update(first=first, stale=stale, fresh=fresh, parameters_per_cut=REMOVE * 3 * model.config.hidden_size,
                  ridge=ridge.item(), healing_tokens=len(features), device=DEVICE)
    torch.save({"first_layer": first, "kept_indices": keep.cpu(), "healed_down_weight": mlp.down_proj.weight.cpu(),
                "candidate_removed": {i: idx.cpu() for i, idx in removed.items()}}, OUT / "surgery.pt")
    (OUT / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print("test", {k: np.mean(v) for k, v in result["test"].items()}, flush=True)


if __name__ == "__main__":
    main()
