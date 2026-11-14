"""Supplemental output-only diagnostic on unchanged teacher residual states.

The primary experiment stays unchanged. Compare its retained-space gain matrix
with folding the original gain into an untied readout exactly. All readout math
is FP32 on the same BF16 teacher activations; this is not a healed student or an
additive decomposition of end-to-end KL.
"""
import argparse
import gc
import json
from pathlib import Path

import torch

from audit_rotation import compare, norm_gain, norm_epsilon
from scale_data import load_tokenizer
from scale_probe import load_source, validation_rows, save


def readout_inputs(hidden, gain, basis, scale, eps):
    h, p, gain = hidden.float(), basis.float(), gain.float()
    reference = h * torch.rsqrt(h.square().mean(-1, keepdim=True) + eps) * gain
    z = h @ p
    zn = z * torch.rsqrt(z.square().mean(-1, keepdim=True) + eps)
    matrix = float(scale) * (p.T @ (gain[:, None] * p))
    retained_matrix = (zn @ matrix) @ p.T
    exact_readout = (zn @ p.T) * gain * float(scale)
    return reference, retained_matrix, exact_readout


@torch.no_grad()
def run(name, root):
    result_dir = root / "results" / name
    result = json.loads((result_dir / "result.json").read_text())
    data = json.loads((root / "data.json").read_text())
    source = root / "models" / name
    tok = load_tokenizer(source)
    rows = validation_rows(data, tok)[:2]
    model, _ = load_source(source, "cuda:0")
    hidden = []
    for row in rows:
        def hook(module, args):
            hidden.append(args[0][0, row["positions"]].clone())
        handle = model.model.norm.register_forward_pre_hook(hook)
        try:
            model.model(input_ids=torch.tensor([row["input_ids"]], device="cuda"), use_cache=False)
        finally:
            handle.remove()
    hidden = torch.cat(hidden)
    q = torch.load(result_dir / "basis.pt", weights_only=True).to("cuda")
    scales = json.loads((result_dir / "rms-scales.json").read_text())
    gain, eps = norm_gain(model.model.norm), norm_epsilon(model.model.norm)
    output = {"model": name, "positions": len(hidden), "rows": [r["id"] for r in rows],
              "primary_data_sha256": result["data_sha256"], "readout_dtype": "float32",
              "scope": "Exploratory readout-only diagnostic; unmodified teacher activations, first two validation conversations. Not full-model generation quality.",
              "source_tied_embeddings": model.model.embed_tokens.weight is model.lm_head.weight,
              "arms": {}}
    for arm in result["plan"]["arms"]:
        k = arm["width"]
        inputs = readout_inputs(hidden, gain, q[:, :k], scales[str(k)][-1], eps)
        logits = [torch.empty(len(hidden), model.config.vocab_size, device="cpu") for _ in inputs]
        for start in range(0, model.config.vocab_size, 2048):
            weight = model.lm_head.weight[start:start + 2048].float()
            for dest, states in zip(logits, inputs):
                dest[:, start:start + len(weight)] = (states @ weight.T).cpu()
        output["arms"][str(k)] = {
            "teacher_vs_retained_matrix": compare([logits[0]], [logits[1]]),
            "teacher_vs_exact_readout": compare([logits[0]], [logits[2]]),
            "exact_vs_retained_matrix": compare([logits[2]], [logits[1]])}
        del inputs, logits
    save(output, root / "results" / f"{name}-readout-diagnostic.json")
    print(json.dumps(output, indent=2), flush=True)
    del model, hidden, q, gain
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("/workspace/scale"))
    ap.add_argument("--models", nargs="+", required=True)
    args = ap.parse_args()
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    for name in args.models:
        run(name, args.root)
