"""Export real 2a_384 activations and check packed integer layer arithmetic.

The six linears use independent NumPy int64 dot products. Attention, GELU and
norms deliberately retain PyTorch's reference implementation at this gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

from arithmetic import dequantize, integer_products, pack_activations, pack_weights, quantize, unpack_weights

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "tern-embed"))
from model import Student, tern_codes, quant_act  # noqa: E402


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metric(got, ref):
    diff = (got - ref).double()
    return {"max_abs": float(diff.abs().max()), "rms": float(diff.square().mean().sqrt())}


def export(checkpoint: Path, work: Path, out: Path):
    torch.set_num_threads(2)
    model = Student.load(checkpoint).eval()
    if (model.dims["d"] != 384 or model.dims["ffn"] != 1536
            or model.loops != 1 or model.dims["max_pos"] < 128 or not model.quantized):
        raise ValueError("This gate requires a quantized native-384/1536 student")
    out.mkdir(parents=True, exist_ok=True)
    ids = np.load(work / "ids.npy", mmap_mode="r")
    offsets = np.load(work / "offsets.npy")
    held = np.load(work / "held.npy")
    lengths = np.diff(offsets)
    matrices, layers, fixtures, checks = {}, {}, [], {}
    for li, layer in enumerate(model.layers):
        for name in ("q", "k", "v", "o", "fc1", "fc2"):
            module = getattr(layer, name)
            codes, scales = tern_codes(module.weight)
            codes, scales = codes.numpy().astype(np.int8), scales.numpy()
            packed = pack_weights(codes)
            decoded = unpack_weights(packed)
            if not np.array_equal(decoded, codes):
                raise AssertionError(f"Packing mismatch: layer {li} {name}")
            key = f"l{li}_{name}"
            packed.tofile(out / f"{key}.w.bin")
            np.savez(out / f"{key}.params.npz", scales=scales, bias=module.bias.detach().numpy())
            matrices[key] = (decoded, scales, module.bias.detach().numpy())
            layers[key] = {"k": codes.shape[1], "n": codes.shape[0], "packed_bytes": packed.nbytes,
                           "weight_sha256": sha256(out / f"{key}.w.bin")}

    # Each sequence trace contains four independent texts. A projection fixture
    # takes the first four flattened token rows of that trace.
    for cap in (8, 32, 128):
        selected = []
        for target in (cap, max(2, cap // 2), max(2, cap // 4), cap - 1):
            selected.append(int(held[np.argmin(np.abs(lengths[held] - target))]))
        rows = [np.array(ids[offsets[i]:min(offsets[i + 1], offsets[i] + cap)], dtype=np.int64) for i in selected]
        token_ids = torch.zeros((4, cap), dtype=torch.long)
        mask = torch.zeros((4, cap), dtype=torch.bool)
        for row, values in enumerate(rows):
            token_ids[row, :len(values)] = torch.from_numpy(values)
            mask[row, :len(values)] = True
        with torch.no_grad():
            ref_states, _, ref_vectors = model(token_ids, mask)
            x = ref_states[0]
            trace = {"ids": token_ids.numpy(), "mask": mask.numpy(), "embedding": x.numpy()}
            for li, layer in enumerate(model.layers):
                phase = {}

                def linear(name, value):
                    flat = value.reshape(-1, value.shape[-1]).numpy()
                    aq, act_scales = quantize(flat)
                    q_ref = quant_act(value).reshape(flat.shape).numpy()
                    rebuilt = (aq.reshape(len(flat), -1, aq.shape[-1] // act_scales.shape[-1])
                               * act_scales[..., None]).reshape(flat.shape)
                    if not np.array_equal(rebuilt, q_ref):
                        raise AssertionError(f"Activation quantization differs: {li}/{name}")
                    key = f"l{li}_{name}"
                    weights, scales, bias = matrices[key]
                    output = np.empty((len(flat), len(weights)), np.float32)
                    for start in range(0, len(flat), 4):
                        products = integer_products(aq[start:start + 4], weights)
                        output[start:start + 4] = dequantize(products, act_scales[start:start + 4], scales, bias)
                        # A real four-row fixture for every projection/layer/cap.
                        if start == 0:
                            prefix = f"s{cap}_{key}"
                            pack_activations(aq[:4]).tofile(out / f"{prefix}.a.bin")
                            products.tofile(out / f"{prefix}.expected.bin")
                            np.savez(out / f"{prefix}.dequant.npz", activation_scales=act_scales[:4],
                                     expected=output[:4])
                            fixtures.append({"name": prefix, "matrix": key, "k": weights.shape[1],
                                             "n": weights.shape[0], "groups": act_scales.shape[-1],
                                             "activation_sha256": sha256(out / f"{prefix}.a.bin"),
                                             "expected_sha256": sha256(out / f"{prefix}.expected.bin")})
                    result = torch.from_numpy(output).reshape(*value.shape[:-1], len(weights))
                    reference = getattr(layer, name)(value)
                    phase[name] = metric(result, reference)
                    torch.testing.assert_close(result, reference, atol=3e-5, rtol=2e-5)
                    trace[f"l{li}_{name}"] = result.numpy()
                    return result

                q, k, v = (linear(name, x).view(4, cap, layer.heads, -1).transpose(1, 2)
                           for name in ("q", "k", "v"))
                attention = ((q @ k.transpose(-1, -2)) / q.shape[-1] ** 0.5).masked_fill(
                    ~mask[:, None, None, :], float("-inf")).softmax(-1)
                context = (attention @ v).transpose(1, 2).reshape(4, cap, 384)
                x = layer.ln1(x + linear("o", context))
                trace[f"l{li}_ln1"] = x.numpy()
                gelu = F.gelu(linear("fc1", x), approximate=layer.gelu)
                trace[f"l{li}_gelu"] = gelu.numpy()
                x = layer.ln2(x + linear("fc2", gelu))
                phase["state"] = metric(x, ref_states[li + 1])
                torch.testing.assert_close(x, ref_states[li + 1], atol=2e-4, rtol=2e-4)
                trace[f"l{li}_output"] = x.numpy()
                checks[f"s{cap}_layer{li}"] = phase
            pooled = x[:, 0] if model.pool == "cls" else (x * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)
            vector = F.normalize(model.out(pooled), dim=-1)
            torch.testing.assert_close(vector, ref_vectors, atol=2e-5, rtol=2e-4)
            checks[f"s{cap}_vectors"] = metric(vector, ref_vectors)
            trace["vectors"] = vector.numpy()
            np.savez(out / f"s{cap}.trace.npz", **trace)

    report = {"checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": sha256(checkpoint),
              "dims": model.dims, "scope": "CPU integer oracle; nonlinear operations remain PyTorch",
              "reference_source_sha256": sha256(ROOT.parent / "tern-embed/model.py"),
              "corpus_sha256": {name: sha256(work / name) for name in ("ids.npy", "offsets.npy", "held.npy")},
              "matrices": layers, "fixtures": fixtures, "checks": checks}
    (out / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    retained = {k: v for k, v in report.items() if k not in ("fixtures", "matrices")}
    retained.update(matrices=len(layers), projection_fixtures=len(fixtures))
    (ROOT / "oracle-result.json").write_text(json.dumps(retained, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=ROOT.parent / "tern-embed/pulls/grid1/runs/2a_384/model.pt")
    parser.add_argument("--work", type=Path, default=ROOT.parent / "tern-embed/work_smoke")
    parser.add_argument("--out", type=Path, default=ROOT / "fixtures")
    args = parser.parse_args()
    report = export(args.checkpoint, args.work, args.out)
    print(json.dumps({"matrices": len(report["matrices"]), "fixtures": len(report["fixtures"]),
                      "checks": report["checks"]}, indent=2))


if __name__ == "__main__":
    main()
