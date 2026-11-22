"""Encoding-only Klein conditioning targets. No downloads, training or NPU dispatch."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import re
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file, save_file

TEACHER = "black-forest-labs/FLUX.2-klein-4B"
REVISION = "e7b7dc27f91deacad38e78976d1f2b499d76a294"
PROTOCOL = "klein-projected-conditioning-v1"
POSITIONS, WIDTH, TAP_WIDTH = 512, 3072, 2560
TAPS = (9, 18, 27)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def case_hash(case: dict) -> str:
    return hashlib.sha256(json.dumps(case, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def read_cases(path: Path) -> list[dict]:
    cases = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    seen, prompts = set(), set()
    for case in cases:
        name, prompt = case.get("id"), case.get("prompt")
        if not isinstance(name, str) or not re.fullmatch(r"[a-zA-Z0-9_-]+", name) or name in seen:
            raise ValueError("Case IDs must be unique safe filenames")
        if not isinstance(prompt, str) or prompt in prompts:
            raise ValueError("Prompts must be unique strings across all splits")
        if case.get("split") not in {"train", "validation", "test"}:
            raise ValueError("Each case needs a train/validation/test split")
        if not isinstance(case.get("seed"), int) or isinstance(case["seed"], bool) or case["seed"] < 0:
            raise ValueError("Each case needs a nonnegative integer image seed")
        seen.add(name)
        prompts.add(prompt)
    if not cases:
        raise ValueError("Empty case set")
    return cases


def chat_text(prompt: str) -> str:
    return (f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            "<think>\n\n</think>\n\n")


def prepare_inputs(tokenizer, prompt: str) -> tuple[dict, int]:
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False,
    )
    if rendered != chat_text(prompt):
        raise ValueError("Tokenizer chat template differs from the frozen Phlegm Klein control")
    if tokenizer.padding_side != "right" or tokenizer.truncation_side != "right" or tokenizer.pad_token_id != 151643:
        raise ValueError("Klein control requires right padding/truncation and pad ID 151643")
    full_ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
    inputs = tokenizer(rendered, return_tensors="pt", padding="max_length",
                       truncation=True, max_length=POSITIONS)
    # Reject tokenizer-specific extra tokens instead of silently changing the native control.
    real = int(inputs["attention_mask"].sum())
    if inputs["input_ids"][0, :real].tolist() != full_ids[:POSITIONS]:
        raise ValueError("Tokenizer adds tokens absent from the Phlegm native control")
    return {k: inputs[k] for k in ("input_ids", "attention_mask")}, len(full_ids)


def text_ids(length: int | None = None) -> torch.Tensor:
    length = POSITIONS if length is None else length
    ids = torch.zeros(length, 4, dtype=torch.int64)
    ids[:, 3] = torch.arange(length)
    return ids


def project(hidden_states, weight: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    taps = [hidden_states[k] for k in TAPS]
    if any(t.shape[-2:] != (POSITIONS, TAP_WIDTH) for t in taps):
        raise ValueError("Teacher tap geometry differs from Klein 4B")
    if tuple(weight.shape) != (WIDTH, 3 * TAP_WIDTH):
        raise ValueError("Projection geometry differs from Klein 4B")
    if bias is not None and tuple(bias.shape) != (WIDTH,):
        raise ValueError("Projection bias geometry differs from Klein 4B")
    # Keep every padding row: the denoiser attends to the whole conditioning sequence.
    return F.linear(torch.cat(taps, dim=-1).to(device=weight.device, dtype=weight.dtype), weight, bias)


def projection_files(model: Path) -> tuple[dict[str, torch.Tensor], list[Path]]:
    folder = model / "transformer"
    index = folder / "diffusion_pytorch_model.safetensors.index.json"
    weight_name, bias_name = "context_embedder.weight", "context_embedder.bias"
    if index.exists():
        mapping = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        names = [weight_name] + ([bias_name] if bias_name in mapping else [])
        files = {name: folder / mapping[name] for name in names}
        provenance = [index]
    else:
        files = {}
        for path in sorted(folder.glob("*.safetensors")):
            with safe_open(str(path), framework="pt", device="cpu") as f:
                for name in (weight_name, bias_name):
                    if name in f.keys():
                        if name in files:
                            raise ValueError(f"Duplicate projection tensor: {name}")
                        files[name] = path
        provenance = []
    if weight_name not in files:
        raise ValueError("Missing context_embedder.weight")
    tensors = {}
    for name, path in files.items():
        if not path.resolve().is_relative_to(folder.resolve()):
            # HF cache snapshots use symlinks into a sibling blobs directory.
            if not path.absolute().is_relative_to(folder.absolute()) or ".." in path.parts:
                raise ValueError("Projection shard escapes transformer directory")
        with safe_open(str(path), framework="pt", device="cpu") as f:
            tensors[name] = f.get_tensor(name)
    return tensors, provenance + sorted(set(files.values()))


def validate_tensors(tensors: dict, prediction: bool = False) -> None:
    for name, shape in {"projected_conditioning": (POSITIONS, WIDTH), "text_ids": (POSITIONS, 4)}.items():
        value = tensors.get(name)
        if value is None or tuple(value.shape) != shape or not torch.isfinite(value).all():
            raise ValueError(f"Invalid {name}: expected finite {shape}")
    if not tensors["projected_conditioning"].is_floating_point():
        raise ValueError("Conditioning must be floating point")
    if tensors["text_ids"].dtype != torch.int64 or not torch.equal(tensors["text_ids"], text_ids()):
        raise ValueError("Text position IDs differ from the 512-position control")
    if not prediction:
        ids, mask = tensors.get("input_ids"), tensors.get("attention_mask")
        if ids is None or mask is None or ids.shape != (POSITIONS,) or mask.shape != (POSITIONS,):
            raise ValueError("Missing token IDs or mask")
        if ids.dtype != torch.int64 or mask.dtype != torch.int64:
            raise ValueError("Token IDs and mask must be int64")
        real = int(mask.sum())
        if not 0 < real <= POSITIONS or not torch.equal(mask, (torch.arange(POSITIONS) < real).long()):
            raise ValueError("Attention mask must describe a nonempty right-padded sequence")
        if (ids < 0).any() or not (ids[real:] == 151643).all():
            raise ValueError("Invalid token or padding IDs")


def export(args) -> None:
    cases = read_cases(args.cases)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    model_path = args.model.resolve()
    config_path = model_path / "text_encoder" / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("model_type") != "qwen3" or config.get("hidden_size") != TAP_WIDTH or config.get("num_hidden_layers", 0) <= max(TAPS):
        raise ValueError("Expected the full Qwen3-4B teacher, with intermediate taps 9/18/27")
    from transformers import AutoTokenizer, Qwen3Model

    tokenizer = AutoTokenizer.from_pretrained(model_path / "tokenizer", local_files_only=True)
    prepared = [prepare_inputs(tokenizer, c["prompt"]) for c in cases]
    for case, (_, length) in zip(cases, prepared):
        if "expected_untruncated_tokens" in case and length != case["expected_untruncated_tokens"]:
            raise ValueError(f"Token-boundary fixture drifted: {case['id']}: {length}")
    weights, files = projection_files(model_path)
    source_files = sorted(set(files + list((model_path / "transformer").glob("*.json"))
                              + list((model_path / "text_encoder").glob("*.safetensors"))
                              + list((model_path / "text_encoder").glob("*.json"))
                              + [p for p in (model_path / "tokenizer").rglob("*") if p.is_file()]))
    provenance = {str(p.relative_to(model_path)).replace("\\", "/"): sha256(p) for p in source_files}
    upstream_path = model_path / "verified-upstream.json"
    hub_verification = None
    if upstream_path.exists():
        upstream = json.loads(upstream_path.read_text(encoding="utf-8"))
        recorded = {r["path"]: r for r in upstream["files"]}
        if upstream.get("repo") != TEACHER or upstream.get("revision") != REVISION:
            raise ValueError("Upstream provenance identifies a different teacher")
        if any(name not in recorded or recorded[name]["sha256"] != sha or not recorded[name]["hub_verified"]
               for name, sha in provenance.items()):
            raise ValueError("Checkpoint files differ from recorded upstream verification")
        hub_verification = {"path": str(upstream_path), "sha256": sha256(upstream_path),
                            "revision": REVISION, "matched_source_files": len(provenance)}
    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    model, loading = Qwen3Model.from_pretrained(
        model_path / "text_encoder", local_files_only=True, output_loading_info=True,
        torch_dtype=dtype, attn_implementation="eager")
    if any(loading.get(k) for k in ("missing_keys", "mismatched_keys", "error_msgs")):
        raise ValueError(f"Incomplete teacher checkpoint: {loading}")
    loading = {k: sorted(v) if isinstance(v, set) else v for k, v in loading.items()}
    model = model.to(args.device).eval()
    weight = weights["context_embedder.weight"].to(device=args.device, dtype=dtype)
    bias = weights.get("context_embedder.bias")
    if bias is not None:
        bias = bias.to(device=args.device, dtype=dtype)
    args.output.mkdir(parents=True)
    manifest = {
        "protocol": PROTOCOL, "teacher": TEACHER, "declared_revision": REVISION,
        "revision_note": ("Source hashes matched the pinned Hub verification recorded in hub_verification."
                          if hub_verification else
                          "Local file hashes identify the actual checkpoint; the revision label is not independently verified."),
        "model_path": str(model_path), "source_sha256": provenance,
        "harness_sha256": sha256(Path(__file__)), "cases_sha256": sha256(args.cases),
        "hub_verification": hub_verification,
        "dtype": args.dtype, "device": args.device, "attention": "eager", "loading_info": loading,
        "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(),
        "versions": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "safetensors")},
        "positions": POSITIONS, "width": WIDTH, "taps": list(TAPS), "cases": [],
    }
    # No complete manifest is published for a failed or interrupted export.
    for case, (inputs, full_length) in zip(cases, prepared):
        with torch.inference_mode():
            output = model(**{k: v.to(args.device) for k, v in inputs.items()},
                           output_hidden_states=True, use_cache=False)
            conditioning = project(output.hidden_states, weight, bias)[0].cpu().contiguous()
        tensors = {"projected_conditioning": conditioning, "text_ids": text_ids(),
                   **{k: v[0].cpu().contiguous() for k, v in inputs.items()}}
        validate_tensors(tensors)
        path = args.output / f"{case['id']}.safetensors"
        save_file(tensors, str(path), metadata={"protocol": PROTOCOL, "case_sha256": case_hash(case)})
        manifest["cases"].append({**case, "case_sha256": case_hash(case), "file": path.name,
                                  "sha256": sha256(path), "untruncated_tokens": full_length,
                                  "truncated": full_length > POSITIONS})
        print(f"{case['id']}: {full_length} tokens, truncated={full_length > POSITIONS}", flush=True)
        del output, conditioning
    tmp = args.output / "manifest.json.tmp"
    tmp.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    tmp.replace(args.output / "manifest.json")


def metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict:
    if prediction.shape != target.shape or not prediction.is_floating_point() or not target.is_floating_point():
        raise ValueError("Metrics require matching floating point tensors")
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ValueError("Nonfinite conditioning")
    if not target.numel():
        return {"positions": 0}
    a, b = prediction.double(), target.double()
    delta = a - b
    norm_a, norm_b = a.norm(), b.norm()
    # Zero reference tensors are valid; report undefined relative statistics as null.
    return {"positions": len(b), "rmse": float(delta.square().mean().sqrt()),
            "relative_l2": float(delta.norm() / norm_b) if norm_b else None,
            "cosine": float((a * b).sum() / (norm_a * norm_b)) if norm_a and norm_b else None,
            "max_absolute_error": float(delta.abs().max())}


def save_prediction(target_file: Path, conditioning: torch.Tensor, output_file: Path) -> None:
    """Student adapter: bind each prediction to an exact teacher target artifact."""
    if output_file.exists():
        raise FileExistsError(output_file)
    target = load_file(str(target_file))
    validate_tensors(target)
    tensors = {"projected_conditioning": conditioning.detach().cpu().contiguous(),
               **{name: target[name] for name in ("text_ids", "input_ids", "attention_mask")}}
    validate_tensors(tensors, prediction=True)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(output_file), metadata={"protocol": PROTOCOL, "reference_sha256": sha256(target_file)})


def evaluate(target_dir: Path, prediction_dir: Path, split: str = "test") -> dict:
    manifest = json.loads((target_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("protocol") != PROTOCOL:
        raise ValueError("Unknown target protocol")
    rows, names = [], set()
    for case in manifest["cases"]:
        if case["split"] != split:
            continue
        filename = case["file"]
        if not re.fullmatch(r"[a-zA-Z0-9_-]+\.safetensors", filename) or filename in names:
            raise ValueError("Invalid or duplicated target filename")
        names.add(filename)
        target_file, pred_file = target_dir / filename, prediction_dir / filename
        reference_hash = sha256(target_file)
        if reference_hash != case["sha256"] or case_hash({k: v for k, v in case.items() if k not in {
                "case_sha256", "file", "sha256", "untruncated_tokens", "truncated"}}) != case["case_sha256"]:
            raise ValueError("Target provenance changed")
        with safe_open(str(target_file), framework="pt") as f:
            if (f.metadata() or {}).get("case_sha256") != case["case_sha256"]:
                raise ValueError("Target case identity differs from manifest")
        with safe_open(str(pred_file), framework="pt") as f:
            metadata = f.metadata() or {}
        if metadata.get("protocol") != PROTOCOL or metadata.get("reference_sha256") != reference_hash:
            raise ValueError("Prediction is not bound to this exact target")
        target, pred = load_file(str(target_file)), load_file(str(pred_file))
        validate_tensors(target)
        validate_tensors(pred, prediction=True)
        mask = target["attention_mask"].bool()
        a, b = pred["projected_conditioning"], target["projected_conditioning"]
        rows.append({"id": case["id"], "target_sha256": reference_hash, "prediction_sha256": sha256(pred_file),
                     "all": metrics(a, b), "real": metrics(a[mask], b[mask]), "padding": metrics(a[~mask], b[~mask])})
    if not rows:
        raise ValueError(f"No targets for split {split}")
    return {"protocol": PROTOCOL, "split": split, "target_manifest_sha256": sha256(target_dir / "manifest.json"),
            "quality_gate": "unmeasured: conditioning metrics do not establish generated-image quality",
            "cases": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    exp = sub.add_parser("export", help="Export targets from an existing local teacher snapshot")
    exp.add_argument("--model", type=Path, required=True)
    exp.add_argument("--cases", type=Path, default=Path(__file__).with_name("heldout.jsonl"))
    exp.add_argument("--output", type=Path, required=True)
    exp.add_argument("--device", default="cpu")
    exp.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    exp.add_argument("--threads", type=int, default=8, help="CPU intra-op threads (recorded in manifest)")
    ev = sub.add_parser("evaluate", help="Compare student outputs against exact teacher target files")
    ev.add_argument("--targets", type=Path, required=True)
    ev.add_argument("--predictions", type=Path, required=True)
    ev.add_argument("--split", choices=("train", "validation", "test"), default="test")
    ev.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "export":
        if args.threads < 1:
            parser.error("--threads must be positive")
        torch.set_num_threads(args.threads)
        export(args)
    else:
        if args.output.exists():
            raise FileExistsError(args.output)
        report = evaluate(args.targets, args.predictions, args.split)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
