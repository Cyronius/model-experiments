"""Compare exports against pinned upstream Klein methods and Qwen3ForCausalLM.

Only the two upstream static text methods are executed; the installed diffusers
package is not required. The CPU device helper returns the model's actual device.
No harness tokenization, tap concatenation or projection function is reused.
"""
from __future__ import annotations

import argparse
import ast
import copy
import importlib.metadata
import json
from pathlib import Path
import time

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

import harness as h

DIFFUSERS_COMMIT = "c60830ee365d520ab52b110dda562dd26f7b4d7f"
PIPELINE_PATH = "src/diffusers/pipelines/flux2/pipeline_flux2_klein.py"


def reference_methods(source: Path, expected_sha256: str) -> dict:
    if h.sha256(source) != expected_sha256:
        raise ValueError("Pinned reference source hash mismatch")
    tree = ast.parse(source.read_text(encoding="utf-8"))
    pipeline = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Flux2KleinPipeline")
    names = {"_get_qwen3_prompt_embeds", "_prepare_text_ids"}
    methods = [copy.deepcopy(n) for n in pipeline.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in methods} != names:
        raise ValueError("Pinned source is missing reference methods")
    for method in methods:
        method.decorator_list = []
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, *methods], type_ignores=[]))
    namespace = {"torch": torch, "get_module_execution_device": lambda model: next(model.parameters()).device}
    exec(compile(module, str(source), "exec"), namespace)
    return {name: namespace[name] for name in names}


def verify_sources(model: Path, manifest: dict) -> tuple[dict, dict]:
    upstream_path = model / "verified-upstream.json"
    upstream = json.loads(upstream_path.read_text(encoding="utf-8"))
    if upstream.get("repo") != h.TEACHER or upstream.get("revision") != h.REVISION:
        raise ValueError("Wrong teacher provenance")
    if upstream.get("diffusers_commit") != DIFFUSERS_COMMIT:
        raise ValueError("Wrong reference commit")
    if manifest.get("declared_revision") != h.REVISION:
        raise ValueError("Targets declare a different teacher revision")
    expected = {r["path"]: r for r in upstream["files"]}
    for name, digest in manifest["source_sha256"].items():
        path = model / name
        if not path.absolute().is_relative_to(model.absolute()) or ".." in Path(name).parts:
            raise ValueError("Source file escapes checkpoint")
        if name not in expected or not expected[name]["hub_verified"] or expected[name]["sha256"] != digest:
            raise ValueError(f"Target checkpoint provenance differs: {name}")
        if h.sha256(path) != digest:
            raise ValueError(f"Current checkpoint changed: {name}")
    url = f"https://raw.githubusercontent.com/huggingface/diffusers/{DIFFUSERS_COMMIT}/{PIPELINE_PATH}"
    record = next(r for r in upstream["reference_source"] if r["url"] == url)
    return upstream, record


def compare(model_path: Path, targets: Path, output: Path, threads: int = 8) -> dict:
    if output.exists():
        raise FileExistsError(output)
    if threads < 1:
        raise ValueError("threads must be positive")
    torch.set_num_threads(threads)
    manifest = json.loads((targets / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("protocol") != h.PROTOCOL or manifest.get("dtype") not in {"bf16", "fp32"}:
        raise ValueError("Unsupported target protocol or dtype")
    upstream, source = verify_sources(model_path, manifest)
    methods = reference_methods(model_path / source["path"], source["sha256"])
    from transformers import AutoTokenizer, Qwen3ForCausalLM

    dtype = torch.bfloat16 if manifest["dtype"] == "bf16" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_path / "tokenizer", local_files_only=True)
    model, loading = Qwen3ForCausalLM.from_pretrained(
        model_path / "text_encoder", local_files_only=True, torch_dtype=dtype,
        attn_implementation="eager", output_loading_info=True)
    if any(loading.get(k) for k in ("missing_keys", "mismatched_keys", "error_msgs")):
        raise ValueError(f"Incomplete reference model: {loading}")
    model = model.cpu().eval()
    # Independently load the published transformer projection and use nn.Linear.
    with safe_open(str(model_path / "transformer" / "diffusion_pytorch_model.safetensors"), framework="pt") as f:
        weight = f.get_tensor("context_embedder.weight")
        bias = f.get_tensor("context_embedder.bias") if "context_embedder.bias" in f.keys() else None
    linear = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=bias is not None, device="meta")
    linear.load_state_dict({"weight": weight, **({"bias": bias} if bias is not None else {})}, assign=True)
    linear = linear.to(dtype=dtype).eval()
    captured = {}

    def capture(module, args, kwargs):
        captured["input_ids"] = kwargs["input_ids"].detach().cpu().clone()
        captured["attention_mask"] = kwargs["attention_mask"].detach().cpu().clone()

    hook = model.register_forward_pre_hook(capture, with_kwargs=True)
    output.mkdir(parents=True)
    report = {
        "protocol": h.PROTOCOL, "diffusers_commit": DIFFUSERS_COMMIT,
        "source": source, "reference_methods": sorted(methods),
        "method_note": "Unmodified upstream method bodies, extracted by AST; CPU helper returns model parameter device. Projection via nn.Linear.",
        "harness_sha256": h.sha256(Path(h.__file__)), "comparison_sha256": h.sha256(Path(__file__)),
        "target_manifest_sha256": h.sha256(targets / "manifest.json"),
        "upstream_manifest_sha256": h.sha256(model_path / "verified-upstream.json"),
        "versions": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "safetensors")},
        "dtype": manifest["dtype"], "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "acceptance": "Exact tensor equality for teacher export plumbing; no image-quality or student gate.", "cases": [],
    }
    try:
        for case in manifest["cases"]:
            name = case["file"]
            if Path(name).name != name or ".." in name:
                raise ValueError("Unsafe target filename")
            target_file = targets / name
            if h.sha256(target_file) != case["sha256"]:
                raise ValueError("Target artifact hash mismatch")
            target = load_file(str(target_file))
            started = time.perf_counter()
            with torch.inference_mode():
                raw = methods["_get_qwen3_prompt_embeds"](model, tokenizer, case["prompt"])
                prediction = linear(raw)[0].contiguous()
                positions = methods["_prepare_text_ids"](raw)[0].contiguous()
            mask = target["attention_mask"].bool()
            equals = {"conditioning": torch.equal(prediction, target["projected_conditioning"]),
                      "text_ids": torch.equal(positions, target["text_ids"]),
                      "input_ids": torch.equal(captured["input_ids"][0], target["input_ids"]),
                      "attention_mask": torch.equal(captured["attention_mask"][0], target["attention_mask"])}
            filename = output / name
            save_file({"projected_conditioning": prediction, "text_ids": positions,
                       "input_ids": captured["input_ids"][0].contiguous(),
                       "attention_mask": captured["attention_mask"][0].contiguous()}, str(filename),
                      metadata={"protocol": h.PROTOCOL, "reference_sha256": case["sha256"]})
            row = {"id": case["id"], "equal": equals, "target_sha256": case["sha256"],
                   "reference_sha256": h.sha256(filename), "seconds": time.perf_counter() - started,
                   "all": h.metrics(prediction, target["projected_conditioning"]),
                   "real": h.metrics(prediction[mask], target["projected_conditioning"][mask]),
                   "padding": h.metrics(prediction[~mask], target["projected_conditioning"][~mask])}
            report["cases"].append(row)
            print(case["id"], "equal=" + str(all(equals.values())), "seconds=" + str(round(row["seconds"], 2)), flush=True)
    finally:
        hook.remove()
    report["all_exact"] = bool(report["cases"]) and all(all(c["equal"].values()) for c in report["cases"])
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    result = compare(args.model, args.targets, args.output, args.threads)
    raise SystemExit(0 if result["all_exact"] else 1)


if __name__ == "__main__":
    main()
