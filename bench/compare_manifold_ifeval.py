"""Paired local IFEval comparison for manifold-compression checkpoints."""
import argparse
import hashlib
import importlib.metadata
import json
from itertools import combinations
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import run_bench as bench

ROOT = Path(__file__).resolve().parents[1]


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=int, help="Legacy single-checkpoint selection")
    ap.add_argument("--checkpoints", type=int, nargs="+", help="Compare these checkpoints with stock")
    ap.add_argument("--sampling", choices=("qwen", "greedy"), default="qwen")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--runid", default="ifeval-stock08-300M-500M-qwen-20261023")
    args = ap.parse_args()
    if args.checkpoint is not None and args.checkpoints:
        ap.error("Use --checkpoint or --checkpoints, not both")
    checkpoints = args.checkpoints or ([args.checkpoint] if args.checkpoint is not None else [300, 500])
    if len(set(checkpoints)) != len(checkpoints):
        ap.error("Checkpoint selections must be unique")
    out = ROOT / "manifold-compression" / "results" / args.runid
    out.mkdir(parents=True, exist_ok=True)
    server = ROOT / "bench/tools/llama-b10673/llama-server.exe"
    models = {
        "stock-0.8B": Path.home() / ".cache/huggingface/hub/models--unsloth--Qwen3.5-0.8B-GGUF/snapshots/6ab461498e2023f6e3c1baea90a8f0fe38ab64d0/Qwen3.5-0.8B-Q8_0.gguf",
    }
    for checkpoint in checkpoints:
        base = ROOT / "models/manifold-compression/distill-08b"
        if checkpoint >= 250:
            base /= "continuation-500m"
        models[f"ours-{checkpoint}M"] = base / f"ckpt_{checkpoint}M.gguf"
    for path in [server, *models.values()]:
        if not path.is_file():
            raise FileNotFoundError(path)
    from langdetect import DetectorFactory
    DetectorFactory.seed = 0
    from lm_eval.tasks.ifeval import utils
    dataset = ROOT / "bench/data/ifeval.jsonl"
    docs = [json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit:
        docs = docs[:args.limit]
    # Exercise every instruction type before starting either model.
    for doc in docs:
        utils.process_results(doc, ["This is a sentence.\n\nA second paragraph."])
    bench.PORT = 8201
    bench.BASE_URL = f"http://127.0.0.1:{bench.PORT}"
    original_post = bench.post_chat
    sampling = {"temperature": 1.0, "top_p": 1.0, "top_k": 20, "min_p": 0.0,
                "repeat_penalty": 1.0, "presence_penalty": 2.0, "frequency_penalty": 0.0}
    if args.sampling == "greedy":
        sampling |= {"temperature": 0, "presence_penalty": 0.0}
    seeds = {doc["prompt"]: int(doc["key"]) for doc in docs}

    def post(payload):
        payload = payload | sampling | {"seed": seeds[payload["messages"][0]["content"]]}
        for attempt in range(3):
            try:
                return original_post(payload)
            except (bench.urllib.error.URLError, TimeoutError, ConnectionError):
                if attempt == 2:
                    raise
                time.sleep(2 * (attempt + 1))

    bench.post_chat = post
    manifest = {
        "runid": args.runid, "started_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_million_tokens": checkpoints,
        "dataset": str(dataset), "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "n_prompts": len(docs), "n_instructions": sum(len(d["instruction_id_list"]) for d in docs),
        "server": str(server), "server_version": subprocess.run([str(server), "--version"], capture_output=True, text=True).stdout.strip(),
        "lm_eval_version": importlib.metadata.version("lm_eval"),
        "settings": {**sampling, "seed_rule": "IFEval document key, identical across models", "enable_thinking": False,
                     "max_tokens": bench.CAPS["ifeval"], "parallel": bench.PARALLEL,
                     "total_context": 16384, "gpu_layers": 99, "sampling_preset": args.sampling,
                     "skip_chat_parsing": True, "repeat_last_n": 64},
        "models": {name: {"gguf": str(path), "size_bytes": path.stat().st_size, "quantization": "Q8_0"} for name, path in models.items()},
    }
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        for field in ("dataset_sha256", "n_prompts", "n_instructions", "settings", "models", "lm_eval_version", "server"):
            if previous.get(field) != manifest[field]:
                raise RuntimeError(f"Existing run differs in {field}; use a new --runid")
        manifest = previous
    else:
        save(manifest_path, manifest)
    summaries, all_rows = {}, {}
    for name, path in models.items():
        work = out / name / "ifeval"
        work.mkdir(parents=True, exist_ok=True)
        expected_keys = {str(doc["key"]) for doc in docs}
        for cache in work.glob("*.json"):
            cached = json.loads(cache.read_text(encoding="utf-8"))
            if cache.stem not in expected_keys:
                raise RuntimeError(f"Unexpected cached item: {cache}")
            if "error" in cached:
                cache.rename(cache.with_suffix(f".error-{time.time_ns()}"))
        result_path = out / f"result-{name}.json"
        if result_path.exists() and (out / f"runtime-{name}.json").exists():
            cached_rows = [json.loads((work / f"{doc['key']}.json").read_text(encoding="utf-8"))
                           for doc in docs if (work / f"{doc['key']}.json").exists()]
            if len(cached_rows) == len(docs) and all("error" not in row for row in cached_rows):
                summaries[name] = json.loads(result_path.read_text(encoding="utf-8"))
                all_rows[name] = {row["key"]: row for row in cached_rows}
                bench.log(f"Reusing completed, validated {name}")
                continue
        # Refuse to disturb another application already using this port.
        with socket.socket() as probe:
            if probe.connect_ex(("127.0.0.1", bench.PORT)) == 0:
                raise RuntimeError(f"Port {bench.PORT} is occupied")
        bench.log(f"=== {name}: {len(docs)} IFEval prompts ===")
        t0 = time.time()
        with (out / f"server-{name}.log").open("a", encoding="utf-8") as log_file:
            proc = subprocess.Popen(
                [str(server), "-m", str(path), "--host", "127.0.0.1", "--port", str(bench.PORT),
                 "-ngl", "99", "-c", "16384", "-np", str(bench.PARALLEL), "--skip-chat-parsing"],
                stdout=log_file, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
            )
            try:
                deadline = time.time() + 180
                while time.time() < deadline:
                    if proc.poll() is not None:
                        raise RuntimeError(f"Server exited {proc.returncode}; see server-{name}.log")
                    try:
                        with bench.urllib.request.urlopen(bench.BASE_URL + "/health", timeout=2) as response:
                            if b"ok" in response.read():
                                break
                    except Exception:
                        pass
                    time.sleep(1)
                else:
                    raise TimeoutError("Server did not become healthy")
                rows = bench.run_ifeval_suite(work, args.limit)
                with bench.urllib.request.urlopen(bench.BASE_URL + "/slots", timeout=10) as response:
                    runtime = json.load(response)
                save(out / f"runtime-{name}.json", runtime)
                assert len(runtime) == bench.PARALLEL
                for slot in runtime:
                    params = slot["params"]
                    assert all(params[k] == v for k, v in sampling.items()), params
                    assert params["repeat_last_n"] == 64
                    assert params["max_tokens"] == bench.CAPS["ifeval"]
                errors = [row for row in rows if "error" in row]
                if errors:
                    save(out / f"errors-{name}.json", errors)
                    raise RuntimeError(f"{len(errors)} requests/scoring errors for {name}; refusing to publish incomplete scores")
                assert len(rows) == len(docs) and {row["key"] for row in rows} == {doc["key"] for doc in docs}
                doc_by_key = {doc["key"]: doc for doc in docs}
                for row in rows:
                    assert len(row["inst_strict"]) == len(row["inst_loose"]) == len(doc_by_key[row["key"]]["instruction_id_list"])
            finally:
                proc.terminate()
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
        result = bench.aggregate({"ifeval": rows}, {}, {}, name, {"gguf": str(path)}, args.runid)
        total_instructions = sum(len(row["inst_loose"]) for row in rows)
        result["ifeval_inst_loose"] = sum(sum(row["inst_loose"]) for row in rows) / total_instructions
        result["prompt_pass_count"] = sum(row["correct"] for row in rows)
        result["instruction_pass_count"] = sum(sum(row["inst_strict"]) for row in rows)
        result["instruction_count"] = total_instructions
        result["errors"] = 0
        result["wall_seconds"] = round(time.time() - t0, 1)
        save(out / f"result-{name}.json", result)
        summaries[name] = result
        all_rows[name] = {row["key"]: row for row in rows}
        bench.log(json.dumps(result))
    paired = {}
    for first, second in combinations(models, 2):
        a, b = all_rows[first], all_rows[second]
        assert a.keys() == b.keys()
        paired[f"{first} vs {second}"] = {
            "first_model": first, "second_model": second,
            "both_pass": sum(a[k]["correct"] and b[k]["correct"] for k in a),
            "first_only_pass": sum(a[k]["correct"] and not b[k]["correct"] for k in a),
            "second_only_pass": sum(not a[k]["correct"] and b[k]["correct"] for k in a),
            "both_fail": sum(not a[k]["correct"] and not b[k]["correct"] for k in a),
        }
    save(out / "comparison.json", {"models": summaries, "paired_prompt_strict": paired})
    lines = [
        "# IFEval: distilled 0.8B against stock Qwen3.5-0.8B", "",
        f"Checkpoints: {', '.join(str(c) + 'M' for c in checkpoints)} distillation tokens. {len(docs)} prompts; {manifest['n_instructions']} instructions.",
        f"All Q8_0, same llama.cpp binary, {args.sampling} decoding, thinking off, {bench.CAPS['ifeval']}-token response cap.",
        "Sampling seeds are IFEval document keys, shared across models. Four slots, 4096 context tokens each.", "",
        "| Model | Prompt strict | Instruction strict | Prompt loose | Instruction loose | Hit response cap | Repeated 4-grams >15% |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, r in summaries.items():
        cols = ["ifeval_prompt_strict", "ifeval_inst_strict", "ifeval_prompt_loose", "ifeval_inst_loose", "cap_hit_rate", "degenerate_rate"]
        lines.append("| " + name + " | " + " | ".join(f"{100*r[c]:.2f}%" for c in cols) + " |")
    lines += ["", "Cap hits can also reflect legitimately long answers. Repetition can be requested by an instruction.", "",
              "Scores measure verifiable instruction compliance, not general factual correctness or coherence.",
              "One sample per prompt. This is a local comparison, not a reproduction of published model-card benchmark scores.",
              "The 1280-token cap can truncate long requested outputs; use cap-hit rates to interpret scores.", "",
              f"Paired strict prompt outcomes: {json.dumps(paired)}", ""]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    bench.log(f"COMPLETE: {out / 'summary.md'}")


if __name__ == "__main__":
    main()

