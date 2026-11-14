# Generate smoke answers with Qwen's non-thinking text preset; greedy is an explicit diagnostic.
import argparse
import json
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SERVER = ROOT / "bench/tools/llama-b10673/llama-server.exe"
MODELS = {
    "stock": ROOT / "models/stock/Qwen3.5-2B-BF16.gguf",
    "mlp75": ROOT / "models/manifold-compression/mc-2b-mlp75.gguf",
    "mlp50-selfchat": ROOT / "models/manifold-compression/mc-2b-mlp50-selfchat.gguf",
    "stock08": Path.home() / ".cache/huggingface/hub/models--unsloth--Qwen3.5-0.8B-GGUF/snapshots/6ab461498e2023f6e3c1baea90a8f0fe38ab64d0/Qwen3.5-0.8B-Q8_0.gguf",
    "proj08": ROOT / "models/manifold-compression/mc-proj-0.8b-tail.gguf",
}
# Part-swap variants from m_part_swap.py; these run with 4 parallel slots.
MODELS |= {f"ps-{v}": ROOT / f"models/manifold-compression/part-swap/ps-{v}.gguf"
           for v in ("ours", "stock", "mlp", "attn", "io", "ctrl-mlp", "ctrl-attn", "ctrl-io", "ctrl-all")}
MODELS |= {"ours9": ROOT / "models/manifold-compression/mc-9b-4b-tail-q8.gguf",
           "stock4": ROOT / "models/stock9/Qwen3.5-4B-Q8_0.gguf", "stock9": ROOT / "models/stock9/Qwen3.5-9B-Q8_0.gguf"}
MODELS |= {f"di-{a}": ROOT / f"models/damage-isolation/di-{a}.gguf" for a in (
    "F_a32", "F_a128", "F_a512", "F_or", "M_a32_f32", "M_a128_f8", "M_a128_f32", "M_a128_f128", "M_a512_f32", "M_or", "FM_a128_f32", "FM_or")}
MODELS |= {f"d{m}M": ROOT / f"models/manifold-compression/distill-08b/ckpt_{m}M.gguf" for m in (5, 20, 50, 100, 200, 350, 500)}
MODELS |= {f"d{m}M": ROOT / f"models/manifold-compression/distill-08b/continuation-500m/ckpt_{m}M.gguf"
           for m in (250, 300, 350, 400, 450, 500)}
SLOTS = 4
# Qwen3.5 model card, non-thinking text tasks.
SAMPLED = {"temperature": 1.0, "top_p": 1.0, "top_k": 20, "min_p": 0.0, "presence_penalty": 2.0, "repeat_penalty": 1.0}
PORT = 8093
prompts = [json.loads(l) for l in (HERE / "prompts.jsonl").open(encoding="utf-8")]
parser = argparse.ArgumentParser(description="Generate smoke answers with Qwen-recommended sampling (default), or --greedy for diagnostics")
parser.add_argument("models", nargs="*")
parser.add_argument("--greedy", action="store_true", help="Use greedy decoding for a separate diagnostic; default is Qwen's recommended preset")
parser.add_argument("--dry-run", action="store_true", help="Print model, output path and decoding settings without starting a server")
parser.add_argument("--tag", default="", help="Separate output names for a new evaluation run")
parser.add_argument("--seed-offset", type=int, default=0)
parser.add_argument("--slots", type=int, choices=range(1, 5), help="Override concurrent server slots; retain 4096 context tokens per slot")
parser.add_argument("--skip-chat-parsing", action="store_true", help="Preserve raw output, including malformed chat syntax")
args = parser.parse_args()
if args.tag and any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in args.tag):
    parser.error("tag must contain only letters, digits, underscores or hyphens")
if args.greedy and any(model.endswith(("-sampled", "-sampled0")) for model in args.models):
    parser.error("--greedy cannot be combined with an explicitly sampled model suffix")


def post(body):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=600).read())


for arg in args.models or MODELS:
    # Bare names now use Qwen's preset. Keep legacy sampled suffixes and output names compatible.
    nopen = arg.endswith("-sampled0")
    name = arg.removesuffix("-sampled0").removesuffix("-sampled")
    if name not in MODELS:
        parser.error(f"unknown model: {name}")
    sampled = not args.greedy
    sampling = (SAMPLED | ({"presence_penalty": 0.0} if nopen else {})) if sampled else {"temperature": 0}
    variant = name + ("-sampled0" if nopen else "-sampled") if sampled else name
    label = variant + ("_" + args.tag if args.tag else "")
    out = HERE / f"answers_{label}.jsonl"
    if args.dry_run:
        print(json.dumps({"model": name, "model_path": str(MODELS[name]), "output": str(out),
                          "sampling": sampling, "thinking": False, "max_tokens": 400,
                          "seed_rule": "prompt_id + seed_offset" if sampled else None,
                          "seed_offset": args.seed_offset}))
        continue
    done = {json.loads(l)["id"] for l in out.open(encoding="utf-8")} if out.exists() else set()
    slots = args.slots or (SLOTS if name.startswith(("ps-", "di-", "ours9", "stock4", "stock9")) else 1)
    context = 4096 * slots if args.slots else 2048 * max(2, slots)
    srv = subprocess.Popen([str(SERVER), "-m", str(MODELS[name]), "--port", str(PORT), "-c", str(context),
                            "-np", str(slots), "-ngl", "99"] + (["--skip-chat-parsing"] if args.skip_chat_parsing else []),
                           stdout=subprocess.DEVNULL, stderr=open(HERE / f"server_{label}.log", "a" if done else "w"))
    metadata = {"model": name, "model_path": str(MODELS[name]), "server": str(SERVER),
                "max_tokens": 400, "thinking": False, "slots": slots, "context": context,
                "sampling": sampling,
                "seed_rule": "prompt_id + seed_offset" if sampled else None, "seed_offset": args.seed_offset,
                "skip_chat_parsing": args.skip_chat_parsing, "started_unix": time.time()}
    metadata_path = HERE / f"metadata_{label}.json"
    if done and metadata_path.exists():
        metadata["previous_run"] = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["resumed_answer_ids"] = sorted(done)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    def answer(p):
        t0 = time.time()
        body = {"messages": [{"role": "user", "content": p["prompt"]}], "max_tokens": 400,
                "chat_template_kwargs": {"enable_thinking": False}} | sampling
        if sampled:
            body["seed"] = p["id"] + args.seed_offset
        r = post(body)
        return {"id": p["id"], "answer": r["choices"][0]["message"]["content"], "finish": r["choices"][0]["finish_reason"],
                "tokens": r["usage"]["completion_tokens"], "secs": round(time.time() - t0, 1)}

    try:
        for _ in range(120):
            try:
                if b"ok" in urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2).read():
                    break
            except Exception:
                time.sleep(2)
        todo = [p for p in prompts if p["id"] not in done]
        with out.open("a", encoding="utf-8") as f, ThreadPoolExecutor(slots) as pool:
            for row in pool.map(answer, todo):
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                print(arg, row["id"], row["tokens"], flush=True)
    finally:
        srv.kill()
