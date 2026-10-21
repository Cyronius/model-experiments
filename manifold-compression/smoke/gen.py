# Generate greedy answers to the smoke prompts with each GGUF model through llama-server (2026-10-19).
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
SLOTS = 4
# Qwen3.5 model card, non-thinking text tasks.
SAMPLED = {"temperature": 1.0, "top_p": 1.0, "top_k": 20, "min_p": 0.0, "presence_penalty": 2.0, "repeat_penalty": 1.0}
PORT = 8093
prompts = [json.loads(l) for l in (HERE / "prompts.jsonl").open(encoding="utf-8")]


def post(body):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=600).read())


for arg in sys.argv[1:] or MODELS:
    # "<model>-sampled" uses the model card settings; a bare name is greedy.
    nopen = arg.endswith("-sampled0")
    name, sampled = arg.removesuffix("-sampled0").removesuffix("-sampled"), arg.endswith(("-sampled", "-sampled0"))
    out = HERE / f"answers_{arg}.jsonl"
    done = {json.loads(l)["id"] for l in out.open(encoding="utf-8")} if out.exists() else set()
    slots = SLOTS if name.startswith("ps-") else 1
    srv = subprocess.Popen([str(SERVER), "-m", str(MODELS[name]), "--port", str(PORT), "-c", str(2048 * max(2, slots)),
                            "-np", str(slots), "-ngl", "99"],
                           stdout=subprocess.DEVNULL, stderr=open(HERE / f"server_{arg}.log", "w"))

    def answer(p):
        t0 = time.time()
        body = {"messages": [{"role": "user", "content": p["prompt"]}], "max_tokens": 400, "temperature": 0,
                "chat_template_kwargs": {"enable_thinking": False}}
        if sampled:
            body |= SAMPLED | {"seed": p["id"]}
            if nopen:
                body["presence_penalty"] = 0.0
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
