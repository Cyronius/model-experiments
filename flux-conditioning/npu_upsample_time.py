"""Stock enrichment baseline on the NPU: FLUX.2's upsample_prompt with Qwen3-4B-Instruct through FastFlowLM; time per prompt (2026-11-20)."""
from __future__ import annotations

import json
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, r"C:\code\model-experiments\artifacts\flux-conditioning\venv-diffusers\Lib\site-packages")
from diffusers.pipelines.flux2.system_messages import SYSTEM_MESSAGE_UPSAMPLING_T2I  # noqa: E402

A = Path(r"C:\code\model-experiments\artifacts\flux-conditioning")
PORT = 52625


def ask(prompt: str) -> dict:
    body = {"model": "qwen3-it:4b", "stream": True, "temperature": 0.15, "max_tokens": 512,
            "messages": [{"role": "system", "content": SYSTEM_MESSAGE_UPSAMPLING_T2I}, {"role": "user", "content": prompt}]}
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.perf_counter()
    first, chunks, text, usage = None, 0, [], None
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line.endswith("[DONE]"):
                continue
            d = json.loads(line[5:])
            usage = d.get("usage") or usage
            for c in d.get("choices", []):
                piece = (c.get("delta") or {}).get("content")
                if piece:
                    first = first or time.perf_counter()
                    chunks += 1
                    text.append(piece)
    end = time.perf_counter()
    n = (usage or {}).get("completion_tokens", chunks)
    return {"prompt": prompt, "ttft_s": round(first - t0, 3), "total_s": round(end - t0, 3), "tokens": n,
            "decode_tok_s": round((n - 1) / (end - first), 2) if n > 1 else None, "usage": usage, "text": "".join(text)}


def main() -> None:
    cases = json.loads((A / "encoder-probe/run1/targets/manifest.json").read_text(encoding="utf-8"))["cases"]
    srv = subprocess.Popen(["flm", "serve", "qwen3-it:4b", "--port", str(PORT), "--quiet"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(120):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=2)
                break
            except OSError:
                time.sleep(1)
        ask("a red fox in snow")
        rows = [ask(c["prompt"]) for c in cases]
    finally:
        srv.terminate()
        srv.wait(timeout=60)
    out = A / "npu-upsample"
    out.mkdir(exist_ok=True)
    (out / "results.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    for r in rows:
        print(f"{r['total_s']:6.2f} s  ttft {r['ttft_s']:.2f}  {r['tokens']} tok  {r['decode_tok_s']} tok/s  {r['prompt'][:40]}")
    med = lambda k: statistics.median(r[k] for r in rows)  # noqa: E731
    print(f"median: total {med('total_s'):.2f} s, ttft {med('ttft_s'):.2f} s, tokens {med('tokens')}, "
          f"decode {med('decode_tok_s')} tok/s")


if __name__ == "__main__":
    main()
