"""Generate and embed the fixed Nanbeige pilot under one Dibs GPU+port run.

This program does not acquire Dibs itself. Invoke it as:
  dibs run gpu port-8400 --label "Nanbeige microdrafter capture" --eta 45m -- python run_capture.py
"""

from __future__ import annotations

import argparse
import contextlib
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import capture


ROOT = Path(__file__).resolve().parent
SERVER = Path("/home/cyrus/code/phlegm/target/npu-dev/xing/src/llama.cpp-7d7f0287530bc8981690f361d8c1627dbae1c99a/build-vk/bin/llama-server")
MODEL = Path("/home/cyrus/models/Nanbeige4.3-33B-Q4pure-NPU2/model.gguf")
TRAIN_PROMPTS = Path("/home/cyrus/code/model-experiments/bench/data/ifeval.jsonl")
EVAL_PROMPTS = Path("/home/cyrus/code/openflowlm-next/utilities/dflash-compare/prompts.jsonl")
URL = "http://127.0.0.1:8400"


@contextlib.contextmanager
def server(embedding: bool, output: Path, binary: Path, threads: int):
    flags = [str(binary), "-m", str(MODEL), "-ngl", "999", "--host", "127.0.0.1",
             "--port", "8400", "-c", "2048", "-b", "512", "-ub", "512", "-np", "1",
             "-t", str(threads)]
    if embedding:
        flags += ["--embedding", "--pooling", "none", "--embd-normalize", "-1"]
    with output.open("w") as log:
        process = subprocess.Popen(flags, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 180
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"llama-server exited with {process.returncode}; see {output}")
                try:
                    with urllib.request.urlopen(URL + "/health", timeout=2) as response:
                        if response.status == 200:
                            break
                except Exception:
                    time.sleep(1)
            else:
                raise TimeoutError(f"llama-server did not become healthy; see {output}")
            yield
        finally:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-limit", type=int, default=64)
    parser.add_argument("--eval-limit", type=int, default=20)
    parser.add_argument("--tokens", type=int, default=96)
    parser.add_argument("--server", type=Path, default=SERVER)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--embed-only", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "data/pilot")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.server.is_file() or not MODEL.is_file():
        raise FileNotFoundError("Nanbeige server or target GGUF is unavailable")
    train_generated = args.output / "train-generated.jsonl"
    eval_generated = args.output / "eval-generated.jsonl"
    if args.embed_only:
        if not train_generated.is_file() or not eval_generated.is_file():
            raise FileNotFoundError("embed-only requires completed generated JSONL files")
    else:
        with server(False, args.output / "generation-server.log", args.server, args.threads):
            for source, limit, destination in ((TRAIN_PROMPTS, args.train_limit, train_generated),
                                               (EVAL_PROMPTS, args.eval_limit, eval_generated)):
                capture.generate(argparse.Namespace(prompts=source, start=0, limit=limit,
                                                    tokens=args.tokens, output=destination, url=URL))
    with server(True, args.output / "embedding-server.log", args.server, args.threads):
        for generated, destination, stride in ((train_generated, args.output / "train.npz", 1),
                                               (eval_generated, args.output / "eval.npz", 7)):
            capture.embed(argparse.Namespace(generated=generated, output=destination, url=URL,
                                             stride=stride))
    print(f"capture complete: {args.output}", flush=True)


if __name__ == "__main__":
    main()
