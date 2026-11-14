#!/usr/bin/env python3
"""Repeatable local pilot. Invoke label under Dibs gpu+port-8400; train/evaluate under cpu.

No hosted inference or paid infrastructure is used. Teacher weights must already
be downloaded and hash-verified; the existing student download is pinned too.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS = ROOT.parents[1] / "models/claim-curation"


def invoke(name, *args):
    subprocess.run([sys.executable, str(ROOT / name), *map(str, args)], check=True)


def label(args):
    from teacher import generate
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGTERM, interrupted)
    checkpoint = MODELS / "teacher-qwen35-27b"
    meta = json.loads((checkpoint / "provenance.json").read_text())
    weights = checkpoint / meta["filename"]
    h = hashlib.sha256()
    with weights.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    if h.hexdigest() != meta["sha256"]:
        raise ValueError("Teacher weight digest differs from pinned provenance")
    out = ROOT / "labels"
    out.mkdir(exist_ok=True)
    with (out / "server.log").open("a") as log:
        command = [str(Path.home() / ".local/bin/llama"), "serve", "-m", str(weights),
                   "--alias", meta["repo"], "--host", "127.0.0.1", "--port", "8400",
                   "-ngl", "all", "-c", "32768", "-np", "2", "-t", "8", "-b", "512",
                   "--reasoning", "off", "--no-webui", "--no-perf"]
        server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 300
            while True:
                if server.poll() is not None:
                    raise RuntimeError("Teacher server exited; see labels/server.log")
                try:
                    with urllib.request.urlopen("http://127.0.0.1:8400/health", timeout=2) as r:
                        if r.status == 200:
                            break
                except Exception:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Teacher server did not become ready")
                    time.sleep(1)
            config = argparse.Namespace(data=ROOT / "data", out=out,
                         endpoint="http://127.0.0.1:8400/v1", teacher_id=meta["repo"],
                         teacher_revision=meta["revision"] + ":" + meta["sha256"],
                         splits=["train"], max_tokens=3072)
            rows = generate(config)
            summary = {s: {"eligible": sum(r["split"] == s and r["training_eligible"] for r in rows),
                           "total": sum(r["split"] == s for r in rows)} for s in config.splits}
            (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            print(json.dumps(summary, indent=2), flush=True)
        finally:
            server.terminate()
            try:
                server.wait(timeout=20)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()


def run(args):
    if args.phase == "label":
        return label(args)
    os.environ["PYTHONPATH"] = str(ROOT / "vendor") + os.pathsep + os.environ.get("PYTHONPATH", "")
    common = ["--checkpoint", MODELS / "nuextract3", "--labels", ROOT / "labels/labels.jsonl"]
    if args.phase == "train":
        invoke("train_lora.py", *common, "--out", ROOT / "runs/lora-pilot", "--device", args.device,
               "--steps", "24", "--threads", "8")
    else:
        invoke("evaluate.py", *common, "--out", ROOT / "runs/base-test", "--device", args.device)
        invoke("evaluate.py", *common, "--adapter", ROOT / "runs/lora-pilot/final",
               "--out", ROOT / "runs/adapter-test", "--device", args.device)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=["label", "train", "evaluate"])
    p.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    run(p.parse_args())
