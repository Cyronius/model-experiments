"""Run inside the disposable GPU pod; contains no RunPod credential."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent


def status(stage, **extra):
    path = ROOT / "results/status.json"
    path.parent.mkdir(exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"stage": stage, "updated_at": time.time(), **extra}, indent=2))
    tmp.replace(path)
    print(stage, extra, flush=True)


def main():
    deadline = float(os.environ["SCALE_JOB_DEADLINE"])
    def expired(signum, frame):
        raise TimeoutError("Authorized job deadline reached")
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(max(1, int(deadline - time.time())))
    manifest = json.loads((ROOT / "bundle-manifest.json").read_text())
    for name, digest in manifest["sha256"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Bundle checksum mismatch: {name}")
    from huggingface_hub import snapshot_download
    from scale_probe import SOURCES
    completed = []
    status("starting", completed=completed)
    (ROOT / "results/environment.txt").write_text(subprocess.check_output(
        [sys.executable, "-m", "pip", "freeze"], text=True))
    (ROOT / "results/gpu.txt").write_text(subprocess.check_output(["nvidia-smi"], text=True))
    # Small control first detects kernel/runtime issues cheaply, then the two
    # large architectures. Every run contains stock / 75% / 50% arms.
    for name in ("qwen2", "qwen27", "mistral24"):
        status("downloading", model=name, completed=completed)
        source = SOURCES[name]
        folder = ROOT / "models" / name
        snapshot_download(source["repo"], revision=source["revision"], local_dir=folder,
            allow_patterns=["config.json", "generation_config.json", "tokenizer*", "special_tokens_map.json",
                            "chat_template*", "model*.safetensors", "model.safetensors.index.json"])
        (folder / "source.json").write_text(json.dumps(source, indent=2) + "\n")
        status("evaluating", model=name, completed=completed)
        remaining = int(deadline - time.time())
        if remaining < 120:
            raise TimeoutError("Too little authorized time remains to start another model")
        with (ROOT / "results" / f"{name}.log").open("w") as log:
            subprocess.run([sys.executable, "-u", str(ROOT / "scale_probe.py"), "run", "--model", name,
                "--source", str(folder), "--data", str(ROOT / "data.json"),
                "--out", str(ROOT / "results" / name), "--threads", "8", "--batch-size", "4"],
                stdout=log, stderr=subprocess.STDOUT, timeout=remaining, check=True)
        completed.append(name)
        status("model complete", model=name, completed=completed)
    results = ROOT / "results"
    artifacts = {str(p.relative_to(results)): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in results.rglob("*") if p.is_file() and "teacher_logits" not in p.parts
                 and p.suffix != ".log" and p.name != "status.json"}
    (results / "artifact-manifest.json").write_text(json.dumps(artifacts, indent=2) + "\n")
    status("complete", completed=completed)


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        status("failed", error=f"{type(error).__name__}: {error}")
        raise
