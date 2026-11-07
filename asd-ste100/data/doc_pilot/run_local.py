"""Run a small pinned Bonsai2 teacher batch on a private localhost server."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/out/doc-teacher-pilot-v1"
MODEL = Path("/home/cyrus/code/openflowlm-next/build/dflash-compare/models/target.gguf")
SERVER = Path("/tmp/memdraft-dflash-prism/build/bin/llama-server")
MODEL_SHA = "3907dc1658db1f78a9826bf8d5bcb8dc65db0d466388937af57f2294fae62ec1"
SERVER_SHA = "c834643d20fb348c94fba033b290ec92e992aeea7f6a1ff9d97b5db04fb973de"
PORT = 8765


def sha(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def status(stage: str, **extra):
    value = {"stage": stage, "time_epoch": time.time(), **extra}
    tmp = OUT / "teacher-run-status.tmp"
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(OUT / "teacher-run-status.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=12)
    args = parser.parse_args()
    if not 1 <= args.limit <= 36:
        raise ValueError("limit must be 1..36")
    subprocess.run([sys.executable, str(Path(__file__).with_name("validate_sources.py"))],
                   cwd=ROOT.parent, check=True, stdout=subprocess.DEVNULL)
    if not MODEL.is_file() or MODEL.stat().st_size != 7206168928 or sha(MODEL) != MODEL_SHA:
        raise ValueError("Bonsai2 file differs from pinned model")
    if not SERVER.is_file() or sha(SERVER) != SERVER_SHA:
        raise ValueError("local llama-server differs from pinned runtime")
    cmd = [str(SERVER), "-m", str(MODEL), "--host", "127.0.0.1", "--port", str(PORT),
           "-ngl", "0", "-t", "8", "-tb", "8", "-np", "1", "-c", "4096",
           "--offline", "--no-warmup", "--jinja"]
    metadata = {"server_command": cmd, "server_sha256": SERVER_SHA,
                "model_sha256": MODEL_SHA, "synthesis_limit": args.limit,
                "model_notice_manifest": str(OUT / "model-notice/manifest.json")}
    (OUT / "teacher-runtime.json").write_text(json.dumps(metadata, indent=2) + "\n")
    server = None
    try:
        status("server_starting", limit=args.limit)
        with (OUT / "teacher-server.log").open("w") as log:
            server = subprocess.Popen(cmd, cwd=ROOT.parent, stdout=log, stderr=subprocess.STDOUT)
            deadline = time.monotonic() + 300
            while time.monotonic() < deadline:
                if server.poll() is not None:
                    raise RuntimeError(f"server exited with code {server.returncode}")
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3) as response:
                        if response.status == 200:
                            break
                except Exception:
                    pass
                time.sleep(2)
            else:
                raise TimeoutError("Bonsai2 server did not become ready")
            status("synthesizing", limit=args.limit, server_pid=server.pid)
            with (OUT / "teacher-synthesis.log").open("a") as synthesis_log:
                subprocess.run([sys.executable, "-u", str(Path(__file__).with_name("synthesize.py")),
                                "--url", f"http://127.0.0.1:{PORT}", "--limit", str(args.limit)],
                               cwd=ROOT.parent, stdout=synthesis_log, stderr=subprocess.STDOUT, check=True)
            if args.limit == 12:
                subprocess.run([sys.executable, str(Path(__file__).with_name("review_packet.py"))],
                               cwd=ROOT.parent, check=True)
            summary = json.loads((OUT / "synthesis-summary.json").read_text())
            status("complete", limit=args.limit, accepted=summary["accepted"], rejected=summary["rejected"])
    except Exception as exc:
        status("failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        if server is not None and server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=20)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=10)


if __name__ == "__main__":
    main()
