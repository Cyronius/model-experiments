"""Run the v3 teacher pilot on the pinned Bonsai2 runtime, private localhost only.

Same model, binary, hash checks and server options as v2; the default is all 36
teacher inputs, and the review packet covers every passage that was run.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run_local import MODEL, MODEL_SHA, PORT, ROOT, SERVER, SERVER_SHA, sha  # noqa: E402

OUT = ROOT / "data/out/doc-teacher-pilot-v3"
CTX = 40960


def status(stage: str, **extra):
    value = {"stage": stage, "time_epoch": time.time(), **extra}
    tmp = OUT / "teacher-run-status.tmp"
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(OUT / "teacher-run-status.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=36)
    args = parser.parse_args()
    if not 1 <= args.limit <= 36:
        raise ValueError("limit must be 1..36")
    (OUT / "slots").mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, str(Path(__file__).with_name("validate_sources.py"))],
                   cwd=ROOT.parent, check=True, stdout=subprocess.DEVNULL)
    if not MODEL.is_file() or MODEL.stat().st_size != 7206168928 or sha(MODEL) != MODEL_SHA:
        raise ValueError("Bonsai2 file differs from pinned model")
    if not SERVER.is_file() or sha(SERVER) != SERVER_SHA:
        raise ValueError("local llama-server differs from pinned runtime")
    cmd = [str(SERVER), "-m", str(MODEL), "--host", "127.0.0.1", "--port", str(PORT),
           "-ngl", "0", "-t", "8", "-tb", "8", "-np", "1", "-c", str(CTX),
           "--offline", "--no-warmup", "--jinja", "--cache-ram", "0",
           "--slot-save-path", str(OUT / "slots")]
    (OUT / "teacher-runtime.json").write_text(json.dumps(
        {"server_command": cmd, "server_sha256": SERVER_SHA, "model_sha256": MODEL_SHA,
         "synthesis_limit": args.limit}, indent=2) + "\n")
    server = None
    try:
        status("server_starting", limit=args.limit)
        with (OUT / "teacher-server.log").open("a") as log:
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
                subprocess.run([sys.executable, "-u", str(Path(__file__).with_name("synthesize_v3.py")),
                                "--url", f"http://127.0.0.1:{PORT}", "--limit", str(args.limit)],
                               cwd=ROOT.parent, stdout=synthesis_log, stderr=subprocess.STDOUT, check=True)
            if args.limit % 3 == 0:
                subprocess.run([sys.executable, str(Path(__file__).with_name("review_packet.py")), "--out", str(OUT),
                                "--limit", str(args.limit)], cwd=ROOT.parent, check=True)
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
