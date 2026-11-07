"""Yield to the active DFlash2 collection and queued timing before teacher CPU work."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/out/doc-teacher-pilot-v1"
PAIR = ROOT.parent / "pair-cache-spec/results"
BROAD = PAIR / "ternary-dflash2-broad-20261025"
CONTROLLED = PAIR / "ternary-dflash2-controlled-20261026"
COMMAND = [shutil.which("dibs"), "run", "cpu", "port-8765",
           "--label", "ASD STE100 Bonsai2 doc synthesis pilot", "--eta", "45m", "--",
           sys.executable, "-u", str(Path(__file__).with_name("run_local.py")), "--limit", "12"]


def state(stage: str, **extra):
    value = {"stage": stage, "updated_epoch": time.time(), **extra}
    tmp = OUT / "teacher-queue-status.tmp"
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(OUT / "teacher-queue-status.json")


def read(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def main():
    started = time.monotonic()
    while time.monotonic() - started < 12 * 3600:
        broad = read(BROAD / "report.json")
        timed = read(CONTROLLED / "status.json")
        if broad.get("complete") and timed.get("phase") in ("complete", "complete_with_noisy_targets"):
            break
        state("waiting_for_dflash_queue", broad_complete=bool(broad.get("complete")),
              timing_phase=timed.get("phase"))
        time.sleep(60)
    else:
        state("timeout", reason="DFlash2 collection/timing queue did not finish in 12h")
        raise TimeoutError("DFlash2 collection/timing queue did not finish in 12h")
    state("dibs_launching")
    for attempt in range(2):
        result = subprocess.run(COMMAND, cwd=ROOT.parent, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if result.returncode == 0:
            state("complete", dibs_output=result.stdout[-1000:])
            return
        if "WAIT" in result.stdout and attempt == 0:
            state("waiting_for_dibs_measurement", dibs_output=result.stdout[-1000:])
            subprocess.run([shutil.which("dibs"), "wait", "--measure"], check=True)
            continue
        state("failed", exit_code=result.returncode, dibs_output=result.stdout[-2000:])
        raise RuntimeError("Dibs teacher pilot failed; see queue status")


if __name__ == "__main__":
    main()
