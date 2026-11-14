#!/usr/bin/env python3
"""Run the complete free local distillation pilot, coordinating each Dibs phase."""
from __future__ import annotations
import datetime as dt
import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def save(status):
    temp = ROOT / "status.json.tmp"
    temp.write_text(json.dumps(status, indent=2) + "\n")
    temp.replace(ROOT / "status.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-phase", choices=["label", "train", "evaluate"], default="label")
    args = parser.parse_args()
    if args.from_phase != "label":
        subprocess.run([sys.executable, str(ROOT / "finalize_annotations.py")], check=True)
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGTERM, interrupted)
    python_gpu = ROOT / "rocm-env/bin/python"
    check = json.loads((ROOT / "gpu-runtime-check.json").read_text())
    smoke = json.loads((ROOT / "work/gpu-backward-smoke/result.json").read_text())
    if not check["bf16_backward_finite"] or smoke["steps_completed"] != 1 or not smoke["changed_adapter_tensors"]:
        raise ValueError("GPU and real student backward checks must pass before the pilot")
    phases = [
        ("label", ["gpu", "port-8400"], sys.executable, "90m", []),
        ("train", ["gpu"], str(python_gpu), "45m", ["--device", "cuda"]),
        ("evaluate", ["gpu"], str(python_gpu), "90m", ["--device", "cuda"]),
    ]
    phases = phases[[p[0] for p in phases].index(args.from_phase):]
    status = {"started_utc": now(), "pid": os.getpid(), "state": "running", "phase": None,
              "completed_phases": [], "paid_compute_cost_usd": 0,
              "protocol": "protocol.json", "interpretation": "Pilot supervision remains proposed, not independently certified truth."}
    status["from_phase"] = args.from_phase
    status["input_hashes"] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in [ROOT / "protocol.json", ROOT / "data/manifest.json", *sorted(ROOT.glob("*.py"))]}
    if args.from_phase != "label":
        status["input_hashes"]["labels/labels.jsonl"] = hashlib.sha256((ROOT / "labels/labels.jsonl").read_bytes()).hexdigest()
    save(status)
    child = None
    try:
        for phase, resources, python, eta, extra in phases:
            command = ["dibs", "run", *resources, "--label", f"XDNA2 claim-distillation {phase}",
                       "--eta", eta, "--", python, str(ROOT / "pilot.py"), phase, *extra]
            status.update(phase=phase, phase_started_utc=now(), command=command)
            save(status)
            print("START", phase, now(), flush=True)
            with (ROOT / f"{phase}.log").open("a") as log:
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                status["child_pid"] = child.pid
                save(status)
                rc = child.wait()
            child = None
            if rc:
                raise RuntimeError(f"{phase} exited {rc}; see {phase}.log")
            if phase == "label":
                subprocess.run([sys.executable, str(ROOT / "finalize_annotations.py")], check=True)
                status["input_hashes"]["labels/labels.jsonl"] = hashlib.sha256((ROOT / "labels/labels.jsonl").read_bytes()).hexdigest()
            status["completed_phases"].append(phase)
            save(status)
            print("DONE", phase, now(), flush=True)
        status.update(state="complete", finished_utc=now())
        save(status)
    except BaseException as exc:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        status.update(state="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                      error=str(exc), finished_utc=now())
        save(status)
        raise


if __name__ == "__main__":
    main()
