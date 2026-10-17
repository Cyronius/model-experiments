import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from common import RESULTS

HERE = Path(__file__).resolve().parent
LOGS = RESULTS / "logs"


def fingerprint_gate():
    fp = json.loads((RESULTS / "fingerprint.json").read_text())
    tern = [t for t in fp["tensors"].values() if "agree" in t]
    linears = [t for t in tern if t["kind"] not in ("embed", "lm_head")]
    n = sum(t["n"] for t in linears)
    agree = sum(t["agree"]["rtn_prism_s"] for t in linears) / max(n, 1)
    if len(linears) == 0 or min(t["exact_groups"] for t in linears) < 0.99:
        return f"ternary decode not exact on linears ({len(linears)} found)"
    # near-chance agreement means a hidden rotation, which this experiment doesn't model
    if agree < 0.6:
        return f"rtn agreement {agree:.3f} suggests a rotated basis"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--start-at", default="tests")
    args = ap.parse_args()
    py, th = sys.executable, ["--threads", str(args.threads)]
    stages = [
        ("tests", "cpu", "5m", [py, str(HERE / "test_quant.py")], None),
        ("fetch", "network", "40m", [py, str(HERE / "fetch.py")], None),
        ("fingerprint", "cpu", "30m", [py, str(HERE / "fingerprint.py"), *th], fingerprint_gate),
        ("layer_error", "cpu", "120m", [py, str(HERE / "layer_error.py"), *th], None),
        ("ppl", "cpu", "150m", [py, str(HERE / "ppl.py"), *th], None),
    ]
    names = [s[0] for s in stages]
    LOGS.mkdir(parents=True, exist_ok=True)
    for name, res, eta, cmd, gate in stages[names.index(args.start_at):]:
        t0 = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] {name} start", flush=True)
        # resumable stages get a few tries because an outside kill must not cost hours
        for attempt in range(3 if name in ("layer_error", "ppl") else 1):
            with open(LOGS / f"{name}.log", "a", encoding="utf-8") as log:
                rc = subprocess.run([shutil.which("dibs.cmd") or "dibs", "run", res, "--new", "--label", f"ternary fingerprint: {name}",
                                     "--eta", eta, "--", *cmd], stdout=log, stderr=subprocess.STDOUT).returncode
            if rc == 0:
                break
            print(f"[{time.strftime('%H:%M:%S')}] {name} attempt {attempt + 1} rc={rc}", flush=True)
        print(f"[{time.strftime('%H:%M:%S')}] {name} rc={rc} ({time.time() - t0:.0f}s)", flush=True)
        if rc != 0:
            sys.exit(f"stage {name} failed; see {LOGS / (name + '.log')}")
        problem = gate() if gate else None
        if problem:
            sys.exit(f"gate after {name}: {problem}")
    print("all stages done", flush=True)


if __name__ == "__main__":
    main()
