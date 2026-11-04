"""Differential NPU check of shape-port answers against the real build B.

    dibs run npu --label "npu-kernel-coder differential check" --eta 30m -- \
        python3 npu_check.py runs/<tag> [runs/<tag> ...] [--only-inexact]

For each answer that builds and passes the rule gate:
  1. rebuild it with npuasm and package an xclbin (B's xclbin as template,
     fresh PDI/xclbin UUIDs);
  2. run B's vendor build twice with the same seeded buffers (determinism);
  3. run the answer once with those buffers; compare every buffer.
Verdicts: `equal` (bitwise), `close` (float buffers within 1e-2 relative to
their scale, integer buffers equal), `differs`, `hang` (no completion),
`b_nondeterministic` (B disagrees with itself; not scored), `vacuous` (B
changed no buffer, so agreement would prove nothing; not scored).
Each run is its own process with a timeout. After any hang, B is rerun as a
health check and the session stops if that fails too.
Results: runs/<tag>/<task>/npu.json and runs/<tag>/npu_summary.json.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "npuasm"))

import score  # noqa: E402
from npuasm import build, xclbin  # noqa: E402

SEQ = re.compile(r"runtime_sequence\s*(?:@\w+)?\s*\(([^)]*)\)")
ARG = re.compile(r"memref<(\d+)x(\w+)>")
SEED = 20261104


def signature(prj: str) -> list[list]:
    t = (Path(prj) / "aie.mlir").read_text()
    m = SEQ.search(t)
    if not m:
        raise ValueError("no runtime_sequence in aie.mlir")
    return [[int(n), dt] for n, dt in ARG.findall(m.group(1))]


def exec_run(x: Path, insts: Path, sig: Path, out: Path, timeout_ms: int = 20000) -> dict:
    try:
        r = subprocess.run([sys.executable, str(HERE / "npu_exec.py"), str(x), str(insts), str(sig), str(SEED), str(out),
                            "--timeout-ms", str(timeout_ms)], capture_output=True, text=True, timeout=timeout_ms / 1000 + 60)
    except subprocess.TimeoutExpired:
        return {"ok": False, "state": "process timeout"}
    info = {"ok": r.returncode == 0}
    try:
        info.update(json.loads(r.stdout.strip().splitlines()[-1]))
    except Exception:
        info["stderr"] = r.stderr[-300:]
    return info


def compare(p: Path, q: Path, sig) -> tuple[str, list[dict]]:
    a, b = np.load(p), np.load(q)
    verdict, rows = "equal", []
    for i, (n, dt) in enumerate(sig):
        x, y = a[f"arr_{i}"], b[f"arr_{i}"]
        same = np.array_equal(x.view(np.uint8), y.view(np.uint8))
        row = {"arg": i, "dtype": dt, "elements": n, "bitwise_equal": bool(same)}
        if not same:
            if dt in ("bf16", "f32", "f16"):
                fx = (x.astype(np.uint32) << 16).view(np.float32) if dt == "bf16" else x.astype(np.float32)
                fy = (y.astype(np.uint32) << 16).view(np.float32) if dt == "bf16" else y.astype(np.float32)
                fin = np.isfinite(fx) & np.isfinite(fy)
                scale = float(np.max(np.abs(fx[fin]))) if fin.any() else 0.0
                err = float(np.max(np.abs(fx[fin] - fy[fin]))) if fin.any() else float("inf")
                row.update(max_abs_err=err, scale=scale, nonfinite_mismatch=int(np.sum(np.isfinite(fx) != np.isfinite(fy))),
                           words_differ=int(np.sum(x != y)))
                close = row["nonfinite_mismatch"] == 0 and err <= 1e-2 * max(scale, 1e-30)
                verdict = "close" if (close and verdict in ("equal", "close")) else "differs"
            else:
                row["words_differ"] = int(np.sum(x != y))
                verdict = "differs"
        rows.append(row)
    return verdict, rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--only-inexact", action="store_true", help="skip exact answers (they are byte-identical to B)")
    a = ap.parse_args()
    for run in a.runs:
        summary = {}
        for rp in sorted(run.glob("*/result.json")):
            res = json.loads(rp.read_text())
            s = res["score"]
            td = HERE / "tasks" / res["id"]
            out_json = rp.parent / "npu.json"
            if out_json.exists():
                summary[res["id"]] = json.loads(out_json.read_text())["verdict"]
                continue
            if not (s["builds"] and s["rules_pass"]):
                continue
            if a.only_inexact and s["exact"]:
                continue
            a_files = {p.name: p.read_text() for p in (td / "a").iterdir()}
            pred = dict(a_files)
            for f, h in score.parse_diff((rp.parent / "answer.txt").read_text()).items():
                if f in a_files:
                    pred[f] = "\n".join(score.apply(a_files[f].splitlines(), h)[0]) + "\n"
            manifest = json.loads((td / "manifest_b.json").read_text())
            sig = signature(res["b_prj"])
            if len(sig) > 5:
                result = {"verdict": "skipped", "why": f"{len(sig)} buffer args (classic path takes 5)"}
            else:
                with tempfile.TemporaryDirectory(prefix="nkc-npu-") as d:
                    d = Path(d)
                    built = build.rebuild(manifest, pred)
                    bdir = Path(res["b_prj"]).parent
                    (d / "sig.json").write_text(json.dumps(sig))
                    (d / "ans.xclbin").write_bytes(xclbin.build((bdir / "final.xclbin").read_bytes(), built["pdi"]))
                    (d / "ans.insts").write_bytes(built["insts"])
                    r1 = exec_run(bdir / "final.xclbin", bdir / "insts.bin", d / "sig.json", d / "b1.npz")
                    r2 = exec_run(bdir / "final.xclbin", bdir / "insts.bin", d / "sig.json", d / "b2.npz")
                    if not (r1["ok"] and r2["ok"]):
                        result = {"verdict": "b_failed", "b_runs": [r1, r2]}
                    else:
                        v_bb, _ = compare(d / "b1.npz", d / "b2.npz", sig)
                        if not any(r1.get("changed_bytes", [1])):
                            result = {"verdict": "vacuous", "why": "B wrote no buffer; the comparison would prove nothing"}
                        elif v_bb != "equal":
                            result = {"verdict": "b_nondeterministic", "b_vs_b": v_bb}
                        else:
                            ra = exec_run(d / "ans.xclbin", d / "ans.insts", d / "sig.json", d / "ans.npz")
                            if not ra["ok"]:
                                result = {"verdict": "hang", "answer_run": ra}
                                health = exec_run(bdir / "final.xclbin", bdir / "insts.bin", d / "sig.json", d / "h.npz")
                                result["health_after"] = health
                            else:
                                v, rows = compare(d / "b1.npz", d / "ans.npz", sig)
                                result = {"verdict": v, "buffers": rows, "b_ms": r1.get("ms"), "answer_ms": ra.get("ms"),
                                          "b_changed_bytes": r1.get("changed_bytes"), "answer_changed_bytes": ra.get("changed_bytes")}
            result.update(exact=s["exact"], gap_closed=s["gap_closed"], difficulty=res["difficulty"])
            out_json.write_text(json.dumps(result, indent=1) + "\n")
            summary[res["id"]] = result["verdict"]
            print(f"{run.name} {res['id']} {res['difficulty']:6s} exact={s['exact']} gap={s['gap_closed']} -> {result['verdict']}", flush=True)
            if result["verdict"] == "hang" and not result.get("health_after", {}).get("ok"):
                print("NPU health check failed after a hang; stopping", flush=True)
                (run / "npu_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
                return
        (run / "npu_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
        counts = {}
        for v in summary.values():
            counts[v] = counts.get(v, 0) + 1
        print(run.name, counts, flush=True)


if __name__ == "__main__":
    main()
