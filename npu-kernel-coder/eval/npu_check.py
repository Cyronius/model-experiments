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
--guard N (bytes after every buffer, see npu_exec.py) also catches answers
that write past the end of B's buffers: any guard byte that B leaves alone
but the answer changes makes the verdict `differs`.
Each run is its own process with a timeout. A hang stops the whole session
(no health-check run, no next answer) and writes runs/.npu_hang_lock; later
sessions refuse to start while a lock younger than 30 minutes exists. On
2026-11-05, 6 hangs within 23 s froze this machine and needed a hard reset,
so hangs are not treated as cheap.
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
SEED = 20261104  # default; --seed overrides


def signature(prj: str) -> list[list]:
    t = (Path(prj) / "aie.mlir").read_text()
    m = SEQ.search(t)
    if not m:
        raise ValueError("no runtime_sequence in aie.mlir")
    return [[int(n), dt] for n, dt in ARG.findall(m.group(1))]


GUARD = 0  # --guard overrides


def exec_run(x: Path, insts: Path, sig: Path, out: Path, timeout_ms: int = 20000) -> dict:
    try:
        r = subprocess.run([sys.executable, str(HERE / "npu_exec.py"), str(x), str(insts), str(sig), str(SEED), str(out),
                            "--timeout-ms", str(timeout_ms), "--guard", str(GUARD)],
                           capture_output=True, text=True, timeout=timeout_ms / 1000 + 60)
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
    for i in range(len(sig)):
        if f"g_{i}" in a.files and f"g_{i}" in b.files:
            n_diff = int(np.count_nonzero(a[f"g_{i}"] != b[f"g_{i}"]))
            if n_diff:
                rows.append({"arg": i, "guard_bytes_differ": n_diff})
                verdict = "differs"
    return verdict, rows


LOCK = HERE / "runs" / ".npu_hang_lock"
COOLDOWN_S = 1800


def main() -> None:
    import time
    stop = HERE / "runs" / ".npu_disabled"
    if stop.exists():
        sys.exit(f"refusing: NPU checks disabled by {stop}: {stop.read_text().strip()}")
    if LOCK.exists() and time.time() - LOCK.stat().st_mtime < COOLDOWN_S:
        sys.exit(f"refusing: an answer hung the NPU at {time.ctime(LOCK.stat().st_mtime)}; wait 30 min or inspect first")
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--only-inexact", action="store_true", help="skip exact answers (they are byte-identical to B)")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--guard", type=int, default=0, help="guard bytes after every buffer (catches out-of-bounds writes)")
    ap.add_argument("--out-name", default="npu.json", help="per-task result file (e.g. npu_seed2.json)")
    ap.add_argument("--recheck-equal", action="store_true",
                    help="only answers whose npu.json verdict is equal/close (second-seed recheck)")
    a = ap.parse_args()
    globals()["SEED"] = a.seed
    globals()["GUARD"] = a.guard
    for run in a.runs:
        summary = {}
        for rp in sorted(run.glob("*/result.json")):
            res = json.loads(rp.read_text())
            s = res["score"]
            td = HERE / "tasks" / res["id"]
            out_json = rp.parent / a.out_name
            if a.recheck_equal:
                first = rp.parent / "npu.json"
                if not first.exists() or json.loads(first.read_text())["verdict"] not in ("equal", "close"):
                    continue
            if out_json.exists():
                summary[res["id"]] = json.loads(out_json.read_text())["verdict"]
                continue
            aligned = not s["rules_pass"]
            if not (s["builds"] and (s["rules_pass"] or s.get("rules_pass_aligned"))):
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
                    built = build.rebuild(manifest, pred, align_targets=aligned)
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
                            else:
                                v, rows = compare(d / "b1.npz", d / "ans.npz", sig)
                                result = {"verdict": v, "buffers": rows, "b_ms": r1.get("ms"), "answer_ms": ra.get("ms"),
                                          "b_changed_bytes": r1.get("changed_bytes"), "answer_changed_bytes": ra.get("changed_bytes"),
                                          "b_guard_changed": r1.get("guard_changed_bytes"),
                                          "answer_guard_changed": ra.get("guard_changed_bytes")}
            result.update(exact=s["exact"], gap_closed=s["gap_closed"], difficulty=res["difficulty"],
                          aligned=aligned, seed=a.seed, guard=a.guard)
            out_json.write_text(json.dumps(result, indent=1) + "\n")
            summary[res["id"]] = result["verdict"]
            print(f"{run.name} {res['id']} {res['difficulty']:6s} exact={s['exact']} gap={s['gap_closed']} -> {result['verdict']}", flush=True)
            if result["verdict"] == "hang":
                LOCK.write_text(f"{run.name} {res['id']}\n")
                print("answer hung the NPU; stopping the session (see runs/.npu_hang_lock)", flush=True)
                (run / a.out_name.replace(".json", "_summary.json")).write_text(json.dumps(summary, indent=1) + "\n")
                return
        (run / a.out_name.replace(".json", "_summary.json")).write_text(json.dumps(summary, indent=1) + "\n")
        counts = {}
        for v in summary.values():
            counts[v] = counts.get(v, 0) + 1
        print(run.name, counts, flush=True)


if __name__ == "__main__":
    main()
