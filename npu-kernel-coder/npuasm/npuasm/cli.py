"""npuasm command line.

  python -m npuasm check  <final.prj>             round-trip one build
  python -m npuasm lift   <final.prj> <out_dir>   write the text form
  python -m npuasm build  <lifted_dir> <out_dir>  rebuild artifacts from text
  python -m npuasm corpus <root>... --out results.jsonl [--jobs N]
  python -m npuasm rules  <lifted_dir>            the rule gate; exit 1 on violations
  python -m npuasm package <pdi> <template.xclbin> <out.xclbin>
                                                  new xclbin around a PDI
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from . import build, xclbin


def _find_prjs(roots: list[Path]) -> list[Path]:
    out = set()
    for root in roots:
        for p in root.rglob("final.prj"):
            if p.is_dir() and any(p.glob("*.pdi")):
                out.add(p.resolve())
    return sorted(out)


def _check(p: str) -> dict:
    t = time.time()
    r = build.check(Path(p))
    d = r.__dict__.copy()
    d["seconds"] = round(time.time() - t, 3)
    return d


def summarize(rows: list[dict]) -> str:
    n = len(rows)
    errs = [r for r in rows if r["error"]]
    done = [r for r in rows if not r["error"]]
    kinds = Counter()
    passed = Counter()
    for r in done:
        for k, v in r["ok"].items():
            k = "core" if k.endswith(".s") else k
            kinds[k] += 1
            passed[k] += bool(v)
    all_ok = sum(1 for r in done if all(r["ok"].values()))
    bundles = sum(r["bundles"] for r in done)
    raw = sum(r["raw_bundles"] for r in done)
    asm = Counter()
    for r in done:
        asm.update(r["asm_errors"])
    lines = [
        f"builds: {n}   lifted and rebuilt: {len(done)}   every artifact identical: {all_ok}",
        "",
        "| artifact | identical | checked |",
        "|---|---:|---:|",
    ]
    for k in sorted(kinds):
        lines.append(f"| {k} | {passed[k]} | {kinds[k]} |")
    lines += [
        "",
        f"core bundles lifted: {bundles}; written as `.byte` because the assembler rejects them: {raw} ({100 * raw / max(bundles, 1):.2f}%)",
        "",
        "assembler rejections by slot (first word):",
    ]
    lines += [f"  {c:7d}  {k}" for k, c in asm.most_common(15)]
    if errs:
        lines += ["", f"builds that failed to lift ({len(errs)}):"]
        for k, c in Counter(r["error"].split(":")[0] + ": " + r["error"].split(":", 1)[1][:90] for r in errs).most_common(10):
            lines.append(f"  {c:5d}  {k}")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="npuasm")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("prj", type=Path)
    l = sub.add_parser("lift")
    l.add_argument("prj", type=Path)
    l.add_argument("out", type=Path)
    b = sub.add_parser("build")
    b.add_argument("lifted", type=Path)
    b.add_argument("out", type=Path)
    ru = sub.add_parser("rules")
    ru.add_argument("lifted", type=Path)
    pk = sub.add_parser("package")
    pk.add_argument("pdi", type=Path)
    pk.add_argument("template", type=Path)
    pk.add_argument("out", type=Path)
    pk.add_argument("--keep-mirror", action="store_true")
    k = sub.add_parser("corpus")
    k.add_argument("roots", type=Path, nargs="+")
    k.add_argument("--out", type=Path, required=True)
    k.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    k.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)

    if a.cmd == "check":
        print(json.dumps(_check(str(a.prj)), indent=1))
        return 0
    if a.cmd == "lift":
        _, texts, rep = build.lift(a.prj, a.out)
        print(f"wrote {len(texts)} files to {a.out}")
        return 0
    if a.cmd == "build":
        manifest = json.loads((a.lifted / "manifest.json").read_text())
        texts = {p.name: p.read_text() for p in a.lifted.iterdir() if p.suffix in (".s", ".cdo", ".txn")}
        out = build.rebuild(manifest, texts)
        a.out.mkdir(parents=True, exist_ok=True)
        names = {"cdo_elfs": "aie_cdo_elfs.bin", "cdo_init": "aie_cdo_init.bin", "cdo_enable": "aie_cdo_enable.bin",
                 "pdi": f"{manifest['device']}.pdi", "insts": "insts.bin"}
        for key, data in out.items():
            if key in names:
                (a.out / names[key]).write_bytes(data)
        print(f"wrote {len(out)} artifacts to {a.out}")
        return 0
    if a.cmd == "rules":
        manifest = json.loads((a.lifted / "manifest.json").read_text())
        texts = {p.name: p.read_text() for p in a.lifted.iterdir() if p.suffix in (".s", ".cdo", ".txn")}
        vs = build.check_rules(manifest, texts)
        for v in vs:
            print(f"{v.rule}: {v.where}: {v.detail}")
        print(f"{len(vs)} violations" if vs else "rule gate: pass")
        return 1 if vs else 0
    if a.cmd == "package":
        data = xclbin.build(a.template.read_bytes(), a.pdi.read_bytes(), keep_mirror=a.keep_mirror)
        a.out.write_bytes(data)
        print(f"wrote {a.out} ({len(data)} bytes)")
        return 0
    if a.cmd == "corpus":
        prjs = _find_prjs(a.roots)
        if a.limit:
            prjs = prjs[: a.limit]
        print(f"{len(prjs)} builds, {a.jobs} jobs", file=sys.stderr)
        rows = []
        with ProcessPoolExecutor(a.jobs) as ex, a.out.open("w") as f:
            for i, row in enumerate(ex.map(_check, map(str, prjs), chunksize=4)):
                rows.append(row)
                f.write(json.dumps(row) + "\n")
                if (i + 1) % 200 == 0:
                    print(f"  {i + 1}/{len(prjs)}", file=sys.stderr)
        s = summarize(rows)
        a.out.with_suffix(".md").write_text(s + "\n")
        print(s)
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
