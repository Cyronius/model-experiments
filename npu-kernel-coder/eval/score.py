"""Score a shape-port answer: apply its unified diff to build A, compare
with build B, and try to rebuild it.

Metrics per task:
  parsed        a diff was found
  hunks         applied / total (a hunk applies where its context and
                removed lines occur; nearest to the stated line wins)
  exact         every file equals B's reference text
  gap_closed    1 - d(answer, B) / d(A, B); d counts changed lines. 1 is
                exact, 0 is no progress, below 0 is worse than A
  builds        npuasm rebuilds the answer (assembles every core)
  rules_pass    and the rule gate finds no violation
  rules_pass_aligned  passes once branch targets are padded to 16 bytes
                (the repair applies only when alignment was the sole problem)
  bytes_match   the rebuilt artifacts equal B's vendor binaries
"""

from __future__ import annotations

import difflib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "npuasm"))

from npuasm import build  # noqa: E402

HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def parse_diff(text: str) -> dict[str, list[tuple[int, list[str]]]]:
    """file -> [(stated old line, hunk lines)]. Tolerates code fences."""
    files: dict[str, list[tuple[int, list[str]]]] = {}
    cur = None
    hunk = None
    for line in text.splitlines():
        if line.startswith("```"):
            continue
        if line.startswith("--- "):
            continue
        if line.startswith("+++ "):
            name = line[4:].strip().split("\t")[0]
            name = name[2:] if name.startswith(("a/", "b/")) else name
            cur = files.setdefault(name, [])
            hunk = None
            continue
        m = HUNK.match(line)
        if m and cur is not None:
            hunk = []
            cur.append((int(m.group(1)), hunk))
            continue
        if hunk is not None and line[:1] in (" ", "-", "+"):
            hunk.append(line)
        elif hunk is not None and line == "":
            hunk.append(" ")  # a blank context line whose space was dropped
    return files


def apply(lines: list[str], hunks, failed: list | None = None) -> tuple[list[str], int]:
    """Apply hunks in order; hunks that can't be placed are skipped (and
    appended to `failed` as (stated line, hunk lines) when a list is given)."""
    out = list(lines)
    ok = 0
    shift = 0
    for start, h in hunks:
        old = [l[1:] for l in h if l[0] in " -"]
        new = [l[1:] for l in h if l[0] in " +"]
        if not old:  # pure insertion at the stated line
            at = max(0, min(len(out), start + shift))
            out[at:at] = new
            shift += len(new)
            ok += 1
            continue
        hint = start - 1 + shift
        found = None
        for key in (lambda s: s, lambda s: s.rstrip(), lambda s: " ".join(s.split())):
            want = [key(x) for x in old]
            have = [key(x) for x in out]
            cands = [i for i in range(len(have) - len(want) + 1) if have[i : i + len(want)] == want]
            if cands:
                found = min(cands, key=lambda i: abs(i - hint))
                break
        if found is None:
            if failed is not None:
                failed.append((start, h))
            continue
        out[found : found + len(old)] = new
        shift += len(new) - len(old)
        ok += 1
    return out, ok


def dist(x: dict[str, str], y: dict[str, str]) -> int:
    n = 0
    for f in set(x) | set(y):
        a, b = x.get(f, "").splitlines(), y.get(f, "").splitlines()
        sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
        n += sum(max(i2 - i1, j2 - j1) for t, i1, i2, j1, j2 in sm.get_opcodes() if t != "equal")
    return n


def score(task_dir: Path, answer: str) -> dict:
    a = {p.name: p.read_text() for p in (task_dir / "a").iterdir()}
    b = {p.name: p.read_text() for p in (task_dir / "b").iterdir()}
    files = parse_diff(answer)
    res = {"parsed": bool(files), "hunks_total": sum(len(h) for h in files.values()), "hunks_ok": 0,
           "unknown_files": sorted(set(files) - set(a))}
    pred = dict(a)
    failed: list[str] = []
    for f, hunks in files.items():
        if f not in a:
            continue
        miss: list = []
        new, ok = apply(a[f].splitlines(), hunks, miss)
        failed += [f"{f} @@ -{start}: " + next((l[1:] for l in h if l[0] in " -"), "").strip()[:100] for start, h in miss]
        res["hunks_ok"] += ok
        pred[f] = "\n".join(new) + "\n"
    res["failed_hunks"] = failed[:8]
    d0, d1 = dist(a, b), dist(pred, b)
    res.update(exact=pred == b, d_a_b=d0, d_answer_b=d1, gap_closed=round(1 - d1 / d0, 4) if d0 else None)
    manifest = json.loads((task_dir / "manifest_b.json").read_text())
    try:
        out = build.rebuild(manifest, pred)
        res["builds"] = True
        vs = build.check_rules(manifest, pred)
        res["rules_pass"] = not vs
        res["rule_violations"] = [f"{v.rule}: {v.detail}" for v in vs[:5]]
        # Repair: when the only problem is branch-target alignment, pad the
        # targets to 16 bytes as Peano does (core.align_targets) and recheck.
        res["rules_pass_aligned"] = res["rules_pass"]
        if vs and all(v.rule == "branch-target" for v in vs):
            try:
                res["rules_pass_aligned"] = not build.check_rules(manifest, pred, align_targets=True)
            except Exception as e:
                res["align_error"] = f"{type(e).__name__}: {str(e)[:200]}"
        h = manifest["hashes"]
        res["bytes_match"] = all(build.sha(out[k]) == h[k] for k in ("cdo_elfs", "cdo_init", "cdo_enable", "insts") if k in h and k in out)
    except Exception as e:  # unparseable or unassemblable answer
        res.update(builds=False, rules_pass=False, rules_pass_aligned=False, bytes_match=False,
                   build_error=f"{type(e).__name__}: {str(e)[:200]}")
    return res
