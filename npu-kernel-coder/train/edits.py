"""Line-numbered edit lists: the port answer format (replaces unified diffs for the small model).

The prompt shows the source with each line numbered (`12| text`), so the model copies positions
instead of counting them. An answer is a list of edits in ascending order, with no context lines
and no hunk counts:

    @@ 12          edit at source line 12
    -old line      source lines removed, starting at line 12 (each must match the source)
    +new line      lines put in their place (none: pure deletion; no `-` lines: insertion before 12)

Every text line carries its prefix, so lines starting with `@`, `+`, `-` or `=` stay unambiguous.
"""

from __future__ import annotations

import difflib
import re

HEAD = re.compile(r"@@ (\d+)$")
NUM = re.compile(r"(\d+)\| ?(.*)$")


def number(text: str, start: int = 1) -> str:
    return "".join(f"{start + i}| {l}\n" for i, l in enumerate(text.splitlines()))


def unnumber(text: str) -> tuple[list[str], int]:
    """Numbered prompt text -> (lines, first line number)."""
    out, first = [], None
    for l in text.splitlines():
        m = NUM.match(l)
        if not m:
            raise ValueError(f"unnumbered line {l!r}")
        first = int(m.group(1)) if first is None else first
        out.append(m.group(2))
    return out, first or 1


def edits(a: str, b: str, start: int = 1) -> str:
    al, bl = a.splitlines(), b.splitlines()
    out = []
    for t, i1, i2, j1, j2 in difflib.SequenceMatcher(None, al, bl, autojunk=False).get_opcodes():
        if t == "equal":
            continue
        out.append(f"@@ {start + i1}")
        out += ["-" + l for l in al[i1:i2]] + ["+" + l for l in bl[j1:j2]]
    return "\n".join(out) + "\n" if out else ""


def apply(lines: list[str], answer: str, start: int = 1) -> list[str] | None:
    """Apply an edit list to source lines numbered from `start`; None if it is malformed or does not match."""
    ops = []
    for l in answer.splitlines():
        m = HEAD.match(l)
        if m:
            ops.append((int(m.group(1)) - start, [], []))
        elif l.startswith("-") and ops:
            ops[-1][1].append(l[1:])
        elif l.startswith("+") and ops:
            ops[-1][2].append(l[1:])
        elif l.strip():
            return None
    out, pos = [], 0
    for at, old, new in ops:
        if at < pos or at > len(lines) or lines[at:at + len(old)] != old:
            return None
        out += lines[pos:at] + new
        pos = at + len(old)
    return out + lines[pos:]
