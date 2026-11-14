"""Line patches: write a near-copy of another text as the edits from it.

    .base core_c0r2
    =120            copy 120 lines of the base
    -1              skip 1 base line
    +	movxm r0, #0x13
    =*              copy the rest

Used for core programs that differ from an earlier one in a few lines
(per-core constants, a different buffer address). `apply` rebuilds the full
text exactly; `make` returns None when a patch would not be much smaller.
"""

from __future__ import annotations

import difflib


def make(base: list[str], lines: list[str], base_name: str) -> list[str]:
    sm = difflib.SequenceMatcher(None, base, lines, autojunk=False)
    out = [f".base {base_name}"]
    ops = sm.get_opcodes()
    for n, (tag, i1, i2, j1, j2) in enumerate(ops):
        last = n == len(ops) - 1
        if tag == "equal":
            out.append("=*" if (last and i2 == len(base)) else f"={i2 - i1}")
            continue
        if i2 > i1:
            out.append(f"-{i2 - i1}")
        out.extend("+" + l for l in lines[j1:j2])
    return out


def size(script: list[str]) -> int:
    return sum(len(l) + 1 for l in script)


def apply(base: list[str], script: list[str]) -> list[str]:
    if not script or not script[0].startswith(".base "):
        raise ValueError("not a patch")
    out: list[str] = []
    pos = 0
    for l in script[1:]:
        if l == "=*":
            out.extend(base[pos:])
            pos = len(base)
        elif l.startswith("="):
            n = int(l[1:])
            out.extend(base[pos : pos + n])
            pos += n
        elif l.startswith("-"):
            pos += int(l[1:])
        elif l.startswith("+"):
            out.append(l[1:])
        else:
            raise ValueError(f"bad patch line {l!r}")
    return out


def base_of(text: str) -> str | None:
    first = text.split("\n", 1)[0]
    return first[6:].strip() if first.startswith(".base ") else None
