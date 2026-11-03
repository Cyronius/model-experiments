"""Fold unrolled loops in line-oriented text, and unfold them.

    repeat 8 i {
    @c{0+1i}r0
    ddr_patch DMA_BD0_1 arg=0 plus={0x0+0x140000i}
    }

A run of N >= 3 consecutive copies of a K-line block folds when every
number that varies between copies steps by a constant: `{start+step i}`
with `i` = 0..N-1. Nested loops use `i`, then `j`, then `k` (innermost
first, since folding runs innermost-out). Numbers are decimal or `0x` hex
as written; one with leading zeros is never varied, and an inner loop's
`{...}` term is never varied by an outer one. `fold` checks that `unfold`
returns exactly its input, so folding never changes meaning.
"""

from __future__ import annotations

import re

NUM = re.compile(r"\{[^{}]*\}|0x[0-9a-f]+|\d+")
VAR = re.compile(r"\{(0x[0-9a-f]+|\d+)([+-])(0x[0-9a-f]+|\d+)([ijk])\}")
HEAD = re.compile(r"^repeat (\d+) ([ijk]) \{$")
LETTERS = "ijk"
MAX_BODY = 64
MIN_COPIES = 3


def _parts(line: str):
    """(shape, [(text, value, style)]); numbers become \\0 in the shape.
    Style is None for numbers that must stay as written."""
    vals, shape, last = [], [], 0
    for m in NUM.finditer(line):
        t = m.group(0)
        if t.startswith("{"):  # an inner loop's term: part of the shape
            continue
        shape.append(line[last : m.start()])
        last = m.end()
        if t.startswith("0x"):
            vals.append((t, int(t, 16), "hex" if (t == "0x0" or t[2] != "0") else None))
        else:
            vals.append((t, int(t), "dec" if (t == "0" or t[0] != "0") else None))
    shape.append(line[last:])
    return "\0".join(shape), vals


def _fmt(v: int, style: str) -> str:
    return f"{v:#x}" if style == "hex" else str(v)


def _fold_once(lines: list[str], var: str) -> list[str]:
    parts = [_parts(l) for l in lines]
    shapes = [p[0] for p in parts]
    out: list[str] = []
    i, n = 0, len(lines)
    while i < n:
        best = None  # (saved, K, N, steps)
        for K in range(1, MAX_BODY + 1):
            if i + 2 * K > n:
                break
            if shapes[i + K] != shapes[i] or shapes[i : i + K] != shapes[i + K : i + 2 * K]:
                continue
            steps = [[y[1] - x[1] for x, y in zip(parts[i + k][1], parts[i + K + k][1])] for k in range(K)]
            if any(
                d and (parts[i + k][1][j][2] is None or parts[i + k][1][j][2] != parts[i + K + k][1][j][2])
                for k in range(K)
                for j, d in enumerate(steps[k])
            ):
                continue
            N = 2
            while i + (N + 1) * K <= n:
                base = i + N * K
                if shapes[base : base + K] != shapes[i : i + K]:
                    break
                good = all(
                    y[1] == x[1] + N * steps[k][j] and (y[2] == x[2] if steps[k][j] else y[0] == x[0])
                    for k in range(K)
                    for j, (x, y) in enumerate(zip(parts[i + k][1], parts[base + k][1]))
                )
                if not good:
                    break
                N += 1
            saved = K * (N - 1) - 2
            # Two copies always "step by a constant", so they say nothing
            # about a loop; fold only three or more.
            if N >= MIN_COPIES and saved > 0 and (best is None or saved > best[0]):
                best = (saved, K, N, steps)
        if best is None:
            out.append(lines[i])
            i += 1
            continue
        _, K, N, steps = best
        body = []
        for k in range(K):
            shape, vals = parts[i + k]
            pieces = shape.split("\0")
            text = [pieces[0]]
            for j, (t, v, style) in enumerate(vals):
                d = steps[k][j]
                text.append(t if d == 0 else "{" + _fmt(v, style) + ("+" if d > 0 else "-") + _fmt(abs(d), style) + var + "}")
                text.append(pieces[j + 1])
            body.append("".join(text))
        cand = [f"repeat {N} {var} {{", *body, "}"]
        if unfold(cand) == lines[i : i + N * K]:
            out.extend(cand)
            i += N * K
        else:
            out.append(lines[i])
            i += 1
    return out


def fold(lines: list[str]) -> list[str]:
    out = list(lines)
    for var in LETTERS:
        nxt = _fold_once(out, var)
        if len(nxt) >= len(out):
            break
        out = nxt
    if unfold(out) != list(lines):
        raise AssertionError("loop folding did not round-trip")
    return out


def _subst(line: str, var: str, i: int) -> str:
    def rep(m):
        if m.group(4) != var:
            return m.group(0)
        s, sign, d = m.group(1), m.group(2), m.group(3)
        v = int(s, 0) + (1 if sign == "+" else -1) * int(d, 0) * i
        if v < 0:
            raise ValueError(f"loop value went negative in {line!r}")
        return _fmt(v, "hex" if s.startswith("0x") else "dec")

    return VAR.sub(rep, line)


def unfold(lines: list[str]) -> list[str]:
    out: list[str] = []
    k = 0
    while k < len(lines):
        m = HEAD.match(lines[k])
        if not m:
            out.append(lines[k])
            k += 1
            continue
        n, var = int(m.group(1)), m.group(2)
        depth, j = 1, k + 1
        while depth:
            if j >= len(lines):
                raise ValueError("unterminated repeat")
            if HEAD.match(lines[j]):
                depth += 1
            elif lines[j] == "}":
                depth -= 1
            j += 1
        body = lines[k + 1 : j - 1]
        for i in range(n):
            out.extend(unfold([_subst(l, var, i) for l in body]))
        k = j
    return out
