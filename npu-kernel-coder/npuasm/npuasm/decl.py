"""Declarative CDO text: register writes as aie-rt register and field names.

Same op list as `cdo.py`, so it is lossless; a smaller, readable spelling:

    @c2r3                                   tile for the lines that follow
    write CORE_CONTROL ENABLE=1             named register, non-zero fields
    write LOCK3_VALUE 0                     all fields zero
    maskwrite DMA_S2MM_0_CTRL RESET=1       mask = the listed fields' bits
    maskwrite X F=1 mask=0x3                mask given when it isn't that
    block DMA_BD3_0 [BUFFER_LENGTH=256] [] [D0_STEPSIZE=1] ...
                                            one DMA write over consecutive
                                            named registers, a [...] per word
    locks 0: 0*14,2,1                       LOCK0_VALUE.. written in order
    @c1r2 =c0r2                             same lines as the last segment
                                            written for c0r2, on this tile
    +0x1f                                   bits outside every known field

Field values print in decimal below 1000 and in hex above; either parses.
Writes the register table can't name, data-memory payloads, program-memory
references (`@core_cCrR`) and NOPs keep the `cdo.py` line form, which this
parser also reads. The register and field names come from aie-rt's
`xaie2pgbl_params.h` via `regs.py`.
"""

from __future__ import annotations

import re

from . import cdo, regs
from .cdo import DMA_WRITE, MASK_POLL64, MASK_WRITE64, NOP, WRITE64, CdoError, Op

LOCK = re.compile(r"^LOCK(\d+)_VALUE$")


def _num(v: int) -> str:
    return str(v) if v < 1000 else f"{v:#x}"


def _tile_of(addr: int) -> tuple[int, int]:
    return addr >> 25, (addr >> 20) & 0x1F


def _fields_text(addr: int, value: int, *, keep: int = 0) -> tuple[str, int, int]:
    """Field tokens for `value` at register `addr`. Fields overlapping `keep`
    are printed even when zero. Returns (text, implied_mask, remainder)."""
    toks, covered, implied = [], 0, 0
    for name, lsb, width in regs.fields(addr):
        fm = ((1 << width) - 1) << lsb
        covered |= fm
        v = (value & fm) >> lsb
        if v or (keep & fm):
            toks.append(f"{name}={_num(v)}")
            implied |= fm
    rest = value & ~covered & 0xFFFFFFFF
    if rest:
        toks.append(f"+{rest:#x}")
    return " ".join(toks), implied, rest


def _word_text(addr: int, value: int) -> str:
    t, _, _ = _fields_text(addr, value)
    return t


def _named(addr: int) -> str | None:
    n = regs.name(addr)
    if n is None or not regs.fields(addr):
        return None
    # Ranges the block() classifier treats as memory are never registers.
    if regs.block(addr).endswith(("PROGRAM_MEMORY", "DATA_MEMORY")):
        return None
    return n


def _rle(vals: list[int]) -> str:
    out, i = [], 0
    while i < len(vals):
        j = i
        while j < len(vals) and vals[j] == vals[i]:
            j += 1
        n = j - i
        out.append(f"{_num(vals[i])}*{n}" if n > 1 else _num(vals[i]))
        i = j
    return ",".join(out)


def to_text(ops: list[Op]) -> str:
    lines: list[str] = []
    tile: tuple[int, int] | None = None
    i = 0

    def at(addr: int) -> None:
        nonlocal tile
        t = _tile_of(addr)
        if t != tile:
            lines.append(f"@c{t[0]}r{t[1]}")
            tile = t

    while i < len(ops):
        op = ops[i]
        a = op.args
        has_addr = op.cmd in (WRITE64, MASK_WRITE64, MASK_POLL64, DMA_WRITE) and a and a[0] == 0
        if not has_addr:
            lines.append(cdo.to_text([op]).rstrip("\n"))
            i += 1
            continue
        addr = a[1]
        name = _named(addr) if not op.ref else None
        # A run of lock-value writes on one tile, lock ids ascending by one.
        m = LOCK.match(name or "") if op.cmd == WRITE64 else None
        if m and (a[2] & ~0x3F) == 0:
            first = int(m.group(1))
            vals = [a[2]]
            j = i + 1
            while j < len(ops):
                o = ops[j]
                if o.cmd != WRITE64 or o.args[0] != 0 or _tile_of(o.args[1]) != _tile_of(addr):
                    break
                mm = LOCK.match(regs.name(o.args[1]) or "")
                if not mm or int(mm.group(1)) != first + len(vals) or (o.args[2] & ~0x3F):
                    break
                vals.append(o.args[2])
                j += 1
            at(addr)
            lines.append(f"locks {first}: {_rle(vals)}")
            i = j
            continue
        if name is None:  # raw form carries its own full address
            lines.append(cdo.to_text([op]).rstrip("\n"))
            i += 1
            continue
        at(addr)
        if op.cmd == WRITE64:
            t, _, _ = _fields_text(addr, a[2])
            lines.append(f"write {name} {t or '0'}")
        elif op.cmd in (MASK_WRITE64, MASK_POLL64):
            mask, value = a[2], a[3]
            t, implied, rest = _fields_text(addr, value, keep=mask)
            extra = "" if (mask == implied and not rest) else f" mask={mask:#x}"
            if op.cmd == MASK_POLL64:
                extra += f" timeout={a[4]}"
            lines.append(f"{'maskwrite' if op.cmd == MASK_WRITE64 else 'maskpoll'} {name} {t or '0'}{extra}")
        else:  # DMA_WRITE over named registers?
            data = a[2:]
            if all(_named(addr + 4 * k) for k in range(len(data))):
                words = " ".join(f"[{_word_text(addr + 4 * k, w)}]" for k, w in enumerate(data))
                lines.append(f"block {name} {words}")
            else:
                lines.append(cdo.to_text([op]).rstrip("\n"))
        i += 1
    return "\n".join(_reuse_segments(lines)) + "\n"


def _reuse_segments(lines: list[str]) -> list[str]:
    """Replace a tile segment (the lines after an `@cCrR` line) that equals
    the last segment written for another tile by `@cCrR =cXrY`."""
    out: list[str] = []
    last: dict[str, tuple[str, ...]] = {}
    i = 0
    while i < len(lines):
        if not lines[i].startswith("@"):
            out.append(lines[i])
            i += 1
            continue
        j = i + 1
        while j < len(lines) and not lines[j].startswith("@"):
            j += 1
        tile, body = lines[i], tuple(lines[i + 1 : j])
        src = next((t for t, b in last.items() if b == body and t != tile), None) if body else None
        if src:
            out.append(f"{tile} ={src[1:]}")
        else:
            out.append(tile)
            out.extend(body)
        last[tile] = body
        i = j
    return out


def _expand_segments(text: str) -> list[str]:
    out: list[str] = []
    last: dict[str, list[str]] = {}
    cur: str | None = None
    for line in text.splitlines():
        if line.startswith("@"):
            tile, _, src = line.partition(" =")
            cur = tile
            last[cur] = []
            out.append(tile)
            if src:
                body = last.get("@" + src.strip())
                if body is None:
                    raise CdoError(f"{line}: no earlier segment for @{src.strip()}")
                last[cur] = list(body)
                out.extend(body)
            continue
        if cur is not None:
            last[cur].append(line)
        out.append(line)
    return out


# ------------------------------------------------------------------ parse


def _int(t: str) -> int:
    return int(t, 0)


def _encode(addr: int, toks: list[str]) -> tuple[int, int, int | None]:
    """Field tokens -> (value, implied_mask, explicit_mask)."""
    fmap = {n: (lsb, w) for n, lsb, w in regs.fields(addr)}
    value = implied = 0
    explicit = None
    for t in toks:
        if t == "0":
            continue
        if t.startswith("+"):
            value |= _int(t[1:])
            continue
        k, v = t.split("=", 1)
        if k == "mask":
            explicit = _int(v)
            continue
        if k not in fmap:
            raise CdoError(f"{regs.name(addr)} has no field {k}")
        lsb, w = fmap[k]
        x = _int(v)
        if x >> w:
            raise CdoError(f"{k}={x} does not fit {w} bits")
        value |= x << lsb
        implied |= ((1 << w) - 1) << lsb
    return value, implied, explicit


BRACKET = re.compile(r"\[([^\]]*)\]")


def from_text(text: str, *, resolve=None) -> list[Op]:
    ops: list[Op] = []
    tile: tuple[int, int] | None = None
    raw: list[str] = []  # pending cdo.py-form lines (dmawrite data follows its header)

    def flush() -> None:
        if raw:
            ops.extend(cdo.from_text("\n".join(raw) + "\n", resolve=resolve))
            raw.clear()

    def reg(name: str) -> int:
        if tile is None:
            raise CdoError(f"register {name} before any @tile line")
        base = (tile[0] << 25) | (tile[1] << 20)
        kind = regs.kind(base)
        off = regs.offsets()[kind].get(name)
        if off is None:
            raise CdoError(f"no register {name} in a {kind} tile")
        return base | off

    for line in _expand_segments(text):
        s = line.split(";", 1)[0].rstrip()
        if not s.strip():
            continue
        if line.startswith("  "):  # data line of a cdo.py dmawrite
            raw.append(line)
            continue
        t = s.split()
        k = t[0]
        if k.startswith("@"):
            flush()
            col, row = t[0][2:].split("r")
            tile = (int(col), int(row))
            continue
        named = len(t) > 1 and k in ("write", "maskwrite", "maskpoll", "block") and "+" not in t[1] and not re.match(r"^c\d+r\d+", t[1])
        if k == "locks":
            flush()
            first = int(t[1].rstrip(":"))
            vals: list[int] = []
            for item in t[2].split(","):
                v, _, n = item.partition("*")
                vals += [_int(v)] * (int(n) if n else 1)
            for idx, v in enumerate(vals):
                ops.append(Op(WRITE64, [0, reg(f"LOCK{first + idx}_VALUE"), v]))
        elif named and k == "write":
            flush()
            addr = reg(t[1])
            value, _, _ = _encode(addr, t[2:])
            ops.append(Op(WRITE64, [0, addr, value]))
        elif named and k in ("maskwrite", "maskpoll"):
            flush()
            addr = reg(t[1])
            toks = [x for x in t[2:] if not x.startswith("timeout=")]
            value, implied, explicit = _encode(addr, toks)
            mask = implied if explicit is None else explicit
            if k == "maskwrite":
                ops.append(Op(MASK_WRITE64, [0, addr, mask, value]))
            else:
                timeout = next(int(x.split("=")[1]) for x in t[2:] if x.startswith("timeout="))
                ops.append(Op(MASK_POLL64, [0, addr, mask, value, timeout]))
        elif named and k == "block":
            flush()
            addr = reg(t[1])
            words = []
            for n, grp in enumerate(BRACKET.findall(s)):
                v, _, _ = _encode(addr + 4 * n, grp.split())
                words.append(v)
            ops.append(Op(DMA_WRITE, [0, addr, *words]))
        else:
            if k in ("write", "maskwrite", "maskpoll", "dmawrite") or k in cdo.IDS or k.startswith("cmd_"):
                flush()
            raw.append(line)
    flush()
    return ops
