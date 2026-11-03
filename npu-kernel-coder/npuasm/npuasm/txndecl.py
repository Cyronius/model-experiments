"""Declarative transaction-stream text: `txn.py` ops with aie-rt names.

    txn 0.1 gen=4 rows=6 cols=8 memtile_rows=1
    @c1r0
    block DMA_BD0_0 [BUFFER_LENGTH=0x2800] [BASE_ADDRESS_LOW=...] ... tile=c0r0
    ddr_patch DMA_BD0_1 arg=0 plus=0x140000
    maskwrite DMA_MM2S_0_CTRL CONTROLLER_ID=15
    write DMA_MM2S_0_TASK_QUEUE START_BD_ID=0 ...
    sync c0r0 dir=0 chan=0 ncol=1 nrow=1

`tile=` keeps the blockwrite's own tile field when it differs from the
address (mlir-aie writes c0r0 there for every column). Unnameable ops keep
the `txn.py` line form, which this parser also reads. Identical tile
segments reuse earlier ones as in `decl.py`.
"""

from __future__ import annotations

import re

from . import decl, loops, regs, txn
from .cdo import CdoError

TILE = re.compile(r"^c\d+r\d+")


def _tile(addr: int) -> tuple[int, int]:
    return addr >> 25, (addr >> 20) & 0x1F


def to_text(h: txn.Header, ops: list[txn.Op]) -> str:
    head, *_ = txn.to_text(h, []).splitlines()
    lines: list[str] = []
    tile = None

    def at(addr: int) -> None:
        nonlocal tile
        t = _tile(addr)
        if t != tile:
            lines.append(f"@c{t[0]}r{t[1]}")
            tile = t

    for op in ops:
        f = op.f
        addr = f.get("addr") if op.kind != "loadpdi" else None
        name = decl._named(addr) if addr is not None else None
        if name is None:
            lines.extend(txn.to_text(h, [op]).splitlines()[1:])
            continue
        at(addr)
        if op.kind == "write":
            t, _, _ = decl._fields_text(addr, f["value"])
            lines.append(f"write {name} {t or '0'}")
        elif op.kind in ("maskwrite", "maskpoll"):
            t, implied, rest = decl._fields_text(addr, f["value"], keep=f["mask"])
            extra = "" if (f["mask"] == implied and not rest) else f" mask={f['mask']:#x}"
            lines.append(f"{op.kind} {name} {t or '0'}{extra}")
        elif op.kind == "blockwrite":
            if not all(decl._named(addr + 4 * k) for k in range(len(op.data))):
                lines.extend(txn.to_text(h, [op]).splitlines()[1:])
                continue
            words = " ".join(f"[{decl._word_text(addr + 4 * k, w)}]" for k, w in enumerate(op.data))
            extra = "" if (f["col"], f["row"]) == _tile(addr) else f" tile=c{f['col']}r{f['row']}"
            lines.append(f"block {name} {words}{extra}")
        elif op.kind == "ddr_patch":
            lines.append(f"ddr_patch {name} arg={f['arg']} plus={f['plus']:#x}")
        else:
            lines.extend(txn.to_text(h, [op]).splitlines()[1:])
    return "\n".join([head, *loops.fold(decl._reuse_segments(lines))]) + "\n"


def from_text(text: str) -> tuple[txn.Header, list[txn.Op]]:
    lines = decl._expand_segments("\n".join(loops.unfold(text.splitlines())))
    head = lines[0]
    h, _ = txn.from_text(head + "\n")
    ops: list[txn.Op] = []
    raw: list[str] = []
    tile = None

    def flush() -> None:
        if raw:
            ops.extend(txn.from_text("\n".join([head, *raw]) + "\n")[1])
            raw.clear()

    def reg(name: str) -> int:
        if tile is None:
            raise CdoError(f"register {name} before any @tile line")
        base = (tile[0] << 25) | (tile[1] << 20)
        off = regs.offsets()[regs.kind(base)].get(name)
        if off is None:
            raise CdoError(f"no register {name} in a {regs.kind(base)} tile")
        return base | off

    for line in lines[1:]:
        s = line.split(";", 1)[0].rstrip()
        if not s.strip():
            continue
        if line.startswith("  "):
            raw.append(line)
            continue
        t = s.split()
        k = t[0]
        if k.startswith("@"):
            flush()
            col, row = k[2:].split("r")
            tile = (int(col), int(row))
            continue
        named = len(t) > 1 and "+" not in t[1] and not TILE.match(t[1])
        if k in ("write", "maskwrite", "maskpoll", "block", "ddr_patch") and named:
            flush()
            addr = reg(t[1])
            if k == "write":
                v, _, _ = decl._encode(addr, t[2:])
                ops.append(txn.Op("write", {"addr": addr, "value": v}))
            elif k in ("maskwrite", "maskpoll"):
                v, implied, explicit = decl._encode(addr, t[2:])
                ops.append(txn.Op(k, {"addr": addr, "value": v, "mask": implied if explicit is None else explicit}))
            elif k == "block":
                words = [decl._encode(addr + 4 * n, g.split())[0] for n, g in enumerate(decl.BRACKET.findall(s))]
                col, row = _tile(addr)
                for x in t[2:]:
                    if x.startswith("tile="):
                        c, r = x[6:].split("r")
                        col, row = int(c), int(r)
                ops.append(txn.Op("blockwrite", {"addr": addr, "col": col, "row": row}, words))
            else:
                kv = dict(x.split("=") for x in t[2:])
                ops.append(txn.Op("ddr_patch", {"addr": addr, "arg": int(kv["arg"]), "plus": int(kv["plus"], 0)}))
        else:
            flush()
            raw.append(line)
    flush()
    return h, ops
