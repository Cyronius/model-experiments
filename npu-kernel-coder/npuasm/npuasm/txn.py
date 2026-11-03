"""NPU transaction stream (`insts.bin`) codec: bytes <-> op list <-> text.

Format reference: mlir-aie `include/aie/Runtime/TxnEncoding.h` (header,
WRITE, BLOCKWRITE, MASKWRITE, MASKPOLL, TCT sync, DDR_PATCH, LOADPDI,
PREEMPT). Every op's reserved words are checked to be zero, so the text form
can leave them out and stay lossless; anything else is rejected by name.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

from . import cdo

WRITE, BLOCKWRITE, MASKWRITE, MASKPOLL, PREEMPT, LOADPDI = 0, 1, 3, 4, 6, 8
TCT, DDR_PATCH = 0x80, 0x81


class TxnError(ValueError):
    pass


@dataclass
class Header:
    major: int = 0
    minor: int = 1
    dev_gen: int = 4  # 4 = NPU2 (Strix, Krackan)
    rows: int = 6
    cols: int = 8
    memtile_rows: int = 1


@dataclass
class Op:
    kind: str
    f: dict = field(default_factory=dict)  # named fields
    data: list[int] = field(default_factory=list)  # blockwrite payload


def _need_zero(words, idx, what):
    for i in idx:
        if words[i] != 0:
            raise TxnError(f"{what}: reserved word {i} is {words[i]:#x}")


def parse(data: bytes) -> tuple[Header, list[Op]]:
    if len(data) % 4 or len(data) < 16:
        raise TxnError("not a whole number of words")
    w = struct.unpack(f"<{len(data) // 4}I", data)
    h = Header(w[0] & 0xFF, (w[0] >> 8) & 0xFF, (w[0] >> 16) & 0xFF, w[0] >> 24, w[1] & 0xFF, (w[1] >> 8) & 0xFF)
    if w[1] >> 16:
        raise TxnError("header word 1 high bits set")
    count, size = w[2], w[3]
    if size != len(data):
        raise TxnError(f"header size {size} != file size {len(data)}")
    ops: list[Op] = []
    i = 4
    while i < len(w):
        opc = w[i] & 0xFF
        if opc == WRITE:
            o = w[i : i + 6]
            _need_zero(o, (1, 3), "write")
            if o[0] != 0 or o[5] != 24:
                raise TxnError(f"write header {o[0]:#x} size {o[5]}")
            ops.append(Op("write", {"addr": o[2], "value": o[4]}))
            i += 6
        elif opc == BLOCKWRITE:
            if w[i] != BLOCKWRITE:
                raise TxnError(f"blockwrite opcode word {w[i]:#x}")
            nbytes = w[i + 3]
            if nbytes % 4 or nbytes < 16:
                raise TxnError(f"blockwrite size {nbytes}")
            if w[i + 1] >> 16:
                raise TxnError("blockwrite tile word high bits set")
            n = nbytes // 4
            ops.append(
                Op(
                    "blockwrite",
                    {"addr": w[i + 2], "col": w[i + 1] & 0xFF, "row": (w[i + 1] >> 8) & 0xFF},
                    list(w[i + 4 : i + n]),
                )
            )
            i += n
        elif opc in (MASKWRITE, MASKPOLL):
            o = w[i : i + 7]
            _need_zero(o, (1, 3), "maskwrite/maskpoll")
            if o[0] != opc or o[6] != 28:
                raise TxnError(f"mask op header {o[0]:#x} size {o[6]}")
            ops.append(Op("maskwrite" if opc == MASKWRITE else "maskpoll", {"addr": o[2], "value": o[4], "mask": o[5]}))
            i += 7
        elif opc == TCT:
            o = w[i : i + 4]
            if o[0] != TCT or o[1] != 16 or o[3] & 0xFF:
                raise TxnError(f"sync words {[hex(x) for x in o]}")
            if o[2] >> 24:
                raise TxnError("sync word 2 high bits set")
            ops.append(
                Op(
                    "sync",
                    {
                        "dir": o[2] & 0xFF,
                        "row": (o[2] >> 8) & 0xFF,
                        "col": (o[2] >> 16) & 0xFF,
                        "nrow": (o[3] >> 8) & 0xFF,
                        "ncol": (o[3] >> 16) & 0xFF,
                        "chan": o[3] >> 24,
                    },
                )
            )
            i += 4
        elif opc == DDR_PATCH:
            o = w[i : i + 12]
            if o[0] != DDR_PATCH or o[1] != 48:
                raise TxnError(f"ddr_patch words {[hex(x) for x in o[:2]]}")
            _need_zero(o, (2, 3, 4, 5, 7, 9), "ddr_patch")
            ops.append(Op("ddr_patch", {"addr": o[6], "arg": o[8], "plus": o[10] | (o[11] << 32)}))
            i += 12
        elif opc == LOADPDI:
            o = w[i : i + 4]
            if (o[0] >> 8) & 0xFF:
                raise TxnError("loadpdi opcode word bits 8-15 set")
            ops.append(Op("loadpdi", {"id": o[0] >> 16, "size": o[1], "addr": o[2] | (o[3] << 32)}))
            i += 4
        elif opc == PREEMPT:
            ops.append(Op("preempt", {"level": w[i] >> 8}))
            i += 1
        else:
            raise TxnError(f"unsupported opcode {opc:#x} at word {i}")
    if i != len(w):
        raise TxnError("last op runs past the end")
    if count != len(ops):
        raise TxnError(f"header op count {count} != parsed {len(ops)}")
    return h, ops


def serialize(h: Header, ops: list[Op]) -> bytes:
    out = [0, 0, 0, 0]
    for op in ops:
        f = op.f
        if op.kind == "write":
            out += [WRITE, 0, f["addr"], 0, f["value"], 24]
        elif op.kind == "blockwrite":
            out += [BLOCKWRITE, f["col"] | (f["row"] << 8), f["addr"], 4 * (4 + len(op.data)), *op.data]
        elif op.kind in ("maskwrite", "maskpoll"):
            out += [MASKWRITE if op.kind == "maskwrite" else MASKPOLL, 0, f["addr"], 0, f["value"], f["mask"], 28]
        elif op.kind == "sync":
            out += [
                TCT,
                16,
                f["dir"] | (f["row"] << 8) | (f["col"] << 16),
                (f["nrow"] << 8) | (f["ncol"] << 16) | (f["chan"] << 24),
            ]
        elif op.kind == "ddr_patch":
            p = f["plus"]
            out += [DDR_PATCH, 48, 0, 0, 0, 0, f["addr"], 0, f["arg"], 0, p & 0xFFFFFFFF, p >> 32]
        elif op.kind == "loadpdi":
            out += [LOADPDI | (f["id"] << 16), f["size"], f["addr"] & 0xFFFFFFFF, f["addr"] >> 32]
        elif op.kind == "preempt":
            out += [PREEMPT | (f["level"] << 8)]
        else:
            raise TxnError(f"unknown op {op.kind}")
    out[0] = (h.rows << 24) | (h.dev_gen << 16) | (h.minor << 8) | h.major
    out[1] = (h.memtile_rows << 8) | h.cols
    out[2] = len(ops)
    out[3] = 4 * len(out)
    return struct.pack(f"<{len(out)}I", *out)


# ---------------------------------------------------------------- text form


def _tile(addr: int) -> str:
    return f"c{addr >> 25}r{(addr >> 20) & 0x1F}+{addr & 0xFFFFF:#07x}"


def _addr(tok: str) -> int:
    if tok.startswith("c") and "+" in tok:
        cr, off = tok[1:].split("+")
        col, row = cr.split("r")
        return (int(col) << 25) | (int(row) << 20) | int(off, 0)
    return int(tok, 0)


def to_text(h: Header, ops: list[Op]) -> str:
    lines = [
        f"txn {h.major}.{h.minor} gen={h.dev_gen} rows={h.rows} cols={h.cols} memtile_rows={h.memtile_rows}"
    ]
    for op in ops:
        f = op.f
        if op.kind == "write":
            lines.append(f"write {_tile(f['addr'])} {f['value']:#x}")
        elif op.kind in ("maskwrite", "maskpoll"):
            lines.append(f"{op.kind} {_tile(f['addr'])} {f['value']:#x} mask={f['mask']:#x}")
        elif op.kind == "blockwrite":
            a = f["addr"]
            tile = ""
            if (f["col"], f["row"]) != (a >> 25, (a >> 20) & 0x1F):
                tile = f" tile=c{f['col']}r{f['row']}"
            lines.append(f"blockwrite {_tile(a)} {len(op.data)}{tile}")
            items = cdo._runs(op.data)
            for j in range(0, len(items), 8):
                lines.append("  " + " ".join(items[j : j + 8]))
        elif op.kind == "sync":
            lines.append(
                "sync c{col}r{row} dir={dir} chan={chan} ncol={ncol} nrow={nrow}".format(**f)
            )
        elif op.kind == "ddr_patch":
            lines.append(f"ddr_patch {_tile(f['addr'])} arg={f['arg']} plus={f['plus']:#x}")
        elif op.kind == "loadpdi":
            lines.append(f"loadpdi id={f['id']} size={f['size']} addr={f['addr']:#x}")
        elif op.kind == "preempt":
            lines.append(f"preempt level={f['level']}")
    return "\n".join(lines) + "\n"


def _kv(tokens) -> dict:
    out = {}
    for t in tokens:
        k, v = t.split("=")
        out[k] = v
    return out


def from_text(text: str) -> tuple[Header, list[Op]]:
    lines = [l.split(";", 1)[0].rstrip() for l in text.splitlines()]
    lines = [l for l in lines if l.strip()]
    if not lines or not lines[0].startswith("txn "):
        raise TxnError("missing txn header line")
    t = lines[0].split()
    major, minor = (int(x) for x in t[1].split("."))
    kv = _kv(t[2:])
    h = Header(major, minor, int(kv["gen"]), int(kv["rows"]), int(kv["cols"]), int(kv["memtile_rows"]))
    ops: list[Op] = []
    want = 0
    for line in lines[1:]:
        if line.startswith("  "):
            if want <= 0:
                raise TxnError("unexpected data line")
            vals = cdo._unruns(line.split())
            ops[-1].data += vals
            want -= len(vals)
            continue
        if want:
            raise TxnError("blockwrite data ended early")
        t = line.split()
        k = t[0]
        if k == "write":
            ops.append(Op("write", {"addr": _addr(t[1]), "value": int(t[2], 0)}))
        elif k in ("maskwrite", "maskpoll"):
            ops.append(Op(k, {"addr": _addr(t[1]), "value": int(t[2], 0), "mask": int(_kv(t[3:])["mask"], 0)}))
        elif k == "blockwrite":
            a = _addr(t[1])
            col, row = a >> 25, (a >> 20) & 0x1F
            extra = _kv(t[3:])
            if "tile" in extra:
                c, r = extra["tile"][1:].split("r")
                col, row = int(c), int(r)
            ops.append(Op("blockwrite", {"addr": a, "col": col, "row": row}))
            want = int(t[2])
        elif k == "sync":
            c, r = t[1][1:].split("r")
            kv = {key: int(v) for key, v in _kv(t[2:]).items()}
            ops.append(Op("sync", {"col": int(c), "row": int(r), **kv}))
        elif k == "ddr_patch":
            kv = _kv(t[2:])
            ops.append(Op("ddr_patch", {"addr": _addr(t[1]), "arg": int(kv["arg"]), "plus": int(kv["plus"], 0)}))
        elif k == "loadpdi":
            kv = _kv(t[1:])
            ops.append(Op("loadpdi", {"id": int(kv["id"]), "size": int(kv["size"]), "addr": int(kv["addr"], 0)}))
        elif k == "preempt":
            ops.append(Op("preempt", {"level": int(_kv(t[1:])["level"])}))
        else:
            raise TxnError(f"unknown op {k!r}")
    if want:
        raise TxnError("file ended inside blockwrite data")
    return h, ops
