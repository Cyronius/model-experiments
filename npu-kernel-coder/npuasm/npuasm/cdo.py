"""CDO (configuration data object) codec: bytes <-> op list <-> text.

Format reference: Xilinx bootgen `cdo-driver/cdo_driver.c` (Apache-2.0).
A CDO is a 5-word header followed by commands. Each command header is
`Reserved[31:24] | Length[23:16] | Handler[15:8] | API-ID[7:0]`; length 255
means the next word holds the real length.

The writer that produced a CDO decides two things the op list does not carry:
NOP padding (inserted so every DMA-write payload starts 16-byte aligned) and
whether a 64-bit command keeps its high address word. Both are recomputed
here; `serialize` takes a `style` that says which rules apply.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

MAGIC = 0x004F4443  # "CDO\0"
VERSION = 0x00000200

# PLM generic module (handler 1) command ids.
MASK_POLL = 0x101
MASK_WRITE = 0x102
WRITE = 0x103
DELAY = 0x104
DMA_WRITE = 0x105
MASK_POLL64 = 0x106
MASK_WRITE64 = 0x107
WRITE64 = 0x108
DMA_XFER = 0x109
NOP = 0x111

NAMES = {
    MASK_POLL: "maskpoll32",
    MASK_WRITE: "maskwrite32",
    WRITE: "write32",
    DELAY: "delay",
    DMA_WRITE: "dmawrite",
    MASK_POLL64: "maskpoll",
    MASK_WRITE64: "maskwrite",
    WRITE64: "write",
    NOP: "nop",
}
IDS = {v: k for k, v in NAMES.items()}


class CdoError(ValueError):
    pass


@dataclass
class Op:
    """One CDO command. `args` are the payload words after the header."""

    cmd: int
    args: list[int] = field(default_factory=list)
    ref: str | None = None  # DMA write whose payload is a named blob (text form only)

    @property
    def name(self) -> str:
        return NAMES.get(self.cmd, f"cmd_{self.cmd:#x}")


def header(length_words: int) -> list[int]:
    s = 4 + MAGIC + VERSION + length_words
    return [4, MAGIC, VERSION, length_words, (~s) & 0xFFFFFFFF]


def parse(data: bytes, *, keep_nops: bool = False) -> list[Op]:
    if len(data) % 4:
        raise CdoError(f"length {len(data)} is not a whole number of words")
    words = struct.unpack(f"<{len(data) // 4}I", data)
    if len(words) < 5 or words[1] != MAGIC:
        raise CdoError("missing CDO header")
    if words[0] != 4 or words[2] != VERSION:
        raise CdoError(f"unsupported header {words[:3]}")
    if list(words[:5]) != header(words[3]):
        raise CdoError("bad header checksum")
    if words[3] != len(words) - 5:
        raise CdoError(f"header length {words[3]} != {len(words) - 5} words")
    return parse_body(words[5:], keep_nops=keep_nops)


def parse_body(words, *, keep_nops: bool = False) -> list[Op]:
    ops: list[Op] = []
    i = 0
    n = len(words)
    while i < n:
        h = words[i]
        cmd = h & 0xFFFF
        length = (h >> 16) & 0xFF
        i += 1
        if (h >> 24) != 0:
            raise CdoError(f"reserved bits set in command header {h:#010x}")
        if length == 255:
            length = words[i]
            i += 1
        if i + length > n:
            raise CdoError(f"command {cmd:#x} runs past the end")
        args = list(words[i : i + length])
        i += length
        if cmd == NOP and any(args):
            raise CdoError("NOP with nonzero payload")
        ops.append(Op(cmd, args))
    if keep_nops:
        return ops
    # Drop only the NOPs the writer re-creates: the pad right before a DMA
    # write. Any other NOP (cdo-driver starts every file with one) stays.
    return [
        op
        for j, op in enumerate(ops)
        if not (op.cmd == NOP and j + 1 < len(ops) and ops[j + 1].cmd == DMA_WRITE)
    ]


@dataclass(frozen=True)
class Style:
    """Writer rules. `aligned_dma`: pad so DMA payloads start 16-byte aligned.
    `narrow`: rewrite 64-bit commands whose high address word is 0 into the
    32-bit forms (what bootgen does when it merges CDOs into a PDI).
    `split_pad`: pad with one-word NOPs (bootgen) instead of one NOP with a
    zero payload (cdo-driver). `base`: byte offset of the first body word in
    the final file, for the alignment rule."""

    aligned_dma: bool = True
    narrow: bool = False
    split_pad: bool = False
    base: int = 20


CDO_DRIVER = Style()
BOOTGEN_MERGED = Style(narrow=True)  # input must keep the part files' NOPs


def _cmd_words(cmd: int, args: list[int]) -> list[int]:
    if len(args) < 255:
        return [(len(args) << 16) | cmd, *args]
    return [(255 << 16) | cmd, len(args), *args]


def _narrowed(op: Op) -> Op:
    a = op.args
    if op.cmd == WRITE64 and a[0] == 0:
        return Op(WRITE, a[1:])
    if op.cmd == MASK_WRITE64 and a[0] == 0:
        return Op(MASK_WRITE, a[1:])
    if op.cmd == MASK_POLL64 and a[0] == 0:
        return Op(MASK_POLL, a[1:])
    return op


def serialize_body(ops: list[Op], style: Style = CDO_DRIVER) -> list[int]:
    out: list[int] = []
    for op in ops:
        if op.cmd == NOP:
            out += _cmd_words(NOP, op.args)
            continue
        if style.narrow:
            op = _narrowed(op)
        if op.cmd == DMA_WRITE and style.aligned_dma:
            pre = 3 if len(op.args) < 255 else 4  # header (+len) + addr hi/lo
            pos = style.base + 4 * len(out) + 4 * pre
            pad = (16 - pos % 16) % 16
            if pad >= 4 and style.split_pad:
                out += [NOP] * (pad // 4)
            elif pad >= 4:
                out += _cmd_words(NOP, [0] * (pad // 4 - 1))
        out += _cmd_words(op.cmd, op.args)
    return out


def serialize(ops: list[Op], style: Style = CDO_DRIVER) -> bytes:
    body = serialize_body(ops, style)
    words = header(len(body)) + body
    return struct.pack(f"<{len(words)}I", *words)


# ---------------------------------------------------------------- text form


def _tile(addr: int) -> str:
    col, row, off = addr >> 25, (addr >> 20) & 0x1F, addr & 0xFFFFF
    return f"c{col}r{row}+{off:#07x}"


def to_text(ops: list[Op], *, names=None) -> str:
    """One op per line. Addresses are written as `cCrR+offset` (array address
    col<<25 | row<<20 | offset). A DMA write's data follows on indented lines,
    eight words each. `names(addr)` may return a register name for comments."""
    lines: list[str] = []
    for op in ops:
        a = op.args
        if op.cmd in (WRITE64, MASK_WRITE64, MASK_POLL64, DMA_WRITE) and a[0] != 0:
            raise CdoError("high address word set; text form assumes 0")
        if op.cmd == WRITE64:
            line = f"write {_tile(a[1])} {a[2]:#x}"
            addr = a[1]
        elif op.cmd == MASK_WRITE64:
            line = f"maskwrite {_tile(a[1])} {a[3]:#x} mask={a[2]:#x}"
            addr = a[1]
        elif op.cmd == MASK_POLL64:
            line = f"maskpoll {_tile(a[1])} {a[3]:#x} mask={a[2]:#x} timeout={a[4]}"
            addr = a[1]
        elif op.cmd == DMA_WRITE:
            data = a[2:]
            addr = a[1]
            if op.ref:
                lines.append(f"dmawrite {_tile(addr)} @{op.ref}" + _comment(names, addr))
                continue
            line = f"dmawrite {_tile(a[1])} {len(data)}"
            lines.append(line + _comment(names, addr))
            items = _runs(data)
            for j in range(0, len(items), 8):
                lines.append("  " + " ".join(items[j : j + 8]))
            continue
        else:
            line = f"{op.name} " + " ".join(f"{x:#x}" for x in a)
            addr = None
        lines.append(line + _comment(names, addr))
    return "\n".join(lines) + "\n"


def _runs(data: list[int]) -> list[str]:
    """Data words as items; a run of 3 or more equal words is `XXXXXXXX*N`."""
    items, i = [], 0
    while i < len(data):
        j = i
        while j < len(data) and data[j] == data[i]:
            j += 1
        n = j - i
        if n >= 3:
            items.append(f"{data[i]:08x}*{n}")
        else:
            items.extend(f"{data[i]:08x}" for _ in range(n))
        i = j
    return items


def _unruns(items: list[str]) -> list[int]:
    out = []
    for t in items:
        v, _, n = t.partition("*")
        out += [int(v, 16)] * (int(n) if n else 1)
    return out


def _comment(names, addr) -> str:
    if names is None or addr is None:
        return ""
    n = names(addr)
    return f"  ; {n}" if n else ""


def _addr(tok: str) -> int:
    if tok.startswith("c") and "r" in tok and "+" in tok:
        cr, off = tok[1:].split("+")
        col, row = cr.split("r")
        return (int(col) << 25) | (int(row) << 20) | int(off, 0)
    return int(tok, 0)


def _kv(tok: str, key: str) -> int:
    k, v = tok.split("=")
    if k != key:
        raise CdoError(f"expected {key}=, got {tok}")
    return int(v, 0)


def from_text(text: str, *, resolve=None) -> list[Op]:
    """Parse the text form. `resolve(name)` returns the words for a
    `dmawrite ADDR @name` line."""
    ops: list[Op] = []
    pending: Op | None = None
    want = 0
    for raw in text.splitlines():
        line = raw.split(";", 1)[0].rstrip()
        if not line.strip():
            continue
        if raw.startswith("  ") and pending is not None:
            vals = _unruns(line.split())
            pending.args += vals
            want -= len(vals)
            if want < 0:
                raise CdoError("too many dmawrite data words")
            if want == 0:
                pending = None
            continue
        if pending is not None:
            raise CdoError("dmawrite data ended early")
        t = line.split()
        kind = t[0]
        if kind == "write":
            ops.append(Op(WRITE64, [0, _addr(t[1]), int(t[2], 0)]))
        elif kind == "maskwrite":
            ops.append(Op(MASK_WRITE64, [0, _addr(t[1]), _kv(t[3], "mask"), int(t[2], 0)]))
        elif kind == "maskpoll":
            ops.append(
                Op(MASK_POLL64, [0, _addr(t[1]), _kv(t[3], "mask"), int(t[2], 0), _kv(t[4], "timeout")])
            )
        elif kind == "dmawrite" and t[2].startswith("@"):
            if resolve is None:
                raise CdoError(f"no resolver for {t[2]}")
            ops.append(Op(DMA_WRITE, [0, _addr(t[1]), *resolve(t[2][1:])], ref=t[2][1:]))
        elif kind == "dmawrite":
            pending = Op(DMA_WRITE, [0, _addr(t[1])])
            want = int(t[2])
            ops.append(pending)
            if want == 0:
                pending = None
        elif kind in IDS:
            ops.append(Op(IDS[kind], [int(x, 0) for x in t[1:]]))
        elif kind.startswith("cmd_"):
            ops.append(Op(int(kind[4:], 0), [int(x, 0) for x in t[1:]]))
        else:
            raise CdoError(f"unknown op {kind!r}")
    if pending is not None:
        raise CdoError("file ended inside dmawrite data")
    return ops
