"""AIE2P register names from aie-rt's `xaie2pgbl_params.h`.

Tile kind by row: row 0 is the shim (NOC + PL modules), row 1 the memory
tile, rows 2+ compute tiles (memory module below 0x30000, core module
above). `name(addr)` gives the register at an array address; `block(addr)`
gives a coarse group (`DMA_BD`, `LOCK_VALUE`, `SS_MASTER`...), plus
`PROGRAM_MEMORY` / `DATA_MEMORY` for the memory ranges.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path

DEFAULT_HEADER = (
    Path.home()
    / "code/openflowlm-next/ironvenv/lib/python3.14/site-packages/mlir_aie/runtime_lib/x86_64/xaiengine/include/xaiengine/xaie2pgbl_params.h"
)
DEFINE = re.compile(r"^#define XAIE2PGBL_([A-Z0-9_]+?)\s+0x([0-9A-Fa-f]+)U?\s*$")
MODULES = {
    "shim": ("NOC_MODULE_", "PL_MODULE_"),
    "memtile": ("MEM_TILE_MODULE_",),
    "core": ("MEMORY_MODULE_", "CORE_MODULE_"),
}
FIELD_SUFFIX = ("_LSB", "_MASK", "_WIDTH", "_DEFVAL")


def kind(addr: int) -> str:
    row = (addr >> 20) & 0x1F
    return "shim" if row == 0 else "memtile" if row == 1 else "core"


@lru_cache(maxsize=1)
def table() -> dict[str, dict[int, str]]:
    path = Path(os.environ.get("NPUASM_REGS_HEADER", DEFAULT_HEADER))
    out: dict[str, dict[int, str]] = {k: {} for k in MODULES}
    for line in path.read_text().splitlines():
        m = DEFINE.match(line)
        if not m or m.group(1).endswith(FIELD_SUFFIX):
            continue
        name, off = m.group(1), int(m.group(2), 16)
        for k, prefixes in MODULES.items():
            for p in prefixes:
                if name.startswith(p):
                    out[k].setdefault(off, name[len(p):])
    return out


def name(addr: int) -> str | None:
    return table()[kind(addr)].get(addr & 0xFFFFF)


FIELD = re.compile(r"^#define XAIE2PGBL_([A-Z0-9_]+)_(LSB|WIDTH)\s+(\d+)U?\s*$")


@lru_cache(maxsize=1)
def fields_table() -> dict[str, dict[int, list[tuple[str, int, int]]]]:
    """kind -> register offset -> [(field, lsb, width)], ordered high bit first
    (the order aie-rt lists them)."""
    path = Path(os.environ.get("NPUASM_REGS_HEADER", DEFAULT_HEADER))
    regs_by_kind = {k: {v: o for o, v in t.items()} for k, t in table().items()}
    raw: dict[str, dict[str, int]] = {}
    for line in path.read_text().splitlines():
        m = FIELD.match(line)
        if m:
            raw.setdefault(m.group(1), {})[m.group(2)] = int(m.group(3))
    out: dict[str, dict[int, list[tuple[str, int, int]]]] = {k: {} for k in MODULES}
    for full, lw in raw.items():
        if "LSB" not in lw or "WIDTH" not in lw:
            continue
        for k, prefixes in MODULES.items():
            for p in prefixes:
                if not full.startswith(p):
                    continue
                rest = full[len(p):]
                # Longest register name that is a prefix of REG_FIELD.
                best = None
                for reg, off in regs_by_kind[k].items():
                    if rest.startswith(reg + "_") and (best is None or len(reg) > len(best[0])):
                        best = (reg, off)
                if best:
                    out[k].setdefault(best[1], []).append((rest[len(best[0]) + 1:], lw["LSB"], lw["WIDTH"]))
    for k in out:
        for off in out[k]:
            out[k][off].sort(key=lambda f: -f[1])
    return out


def fields(addr: int) -> list[tuple[str, int, int]]:
    return fields_table()[kind(addr)].get(addr & 0xFFFFF, [])


@lru_cache(maxsize=1)
def offsets() -> dict[str, dict[str, int]]:
    """kind -> register name -> offset."""
    return {k: {v: o for o, v in t.items()} for k, t in table().items()}


_DIGITS = re.compile(r"\d+")


def block(addr: int) -> str:
    k, off = kind(addr), addr & 0xFFFFF
    if k == "core" and 0x20000 <= off < 0x24000:
        return "core:PROGRAM_MEMORY"
    if (k == "core" and off < 0x10000) or (k == "memtile" and off < 0x80000):
        return f"{k}:DATA_MEMORY"
    n = name(addr)
    if n is None:
        return f"{k}:unnamed"
    n = _DIGITS.sub("#", n)
    for pfx, grp in (
        ("DMA_BD#", "DMA_BD"),
        ("LOCK#_VALUE", "LOCK_VALUE"),
        ("STREAM_SWITCH_MASTER_CONFIG", "SS_MASTER"),
        ("STREAM_SWITCH_SLAVE_", "SS_SLAVE"),
        ("STREAM_SWITCH_DETERMINISTIC_MERGE", "SS_MERGE"),
        ("DMA_S#MM_#_CTRL", "DMA_CH_CTRL"),
        ("DMA_MM#S_#_CTRL", "DMA_CH_CTRL"),
        ("DMA_S#MM_#_START_QUEUE", "DMA_CH_QUEUE"),
        ("DMA_MM#S_#_START_QUEUE", "DMA_CH_QUEUE"),
        ("CORE_CONTROL", "CORE_CONTROL"),
    ):
        if n.startswith(pfx):
            return f"{k}:{grp}"
    return f"{k}:{n}"
