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
