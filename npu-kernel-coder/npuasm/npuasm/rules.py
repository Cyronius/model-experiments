"""Static checks on npuasm output before it may run on the NPU.

Each rule cites the finding it encodes (xdna2-notes) and was calibrated on
the local corpus: every program there runs in a working engine, so a rule
that flags one of them is wrong. Rules that failed that test were removed
or loosened, with the reason noted.

Core program rules:
- size: code fits in 16 KiB program memory (F-0006, F-0242).
- branch-target: an immediate branch target sits on a 16-byte boundary, or
  only whole NOP bundles precede it in its 16-byte line (F-0288; hipfire
  `isa::rules::BranchTarget`). Indirect transfers (`ret lr`, register
  targets) are allowed: Peano uses them everywhere.
- delay-slots: every control transfer is followed by at least 5 bundles,
  none of which is another control transfer or `done`, and no immediate
  branch target lands inside them (hipfire `DelaySlot`). hipfire also bans
  lock requests there because it never ran one; Peano puts `acq`/`rel` in
  call delay slots in every local build (5,139 cases in a sample of 473
  programs), so that part is dropped.
- lock-spacing: lock requests at least 4 issue bundles apart in straight-
  line order (hipfire `LockSpacing`).
Hardware zero-overhead loops are allowed: Peano's form ran exactly on Strix
Halo (npuasm/silicon/zol), unlike hipfire's own (`ZeroOverheadLoop` dropped).

Array configuration rules (CDO and transaction ops):
- zero-length-bd: no valid buffer descriptor with BUFFER_LENGTH 0 (F-0059).
- shim-axqos-15: no shim BD with AxQoS 15 (hipfire: wedged the NPU until
  reboot on Strix Halo, Linux 7.0).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from . import cdo, regs

PROGRAM_MEMORY = 16 * 1024
DELAY_SLOTS = 5
LOCK_SPACING = 4
CONTROL = {"j", "jl", "jz", "jnz", "jnzd", "ret"}
LOCKREQ = {"acq", "rel", "acq.cond", "rel.cond"}
TARGET = re.compile(r"^(?:j|jl|jz|jnz)\s+(?:[^#;]*,\s*)?#(0x[0-9a-f]+)$")
NOP = re.compile(r"^nop[a-z0-9]*$")


@dataclass
class Violation:
    rule: str
    where: str
    detail: str


def _slots(text: str) -> list[str]:
    return [s.strip() for s in text.split(";") if s.strip()]


def _mn(slot: str) -> str:
    return slot.split()[0] if slot else ""


def check_core(insns, name: str = "core") -> list[Violation]:
    """`insns`: objects with .addr, .raw, .text (core.Insn), address order."""
    out: list[Violation] = []
    if not insns:
        return out
    end = insns[-1].addr + len(insns[-1].raw)
    if end > PROGRAM_MEMORY:
        out.append(Violation("size", name, f"{end} B > {PROGRAM_MEMORY} B program memory"))
    idx = {ins.addr: i for i, ins in enumerate(insns)}
    all_nop = [all(NOP.match(s) for s in _slots(ins.text)) for ins in insns]
    targets = set()
    for ins in insns:
        for s in _slots(ins.text):
            m = TARGET.match(s)
            if m:
                targets.add(int(m.group(1), 16))
    for t in sorted(targets):
        if t % 16 == 0:
            continue
        i = idx.get(t)
        if i is None:
            out.append(Violation("branch-target", name, f"target {t:#x} is not a bundle start"))
            continue
        line_start = t - t % 16
        ok = all(all_nop[j] for j in range(i) if insns[j].addr >= line_start)
        if not ok:
            out.append(Violation("branch-target", name, f"target {t:#x} is unaligned after real instructions in its 16-byte line"))
    last_lock = None
    for i, ins in enumerate(insns):
        mns = [_mn(s) for s in _slots(ins.text)]
        if any(m in LOCKREQ for m in mns):
            if last_lock is not None and i - last_lock < LOCK_SPACING:
                out.append(Violation("lock-spacing", name, f"{ins.addr:#x}: lock request {i - last_lock} bundles after the last"))
            last_lock = i
        if any(m in CONTROL for m in mns):
            if i + DELAY_SLOTS >= len(insns) + 1:
                out.append(Violation("delay-slots", name, f"{ins.addr:#x}: fewer than {DELAY_SLOTS} bundles follow"))
                continue
            for j in range(i + 1, min(i + 1 + DELAY_SLOTS, len(insns))):
                dm = [_mn(s) for s in _slots(insns[j].text)]
                bad = [m for m in dm if m in CONTROL or m == "done"]
                if bad:
                    out.append(Violation("delay-slots", name, f"{insns[j].addr:#x}: {bad[0]} in a delay slot of {ins.addr:#x}"))
                if insns[j].addr in targets:
                    out.append(Violation("delay-slots", name, f"{insns[j].addr:#x}: branch target inside the delay slots of {ins.addr:#x}"))
    return out


def _bd_fields(addr: int, words: list[int]) -> dict[str, int]:
    """Named field values over consecutive BD registers."""
    vals: dict[str, int] = {}
    for k, w in enumerate(words):
        for fname, lsb, width in regs.fields(addr + 4 * k):
            vals[fname] = (w >> lsb) & ((1 << width) - 1)
    return vals


def check_bd(addr: int, words: list[int], where: str) -> list[Violation]:
    n = regs.name(addr) or ""
    if not re.match(r"^DMA_BD\d+_0$", n):
        return []
    f = _bd_fields(addr, words)
    out = []
    if f.get("VALID_BD") == 1 and f.get("BUFFER_LENGTH", 1) == 0:
        out.append(Violation("zero-length-bd", where, f"{n} at c{addr >> 25}r{(addr >> 20) & 31} is valid with BUFFER_LENGTH 0"))
    if regs.kind(addr) == "shim" and f.get("AXQOS") == 15:
        out.append(Violation("shim-axqos-15", where, f"{n} at c{addr >> 25}r0 sets AxQoS 15"))
    return out


def check_cdo(ops, where: str) -> list[Violation]:
    out = []
    for op in ops:
        if op.cmd == cdo.DMA_WRITE and not op.ref:
            out += check_bd(op.args[1], list(op.args[2:]), where)
    return out


def check_txn(ops, where: str) -> list[Violation]:
    out = []
    for op in ops:
        if op.kind == "blockwrite":
            out += check_bd(op.f["addr"], op.data, where)
    return out
