"""AIE2P core programs: ELF -> assembly text -> bytes.

Two backends:
- `core` (default): `rust/target/release/npuasm-core`, built on hipfire's
  peacemaker-npu tables (generated from llvm-aie TableGen). It decodes
  bundles to llvm-objdump's syntax and assembles that text back to the same
  bytes, with no vendor tools. Every AIE2P instruction in the local corpus is
  covered, including the three Peano's assembler rejects.
- `peano` (`NPUASM_BACKEND=peano`): Peano's `llvm-objdump` and `clang -c`,
  kept as an oracle. Bundles its assembler rejects (`vinsert.32`, `event`,
  `vinsert.16`) fall back to `.byte`.

The lifted text keeps absolute branch targets and addresses (the linked
image starts at 0), so it reassembles without relocations. Symbols become
`//` comments, which also avoids duplicate local labels across objects.
"""

from __future__ import annotations

import os
import re
import struct
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

PT_LOAD = 1


class CoreError(RuntimeError):
    pass


def core_bin() -> Path | None:
    env = os.environ.get("NPUASM_CORE")
    p = Path(env) if env else Path(__file__).resolve().parents[1] / "rust/target/release/npuasm-core"
    return p if p.exists() else None


def backend() -> str:
    want = os.environ.get("NPUASM_BACKEND", "core")
    if want == "core" and core_bin() is None:
        raise CoreError("npuasm-core not built: run `cargo build --release` in npuasm/rust (or set NPUASM_BACKEND=peano)")
    if want not in ("core", "peano"):
        raise CoreError(f"unknown NPUASM_BACKEND {want!r}")
    return want


def peano_bin() -> Path:
    env = os.environ.get("NPUASM_PEANO")
    cands = [Path(env)] if env else []
    cands.append(Path.home() / "code/openflowlm-next/ironvenv/lib/python3.14/site-packages/llvm-aie/bin")
    for c in cands:
        if (c / "clang").exists() and (c / "llvm-objdump").exists():
            return c
    raise CoreError("Peano not found; set NPUASM_PEANO to llvm-aie/bin")


@dataclass
class Segment:
    vaddr: int
    data: bytes
    memsz: int
    flags: int


@dataclass
class Elf:
    machine: int
    flags: int
    segments: list[Segment]
    sections: dict[str, tuple[int, int, int]]  # name -> (addr, offset, size)
    raw: bytes

    def section(self, name: str) -> bytes:
        _, off, size = self.sections[name]
        return self.raw[off : off + size]


def read_elf(data: bytes) -> Elf:
    if data[:4] != b"\x7fELF" or data[4] != 1 or data[5] != 1:
        raise CoreError("not a little-endian ELF32 file")
    (machine,) = struct.unpack_from("<H", data, 0x12)
    flags, = struct.unpack_from("<I", data, 0x24)
    phoff, shoff = struct.unpack_from("<II", data, 0x1C)
    phentsize, phnum, shentsize, shnum, shstrndx = struct.unpack_from("<HHHHH", data, 0x2A)
    segs = []
    for i in range(phnum):
        p_type, p_off, p_vaddr, _p_paddr, p_filesz, p_memsz, p_flags, _ = struct.unpack_from(
            "<8I", data, phoff + i * phentsize
        )
        if p_type == PT_LOAD:
            segs.append(Segment(p_vaddr, data[p_off : p_off + p_filesz], p_memsz, p_flags))
    shdrs = [struct.unpack_from("<10I", data, shoff + i * shentsize) for i in range(shnum)]
    sections = {}
    if shnum:
        stro = shdrs[shstrndx][4]
        for sh in shdrs:
            nm = data[stro + sh[0] : data.index(b"\0", stro + sh[0])].decode()
            sections[nm] = (sh[3], sh[4], sh[5] if sh[1] != 8 else 0)  # NOBITS has no file bytes
    return Elf(machine, flags, segs, sections, data)


def text_symbols(data: bytes) -> dict[int, list[str]]:
    """Function and label symbols defined in `.text`, by address."""
    shoff, = struct.unpack_from("<I", data, 0x20)
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", data, 0x2E)
    shdrs = [struct.unpack_from("<10I", data, shoff + i * shentsize) for i in range(shnum)]
    if not shnum:
        return {}
    stro = shdrs[shstrndx][4]
    name = lambda tab, off: data[tab + off : data.index(b"\0", tab + off)].decode()  # noqa: E731
    text_idx = next((i for i, sh in enumerate(shdrs) if name(stro, sh[0]) == ".text"), None)
    out: dict[int, list[str]] = {}
    for sh in shdrs:
        if sh[1] != 2:  # SHT_SYMTAB
            continue
        strtab = shdrs[sh[6]][4]
        for j in range(sh[5] // 16):
            st_name, st_value, _size, st_info, _other, st_shndx = struct.unpack_from("<IIIBBH", data, sh[4] + 16 * j)
            if st_shndx != text_idx or (st_info & 0xF) not in (0, 2) or not st_name:
                continue
            nm = name(strtab, st_name)
            if nm and not nm.startswith(("$", "_ctors_", "_dtors_", "_init_array_", "_fini_array_")):
                out.setdefault(st_value, []).append(nm)
    return {a: sorted(v) for a, v in out.items()}


TARGET = re.compile(r"^(?:(?:j|jl|jz|jnz)\s+(?:[^#]*,\s*)?|movxm\s+(?:ls|le),\s*)#(0x[0-9a-f]+)$")
LABEL = re.compile(r"^[A-Za-z_.$][A-Za-z0-9_.$]*$")
LINE = re.compile(r"^\s*([0-9a-f]+): ((?:[0-9a-f]{2} ?)+?) *\t(.*)$")
SYM = re.compile(r"^([0-9a-f]{8}) <(.+)>:$")


@dataclass
class Insn:
    addr: int
    raw: bytes
    text: str


@dataclass
class Listing:
    """Disassembled program: instructions in address order, plus symbols."""

    insns: list[Insn]
    symbols: dict[int, list[str]]  # addr -> names

    def labels(self) -> dict[int, str]:
        """A name for every code address a branch (`j`, `jl`, `jz`, `jnz`) or
        hardware-loop bound (`movxm ls/le`) points at, when that address
        starts a bundle (or is the end of the program)."""
        starts = {ins.addr for ins in self.insns}
        if self.insns:
            starts.add(self.insns[-1].addr + len(self.insns[-1].raw))
        targets = set()
        for ins in self.insns:
            for slot in ins.text.split(";"):
                m = TARGET.match(slot.strip())
                if m and int(m.group(1), 16) in starts:
                    targets.add(int(m.group(1), 16))
        count: dict[str, int] = {}
        for names in self.symbols.values():
            for n in names:
                count[n] = count.get(n, 0) + 1
        out = {}
        for a in sorted(targets):
            good = [n for n in self.symbols.get(a, ()) if count[n] == 1 and LABEL.match(n) and not n.startswith(".L")]
            out[a] = good[0] if good else f".L{a:x}"
        return out

    def render(self, raw_lines: set[int] = frozenset(), *, labels: bool = False) -> str:
        """Assembly text. Instructions whose index is in `raw_lines` are
        written as `.byte` with their text as a comment (the assembler can't
        parse everything the disassembler prints). With `labels`, branch and
        loop-bound targets become `name:` lines and `#name` operands (only
        npuasm-core assembles those)."""
        names = self.labels() if labels else {}

        def relabel(text: str) -> str:
            slots = []
            for slot in text.split(";"):
                m = TARGET.match(slot.strip())
                if m and int(m.group(1), 16) in names:
                    slot = slot.replace("#" + m.group(1), "#" + names[int(m.group(1), 16)])
                slots.append(slot)
            return ";".join(slots)

        out = ["\t.text"]
        for i, ins in enumerate(self.insns):
            for name in self.symbols.get(ins.addr, ()):
                if names.get(ins.addr) != name:
                    out.append(f"// {name} @ {ins.addr:#x}")
            if ins.addr in names:
                out.append(f"{names[ins.addr]}:")
            if i in raw_lines:
                out.append("\t.byte " + ",".join(f"{b:#04x}" for b in ins.raw) + "  // " + ins.text)
            else:
                out.append("\t" + (relabel(ins.text) if names else ins.text))
        if self.insns:
            end = self.insns[-1].addr + len(self.insns[-1].raw)
            if end in names:
                out.append(f"{names[end]}:")
        return "\n".join(out) + "\n"

    def line_of(self) -> dict[int, int]:
        """1-based line number in `render()` -> instruction index."""
        m, line = {}, 1
        for i, ins in enumerate(self.insns):
            line += len(self.symbols.get(ins.addr, ()))
            line += 1
            m[line] = i
        return m


def lift(elf_path: Path, *, peano: Path | None = None) -> Listing:
    if peano is None and backend() == "core":
        return _lift_core(elf_path)
    peano = peano or peano_bin()
    out = subprocess.run(
        [str(peano / "llvm-objdump"), "-d", "-z", "--triple=aie2p", str(elf_path)],
        check=True, capture_output=True, text=True,
    ).stdout
    insns, symbols = [], {}
    expect = 0
    for raw in out.splitlines():
        m = SYM.match(raw)
        if m:
            symbols.setdefault(int(m.group(1), 16), []).append(m.group(2))
            continue
        m = LINE.match(raw)
        if not m:
            continue
        addr = int(m.group(1), 16)
        b = bytes.fromhex(m.group(2).replace(" ", ""))
        if addr != expect:
            raise CoreError(f"{elf_path}: gap in disassembly at {addr:#x} (expected {expect:#x})")
        expect = addr + len(b)
        insn = m.group(3).strip()
        if not insn or insn.startswith("<unknown>"):
            raise CoreError(f"{elf_path}: undecodable bytes at {addr:#x}")
        insns.append(Insn(addr, b, re.sub(r"\s+", " ", insn).replace(" ;", ";")))
    return Listing(insns, symbols)


def _lift_core(elf_path: Path) -> Listing:
    r = subprocess.run([str(core_bin()), "lift", str(elf_path)], capture_output=True, text=True)
    if r.returncode:
        raise CoreError(f"{elf_path}: {r.stderr.strip()}")
    insns = []
    expect = 0
    for line in r.stdout.splitlines():
        a, hx, text = line.split("\t", 2)
        addr = int(a, 16)
        if addr != expect:
            raise CoreError(f"{elf_path}: gap in disassembly at {addr:#x}")
        b = bytes.fromhex(hx)
        expect = addr + len(b)
        insns.append(Insn(addr, b, re.sub(r"\s+", " ", text).replace(" ;", ";").strip()))
    return Listing(insns, text_symbols(Path(elf_path).read_bytes()))


def lift_bytes(code: bytes) -> Listing:
    """Decode raw program bytes (from address 0) with npuasm-core."""
    with tempfile.TemporaryDirectory(prefix="npuasm-") as d:
        b = Path(d, "code.bin")
        b.write_bytes(code)
        r = subprocess.run([str(core_bin()), "dis", str(b)], capture_output=True, text=True)
        if r.returncode:
            raise CoreError(r.stderr.strip())
    insns = []
    for line in r.stdout.splitlines():
        a, hx, text = line.split("\t", 2)
        insns.append(Insn(int(a, 16), bytes.fromhex(hx), re.sub(r"\s+", " ", text).replace(" ;", ";").strip()))
    return Listing(insns, {})


def lift_text(elf_path: Path, *, peano: Path | None = None) -> str:
    return lift(elf_path, peano=peano).render()


ERR = re.compile(r"core\.s:(\d+):(\d+): error: (.*)")


def _run_as(asm: str, peano: Path) -> tuple[bytes | None, list[tuple[int, str]]]:
    with tempfile.TemporaryDirectory(prefix="npuasm-") as d:
        s, o = Path(d, "core.s"), Path(d, "core.o")
        s.write_text(asm)
        r = subprocess.run(
            [str(peano / "clang"), "--target=aie2p", "-c", str(s), "-o", str(o)],
            capture_output=True, text=True,
        )
        if r.returncode:
            errs = [(int(m.group(1)), int(m.group(2)), m.group(3)) for m in ERR.finditer(r.stderr)]
            if not errs:
                raise CoreError("assembler failed:\n" + r.stderr[-2000:])
            return None, errs
        return read_elf(o.read_bytes()).section(".text"), []


def _asm_core(asm: str) -> bytes:
    with tempfile.TemporaryDirectory(prefix="npuasm-") as d:
        s, o = Path(d, "core.s"), Path(d, "core.bin")
        s.write_text(asm)
        r = subprocess.run([str(core_bin()), "asm", str(s), str(o)], capture_output=True, text=True)
        if r.returncode:
            raise CoreError("npuasm-core asm failed: " + r.stderr.strip()[-2000:])
        return o.read_bytes()


def assemble(asm: str, *, peano: Path | None = None) -> bytes:
    """Assemble text; return the `.text` section bytes."""
    if peano is None and backend() == "core":
        return _asm_core(asm)
    out, errs = _run_as(asm, peano or peano_bin())
    if out is None:
        raise CoreError("assembler failed: " + "; ".join(f"line {l}: {e}" for l, _, e in errs[:5]))
    return out


def _slot_at(line: str, col: int) -> str:
    """The `;`-separated slot of a bundle that contains column `col`."""
    a = line.rfind(";", 0, max(col, 0)) + 1
    b = line.find(";", col)
    return line[a : b if b >= 0 else len(line)].strip()


def assemble_listing(lst: Listing, *, peano: Path | None = None, max_rounds: int = 6):
    """Assemble a listing, falling back to `.byte` for bundles the assembler
    rejects. Returns (bytes, raw_line_indices, errors_seen), where each error
    is (slot text at the error column, message). The core backend needs
    no fallback."""
    if peano is None and backend() == "core":
        return _asm_core(lst.render(labels=True)), set(), []
    peano = peano or peano_bin()
    raw: set[int] = set()
    seen: list[tuple[str, str]] = []
    for _ in range(max_rounds):
        out, errs = _run_as(lst.render(raw), peano)
        if out is not None:
            return out, raw, seen
        lines = lst.line_of()
        # After a bundle fails to parse, the assembler often rejects the next
        # one as "incorrect bundle" too. Mark real rejections first; mark
        # knock-on errors only if nothing else is left.
        real = [x for x in errs if x[2] != "incorrect bundle"]
        errs = real or errs
        new = {lines[l] for l, _, _ in errs if l in lines} - raw
        if not new:
            raise CoreError("assembler errors outside instruction lines: " + str(errs[:3]))
        for l, col, e in errs:
            if l in lines:
                seen.append((_slot_at("\t" + lst.insns[lines[l]].text, col - 1), e))
        raw |= new
    raise CoreError("assembler still failing after fallback rounds")


def code_segment(elf: Elf) -> Segment:
    """The executable PT_LOAD segment (program memory, vaddr 0)."""
    segs = [s for s in elf.segments if s.flags & 1]
    if len(segs) != 1:
        raise CoreError(f"expected one executable segment, found {len(segs)}")
    return segs[0]


def have_peano() -> bool:
    try:
        peano_bin()
        return True
    except CoreError:
        return False
