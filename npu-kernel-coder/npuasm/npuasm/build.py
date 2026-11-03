"""Lift a whole aiecc build (final.prj + insts.bin) to text, rebuild it,
and compare every artifact byte for byte.

Lifted directory:
    manifest.json        sources, hashes, PDI revoke word, core table
    core_cCrR.s          one per distinct core program (identical programs
                         share one file; manifest maps every core to it)
    elfs.cdo, init.cdo, enable.cdo
                         declarative CDO text (decl.py): aie-rt register and
                         field names; a program-memory write whose payload is
                         a core's code segment becomes `dmawrite ... @core_cCrR`
    insts.txn            transaction stream text (if insts.bin exists)

Rebuild order: assemble cores -> expand CDO references -> write the three
CDO files -> merge into the PDI -> write insts.bin.
"""

from __future__ import annotations

import hashlib
import json
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path

from . import cdo, core, decl, pdi, txn

CORE_DIR = re.compile(r"^elfs_(.+)_core_(\d+)_(\d+)$")


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]


@dataclass
class BuildFiles:
    prj: Path
    device: str
    pdi: Path
    cdos: dict[str, Path]  # elfs/init/enable
    cores: dict[tuple[int, int], Path]
    insts: Path | None
    xclbin: Path | None


def find(prj: Path) -> BuildFiles:
    pdis = sorted(prj.glob("*.pdi"))
    if len(pdis) != 1:
        raise FileNotFoundError(f"{prj}: expected one .pdi, found {len(pdis)}")
    device = pdis[0].stem
    cdos = {k: prj / f"cdo_{device}" / f"{device}_aie_cdo_{k}.bin" for k in ("elfs", "init", "enable")}
    for p in cdos.values():
        if not p.exists():
            raise FileNotFoundError(p)
    cores = {}
    for d in prj.iterdir():
        m = CORE_DIR.match(d.name)
        if m and m.group(1) == device and d.is_dir():
            elfs = list(d.glob("*.elf"))
            if len(elfs) == 1:
                cores[(int(m.group(2)), int(m.group(3)))] = elfs[0]
    parent = prj.parent
    insts = parent / "insts.bin"
    xcl = sorted(parent.glob("*.xclbin"))
    return BuildFiles(prj, device, pdis[0], cdos, cores, insts if insts.exists() else None, xcl[0] if xcl else None)


@dataclass
class Report:
    prj: str
    ok: dict[str, bool] = field(default_factory=dict)
    sizes: dict[str, int] = field(default_factory=dict)  # bytes of each original artifact
    text_chars: dict[str, int] = field(default_factory=dict)
    cores: int = 0
    distinct_cores: int = 0
    bundles: int = 0
    raw_bundles: int = 0
    asm_errors: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    error: str | None = None


def lift(prj: Path, out: Path | None, rep: Report | None = None) -> tuple[dict, dict[str, str], Report]:
    """Returns (manifest, {filename: text}, report). Writes files if `out`."""
    f = find(prj)
    rep = rep or Report(str(prj))
    texts: dict[str, str] = {}
    manifest: dict = {"device": f.device, "source": str(prj), "cores": {}, "hashes": {}}

    # Cores: lift, assemble (with .byte fallback), dedupe by code bytes.
    by_code: dict[bytes, str] = {}
    code_words: dict[str, list[int]] = {}
    for (c, r), elf_path in sorted(f.cores.items()):
        elf = core.read_elf(elf_path.read_bytes())
        seg = core.code_segment(elf)
        if seg.vaddr != 0:
            raise core.CoreError(f"{elf_path}: code segment at {seg.vaddr:#x}")
        rep.cores += 1
        rep.sizes["core_code"] = rep.sizes.get("core_code", 0) + len(seg.data)
        name = by_code.get(seg.data)
        if name is None:
            name = f"core_c{c}r{r}"
            by_code[seg.data] = name
            lst = core.lift(elf_path)
            out_bytes, raw, seen = core.assemble_listing(lst)
            ok = out_bytes[: len(seg.data)] == seg.data and not any(out_bytes[len(seg.data):])
            rep.ok[f"{name}.s"] = ok
            rep.bundles += len(lst.insns)
            rep.raw_bundles += len(raw)
            for slot, e in seen:
                key = f"{e}: {slot.split()[0] if slot else '?'}"
                rep.asm_errors[key] = rep.asm_errors.get(key, 0) + 1
            texts[f"{name}.s"] = lst.render(raw, labels=core.backend() == "core")
            rep.distinct_cores += 1
            if len(seg.data) % 4:
                raise core.CoreError(f"{elf_path}: code size not a word multiple")
            code_words[name] = list(struct.unpack(f"<{len(seg.data) // 4}I", seg.data))
        manifest["cores"][f"c{c}r{r}"] = name

    # CDOs: text, with program-memory payloads replaced by core references.
    words_to_core = {tuple(w): n for n, w in code_words.items()}
    for part, path in f.cdos.items():
        data = path.read_bytes()
        rep.sizes[f"cdo_{part}"] = len(data)
        ops = cdo.parse(data)
        for op in ops:
            if op.cmd == cdo.DMA_WRITE and (op.args[1] & 0xFFFFF) == 0x20000:
                n = words_to_core.get(tuple(op.args[2:]))
                if n:
                    op.ref = n
        texts[f"{part}.cdo"] = decl.to_text(ops)
        manifest["hashes"][f"cdo_{part}"] = sha(data)

    pdata = f.pdi.read_bytes()
    rep.sizes["pdi"] = len(pdata)
    manifest["pdi_revoke_word"] = pdi.revoke_id(pdata)
    manifest["hashes"]["pdi"] = sha(pdata)

    if f.insts:
        ib = f.insts.read_bytes()
        rep.sizes["insts"] = len(ib)
        h, ops = txn.parse(ib)
        texts["insts.txn"] = txn.to_text(h, ops)
        manifest["hashes"]["insts"] = sha(ib)
    if f.xclbin:
        xb = f.xclbin.read_bytes()
        manifest["xclbin_embeds_pdi"] = pdata in xb
        rep.ok["xclbin_has_pdi"] = pdata in xb

    for k, v in texts.items():
        rep.text_chars[k] = len(v)
    if out:
        out.mkdir(parents=True, exist_ok=True)
        for k, v in texts.items():
            (out / k).write_text(v)
        (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest, texts, rep


def rebuild(manifest: dict, texts: dict[str, str]) -> dict[str, bytes]:
    """Artifacts from lifted text: cdo_elfs/init/enable, pdi, insts."""
    out: dict[str, bytes] = {}
    code: dict[str, list[int]] = {}
    for name in sorted(set(manifest["cores"].values())):
        lst_text = texts[f"{name}.s"]
        b = core.assemble(lst_text)
        code[name] = list(struct.unpack(f"<{len(b) // 4}I", b))

    def resolve(ref: str) -> list[int]:
        return code[ref]

    parts = []
    for part in ("elfs", "init", "enable"):
        ops = decl.from_text(texts[f"{part}.cdo"], resolve=resolve)
        b = cdo.serialize(ops)
        out[f"cdo_{part}"] = b
        parts.append(b)
    out["pdi"] = pdi.build(pdi.merge(parts), revoke_id=manifest["pdi_revoke_word"])
    if "insts.txn" in texts:
        h, ops = txn.from_text(texts["insts.txn"])
        out["insts"] = txn.serialize(h, ops)
    return out


def check(prj: Path) -> Report:
    rep = Report(str(prj))
    try:
        manifest, texts, rep = lift(prj, None, rep)
        built = rebuild(manifest, texts)
        f = find(prj)
        rep.ok["cdo_elfs"] = built["cdo_elfs"] == f.cdos["elfs"].read_bytes()
        rep.ok["cdo_init"] = built["cdo_init"] == f.cdos["init"].read_bytes()
        rep.ok["cdo_enable"] = built["cdo_enable"] == f.cdos["enable"].read_bytes()
        rep.ok["pdi"] = built["pdi"] == f.pdi.read_bytes()
        if f.insts:
            rep.ok["insts"] = built["insts"] == f.insts.read_bytes()
    except Exception as e:  # report and keep going over the corpus
        rep.error = f"{type(e).__name__}: {str(e)[:300]}"
    return rep
