"""xclbin container and its AIE_PARTITION section, read and written exactly.

Layout from XRT's `xclbin.h` (vendored in mlir-aie's hrx-xclbinutil):
`axlf` header (magic, signature length, reserved, key block, unique id,
`axlf_header` at 304, section headers at 456, 40 bytes each), then the
sections back to back on 8-byte boundaries. xclbinutil appends a JSON copy
of the metadata between `XCLBIN_MIRROR_DATA_START` and `..._END`; it is kept
as opaque bytes here, so a vendor xclbin rebuilds exactly.

AIE_PARTITION (kind 32): a 184-byte `aie_partition`, then, each padded to
8 bytes: its name and kernel-commit strings, the start-column array, and
per PDI the image, each CDO group's name, DPU kernel ids and pre-group ids,
the CDO group structs, and last the `aie_pdi` structs.

`build` makes a new xclbin for npuasm output: the other six sections come
from a template xclbin (all 1,873 local aiecc xclbins share their layout,
with at most three variants of each section), the partition is rebuilt
around the new PDI under fresh UUIDs, and the JSON copy is dropped.
"""

from __future__ import annotations

import struct
import time
import uuid as uuidlib
from dataclasses import dataclass, field

MAGIC = b"xclbin2\0"
HDR = 304
SECTIONS = 456
AIE_PARTITION = 32
MIRROR = b"XCLBIN_MIRROR_DATA_START"


class XclbinError(ValueError):
    pass


def _align(n: int, a: int = 8) -> int:
    return (n + a - 1) // a * a


@dataclass
class Section:
    kind: int
    name: bytes  # 16 bytes as stored
    data: bytes


@dataclass
class Xclbin:
    head: bytes  # bytes 0..HDR-1 (magic .. unique id)
    header: bytes  # axlf_header without m_length and m_numSections: bytes 312..448
    sections: list[Section]
    tail: bytes = b""  # the JSON copy, if any

    @property
    def uuid(self) -> bytes:
        return self.header[416 - 312 : 432 - 312]


def parse(data: bytes) -> Xclbin:
    if data[:8] != MAGIC:
        raise XclbinError("not an xclbin2 file")
    (length,) = struct.unpack_from("<Q", data, HDR)
    if length != len(data):
        raise XclbinError(f"m_length {length} != file size {len(data)}")
    (n,) = struct.unpack_from("<I", data, 448)
    if data[452:456] != b"\0\0\0\0":
        raise XclbinError("header padding not zero")
    secs, pos = [], SECTIONS + 40 * n
    for i in range(n):
        h = SECTIONS + 40 * i
        (kind,) = struct.unpack_from("<I", data, h)
        name = data[h + 4 : h + 20]
        if data[h + 20 : h + 24] != b"\0\0\0\0":
            raise XclbinError("section header padding not zero")
        off, size = struct.unpack_from("<QQ", data, h + 24)
        if off != _align(pos):
            raise XclbinError(f"section {i} at {off}, expected {_align(pos)}")
        if any(data[pos:off]):
            raise XclbinError("nonzero bytes between sections")
        secs.append(Section(kind, name, data[off : off + size]))
        pos = off + size
    return Xclbin(data[:HDR], data[312:448], secs, data[pos:])


def serialize(x: Xclbin) -> bytes:
    n = len(x.sections)
    pos = SECTIONS + 40 * n
    table, body = bytearray(), bytearray()
    start = pos
    for s in x.sections:
        off = _align(pos)
        body += b"\0" * (off - pos) + s.data
        table += struct.pack("<I", s.kind) + s.name.ljust(16, b"\0")[:16] + b"\0" * 4 + struct.pack("<QQ", off, len(s.data))
        pos = off + len(s.data)
    total = pos + len(x.tail)
    out = bytearray(x.head) + struct.pack("<Q", total) + x.header + struct.pack("<I", n) + b"\0" * 4
    assert len(out) == SECTIONS
    out += table
    assert len(out) == start
    out += body + x.tail
    return bytes(out)


# ------------------------------------------------------------ AIE_PARTITION


@dataclass
class CdoGroup:
    name: str
    cdo_type: int
    pdi_id: int
    dpu_kernel_ids: list[int]
    pre_cdo_groups: list[int]


@dataclass
class Pdi:
    uuid: bytes
    image: bytes
    cdo_groups: list[CdoGroup]


@dataclass
class Partition:
    name: str
    kernel_commit_id: str
    operations_per_cycle: int
    inference_fingerprint: int
    pre_post_fingerprint: int
    column_width: int
    start_columns: list[int]
    pdis: list[Pdi] = field(default_factory=list)


def _cstr(b: bytes, off: int) -> str:
    return b[off : b.index(b"\0", off)].decode()


def parse_partition(sec: bytes) -> Partition:
    if sec[0] != 0 or any(sec[1:4]) or any(sec[12:16]):
        raise XclbinError("unexpected partition schema or padding")
    mpo, ops = struct.unpack_from("<II", sec, 4)
    inf, pp = struct.unpack_from("<QQ", sec, 16)
    (cw,) = struct.unpack_from("<H", sec, 32)
    sc_n, sc_o = struct.unpack_from("<II", sec, 40)
    pdi_n, pdi_o = struct.unpack_from("<II", sec, 120)
    (kcid,) = struct.unpack_from("<I", sec, 128)
    p = Partition(
        _cstr(sec, mpo), _cstr(sec, kcid), ops, inf, pp, cw,
        [struct.unpack_from("<H", sec, sc_o + 2 * i)[0] for i in range(sc_n)],
    )
    for i in range(pdi_n):
        b = pdi_o + 96 * i
        im_n, im_o, cg_n, cg_o = struct.unpack_from("<IIII", sec, b + 16)
        groups = []
        for g in range(cg_n):
            gb = cg_o + 96 * g
            nm, ct = struct.unpack_from("<IB", sec, gb)
            (pid,) = struct.unpack_from("<Q", sec, gb + 8)
            dk_n, dk_o, pc_n, pc_o = struct.unpack_from("<IIII", sec, gb + 16)
            groups.append(CdoGroup(
                _cstr(sec, nm), ct, pid,
                [struct.unpack_from("<Q", sec, dk_o + 8 * k)[0] for k in range(dk_n)],
                [struct.unpack_from("<I", sec, pc_o + 4 * k)[0] for k in range(pc_n)],
            ))
        p.pdis.append(Pdi(sec[b : b + 16], sec[im_o : im_o + im_n], groups))
    return p


def serialize_partition(p: Partition) -> bytes:
    buf = bytearray(184)

    def put(data: bytes) -> int:
        nonlocal buf
        off = len(buf)
        buf += data + b"\0" * (_align(len(data)) - len(data))
        return off

    name_o = put(p.name.encode() + b"\0")
    kcid_o = put(p.kernel_commit_id.encode() + b"\0")
    sc_o = put(b"".join(struct.pack("<H", c) for c in p.start_columns))
    pdi_meta = []
    for pdi in p.pdis:
        im_o = put(pdi.image)
        gmeta = []
        for g in pdi.cdo_groups:
            nm_o = put(g.name.encode() + b"\0")
            dk_o = put(b"".join(struct.pack("<Q", k) for k in g.dpu_kernel_ids))
            pc_o = put(b"".join(struct.pack("<I", k) for k in g.pre_cdo_groups))
            gmeta.append((nm_o, dk_o, pc_o))
        cg_o = put(b"".join(
            struct.pack("<IB3xQIIII", nm_o, g.cdo_type, g.pdi_id, len(g.dpu_kernel_ids), dk_o, len(g.pre_cdo_groups), pc_o) + b"\0" * 64
            for g, (nm_o, dk_o, pc_o) in zip(pdi.cdo_groups, gmeta)
        ))
        pdi_meta.append((im_o, cg_o))
    pdi_o = put(b"".join(
        pdi.uuid + struct.pack("<IIII", len(pdi.image), im_o, len(pdi.cdo_groups), cg_o) + b"\0" * 64
        for pdi, (im_o, cg_o) in zip(p.pdis, pdi_meta)
    ))
    head = struct.pack("<B3xII4xQQ", 0, name_o, p.operations_per_cycle, p.inference_fingerprint, p.pre_post_fingerprint)
    info = struct.pack("<H6xII", p.column_width, len(p.start_columns), sc_o) + b"\0" * 72
    tailf = struct.pack("<III", len(p.pdis), pdi_o, kcid_o) + b"\0" * 52
    struct_bytes = head + info + tailf
    assert len(struct_bytes) == 184, len(struct_bytes)
    buf[:184] = struct_bytes
    return bytes(buf)


def build(template: bytes, pdi_image: bytes, *, pdi_uuid: bytes | None = None, xclbin_uuid: bytes | None = None,
          timestamp: int | None = None, keep_mirror: bool = False) -> bytes:
    """A new xclbin: the template's other sections, a partition rebuilt
    around `pdi_image`, and fresh UUIDs and timestamp unless given."""
    x = parse(template)
    secs = []
    for s in x.sections:
        if s.kind == AIE_PARTITION:
            p = parse_partition(s.data)
            if len(p.pdis) != 1:
                raise XclbinError("template partition must hold one PDI")
            p.pdis[0] = Pdi(pdi_uuid or uuidlib.uuid4().bytes, pdi_image, p.pdis[0].cdo_groups)
            s = Section(s.kind, s.name, serialize_partition(p))
        secs.append(s)
    hdr = bytearray(x.header)
    struct.pack_into("<Q", hdr, 0, int(time.time()) if timestamp is None else timestamp)  # m_timeStamp
    hdr[416 - 312 : 432 - 312] = xclbin_uuid or uuidlib.uuid4().bytes
    return serialize(Xclbin(x.head, bytes(hdr), secs, x.tail if keep_mirror else b""))
