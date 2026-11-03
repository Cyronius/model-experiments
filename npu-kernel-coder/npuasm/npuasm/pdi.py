"""PDI writer for the one-image, one-CDO-partition PDIs that mlir-aie's
aiecc builds with bootgen (`bif_main.bif`: id_code 0x14ca8093,
extended_id_code 0x01, image `aie_image` id 0x1c000000, three CDO files).

Layout (bootgen `versal/include/imageheadertable-versal.h`): 16 B of SMAP
sync words, the image header table at 0x10, the image header at 0x90, the
partition header at 0xd0, and the merged CDO at 0x150. Each header ends in
`~sum(words)`.

bootgen merges the three CDO files part by part: it copies every command (NOPs
included), narrows 64-bit commands whose high address word is 0, and adds a
NOP before each DMA write whose payload would start unaligned *within that
part file* (offsets counted from the part's own 20-byte header, not from the
merged stream; `cdo.BOOTGEN_MERGED`). The partition is then zero-padded to a
16-byte multiple; the partition header gives both the padded and unpadded
lengths.

The image header's `metaHdrRevokeId` word (file offset 0x98) is the one field
that varies between otherwise identical builds. Upstream bootgen now zeroes
it (`common/src/bifoptions.cpp`); the copy bundled with this mlir-aie seems
to leave it unset, so it holds leftover memory. We carry it through to
reproduce a vendor PDI exactly and write 0 otherwise.
"""

from __future__ import annotations

import struct

from . import cdo

SMAP = [0x000000DD, 0x11223344, 0x55667788, 0x99AABBCC]
IHT_OFF, IH_OFF, PH_OFF, DATA_OFF = 0x10, 0x90, 0xD0, 0x150


class PdiError(ValueError):
    pass


def _csum(words) -> int:
    return (~sum(words)) & 0xFFFFFFFF


def _name_words(name: str) -> list[int]:
    b = name.encode().ljust(16, b"\0")
    if len(b) > 16:
        raise PdiError("image name longer than 16 bytes")
    return list(struct.unpack("<4I", b))


def build(merged_body: list[int], *, revoke_id: int = 0, id_code: int = 0x14CA8093,
          extended_id_code: int = 1, image_name: str = "aie_image", image_id: int = 0x1C000000) -> bytes:
    data = cdo.header(len(merged_body)) + merged_body
    data_words = len(data)
    data += [0] * (-len(data) % 4)
    part_words = len(data)
    iht = [0x00040000, 1, IH_OFF // 4, 1, PH_OFF // 4, 0, id_code, 0, 0, 0, 0x50504449, 0x00201020, 0x30,
           0, 0, 0, 0, extended_id_code] + [0] * 13
    iht.append(_csum(iht))
    ih = [PH_OFF // 4, 1, revoke_id, 0, *_name_words(image_name), image_id, 0, 0, 0, 0, 0, 0xFFFFFFFF]
    ih.append(_csum(ih))
    ph = [part_words, data_words, part_words, 0, 0, 0, 0xFFFFFFFF, 0xFFFFFFFF, DATA_OFF // 4, 0x02000006, 1] + [0] * 20
    ph.append(_csum(ph))
    words = SMAP + iht + ih + ph + data
    assert len(SMAP + iht + ih + ph) * 4 == DATA_OFF
    return struct.pack(f"<{len(words)}I", *words)


def merge(parts: list[bytes]) -> list[int]:
    """Merged CDO body, as bootgen writes it, from the part files' bytes."""
    body: list[int] = []
    for p in parts:
        body += cdo.serialize_body(cdo.parse(p, keep_nops=True), cdo.BOOTGEN_MERGED)
    return body


def revoke_id(pdi: bytes) -> int:
    return struct.unpack_from("<I", pdi, IH_OFF + 8)[0]


def body(pdi: bytes) -> bytes:
    """The merged CDO (with its header) inside a PDI built by `build`."""
    if len(pdi) < DATA_OFF + 20 or struct.unpack_from("<I", pdi, DATA_OFF + 4)[0] != cdo.MAGIC:
        raise PdiError("no CDO at 0x150")
    return pdi[DATA_OFF:]
