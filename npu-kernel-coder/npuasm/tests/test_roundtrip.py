import struct
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from npuasm import build, cdo, core, pdi, txn  # noqa: E402

A = (2 << 25) | (3 << 20)  # tile c2r3


def ops_sample():
    return [
        cdo.Op(cdo.NOP, []),
        cdo.Op(cdo.MASK_WRITE64, [0, A | 0x32000, 0x1, 0x0]),
        cdo.Op(cdo.WRITE64, [0, A | 0x1D000, 0x1234]),
        cdo.Op(cdo.DMA_WRITE, [0, A | 0x20000, *range(7)]),
        cdo.Op(cdo.WRITE64, [0, A | 0x1D004, 5]),
        cdo.Op(cdo.DMA_WRITE, [0, A | 0x20100, *range(300)]),  # long form
        cdo.Op(cdo.MASK_POLL64, [0, A | 0x1D200, 0xF, 0x1, 100]),
    ]


def test_cdo_header_and_bytes_roundtrip():
    data = cdo.serialize(ops_sample())
    w = struct.unpack(f"<{len(data) // 4}I", data)
    assert w[1] == cdo.MAGIC and w[3] == len(w) - 5
    assert w[4] == (~(w[0] + w[1] + w[2] + w[3])) & 0xFFFFFFFF
    ops = cdo.parse(data)
    assert cdo.serialize(ops) == data


def test_cdo_dma_payloads_are_aligned():
    data = cdo.serialize(ops_sample())
    off = 20
    for op in cdo.parse(data, keep_nops=True):
        n = len(op.args)
        if op.cmd == cdo.DMA_WRITE:
            pre = 3 if n < 255 else 4
            assert (off + 4 * pre) % 16 == 0
        off += 4 * (1 + n + (1 if n >= 255 else 0))


def test_cdo_text_roundtrip_and_refs():
    ops = ops_sample()
    data = cdo.serialize(ops)
    parsed = cdo.parse(data)
    parsed[3].ref = "core_c2r3"
    text = cdo.to_text(parsed)
    assert "dmawrite c2r3+0x20000 @core_c2r3" in text
    again = cdo.from_text(text, resolve=lambda name: list(range(7)))
    assert cdo.serialize(again) == data


def test_txn_roundtrip():
    h = txn.Header()
    ops = [
        txn.Op("write", {"addr": A | 0x1D000, "value": 0x800}),
        txn.Op("blockwrite", {"addr": A | 0x1D000, "col": 2, "row": 3}, [1, 2, 3, 4, 5, 6, 7, 8, 9]),
        txn.Op("blockwrite", {"addr": 0x1D000, "col": 1, "row": 0}, [7]),  # tile field differs from addr
        txn.Op("maskwrite", {"addr": A | 0x1D200, "value": 1, "mask": 3}),
        txn.Op("maskpoll", {"addr": A | 0x1D200, "value": 0, "mask": 1}),
        txn.Op("sync", {"col": 0, "row": 0, "dir": 0, "chan": 0, "ncol": 1, "nrow": 1}),
        txn.Op("ddr_patch", {"addr": 0x1D004, "arg": 1, "plus": 0x80000040}),
        txn.Op("preempt", {"level": 2}),
    ]
    data = txn.serialize(h, ops)
    h2, ops2 = txn.parse(data)
    assert txn.serialize(h2, ops2) == data
    assert txn.serialize(*txn.from_text(txn.to_text(h2, ops2))) == data


def test_pdi_build_layout():
    parts = [cdo.serialize(ops_sample()), cdo.serialize(ops_sample()[:3]), cdo.serialize([])]
    p = pdi.build(pdi.merge(parts), revoke_id=0x5581)
    w = lambda o: struct.unpack_from("<I", p, o)[0]  # noqa: E731
    assert w(0) == 0xDD and w(0x28) == 0x14CA8093
    for start, end in ((0x10, 0x8C), (0x90, 0xCC), (0xD0, 0x14C)):
        assert w(end) == (~sum(w(o) for o in range(start, end, 4))) & 0xFFFFFFFF
    assert pdi.revoke_id(p) == 0x5581
    assert (len(p) - 0x150) % 16 == 0 and w(0xD0) * 4 == len(p) - 0x150


@pytest.mark.skipif(not core.have_peano(), reason="needs Peano (NPUASM_PEANO)")
def test_core_assemble_matches_known_bytes():
    asm = "\t.text\n\tpaddxm [sp], #0x80\n\tst r11, [sp, #-56]; nopx\n"
    out = core.assemble(asm)
    assert out.startswith(bytes.fromhex("c40100001000" "5c0000b02ef9"))


SAMPLE = Path.home() / "code/openflowlm-next/open_kernels/designs/layer_x/build_qwen35_lx_h2048_ef57108f/final.prj"


@pytest.mark.skipif(not (SAMPLE.exists() and core.have_peano()), reason="needs the local build corpus and Peano")
def test_real_build_roundtrip():
    rep = build.check(SAMPLE)
    assert rep.error is None
    assert rep.ok and all(rep.ok.values()), rep.ok


@pytest.mark.skipif(core.core_bin() is None, reason="needs rust/target/release/npuasm-core")
def test_core_backend_assembles_peano_gaps(monkeypatch):
    # The three instructions Peano's assembler rejects, plus a vldb.fill bundle
    # whose FIFO and position registers have no encoding field.
    monkeypatch.setenv("NPUASM_BACKEND", "core")
    asm = (
        "\t.text\n"
        "\tst p2, [sp, #-60]; vinsert.32 x6, x1, #0, r3\n"
        "\tvlda bmhh4, [sp, #-0x80]; vldb.fill.512 [p1, lf1, r25]; movs p0, r21; mov r24, r19\n"
    )
    out = core.assemble(asm)
    assert out == bytes.fromhex("02b0388401b0a3f8" "f670d00c030b952828ba4dff")


# ---------------------------------------------------------------- rule gate

from npuasm import rules  # noqa: E402
from npuasm.core import Insn  # noqa: E402


def _prog(texts, size=16):
    return [Insn(size * i, b"\0" * size, t) for i, t in enumerate(texts)]


NOP5 = ["nop"] * 5


def test_rules_accept_clean_program():
    p = _prog(["mova r0, #0x0", "jl #0x80", *NOP5, "acq #0x30, r1", "nop", "nop", "nop", "rel #0x31, r1"])
    assert rules.check_core(p) == []


def test_rules_catch_each_core_violation():
    def hit(p):
        return {v.rule for v in rules.check_core(p)}

    assert "size" in hit(_prog(["nop"] * 1025))  # 1025 x 16 B > 16 KiB
    # target 0x18 is mid-line after a real instruction
    p = [Insn(0, b"\0" * 8, "jl #0x18"), Insn(8, b"\0" * 8, "mova r0, #0x1"), Insn(16, b"\0" * 8, "mova r1, #0x2"), Insn(24, b"\0" * 8, "nop")]
    p += [Insn(32 + 8 * i, b"\0" * 8, "nop") for i in range(5)]
    assert "branch-target" in hit(p)
    assert "delay-slots" in hit(_prog(["jl #0x0", "jl #0x0", *NOP5]))  # branch in a delay slot
    assert "delay-slots" in hit(_prog(["jl #0x0", "nop", "done", "nop", "nop", "nop"]))
    assert "delay-slots" in hit(_prog(["jl #0x0", "nop"]))  # too few bundles after
    assert "lock-spacing" in hit(_prog(["acq #0x30, r1", "nop", "rel #0x31, r1"]))


def test_rules_allow_peano_hardware_loop_and_locks_in_delay_slots():
    p = _prog(["movxm ls, #0x40", "movxm le, #0x40", "add.nc lc, r2, #-0x1", "nop", "nop", "jl #0x0", "acq #0x30, r1", *NOP5])
    assert rules.check_core(p) == []


def test_rules_catch_bd_violations():
    core_bd = (2 << 25) | (3 << 20) | 0x1D000
    valid = 1 << 25  # DMA_BD0_5 VALID_BD
    v = rules.check_bd(core_bd, [0, 0, 0, 0, 0, valid], "t")
    assert [x.rule for x in v] == ["zero-length-bd"]
    assert rules.check_bd(core_bd, [16, 0, 0, 0, 0, valid], "t") == []
    shim_bd = 0x1D000
    axqos15 = 15 << 20  # DMA_BD0_5 AXQOS, bits 23:20
    v = rules.check_bd(shim_bd, [64, 0, 0, 0, 0, axqos15, 0, 1 << 25], "t")
    assert "shim-axqos-15" in [x.rule for x in v]


# ------------------------------------------------------------------ xclbin

from npuasm import xclbin  # noqa: E402

PROBE = Path(__file__).resolve().parents[1] / "silicon/zol/build/final.xclbin"


@pytest.mark.skipif(not PROBE.exists(), reason="needs the zol probe build")
def test_xclbin_roundtrip_and_build():
    data = PROBE.read_bytes()
    x = xclbin.parse(data)
    assert xclbin.serialize(x) == data
    part = next(s for s in x.sections if s.kind == xclbin.AIE_PARTITION)
    p = xclbin.parse_partition(part.data)
    assert xclbin.serialize_partition(p) == part.data
    new = xclbin.build(data, p.pdis[0].image, pdi_uuid=b"\1" * 16, xclbin_uuid=b"\2" * 16, timestamp=0)
    y = xclbin.parse(new)
    assert y.tail == b"" and y.uuid == b"\2" * 16
    q = xclbin.parse_partition(next(s for s in y.sections if s.kind == xclbin.AIE_PARTITION).data)
    assert q.pdis[0].uuid == b"\1" * 16 and q.pdis[0].image == p.pdis[0].image
