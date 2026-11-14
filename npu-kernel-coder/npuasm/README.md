# npuasm: a lossless text form for XDNA2 build artifacts

npuasm turns a finished aiecc build (`final.prj/` plus `insts.bin`) into text, then rebuilds the binaries from that text. It covers the first four gates in [the revised plan](../plan.md#revision-2026-11-03-direct-emission):
- **A:** a text form that loses nothing;
- **B:** rebuilt to the exact bytes without the vendor toolchain;
- **C:** compact enough to read and write;
- **D:** a run path for edited programs, with a rule gate before anything reaches the NPU.

**Result, 2026-11-03: all four pass on every build on this machine.** All 1,873 builds rebuild from text with every artifact byte-identical. Only `npuasm-core` and Python run; there is no Peano, bootgen, mlir-aie or XRT in the loop. A median build is 57,818 Qwen tokens of text, down from 127,406 for the first raw text. Details are in [Results](#results).

## What it covers

| Artifact | Text form | Rebuilt by |
|---|---|---|
| Core programs (`elfs_*_core_C_R/*.elf`, code segment) | `core_cCrR.s`: AIE2P assembly in llvm-objdump syntax, one bundle per line, with labels for branch and hardware-loop targets. Each distinct program is stored once. A near-copy of an earlier program is a patch on it (`.base core_cXrY` + line edits, `patch.py`). | [`rust/`](rust/) `npuasm-core asm`: our text parser over hipfire's encoder tables, two-pass for labels |
| CDOs (`*_aie_cdo_{elfs,init,enable}.bin`) | `elfs.cdo`, `init.cdo`, `enable.cdo` in the declarative form (`decl.py`, below). A program-memory write becomes `dmawrite c0r2+0x20000 @core_c0r2`. | `npuasm.decl` + `npuasm.cdo` (Python, no dependencies) |
| PDI (`*.pdi`) | Nothing extra: built from the three CDOs, plus one junk word kept in `manifest.json` | `npuasm.pdi` |
| Transaction stream (`insts.bin`) | `insts.txn` in the same declarative form (`txndecl.py`) | `npuasm.txndecl` + `npuasm.txn` |

### The declarative form

Register writes use aie-rt's register and field names (`xaie2pgbl_params.h`, via `regs.py`). Only non-zero fields are printed. Four more lossless steps:
- **Tile context:** an `@cCrR` line sets the tile for the lines after it.
- **Lock runs:** `locks 0: 0*14,2,1` writes `LOCK0_VALUE` onward in order.
- **Segment reuse:** `@c1r2 =c0r2` replays the last segment written for c0r2.
- **Loops** (`loops.py`): an unrolled run of 3 or more copies whose varying numbers step by a constant folds into `repeat N i { … {start+step i} … }`, nesting as `i`, `j`, `k`.

Anything that can't be named keeps the raw line form, which the parsers still read. Data payloads print runs of 3 or more equal words as `00000000*12`.

```
@c0r0
block DMA_BD0_0 [BUFFER_LENGTH=0x800] [] [] [] [BURST_LENGTH=3] [AXCACHE=2] [] [VALID_BD=1]
ddr_patch DMA_BD0_1 arg=1 plus=0x0
repeat 8 i {
@c{0+1i}r0
write DMA_MM2S_0_TASK_QUEUE ENABLE_TOKEN_ISSUE=1 START_BD_ID={0+1i}
}
```
| xclbin | Not part of the text. `xclbin.py` parses and rewrites it exactly (all 1,873 local xclbins). `package` builds a new one around a PDI from a template's other sections, with fresh UUIDs and without xclbinutil's JSON copy. | `npuasm.xclbin` |

Lifting a core uses `npuasm-core lift`, hipfire's decoder plus one extension. Its text matches llvm-objdump's exactly on all 5.5M bundles of the 2,267 distinct programs here. Peano is kept as an oracle: `NPUASM_BACKEND=peano` lifts with its `llvm-objdump` and assembles with its `clang -c`, falling back to `.byte` for the instructions that assembler rejects.

## Run

```sh
# From this directory. Build the core tool once (needs ~/code/hipfire, see rust/README.md):
(cd rust && cargo build --release)

python3 -m npuasm check  <build>/final.prj          # round-trip one build, JSON report
python3 -m npuasm lift   <build>/final.prj <dir>    # write the text form
python3 -m npuasm build  <dir> <out>                # rebuild the binaries from text
python3 -m npuasm corpus ~/code --out results/corpus.jsonl --jobs 12

python3 -m npuasm rules  <dir>                      # rule gate on lifted/edited text; exit 1 on violations
python3 -m npuasm package <pdi> <template.xclbin> <out.xclbin>   # xclbin without xclbinutil
rust/target/release/npuasm-core check CORE.elf...   # decode -> text -> assemble, per bundle
rust/target/release/npuasm-core rules CORE.elf...   # hipfire's silicon rules, first violation per program

../../target/manifold-audit/env/bin/python -m pytest -q tests
../../target/manifold-audit/env/bin/python scripts/count_tokens.py \
    results/corpus.jsonl ../../models/manifold-audit/Qwen3.5-9B/tokenizer.json --out results/tokens.md
```

## Results

Measured on 2026-11-03 on the Ryzen AI MAX+ 395 box. The builds came from whatever toolchain each project used when it built them.

**Gate D: run path and rule gate.**
- **Run path:** `silicon/zol/edit_build.sh` edits a core program as text, runs the rule gate, rebuilds with npuasm, and packages with `npuasm package`. No vendor tool is involved; running still uses XRT's runtime. The multiplier-7 edit gave 1,024/1,024 correct words in 3 runs ([silicon/zol](silicon/zol/README.md)). XRT accepts an xclbin without xclbinutil's JSON copy.
- **Rule gate:** `rules.py` checks program size, branch-target alignment, delay slots, lock spacing, zero-length valid descriptors and shim AxQoS 15. Over all 1,873 builds it finds no violations, and each rule fires on its unit-test case.
- **Calibration:** the corpus showed that hipfire's ban on lock requests in delay slots was too strict. Peano puts `acq`/`rel` in call delay slots in every build, so that part was dropped.

**Gate C: size in tokens** ([tokens.md](results/tokens.md), the first raw text in [tokens-v1.md](results/tokens-v1.md); Qwen3.5 tokenizer, the same 248k vocabulary as Qwen3.8; 118 design families, one build each, the one with the most core code):

| Part | Median | p90 | Max | Total change |
|---|---:|---:|---:|---:|
| Core assembly | 17,014 → 16,832 | 316,000 → 82,060 | 1,651,891 → 188,888 | −72.9% |
| CDOs | 73,914 → 16,359 | 82,994 → 25,716 | 349,954 → 25,799 | −75.9% |
| `insts.txn` | 20,547 → 8,662 | 108,821 → 85,120 | 899,296 → 221,076 | −48.6% |
| Whole build | 127,406 → 57,818 | 490,286 → 154,700 | 2,019,124 → 281,565 | −68.0% |

Where the assembly savings came from, and what's left:
- **Patches:** most of the cut is near-copy programs written as patches. The 32 `trit_embed` programs differ in about four lines each (the core's own function label, one constant, one jump), so 1.66M tokens become 77k.
- **What's left:** NOPs are 7.7% of assembly tokens, comments 2.3% and labels 1.3%; the rest is real instructions, about 3.3 tokens per code byte.
- **Not used:** loop folding would save only 5% on the assembly, so it isn't applied there.

Getting much smaller than this needs a higher layer that isn't lossless. For example, the model could write a runtime sequence as loops in its own terms, or only the parts of a build that change.

**Gate B: no vendor toolchain** ([corpus.md](results/corpus.md); per-build rows in `results/corpus.jsonl`, untracked and regenerated by `corpus`):

- 1,863 builds then (1,873 now, with the gate C form). Every artifact is identical in all of them: 5,919 programs (each build's distinct cores, summed), all three CDOs, the PDI and `insts.bin`. No bundle needs `.byte`.
- 426 CPU-seconds in total, a median of 0.061 s per build. A subprocess log of one build shows `npuasm-core` as the only program run.
- `npuasm-core check` over the 2,267 programs that are distinct across the corpus (5,501,116 bundles): every bundle decodes, re-encodes and assembles from its text to its exact bytes. No bundle text matched more than one encoding.
- Lifted text equals Peano llvm-objdump's text for every one of those bundles.

**Gate A: lossless text, with Peano's assembler** (the first run): 1,849 builds, every artifact identical. Peano llvm-aie 21.0.0.2026080301+c9c5ecb7 rejects 38,176 of 11,828,654 bundles (0.32%), which it carried as `.byte`: `vinsert.32` (33,284), `event` (4,886) and `vinsert.16` (6). No operand spelling assembles. In hipfire's tables, `vinsert.32 $dst, $s1, #0, $src` has a literal `#0` in its syntax rather than an operand; npuasm-core assembles all three.

## What the round trip showed about the formats

- **CDO NOPs** (bootgen `cdo-driver`): a NOP right before a DMA write is padding that the writer recreates, so the text leaves it out. Every other NOP is kept as an op. For example, each cdo-driver file starts with one.
- **Merging into the PDI**: bootgen does more than concatenate. For each part file it copies every command, NOPs included, rewrites 64-bit commands with a zero high address word into the 32-bit forms, and adds a NOP before any DMA write whose payload would start unaligned *within that part file*. Alignment is counted from the part's own header, not from the merged stream. The partition is then zero-padded to a 16-byte multiple, and the partition header records both lengths.
- **Image header word at offset 0x98** (`metaHdrRevokeId`): this is the only header word that varies between otherwise identical builds. It is zero in 1,275 PDIs and a pointer-like value (`0x5581`, `0x7fda`…) in 574, mixed within every project. Upstream bootgen now initializes it to 0 (`common/src/bifoptions.cpp`). The bundled copy seemingly doesn't, so the word holds leftover memory (inferred). npuasm keeps it so vendor PDIs reproduce exactly, and writes 0 otherwise.
- **The `elfs` CDO** is aie-rt's ELF loader output: a reset sequence for each core, the code segment written to `0x20000`, and data segments written to data memory.
- **`vldb.fill.512` / `vldb.fillx.512`** print `[pN, lfN, r2N]` but encode only the pointer bit. The FIFO and position registers have no field, so hipfire's decoder rejects these bundles: 230 bundles in 19 programs. In npuasm-core's fallback decoder, an unfielded register takes the same index as a fielded operand whose register class has the same size. The result matches llvm-objdump on all 230.
- **Hardware loops:** 2,120 of the 2,267 distinct programs set up zero-overhead loops (`movxm ls`, `movxm le`, `add.nc lc`; 35,706 setups). hipfire's rules reject any hardware loop, because in hipfire's own encoder the loop body ran once on silicon (hearsay). hipfire's `rules::check` also rejects every Peano program at its first `ret lr`, since it can't check indirect branch targets. So those rules can't be our rule gate unchanged. This conflict is logged for xdna2-notes.

## Not covered yet

- Data addresses (`movxm p1, #0x71800`) and DMA buffer addresses stay numeric. Branch and loop targets are labels, so code can be inserted, but moving data buffers means editing numbers.
- `insts.elf` (full-ELF flow, `aiebu-asm`) and a raw-ioctl loader to replace XRT's runtime (hipfire has one).
- FastFlowLM's closed xclbins have no `final.prj`, so they are out of scope.
- The rule gate covers the core-program and descriptor rules above. Not yet covered: routing and lock-protocol deadlock, queue depth (F-0001), memory-tile lane rules.

## Sources

- **Read, not copied:** Xilinx bootgen `cdo-driver/cdo_driver.{c,h}`, `versal/include/imageheadertable-versal.h` and `common/src/bifoptions.cpp` (Apache-2.0); mlir-aie `include/aie/Runtime/TxnEncoding.h`.
- **Used and adapted:** hipfire's `peacemaker-npu` (Apache-2.0; its tables are Apache-2.0 WITH LLVM-exception, derived from llvm-aie TableGen). See [rust/README.md](rust/README.md) and `rust/NOTICE.hipfire`.
- **Format notes:** hipfire's PDI and transaction notes are in `xdna2-notes/raw/hipfire/2026-10-03-amdxdna-linux-driver-and-aie2p-format-facts-documented.md`.
