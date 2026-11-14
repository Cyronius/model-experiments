# npuasm-core

AIE2P core programs without the vendor toolchain: `lift` decodes a core ELF to
llvm-objdump syntax, `asm` assembles that text back to the same bytes, `check`
does both and compares.

```sh
cargo build --release
./target/release/npuasm-core check path/to/elfs_main_core_0_2.elf
```

It builds on hipfire's `peacemaker-npu`
([warpfront/hipfire](https://github.com/warpfront/hipfire), Apache-2.0), used
as a path dependency on a checkout at `~/code/hipfire` (commit `efa8fbb`,
sparse: `crates/peacemaker-npu` is the only crate needed). Its generated ISA
tables are Apache-2.0 WITH LLVM-exception, derived from Xilinx llvm-aie
TableGen. `src/decode_ext.rs` is adapted from its `src/isa/decode.rs` and
keeps that license; `NOTICE.hipfire` is hipfire's NOTICE, included as
Apache-2.0 §4(d) requires. `src/asm.rs` (the text parser) and `src/main.rs`
are new.

To get the dependency:

```sh
git clone --depth 1 --filter=blob:none --sparse https://github.com/warpfront/hipfire.git ~/code/hipfire
git -C ~/code/hipfire sparse-checkout set --no-cone '/LICENSE*' '/NOTICE' '/Cargo.toml' '/crates/peacemaker-npu/'
```
