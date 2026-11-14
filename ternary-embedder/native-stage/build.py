"""Build complete projections; reuses the locally built width-probe packager."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--array", action="store_true", help="32 workers / 128 token rows")
    args = parser.parse_args()
    if (args.k, args.n) not in ((384, 384), (384, 1536), (1536, 384)):
        parser.error("Supported shapes: 384x384, 384x1536, 1536x384")
    os.environ.update(STAGE_K=str(args.k), STAGE_N=str(args.n))
    packager_root = ROOT.parent / "npu-width-probe/tools"
    packager = packager_root / "xclbinutil-build/Release/xclbinutil.exe"
    if not packager.is_file():
        raise FileNotFoundError(f"Build the width-probe packager first: {packager}")
    os.environ["AIE_XCLBINUTIL"] = str(packager)
    os.environ["PATH"] = os.pathsep.join([str(packager.parent), str(packager_root / "deps/x64-windows/bin"), os.environ["PATH"]])
    import aie.iron as iron
    from aie.iron.device import from_name
    iron.set_current_device(from_name("npu2", n_cols=8 if args.array else 1))
    if args.array:
        from array_design import DESIGN
    else:
        from design import DESIGN
    out = ROOT / f"build_{'array_' if args.array else ''}k{args.k}_n{args.n}"
    out.mkdir(exist_ok=True)
    work = (out / "final.prj").resolve()
    if not work.is_relative_to(ROOT.resolve()) or work.name != "final.prj":
        raise ValueError(f"Build workdir must remain inside this experiment: {work}")
    if work.exists():
        shutil.rmtree(work)
    start = time.perf_counter()
    DESIGN.specialize().compile(xclbin_path=str(out / "final.xclbin"), inst_path=str(out / "insts.bin"))
    size_tool = Path(os.environ["PEANO_INSTALL_DIR"]) / "bin/llvm-size.exe"
    elf = work / "elfs_main_core_0_2/elfs_main_core_0_2.elf"
    sections = subprocess.check_output([str(size_tool), "-A", str(elf)], text=True)
    text_bytes = int(re.search(r"^\.text\s+(\d+)", sections, re.MULTILINE)[1])
    groups = 6 if args.k == 1536 else 1
    report = {"k": args.k, "n": args.n, "rows": 128 if args.array else 4, "groups": 6 if args.k == 1536 else 1,
              "weight_residency": "streamed from host for each dispatch",
              "mlir_aie": importlib.metadata.version("mlir_aie"),
              "llvm_aie": importlib.metadata.version("llvm-aie"),
              "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                for name in ("projection.cc", "array_design.py" if args.array else "design.py")},
              "artifact_sha256": {name: hashlib.sha256((out / name).read_bytes()).hexdigest()
                                  for name in ("final.xclbin", "insts.bin")},
              "resources": {"cores": 32 if args.array else 1, "program_text_bytes": text_bytes, "stack_bytes_per_core": 4096,
                            "activation_buffer_bytes": 4 * args.k, "weight_buffer_bytes": 8 * args.k,
                            "output_buffer_bytes": 512 * groups,
                            "memtile_buffer_bytes_per_column": 4*(4*args.k+512*groups) if args.array else 0,
                            "shim_mm2s_channels": 9 if args.array else 2, "shim_s2mm_channels": 8 if args.array else 1},
              "build_seconds": time.perf_counter() - start}
    (out / "build.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
