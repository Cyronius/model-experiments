"""Build locally with C:/dev/mlir-aie/iron_env.ps1 active. Artifacts stay here."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, required=True, choices=(384, 512, 1536, 2048))
    p.add_argument("--repeats", type=int, default=1024)
    args = p.parse_args()
    os.environ["WIDTH_K"] = str(args.k)
    os.environ["WIDTH_REPEATS"] = str(args.repeats)
    packager = ROOT / "tools/xclbinutil-build/Release/xclbinutil.exe"
    if packager.is_file():
        os.environ["AIE_XCLBINUTIL"] = str(packager)
        os.environ["PATH"] = os.pathsep.join([
            str(packager.parent), str(ROOT / "tools/deps/x64-windows/bin"),
            os.environ["PATH"]])
    import aie.iron as iron
    import aie.utils.compile.jit.compilabledesign as compilable
    compilable._cleanup_failed_compilation = lambda *a, **kw: None
    from aie.iron.device import from_name
    iron.set_current_device(from_name("npu2", n_cols=1))
    out = ROOT / f"build_k{args.k}_r{args.repeats}"
    # The JIT preserves kernel objects. Remove only this probe's verified workdir.
    work = (out / "final.prj").resolve()
    assert work.is_relative_to(ROOT.resolve()) and work.name == "final.prj"
    if work.exists():
        shutil.rmtree(work)
    out.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location("width_design", ROOT / "design.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    t = time.perf_counter()
    mod.DESIGN.specialize().compile(xclbin_path=str(out / "final.xclbin"),
                                    inst_path=str(out / "insts.bin"))
    import importlib.metadata as metadata
    report = {"k": args.k, "m": 4, "n": 32, "k_blocks": args.k // 16,
              "padded_k": args.k, "repeats": args.repeats,
              "mlir_aie": metadata.version("mlir_aie"),
              "llvm_aie": metadata.version("llvm-aie"),
              "build_seconds": time.perf_counter() - t}
    (out / "build.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
