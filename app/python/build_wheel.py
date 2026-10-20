"""Builds a platform wheel that holds the C library and the model (2026-10-20).

    cargo build --release -p titler-capi      # from app/
    python build_wheel.py                     # from app/python; writes dist/titler-*.whl
"""

import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

HERE = Path(__file__).parent
TARGET = HERE.parent / "target" / "release"
LIBS = {"win32": "titler_c.dll", "darwin": "libtitler_c.dylib"}
MODEL = HERE.parent / "model" / "titler-v3-q4_k_m.gguf"


def main() -> None:
    lib = TARGET / LIBS.get(sys.platform, "libtitler_c.so")
    if not lib.exists():
        sys.exit(f"{lib} is missing: run `cargo build --release -p titler-capi` in app/ first")
    package = HERE / "titler"
    staged = [package / lib.name, package / "titler.gguf"]
    shutil.copy(lib, staged[0])
    shutil.copy(MODEL, staged[1])
    dist = HERE / "dist"
    shutil.rmtree(dist, ignore_errors=True)
    try:
        subprocess.run([sys.executable, "-m", "build", "--wheel", "--outdir", str(dist)], cwd=HERE, check=True)
        plat = sysconfig.get_platform().replace("-", "_").replace(".", "_")
        wheel = next(dist.glob("*.whl"))
        subprocess.run([sys.executable, "-m", "wheel", "tags", "--remove", "--python-tag", "py3", "--abi-tag", "none", "--platform-tag", plat, str(wheel)], check=True)
    finally:
        for f in staged:
            f.unlink(missing_ok=True)
        shutil.rmtree(HERE / "build", ignore_errors=True)
        shutil.rmtree(HERE / "titler.egg-info", ignore_errors=True)
    print(*dist.glob("*.whl"))


if __name__ == "__main__":
    main()
