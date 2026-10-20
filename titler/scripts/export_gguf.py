"""Export a trained titler checkpoint to GGUF (llama.cpp), quantized.

llama.cpp's T5 converter hard-requires a real SentencePiece protobuf (there is
no tokenizer.json fallback for T5 -- see make_spiece.py), so this script:
  1. stages a copy of the checkpoint + a synthesized tokenizer.model
  2. fetches llama.cpp's convert_hf_to_gguf.py + conversion/ package (not
     vendored here; cached after first run) and runs it -> f16 GGUF
  3. quantizes via the ghcr.io/ggml-org/llama.cpp Docker image

Q8_0 vs Q4_0 vs f16 was benchmarked on the 339k-trained v1.2 checkpoint
(2026-08-21, n=100 test prompts, greedy decode): Q8_0 changed 3/100 outputs
from the f16 baseline (paraphrases, not garbles), ROUGE-L -1.2%. Q4_0 changed
24/100, with visible degradation (repetition loops, truncated output),
ROUGE-L -2.1%. Q8_0 is the recommended default -- it's smaller than the
project's int8 ONNX export (7.9MB vs 10.3MB) at negligible quality cost;
Q4_0 saves only ~3MB further for a real quality hit and isn't worth it here.

Requires Docker Desktop running. Python deps beyond the base project:
    uv run --extra gguf python scripts/export_gguf.py ...

Usage:
    uv run --extra gguf python scripts/export_gguf.py --model model/titler-v12-339k --out model/titler-v12-gguf --name titler-v1.2
"""

import argparse
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

from make_spiece import write_spiece

LLAMACPP_TARBALL = "https://github.com/ggml-org/llama.cpp/archive/refs/heads/master.tar.gz"
DOCKER_IMAGE = "ghcr.io/ggml-org/llama.cpp:light"


def fetch_converter(cache_dir: Path) -> Path:
    """Downloads just convert_hf_to_gguf.py + conversion/ + gguf-py from llama.cpp
    master (a few hundred KB, not the ~40MB full tarball) into cache_dir, skipping
    if already present. Returns the extracted repo root."""
    root = cache_dir / "llama.cpp-master"
    if (root / "convert_hf_to_gguf.py").exists():
        print(f"using cached converter: {root}")
        return root

    cache_dir.mkdir(parents=True, exist_ok=True)
    tar_path = cache_dir / "repo.tar.gz"
    print(f"fetching llama.cpp conversion tooling from {LLAMACPP_TARBALL} ...")
    urllib.request.urlretrieve(LLAMACPP_TARBALL, tar_path)

    wanted_prefixes = ("llama.cpp-master/convert_hf_to_gguf.py", "llama.cpp-master/conversion/", "llama.cpp-master/gguf-py/")
    with tarfile.open(tar_path) as tf:
        members = [m for m in tf.getmembers() if m.name.startswith(wanted_prefixes)]
        tf.extractall(cache_dir, members=members)
    tar_path.unlink()
    return root


def stage_checkpoint(model_dir: Path, tokenizer_json: Path, work_dir: Path) -> Path:
    src = work_dir / "src"
    src.mkdir(parents=True, exist_ok=True)
    for f in ("config.json", "model.safetensors", "generation_config.json"):
        shutil.copy(model_dir / f, src / f)
    n = write_spiece(str(tokenizer_json), str(src / "tokenizer.model"))
    print(f"synthesized tokenizer.model: {n} pieces")
    return src


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="fp32 HF checkpoint dir (config.json + model.safetensors)")
    ap.add_argument("--tokenizer", default="model/tokenizer/tokenizer.json")
    ap.add_argument("--out", required=True, help="output dir for the GGUF files")
    ap.add_argument("--name", default="titler", help="output filename stem")
    ap.add_argument("--quant", default="Q8_0", help="quant type for llama-quantize (default: Q8_0 -- see module docstring)")
    ap.add_argument("--cache-dir", default=".cache/llamacpp-convert")
    ap.add_argument("--llamacpp", default=None, help="path to an existing llama.cpp checkout; skips auto-fetch")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    llamacpp_root = Path(args.llamacpp) if args.llamacpp else fetch_converter(Path(args.cache_dir))
    src = stage_checkpoint(Path(args.model), Path(args.tokenizer), out_dir)

    f16_path = out_dir / f"{args.name}-f16.gguf"
    subprocess.run(
        [sys.executable, str(llamacpp_root / "convert_hf_to_gguf.py"), str(src), "--outfile", str(f16_path), "--outtype", "f16"],
        check=True,
    )

    quant_path = out_dir / f"{args.name}-{args.quant.lower()}.gguf"
    subprocess.run(
        [
            "docker", "run", "--rm", "--entrypoint", "/app/llama",
            "-v", f"{out_dir.resolve()}:/models",
            DOCKER_IMAGE, "quantize", f"/models/{f16_path.name}", f"/models/{quant_path.name}", args.quant,
        ],
        check=True,
    )
    print(f"done: {f16_path} , {quant_path}")


if __name__ == "__main__":
    main()
