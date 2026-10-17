"""One-command setup for the router demo.

Downloads google/gemma-3n-E4B-it, produces a 50%-FFN-sliced copy (the "half"
tier), fetches llama.cpp (source, for the GGUF converter; prebuilt Windows
binaries, for serving/quantizing), converts both checkpoints to GGUF bf16,
and quantizes both to Q4_K_M.

Every step checks whether its output already exists and skips if so, so
re-running this script after an interrupted run resumes rather than
restarting.

Requirements to RUN this script (not the demo server):
    pip install huggingface_hub safetensors torch
Torch is CPU-only here -- it is only used to slice tensors, never to run
inference. huggingface_hub needs a token: set HF_TOKEN, or run
`huggingface-cli login` first (either satisfies Gemma's gated-access terms).

The GGUF converter (llama.cpp/convert_hf_to_gguf.py) has its own, heavier
dependency set (transformers, numpy, sentencepiece, gguf, ...). Those are
installed into a dedicated venv at router_demo/.venv, created by this script,
so they never pollute the interpreter running this file.

Layout produced under router_demo/:
    models/hf-full/              raw HF snapshot
    models/hf-half/              50%-FFN-sliced HF snapshot
    models/full-bf16.gguf        converted, unquantized
    models/half-bf16.gguf
    models/full-Q4_K_M.gguf      what router_server.py actually serves
    models/half-Q4_K_M.gguf
    llama.cpp/                   shallow clone (converter script + requirements)
    bin/                         llama-server.exe, llama-quantize.exe, llama-bench.exe, DLLs
    .venv/                       converter's Python environment
"""

import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
BIN_DIR = ROOT / "bin"
LLAMA_CPP_DIR = ROOT / "llama.cpp"
VENV_DIR = ROOT / ".venv"

HF_REPO_ID = "google/gemma-3n-E4B-it"

# Gemma 3n E4B text-tower shapes. hidden=2048 is fixed; matching on the full
# tuple (not just a dim) keeps the multimodal towers untouched, same as
# gemma_phase1/slice_safetensors.py.
HIDDEN = 2048
FULL_INTERMEDIATE = 16384
HALF_INTERMEDIATE = 8192
EXPECTED_SLICED_TENSORS = 105  # 35 text layers x 3 projections (gate/up/down)

GITHUB_RELEASE_API = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"
LLAMA_CPP_GIT_URL = "https://github.com/ggml-org/llama.cpp.git"

WINDOWS_CUDA_ASSET = re.compile(r"win.*cuda.*x64", re.IGNORECASE)
WINDOWS_CPU_ASSET = re.compile(r"win.*(avx2|cpu).*x64", re.IGNORECASE)


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("  $", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


# ---------------------------------------------------------------------------
# Step 1: download the full checkpoint
# ---------------------------------------------------------------------------

def download_full_snapshot() -> Path:
    dst = MODELS_DIR / "hf-full"
    if (dst / "config.json").exists() and any(dst.glob("*.safetensors")):
        print(f"[1/6] full snapshot already present: {dst}")
        return dst

    from huggingface_hub import snapshot_download

    print(f"[1/6] downloading {HF_REPO_ID} ...")
    token = os.environ.get("HF_TOKEN")  # None -> huggingface_hub uses cached login
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=HF_REPO_ID, local_dir=str(dst), token=token)
    print(f"[1/6] downloaded to {dst}")
    return dst


# ---------------------------------------------------------------------------
# Step 2: slice the FFNs to 50% width
# ---------------------------------------------------------------------------

def slice_checkpoint(src_dir: Path, dst_dir: Path, target_intermediate: int,
                      expected_sliced: int) -> Path:
    marker = dst_dir / ".slice_complete"
    if marker.exists() and marker.read_text().strip() == str(target_intermediate):
        print(f"[2/6] sliced snapshot already present: {dst_dir}")
        return dst_dir

    print(f"[2/6] slicing FFNs to intermediate_size={target_intermediate} ...")
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    dst_dir.mkdir(parents=True)

    sliced = 0
    total_bytes = 0
    for item in sorted(src_dir.iterdir()):
        dst_item = dst_dir / item.name
        if item.is_dir():
            shutil.copytree(item, dst_item)
            continue
        if item.suffix == ".safetensors":
            sd = load_file(str(item))
            out = {}
            for k, v in sd.items():
                if k.endswith(("mlp.gate_proj.weight", "mlp.up_proj.weight")) \
                        and tuple(v.shape) == (FULL_INTERMEDIATE, HIDDEN):
                    v = v[:target_intermediate].contiguous()
                    sliced += 1
                elif k.endswith("mlp.down_proj.weight") \
                        and tuple(v.shape) == (HIDDEN, FULL_INTERMEDIATE):
                    v = v[:, :target_intermediate].contiguous()
                    sliced += 1
                out[k] = v
                total_bytes += v.numel() * v.element_size()
            save_file(out, str(dst_item), metadata={"format": "pt"})
            print(f"  sliced shard {item.name}")
        else:
            shutil.copy2(item, dst_item)

    assert sliced == expected_sliced, (
        f"expected {expected_sliced} sliced tensors, got {sliced} -- "
        "checkpoint layout may have changed; check the shape-matching rules"
    )

    cfg_path = dst_dir / "config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["text_config"]["intermediate_size"] = target_intermediate
    cfg_path.write_text(json.dumps(cfg, indent=2))

    idx_path = dst_dir / "model.safetensors.index.json"
    if idx_path.exists():
        idx = json.loads(idx_path.read_text())
        idx.setdefault("metadata", {})["total_size"] = total_bytes
        idx_path.write_text(json.dumps(idx))

    marker.write_text(str(target_intermediate))
    print(f"[2/6] sliced {sliced} tensors, {total_bytes / 1e9:.2f} GB")
    return dst_dir


# ---------------------------------------------------------------------------
# Step 3: llama.cpp source (converter script + its requirements.txt)
# ---------------------------------------------------------------------------

def fetch_llama_cpp_source() -> Path:
    if (LLAMA_CPP_DIR / "convert_hf_to_gguf.py").exists():
        print(f"[3/6] llama.cpp source already present: {LLAMA_CPP_DIR}")
        return LLAMA_CPP_DIR
    if shutil.which("git") is None:
        raise SystemExit("git is required to fetch llama.cpp source; install git and re-run")
    print("[3/6] cloning llama.cpp (shallow) ...")
    run(["git", "clone", "--depth", "1", LLAMA_CPP_GIT_URL, str(LLAMA_CPP_DIR)])
    return LLAMA_CPP_DIR


# ---------------------------------------------------------------------------
# Step 4: llama.cpp prebuilt Windows binaries
# ---------------------------------------------------------------------------

def fetch_latest_release_assets() -> tuple[list[dict], str]:
    req = urllib.request.Request(
        GITHUB_RELEASE_API,
        headers={"User-Agent": "router-demo-setup", "Accept": "application/vnd.github+json"},
    )
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
    return data["assets"], data["tag_name"]


def pick_release_asset(assets: list[dict]) -> dict:
    for pattern in (WINDOWS_CUDA_ASSET, WINDOWS_CPU_ASSET):
        for asset in assets:
            if pattern.search(asset["name"]):
                return asset
    names = "\n".join(f"  - {a['name']}" for a in assets)
    print(f"No Windows CUDA or CPU/AVX2 x64 asset matched. Available assets:\n{names}")
    raise SystemExit(
        "Cannot pick a llama.cpp release asset automatically. Update the regexes "
        "in setup_models.py (WINDOWS_CUDA_ASSET / WINDOWS_CPU_ASSET) or download "
        "one manually and extract it into router_demo/bin/."
    )


def download_file(url: str, dest: Path, chunk_size: int = 1 << 20) -> None:
    print(f"  downloading {url}")
    tmp = dest.with_name(dest.name + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "router-demo-setup"})
    with urllib.request.urlopen(req, timeout=60) as resp, open(tmp, "wb") as f:
        total = int(resp.headers.get("Content-Length", 0))
        read = 0
        last_pct = -1
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
            read += len(chunk)
            if total:
                pct = (read * 100) // total
                if pct != last_pct and pct % 10 == 0:
                    print(f"    {pct}% ({read / 1e6:.0f}/{total / 1e6:.0f} MB)")
                    last_pct = pct
    tmp.replace(dest)


def fetch_llama_cpp_binaries() -> Path:
    server_exe = BIN_DIR / "llama-server.exe"
    if server_exe.exists():
        print(f"[4/6] llama.cpp binaries already present: {BIN_DIR}")
        return BIN_DIR

    print("[4/6] querying latest llama.cpp release ...")
    assets, tag = fetch_latest_release_assets()
    asset = pick_release_asset(assets)
    print(f"[4/6] selected asset: {asset['name']} (release {tag})")

    BIN_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = BIN_DIR / "_download.zip"
    download_file(asset["browser_download_url"], zip_path)

    extract_dir = BIN_DIR / "_pkg"
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)
    zip_path.unlink()

    # Flatten: DLL search on Windows is same-directory, so every exe and dll
    # from the archive (regardless of its internal folder structure) lands
    # directly in BIN_DIR.
    flattened = 0
    for f in extract_dir.rglob("*"):
        if f.is_file():
            dst = BIN_DIR / f.name
            if not dst.exists():
                shutil.copy2(f, dst)
                flattened += 1
    shutil.rmtree(extract_dir)
    print(f"[4/6] extracted {flattened} files into {BIN_DIR}")

    for needed in ("llama-server.exe", "llama-quantize.exe", "llama-bench.exe"):
        if not (BIN_DIR / needed).exists():
            raise SystemExit(f"expected {needed} not found in release asset {asset['name']}")
    return BIN_DIR


# ---------------------------------------------------------------------------
# Step 5: converter venv
# ---------------------------------------------------------------------------

def venv_python_path() -> Path:
    if os.name == "nt":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def setup_converter_venv() -> Path:
    py = venv_python_path()
    marker = VENV_DIR / ".deps_installed"
    if py.exists() and marker.exists():
        print(f"[5/6] converter venv already set up: {VENV_DIR}")
        return py

    print("[5/6] creating converter venv ...")
    run([sys.executable, "-m", "venv", str(VENV_DIR)])

    req_file = LLAMA_CPP_DIR / "requirements" / "requirements-convert_hf_to_gguf.txt"
    if not req_file.exists():
        req_file = LLAMA_CPP_DIR / "requirements.txt"
    if not req_file.exists():
        raise SystemExit(f"no requirements file found under {LLAMA_CPP_DIR}")

    run([str(py), "-m", "pip", "install", "-U", "pip"])
    run([str(py), "-m", "pip", "install", "-r", str(req_file)])
    marker.write_text("ok")
    print("[5/6] converter venv ready")
    return py


# ---------------------------------------------------------------------------
# Step 6: convert to GGUF, then quantize
# ---------------------------------------------------------------------------

def convert_to_gguf(hf_dir: Path, out_file: Path, venv_python: Path) -> None:
    if out_file.exists():
        print(f"[6/6] {out_file.name} already converted, skipping")
        return
    print(f"[6/6] converting {hf_dir.name} -> {out_file.name} (bf16) ...")
    script = LLAMA_CPP_DIR / "convert_hf_to_gguf.py"
    run([str(venv_python), str(script), str(hf_dir), "--outfile", str(out_file), "--outtype", "bf16"])


def quantize(in_file: Path, out_file: Path, quantize_bin: Path) -> None:
    if out_file.exists():
        print(f"[6/6] {out_file.name} already quantized, skipping")
        return
    print(f"[6/6] quantizing {in_file.name} -> {out_file.name} (Q4_K_M) ...")
    run([str(quantize_bin), str(in_file), str(out_file), "Q4_K_M"])


# ---------------------------------------------------------------------------

def main() -> None:
    t0 = time.time()
    print("=== router_demo setup_models ===")

    full_hf = download_full_snapshot()
    half_hf = slice_checkpoint(MODELS_DIR / "hf-full", MODELS_DIR / "hf-half",
                                HALF_INTERMEDIATE, EXPECTED_SLICED_TENSORS)
    fetch_llama_cpp_source()
    fetch_llama_cpp_binaries()
    venv_python = setup_converter_venv()

    convert_to_gguf(full_hf, MODELS_DIR / "full-bf16.gguf", venv_python)
    convert_to_gguf(half_hf, MODELS_DIR / "half-bf16.gguf", venv_python)

    quantize_bin = BIN_DIR / "llama-quantize.exe"
    quantize(MODELS_DIR / "full-bf16.gguf", MODELS_DIR / "full-Q4_K_M.gguf", quantize_bin)
    quantize(MODELS_DIR / "half-bf16.gguf", MODELS_DIR / "half-Q4_K_M.gguf", quantize_bin)

    print(f"=== done in {time.time() - t0:.0f}s ===")
    for name in ("full-Q4_K_M.gguf", "half-Q4_K_M.gguf"):
        p = MODELS_DIR / name
        if p.exists():
            print(f"  {p}  ({p.stat().st_size / 1e9:.2f} GB)")
    print("Next: python router_demo/router_server.py")


if __name__ == "__main__":
    main()
