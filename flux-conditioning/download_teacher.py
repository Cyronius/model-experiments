"""Download and verify only the original Klein components needed for text targets."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
import time
import urllib.request

from huggingface_hub import HfApi, hf_hub_download

REPO = "black-forest-labs/FLUX.2-klein-4B"
REVISION = "e7b7dc27f91deacad38e78976d1f2b499d76a294"
SOURCE_COMMIT = "c60830ee365d520ab52b110dda562dd26f7b4d7f"
SOURCES = {
    "src/diffusers/pipelines/flux2/pipeline_flux2_klein.py": "8afefd06451d2bed0f34b59e930e60383c6a47fc8cebbec74ea30bfcdc7b79a9",
    "src/diffusers/models/transformers/transformer_flux2.py": "88cbc0569409698efbdaa5cd25403b78fb6607b9cf1b15855bc392b2ad0c1607",
}


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as reader:
        for block in iter(lambda: reader.read(8 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def download(destination: Path, workers: int = 4) -> None:
    if workers < 1:
        raise ValueError("workers must be positive")
    destination.mkdir(parents=True, exist_ok=True)
    info = HfApi().model_info(REPO, revision=REVISION, files_metadata=True)
    if info.sha != REVISION:
        raise ValueError("Unexpected Hub revision")
    files = [f for f in info.siblings if f.rfilename.startswith(("text_encoder/", "tokenizer/"))
             or f.rfilename in {"transformer/config.json", "transformer/diffusion_pytorch_model.safetensors"}]
    needed = sum(f.size for f in files if not (destination / f.rfilename).exists())
    if shutil.disk_usage(destination).free < needed + 512 * 1024 * 1024:
        raise OSError("Insufficient free space for selected teacher files and working margin")
    print("Selected", len(files), "files;", sum(f.size for f in files), "bytes", flush=True)
    started = time.time()

    def fetch(file):
        path = Path(hf_hub_download(REPO, file.rfilename, revision=REVISION, local_dir=destination))
        if path.stat().st_size != file.size:
            raise ValueError(f"Size mismatch: {file.rfilename}")
        sha = digest(path)
        if file.lfs:
            if sha != file.lfs.sha256:
                raise ValueError(f"LFS SHA-256 mismatch: {file.rfilename}")
        else:
            blob = b"blob " + str(file.size).encode() + b"\0" + path.read_bytes()
            if hashlib.sha1(blob).hexdigest() != file.blob_id:
                raise ValueError(f"Git blob mismatch: {file.rfilename}")
        return {"path": file.rfilename, "size": file.size, "sha256": sha,
                "hub_blob_id": file.blob_id, "hub_verified": True}

    records = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(fetch, f) for f in files]):
            record = future.result()
            records.append(record)
            print("Verified", record["path"], flush=True)
    sources = []
    for relative, expected in SOURCES.items():
        url = f"https://raw.githubusercontent.com/huggingface/diffusers/{SOURCE_COMMIT}/{relative}"
        request = urllib.request.Request(url, headers={"User-Agent": "Phlegm-research"})
        with urllib.request.urlopen(request, timeout=60) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f"Reference source SHA-256 mismatch: {relative}")
        path = destination / "reference-source" / Path(relative).name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(data)
        sources.append({"url": url, "path": path.relative_to(destination).as_posix(), "sha256": expected})
    manifest = {"repo": REPO, "revision": REVISION, "files": sorted(records, key=lambda r: r["path"]),
                "diffusers_commit": SOURCE_COMMIT, "reference_source": sources,
                "download_script_sha256": digest(Path(__file__)), "elapsed_seconds": time.time() - started}
    temporary = destination / "verified-upstream.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination / "verified-upstream.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    download(args.output, args.workers)


if __name__ == "__main__":
    main()
