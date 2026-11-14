#!/usr/bin/env python3
"""Download only public open weights; no hosted inference calls."""
import argparse
import os

os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
from huggingface_hub import snapshot_download

MODELS = {
    "nuextract2b": ("numind/NuExtract-2.0-2B", "5eb4a99746edc2ac44900f56a5c9431d3c03b16d"),
    "qwen4b": ("Qwen/Qwen3-4B-Instruct-2507", "cdbee75f17c01a7cc42f958dc650907174af0554"),
    "nuextract3": ("numind/NuExtract3", "c99dc8f5641b866aa0192b6ea78f84bf9f3535f1"),
}
p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--model", choices=list(MODELS), default="nuextract2b")
p.add_argument("--out", required=True)
a = p.parse_args()
name, revision = MODELS[a.model]
snapshot_download(name, revision=revision, local_dir=a.out, token=False, max_workers=4)
