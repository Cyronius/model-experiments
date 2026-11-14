#!/usr/bin/env python3
"""Download the pinned public checkpoint. Never calls a paid inference API."""
import argparse
import os

os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")

from huggingface_hub import snapshot_download

MODEL_ID = "latent-context/0.6b-4b-LCLM-4x"
MODEL_REVISION = "d45629704bbc8de96d4b847c0098bf7ccebb3aa4"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out", required=True)
parser.add_argument("--model", choices=["lclm", "stock"], default="lclm")
args = parser.parse_args()
if args.model == "stock":
    MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
    MODEL_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
snapshot_download(MODEL_ID, revision=MODEL_REVISION, local_dir=args.out, token=False, max_workers=4)
