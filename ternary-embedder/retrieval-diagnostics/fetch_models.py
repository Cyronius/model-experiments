"""Download only the baseline model and student tokenizer into this workspace."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ["HF_HOME"] = str(ROOT / "cache/huggingface")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
from huggingface_hub import snapshot_download

TOKEN_FILES = ["config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.txt"]

if __name__ == "__main__":
    for repo, revision, files in (
        ("sentence-transformers/all-MiniLM-L6-v2", "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
         TOKEN_FILES + ["model.safetensors", "1_Pooling/config.json", "modules.json"]),
        ("BAAI/bge-small-en-v1.5", "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a", TOKEN_FILES),
    ):
        print(snapshot_download(repo, revision=revision, allow_patterns=files), flush=True)
