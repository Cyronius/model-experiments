"""Quantize the optimum ONNX export to int8 and package it in the layout
transformers.js expects (onnx/ subdir, *_quantized.onnx naming for dtype 'q8').

Usage:
    python scripts/quantize.py --input model/titler-spike-onnx --output model/titler-spike-web
"""

import argparse
import shutil
from pathlib import Path

from onnxruntime.quantization import QuantType, quantize_dynamic

METADATA_FILES = [
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
]
# transformers.js seq2seq with dtype 'q8' loads exactly these two graphs
GRAPHS = ["encoder_model", "decoder_model_merged"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="optimum ONNX export directory")
    ap.add_argument("--output", required=True, help="web-ready model directory")
    args = ap.parse_args()

    src, dst = Path(args.input), Path(args.output)
    onnx_dir = dst / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)

    for name in METADATA_FILES:
        if (src / name).exists():
            shutil.copy2(src / name, dst / name)

    total = 0
    for graph in GRAPHS:
        model_in = src / f"{graph}.onnx"
        model_out = onnx_dir / f"{graph}_quantized.onnx"
        quantize_dynamic(
            model_input=model_in,
            model_output=model_out,
            weight_type=QuantType.QInt8,
            per_channel=True,
            extra_options={"EnableSubgraph": True},  # decoder_model_merged uses If-node subgraphs
        )
        size = model_out.stat().st_size
        total += size
        print(f"{model_out.name}: {size / 1e6:.2f} MB")

    print(f"total quantized: {total / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
