"""Phase-0 spike: load the quantized ONNX model in Python ONNX Runtime,
dump tokenizations of tricky strings (for cross-checking against
transformers.js), and run a generation.

Usage:
    python scripts/spike_check_python.py --model model/titler-spike-web --out data/spike_tokenization_python.json
"""

import argparse
import json
import time
from pathlib import Path

from optimum.onnxruntime import ORTModelForSeq2SeqLM
from transformers import AutoTokenizer

STRINGS = json.loads((Path(__file__).parent / "spike_test_strings.json").read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dump = {s: tokenizer(s)["input_ids"] for s in STRINGS}
    Path(args.out).write_text(json.dumps(dump, indent=2), encoding="utf-8")
    print(f"wrote tokenization dump for {len(STRINGS)} strings to {args.out}")

    # NOTE: do not pass decoder_file_name here — optimum treats an explicit name
    # as a request for the split decoder + decoder_with_past pair and errors.
    # The *_quantized.onnx names match its merged-decoder pattern, so inference works.
    model = ORTModelForSeq2SeqLM.from_pretrained(
        args.model,
        subfolder="onnx",
        use_merged=True,
        use_cache=True,
    )
    prompt = STRINGS[0]
    inputs = tokenizer(prompt, return_tensors="pt")
    start = time.perf_counter()
    gen = model.generate(**inputs, max_new_tokens=16, num_beams=2)
    elapsed = (time.perf_counter() - start) * 1000
    text = tokenizer.decode(gen[0], skip_special_tokens=True)
    print(f"python ORT generation ({elapsed:.0f}ms): {text!r}")


if __name__ == "__main__":
    main()
