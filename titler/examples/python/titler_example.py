"""Minimal Python inference example (quantized ONNX, CPU).

    uv run python examples/python/titler_example.py [model_dir]
"""

import re
import sys
import time

from optimum.onnxruntime import ORTModelForSeq2SeqLM
from transformers import AutoTokenizer


def clamp_title(t: str) -> str:
    """Product-boundary clamp: strip stray quotes/punctuation, cap at 8 words."""
    t = re.sub(r"[\"'“”‘’`]", "", t)
    t = re.sub(r"[.!?:;,\s]+$", "", t)
    return " ".join(t.split()[:8])

model_dir = sys.argv[1] if len(sys.argv) > 1 else "model/titler-v1-web"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = ORTModelForSeq2SeqLM.from_pretrained(model_dir, subfolder="onnx", use_merged=True, use_cache=True)

prompts = [
    "how do I center a div horizontally with flexbox",
    "write a poem about the ocean at sunset",
    "my docker compose file keeps failing with port is already allocated",
    "hi",
    "explain quantum entanglement like I'm five",
]
for prompt in prompts:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    start = time.perf_counter()
    out = model.generate(**inputs, max_new_tokens=32, num_beams=2, no_repeat_ngram_size=2)
    ms = (time.perf_counter() - start) * 1000
    print(f"({ms:4.0f}ms) {prompt[:50]!r} -> {clamp_title(tokenizer.decode(out[0], skip_special_tokens=True))!r}")
