"""How much system RAM does loading a checkpoint onto the GPU take here (unified memory, ROCm)?

    dibs run gpu -- .venv/bin/python load_probe.py MODEL_DIR [--cls auto|qwen3_5_moe]

Prints MemAvailable, process RSS and torch's allocated bytes before and after
from_pretrained(device_map cuda), then after dropping the CPU-side caches.
A loaded model should cost about its bf16 size once; more means a second copy somewhere.
"""
import argparse
import gc
import os
import sys
import time


def meminfo():
    d = {}
    for line in open("/proc/meminfo"):
        k, v = line.split(":")
        d[k] = int(v.split()[0]) / 2**20
    rss = int(open(f"/proc/{os.getpid()}/statm").read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**30
    return {"avail_gb": round(d["MemAvailable"], 1), "cached_gb": round(d["Cached"], 1), "rss_gb": round(rss, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--cls", default="auto")
    a = ap.parse_args()
    import memguard
    memguard.start()
    import torch
    from transformers import AutoModelForCausalLM, Qwen3_5MoeForConditionalGeneration
    import transformers.modeling_utils as _mu
    # transformers pre-allocates the model's size on the GPU before loading. On this ROCm APU
    # that block wasn't reused, so loading took ~2x the weights in system RAM (2026-11-08).
    _mu.caching_allocator_warmup = lambda *args, **kwargs: None
    print("start", meminfo(), flush=True)
    torch.zeros(1, device="cuda")
    print("cuda init", meminfo(), flush=True)
    t = time.time()
    cls = Qwen3_5MoeForConditionalGeneration if a.cls == "qwen3_5_moe" else AutoModelForCausalLM
    m = cls.from_pretrained(a.model, dtype=torch.bfloat16, device_map={"": 0}, disable_mmap=True)
    n = sum(p.numel() * p.element_size() for p in m.parameters()) / 2**30
    print(f"loaded {time.time() - t:.0f}s params {n:.1f} GB torch {torch.cuda.memory_allocated() / 2**30:.1f} GB",
          meminfo(), flush=True)
    gc.collect()
    print("after gc", meminfo(), flush=True)


if __name__ == "__main__":
    sys.exit(main())
