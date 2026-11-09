"""Which SDPA backend handles Qwen3.6's full attention (16 q heads, 2 kv heads, head_dim 256,
causal) on this GPU, and at what cost?   dibs run gpu -- .venv/bin/python bench_attn.py"""
import time
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


def run(T, backend):
    q = torch.randn(1, 16, T, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(1, 16, T, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(1, 16, T, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    with sdpa_kernel([backend]):
        F.scaled_dot_product_attention(q, k, v, is_causal=True).float().sum().backward()
        torch.cuda.synchronize()
        t = time.time()
        F.scaled_dot_product_attention(q, k, v, is_causal=True).float().sum().backward()
        torch.cuda.synchronize()
    return time.time() - t, (torch.cuda.max_memory_allocated() - base) / 2**30


for T in (4096, 16384):
    for b in (SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH):
        if b == SDPBackend.MATH and T > 8192:
            continue
        try:
            dt, mem = run(T, b)
            print(f"T={T} {b.name}: fwd+bwd {dt * 1000:.0f} ms, extra peak {mem:.1f} GB", flush=True)
        except Exception as e:
            print(f"T={T} {b.name}: {type(e).__name__}: {str(e)[:150]}", flush=True)
