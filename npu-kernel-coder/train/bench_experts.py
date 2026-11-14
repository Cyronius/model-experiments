"""Micro-benchmark of one routed-experts layer (256 experts, H=2048, I=512, top-8) on this GPU.

    dibs run gpu -- .venv/bin/python bench_experts.py [--tokens 4096,16384]

Times, per layer forward: dequantizing all experts from 4 bits; the current per-expert loop;
torch._grouped_mm on bf16 weights (if this ROCm build has it); and a padded batched matmul.
"""
import argparse
import time

import torch
import torch.nn.functional as F

import q4moe

E, H, I, K = 256, 2048, 512, 8


def timeit(fn, n=5):
    fn()
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t) / n * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", default="4096,16384")
    a = ap.parse_args()
    dev = "cuda"
    w_gu = (torch.randn(E, 2 * I, H, device=dev) * 0.02).bfloat16()
    w_d = (torch.randn(E, H, I, device=dev) * 0.02).bfloat16()
    qg, qd = q4moe.quantize(w_gu), q4moe.quantize(w_d)
    mod = q4moe.Q4Experts(E, H, I, F.silu, device=dev)
    for name, parts in (("gate_up", qg), ("down", qd)):
        for p, v in parts.items():
            getattr(mod, f"{name}_{p}").copy_(v)
    print("dequant all experts, torch:  %.1f ms" % timeit(lambda: (q4moe.dequantize_torch(**qg), q4moe.dequantize_torch(**qd))))
    print("dequant all experts, triton: %.1f ms" % timeit(lambda: (q4moe.dequantize(**qg), q4moe.dequantize(**qd))))
    print("triton == torch dequant:", torch.equal(q4moe.dequantize(**qg), q4moe.dequantize_torch(**qg)),
          torch.equal(q4moe.dequantize(**qd), q4moe.dequantize_torch(**qd)))
    print("has torch._grouped_mm:", hasattr(torch, "_grouped_mm"), " F.grouped_mm:", hasattr(F, "grouped_mm"))
    for T in (int(x) for x in a.tokens.split(",")):
        x = torch.randn(T, H, device=dev, dtype=torch.bfloat16)
        logits = torch.randn(T, E, device=dev)
        wts, idx = torch.topk(logits.softmax(-1), K)
        wts = (wts / wts.sum(-1, keepdim=True)).bfloat16()
        print(f"--- {T} tokens ({T * K} assignments)")
        print("  Q4Experts (bmm):     %.1f ms" % timeit(lambda: mod(x, idx, wts)))
        xg = x.clone().requires_grad_(True)
        print("  Q4Experts fwd+bwd:   %.1f ms" % timeit(lambda: mod(xg, idx, wts).float().pow(2).mean().backward()))

        flat = idx.reshape(-1)
        order = torch.argsort(flat, stable=True)
        tok = order // K
        counts = torch.bincount(flat, minlength=E)
        offs = torch.cumsum(counts, 0).to(torch.int32)
        xs = x[tok]

        def grouped():
            h = torch._grouped_mm(xs, w_gu.transpose(1, 2), offs=offs)
            g, u = h.chunk(2, dim=-1)
            return torch._grouped_mm(F.silu(g) * u, w_d.transpose(1, 2), offs=offs)
        try:
            print("  grouped_mm (bf16 w): %.1f ms" % timeit(grouped))
        except Exception as e:
            print("  grouped_mm failed:", type(e).__name__, str(e)[:200])

        cmax = int(counts.max())
        def padded():
            buf = x.new_zeros(E, cmax, H)
            pos = torch.arange(T * K, device=dev) - torch.repeat_interleave(offs - counts.to(torch.int32), counts)
            buf[flat[order], pos] = xs
            h = torch.bmm(buf, w_gu.transpose(1, 2))
            g, u = h.chunk(2, dim=-1)
            return torch.bmm(F.silu(g) * u, w_d.transpose(1, 2))
        print(f"  padded bmm (max {cmax}/expert, mean {T * K // E}): %.1f ms" % timeit(padded))


if __name__ == "__main__":
    main()
