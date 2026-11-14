"""Micro-benchmark of one Qwen3.6 linear-attention (Gated DeltaNet) layer on this GPU, by part.

    dibs run gpu -- [PYTHONPATH=.fla] .venv/bin/python bench_linattn.py Q4_MODEL_DIR [--tokens 4096,16384]

Parts: the four input projections, the depthwise causal conv (nn.Conv1d via MIOpen, against a
4-tap shift-and-add), the chunked gated delta rule (torch fallback, and fla's Triton kernel
when importable), the gated norm and the output projection. Also forward+backward of the whole
layer, the way training runs it.
"""
import argparse
import time

import torch
import torch.nn.functional as F


def timeit(fn, n=3):
    fn()
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t) / n * 1000


def shift_conv(x, weight, bias=None, activation="silu", seq_idx=None):
    """Causal depthwise conv1d as a sum of shifted copies. x [B, C, T], weight [C, K]."""
    k = weight.shape[1]
    t = x.shape[-1]
    xp = F.pad(x, (k - 1, 0))
    y = sum(xp[..., i:i + t] * weight[:, i, None] for i in range(k))
    if bias is not None:
        y = y + bias[:, None]
    return F.silu(y) if activation in ("silu", "swish") else y


def btc_conv(x, weight, bias=None, activation="silu", seq_idx=None):
    """Same conv computed in [B, T, C] layout: x arrives as a transposed view of a contiguous
    [B, T, C] tensor, so shifting along T keeps the channel dimension contiguous."""
    xt = x.transpose(1, 2)
    k, t = weight.shape[1], xt.shape[1]
    xp = F.pad(xt, (0, 0, k - 1, 0))
    w = weight.t()  # [K, C]
    y = xp[:, k - 1:k - 1 + t] * w[k - 1]
    for i in range(k - 1):
        y = y + xp[:, i:i + t] * w[i]
    if bias is not None:
        y = y + bias
    y = F.silu(y) if activation in ("silu", "swish") else y
    return y.transpose(1, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--tokens", default="4096,16384")
    a = ap.parse_args()
    from transformers import Qwen3_5MoeTextConfig
    from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as mq

    cfg = Qwen3_5MoeTextConfig.from_pretrained(a.model)
    torch.manual_seed(0)
    layer = mq.Qwen3_5MoeGatedDeltaNet(cfg, 0).cuda().to(torch.bfloat16)
    for p in layer.parameters():
        if p.dim() > 1:
            torch.nn.init.normal_(p, std=0.02)
    print("chunk rule:", layer.chunk_gated_delta_rule.__module__, "| conv fn:", layer.causal_conv1d_fn)
    for T in (int(x) for x in a.tokens.split(",")):
        x = torch.randn(1, T, cfg.hidden_size, device="cuda", dtype=torch.bfloat16)
        print(f"--- {T} tokens")
        with torch.no_grad():
            print("  forward:             %.1f ms" % timeit(lambda: layer(x)))
            qkv = layer.in_proj_qkv(x).transpose(1, 2)
            print("  projections:         %.1f ms" % timeit(lambda: (layer.in_proj_qkv(x), layer.in_proj_z(x), layer.in_proj_b(x), layer.in_proj_a(x))))
            print("  conv1d (nn.Conv1d):  %.1f ms" % timeit(lambda: F.silu(layer.conv1d(qkv)[:, :, :T])))
            print("  conv1d (shift-add):  %.1f ms" % timeit(lambda: shift_conv(qkv, layer.conv1d.weight.squeeze(1), layer.conv1d.bias)))
            print("  conv1d (BTC layout): %.1f ms" % timeit(lambda: btc_conv(qkv, layer.conv1d.weight.squeeze(1), layer.conv1d.bias)))
            print("  BTC conv max diff:   %.2e" % (btc_conv(qkv, layer.conv1d.weight.squeeze(1), layer.conv1d.bias).float() - F.silu(layer.conv1d(qkv)[:, :, :T]).float()).abs().max().item())
            ref = F.silu(layer.conv1d(qkv)[:, :, :T])
            got = shift_conv(qkv, layer.conv1d.weight.squeeze(1), layer.conv1d.bias)
            print("  conv max abs diff:   %.2e" % (got.float() - ref.float()).abs().max().item())
            H, D = cfg.linear_num_value_heads, cfg.linear_key_head_dim
            q = torch.randn(1, T, H, D, device="cuda", dtype=torch.bfloat16)
            k = torch.randn(1, T, H, D, device="cuda", dtype=torch.bfloat16)
            v = torch.randn(1, T, H, cfg.linear_value_head_dim, device="cuda", dtype=torch.bfloat16)
            g = -torch.rand(1, T, H, device="cuda") * 0.1
            beta = torch.rand(1, T, H, device="cuda", dtype=torch.bfloat16)
            rule = lambda f: f(q, k, v, g=g, beta=beta, initial_state=None, output_final_state=False, use_qk_l2norm_in_kernel=True)
            print("  delta rule (torch):  %.1f ms" % timeit(lambda: rule(mq.torch_chunk_gated_delta_rule)))
            if mq.chunk_gated_delta_rule is not None:
                print("  delta rule (fla):    %.1f ms" % timeit(lambda: rule(mq.chunk_gated_delta_rule)))
                o1, o2 = rule(mq.torch_chunk_gated_delta_rule)[0], rule(mq.chunk_gated_delta_rule)[0]
                print("  fla vs torch rel err: %.2e" % ((o1.float() - o2.float()).norm() / o1.float().norm()).item())
        xg = x.clone().requires_grad_(True)
        print("  fwd+bwd:             %.1f ms" % timeit(lambda: layer(xg).float().pow(2).mean().backward()))
        layer.causal_conv1d_fn = btc_conv
        print("  fwd+bwd, BTC conv:   %.1f ms" % timeit(lambda: layer(xg).float().pow(2).mean().backward()))
        with torch.no_grad():
            print("  forward, BTC conv:   %.1f ms" % timeit(lambda: layer(x)))
        layer.causal_conv1d_fn = None


if __name__ == "__main__":
    main()
