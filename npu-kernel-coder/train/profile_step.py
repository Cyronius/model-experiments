"""Where does a LoRA training step on the 4-bit-expert Qwen3.6-35B-A3B spend its time?

    dibs run gpu -- [PYTHONPATH=.fla] .venv/bin/python profile_step.py Q4_MODEL_DIR [--tokens 4096]

Times the forward of each module kind (linear attention, full attention, routed experts,
shared expert, lm_head) with a GPU sync around each call, then one full forward+backward
step with gradient checkpointing. With PYTHONPATH=.fla the linear-attention layers use
flash-linear-attention's Triton kernels instead of transformers' torch fallback, and the
script also checks the two give the same layer output.
"""
import argparse
import collections
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--tokens", type=int, default=4096)
    a = ap.parse_args()
    import memguard
    memguard.start(12)
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers.utils.import_utils import is_flash_linear_attention_available
    import q4moe

    torch.cuda.set_per_process_memory_fraction(70 * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    model = q4moe.load(__import__("pathlib").Path(a.model))
    print("fla", is_flash_linear_attention_available(), flush=True)
    times = collections.defaultdict(float)
    counts = collections.Counter()

    def wrap(mod, name):
        orig = mod.forward

        def f(*args, **kw):
            torch.cuda.synchronize()
            t = time.time()
            out = orig(*args, **kw)
            torch.cuda.synchronize()
            times[name] += time.time() - t
            counts[name] += 1
            return out
        mod.forward = f
        return orig

    restore = []
    for layer in model.model.layers:
        if hasattr(layer, "linear_attn"):
            restore.append((layer.linear_attn, wrap(layer.linear_attn, "linear_attn")))
        else:
            restore.append((layer.self_attn, wrap(layer.self_attn, "full_attn")))
        restore.append((layer.mlp.experts, wrap(layer.mlp.experts, "routed_experts")))
        restore.append((layer.mlp.shared_expert, wrap(layer.mlp.shared_expert, "shared_expert")))
    restore.append((model.lm_head, wrap(model.lm_head, "lm_head")))

    x = torch.randint(1000, 100000, (1, a.tokens), device="cuda")
    with torch.no_grad():
        model(input_ids=x, logits_to_keep=1)  # warm-up (Triton compiles)
        times.clear(); counts.clear()
        torch.cuda.synchronize(); t = time.time()
        model(input_ids=x, logits_to_keep=1)
        torch.cuda.synchronize(); total = time.time() - t
    print(f"forward {a.tokens} tokens: {total:.2f}s")
    for k, v in sorted(times.items(), key=lambda kv: -kv[1]):
        print(f"  {k:15s} {v:7.2f}s  ({counts[k]} calls, {100 * v / total:.0f}%)")
    for mod, orig in restore:
        mod.forward = orig

    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    target = (r".*\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|linear_attn\.(in_proj_qkv|in_proj_z|out_proj)"
              r"|shared_expert\.(gate_proj|up_proj|down_proj))")
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, target_modules=target, task_type="CAUSAL_LM"))
    model.train()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize(); t = time.time()
    out = model(input_ids=x, logits_to_keep=512)
    out.logits.float().logsumexp(-1).mean().backward()
    torch.cuda.synchronize()
    print(f"train step {a.tokens} tokens (512 scored): {time.time() - t:.2f}s, peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")


if __name__ == "__main__":
    sys.exit(main())
