"""Can Qwen3.6-35B-A3B be LoRA-trained on this machine's GPU (ROCm, Strix Halo) at stage-1 lengths?

    dibs run gpu --label ... -- .venv/bin/python lora_check.py Q4_MODEL_DIR data/train.jsonl data/val.jsonl \
        [--lengths 8000,16000,24000,32000] [--steps 4] [--out runs/lora-check.json]

Loads the text model with 4-bit routed experts (q4moe.py; the bf16 checkpoint doesn't fit in this
machine's shared RAM), puts LoRA on the bf16 attention (full and linear: q/k/v/o, in_proj_qkv,
in_proj_z, out_proj) and shared-expert MLPs, with the router and the 256 routed experts frozen
(plan.md: "freeze the router, LoRA on attention + shared experts first"). Gradient checkpointing,
loss on the answer tokens only. For each target length it takes the SFT example closest to it
and runs `--steps` optimizer steps on that one example, recording step time, peak memory and
loss (it should fall, since the same example repeats). Writes a JSON report.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", type=Path)
    ap.add_argument("data", type=Path, nargs="+")
    ap.add_argument("--lengths", default="4000,8000,16000,24000,32000")
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--out", type=Path, default=Path("runs/lora-check.json"))
    ap.add_argument("--mem-floor-gb", type=float, default=12.0, help="exit when MemAvailable falls below this")
    ap.add_argument("--mem-cap-gb", type=float, default=70.0,
                    help="torch allocator cap; GPU memory is system RAM here, so stay clear of the OOM killer")
    a = ap.parse_args()

    import memguard
    memguard.start(a.mem_floor_gb)
    import torch
    import transformers
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer
    from transformers.utils.import_utils import is_flash_linear_attention_available
    import q4moe
    from sft_loss import answer_loss

    report = {"model": str(a.model), "torch": torch.__version__, "hip": torch.version.hip,
              "transformers": transformers.__version__, "device": torch.cuda.get_device_name(0),
              "fla": is_flash_linear_attention_available(), "experts": "q4moe", "rank": a.rank,
              "host": platform.node(), "runs": []}
    a.out.parent.mkdir(parents=True, exist_ok=True)

    def save():
        a.out.write_text(json.dumps(report, indent=1) + "\n")

    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, a.mem_cap_gb * 2**30 / total))
    report["mem_cap_gb"] = a.mem_cap_gb
    tok = AutoTokenizer.from_pretrained(a.model)
    t0 = time.time()
    model = q4moe.load(a.model)
    ex0 = model.model.layers[0].mlp.experts
    d_tri = q4moe.dequantize(ex0.gate_up_q[:8], ex0.gate_up_s[:8], ex0.gate_up_m[:8]).float()
    ref = ((torch.stack((ex0.gate_up_q[:8] & 15, ex0.gate_up_q[:8] >> 4), -1).reshape(8, d_tri.shape[1], -1, q4moe.GROUP).float()
            * ex0.gate_up_s[:8].float().unsqueeze(-1) + ex0.gate_up_m[:8].float().unsqueeze(-1)).reshape(d_tri.shape))
    report["triton_dequant_max_rel_diff"] = float(((d_tri - ref).abs().max() / ref.abs().max()).item())
    print("triton dequant vs fp32 reference, max rel diff", report["triton_dequant_max_rel_diff"], flush=True)
    with torch.no_grad():  # the training attention must match transformers' own sdpa path
        probe = torch.randint(1000, 100000, (1, 512), device="cuda")
        a_out = model(input_ids=probe).logits.float()
        model.set_attn_implementation("sdpa")
        b_out = model(input_ids=probe).logits.float()
        model.set_attn_implementation("causal_sdpa")
        report["causal_sdpa_vs_sdpa_rel_diff"] = float(((a_out - b_out).norm() / b_out.norm()).item())
        print("causal_sdpa vs sdpa logits rel diff", report["causal_sdpa_vs_sdpa_rel_diff"], flush=True)
        del a_out, b_out
    report["load_seconds"] = round(time.time() - t0, 1)
    report["mem_after_load_gb"] = round(torch.cuda.memory_allocated() / 2**30, 1)
    print("loaded", report["load_seconds"], "s", report["mem_after_load_gb"], "GB", flush=True)
    save()

    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    target = (r".*\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)"
              r"|linear_attn\.(in_proj_qkv|in_proj_z|out_proj)"
              r"|shared_expert\.(gate_proj|up_proj|down_proj))")
    model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.0,
                                             target_modules=target, task_type="CAUSAL_LM"))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    report["trainable_params"] = trainable
    print("trainable", trainable, flush=True)

    examples = [json.loads(l) for f in a.data for l in f.open()]

    def encode(e):
        msgs = e["messages"]
        kw = dict(tokenize=False, enable_thinking=False)
        full = tok.apply_chat_template(msgs, **kw)
        prefix = tok.apply_chat_template(msgs[:1], add_generation_prompt=True, **kw)
        ids = tok(full, add_special_tokens=False)["input_ids"]
        n_prefix = len(tok(prefix, add_special_tokens=False)["input_ids"])
        labels = [-100] * n_prefix + ids[n_prefix:]
        return ids, labels

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    model.train()
    for target_len in (int(x) for x in a.lengths.split(",")):
        e = min(examples, key=lambda e: abs(e["meta"]["tokens"] - target_len))
        ids, labels = encode(e)
        x = torch.tensor([ids], device="cuda")
        y = torch.tensor([labels], device="cuda")
        run = {"target": target_len, "tokens": len(ids), "answer_tokens": sum(l != -100 for l in labels),
               "example": e["id"], "steps": []}
        report["runs"].append(run)
        torch.cuda.reset_peak_memory_stats()
        try:
            for step in range(a.steps):
                torch.cuda.synchronize()
                t = time.time()
                loss = answer_loss(model, x, run["answer_tokens"])
                loss.backward()
                opt.step()
                opt.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                dt = time.time() - t
                run["steps"].append({"loss": round(loss.item(), 4), "seconds": round(dt, 2)})
                print(f"len {len(ids)} step {step} loss {loss.item():.4f} {dt:.1f}s "
                      f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)
                save()
            run["peak_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 1)
            later = [s["seconds"] for s in run["steps"][1:]] or [run["steps"][0]["seconds"]]
            run["tokens_per_second"] = round(len(ids) / (sum(later) / len(later)), 1)
        except torch.OutOfMemoryError as err:
            import traceback
            tb = traceback.format_exc()
            run["error"] = f"OOM: {str(err)[:300]}"
            run["traceback"] = tb[-3000:]
            print("OOM at", len(ids), "\n" + tb[-3000:], flush=True)
            save()
            break
        except Exception as err:  # record and stop; the report says how far it got
            run["error"] = f"{type(err).__name__}: {str(err)[:500]}"
            print(run["error"], flush=True)
            save()
            break
        save()
    print(json.dumps({k: v for k, v in report.items() if k != "runs"}, indent=1))
    for r in report["runs"]:
        print(r["tokens"], r.get("peak_gb"), r.get("tokens_per_second"), [s["loss"] for s in r["steps"]], r.get("error", ""))


if __name__ == "__main__":
    sys.exit(main())
