"""Does llama.cpp apply the converted LoRA the way training did?

    dibs run gpu -- PYTHONPATH=.fla .venv/bin/python lora_agree.py Q4_MODEL_DIR ADAPTER.pt \
        PROMPT.txt ANSWER_LORA.txt ANSWER_BASE.txt

Teacher-forces each llama-server answer (the LoRA run's and the base run's, same task) through the
PyTorch model, with and without the adapter, and reports how often the PyTorch greedy next token
equals the token llama.cpp produced, plus the mean loss. If llama.cpp applies the adapter
correctly, the adapter model should agree with the LoRA answer about as well as the base model
agrees with the base answer; if the adapter is mangled in conversion, it won't.
"""
import sys
from pathlib import Path


def main():
    model_dir, adapter, prompt_f, ans_lora_f, ans_base_f = map(Path, sys.argv[1:6])
    import memguard
    memguard.start(12)
    import torch
    import torch.nn.functional as F
    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
    from transformers import AutoTokenizer
    import q4moe

    tok = AutoTokenizer.from_pretrained(model_dir)
    model = q4moe.load(model_dir)
    model.set_attn_implementation("sdpa")
    target = (r".*\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|linear_attn\.(in_proj_qkv|in_proj_z|out_proj)"
              r"|shared_expert\.(gate_proj|up_proj|down_proj))")
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules=target,
                                             task_type="CAUSAL_LM"))
    set_peft_model_state_dict(model, torch.load(adapter, map_location="cuda"))
    model.eval()
    prompt = prompt_f.read_text()
    prefix = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                     add_generation_prompt=True, enable_thinking=False)
    p_ids = tok(prefix, add_special_tokens=False)["input_ids"]

    def score(answer_text, use_adapter):
        a_ids = tok(answer_text, add_special_tokens=False)["input_ids"]
        x = torch.tensor([p_ids + a_ids], device="cuda")
        n = len(a_ids)
        with torch.no_grad():
            if use_adapter:
                h = model.get_base_model().model(input_ids=x).last_hidden_state[0, -(n + 1):-1]
            else:
                with model.disable_adapter():
                    h = model.get_base_model().model(input_ids=x).last_hidden_state[0, -(n + 1):-1]
            logits = model.get_base_model().lm_head(h).float()
        t = x[0, -n:]
        agree = (logits.argmax(-1) == t).float()
        first = int((agree == 0).nonzero()[0]) if (agree == 0).any() else n
        return agree.mean().item(), F.cross_entropy(logits, t).item(), first, n

    for name, f in (("LoRA answer", ans_lora_f), ("base answer", ans_base_f)):
        text = f.read_text()
        for use in (True, False):
            acc, loss, first, n = score(text, use)
            print(f"{name:12s} {'with adapter' if use else 'no adapter  '}: argmax agreement {acc:.4f}, "
                  f"loss {loss:.4f}, first disagreement at token {first}/{n}", flush=True)


if __name__ == "__main__":
    main()
