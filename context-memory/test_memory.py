"""Meaningful mechanics checks, independent of the multi-GB checkpoint."""
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from memory import INSTRUCTION, answer, make_prefix, prefill, question_suffix, setup, text_embeddings
from run import score_answer


class TinyTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [2 + ord(c) % 60 for c in text]

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(65 + (int(i) % 26)) for i in ids)


def tiny_model():
    setup(2)
    model = Qwen3ForCausalLM(Qwen3Config(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        max_position_embeddings=2048, tie_word_embeddings=True,
        eos_token_id=1, pad_token_id=0,
    )).eval()
    return model, TinyTokenizer()


def test_injection_equals_original_token_embeddings():
    """If memory is token embeddings, injected and direct paths must agree."""
    model, tokenizer = tiny_model()
    document = "encoded source facts"
    with torch.inference_mode():
        memory = text_embeddings(model, tokenizer, document)[0]
        injected = make_prefix(model, tokenizer, memory=memory)
        text = "<|im_start|>user\n" + INSTRUCTION + "<|memory_start|>" + document + "<|memory_end|>"
        direct = text_embeddings(model, tokenizer, text)
        torch.testing.assert_close(injected, direct, rtol=0, atol=0)
        a = model(inputs_embeds=injected).logits
        b = model(input_ids=torch.tensor([tokenizer.encode(text)])).logits
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_cached_prefix_agrees_and_queries_do_not_contaminate_each_other():
    model, tokenizer = tiny_model()
    with torch.inference_mode():
        prefix = make_prefix(model, tokenizer, document="memory notes")
        cache = prefill(model, prefix)
        question = "What was measured?"
        a = answer(model, tokenizer, cache, question, max_tokens=3)
        answer(model, tokenizer, cache, "A different intervening question", max_tokens=4)
        b = answer(model, tokenizer, cache, question, max_tokens=3)
        assert a["answer"] == b["answer"]
        assert cache.get_seq_length() == prefix.shape[1]
        # Direct concatenation must give the same first generated token as reuse.
        suffix = question_suffix(question)
        joined = torch.cat([prefix, text_embeddings(model, tokenizer, suffix)], dim=1)
        first = int(model(inputs_embeds=joined).logits[0, -1].argmax())
        assert a["answer"].startswith(tokenizer.decode([first]))


def test_scoring_distinguishes_stack_context_and_swapped_canary():
    stack = {"patterns": ["0x1800"], "forbidden": ["0x3000"]}
    assert score_answer(stack, "latent", "0x1800")[0]
    assert not score_answer(stack, "latent", "0x1800 or 0x3000")[0]
    canary = {"patterns": ["velvet-heron-7316"], "counterfactual_patterns": ["copper-badger-9527"]}
    assert not score_answer(canary, "counterfactual", "velvet-heron-7316")[0]
    assert score_answer(canary, "counterfactual", "copper-badger-9527")[0]
