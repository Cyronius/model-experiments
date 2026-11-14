"""CPU tests on a tiny random Qwen3: identity at init, isolation, gradients, generation."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
from common import score_qa, user_prompt  # noqa: E402
from model import MemoryModel  # noqa: E402

CFG = {"before_layers": [0, 2], "heads": 4, "kv_heads": 2, "head_dim": 16, "max_slot": 8}


def tiny():
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=97, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=256)
    lm = Qwen3ForCausalLM(cfg).eval()
    return MemoryModel(lm, 24, CFG).eval()


def logits(m, ids, banks=None):
    m.set_memory(banks)
    with torch.no_grad():
        out = m.lm(input_ids=ids).logits
    m.set_memory(None)
    return out


def randomize_out(m):
    for b in m.blocks:
        torch.nn.init.normal_(b.o.weight, std=0.5)


def test_identity_at_init():
    m = tiny()
    ids = torch.randint(0, 97, (2, 11))
    base = logits(m, ids)
    with_mem = logits(m, ids, [torch.randn(5, 24), torch.randn(9, 24)])
    assert torch.equal(base, with_mem)
    assert not any(p.requires_grad for p in m.lm.parameters())
    assert sum(p.numel() for p in m.trainable()) == sum(
        p.numel() for p in list(m.adapter.parameters()) + list(m.blocks.parameters()))


def test_memory_changes_output_and_rows_are_isolated():
    m = tiny()
    randomize_out(m)
    ids = torch.randint(0, 97, (2, 7))
    a, b = torch.randn(5, 24), torch.randn(3, 24)
    out = logits(m, ids, [a, b])
    assert not torch.allclose(out, logits(m, ids))
    # Row 0's output depends only on bank a, regardless of bank b's length or content.
    out2 = logits(m, ids, [a, torch.randn(8, 24)])
    assert torch.allclose(out[0], out2[0], atol=1e-5)
    single = logits(m, ids[:1], [a])
    assert torch.allclose(out[0], single[0], atol=1e-5)


def test_memory_occupies_no_positions():
    m = tiny()
    randomize_out(m)
    ids = torch.randint(0, 97, (1, 6))
    assert logits(m, ids, [torch.randn(40, 24)]).shape == logits(m, ids).shape


def test_gradients_reach_only_memory_modules():
    m = tiny().train()
    ids = torch.randint(0, 97, (2, 9))
    m.set_memory([torch.randn(4, 24), torch.randn(6, 24)])
    m.lm(input_ids=ids, labels=ids).loss.backward()
    names = {n for n, p in m.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0}
    assert any(n.startswith("blocks.") and n.endswith("o.weight") for n in names)
    assert not any(n.startswith("lm.") for n in names)
    assert {n for n, _ in m.named_parameters() if not n.startswith("lm.")} == {
        n for n, p in m.named_parameters() if p.requires_grad}


def test_generate_uses_memory_and_cache():
    m = tiny()
    randomize_out(m)
    ids = torch.randint(0, 97, (1, 5))
    bank = torch.randn(6, 24)
    m.set_memory([bank])
    with torch.no_grad():
        cached = m.lm.generate(input_ids=ids, max_new_tokens=6, do_sample=False, use_cache=True)
        m.set_memory([bank])
        uncached = m.lm.generate(input_ids=ids, max_new_tokens=6, do_sample=False, use_cache=False)
    m.set_memory(None)
    assert torch.equal(cached, uncached)


def test_score_qa():
    assert score_qa("The Eiffel Tower", ["Eiffel Tower"])["em"] == 1.0
    assert score_qa("UNKNOWN", [])["em"] == 1.0
    assert score_qa("UNKNOWN", ["x"])["f1"] == 0.0
    assert score_qa("Paris, France\nmore", ["Paris"])["f1"] == pytest.approx(2 / 3)


def test_prompt_never_contains_answers_for_memory():
    p = user_prompt("qa", "Who?")
    assert "[supplied as memory]" in p and p.endswith("Question: Who?\n")
