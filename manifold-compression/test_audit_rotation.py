"""Independent small-model checks before rotating pretrained models."""
import copy

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM, Qwen3_5TextConfig, Qwen3_5ForCausalLM

from audit_rotation import rotate_full_width, cache_check, compare


def test_numerical_gate_rejects_nonfinite_logits():
    reference = torch.zeros(1, 1, 3)
    for value in (float("nan"), float("inf")):
        candidate = reference.clone()
        candidate[0, 0, 0] = value
        with pytest.raises(ValueError, match="non-finite"):
            compare([reference], [candidate])


@pytest.mark.parametrize("architecture", ["llama", "qwen"])
@pytest.mark.parametrize("tied", [True, False])
def test_nontrivial_rotation_preserves_outputs_and_detects_diagonal_loss(architecture, tied):
    torch.set_num_threads(2)
    torch.manual_seed(123)
    kw = dict(vocab_size=101, hidden_size=32, intermediate_size=48, num_hidden_layers=2,
              num_attention_heads=4, num_key_value_heads=2, tie_word_embeddings=tied,
              head_dim=8, attention_dropout=0.0)
    if architecture == "llama":
        model = LlamaForCausalLM(LlamaConfig(**kw))
    else:
        config = Qwen3_5TextConfig(**kw, linear_num_key_heads=2, linear_num_value_heads=2,
                                  linear_key_head_dim=8, linear_value_head_dim=8,
                                  layer_types=["linear_attention", "full_attention"],
                                  rope_parameters={"rope_type": "default", "rope_theta": 10000.,
                                                   "partial_rotary_factor": 1., "mrope_section": [1, 1, 2]})
        model = Qwen3_5ForCausalLM(config)
    model.eval().requires_grad_(False)
    # Default initial gains are uniform: that would hide the diagonal approximation.
    for name, parameter in model.named_parameters():
        if name.endswith("norm.weight") and parameter.numel() == 32:
            parameter.copy_(torch.linspace(0.2, 1.8, 32))
    ids = torch.randint(0, 101, (1, 19))
    q = torch.linalg.qr(torch.randn(32, 32, dtype=torch.float64)).Q.float()
    reference = model(input_ids=ids, use_cache=False).logits
    rotated = copy.deepcopy(model)
    _, diagonal, stats = rotate_full_width(rotated, q)
    exact = rotated(input_ids=ids, use_cache=False).logits
    torch.testing.assert_close(exact, reference, atol=2e-6, rtol=2e-5)
    assert cache_check(rotated, ids)["numerical_pass"]
    rotated.model.norm = diagonal
    approximate = rotated(input_ids=ids, use_cache=False).logits
    assert (approximate - reference).abs().max() > 0.01
    assert stats["discarded_off_diagonal_frobenius_fraction"] > 0.05
