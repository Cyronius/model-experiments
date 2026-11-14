import copy
import json
import math
import runpy
import sys
from pathlib import Path

import pytest
import torch
from transformers import (AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM,
                          MistralConfig, MistralForCausalLM, Qwen3_5TextConfig, Qwen3_5ForCausalLM)

from audit_rotation import rotate_full_width, cache_check
from projection_models import (MatrixRMSNorm, install_output_matrix, load_causal_model,
                               materialize_stock_output, projected_output_matrix, project_residual_width)


def tiny_model(architecture, tied):
    torch.set_num_threads(2)
    torch.manual_seed(7)
    kw = dict(vocab_size=101, hidden_size=32, intermediate_size=48, num_hidden_layers=2,
              num_attention_heads=4, num_key_value_heads=2, head_dim=8, tie_word_embeddings=tied)
    if architecture == "llama":
        model = LlamaForCausalLM(LlamaConfig(**kw))
    elif architecture == "mistral":
        model = MistralForCausalLM(MistralConfig(**kw))
    else:
        config = Qwen3_5TextConfig(**kw, linear_num_key_heads=2, linear_num_value_heads=2,
                                  linear_key_head_dim=8, linear_value_head_dim=8,
                                  layer_types=["linear_attention", "full_attention"],
                                  rope_parameters={"rope_type": "default", "rope_theta": 10000.,
                                                   "partial_rotary_factor": 1., "mrope_section": [1, 1, 2]})
        model = Qwen3_5ForCausalLM(config)
    model.eval().requires_grad_(False)
    model.model.norm.weight.copy_(torch.linspace(0.2, 1.8, 32))
    return model


@pytest.mark.parametrize("architecture", ["qwen", "llama", "mistral"])
@pytest.mark.parametrize("tied", [False, True])
def test_rotation_adapter_survives_hf_roundtrip_and_stock_export(tmp_path, architecture, tied):
    model = tiny_model(architecture, tied)
    ids = torch.randint(0, 101, (1, 17))
    reference = model(ids, use_cache=False).logits
    q = torch.linalg.qr(torch.randn(32, 32, dtype=torch.float64)).Q.float()
    exact, _, _ = rotate_full_width(model, q)
    # Replace the diagnostic buffer module with the production serializable one.
    install_output_matrix(model, exact.matrix)
    torch.testing.assert_close(model(ids, use_cache=False).logits, reference, atol=2e-6, rtol=2e-5)
    folder = tmp_path / "adapter"
    model.save_pretrained(folder)
    reloaded, info = load_causal_model(folder, dtype=torch.float32, output_loading_info=True)
    assert not info["missing_keys"] and not info["unexpected_keys"]
    assert isinstance(reloaded.model.norm, MatrixRMSNorm)
    assert (reloaded.lm_head.weight is reloaded.model.embed_tokens.weight) == tied
    torch.testing.assert_close(reloaded(ids, use_cache=False).logits, reference, atol=2e-6, rtol=2e-5)
    # The directory is self-contained for callers outside this repository.
    external = AutoModelForCausalLM.from_pretrained(folder, trust_remote_code=True, dtype=torch.float32)
    torch.testing.assert_close(external(ids, use_cache=False).logits, reference, atol=2e-6, rtol=2e-5)
    assert cache_check(reloaded, ids)["numerical_pass"]
    # Also check materialization after the independently trainable gain changes.
    reloaded.model.norm.weight.data.add_(0.1)
    before_export = reloaded(ids, use_cache=False).logits
    standard = materialize_stock_output(reloaded)
    standard.save_pretrained(tmp_path / "stock")
    stock = AutoModelForCausalLM.from_pretrained(tmp_path / "stock", dtype=torch.float32)
    assert not stock.config.tie_word_embeddings
    torch.testing.assert_close(stock(ids, use_cache=False).logits, before_export, atol=2e-6, rtol=2e-5)


def test_rectangular_rms_scale_multiplies_instead_of_divides():
    torch.manual_seed(9)
    d, k = 12, 6
    p = torch.linalg.qr(torch.randn(d, d, dtype=torch.float64)).Q[:, :k].float()
    z = torch.randn(7, k)
    gain = torch.linspace(0.4, 2.0, d)
    h = z @ p.T
    target = (h / h.square().mean(-1, keepdim=True).sqrt() * gain) @ p
    c = math.sqrt(d / k)
    norm = MatrixRMSNorm(k, eps=0., offset=0.)
    with torch.no_grad():
        norm.matrix.copy_(projected_output_matrix(gain, p, c))
    torch.testing.assert_close(norm(z), target, atol=2e-6, rtol=2e-6)
    assert (norm(z) / (c * c) - target).abs().max() > 0.1


@pytest.mark.parametrize("architecture", ["qwen", "llama", "mistral"])
@pytest.mark.parametrize("width", [24, 32])
def test_width_builder_and_saved_adapter(tmp_path, architecture, width):
    model = tiny_model(architecture, True)
    ids = torch.randint(0, 101, (1, 15))
    reference = model(ids, use_cache=False).logits
    gain = model.model.norm.weight.clone() + (1 if architecture == "qwen" else 0)
    basis = torch.linalg.qr(torch.randn(32, 32, dtype=torch.float64)).Q[:, :width].float()
    project_residual_width(model, basis, [1.] * 5)
    install_output_matrix(model, projected_output_matrix(gain, basis, 1.))
    expected = model(ids, use_cache=False).logits
    if width == 32:
        torch.testing.assert_close(expected, reference, atol=3e-6, rtol=3e-5)
    model.save_pretrained(tmp_path)
    reloaded = load_causal_model(tmp_path, dtype=torch.float32)
    assert reloaded.config.hidden_size == width
    assert reloaded.config.head_dim == 8
    torch.testing.assert_close(reloaded(ids, use_cache=False).logits, expected, atol=0., rtol=0.)
    assert cache_check(reloaded, ids)["numerical_pass"]


def test_bf16_checkpoint_keeps_matrix_and_can_train_it(tmp_path):
    model = tiny_model("qwen", True)
    matrix = torch.eye(32) + 0.02 * torch.randn(32, 32)
    install_output_matrix(model, matrix)
    ids = torch.randint(0, 101, (1, 13))
    model.to(torch.bfloat16).save_pretrained(tmp_path)
    # Compare identical rounded weights in FP32 to isolate serialization from precision.
    model.float()
    expected = model(ids, use_cache=False).logits
    reloaded = load_causal_model(tmp_path, dtype=torch.float32)
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(dict(reloaded.named_parameters())[name], parameter, atol=0., rtol=0.)
    # Rotary buffers are regenerated in FP32 on reload; model.float() instead
    # retains their earlier BF16 rounding. All persisted parameters match exactly.
    torch.testing.assert_close(reloaded(ids, use_cache=False).logits, expected, atol=2e-5, rtol=1e-4)
    reloaded.requires_grad_(False)
    reloaded.model.norm.requires_grad_(True)
    reloaded(ids, use_cache=False).logits.square().mean().backward()
    assert reloaded.model.norm.matrix.grad.abs().sum() > 0
    assert reloaded.model.norm.weight.grad.abs().sum() > 0


def test_pretrained_qwen_prefix_conversion_survives_adapter(tmp_path):
    from safetensors.torch import save_file
    source = tiny_model("qwen", True)
    original = tmp_path / "original"
    original.mkdir()
    source.config.save_pretrained(original)
    # Reproduce the multimodal prefix used by the actual Qwen3.5 checkpoint.
    weights = {name.replace("model.", "model.language_model.", 1): value.clone()
               for name, value in source.state_dict().items() if name != "lm_head.weight"}
    save_file(weights, original / "model.safetensors", metadata={"format": "pt"})
    model, info = load_causal_model(original, dtype=torch.float32, output_loading_info=True)
    assert not info["missing_keys"] and not info["unexpected_keys"]
    install_output_matrix(model, torch.eye(32) + 0.02 * torch.randn(32, 32))
    ids = torch.randint(0, 101, (1, 13))
    expected = model(ids, use_cache=False).logits
    adapter = tmp_path / "adapter"
    model.save_pretrained(adapter)
    for loader, options in [(load_causal_model, {}),
                            (AutoModelForCausalLM.from_pretrained, {"trust_remote_code": True})]:
        reloaded, info = loader(adapter, dtype=torch.float32, output_loading_info=True, **options)
        assert not info["missing_keys"] and not info["unexpected_keys"]
        torch.testing.assert_close(reloaded(ids, use_cache=False).logits, expected, atol=0., rtol=0.)


@pytest.mark.parametrize("mode", ["matrix", "legacy-diagonal"])
@pytest.mark.parametrize("tied", [False, True])
def test_production_builder_rotation_gate(tmp_path, monkeypatch, mode, tied):
    import common
    model = tiny_model("qwen", tied)
    ids = torch.randint(0, 101, (4, 24)).tolist()
    rows = [("chat", row, [1] * 24) for row in ids]
    monkeypatch.setattr(common, "load_model", lambda *a, **kw: (model, None))
    monkeypatch.setattr(common, "build_data", lambda *a, **kw: {"fit": rows[:2], "eval": rows[2:]})
    monkeypatch.setattr(common, "RESULTS", tmp_path)
    monkeypatch.setattr(common, "session_from", lambda args: common.Session("tiny", T=24))
    monkeypatch.setattr(sys, "argv", ["build_projected.py", "--model", "tiny",
                                     "--rotation-test", "--output-norm", mode, "--threads", "2"])
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_path(str(Path(__file__).with_name("build_projected.py")), run_name="__main__")
    assert exit_info.value.code is None
    result = json.loads((tmp_path / "tiny" / "projected_rotation.json").read_text())
    kl = result["rotation_kl"]["all"]["mean"]
    assert abs(kl) < 2e-6 if mode == "matrix" else kl > 1e-5
