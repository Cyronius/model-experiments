import copy
import json

import pytest
import torch

from audit_rotation import collect_basis as original_basis, norm_gain
from projection_models import install_output_matrix, projected_output_matrix, project_residual_width, load_causal_model
from scale_probe import parameter_plan, collect_basis, collect_scales, count_config, validation_rows
from test_projection_models import tiny_model


@pytest.mark.parametrize("width", [16, 32])
def test_exact_readout_preserves_in_subspace_states(width):
    from probe_scale_readout import readout_inputs
    rng = torch.Generator().manual_seed(813)
    q = torch.linalg.qr(torch.randn(32, 32, generator=rng)).Q[:, :width]
    h = torch.randn(7, width, generator=rng) @ q.T
    gain = torch.linspace(.2, 2., 32)
    weight = torch.randn(41, 32, generator=rng)
    scale = (32 / width) ** .5
    original, retained, exact = readout_inputs(h, gain, q, scale, 0.)
    z = h @ q
    zn = z * torch.rsqrt(z.square().mean(-1, keepdim=True))
    folded = zn @ (((weight * gain) @ q) * scale).T
    torch.testing.assert_close(folded, exact @ weight.T, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(exact, original, atol=2e-6, rtol=2e-6)
    if width == 32:
        torch.testing.assert_close(retained, original, atol=3e-6, rtol=3e-6)
    else:
        assert (retained - original).abs().max() > .1


@pytest.mark.parametrize("architecture", ["qwen", "mistral", "llama"])
@pytest.mark.parametrize("width", [20, 32])
def test_chunked_bf16_projection_matches_fp32_then_rounding(tmp_path, architecture, width):
    low = tiny_model(architecture, False).to(torch.bfloat16)
    high = copy.deepcopy(low).float()
    q = torch.linalg.qr(torch.randn(32, 32, dtype=torch.float64)).Q[:, :width].float()
    scales = [1.03, 1.1, .97, 1.12, 1.04]
    project_residual_width(high, q, scales)
    high.to(torch.bfloat16)
    project_residual_width(low, q, scales, chunk_size=7)
    assert count_config(low.config) == sum(p.numel() for p in low.parameters())
    for name, weight in low.named_parameters():
        other = dict(high.named_parameters())[name]
        # Chunked GEMMs can land on opposite sides of a BF16 rounding midpoint.
        # Permit at most one BF16 relative ULP, not arbitrary output drift.
        torch.testing.assert_close(weight, other, atol=1e-9, rtol=2**-7)
    install_output_matrix(low, torch.eye(width, dtype=torch.bfloat16))
    low.save_pretrained(tmp_path)
    reloaded = load_causal_model(tmp_path, dtype=torch.bfloat16)
    ids = torch.randint(0, 101, (1, 12))
    torch.testing.assert_close(low(ids, use_cache=False).logits, reloaded(ids, use_cache=False).logits,
                               atol=.01, rtol=.01)


@pytest.mark.parametrize("architecture", ["qwen", "mistral"])
@pytest.mark.parametrize("tied", [True, False])
def test_predicted_parameters_equal_actual_projected_model(architecture, tied):
    model = tiny_model(architecture, tied)
    plan = parameter_plan(model.config, alignment=1)
    for arm in plan["arms"]:
        student = copy.deepcopy(model)
        p = torch.eye(32)[:, :arm["width"]]
        project_residual_width(student, p, [1.] * 5, chunk_size=7)
        install_output_matrix(student, torch.eye(arm["width"]))
        assert sum(p.numel() for p in student.parameters()) == arm["parameters"]
        assert arm["retained_fraction"] <= arm["target_fraction"]
        assert student.config.head_dim == model.config.head_dim


def test_gpu_ready_calibration_matches_original_covariance_and_scales():
    from probe_output_fix import rms_scales
    model = tiny_model("mistral", False)
    ids = [torch.randint(0, 101, (1, 20)), torch.randint(0, 101, (1, 17))]
    rows = [{"input_ids": x[0].tolist()} for x in ids]
    q, info = collect_basis(model, rows)
    original, _ = original_basis(model, ids)
    torch.testing.assert_close(q[:, :16] @ q[:, :16].T,
                               original[:, :16] @ original[:, :16].T, atol=2e-5, rtol=2e-5)
    scales = collect_scales(model, rows, q, [16, 24])
    for k in (16, 24):
        expected = rms_scales(model, ids, q[:, :k])
        torch.testing.assert_close(torch.tensor(scales[str(k)]), torch.tensor(expected), atol=2e-6, rtol=2e-6)
    assert info["tokens_per_site"] == [37] * 5


def test_frozen_data_is_disjoint_and_complete():
    from pathlib import Path
    from scale_data import load_tokenizer, encoding_digest
    from healing_data import encode_complete
    root = Path(__file__).parent.parent
    path = root / "manifold-compression/results/scale-probe-20261024/data.json"
    if not path.exists():
        pytest.skip("Frozen data has not been prepared")
    data = json.loads(path.read_text())
    fit = {r["messages"][0]["content"].strip().casefold() for r in data["calibration"]}
    held = {r["messages"][0]["content"].strip().casefold() for r in data["validation"] + data["generation_eval"]}
    assert not fit & held
    assert len(data["generation_eval"]) == 72
    paths = {"qwen27": root / "target/manifold-scale/qwen27", "mistral24": root / "target/manifold-scale/mistral24",
             "qwen2": root / "models/manifold-audit/Qwen3.5-2B"}
    for name, path in paths.items():
        if not path.exists():
            continue
        tok = load_tokenizer(path)
        assert encoding_digest(data, tok) == data["encoded_sha256"][name]
        for row in data["calibration"] + data["validation"]:
            enc = encode_complete(tok, row["messages"], data["max_length"])
            assert enc and enc["targets"][-1] == tok.eos_token_id
        for row in validation_rows(data, tok):
            assert row["positions"][-1] == len(row["input_ids"]) - 2
            assert len(row["positions"]) <= 16
