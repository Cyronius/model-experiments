import json
from pathlib import Path

import pytest
import torch
from transformers import AutoTokenizer

from healing_data import encode_complete, pair, check_answer, generation_metrics, scoring_spec
from healing_pilot import legacy_initialization, make_batch, loss_parts
from projection_models import install_output_matrix
from test_projection_models import tiny_model


@pytest.mark.parametrize("name", ["Qwen3.5-2B", "SmolLM2-1.7B-Instruct"])
def test_complete_answer_mask_includes_end_and_excludes_context(name):
    path = Path("models/manifold-audit") / name
    if not path.exists():
        pytest.skip("Local tokenizer is required")
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    messages = pair("Remember the number 18.", "The number is 18.") + pair("Add 2. What is the result?", "20.")
    row = encode_complete(tok, messages)
    assert row["targets"][-1] == tok.eos_token_id
    assert row["positions"][0] == row["answer_start"] - 1
    assert row["positions"][-1] == len(row["input_ids"]) - 2
    assert tok.decode(row["targets"], skip_special_tokens=True).strip() == "20."
    assert encode_complete(tok, messages, max_length=len(row["input_ids"]) - 1) is None


def test_padding_and_loss_use_only_assistant_positions():
    rows = [{"input_ids": [4, 5, 6, 7], "positions": [1, 2], "targets": [6, 7]},
            {"input_ids": [1, 2, 3], "positions": [1], "targets": [3]}]
    ids, attention, bi, pos, targets = make_batch(rows, 0)
    assert attention.tolist() == [[1, 1, 1, 1], [1, 1, 1, 0]]
    assert ids[bi, pos + 1].tolist() == targets.tolist() == [6, 7, 3]
    logits = torch.randn(2, 4, 9, requires_grad=True)
    teacher = logits[bi, pos].detach().clone()
    kl, ce = loss_parts(logits[bi, pos], teacher, targets)
    assert abs(kl.item()) < 1e-6
    (kl + .1 * ce).backward()
    selected = torch.zeros(2, 4, dtype=torch.bool)
    selected[bi, pos] = True
    assert logits.grad[~selected].abs().sum() == 0
    assert logits.grad[selected].abs().sum() > 0


def test_legacy_reconstruction_preserves_starting_diagonal_function():
    model = tiny_model("qwen", True)
    original = torch.randn(32, 32)
    c = 1.07
    install_output_matrix(model, c * original)
    legacy_initialization(model, c)
    torch.testing.assert_close(model.model.norm.matrix, torch.diag(original.diag() / c))


def test_checks_reject_extra_text_and_report_repetition_separately():
    spec = {"kind": "json", "expected": {"city": "Oslo", "count": 3}}
    assert check_answer('{"count": 3, "city": "Oslo"}', spec)
    assert not check_answer('Here: {"count": 3, "city": "Oslo"}', spec)
    assert not check_answer("25", {"kind": "exact", "expected": "2500"})
    assert check_answer("Yes", scoring_spec({"id": "logic", "check": {"kind": "exact", "expected": "yes"}}))
    assert not check_answer("a7x-P4q", scoring_spec({"id": "copy", "check": {"kind": "exact", "expected": "A7x-P4q"}}))
    text = "This is a repeated sentence. " * 20
    result = generation_metrics(text, [4] * 100 + [2], 2)
    assert result["loop_flag"] and result["ended"]


def test_prepared_splits_and_ending_counts():
    path = Path("manifold-compression/results/healing-pilot-20261024/data.json")
    if not path.exists():
        pytest.skip("Prepare pilot data first")
    data = json.loads(path.read_text())
    first = lambda row: row["messages"][0]["content"].strip().casefold()
    train = {first(r) for r in data["train"]}
    val = {first(r) for r in data["validation"]}
    evaluation = {first(r) for r in data["generation_eval"]}
    assert not train & val and not train & evaluation
    assert len(train) == 64 and len(val) == 12
    for counts in data["counts"].values():
        assert counts["train"]["ending_tokens"] == 64
