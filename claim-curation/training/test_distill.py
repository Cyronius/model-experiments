import json
from pathlib import Path

import pytest

from evaluate import compare
from teacher import mechanically_validate, request, align_evidence
from train_lora import load_rows


def test_source_quote_does_not_make_wrong_statement_safe():
    document = "Kernel ash worked only with compiler 1.2. Compiler 1.3 was not tested."
    correct = {"claims": [{"statement": "Kernel ash worked only with compiler 1.2.",
                           "quote": "Kernel ash worked only with compiler 1.2.",
                           "conditions": ["only with compiler 1.2"]}]}
    assert not mechanically_validate(correct, document)
    missing = {"claims": [{"statement": "Kernel ash always works.",
                          "quote": correct["claims"][0]["quote"], "conditions": []}]}
    # Citation validity is intentionally distinct from qualifier retention.
    assert not mechanically_validate(missing, document)
    assert compare(correct, missing, document)["reference_qualifiers_covered"] == 0


def test_teacher_rejects_invented_evidence():
    c = {"claims": [{"statement": "x", "quote": "made up", "conditions": []}]}
    assert mechanically_validate(c, "The actual source.")
    c["claims"][0]["quote"] = "actual source"
    c["claims"][0]["conditions"] = ["unmentioned hardware"]
    assert mechanically_validate(c, "The actual source.")


def test_remote_paid_teacher_cannot_be_called(tmp_path):
    with pytest.raises(ValueError, match="local HTTP"):
        request("https://example.com/v1", "model", "revision", "instruction", "source", tmp_path, 100)


def test_split_leakage_blocks_training(tmp_path):
    rows = [{"source_key": "same-doc", "split": "train", "training_eligible": True},
            {"source_key": "same-doc", "split": "test", "training_eligible": True}]
    p = tmp_path / "labels.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ValueError, match="leaks"):
        load_rows(p)


def test_nonaccepted_supervision_is_excluded(tmp_path):
    rows = [{"source_key": "a", "split": "train", "training_eligible": False},
            {"source_key": "b", "split": "train", "training_eligible": True}]
    p = tmp_path / "labels.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert len(load_rows(p)) == 1


def test_renamed_duplicate_document_cannot_cross_splits(tmp_path):
    rows = [{"source_key": "a", "source_sha256": "identical", "split": "train"},
            {"source_key": "b", "source_sha256": "identical", "split": "test"}]
    p = tmp_path / "labels.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ValueError, match="content leaks"):
        load_rows(p)


def test_evidence_alignment_preserves_original_words_and_logs_changes():
    source = "Kernel ash took\n9 ms on Birch."
    candidate = {"claims": [{"statement": "ash took 9 ms", "quote": "Kernel ash took 9 ms on Birch.", "conditions": ["on Birch"]}]}
    aligned, changes = align_evidence(candidate, source)
    assert not mechanically_validate(aligned, source)
    assert changes[0]["proposed"] == candidate["claims"][0]["quote"]
    assert candidate["claims"][0]["quote"] != aligned["claims"][0]["quote"]
    candidate["claims"][0]["quote"] = "Kernel ash took 8 ms on Birch."
    aligned, changes = align_evidence(candidate, source)
    assert mechanically_validate(aligned, source)
    assert not changes


def test_ambiguous_whitespace_alignment_is_rejected():
    source = "a\nb then a\t b"
    c = {"claims": [{"statement": "a b", "quote": "a b", "conditions": []}]}
    aligned, changes = align_evidence(c, source)
    assert mechanically_validate(aligned, source)
    assert not changes
