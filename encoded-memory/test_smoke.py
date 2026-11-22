"""Offline checks of leakage boundaries, frozen-backbone gradients, and scoring."""
import copy

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from data import BM25, exact_match, load_findings, numeric_match, synthetic_records
from memory import EncoderConfig, MemoryEncoder, answer_loss, greedy_answer, make_batch, prompt_parts
from run import summarize


class TinyTokenizer:
    bos_token_id = 1
    eos_token_id = 2

    def apply_chat_template(self, messages, **kwargs):
        return "^User: " + messages[0]["content"] + "\nAssistant:"

    def __call__(self, text, **kwargs):
        class Encoded:
            input_ids = [3 + ord(char) % 90 for char in text]
        return Encoded()

    def decode(self, ids, **kwargs):
        return " ".join(map(str, ids))


def tiny_model():
    torch.manual_seed(7)
    model = LlamaForCausalLM(LlamaConfig(vocab_size=96, hidden_size=32, intermediate_size=64,
                                        num_hidden_layers=2, num_attention_heads=4,
                                        num_key_value_heads=2, max_position_embeddings=512,
                                        bos_token_id=1, eos_token_id=2, pad_token_id=0))
    return model.eval().requires_grad_(False)


def test_unseen_facts_and_retrieval():
    docs, questions = synthetic_records(12, 4, 5)
    ids = {split: {d["id"] for d in docs if d["split"] == split} for split in ["train", "dev", "test"]}
    assert not ids["train"] & (ids["dev"] | ids["test"])
    entities = [d["entity"] for d in docs]
    assert len(set(entities)) == len(entities)
    retriever = BM25(docs)
    assert all(retriever.retrieve(q["question"])[0] == q["doc_id"] for q in questions)


def test_source_dialect_preserves_scope(tmp_path):
    folder = tmp_path / "findings"
    folder.mkdir()
    (folder / "F-0001-test.md").write_text(
        "---\nid: F-0001\nevidence: measured\nstack: Tool 1; repeat: Windows\n"
        "last_verified: 2026-10-23\n---\n# Claim\n**Claim:** 64 B.\n"
        "**Where it was found:** lengthy source\n", encoding="utf-8")
    doc = load_findings(tmp_path)[0]
    assert doc["metadata"]["stack"] == "Tool 1; repeat: Windows"
    assert "64 B" in doc["text"] and "Evidence: measured" in doc["text"]
    assert "lengthy source" not in doc["text"]
    assert doc["split"] == "test"


def test_only_encoder_updates_and_target_mask_is_correct():
    torch.set_num_threads(2)
    model, tokenizer = tiny_model(), TinyTokenizer()
    before = copy.deepcopy(model.state_dict())
    encoder = MemoryEncoder(EncoderConfig(hidden_size=32, width=16, heads=4, slots=4, max_tokens=20))
    features = torch.randn(2, 12, 32)
    mask = torch.tensor([[1] * 12, [1] * 7 + [0] * 5], dtype=torch.bool)
    memory = encoder(features, mask)
    qs = [{"question": "What alignment?", "answers": ["64"]},
          {"question": "What is the queue capacity?", "answers": ["4"]}]
    batch = make_batch(model, tokenizer, qs, memory)
    for i, q in enumerate(qs):
        targets = batch["labels"][i][batch["labels"][i] != -100].tolist()
        assert targets == tokenizer(q["answers"][0]).input_ids + [tokenizer.eos_token_id]
    loss = answer_loss(model, batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert encoder.output.weight.grad is not None and encoder.output.weight.grad.abs().sum() > 0
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=0.01)
    optimizer.step()
    assert all(param.grad is None for param in model.parameters())
    assert all(torch.equal(value, before[key]) for key, value in model.state_dict().items())
    # Padding must not influence a document's encoded memory.
    encoder.eval()
    mutated = features.clone()
    mutated[1, 7:] = 1e5
    assert torch.allclose(encoder(features, mask)[1], encoder(mutated, mask)[1])


def test_cached_generation_accepts_soft_memory():
    torch.set_num_threads(2)
    model, tokenizer = tiny_model(), TinyTokenizer()
    # Inference must work without access to an answer key at all.
    question = {"question": "How many?"}
    memory = torch.randn(4, 32) * 0.02
    result = greedy_answer(model, tokenizer, question, memory=memory, max_new_tokens=3)
    assert isinstance(result, str)


def test_reference_slots_are_inside_the_user_message():
    model, tokenizer = tiny_model(), TinyTokenizer()
    question = {"question": "How many?", "answers": ["4"]}
    memory = torch.randn(4, 32)
    before, after = prompt_parts(tokenizer, question, memory_site="reference")
    batch = make_batch(model, tokenizer, [question], [memory], include_answers=False, memory_site="reference")
    assert len(before) > 1 and len(after) > 1
    assert torch.equal(batch["inputs_embeds"][0, len(before):len(before) + len(memory)], memory)
    assert len(batch["inputs_embeds"][0]) == len(before) + len(memory) + len(after)


def test_scoring_rejects_conflicting_numbers_and_pairs_recovery():
    assert exact_match("65,536.", ["65536"])
    assert not exact_match("64 bytes", ["64"])
    assert numeric_match("64 bytes", ["64"])
    assert not numeric_match("32 or 64 bytes", ["64"])
    assert not numeric_match("128", ["28"])
    rows = [{"id": "a", "domain": "synthetic", "condition": condition,
             "exact_match": correct, "numeric_match": correct,
             "elapsed_seconds": 1, "retrieval_hit": True}
            for condition, correct in [("none", False), ("text", True), ("oracle", True), ("retrieved", False)]]
    summary = summarize(rows)
    assert summary["paired_recovery"]["synthetic"] == {"eligible": 1, "oracle": 1.0, "retrieved": 0.0}
    assert summary["retrieval_by_domain"] == {"synthetic": {"hits": 1, "n": 1}}
