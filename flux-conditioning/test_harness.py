import json
import sys
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file

import harness as h


@pytest.fixture(autouse=True)
def small_geometry(monkeypatch):
    monkeypatch.setattr(h, "POSITIONS", 4)
    monkeypatch.setattr(h, "WIDTH", 3)
    monkeypatch.setattr(h, "TAP_WIDTH", 2)
    torch.set_num_threads(1)


def test_projection_keeps_tap_order_and_padding():
    states = [torch.full((1, 4, 2), float(k)) for k in range(28)]
    weight = torch.tensor([[1., 0, 0, 0, 0, 0], [0., 0, 1, 0, 0, 0], [0., 0, 0, 0, 1, 0]])
    actual = h.project(states, weight, torch.tensor([1., 2, 3]))
    assert torch.equal(actual, torch.tensor([[[10., 20., 30.]] * 4]))
    with pytest.raises(ValueError, match="geometry"):
        h.project(states, weight[:, :-1])


class Tokenizer:
    padding_side = truncation_side = "right"
    pad_token_id = 151643

    def __init__(self, tokens):
        self.tokens = tokens

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
        return h.chat_text(messages[0]["content"])

    def __call__(self, text, **kwargs):
        if "return_tensors" not in kwargs:
            return {"input_ids": self.tokens}
        real = min(len(self.tokens), h.POSITIONS)
        return {"input_ids": torch.tensor([self.tokens[:real] + [self.pad_token_id] * (h.POSITIONS - real)]),
                "attention_mask": torch.tensor([[1] * real + [0] * (h.POSITIONS - real)])}


@pytest.mark.parametrize("tokens", [[1, 2], [1, 2, 3, 4], [1, 2, 3, 4, 5]])
def test_tokenizer_padding_and_truncation(tokens):
    tensors, length = h.prepare_inputs(Tokenizer(tokens), "test")
    assert length == len(tokens)
    assert tensors["input_ids"].shape == (1, 4)
    assert int(tensors["attention_mask"].sum()) == min(len(tokens), 4)


def test_reject_changed_tokenizer_semantics():
    tokenizer = Tokenizer([1, 2])
    tokenizer.padding_side = "left"
    with pytest.raises(ValueError, match="right padding"):
        h.prepare_inputs(tokenizer, "test")


def test_metrics_known_errors_and_zeros():
    target = torch.tensor([[3., 4.]])
    result = h.metrics(2 * target, target)
    assert result["relative_l2"] == 1
    assert result["cosine"] == pytest.approx(1)
    assert result["rmse"] == pytest.approx(12.5 ** 0.5)
    assert h.metrics(torch.zeros(1, 2), torch.zeros(1, 2))["relative_l2"] is None
    assert h.metrics(torch.zeros(0, 2), torch.zeros(0, 2)) == {"positions": 0}
    with pytest.raises(ValueError, match="Nonfinite"):
        h.metrics(torch.tensor([[float("nan"), 1.]]), target)


def make_target(tmp_path):
    folder = tmp_path / "targets"
    folder.mkdir()
    case = {"id": "a", "prompt": "one cup", "split": "test", "seed": 17}
    path = folder / "a.safetensors"
    tensors = {"projected_conditioning": torch.ones(4, 3), "text_ids": h.text_ids(),
               "input_ids": torch.tensor([1, 2, 151643, 151643]), "attention_mask": torch.tensor([1, 1, 0, 0])}
    save_file(tensors, str(path), metadata={"protocol": h.PROTOCOL, "case_sha256": h.case_hash(case)})
    record = {**case, "file": path.name, "sha256": h.sha256(path), "case_sha256": h.case_hash(case),
              "untruncated_tokens": 2, "truncated": False}
    (folder / "manifest.json").write_text(json.dumps({"protocol": h.PROTOCOL, "cases": [record]}))
    return folder, path


def test_end_to_end_artifact_identity_and_padding_diagnostic(tmp_path):
    targets, target = make_target(tmp_path)
    prediction = load_file(str(target))["projected_conditioning"]
    prediction[2:] *= 2
    h.save_prediction(target, prediction, tmp_path / "pred" / "a.safetensors")
    saved = load_file(str(tmp_path / "pred" / "a.safetensors"))
    source = load_file(str(target))
    assert torch.equal(saved["input_ids"], source["input_ids"])
    assert torch.equal(saved["attention_mask"], source["attention_mask"])
    report = h.evaluate(targets, tmp_path / "pred")
    row = report["cases"][0]
    assert row["real"]["relative_l2"] == 0
    assert row["padding"]["relative_l2"] == 1
    assert row["all"]["relative_l2"] == pytest.approx(0.5 ** 0.5)
    assert "unmeasured" in report["quality_gate"]
    with pytest.raises(FileExistsError):
        h.save_prediction(target, prediction, tmp_path / "pred" / "a.safetensors")


def test_wrong_prediction_binding_rejected(tmp_path):
    targets, target = make_target(tmp_path)
    pred = tmp_path / "pred"
    pred.mkdir()
    save_file({"projected_conditioning": torch.ones(4, 3), "text_ids": h.text_ids()}, str(pred / "a.safetensors"),
              metadata={"protocol": h.PROTOCOL, "reference_sha256": "wrong"})
    with pytest.raises(ValueError, match="bound"):
        h.evaluate(targets, pred)


def test_changed_target_rejected(tmp_path):
    targets, target = make_target(tmp_path)
    h.save_prediction(target, torch.ones(4, 3), tmp_path / "pred" / "a.safetensors")
    target.write_bytes(target.read_bytes() + b"altered")
    with pytest.raises(ValueError, match="provenance"):
        h.evaluate(targets, tmp_path / "pred")


def test_wrong_position_ids_rejected():
    with pytest.raises(ValueError, match="position"):
        h.validate_tensors({"projected_conditioning": torch.ones(4, 3), "text_ids": h.text_ids().flip(0)}, prediction=True)


def test_dataset_split_leakage_rejected(tmp_path):
    cases = tmp_path / "cases.jsonl"
    cases.write_text('\n'.join(json.dumps({"id": name, "prompt": "same", "split": split, "seed": 1})
                               for name, split in [("a", "train"), ("b", "test")]))
    with pytest.raises(ValueError, match="unique strings"):
        h.read_cases(cases)


def test_projection_loading_with_bias_and_index(tmp_path):
    folder = tmp_path / "transformer"
    folder.mkdir()
    tensors = {"context_embedder.weight": torch.ones(3, 6), "context_embedder.bias": torch.arange(3.)}
    save_file(tensors, str(folder / "shard.safetensors"))
    (folder / "diffusion_pytorch_model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: "shard.safetensors" for name in tensors}}))
    actual, files = h.projection_files(tmp_path)
    assert torch.equal(actual["context_embedder.bias"], tensors["context_embedder.bias"])
    assert len(files) == 2


def test_export_refuses_existing_output_before_model_load(tmp_path):
    cases = tmp_path / "cases.jsonl"
    cases.write_text(json.dumps({"id": "a", "prompt": "cup", "split": "test", "seed": 1}))
    with pytest.raises(FileExistsError):
        h.export(SimpleNamespace(cases=cases, output=tmp_path, model=tmp_path / "absent"))


def test_export_manifest_and_evaluation_with_fixture_teacher(tmp_path, monkeypatch):
    model_dir = tmp_path / "model"
    for name in ("transformer", "text_encoder", "tokenizer"):
        (model_dir / name).mkdir(parents=True)
    (model_dir / "text_encoder" / "config.json").write_text(json.dumps(
        {"model_type": "qwen3", "hidden_size": 2, "num_hidden_layers": 36}))
    save_file({"fixture": torch.ones(1)}, str(model_dir / "text_encoder" / "model.safetensors"))
    save_file({"context_embedder.weight": torch.ones(3, 6)},
              str(model_dir / "transformer" / "diffusion_pytorch_model.safetensors"))
    (model_dir / "tokenizer" / "tokenizer.json").write_text("{}")

    class Model:
        def to(self, device):
            return self

        def eval(self):
            return self

        def __call__(self, **kwargs):
            assert kwargs["use_cache"] is False and kwargs["output_hidden_states"] is True
            return SimpleNamespace(hidden_states=[torch.full((1, 4, 2), float(i)) for i in range(37)])

    def load_model(path, **kwargs):
        assert kwargs["local_files_only"] is True
        assert kwargs["output_loading_info"] is True
        return Model(), {"missing_keys": set(), "unexpected_keys": {"lm_head.weight"},
                         "mismatched_keys": set(), "error_msgs": []}

    def load_tokenizer(path, **kwargs):
        assert kwargs["local_files_only"] is True
        return Tokenizer([1, 2])

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        Qwen3Model=SimpleNamespace(from_pretrained=load_model),
        AutoTokenizer=SimpleNamespace(from_pretrained=load_tokenizer)))
    cases = tmp_path / "cases.jsonl"
    cases.write_text(json.dumps({"id": "a", "prompt": "cup", "split": "test", "seed": 1}))
    output = tmp_path / "export"
    h.export(SimpleNamespace(cases=cases, output=output, model=model_dir, dtype="bf16", device="cpu"))
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["taps"] == [9, 18, 27]
    assert manifest["loading_info"]["unexpected_keys"] == ["lm_head.weight"]
    assert len(manifest["source_sha256"]) == 4
    assert manifest["cases"][0]["truncated"] is False
    target = output / "a.safetensors"
    conditioning = load_file(str(target))["projected_conditioning"]
    assert torch.equal(conditioning, torch.full((4, 3), 108.))
    h.save_prediction(target, conditioning, tmp_path / "pred" / "a.safetensors")
    assert h.evaluate(output, tmp_path / "pred")["cases"][0]["all"]["relative_l2"] == 0


def test_export_with_real_tiny_qwen3_checkpoint(tmp_path, monkeypatch):
    """Exercise installed Transformers loading/hidden-state/manifest APIs, without teacher weights."""
    from transformers import AutoTokenizer, Qwen3Config, Qwen3Model

    model_dir = tmp_path / "tiny"
    config = Qwen3Config(hidden_size=2, intermediate_size=4, num_hidden_layers=28,
                         num_attention_heads=1, num_key_value_heads=1, head_dim=2, vocab_size=8)
    Qwen3Model(config).save_pretrained(model_dir / "text_encoder")
    (model_dir / "transformer").mkdir()
    (model_dir / "tokenizer").mkdir()
    save_file({"context_embedder.weight": torch.ones(3, 6)},
              str(model_dir / "transformer" / "diffusion_pytorch_model.safetensors"))
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *args, **kwargs: Tokenizer([1, 2, 3, 4]))
    cases = tmp_path / "cases.jsonl"
    cases.write_text(json.dumps({"id": "tiny", "prompt": "cup", "split": "test", "seed": 1}))
    output = tmp_path / "export"
    h.export(SimpleNamespace(cases=cases, output=output, model=model_dir, dtype="bf16", device="cpu"))
    manifest = json.loads((output / "manifest.json").read_text())
    assert not manifest["loading_info"]["missing_keys"]
    assert load_file(str(output / "tiny.safetensors"))["projected_conditioning"].dtype == torch.bfloat16


def test_boundary_fixture_token_count_is_enforced(tmp_path, monkeypatch):
    from transformers import AutoTokenizer

    model_dir = tmp_path / "model"
    (model_dir / "text_encoder").mkdir(parents=True)
    (model_dir / "text_encoder" / "config.json").write_text(json.dumps(
        {"model_type": "qwen3", "hidden_size": 2, "num_hidden_layers": 36}))
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *args, **kwargs: Tokenizer([1, 2]))
    cases = tmp_path / "cases.jsonl"
    cases.write_text(json.dumps({"id": "a", "prompt": "cup", "split": "test", "seed": 1,
                                 "expected_untruncated_tokens": 3}))
    with pytest.raises(ValueError, match="fixture drifted"):
        h.export(SimpleNamespace(cases=cases, output=tmp_path / "out", model=model_dir))


def test_pinned_reference_extraction_and_hash_check(tmp_path):
    from compare_reference import reference_methods

    path = tmp_path / "reference.py"
    path.write_text('''raise RuntimeError("module imports must not run")
class Flux2KleinPipeline:
    @staticmethod
    def _get_qwen3_prompt_embeds(x: UnknownAnnotation):
        return torch.tensor(x)
    @staticmethod
    def _prepare_text_ids(x):
        return x + 1
''')
    methods = reference_methods(path, h.sha256(path))
    assert int(methods["_get_qwen3_prompt_embeds"](3)) == 3
    assert methods["_prepare_text_ids"](3) == 4
    with pytest.raises(ValueError, match="hash mismatch"):
        reference_methods(path, "incorrect")
