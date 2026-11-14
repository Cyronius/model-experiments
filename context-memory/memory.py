"""Encode documents once; inject saved continuous memory into Qwen attention.

The Encoder and Adapter are the authors' pinned LCLM implementation.  This
module deliberately does not implement a new compressor or train new weights.
"""
from __future__ import annotations

import copy
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "latent-context/0.6b-4b-LCLM-4x"
MODEL_REVISION = "d45629704bbc8de96d4b847c0098bf7ccebb3aa4"
CODE_URL = "https://github.com/LeonLixyz/LCLM"
CODE_REVISION = "45de65087d910c7d6e5e478e0930a716c34d4d10"
INSTRUCTION = (
    "Answer the question using the supplied notes. Give only the short answer, "
    "with no explanation. Preserve exact numbers, units, identifiers, and codes. "
    "If the requested fact is absent from the notes, answer UNKNOWN.\n\nNotes:\n"
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def setup(threads: int = 8):
    torch.set_num_threads(threads)
    torch.manual_seed(7316)


def encode_banks(checkpoint: Path, upstream: Path, banks: dict[str, str], out: Path):
    """The query and gold answers are never arguments to this function."""
    sys.path.insert(0, str(upstream.resolve()))
    from latent_context import Adapter, Encoder

    config = json.loads((checkpoint / "model_config.json").read_text())
    if config["adapter_type"] != "mlp":
        raise ValueError("This pilot expects the released MLP bridge")
    enc_tokenizer = AutoTokenizer.from_pretrained(checkpoint / "encoder", local_files_only=True)
    enc_model = AutoModel.from_pretrained(
        checkpoint / "encoder", dtype=torch.bfloat16,
        attn_implementation="sdpa", local_files_only=True,
    ).eval()
    encoder = Encoder(
        compression_ratio=config["compression_ratio"], max_length=32768,
        encoder_model=enc_model, encoder_tokenizer=enc_tokenizer,
        train_encoder=False, max_encode_batch_size=2,
        pooling=config["pooling"], encoder_mask_type=config["encoder_mask_type"],
        encoder_window_size=config["encoder_window_size"],
        boundary_overlap=config["boundary_overlap"],
    ).eval()
    decoder_config = json.loads((checkpoint / "decoder" / "config.json").read_text())
    adapter = Adapter(encoder.embedding_dim, decoder_config["hidden_size"]).to(torch.bfloat16).eval()
    adapter.load_state_dict(load_file(checkpoint / "adapter" / "adapter.safetensors"), strict=True)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
        "upstream_code_revision": CODE_REVISION, "config": config,
        "dtype": "bfloat16", "banks": {},
    }
    for name, document in banks.items():
        ids = enc_tokenizer.encode(document, add_special_tokens=False)
        start = time.perf_counter()
        with torch.inference_mode():
            memory = adapter(encoder([ids])[0]).contiguous()
        if not torch.isfinite(memory).all():
            raise ValueError("Non-finite document memory")
        path = out / f"{name}.safetensors"
        save_file({"memory": memory}, path, metadata={
            "model_revision": MODEL_REVISION, "source_sha256": sha256(document.encode()),
        })
        manifest["banks"][name] = {
            "source_sha256": sha256(document.encode()), "source_tokens": len(ids),
            "memory_slots": memory.shape[0], "hidden_size": memory.shape[1],
            "memory_bytes": memory.numel() * memory.element_size(),
            "file_sha256": sha256(path.read_bytes()),
            "encode_seconds": time.perf_counter() - start,
        }
        print("ENCODED", name, json.dumps(manifest["banks"][name]), flush=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_decoder(checkpoint: Path):
    directory = checkpoint / "decoder" if (checkpoint / "decoder").is_dir() else checkpoint
    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        directory, dtype=torch.bfloat16,
        attn_implementation="sdpa", local_files_only=True,
    ).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tokenizer


def text_embeddings(model, tokenizer, text: str):
    ids = torch.tensor([tokenizer.encode(text, add_special_tokens=False)], dtype=torch.long)
    return model.get_input_embeddings()(ids)


def make_prefix(model, tokenizer, *, memory=None, document=None):
    """Only one representation of the documents can reach the decoder."""
    if memory is not None and document is not None:
        raise ValueError("Choose encoded memory or original text")
    start = "<|im_start|>user\n" + INSTRUCTION
    if memory is None:
        content = ("<|memory_start|>" + document + "<|memory_end|>"
                   if document is not None else "[No notes supplied.]")
        return text_embeddings(model, tokenizer, start + content)
    if memory.ndim != 2 or memory.shape[1] != model.config.hidden_size:
        raise ValueError("Memory width does not match the decoder")
    if not torch.isfinite(memory).all():
        raise ValueError("Non-finite document memory")
    # Same interface used by LCLM.generate: replace memory placeholder input
    # embeddings with projected continuous vectors. All decoder attention layers
    # see those positions. There is no document text in this input.
    left = text_embeddings(model, tokenizer, start + "<|memory_start|>")
    right = text_embeddings(model, tokenizer, "<|memory_end|>")
    return torch.cat([left, memory.to(left).unsqueeze(0), right], dim=1)


def prefill(model, prefix):
    with torch.inference_mode():
        result = model(inputs_embeds=prefix, use_cache=True, logits_to_keep=1)
    return result.past_key_values


def question_suffix(question: str):
    # The released SFT decoder can emit <think> even though its base model is
    # named Instruct. Close that channel explicitly for a short-answer test;
    # otherwise the output cap can end in reasoning rather than an answer.
    return (f"\n\nQuestion: {question}\n<|im_end|>\n<|im_start|>assistant\n"
            "<think>\n\n</think>\n\n")


def answer(model, tokenizer, prefix_cache, question: str, max_tokens: int = 40):
    """Decode only the question from an independently cloned prefix cache.

    Gold answers are intentionally absent from this interface. Cloning prevents
    questions and previous generated answers from contaminating later trials.
    """
    suffix = question_suffix(question)
    ids = torch.tensor([tokenizer.encode(suffix, add_special_tokens=False)], dtype=torch.long)
    cache = copy.deepcopy(prefix_cache)
    original_length = prefix_cache.get_seq_length()
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    generated = []
    start = time.perf_counter()
    with torch.inference_mode():
        for _ in range(max_tokens):
            result = model(input_ids=ids, past_key_values=cache, use_cache=True, logits_to_keep=1)
            cache = result.past_key_values
            token = int(result.logits[0, -1].argmax())
            if token in eos:
                break
            generated.append(token)
            decoded = tokenizer.decode(generated, skip_special_tokens=True)
            if "\n" in decoded.strip():
                break
            ids = torch.tensor([[token]], dtype=torch.long)
    if prefix_cache.get_seq_length() != original_length:
        raise AssertionError("Shared prefix cache was mutated")
    return {
        "answer": tokenizer.decode(generated, skip_special_tokens=True).strip(),
        "generated_tokens": len(generated), "seconds": time.perf_counter() - start,
        "question_tokens": len(tokenizer.encode(suffix, add_special_tokens=False)),
    }
