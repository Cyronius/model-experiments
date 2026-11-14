"""End-to-end greedy tokens/sec: exact GGUF plain decode vs external MTP-1101.

This measures a GPU MTP speculative implementation, not the future XDNA2
microdrafter. Both arms use one llama.cpp model/context and identical prompts.
"""

from __future__ import annotations

import argparse
import ctypes as C
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from transformers import AutoConfig

from init_head import GGMLQuantizationType, GGUFReader, head_rows
from mtp_experiment import REFERENCE, embedding_rows, run_capture
from drafter import Drafter

sys.path.insert(0, str(REFERENCE))
from mtp_head import load_mtp_head  # noqa: E402
from speed_test import head_kv_empty, head_prefill_kv, head_step  # noqa: E402

ROOT = Path(__file__).resolve().parent
INT = C.c_int32
PTR_INT = C.POINTER(INT)
PTR_FLOAT = C.POINTER(C.c_float)


class Target:
    def __init__(self, library: Path, model: Path, context: int, batch: int,
                 threads: int, gpu_layers: int = 999):
        self.lib = C.CDLL(str(library.resolve()))
        signatures = {
            "bridge_open": ([C.c_char_p, INT, INT, INT, INT], C.c_void_p),
            "bridge_close": ([C.c_void_p], None),
            "bridge_vocab_size": ([C.c_void_p], INT),
            "bridge_hidden_size": ([C.c_void_p], INT),
            "bridge_eos": ([C.c_void_p], INT),
            "bridge_is_eog": ([C.c_void_p, INT], C.c_bool),
            "bridge_tokenize": ([C.c_void_p, C.c_char_p, INT, PTR_INT, INT], INT),
            "bridge_clear": ([C.c_void_p], None),
            "bridge_rollback": ([C.c_void_p, INT], C.c_bool),
            "bridge_decode": ([C.c_void_p, PTR_INT, INT, INT, C.c_bool], INT),
            "bridge_logits": ([C.c_void_p, INT], PTR_FLOAT),
            "bridge_hidden": ([C.c_void_p, INT], PTR_FLOAT),
        }
        for name, (arguments, result) in signatures.items():
            function = getattr(self.lib, name)
            function.argtypes, function.restype = arguments, result
        self.handle = self.lib.bridge_open(str(model).encode(), gpu_layers, context, batch, threads)
        if not self.handle:
            raise RuntimeError("could not load GGUF target with llama.cpp bridge")
        self.vocab = self.lib.bridge_vocab_size(self.handle)
        self.hidden_size = self.lib.bridge_hidden_size(self.handle)

    def close(self):
        if self.handle:
            self.lib.bridge_close(self.handle)
            self.handle = None

    def clear(self):
        self.lib.bridge_clear(self.handle)

    def rollback(self, position: int):
        if not self.lib.bridge_rollback(self.handle, position):
            raise RuntimeError(f"target KV rollback failed from position {position}")

    def is_eog(self, token: int) -> bool:
        return self.lib.bridge_is_eog(self.handle, token)

    def tokenize(self, text: str) -> np.ndarray:
        raw = text.encode()
        needed = self.lib.bridge_tokenize(self.handle, raw, len(raw), None, 0)
        needed = -needed if needed < 0 else needed
        result = np.empty(needed, np.int32)
        actual = self.lib.bridge_tokenize(self.handle, raw, len(raw), result.ctypes.data_as(PTR_INT), needed)
        if actual != needed:
            raise ValueError(f"tokenizer returned {actual} tokens, expected {needed}")
        return result

    def decode(self, tokens, position: int, all_outputs: bool):
        values = np.ascontiguousarray(tokens, dtype=np.int32)
        status = self.lib.bridge_decode(self.handle, values.ctypes.data_as(PTR_INT), len(values),
                                        position, all_outputs)
        if status != 0:
            raise RuntimeError(f"llama.cpp decode failed with status {status} at position {position}")
        indices = range(len(values)) if all_outputs else [len(values) - 1]
        logits, hidden = [], []
        for i in indices:
            lp = self.lib.bridge_logits(self.handle, i)
            hp = self.lib.bridge_hidden(self.handle, i)
            if not lp or not hp:
                raise RuntimeError(f"missing logits or hidden state at output {i}")
            logits.append(np.ctypeslib.as_array(lp, shape=(self.vocab,)).copy())
            hidden.append(np.ctypeslib.as_array(hp, shape=(self.hidden_size,)).copy())
        return np.stack(logits), np.stack(hidden)


def load_tables(model: Path, device):
    reader = GGUFReader(str(model), mode="r")
    token = next(t for t in reader.tensors if t.name == "token_embd.weight")
    output = next(t for t in reader.tensors if t.name == "output.weight")
    if token.tensor_type != GGMLQuantizationType.Q4_0 or output.tensor_type != GGMLQuantizationType.Q8_0:
        raise ValueError("unexpected GGUF embedding or output-head quantization")
    vocab, hidden = 166912, 7168
    token_raw = token.data.reshape(vocab, hidden // 32 * 18)
    output_raw = output.data.reshape(vocab, hidden // 32, 34)
    embed_table = np.empty((vocab, hidden), np.float16)
    head_table = np.empty((vocab, hidden), np.float16)
    for start in range(0, vocab, 1024):
        stop = min(start + 1024, vocab)
        embed_table[start:stop] = embedding_rows(token_raw, token.tensor_type,
                                                 np.arange(start, stop, dtype=np.int32)).astype(np.float16)
        head_table[start:stop] = head_rows(output_raw, start, stop, hidden).astype(np.float16)
        if start % (1024 * 40) == 0:
            print(f"dequantized {stop}/{vocab} vocabulary rows", flush=True)
    embeddings = torch.from_numpy(embed_table).to(device)
    lm_head = torch.from_numpy(head_table).to(device)
    del embed_table, head_table
    return embeddings, lm_head


def gpu_embedding(table, tokens):
    ids = torch.as_tensor(tokens, dtype=torch.long, device=table.device)
    return F.embedding(ids, table).to(torch.bfloat16)


def prefill(target: Target, prompt: np.ndarray, all_outputs: bool):
    target.clear()
    return target.decode(prompt, 0, all_outputs)


def plain(target: Target, prompt: np.ndarray, max_new: int):
    started = time.perf_counter()
    logits, hidden = prefill(target, prompt, False)
    prefill_seconds = time.perf_counter() - started
    current_logits = logits[-1]
    emitted = []
    started = time.perf_counter()
    for step in range(max_new):
        token = int(current_logits.argmax())
        emitted.append(token)
        if target.is_eog(token) or len(emitted) == max_new:
            break
        logits, hidden = target.decode([token], len(prompt) + step, False)
        current_logits = logits[-1]
    decode_seconds = time.perf_counter() - started
    return {"tokens": emitted, "prefill_seconds": prefill_seconds,
            "decode_seconds": decode_seconds, "accepted": 0, "drafted": 0,
            "target_passes": len(emitted) - 1}


@torch.inference_mode()
def speculative(target: Target, prompt: np.ndarray, max_new: int, head, embeddings, lm_head):
    started = time.perf_counter()
    logits, hidden = prefill(target, prompt, True)
    h = hidden[-1]
    current_logits = logits[-1]
    prior_h = torch.from_numpy(hidden[:-1].astype(np.float32)).to(head.eh_proj.weight.device,
                                                                    dtype=torch.bfloat16)[None]
    prior_e = gpu_embedding(embeddings, prompt[1:])[None]
    kv = head_prefill_kv(head, prior_h, prior_e, kv=head_kv_empty())
    if head.eh_proj.weight.device.type == "cuda":
        torch.cuda.synchronize()
    prefill_seconds = time.perf_counter() - started
    emitted, accepted, drafted, passes = [], 0, 0, 0
    position = len(prompt) - 1
    started = time.perf_counter()
    while len(emitted) < max_new:
        seed = int(current_logits.argmax())
        emitted.append(seed)
        if target.is_eog(seed) or len(emitted) >= max_new:
            break
        h_tensor = torch.from_numpy(h).to(head.eh_proj.weight.device, dtype=torch.bfloat16)[None]
        z, kv = head_step(head, h_tensor, gpu_embedding(embeddings, [seed]),
                          torch.tensor([position], device=h_tensor.device), kv)
        draft_logits = F.linear(z.to(lm_head.dtype), lm_head)
        draft = int(draft_logits[0].argmax())
        drafted += 1
        verification, states = target.decode([seed, draft], position + 1, True)
        passes += 1
        actual = int(verification[0].argmax())
        if actual == draft:
            accepted += 1
            emitted.append(draft)
            if target.is_eog(draft) or len(emitted) >= max_new:
                break
            # Carry the verified pair into the MTP cache before the next draft.
            _, kv = head_step(head,
                              torch.from_numpy(states[0]).to(h_tensor.device, dtype=torch.bfloat16)[None],
                              gpu_embedding(embeddings, [draft]),
                              torch.tensor([position + 1], device=h_tensor.device), kv)
            h, current_logits, position = states[1], verification[1], position + 2
        else:
            target.rollback(position + 2)
            h, current_logits, position = states[0], verification[0], position + 1
    decode_seconds = time.perf_counter() - started
    return {"tokens": emitted, "prefill_seconds": prefill_seconds,
            "decode_seconds": decode_seconds, "accepted": accepted, "drafted": drafted,
            "target_passes": passes}


@torch.inference_mode()
def tiny_speculative(target: Target, prompt: np.ndarray, max_new: int, tiny: Drafter,
                     passes_per_draft: int, head=None, embeddings=None):
    use_mtp = head is not None
    started = time.perf_counter()
    logits, hidden = prefill(target, prompt, use_mtp)
    h, current_logits = hidden[-1], logits[-1]
    kv = None
    if use_mtp:
        prior_h = torch.from_numpy(hidden[:-1].astype(np.float32)).to(
            head.eh_proj.weight.device, dtype=torch.bfloat16)[None]
        prior_e = gpu_embedding(embeddings, prompt[1:])[None]
        kv = head_prefill_kv(head, prior_h, prior_e, kv=head_kv_empty())
        if head.eh_proj.weight.device.type == "cuda":
            torch.cuda.synchronize()
    prefill_seconds = time.perf_counter() - started
    emitted, accepted, drafted, passes = [], 0, 0, 0
    position = len(prompt) - 1
    started = time.perf_counter()
    while len(emitted) < max_new:
        seed = int(current_logits.argmax())
        emitted.append(seed)
        if target.is_eog(seed) or len(emitted) >= max_new:
            break
        if use_mtp:
            h_tensor = torch.from_numpy(h).to(head.eh_proj.weight.device,
                                               dtype=torch.bfloat16)[None]
            condition, kv = head_step(head, h_tensor, gpu_embedding(embeddings, [seed]),
                                      torch.tensor([position], device=h_tensor.device), kv)
        else:
            condition = torch.from_numpy(h).to(tiny.factors.device)[None]
        candidates = tiny.draft(
            condition.float(), torch.tensor([seed], device=tiny.factors.device),
            passes=passes_per_draft)[0, 1:].cpu().numpy().astype(np.int32)
        candidates = candidates[:max_new - len(emitted)]
        drafted += len(candidates)
        verification, states = target.decode(np.concatenate(([seed], candidates)), position + 1, True)
        passes += 1
        matched = 0
        for j, candidate in enumerate(candidates):
            if int(verification[j].argmax()) != int(candidate):
                break
            emitted.append(int(candidate))
            accepted += 1
            matched += 1
            if target.is_eog(int(candidate)) or len(emitted) >= max_new:
                break
        if target.is_eog(emitted[-1]) or len(emitted) >= max_new:
            break
        if matched < len(candidates):
            target.rollback(position + 2 + matched)
        if use_mtp:
            for j in range(matched):
                _, kv = head_step(head,
                                  torch.from_numpy(states[j]).to(h_tensor.device,
                                                                dtype=torch.bfloat16)[None],
                                  gpu_embedding(embeddings, [int(candidates[j])]),
                                  torch.tensor([position + 1 + j], device=h_tensor.device), kv)
        h, current_logits, position = (states[matched], verification[matched],
                                       position + 1 + matched)
    decode_seconds = time.perf_counter() - started
    return {"tokens": emitted, "prefill_seconds": prefill_seconds,
            "decode_seconds": decode_seconds, "accepted": accepted, "drafted": drafted,
            "target_passes": passes}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, default=ROOT / "llama_bridge.so")
    parser.add_argument("--model", type=Path, default=run_capture.MODEL)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--max-new", type=int, default=64)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--gpu-layers", type=int, default=999)
    parser.add_argument("--warmup-tokens", type=int, default=8)
    parser.add_argument("--tiny-checkpoint", type=Path)
    parser.add_argument("--tiny-passes", type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    target = Target(args.library, args.model, args.context, args.batch,
                    args.threads, args.gpu_layers)
    if (target.vocab, target.hidden_size) != (166912, 7168):
        raise ValueError("GGUF bridge model dimensions are wrong")
    config = AutoConfig.from_pretrained(REFERENCE / "base", local_files_only=True)
    head = load_mtp_head(REFERENCE / "mtp_head.safetensors", config,
                         dtype=torch.bfloat16, device=device)
    embeddings, lm_head = load_tables(args.model, device)
    tiny = None
    if args.tiny_checkpoint:
        checkpoint = torch.load(args.tiny_checkpoint, map_location="cpu", weights_only=True)
        tiny = Drafter().to(device).eval()
        tiny.load_state_dict(checkpoint["model"], strict=True)
    rows = [json.loads(line) for line in args.prompts.read_text().splitlines() if line][
        args.start:args.start + args.limit]
    if not rows:
        raise ValueError("no benchmark prompts")
    if args.warmup_tokens:
        warmup = target.tokenize(rows[0]["prompt"])
        plain(target, warmup, args.warmup_tokens)
        speculative(target, warmup, args.warmup_tokens, head, embeddings, lm_head)
        if tiny is not None:
            tiny_speculative(target, warmup, args.warmup_tokens, tiny,
                             args.tiny_passes, head, embeddings)
        if device.type == "cuda":
            torch.cuda.synchronize()
    cases = []
    for index, row in enumerate(rows):
        prompt = target.tokenize(row["prompt"])
        if len(prompt) + args.max_new > args.context:
            raise ValueError(f"prompt {row['id']} exceeds context")
        if index % 2:
            spec = speculative(target, prompt, args.max_new, head, embeddings, lm_head)
            base = plain(target, prompt, args.max_new)
        else:
            base = plain(target, prompt, args.max_new)
            spec = speculative(target, prompt, args.max_new, head, embeddings, lm_head)
        tiny_result = (tiny_speculative(target, prompt, args.max_new, tiny,
                                        args.tiny_passes, head, embeddings)
                       if tiny is not None else None)
        equal = base["tokens"] == spec["tokens"]
        case = {"id": row["id"], "prompt_tokens": len(prompt), "output_identical": equal,
                "plain_tokens": base["tokens"], "mtp_tokens": spec["tokens"],
                "plain": {"output_tokens": len(base["tokens"]),
                          **{key: value for key, value in base.items() if key != "tokens"}},
                "mtp": {"output_tokens": len(spec["tokens"]),
                        **{key: value for key, value in spec.items() if key != "tokens"}},
                "plain_output_sha256": hashlib.sha256(np.asarray(base["tokens"], np.int32).tobytes()).hexdigest(),
                "mtp_output_sha256": hashlib.sha256(np.asarray(spec["tokens"], np.int32).tobytes()).hexdigest()}
        cases.append(case)
        if tiny_result is not None:
            case["tiny_tokens"] = tiny_result["tokens"]
            case["mtp_tiny"] = {"output_tokens": len(tiny_result["tokens"]),
                                **{key: value for key, value in tiny_result.items() if key != "tokens"}}
            case["tiny_output_identical"] = base["tokens"] == tiny_result["tokens"]
            case["tiny_output_sha256"] = hashlib.sha256(
                np.asarray(tiny_result["tokens"], np.int32).tobytes()).hexdigest()
        print(f"{index + 1}/{len(rows)} {row['id']} plain={len(base['tokens']) / base['decode_seconds']:.2f} "
              f"MTP={len(spec['tokens']) / spec['decode_seconds']:.2f} tok/s "
              f"accept={spec['accepted']}/{spec['drafted']} identical={equal}"
              + (f" tiny={len(tiny_result['tokens']) / tiny_result['decode_seconds']:.2f} tok/s "
                 f"tiny_accept={tiny_result['accepted']}/{tiny_result['drafted']} "
                 f"tiny_identical={case['tiny_output_identical']}" if tiny_result is not None else ""),
              flush=True)
    summary = {}
    for arm in (("plain", "mtp", "mtp_tiny") if tiny is not None else ("plain", "mtp")):
        count = sum(case[arm]["output_tokens"] for case in cases)
        decode_time = sum(case[arm]["decode_seconds"] for case in cases)
        total_time = decode_time + sum(case[arm]["prefill_seconds"] for case in cases)
        summary[arm] = {"tokens": count, "decode_seconds": decode_time,
                        "decode_tokens_per_second": count / decode_time,
                        "request_tokens_per_second": count / total_time,
                        "target_passes": sum(case[arm]["target_passes"] for case in cases)}
    summary["mtp"]["draft_acceptance"] = (sum(c["mtp"]["accepted"] for c in cases) /
                                            max(sum(c["mtp"]["drafted"] for c in cases), 1))
    summary["speed_ratio_decode"] = summary["mtp"]["decode_tokens_per_second"] / summary["plain"]["decode_tokens_per_second"]
    summary["output_identical_cases"] = sum(c["output_identical"] for c in cases)
    if tiny is not None:
        summary["mtp_tiny"]["draft_acceptance"] = (sum(c["mtp_tiny"]["accepted"] for c in cases) /
                                                     max(sum(c["mtp_tiny"]["drafted"] for c in cases), 1))
        summary["tiny_speed_ratio_decode"] = (summary["mtp_tiny"]["decode_tokens_per_second"] /
                                               summary["plain"]["decode_tokens_per_second"])
        summary["tiny_output_identical_cases"] = sum(c["tiny_output_identical"] for c in cases)
    report = {"target": str(args.model), "head": str(REFERENCE / "mtp_head.safetensors"),
              "method": "external MTP, optional tiny drafter, exact GGUF verifier; no NPU runtime",
              "draft_device": args.device, "target_gpu_layers_requested": args.gpu_layers,
              "library": str(args.library),
              "tiny_checkpoint": str(args.tiny_checkpoint) if tiny is not None else None,
              "max_new": args.max_new, "cases": cases, "summary": summary}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    target.close()


if __name__ == "__main__":
    main()
