"""Capture exact Q4 features and test the external Nanbeige MTP head.

Run `capture` under `dibs run gpu port-8400`; run `features` under
`dibs run gpu` (or `dibs run cpu` with `--device cpu`). The reference head
and its cache implementation are pinned in data/mtp-1101.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

import capture
import run_capture
from init_head import GGMLQuantizationType, GGUFReader, GGUF_PY

sys.path.insert(0, str(GGUF_PY))
from gguf import dequantize  # noqa: E402

ROOT = Path(__file__).resolve().parent
REFERENCE = ROOT / "data/mtp-1101"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sequence_path(directory: Path, index: int) -> Path:
    return directory / f"{index:04d}.npz"


def chat_prompts(args: argparse.Namespace) -> None:
    from jinja2 import Environment

    reader = GGUFReader(str(run_capture.MODEL), mode="r")
    source = reader.fields["tokenizer.chat_template"].contents()
    environment = Environment(autoescape=False)

    def raise_exception(message):
        raise ValueError(message)

    environment.globals["raise_exception"] = raise_exception
    template = environment.from_string(source)
    rows = list(capture.prompts(args.source))[args.start:args.start + args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as output:
        for row in rows:
            rendered = template.render(messages=[{"role": "user", "content": row["prompt"]}],
                                       tools=None, add_generation_prompt=True,
                                       enable_thinking=True)
            if not rendered.endswith("<think>\n"):
                raise ValueError("unexpected chat template generation prefix")
            output.write(json.dumps({"id": row["id"], "prompt": rendered,
                                     "source_prompt": row["prompt"]}, ensure_ascii=False) + "\n")
    print(f"rendered {len(rows)} chat prompts to {args.output}", flush=True)


def capture_sequences(args: argparse.Namespace) -> None:
    rows = [json.loads(line) for line in args.generated.read_text().splitlines() if line]
    args.output.mkdir(parents=True, exist_ok=True)
    with run_capture.server(True, args.output / "embedding-server.log", args.server, args.threads):
        for index, row in enumerate(rows):
            destination = sequence_path(args.output, index)
            if destination.exists():
                with np.load(destination) as old:
                    generated = row["generated_tokens"]
                    if (old["row_id"].item() != row["id"]
                            or len(old["ids"]) != len(old["hidden"])
                            or (generated and old["ids"][-len(generated):].tolist() != generated)):
                        raise ValueError(f"existing sequence disagrees with row {index}")
                continue
            prompt_ids = capture.post(run_capture.URL, "/tokenize", {
                "content": row["prompt"], "add_special": True,
            })["tokens"]
            ids = np.asarray(prompt_ids + row["generated_tokens"], dtype=np.int32)
            result = capture.post(run_capture.URL, "/embeddings", {"content": ids.tolist()})
            hidden = np.asarray(result[0]["embedding"], dtype=np.float32)
            if hidden.shape != (len(ids), 7168) or not np.isfinite(hidden).all():
                raise ValueError(f"invalid hidden sequence for row {index}: {hidden.shape}")
            np.savez(destination, ids=ids, hidden=hidden.astype(np.float16),
                     prompt_length=np.int32(len(prompt_ids)), row_id=np.asarray(row["id"]))
            print(f"captured {index + 1}/{len(rows)} {row['id']} tokens={len(ids)}", flush=True)
    manifest = {"generated": str(args.generated),
                "generated_sha256": hashlib.sha256(args.generated.read_bytes()).hexdigest(),
                "rows": len(rows), "target": str(run_capture.MODEL)}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def embedding_rows(raw: np.ndarray, qtype: GGMLQuantizationType, ids: np.ndarray) -> np.ndarray:
    unique, inverse = np.unique(ids, return_inverse=True)
    decoded = dequantize(raw[unique], qtype)
    if decoded.shape != (len(unique), 7168):
        raise ValueError(f"decoded embeddings have shape {decoded.shape}")
    return decoded[inverse]


def head_states(head, hidden, embeddings, device, chunk: int):
    """Causal reference head with the same RoPE positions as speed_test.py."""
    import torch
    from speed_test import head_kv_empty, head_layer_pass

    kv = head_kv_empty()
    states = []
    with torch.inference_mode():
        for start in range(0, len(hidden), chunk):
            stop = min(start + chunk, len(hidden))
            h = torch.from_numpy(hidden[start:stop].astype(np.float32)).to(device=device, dtype=torch.bfloat16)[None]
            e = torch.from_numpy(embeddings[start:stop].astype(np.float32)).to(device=device, dtype=torch.bfloat16)[None]
            x = head.eh_proj(torch.cat([head.enorm(e), head.hnorm(h)], dim=-1))
            pos = torch.arange(start, stop, device=device)
            out, k, v = head_layer_pass(head, x, pos, kv)
            kv["k"] = torch.cat([kv["k"], k], dim=2) if kv["k"] is not None else k
            kv["v"] = torch.cat([kv["v"], v], dim=2) if kv["v"] is not None else v
            # Preserve BF16 values exactly for the output-head compatibility probe.
            states.append(head.shared_head_norm(out)[0].float().cpu().numpy())
    return np.concatenate(states, axis=0)


def extract_features(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoConfig

    sys.path.insert(0, str(REFERENCE))
    from mtp_head import load_mtp_head

    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    config = AutoConfig.from_pretrained(REFERENCE / "base", local_files_only=True)
    head = load_mtp_head(REFERENCE / "mtp_head.safetensors", config,
                         dtype=torch.bfloat16, device=device)
    reader = GGUFReader(str(run_capture.MODEL), mode="r")
    tensor = next(t for t in reader.tensors if t.name == "token_embd.weight")
    if tensor.tensor_type != GGMLQuantizationType.Q4_0:
        raise ValueError(f"expected Q4_0 target embeddings, got {tensor.tensor_type}")
    raw = tensor.data.reshape(166912, 4032)
    base, mtp, seeds, futures, owners, anchors = [], [], [], [], [], []
    skipped_short = []
    files = sorted(args.sequences.glob("[0-9][0-9][0-9][0-9].npz"))
    if not files:
        raise ValueError(f"no sequences in {args.sequences}")
    manifest = json.loads((args.sequences / "manifest.json").read_text())
    if len(files) != manifest["rows"] or [path.stem for path in files] != [f"{i:04d}" for i in range(len(files))]:
        raise ValueError("sequence capture is incomplete")
    with torch.no_grad():
        for index, path in enumerate(files):
            with np.load(path) as row:
                ids = row["ids"]
                hidden = row["hidden"]
                prompt_length = int(row["prompt_length"])
            if len(ids) != len(hidden):
                raise ValueError(f"invalid sequence {path}")
            if len(ids) < prompt_length + 7:
                skipped_short.append(path.stem)
                print(f"skipped short continuation {path.stem}", flush=True)
                continue
            embeddings = embedding_rows(raw, tensor.tensor_type, ids[1:])
            z = head_states(head, hidden[:-1], embeddings, device, args.chunk)
            selected = np.arange(prompt_length - 1, len(ids) - 7, args.stride)
            base.append(hidden[selected])
            mtp.append(z[selected])
            seeds.append(ids[selected + 1])
            futures.append(np.stack([ids[selected + j] for j in range(2, 8)], axis=1))
            owners.extend([int(path.stem)] * len(selected))
            anchors.extend(selected.tolist())
            print(f"features {index + 1}/{len(files)} {path.stem} blocks={len(selected)}", flush=True)
    if not base:
        raise ValueError("no continuation has enough tokens for a draft block")
    common = {"seed": np.concatenate(seeds), "future": np.concatenate(futures),
              "owner": np.asarray(owners, np.int32), "anchor": np.asarray(anchors, np.int32)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output.with_name(args.output.stem + "-base.npz"),
             hidden=np.concatenate(base), **common)
    np.savez(args.output.with_name(args.output.stem + "-mtp.npz"),
             hidden=np.concatenate(mtp), **common)
    manifest = {"sequences": str(args.sequences), "head": str(REFERENCE / "mtp_head.safetensors"),
                "head_sha256": file_sha256(REFERENCE / "mtp_head.safetensors"),
                "target": str(run_capture.MODEL), "stride": args.stride,
                "rows": len(files), "blocks": len(owners), "skipped_short_rows": skipped_short,
                "device": str(device)}
    args.output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    chat = sub.add_parser("chat-prompts")
    chat.add_argument("--source", type=Path, required=True)
    chat.add_argument("--output", type=Path, required=True)
    chat.add_argument("--start", type=int, default=0)
    chat.add_argument("--limit", type=int, required=True)
    cap = sub.add_parser("capture")
    cap.add_argument("--generated", type=Path, required=True)
    cap.add_argument("--output", type=Path, required=True)
    cap.add_argument("--server", type=Path, default=run_capture.SERVER)
    cap.add_argument("--threads", type=int, default=16)
    feat = sub.add_parser("features")
    feat.add_argument("--sequences", type=Path, required=True)
    feat.add_argument("--output", type=Path, required=True)
    feat.add_argument("--stride", type=int, default=1)
    feat.add_argument("--chunk", type=int, default=128)
    feat.add_argument("--threads", type=int, default=16)
    feat.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()
    if args.command == "chat-prompts":
        chat_prompts(args)
    elif args.command == "capture":
        capture_sequences(args)
    else:
        if args.stride < 1 or args.chunk < 1:
            parser.error("stride and chunk must be positive")
        extract_features(args)


if __name__ == "__main__":
    main()
