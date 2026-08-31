"""Synthesize a SentencePiece .model file from our HF `tokenizers`-library
Unigram tokenizer.json. llama.cpp's GGUF converter (conversion/t5.py) hard-
requires a real SentencePiece protobuf for T5 models -- there is no
tokenizer.json fallback -- so this is the bridge for `gguf-my-model`-style
conversion. Only used for the llama.cpp/GGUF export path; every other
runtime (Python ORT, Node, browser) uses tokenizer.json directly and is
unaffected.

Usage:
    python scripts/make_spiece.py --tokenizer model/tokenizer/tokenizer.json --out model/titler-v12-339k/tokenizer.model
"""

import argparse
import json
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
from sentencepiece import sentencepiece_model_pb2 as spm_pb2  # noqa: E402

PAD, EOS, UNK = "<pad>", "</s>", "<unk>"
CONTROL, UNKNOWN, NORMAL = 3, 2, 1  # spm_pb2 PieceType values


def write_spiece(tokenizer_json: str, out_path: str) -> int:
    """Returns the vocab size written."""
    d = json.load(open(tokenizer_json, encoding="utf-8"))
    vocab = d["model"]["vocab"]  # [[piece, score], ...] -- id == list index
    pre = d.get("pre_tokenizer", {})
    add_prefix = pre.get("prepend_scheme") == "always" or pre.get("add_prefix_space", False)

    m = spm_pb2.ModelProto()
    m.trainer_spec.model_type = 1  # UNIGRAM
    m.trainer_spec.vocab_size = len(vocab)
    m.normalizer_spec.add_dummy_prefix = add_prefix
    m.normalizer_spec.remove_extra_whitespaces = True
    # no precompiled_charsmap: our NFKC + whitespace-collapse normalization already
    # happened at train time (baked into which pieces exist); llama.cpp's own
    # runtime tokenizer won't re-apply NFKC to new input, so exotic Unicode input
    # (fullwidth forms, combining marks) may tokenize slightly differently than
    # the Python/JS runtimes. Negligible for ordinary text.

    UNUSED = 5
    for i, (piece, score) in enumerate(vocab):
        p = m.pieces.add()
        if "\x00" in piece:
            # sentencepiece's proto loader rejects literal NUL in a piece (we picked
            # one up from a binary/garbled paste during tokenizer training). Rename
            # only -- id/score/row must stay put to match the embedding matrix.
            p.piece = f"<0xNUL-{i}>"
            p.score = score
            p.type = UNUSED
            continue
        p.piece = piece
        p.score = score
        p.type = {PAD: CONTROL, EOS: CONTROL, UNK: UNKNOWN}.get(piece, NORMAL)

    with open(out_path, "wb") as f:
        f.write(m.SerializeToString())
    return len(vocab)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True, help="path to tokenizer.json")
    ap.add_argument("--out", required=True, help="output path, e.g. .../tokenizer.model")
    args = ap.parse_args()

    n = write_spiece(args.tokenizer, args.out)
    print(f"wrote {args.out}: {n} pieces")


if __name__ == "__main__":
    main()
