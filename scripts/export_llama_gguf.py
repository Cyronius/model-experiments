"""Export a v2 Llama titler checkpoint to GGUF with a Unigram ("t5") tokenizer (2026-10-19).

llama.cpp's regular SentencePiece path merges greedily and would not match our Unigram tokenizer, so
this reuses its converter but swaps the vocabulary step for the Unigram one (true Viterbi search, plus
an NFKC table). Out-of-vocabulary characters become <unk> there; byte tokens stay in the vocab for callers
that tokenize themselves.

    uv run python scripts/export_llama_gguf.py --model model/v3-title-8L --out model/v3-title-8L.f16.gguf --outtype f16
"""

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

LLAMACPP = Path(os.environ.get("LLAMACPP_DIR", "C:/code/llama.cpp"))
sys.path.insert(0, str(LLAMACPP))
sys.path.insert(1, str(LLAMACPP / "gguf-py"))
sys.path.insert(2, str(Path(__file__).parent))

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"


def unigram_set_vocab(self) -> None:
    from sentencepiece import sentencepiece_model_pb2 as model

    proto = model.ModelProto()
    proto.ParseFromString((self.dir_model / "tokenizer.model").read_bytes())
    tokens = [p.piece.encode("utf-8") for p in proto.pieces]
    scores = [p.score for p in proto.pieces]
    types = [p.type for p in proto.pieces]  # sentencepiece piece types match the GGUF token types

    w = self.gguf_writer
    w.add_tokenizer_model("t5")
    w.add_tokenizer_pre("default")
    w.add_token_list(tokens)
    w.add_token_scores(scores)
    w.add_token_types(types)
    w.add_add_space_prefix(proto.normalizer_spec.add_dummy_prefix)
    w.add_remove_extra_whitespaces(proto.normalizer_spec.remove_extra_whitespaces)
    if proto.normalizer_spec.precompiled_charsmap:
        w.add_precompiled_charsmap(proto.normalizer_spec.precompiled_charsmap)
    ids = {p.piece: i for i, p in enumerate(proto.pieces)}
    w.add_pad_token_id(ids["<pad>"])
    w.add_bos_token_id(ids["<s>"])
    w.add_eos_token_id(ids["</s>"])
    w.add_unk_token_id(ids["<unk>"])
    w.add_sep_token_id(ids["<sep>"])
    w.add_add_bos_token(True)
    w.add_add_eos_token(False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="checkpoint dir with config.json, model.safetensors, tokenizer.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--outtype", default="f16", choices=["f32", "f16", "bf16", "q8_0", "tq1_0", "tq2_0"])
    args = ap.parse_args()

    from make_spiece import write_spiece

    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp) / "model"
        shutil.copytree(args.model, stage)
        write_spiece(str(stage / "tokenizer.json"), str(stage / "tokenizer.model"), with_charsmap=True)

        import convert_hf_to_gguf as converter
        from conversion import get_model_class

        get_model_class("LlamaForCausalLM").set_vocab = unigram_set_vocab
        sys.argv = ["convert_hf_to_gguf.py", str(stage), "--outfile", str(Path(args.out).resolve()), "--outtype", args.outtype]
        converter.main()


if __name__ == "__main__":
    main()
