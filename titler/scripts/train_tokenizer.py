"""Train the 8k Unigram tokenizer on prompt+title text and save it in
fast-tokenizer format (tokenizer.json) that both transformers (Python) and
transformers.js can load.

Usage:
    python scripts/train_tokenizer.py --input data/spike_pairs.jsonl --output model/tokenizer --vocab-size 8192
"""

import argparse
import json
from pathlib import Path

from tokenizers import Regex, Tokenizer, decoders, normalizers, pre_tokenizers, trainers
from tokenizers.models import Unigram
from tokenizers.processors import TemplateProcessing

PAD, EOS, UNK = "<pad>", "</s>", "<unk>"


CHUNK = 1000  # the Unigram trainer NaNs on very long single sentences; chunking is quality-neutral


def iter_text(paths: list[Path]):
    for path in paths:
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                for text in (rec.get("prompt"), rec.get("title")):
                    if not text:
                        continue
                    for i in range(0, len(text), CHUNK):
                        yield text[i : i + CHUNK]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", nargs="+", required=True, help="JSONL files with prompt/title fields")
    ap.add_argument("--output", required=True, help="Output directory for tokenizer files")
    ap.add_argument("--vocab-size", type=int, default=8192)
    args = ap.parse_args()

    tokenizer = Tokenizer(Unigram())
    tokenizer.normalizer = normalizers.Sequence(
        [normalizers.NFKC(), normalizers.Replace(Regex(r"\s+"), " ")]
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()
    tokenizer.decoder = decoders.Metaspace()

    trainer = trainers.UnigramTrainer(
        vocab_size=args.vocab_size,
        special_tokens=[PAD, EOS, UNK],
        unk_token=UNK,
    )
    tokenizer.train_from_iterator(iter_text([Path(p) for p in args.input]), trainer=trainer)

    # T5 convention: every encoded sequence ends with </s>
    eos_id = tokenizer.token_to_id(EOS)
    tokenizer.post_processor = TemplateProcessing(
        single="$A </s>",
        pair="$A </s> $B </s>",
        special_tokens=[(EOS, eos_id)],
    )

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(out / "tokenizer.json"))

    (out / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "PreTrainedTokenizerFast",
                "model_input_names": ["input_ids", "attention_mask"],
                "model_max_length": 512,
                "pad_token": PAD,
                "eos_token": EOS,
                "unk_token": UNK,
                "clean_up_tokenization_spaces": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (out / "special_tokens_map.json").write_text(
        json.dumps({"pad_token": PAD, "eos_token": EOS, "unk_token": UNK}, indent=2),
        encoding="utf-8",
    )

    actual = tokenizer.get_vocab_size()
    ids = {t: tokenizer.token_to_id(t) for t in (PAD, EOS, UNK)}
    print(f"vocab size: {actual} (requested {args.vocab_size})")
    print(f"special token ids: {ids}")
    assert ids[PAD] == 0 and ids[EOS] == 1 and ids[UNK] == 2, "special tokens must be ids 0/1/2 (T5 convention)"


if __name__ == "__main__":
    main()
