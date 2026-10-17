"""Train the 4,096-token Unigram tokenizer for the v2 Llama titler on data/v3 (2026-10-17).

Format the model sees: <s>{prompt}<sep>{title, no Title Case}</s>. Prints the round-trip and
token-length checks the plan asks for.
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from tokenizers import Regex, Tokenizer, decoders, normalizers, pre_tokenizers, trainers
from tokenizers.models import Unigram

from common.validation import clean_title, decase_title

SPECIALS = ["<pad>", "<s>", "</s>", "<sep>", "<unk>"]
MIN_CHAR_COUNT = 200
CHUNK = 1000  # the Unigram trainer NaNs on very long single sentences


def read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/v3")
    ap.add_argument("--output", default="model/tokenizer-v2")
    ap.add_argument("--vocab-size", type=int, default=4096)
    ap.add_argument("--prompt-rows", type=int, default=150_000, help="prompts sampled for training")
    args = ap.parse_args()

    train = read(Path(args.data) / "train.jsonl")
    test = read(Path(args.data) / "test.jsonl")
    misses = [r["title"] for r in train if clean_title(decase_title(r["title"]), clamp=False) != r["title"]]
    print(f"decase round trip: {len(train) - len(misses):,} of {len(train):,} lossless; examples: {ascii(misses[:5])}")

    def raw_texts():
        rng = random.Random(0)
        for r in rng.sample(train, min(args.prompt_rows, len(train))):
            for i in range(0, min(len(r["prompt"]), 3 * CHUNK), CHUNK):
                yield r["prompt"][i : i + CHUNK]
        for r in train:
            yield decase_title(r["title"])

    chars = Counter(c for t in raw_texts() for c in t)
    keep = {c for c, n in chars.items() if n >= MIN_CHAR_COUNT}  # rare scripts become <unk> instead of eating the vocab
    print(f"alphabet: {len(keep)} kept of {len(chars)} characters")

    def texts():
        return ("".join(c for c in t if c in keep) for t in raw_texts())

    tok = Tokenizer(Unigram())
    tok.normalizer = normalizers.Sequence([normalizers.NFKC(), normalizers.Replace(Regex(r"\s+"), " ")])
    tok.pre_tokenizer = pre_tokenizers.Metaspace()
    tok.decoder = decoders.Metaspace()
    tok.train_from_iterator(
        texts(), trainer=trainers.UnigramTrainer(vocab_size=args.vocab_size, special_tokens=SPECIALS, unk_token="<unk>")
    )
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    tok.save(str(out / "tokenizer.json"))

    unk = tok.token_to_id("<unk>")
    title_len = sorted(len(tok.encode(decase_title(r["title"])).ids) for r in test)
    prompt_len = sorted(len(tok.encode(r["prompt"]).ids) for r in test)
    n = len(test)
    unk_titles = sum(unk in tok.encode(decase_title(r["title"])).ids for r in test)
    print(f"vocab {tok.get_vocab_size()}; test titles: mean {sum(title_len) / n:.1f} tokens, p90 {title_len[int(n * .9)]}, "
          f"max {title_len[-1]}; titles with <unk>: {unk_titles}")
    print(f"test prompts: median {prompt_len[n // 2]} tokens, share <=256: {sum(x <= 256 for x in prompt_len) / n:.1%}")


if __name__ == "__main__":
    main()
