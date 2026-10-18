"""Greedy titles from a v2 Llama checkpoint for the fixed eval set (2026-10-17).

    python scripts/15_gen_v2_titles.py --model model/v2-title-5L --out data/eval/titles_v2_5L.json
"""

import argparse
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer
from transformers import LlamaForCausalLM, LogitsProcessor, LogitsProcessorList

from common.validation import clean_title

PAD, BOS, EOS, SEP = 0, 1, 2, 3
MAX_PROMPT, MAX_NEW = 256, 40


class PenalizeTokens(LogitsProcessor):
    """Soft push away from filler words ("request", "inquiry"); not a ban, so a real use can still win."""

    def __init__(self, ids: list[int], penalty: float):
        self.ids, self.penalty = ids, penalty

    def __call__(self, input_ids, scores):
        scores[:, self.ids] -= self.penalty
        return scores


class NoRepeatInTitle(LogitsProcessor):
    """Inside the title only (the stock n-gram ban also counts the prompt): no repeated token pair, and a
    one-token word is never followed by itself."""

    def __init__(self, prompt_len: int, word_starts: set[int]):
        self.prompt_len = prompt_len
        self.word_starts = word_starts

    def __call__(self, input_ids, scores):
        for row in range(input_ids.size(0)):
            gen = input_ids[row, self.prompt_len:].tolist()
            if gen and gen[-1] in self.word_starts:
                scores[row, gen[-1]] = float("-inf")
            if len(gen) >= 2:
                scores[row, [b for a, b in zip(gen, gen[1:]) if a == gen[-1]]] = float("-inf")
        return scores


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--eval-set", default="data/eval/eval_set.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--beams", type=int, default=2)
    ap.add_argument("--filler-penalty", type=float, default=3.0)
    ap.add_argument("--no-repeat", type=int, default=1, help="1 = ban repeated token pairs inside the title")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LlamaForCausalLM.from_pretrained(args.model).to(device).eval()
    tok = Tokenizer.from_file(f"{args.model}/tokenizer.json")
    word_starts = {i for i in range(tok.get_vocab_size()) if tok.id_to_token(i).startswith("▁")}
    filler = sorted({tok.encode(w).ids[0] for w in ("request", "inquiry")})
    rows = [json.loads(line) for line in Path(args.eval_set).open(encoding="utf-8") if line.strip()]
    titles = {}
    with torch.no_grad():
        for r in rows:
            ids = [BOS] + tok.encode(r["prompt"]).ids[:MAX_PROMPT] + [SEP]
            x = torch.tensor([ids], device=device)
            gen = model.generate(x, attention_mask=torch.ones_like(x), max_new_tokens=MAX_NEW, do_sample=False,
                                 eos_token_id=EOS, pad_token_id=PAD, num_beams=args.beams,
                                 logits_processor=LogitsProcessorList(([NoRepeatInTitle(len(ids), word_starts)] if args.no_repeat else []) + ([PenalizeTokens(filler, args.filler_penalty)] if args.filler_penalty else [])))
            titles[r["id"]] = clean_title(tok.decode(gen[0, len(ids):].tolist(), skip_special_tokens=True))
    Path(args.out).write_text(json.dumps(titles, ensure_ascii=False), encoding="utf-8")
    print(f"{len(titles)} titles -> {args.out}")


if __name__ == "__main__":
    main()
