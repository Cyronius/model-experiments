"""Titles from a v2 Llama checkpoint for the fixed eval set: beam search, or sample N and pick one (2026-10-19).

    python scripts/15_gen_v2_titles.py --model model/v2-title-5L --out data/eval/titles_v2_5L.json
    python scripts/15_gen_v2_titles.py --model model/v3-title-8L --samples 8 --pick mbr --out data/eval/titles_v3_mbr8.json
"""

import argparse
import json
import re
from collections import Counter
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


def words(title: str) -> list[str]:
    return re.findall(r"\w+", title.lower())


def word_f1(a: list[str], b: list[str]) -> float:
    common = sum((Counter(a) & Counter(b)).values())
    return 2 * common / (len(a) + len(b)) if common else 0.0


def pick(cands: list[str], logps: list[float], how: str) -> str:
    """mbr: most words shared with the other candidates (ties go to the likelier); rerank: best mean log-prob per token."""
    idx = [i for i, c in enumerate(cands) if c] or list(range(len(cands)))
    if how == "rerank":
        return cands[max(idx, key=lambda i: logps[i])]
    ws = {i: words(cands[i]) for i in idx}
    return cands[max(idx, key=lambda i: (sum(word_f1(ws[i], ws[j]) for j in idx if j != i), logps[i]))]


def sample_and_pick(model, x, procs, args, tok) -> str:
    out = model.generate(x, attention_mask=torch.ones_like(x), max_new_tokens=MAX_NEW, do_sample=True, temperature=1.0,
                         top_k=0, top_p=1.0, epsilon_cutoff=args.epsilon, num_return_sequences=args.samples,
                         eos_token_id=EOS, pad_token_id=PAD, logits_processor=procs,
                         return_dict_in_generate=True, output_logits=True)
    gen = out.sequences[:, x.size(1):]
    # Raw logits, before the repeat ban and penalty, so the score is the model's own likelihood.
    lp = torch.stack(out.logits, dim=1).log_softmax(-1).gather(-1, gen.unsqueeze(-1)).squeeze(-1)
    is_eos = (gen == EOS).int()
    keep = (is_eos.cumsum(1) - is_eos) == 0
    logps = ((lp * keep).sum(1) / keep.sum(1)).tolist()
    cands = [clean_title(tok.decode(g.tolist(), skip_special_tokens=True)) for g in gen]
    return pick(cands, logps, args.pick)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--eval-set", default="data/eval/eval_set.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--beams", type=int, default=4)
    ap.add_argument("--filler-penalty", type=float, default=3.0)
    ap.add_argument("--no-repeat", type=int, default=1, help="1 = ban repeated token pairs inside the title")
    ap.add_argument("--samples", type=int, default=0, help="sample this many titles and pick one; 0 = beam search")
    ap.add_argument("--pick", choices=["mbr", "rerank"], default="mbr")
    ap.add_argument("--epsilon", type=float, default=0.02, help="drop tokens below this probability when sampling")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)

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
            procs = LogitsProcessorList(([NoRepeatInTitle(len(ids), word_starts)] if args.no_repeat else []) + ([PenalizeTokens(filler, args.filler_penalty)] if args.filler_penalty else []))
            if args.samples:
                titles[r["id"]] = sample_and_pick(model, x, procs, args, tok)
                continue
            gen = model.generate(x, attention_mask=torch.ones_like(x), max_new_tokens=MAX_NEW, do_sample=False,
                                 eos_token_id=EOS, pad_token_id=PAD, num_beams=args.beams, logits_processor=procs)
            titles[r["id"]] = clean_title(tok.decode(gen[0, len(ids):].tolist(), skip_special_tokens=True))
    Path(args.out).write_text(json.dumps(titles, ensure_ascii=False), encoding="utf-8")
    print(f"{len(titles)} titles -> {args.out}")


if __name__ == "__main__":
    main()
