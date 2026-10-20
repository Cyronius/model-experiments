"""Golden files for the Rust core: tokenizer ids, clean_title output, and HF beam search on a toy bigram model (2026-10-19).

    uv run python scripts/24_make_app_golden.py
"""

import json
import random
import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer
from transformers import LlamaConfig, LlamaForCausalLM, LogitsProcessorList
from transformers.modeling_outputs import CausalLMOutputWithPast

sys.path.insert(0, str(Path(__file__).parent))
from common.validation import clean_title, decase_title  # noqa: E402

OUT = Path("app/core/tests/golden")
TOKENIZER = "model/v3-title-8L/tokenizer.json"
PAD, BOS, EOS, SEP = 0, 1, 2, 3

TOKENIZER_EDGE = [
    "", " ", "  hello  world  ", "hello\tworld\n\nnew line", "Ｆｕｌｌｗｉｄｔｈ ＡＢＣ １２３", "café naïve résumé", "é combining",
    "日本語のテキストです", "Привет, как дела?", "مرحبا بالعالم", "emoji 😀 test 🚀🚀", "a" * 300, "x y z", "​zero width",
    "tab\there", "print('hi')\n    return 1", "ß ǅ ﬁ ligature", "Ω≈ç√∫", "� replacement", "mixed English и русский 中文 mix",
]

CLEAN_EDGE = [
    "", "  ", "fixing my children’s bike", "“fixing my children’s bike”", "'don't panic' guide", "title: python vs rust",
    "iphone vs iPhone mRNA SQL v2 setup", "one two three four five six seven eight nine ten", "first line\nsecond line", "ends with period.",
    "trailing punctuation?!;:,", "**bold** words here", 'inner "quotes" here', "multiple   spaces   here", "a guide to the web", "the end of an era",
    "python for loops in the", "straße ǆ ǳ start", "ǳ", "ñandú viaje", "3d printing tips", "x y second", "`code` ticks", "it's a dog's life",
    "rock 'n' roll", "5-10 words: leaked", "vs", "of", "a", "A of B", "the quick brown fox jumps over the lazy dog today",
]


def tokenizer_cases(tok: Tokenizer) -> list[dict]:
    texts = [json.loads(l)["prompt"] for l in open("data/eval/eval_set.jsonl", encoding="utf-8") if l.strip()]
    rng = random.Random(0)
    test = [json.loads(l)["prompt"] for l in open("data/v5/test.jsonl", encoding="utf-8") if l.strip()]
    non_ascii = [t for t in test if any(ord(c) > 127 for c in t)]
    texts += rng.sample(non_ascii, min(300, len(non_ascii))) + rng.sample(test, 300)
    texts = [t[:500] for t in texts] + TOKENIZER_EDGE
    return [{"text": t, "ids": tok.encode(t).ids} for t in texts]


def clean_cases() -> list[dict]:
    val = [json.loads(l)["title"] for l in open("data/v5/val.jsonl", encoding="utf-8") if l.strip()]
    raws = [decase_title(t) for t in val[::3]] + CLEAN_EDGE
    return [{"raw": r, "clean": clean_title(r)} for r in raws]


class Bigram(LlamaForCausalLM):
    """Logits = table[token] + bias[position]; Rust rebuilds them from the same two tables.
    The position term breaks exact ties between cyclic paths, which float rounding would settle at random."""

    table: torch.Tensor
    bias: torch.Tensor

    def forward(self, input_ids=None, past_key_values=None, cache_position=None, **kwargs):
        pos = cache_position if cache_position is not None else torch.arange(input_ids.shape[1])
        return CausalLMOutputWithPast(logits=self.table[input_ids] + self.bias[pos], past_key_values=past_key_values)


def decode_cases() -> list[dict]:
    from importlib import import_module
    gen = import_module("15_gen_v2_titles")
    cases = []
    for seed in range(24):
        rng = torch.Generator().manual_seed(seed)
        vocab = 32
        cfg = LlamaConfig(vocab_size=vocab, hidden_size=8, intermediate_size=8, num_hidden_layers=1, num_attention_heads=1,
                          num_key_value_heads=1, bos_token_id=BOS, eos_token_id=EOS, pad_token_id=PAD)
        model = Bigram(cfg).eval()
        sharp = [0.5, 1.0, 2.0, 4.0][seed % 4]
        table = (torch.randn(vocab, vocab, generator=rng) * sharp).round(decimals=4)
        table[:, EOS] += [-2.0, 0.0, 1.0][seed % 3]
        table[:, PAD] = -20.0
        decimals = [round(float(x), 4) for x in table.flatten()]
        table = torch.tensor(decimals, dtype=torch.float32).view(vocab, vocab)
        model.table = table
        bias_dec = [round(float(x), 4) for x in (torch.randn(96, vocab, generator=rng) * 0.3).flatten()]
        model.bias = torch.tensor(bias_dec, dtype=torch.float32).view(96, vocab)
        beams = [1, 2, 4, 4, 6][seed % 5]
        max_new = [40, 12, 40, 6][seed % 4]
        prompt = [BOS] + [int(torch.randint(5, vocab, (1,), generator=rng)) for _ in range(5 + seed % 7)] + [SEP]
        word_starts = sorted(int(i) for i in torch.randperm(vocab - 5, generator=rng)[:8] + 5)
        filler = sorted(int(i) for i in torch.randperm(vocab - 5, generator=rng)[:2] + 5)
        use_ban, penalty = (seed % 3 != 0), (3.0 if seed % 2 == 0 else 0.0)
        procs = ([gen.NoRepeatInTitle(len(prompt), set(word_starts))] if use_ban else []) + ([gen.PenalizeTokens(filler, penalty)] if penalty else [])
        x = torch.tensor([prompt])
        with torch.no_grad():
            out = model.generate(x, attention_mask=torch.ones_like(x), max_new_tokens=max_new, do_sample=False, eos_token_id=EOS,
                                 pad_token_id=PAD, num_beams=beams, logits_processor=LogitsProcessorList(procs))
        ids = out[0, len(prompt):].tolist()
        if EOS in ids:
            ids = ids[:ids.index(EOS)]
        cases.append({"table": decimals, "bias": bias_dec, "vocab": vocab, "prompt": prompt, "beams": beams, "max_new": max_new,
                      "word_starts": word_starts if use_ban else None, "filler": filler if penalty else [], "penalty": penalty,
                      "expected": ids})
    return cases


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer.from_file(TOKENIZER)
    for name, data in [("tokenizer.json", tokenizer_cases(tok)), ("clean.json", clean_cases()), ("decode.json", decode_cases())]:
        (OUT / name).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        print(name, len(data))


if __name__ == "__main__":
    main()
