# Stage 1: counting pass and n-gram selection (2026-11-12, CPU only, no teacher)

## Corpus

460,869,998 tokens: the Stage 0 Jaswanth training texts (96.4M tokens) plus
the local candidate corpora (`data-first/candidates`, every
`broad-data/candidates/<source>`, `broad-data/ready/train.jsonl`; both
`query` and `positive`), 8,329,179 strings after exact-duplicate removal
across all sources and removal of strings in the validation split (contract
rule), each cut to 504 tokens. This is below the 1-billion-token cap, so
nothing was sampled. Per-source counts are in `result.json`
(`counting_corpus.local_sources`).

## Method

Unigrams exact. Bigrams and trigrams: count-min sketch (2 rows x 2^27 uint32
per order, 1 GiB per order), then an exact second pass over keys whose
sketch count is at least 20; keys with an exact count of at least 20 are
candidates. Selection drops n-grams containing `[UNK]` or another special
token, and n-grams made only of punctuation/symbol tokens and the 150 most
frequent tokens; the rest are ranked by count (ties by key).

| | bigrams | trigrams |
|---|---|---|
| exact count >= 20 | 1,606,055 | 2,253,491 |
| dropped: special token | 697 | 976 |
| dropped: punctuation / top-150 only | 22,822 | 114,089 |
| kept, ranked | 1,582,536 | 2,138,426 |
| T1 (top 250,000), minimum count | 194 | 126 |
| T2 (top 1,000,000), minimum count | 36 | 39 |
| T3 size (target 2,000,000) | **1,582,536** (all kept) | 2,000,000 (min count 21) |

**Deviation:** bigram T3 holds 1,582,536 rows, not 2,000,000: only that many
bigrams occur at least 20 times in 461M tokens. Getting more would need more
text or a lower threshold; T3 is only used for the size report, so I did not
change the threshold.

## Coverage of the shared validation texts

Jaswanth validation split, texts that fit the 504 contract (9,204 texts,
767,885 tokens). Denominator: positions that end an n-gram of that order
inside the document.

| texts | n | bigram T1 | T2 | T3 | trigram T1 | T2 | T3 |
|---|---|---|---|---|---|---|---|
| <= 128 tokens | 7,118 | 50.7% | 63.7% | 67.0% | 22.4% | 33.1% | 38.9% |
| > 128 tokens | 2,086 | 48.9% | 61.8% | 65.1% | 20.5% | 30.8% | 36.6% |
| all | 9,204 | 49.5% | 62.4% | 65.7% | 21.1% | 31.5% | 37.4% |

T2 raises coverage over T1 by 12.9 points for bigrams and 10.4 points for
trigrams, so the coverage condition for arm S6 (at least 10 points) holds for
both orders. `[UNK]` share: 0.02% of tokens.

Unigram frequencies `p[t]` are in `training-v2/work/ngram-space/stage1/unigram_p.npy`
(29,016 of 30,522 ids seen; most frequent id 1996 "the", p = 0.048).

## Table bytes (w_ng = 512, bigram + trigram rows at the same tier)

| tier | rows | float32 | int8 + one float scale per row | hash index (load 0.5, uint64 keys) |
|---|---|---|---|---|
| T1 | 500,000 | 1.02 GB | 0.26 GB | 0.008 GB |
| T2 | 2,000,000 | 4.10 GB | 1.03 GB | 0.032 GB |
| T3 | 3,582,536 | 7.34 GB | 1.85 GB | 0.057 GB |

Unigram table at d = 1024: 125 MB float32, 31 MB int8. With int8 rows (the
Stage 6 format) the 0.5 / 1 / 2 GB budget points correspond roughly to T1+,
T2 and T3.

## Surface strings

Most frequent original surface (`text[start of first token : end of last token]`)
counted over the Jaswanth training texts plus the `data-first` candidates.
Unseen in that sample (decode fallback, lower case): bigrams 130 of T1,
2,851 of T2, 13,048 of all; trigrams 1,323 of T1, 8,795 of T2, 30,026 of T3.
Examples (trigram ranks 1-5): "don't", "the difference between", ". However,",
"U.S", "I'm". N-grams that start inside a word have surfaces like
"ationships"; they are sent to the teacher as they are.

## Files and commands

Work folder `training-v2/work/ngram-space/stage1/`: `local/` (tokenized local
corpus), `unigram_counts.npy`, `unigram_p.npy`, `ranked_n{2,3}_keys.npy`
(rank order; tiers are prefixes), `ranked_n{2,3}_counts.npy`,
`ranked_n{2,3}_surface.jsonl`, `count_meta.json`, `coverage.json`,
`surface_meta.json`.

```
dibs run cpu -- .venv/bin/python -I ngram-space/stage1/tokenize_local.py
dibs run cpu -- .venv/bin/python -I ngram-space/stage1/count.py training-v2/work/ngram-space/stage1/local training-v2/work/ngram-space/stage0/train
dibs run cpu -- .venv/bin/python -I ngram-space/stage1/surfaces.py
.venv/bin/python -I ngram-space/stage1/coverage.py
```

`result.json` keys: counting_corpus, method, selection, tier_sizes,
coverage_validation, unigram, table_bytes_w512, unigram_table_bytes, surfaces.
(`surfaces.n2.T3_unseen` corrected by hand: the script counted against a
2,000,000-row tier that has 1,582,536 rows.)
