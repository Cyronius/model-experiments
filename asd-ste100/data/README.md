# Preparing STE pairs

For the separate million-pair expansion using free datasets and local rule noise, see [LOCAL_CORPUS.md](LOCAL_CORPUS.md). The commands below prepare the existing meaning-judged teacher-target run.

Run from the repository root with Python 3.10 or later. This step uses local files and the standard library. Rebuild the ignored lexicon with `python asd-ste100/lexicon/build_lexicon.py --fetch` if it is absent, and extract the completed pod run before building pairs.

```powershell
python asd-ste100/data/build_pairs.py --run-dir asd-ste100/data/out/full1/ste-out --out asd-ste100/data/out/pairs-v2
python asd-ste100/data/validate_pairs.py --pairs asd-ste100/data/out/pairs-v2 --run-dir asd-ste100/data/out/full1/ste-out
python -m unittest discover -s asd-ste100/data/tests -v
```

`build_pairs.py` reads `judged.jsonl` and `messy.jsonl`. It requires targets rated 5 with `changed: false`, rechecks error-level STE violations, and excludes dictionary evaluation texts. It adds the original source, matching LLM back-translations, and up to two rule-noised variants per target. Empty, unchanged, preamble, protected-content mismatch, oversized, duplicate, and conflicting-label pairs are rejected with reasons in `rejected.jsonl`.

The corrected October 23 build in `pairs-v2` has 28,381 pairs. The earlier `pairs-v1` snapshot predates the capital-variable and pronoun noise corrections; use `pairs-v2` for new training work.

The rule noiser reverses dictionary alternatives with one alternative and a matching, unambiguous part of speech. It excludes function words and auxiliaries, and limits verb substitutions to imperatives and infinitives. Structural operations drop articles, merge sentences with “and,” add formal phrases, nominalize simple imperatives, make them passive, and change explicit-pronoun regular past clauses to perfect tense. Tense and voice rules deliberately cover only narrow patterns. Each variant applies one operation. Rules reduce obvious mistakes but cannot establish word sense or meaning equivalence.

`--rule-variants`, `--seed`, `--dev-fraction`, and `--test-fraction` control the build. Defaults are two variants, seed 0, and 90/5/5 train/dev/test allocation by connected groups. All variants of an origin stay together. Identical source or target texts also link groups, across modes and across source/target roles. Deduplication happens after grouping, so a discarded duplicate cannot cause its remaining variants to leak into another split. Fractions are approximate because groups have different sizes.

The noiser can also run by itself:

```powershell
python asd-ste100/data/noise_rules.py --input asd-ste100/data/out/full1/ste-out/clean5.jsonl --out asd-ste100/data/out/noised.jsonl --variants 2
```

The standalone command only makes variants. Use the pair builder to apply judge, checker, gold exclusion, and split filters.

## Output format

`train.jsonl`, `dev.jsonl`, and `test.jsonl` have schema `ste-surface-pairs-v1`. Each record contains:

- `src`, `ste`, and `mode`; content-based `id`, `origin_id`, `kind`, `noise`, and `meaning_review` provenance.
- `src_tokens` and `tgt_tokens`: case-preserving surface tokens, with character offsets and protected-token indices. URLs, emails, paths, dotted names, snake/camel identifiers, digit-bearing tokens, inline code and fenced code remain whole.
- `edits`: unit-cost Levenshtein steps as `[keep|replace|delete|insert, source_cursor, target_cursor]`. Cursors point before consumption of that step. Ties prefer diagonal steps, then deletion, then insertion.
- `source_canvas`: source tokens at even indices, with JSON `null` elsewhere. `target_positions` maps target tokens into `target_canvas` in order. Insertions can push later targets right.
- `bucket`: the smallest of 64, 128, and 256 that holds both canvases. A canvas reserves at least twice the source-token count. An example that cannot fit is dropped. Mode is a separate input, outside the canvas.
- `split` and `group`: deterministic group assignment.

JSON `null` means a vacant canvas position, not the literal text `NULL`. The separate [model tokenizer stage](../tokenizer/README.md) converts these surface tokens into input/output vocabulary IDs, handles fallback pieces and unknown-word slots, and rechecks capacity. It applies stricter vocabulary/copy gates than this surface build, so some checker-clean pairs are rejected.

`summary.json` records counts, rejection reasons, parameters, and input/code SHA-256 hashes. Repeating the command with the same inputs and code produces identical files. All derived files live under the existing ignored `data/out/` directory; do not commit them.

## What the checks establish

The validator visits every pair and checks judged-target provenance, checker errors, dictionary exclusion, protected-token equality, duplicate IDs, split separation, token offsets, alignment coverage, and canvas reconstruction. Unit tests independently check minimum edit costs over exhaustive short sequences, bucket promotion, transitive split grouping, filtering, and deterministic output under reordered input.

Checker-clean means no **error-level violations from the current checker subset**. Warnings and unknown words remain possible. The score applies to the original-to-STE rewrite; neither messy back-translations nor rule noise has a meaning judge score. Their `meaning_review` values explicitly say so. The span detector does not identify every proper name or unit. Dev/test here are generated-data holdouts; the owner-verified gold set and real-prompt meaning evaluation are still needed.
