# Model tokenizer and copy slots

This is the next data stage after the surface-pair builders. It fits a closed word-level output vocabulary, adds input-only rejected forms and a byte-level BPE fallback, creates request-local copy slots, and exports aligned vocabulary IDs in 64/128/256 canvases. It does not train a model or establish meaning preservation.

Run from the repository root, using Python 3.10 or later. The only extra dependency is `tokenizers`; the tested version is pinned in `requirements.txt`. The ignored lexicon files must already exist.

```powershell
python -m pip install -r asd-ste100/tokenizer/requirements.txt
python asd-ste100/tokenizer/prepare.py fit --pairs asd-ste100/data/out/pairs-v2 --vocab asd-ste100/data/out/tokenizer-v3
python asd-ste100/tokenizer/prepare.py export --pairs asd-ste100/data/out/pairs-v2 --vocab asd-ste100/data/out/tokenizer-v3 --out asd-ste100/data/out/id-pairs-v3 --workers 4
python asd-ste100/tokenizer/prepare.py validate --vocab asd-ste100/data/out/tokenizer-v3 --out asd-ste100/data/out/id-pairs-v3
python asd-ste100/tokenizer/prepare.py export --pairs asd-ste100/data/out/local-million-v1 --vocab asd-ste100/data/out/tokenizer-v3 --out asd-ste100/data/out/id-local-million-v3 --workers 4
python asd-ste100/tokenizer/prepare.py validate --vocab asd-ste100/data/out/tokenizer-v3 --out asd-ste100/data/out/id-local-million-v3
python -m unittest discover -s asd-ste100/tokenizer/tests -v
```

Use a new or empty output directory for each export. `--limit` makes a diagnostic prefix export and records the limit in its summary; it is not a full-corpus result. Input shards are read from their manifest and verified by SHA-256. Gzip timestamps and ordering are fixed. Vocabulary artifacts, exports and rejection logs stay under ignored `data/out/`, because the vocabulary includes AGPL-derived dictionary material. Do not commit them.

## Vocabulary and protection

The fitted vocabulary has **2,568 output IDs and 9,458 input IDs**, including 4,096 BPE tokens. Output IDs form the exact prefix of the input vocabulary. Approved forms use the checker's lexicon and noun plural expansion, word components of approved phrases, the software word list, and the function words explicitly listed in `ste_tokenizer.py`. Encoder-only regular forms expand rejected verb/noun lemmas conservatively; irregular morphology is incomplete. The fallback is fitted exclusively on `pairs-v2` training sources. Dev/test texts and the local augmentation corpus do not participate in fitting.

`SteTokenizer.encode_source()` builds its slot table from the source alone. The shared detector protects backticks/fences, paths, URLs, email addresses, identifiers and digit-bearing strings. This stage also combines signed numbers with a fixed list of units and detects capitalized multiword name candidates. Common command/function words are excluded as name starts. Remaining unknown tokens and mixed-case words become `PROTECT` slots. Isolated capital letters are protected, with `I` treated as a pronoun; `A` stays protected because it can name a variable.

Numeric spans use `NUM_1..16`, name candidates use `TN_1..32`, and other spans use `PROTECT_1..16`. Identical surfaces reuse one slot with an occurrence count. `TV_1..16` is reserved; there is no glossary or technical-verb inflection implementation. Detectors are conservative heuristics, not a complete recognizer of names, units or code. A miss on a known dictionary word can still allow an edit. Oversized requests or exhausted slot pools raise an explicit error; paragraph/sentence chunking belongs to the future host loop.

Inference slots every detected span. Training uses a deterministic 70% slot / 30% raw decision per occurrence. Raw spans are encoded with lossless UTF-8 byte BPE; each piece carries its source slot reference so an output slot remains copyable. This is raw exposure of *detected* spans, not a simulation of detector misses. The model must embed `source_slot_refs` as well as token and case IDs. Targets always use slots for protected/unknown content; byte pieces never enter the output head.

`additive_mask()` permits ordinary output words/punctuation, `NULL`, and active request slots. It masks inactive slots and the other control tokens. `restore_tokens()` separately enforces exact slot occurrence counts and rejects missing, duplicated, inactive and invalid IDs. Both checks are needed. `detokenize()` restores protected strings atomically and normalizes prose spacing; case labels support lower, initial capital, and upper case. Mixed-case text is copied through slots. The checker is still needed for rule compliance: a closed word vocabulary alone does not guarantee STE grammar or meaning.

## ID-pair records

Each `ste-id-pairs-v1` record retains source/target text, pair/origin/group IDs, split, mode, review labels, noise operation, and source attribution. The surface canvases are replaced with:

- `source_ids`, `source_case`, and `source_slot_refs` (`-1` means no slot reference).
- `source_spans`: `[piece_start, piece_end, char_start, char_end]` for each model surface span. Piece/character ends are exclusive.
- `slots`: `{token, surface, count}` entries built only from the source.
- `target_ids`, `target_case`, `target_positions`, minimum-cost `edits`, and the recomputed `bucket`; `surface_bucket` records the old size.
- `mode_id`: conditioning outside the canvas.

To materialize a canvas, fill `bucket` positions with the output ID for `NULL`, place source IDs at `2*i`, and place target IDs at `target_positions`. The source canvas uses input IDs; the target canvas uses output IDs. `PAD` is distinct from a vacant canvas position. Alignment trims equal edges before dynamic programming and retains an optimal edit cost; tied paths can differ from the older surface aligner.

The audit verifies shard hashes, ID ranges, exact source and target token reconstruction, active slots and source/target occurrence counts, case and slot-reference features, edit coverage, target positions, bucket capacity, duplicate IDs, existing group separation, and rejection accounting. It requires Python assertions enabled. It does not rerun the upstream checker or teacher judge, and it does not establish semantic quality or joint grouping across corpora.

## October 23, 2026 results

The teacher-target corpus retains **13,414 of 28,381 pairs**: 12,067 train, 636 dev and 711 test. All retained pairs pass the ID audit. The rejected pairs comprise 4,928 targets with an encoder-only word, 5,184 with a target slot absent from the source, 3,874 with changed slot counts, 974 exceeding the general slot pool, six exceeding the numeric pool and one exceeding input capacity. These are representation-policy failures, not new meaning judgments. Surface pairs remain available unchanged.

Retained slices are 1,876 original teacher rewrites, 682 messy back-translations and 10,856 rule augmentations. This is a substantial loss of original rewriting examples. Thirteen tokenizer tests and nineteen upstream data tests pass. Real teacher-corpus exports with one and four workers have identical shard hashes. Exploratory `v1/v2` tokenizer/ID artifacts are historical; the interrupted `id-local-million-v2` has no completed manifest and must not be consumed.

Of retained teacher-corpus pairs, 737 move from 64 to 128 positions and 192 from 128 to 256; name grouping also shrinks 43 pairs to a smaller bucket. Final buckets contain 9,613 / 3,449 / 352 pairs. Checker-clean data can still contain words outside the model vocabulary because the current checker permits unknown/technical nouns and some rejected verbs in noun-like contexts. The exporter reports that gap instead of silently approving new output words.

The local million-pair corpus retains **610,642 pairs**: 553,586 train, 29,262 dev and 27,794 test, in 27 gzip shards totaling 348.4 MB. All retained rows pass the full ID audit recorded in `validation.json`. Rejections are 384,887 targets with an encoder-only word, 3,395 with a target slot absent from the source, 898 with changed slot counts, 129 exceeding the general slot pool, eleven exceeding the numeric pool and 38 exceeding input capacity. Retained buckets contain 501,317 / 101,417 / 7,908 pairs. The export preserves `checker_only` / `unjudged_rule` labels; none of these are new teacher rewrites or meaning judgments.

The two corpora keep their original splits and review labels. Do not concatenate them for training with separate holdouts: first recompute joint connected groups across their texts and recorded document provenance. Generated holdouts remain distinct from the owner-verified gold set. The next modeling step is the iterative tagger baseline; the gold set and real-input meaning evaluation are still outstanding.
