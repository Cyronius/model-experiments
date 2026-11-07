# Joint surface splits

`joint_splits.py` builds connected groups across the **unfiltered surface** teacher and local corpora. Run this before strict tokenizer filtering; do not concatenate independently split ID exports. This tool uses only the Python standard library. All generated artifacts belong under ignored `data/out/`.

After restoring the transfer archive from the repository root:

```bash
python asd-ste100/data/joint_splits.py build --corpus teacher=asd-ste100/data/out/pairs-v2 --corpus local=asd-ste100/data/out/local-million-v1 --out asd-ste100/data/out/joint-surface-v1
python asd-ste100/data/joint_splits.py validate --corpus teacher=asd-ste100/data/out/pairs-v2 --corpus local=asd-ste100/data/out/local-million-v1 --out asd-ste100/data/out/joint-surface-v1
python -m unittest discover -s asd-ste100/data/tests -p test_joint_splits.py -v
```

Use a new or empty output directory. Inputs come from `summary.json`, with shard hash checks and Windows separator normalization. Older unsharded teacher exports use the three split filenames. Input files are never rewritten. Original ID exports are rejected as grouping inputs.

Each source and target contributes an NFC/casefold/whitespace-normalized text key, shared across modes and corpora. Origins are scoped by corpus name; recorded document IDs are shared. All keys are joined transitively before duplicate/conflict removal, so a discarded example still connects its other variants and documents. Conflicting normalized targets for the same normalized source and mode are excluded. Identical normalized `(mode, source, target)` pairs are merged, preferring original teacher rewrites, then messy rewrites, then teacher rule variants, then local rules.

The emitted canonical row preserves its surface alignment, mode, review labels and attribution. `joint_members` retains each original member's metadata, text, review labels, document provenance, ID and old split/group. Large alignment arrays are stored only for the canonical row. A duplicate local rule does not become teacher-reviewed: its original review labels remain in its member record. No new checker or meaning judgments are made.

Groups receive deterministic SHA-256 IDs and default 90/5/5 train/dev/test assignment. Components larger than 2% of scanned records are kept in train, as in the existing local exporter. Counts include duplicates and conflicts during component sizing. The summary reports actual splits, largest component, merged duplicates and discarded conflicting records. Hash assignment cannot guarantee exact split proportions.

SQLite scratch tables keep surface rows off the Python heap; union-find link keys remain in memory. Output ordering and gzip timestamps are deterministic. The separate validator rereads both original corpora and the output, checks input and output hashes, deduplication/record accounting and metadata equality, and propagates assignments through all original text/origin/document links, including discarded conflicts. It checks that connected keys share a group and split. It does not rerun upstream checker/alignment audits, prove semantic quality or provide owner-verified gold.

## Refit after joint splitting

The transferred `tokenizer-v3` was fitted on the **old teacher train split**. A new joint holdout may contain those training sources. Refit the byte fallback on joint train only, then export and audit fresh IDs; keep the historical artifacts intact:

```bash
python asd-ste100/tokenizer/prepare.py fit --pairs asd-ste100/data/out/joint-surface-v1 --vocab asd-ste100/data/out/tokenizer-joint-v1
python asd-ste100/tokenizer/prepare.py export --pairs asd-ste100/data/out/joint-surface-v1 --vocab asd-ste100/data/out/tokenizer-joint-v1 --out asd-ste100/data/out/id-joint-v1 --workers 4
python asd-ste100/tokenizer/prepare.py validate --vocab asd-ste100/data/out/tokenizer-joint-v1 --out asd-ste100/data/out/id-joint-v1
```

This preserves the closed output vocabulary policy and the existing 70% slotted / 30% raw exposure. Input fallback tokens can change. Report tokenizer rejection counts and retained review slices before training. Training still requires explicit paid-session authorization.

## October 24, 2026 results

The transfer archive was fetched from the owner's FTP server, verified against the handoff SHA-256, and all 120 restored files matched their manifest hashes. The full joint build and separate audit passed on 1,028,381 surface records. There were no cross-corpus duplicates or conflicting targets. There are 166,913 components; the largest has 37,206 records and is assigned to train.

| Stage | Train | Dev | Test | Total |
|---|---:|---:|---:|---:|
| Joint surface | 931,571 | 49,788 | 47,022 | 1,028,381 |
| Joint model IDs | 565,599 | 29,977 | 28,496 | 624,072 |

The refitted vocabulary retains the exact historical 2,568 output IDs and 9,458 input IDs. The 4,096-token byte fallback changed and sees only joint train sources. Every retained ID record passed the reconstruction audit. Of 404,309 rejected records, 389,822 have a target outside the closed vocabulary, 8,579 an unseen target slot, 4,772 changed slot counts, 1,103 general slot overflow, 17 numeric slot overflow, and 16 source capacity overflow.

Retained kinds are 610,659 local rule pairs, 10,855 teacher rule pairs, 1,876 original teacher rewrites and 682 messy back-translations. The original rewrite slice splits 1,688 / 103 / 85 across train/dev/test. The corpus is still dominated by unjudged augmentation; these holdouts do not replace owner-verified gold. The changed byte segmentation changes capacity outcomes slightly relative to the old separately exported corpora.

Audits and manifests are under `joint-surface-v1/`, `tokenizer-joint-v1/` and `id-joint-v1/`; retained slice counts are in `id-joint-v1/slices.json`. Local logs are `joint-build.log`, `joint-audit.log`, `joint-tokenizer-fit.log`, `joint-id-export.log`, `joint-id-audit.log` and `joint-id-slices.log` under `data/out/`. The environment is `data/out/.venv/` (Python 3.14).

Twenty-six data tests (including seven joint-split tests) and thirteen tokenizer tests pass with the restored lexicon and dependencies. Synthetic fixtures cover transitive origin/text/document links, removed conflict bridges, normalized cross-mode links, deduplication and review preservation, holdout creation, reproducible compressed output, corrupt manifests, tampered audit evidence and changed canonical alignment arrays. A separate 20,000-pair synthetic build/audit also passed. Multiprocessing tests/export require permission for Python forkserver sockets in the restricted environment; the single-process joint tools need no network.

The [offline tagger baseline](../model_tagger/README.md) is implemented and smoke-tested. Full multi-pass label coverage, owner-verified gold, real-input meaning evaluation and authorized training remain outstanding. No paid compute or checker changes were made.

## Assistant spot check, October 24, 2026

`spot_check.py` selected 150 distinct retained pairs with a recorded SHA-256 sampling seed: 60 uniform pair-level examples, 20 each from original/messy/teacher-rule slices, and five each from six local noise operations. The assistant read every sampled source and target. Results are linguistic judgments, not owner-verified gold, full STE compliance or factual validation.

| Sample | No obvious issue | Needs review/context | Recommend exclude/repair |
|---|---:|---:|---:|
| Uniform 60 | 54 | 3 | 3 |
| Diagnostic oversample 90 | 59 | 18 | 13 |

The diagnostic group deliberately oversamples small and rare slices; do not treat its aggregate exclusion rate as a corpus-wide estimate. Even the uniform sample is small. The original teacher slice had 13 / 4 / 3 and messy had 9 / 7 / 4, illustrating that the old teacher judge's score 5 does not establish perfect meaning preservation.

Confirmed mechanisms reproduced locally: article deletion can edit a lowercase mathematical variable or a string inside unfenced code; sentence merging can consume a numbered-list marker; HTML stripping can remove an inline tag from inside backticks. Other reviewed exclusions include incomplete/truncated targets, broken target grammar, changed question intent, changed pronoun referents and strengthened modality. The checker reported zero error-level target violations and two warning-only targets across the sample; these failures mostly lie outside its current gates.

Artifacts are under ignored `data/out/spot-check-20261024-v1/`: `manifest.json`, `sample.jsonl`, `reviewed.jsonl`, `review-summary.json`, `review.txt` and `mechanism-reproductions.json`. The original corpus, labels, checker and tokenizer are unchanged. The new sampler is reproducible with:

```bash
python asd-ste100/data/spot_check.py --pairs asd-ste100/data/out/id-joint-v1 --out asd-ste100/data/out/spot-check-20261024-v1
```

Use a fresh directory when repeating. Before training, repair the identified extraction/noise gates and measure their affected population; do not simply relabel all 624,072 records as meaning-verified. A reviewed sample can inform targeted cleanup without discarding the largely useful article-restoration corpus.

## Deterministic cleanup, October 24, 2026

At the owner's request, `quality_gates.py` adds dataset quality gates separate from the STE checker. `check_quality.py` scans manifest-listed data, records every finding/rejection and creates an immutable filtered subset. `validate_quality.py` compares every parent decision and retained payload. The original `id-joint-v1` and all historical surface/teacher artifacts remain intact.

The filtered export is `data/out/id-joint-clean-v2/`: **607,040 pairs**, with **550,129 train / 29,263 dev / 27,648 test**. It excludes **17,032 of 624,072 pairs (2.73%)**. It retains 593,638 local rules, 10,845 teacher rules, all 1,876 original teacher rewrites and 681 messy back-translations. Splits/groups, IDs, text, token IDs, slot/case features, attribution and meaning-review labels are unchanged for kept records. Another 3,035 retained records have review flags; they are not silently discarded.

| Automatic exclusion rule | Pair matches |
|---|---:|
| Contractions in target prose (including curly apostrophes and pronoun + apostrophe-s) | 12,694 |
| Empty inline tag/element references | 2,029 |
| Synthetic merging of a numbered-list marker | 1,117 |
| Explicit truncation markers | 407 |
| Changed recognized code literals/calls/blocks | 376 |
| Changed single-letter variables adjacent to mathematical operators | 321 |
| Clearly unfinished copula clauses | 135 |

Counts overlap; they do not sum to the unique exclusion count. Possessives, contractions quoted as examples, and contractions inside code are not automatically excluded. Generic empty inline literals, ellipses, dangling step headings and potentially complete relative clauses receive review flags. A blank delimiter, an infinite series ending in ellipses, a cross-reference to a step, a sentence ending in a number, and a comparative construction such as “the more … the more … will be” must not trigger broad hard filters. The code recognizer handles a limited set of high-confidence syntaxes, not every programming language or mathematical expression.

The assistant inspected deterministic per-rule samples and narrowed initial patterns that caught valid numeric sentence endings and complete comparative/relative clauses. On the prior 150-pair reviewed sample, hard gates catch eight examples previously marked exclude and no examples previously marked usable or needing review. The remaining eight previously marked exclude include meaning changes and grammar defects requiring judgment. This cleanup does not certify meaning, facts or full STE compliance.

Future builders also use the hard target gates. The noiser now rejects edits to recognized code/math/markup and synthetic list-marker merges. Extraction preserves inline code and recognized calls/blocks while stripping HTML or normalizing prose, preventing inline HTML examples from becoming empty backticks. These changes are fingerprinted by the builders. The STE checker and output vocabulary are unchanged.

Reproduce with new/empty output directories:

```bash
python asd-ste100/data/check_quality.py scan --pairs asd-ste100/data/out/id-joint-v1 --out asd-ste100/data/out/quality-scan-final
python asd-ste100/data/check_quality.py filter --pairs asd-ste100/data/out/id-joint-v1 --out asd-ste100/data/out/id-joint-clean-v2
python asd-ste100/data/validate_quality.py --parent asd-ste100/data/out/id-joint-v1 --out asd-ste100/data/out/id-joint-clean-v2
python asd-ste100/tokenizer/prepare.py validate --vocab asd-ste100/data/out/tokenizer-joint-v1 --out asd-ste100/data/out/id-joint-clean-v2
```

`quality-report.json` contains unique totals, overlapping rule counts, retained slices and code/input fingerprints. `findings.jsonl.gz` records hard and review-only findings; `rejected.jsonl` accounts for every exclusion; `examples.jsonl` contains deterministic per-rule review samples. Subset and ID audit outputs are `quality-validation.json` and `validation.json`. The preliminary `id-joint-clean-v1` excluded five quoted passages using curly quotation marks; `v2` restores those rows. Exploratory `quality-scan-v1/v2/v3` outputs are historical pattern-development scans; use the report attached to the filtered export for final decisions.

The existing tokenizer remains valid for this subset: no holdout changes or new fit occurred. Its fallback was fitted on the parent joint training split, which also includes training sources removed by these gates; it was not refitted exclusively on the filtered training rows. A future fresh fit can use the filtered train texts. Do not rebuild joint groups after these exclusions, since discarded examples still establish links.

Validation: 34 data tests, 13 tokenizer tests and six tagger tests pass. The eight new data tests cover copy zones, quoted text, contractions versus possessives, literal/heading counterexamples, numeric/comparative/relative-clause false positives, future extraction/noise safeguards, exact filter accounting and subset validation. The checker was not changed; no checker recalibration claim is made.
