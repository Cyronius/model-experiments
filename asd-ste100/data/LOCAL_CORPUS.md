# Free local corpus

The owner selected **free downloads and local processing only** on October 23, 2026. This route builds a separately labeled augmentation corpus. It does not call a teacher or meaning judge, rent a pod, or replace the reviewed teacher-target corpus in `out/pairs-v1/`.

## Sources

The tracked `source_catalog.json` pins dataset revisions and selected files. Downloaded cards, manifests and data remain under the ignored `out/million/` directory.

| Source | Selected material | Published license |
|---|---|---|
| [Magicoder OSS-Instruct 75K](https://huggingface.co/datasets/ise-uiuc/Magicoder-OSS-Instruct-75K) | Coding problems and solution prose; code fences are excluded from source extraction | MIT |
| [Orca AgentInstruct 1M](https://huggingface.co/datasets/microsoft/orca-agentinstruct-1M-v1) | Code, tool use, web agents, text modification/extraction and structured-to-text subsets | CDLA-Permissive-2.0 |
| [SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk) | Only smol-magpie-ultra, smol-constraints, smol-rewrite and smol-summarize training files | Apache-2.0 for these four new subsets |

The downloader checks the non-SmolTalk license metadata, saves the cards, verifies pinned revisions and available LFS SHA-256 checksums, and enforces a six-GB download cap. The selected 17 files total approximately 2.62 GB. SmolTalk's mixed `all` configuration is deliberately excluded because its component licenses differ. Each final pair retains dataset, revision, file, record index, role, license, source URL and document attribution. Redistribution must retain the applicable source terms; this repository's code license does not relabel the downloaded data.

All three are synthetic source datasets. Free access and the license labels do not establish factual correctness or STE compliance.

## Build and resume

Run from the repository root. The downloader requires network access; decoding Parquet uses `pyarrow`. Other processing uses the Python standard library. The locally rebuilt lexicon must already exist.

```powershell
python asd-ste100/data/harvest_sources.py download --out asd-ste100/data/out/million
python asd-ste100/data/parallel_local_stage.py --sources asd-ste100/data/out/million --out asd-ste100/data/out/local-million-v1 --n 1000000 --workers 4
python asd-ste100/data/export_local_million.py --out asd-ste100/data/out/local-million-v1 --workers 4
python asd-ste100/data/validate_local_million.py --out asd-ste100/data/out/local-million-v1
python -m unittest discover -s asd-ste100/data/tests -v
```

Staging streams the pinned raw files directly and keeps its state in SQLite. It commits every thousand scanned sources and resumes by normalized source-content IDs. Dataset streams are interleaved with coding/agent sources weighted ahead of general instructions. Interrupted downloads restart incomplete files; completed files are checked and reused. Repeating staging with a different seed, filter, noiser, source manifest or exclusion directory requires a new output directory. The optional `harvest_sources.py prepare` command can materialize a separate raw-source database and JSONL for future teacher runs; the local build does not need that extra disk copy.

The parallel stage uses four CPU processes for extraction, checking and noising; a single writer owns SQLite. It submits bounded batches and consumes their results in a stable order. The last retained record in each dataset lets it skip old contents without re-tokenizing them. It restarts one record earlier to cover partially processed documents, preserves original record indices, verifies file hashes, and still checks processed content IDs. `parallel_driver.json` records worker count, offsets and driver hash. On a fresh output directory it starts from the beginning. `resume_local_stage.py` provides the single-process alternative. Parallel and single-process ordering can select different valid subsets when the requested count stops a larger input stream; review labels and validation gates are the same.

Targets are unmodified extracted chunks, not newly generated STE rewrites. Extraction removes code fences and markup, groups up to three whole sentences, and keeps chunks of 5–60 words that fit a surface-token canvas. It rejects long sentences rather than truncating them. Prompts/system messages use `PROC`; assistant prose uses `DESC` unless its first word is an instruction.

Conversation messages are processed independently. A user prompt and its assistant answer are never treated as parallel rewrite targets; each pair is made by noising one retained text and preserving that text as its target.

The stage requires:

- No checker errors or warnings, at most 20% unknown unprotected prose words, and no detected dangling heading or unfenced code fragment.
- No dictionary evaluation text or text already in any `pairs-v1` split.
- A changed source/target pair with identical detected protected spans, fitting a 64/128/256 surface-token canvas.
- At most six distinct variants per target. The raw target itself is never counted as another pair.
- One stable target label for each normalized noisy source and mode, with duplicate pairs removed.
- At most 70% of pairs in either mode and at most 60% from one dataset.

Noise is limited to dropped articles, sentence merges, formal phrases, simple passive/nominalization patterns, and regular-past-to-perfect patterns. Dictionary substitutions are excluded to reduce word-sense drift. The operation mix comes from applicable patterns in the retained targets; a million total pairs does not imply balanced coverage of all rewrite rules. Article restoration is expected to dominate.

Capital `A` is never dropped as an article because it can name a variable. Perfect-tense noise requires a pronoun at a clause boundary, avoiding object-pronoun phrases such as “get you started.” The October 23 run corrected these rules mid-build and filtered its staged rows before resuming. `curation_history.json` records the prior/current code hashes and removal counts; the previous source versions are saved under the ignored `previous-code/` directory. `curate_local_stage.py` applies current gates to an interrupted stage; it must run while staging is stopped. The final export is reproducible from the curated staging database. A fresh build using corrected rules can choose a different million-pair subset than an older, interrupted-and-curated build.

`stage_summary.json` records the actual number of pairs, independent targets, mode/dataset mix, operations and rejection reasons. A shortfall is reported as a shortfall; the pipeline does not duplicate rows or invent review scores to meet the requested count.

## Shards and checks

The compressed export has schema `ste-local-surface-pairs-v1`. It retains the token offsets, edit paths, source/target canvases and provenance described in `README.md`, plus:

```json
{"kind":"local_rule","target_review":"checker_only","meaning_review":"unjudged_rule"}
```

It has no fabricated `score` or `changed` judge fields. The separate [model tokenizer stage](../tokenizer/README.md) now provides vocabulary IDs, BPE fallback and unknown-word slots. Its stricter vocabulary/copy gates retain 610,642 of these million surface pairs in `out/id-local-million-v3/`; the original surface corpus remains unchanged.

All recorded documents, shared targets and identical input/output texts link connected components before train/dev/test assignment. All pairs from each component stay together. Components larger than 2% of the corpus are assigned to train so common boilerplate cannot put a large part of the corpus into a holdout. Other components receive deterministic approximate 90/5/5 assignment. The summary reports the largest component and the number of large components assigned to train. This is exact content/document grouping over recorded provenance, not semantic near-duplicate detection.

The output consists of manifest-listed `shards/{train,dev,test}-NNNNN.jsonl.gz` files, with 25,000 records per full shard. Gzip uses a fixed timestamp for reproducibility. Use the manifest rather than globbing a directory that might also contain files from an older export. For example:

```python
import gzip, json
from pathlib import Path
root = Path('asd-ste100/data/out/local-million-v1')
summary = json.loads((root / 'summary.json').read_text(encoding='utf-8'))
for shard in summary['shards']:
    with gzip.open(root / shard['path'], 'rt', encoding='utf-8') as stream:
        for line in stream:
            pair = json.loads(line)
```

The full validator checks every pair: hashes, review labels, target quality, source attribution, reserved-text exclusions, shared documents/texts across splits, conflicting labels, token offsets, protected spans, edit coverage and canvas reconstruction. It also regenerates the rule candidates to verify that each noisy input matches its recorded operation. Its result is `validation.json`. Unit tests also verify optimal edit costs, interrupted-build resumption, and identical gzip output with one versus two export workers.

These are **checker-filtered augmentation pairs**, suitable for testing scale and the baseline. They are not a million hand-verified STE rewrites. The checker omits several rules, allows technical words and unknowns, and does not check meaning. Rule noising can create awkward or ambiguous inputs. Evaluation still needs owner-verified real prompts and documentation; keep the reviewed teacher-target corpus identified separately.

## October 23 result

The completed `out/local-million-v1/` build passed its full audit.

| Measure | Result |
|---|---:|
| Unique pairs | 1,000,000 |
| Distinct target texts | 403,989 |
| Recorded documents audited | 180,754 |
| Train / dev / test | 906,045 / 48,332 / 45,623 |
| Instruction / description labels | 385,405 / 614,595 |
| Compressed output | 41 gzip shards, 607.9 MB |
| Teacher / judge calls | 0 / 0 |
| New paid compute | $0 |

| Noise operation | Pairs |
|---|---:|
| Drop article | 895,240 |
| Merge sentences | 71,338 |
| Formal phrase | 32,884 |
| Perfect tense | 486 |
| Passive | 26 |
| Nominalization | 26 |

Source contributions are 547,404 AgentInstruct pairs, 199,972 Magicoder pairs and 252,624 SmolTalk pairs. One component of 37,206 linked pairs was assigned to train. All pairs passed target gates, recorded-rule regeneration, protected-token equality, reconstruction, attribution, reserved-text exclusion, conflict and split checks. The 19-test suite passed; local-corpus tests were rerun after audit refinements.

`review_sample.jsonl` contains 200 examples with rare operations deliberately overrepresented. It is a review aid, not gold data or a representative quality estimate. Hard structural rewrites and meaning-preservation evaluation still need targeted work. The corrected teacher-target corpus is `out/pairs-v2/`; constructing a combined dataset requires joint deduplication and split grouping rather than concatenating independently assigned splits.
