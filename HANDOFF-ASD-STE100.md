# ASD-STE100: continue on another computer

Updated October 23, 2026. Start here, then read [the experiment handoff](asd-ste100/HANDOFF.md), [the design](asd-ste100/PLAN.md), and [the tokenizer notes](asd-ste100/tokenizer/README.md).

## Where we stopped

We are building a tiny non-autoregressive English-to-Simplified-Technical-English editor. The lexicon converter, deterministic checker, data preparation and model tokenizer/slotter are implemented. **There is no trained student model, model implementation, active training job, GGUF writer, llama.cpp patch or NPU engine for this experiment.** The last completed job was local data export and validation. No new money was spent.

The baseline is an iterative tagger: six bidirectional transformer layers, width 512, eight heads and FFN 2048. It comes before the canvas editor. Output is a closed STE/software vocabulary plus copy slots. Code, identifiers, numbers and detected names/units must be restored unchanged. The final model format is GGUF. Target size/latency are under 100 MB and under 500 ms per sentence on CPU; NPU work waits for measurements. Keep the existing decisions in the experiment handoff.

## Completed data work

| Artifact under `asd-ste100/data/out/` | State |
|---|---|
| `full1/ste-out/` | Saved teacher/judge run; 8,587 score-5 targets before final pair filtering. Original pod cost was $1.71. Do not rerun paid generation to recover local files. |
| `pairs-v2/` | Corrected surface corpus: 28,381 pairs, fully audited. `pairs-v1` is historical and was the exclusion set for local augmentation. |
| `local-million-v1/` | One million unique checker-only rule augmentations from 403,989 target texts; full surface audit passed. Article restoration dominates. |
| `tokenizer-v3/` | Final tokenizer: 2,568 output IDs, 9,458 input IDs, including a 4,096-token byte fallback fitted on teacher-corpus training sources only. |
| `id-pairs-v3/` | 13,414 retained teacher-corpus pairs: 12,067 train / 636 dev / 711 test; full ID audit passed. Only 1,876 are original teacher rewrites; the rest are augmentations. |
| `id-local-million-v3/` | 610,642 retained local pairs: 553,586 train / 29,262 dev / 27,794 test; full ID audit passed. 27 compressed shards, 348.4 MB. |

The strict tokenizer rejects many checker-clean targets: the checker permits unknown technical nouns and some rejected words in noun-like contexts, while the output vocabulary is closed. Do not silently approve those words to recover pair counts. Original surface datasets remain unchanged. Local augmentation labels remain `target_review: checker_only` / `meaning_review: unjudged_rule`; they are not meaning-verified gold.

Training uses 70% slotted / 30% raw input exposure for detected spans. Raw spans use byte BPE and carry `source_slot_refs`; the model must consume slot-reference and case features. Targets always use copy slots. Inference slots every detected span. Mask inactive slots and control tokens, and separately check exact slot counts before returning decoded output. Detection is heuristic. Technical-verb slots are reserved; no glossary implementation exists.

Thirteen tokenizer tests and nineteen data tests passed. Both retained ID corpora passed full reconstruction audits. Repeated tokenizer fits were byte-identical; one-worker and four-worker teacher exports produced identical compressed shards. The checker was not changed in this continuation.

## What Git does and does not transfer

The `main` commit contains this root handoff, experiment documentation, data builders, pinned public source catalog, tokenizer, and tests. Other experiments' uncommitted changes are outside this handoff.

**Git does not contain the data or fitted tokenizer.** Everything under `asd-ste100/data/out/` and the dictionary-derived files under `asd-ste100/lexicon/` are ignored. The dictionary packages are AGPL; the existing project decision is to keep derived artifacts out of Git. A fresh clone alone supports code work and synthetic tokenizer tests, but it cannot reproduce the exact teacher corpus without the saved teacher output.

A local transfer archive is prepared at:

```text
asd-ste100/data/out/asd-ste100-transfer-20261023.tar
asd-ste100/data/out/asd-ste100-transfer-20261023.manifest.json
asd-ste100/data/out/asd-ste100-transfer-20261023.tar.sha256
```

Copy those files separately using your own drive or file transfer. They are not uploaded with the Git commit. The archive is 1,207,521,280 bytes (about 1.21 GB), contains 120 files, and has SHA-256 `4ff323b7acacfd46961395573fd66a7c6d3600b98a4e77b25900de0d2b46b331`. Every archived file was read back and checked against its source hash.

The archive uses repository-relative paths and contains the exact fitted vocabulary, audited surface/ID exports, saved teacher outputs, small provenance/audit files, and exact derived lexicon. The companion manifest records SHA-256 for every included file. It excludes the 2.62 GB raw source downloads, the 2 GB staging SQLite database, exploratory tokenizer/ID versions and the interrupted `id-local-million-v2` export. It supports continuing model/data-split work, not resuming the old staging database.

## Set up the other computer

```powershell
git clone https://github.com/Cyronius/model-experiments.git
cd model-experiments
git switch main
git pull --ff-only
python -m venv .venv
# Windows:
.venv\Scripts\Activate.ps1
# Linux/macOS instead: source .venv/bin/activate
python -m pip install -r asd-ste100/tokenizer/requirements.txt pyarrow
```

Use Python 3.10 or newer. No paid account or GPU is needed for this setup. Check the transferred archive's SHA-256 against the `.sha256` file, then extract it **from the repository root**:

```powershell
Get-FileHash /path/to/asd-ste100-transfer-20261023.tar -Algorithm SHA256
tar -xf /path/to/asd-ste100-transfer-20261023.tar
python -m unittest discover -s asd-ste100/data/tests -v
python -m unittest discover -s asd-ste100/tokenizer/tests -v
python asd-ste100/tokenizer/prepare.py validate --vocab asd-ste100/data/out/tokenizer-v3 --out asd-ste100/data/out/id-pairs-v3
python asd-ste100/tokenizer/prepare.py validate --vocab asd-ste100/data/out/tokenizer-v3 --out asd-ste100/data/out/id-local-million-v3
```

On Linux/macOS, use `sha256sum` in place of `Get-FileHash`. Do not use Python `-O` for audits: they depend on assertions. Read compressed shards from `summary.json`, not a directory glob. The original local-surface manifest uses Windows separators; normalize `\\` to `/` when reading it on another OS, as `tokenizer/prepare.py` already does.

If the archive is unavailable, the exact AGPL-derived lexicon can be rebuilt with `python asd-ste100/lexicon/build_lexicon.py --fetch`; compare it with the hashes recorded in existing artifact manifests before assuming identical results. The public raw datasets can be downloaded for free with `harvest_sources.py`; see [the local corpus notes](asd-ste100/data/LOCAL_CORPUS.md). Teacher outputs cannot be recreated by those downloads. A fresh local-million build may select a different valid subset from the interrupted-and-curated historical run. Do not present that as a byte-identical rebuild.

## Do next

1. Build a combined split assignment across teacher and local corpora. Recompute connected groups using normalized source/target text, teacher origin IDs and recorded local document provenance. Do not concatenate the existing independently assigned splits: their holdouts are not jointly verified. Preserve attribution and review labels, deduplicate pairs, and audit the combined groups before training. Use the surface corpora to establish grouping before strict tokenizer filtering so discarded examples cannot hide links.
2. Implement the iterative tagger baseline locally. Share the tokenizer, closed output vocabulary, slot restoration and checker. See design section 3.5 for tags and architecture. Build an offline forward/loss/decode smoke test before renting compute.
3. Prepare the owner-verified gold set (300 sentences and 50 paragraphs) and real-input meaning evaluation. The owner must verify the gold targets; generated dev/test sets do not replace it. Hard structural rewrites and original teacher examples remain data gaps.
4. Train only after the next paid session is explicitly authorized. Then evaluate the tagger before building the canvas model or runtime.

Current authorization is free downloads and local processing. Ask before spending money, launching/stopping a pod, or changing established design decisions. Never commit derived dictionary/vocabulary artifacts or weights. If the checker changes, rerun calibration and report strict and software-word-list results.

Suggested next-agent prompt:

> Read HANDOFF-ASD-STE100.md and continue the ASD-STE100 experiment. First build and audit joint dataset groups/splits, then implement an offline iterative tagger baseline using the existing tokenizer and slot/case features. Stay local and free; do not launch a pod or change the established design.
