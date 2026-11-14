# Repeatable claim extraction and contradiction screening

This is a local proving ground for document curation. It takes source passages,
extracts individual assertions with their conditions and supporting quotations,
checks the citations, and exports proposed claim records. A separate benchmark
tests whether models distinguish contradictions from different scopes.

The initial comparison uses stock checkpoints and local inference. The original XDNA2 notes are
never edited. The experiment lives here so the curation process can evolve
independently of the eventual XDNA2 programming model and any retrieval backend.

## Pilot

- **12 passages:** nine real excerpts from five XDNA2 project notes, plus three
  explicitly synthetic controls for negation, absent information, and questions.
- **33 named claim anchors:** fixed before benchmark inference, including
  numeric values, failures, exceptions and workload/version qualifications.
- **10 claim pairs:** contradictions, equivalent units, paraphrases, different
  versions/workloads, and insufficient information.
- **Models:** NuExtract 2.0 2B; stock Qwen3-4B-Instruct-2507; current NuExtract3
  4B added after the older 2B model returned empty lists on the first real cases.
- **One excluded development example:** fictional hardware and different
  values, used to select native prompt formatting and a one-shot example. The
  older NuExtract needed an instruction and example to extract both a simple
  latency and the fact that firmware information was missing.

[Measurements and limitations](results/summary.md) are generated from retained
outputs. These are agent-authored pilot annotations, not independently reviewed
human ground truth. No output should automatically replace a finding.

### First measured result

| Model | Named target facts matched | Scoped and exactly cited targets | Supplied pair labels |
|---|---:|---:|---:|
| NuExtract 2.0 2B | 9/33 | 6/33 | 4/10 |
| NuExtract3 4B | 30/33 | 27/33 | 8/10 |
| Stock Qwen3-4B | 26/33 | 17/33 | 10/10 |

NuExtract3 produced 66 proposed records with exact quotations/condition
fragments, all 66 mechanically valid. It still lost some crucial context and
strengthened a recommendation into a guarantee. The older 2B model missed most
technical passages and turned questions into claims. Qwen handled the pair
labels best here but produced malformed JSON for two extraction passages.

These are lexical coverage diagnostics and a qualitative spot review, not a
fully adjudicated accuracy score. A post-inference matcher fix prevents `work`
from matching `workers`; original frozen inputs and scores are preserved, and
refined anchors/scores are separate artifacts. See
[qualitative review](qualitative-review.md) for concrete failures.

The evidence supports using NuExtract3 as a **proposal extractor** and testing
Qwen4B for the separate comparison/review pass. It does not establish automatic
document correction, contradiction discovery, or semantic truth verification.
The stock-model comparison does not show that fine-tuning is necessary. The separate teacher-supervised LoRA pilot under `training/` is complete; its adapter was not promoted.

## Record format and provenance

The model emits only:

```json
{
  "claims": [{
    "statement": "One scoped factual assertion.",
    "quote": "An exact passage from the input.",
    "conditions": ["An exact condition fragment from the input."]
  }]
}
```

Code adds stable content-derived claim/evidence IDs, the source path and snapshot
hash, exact character/line spans, the supplied stack and evidence labels, the
model/prompt cache key, and `review_status: proposed`. Source evidence labels
describe the note; they do not establish that each extracted assertion was
measured or true. `semantic_support_verified` remains false.

The five input documents are project notes/digests. Their references to upstream
READMEs, commits and code locations are retained verbatim but **not independently
resolved or verified**. The checked citation reaches the exact ingested note.
Resolving those upstream references is a later provenance stage; the pipeline
does not invent an original-source citation when it has not read that source.

Records with nonexistent quotations or invented condition fragments go to
`quarantine.jsonl` and are excluded from the derived SQLite claim table. A
verbatim quotation alone does not prove that the statement correctly interprets
it. There is no semantic truth certification, automatic source correction,
cross-document entity resolution, or GraphRAG in this pilot.

## Repeatability

`data/manifest.json` records the input snapshot hashes and originating repository
commit. `preregistered.json` fixes the prompt, annotations and pair labels.
`prompt.json` contains the complete instructions, native templates and excluded
development examples. Gold labels are not passed to the models.

The cache key covers the exact rendered prompt, source text, model revision,
library versions and generation settings. An unchanged run replays saved
outputs without loading the model. Changing inputs, prompts, revisions or
settings invalidates the affected entries. Rendered prompts, raw completions,
truncation flags, batch timing and settings are retained. Repeatability means
auditable inputs and retained outputs; fresh floating-point inference is not
promised bit-identical across different hardware or batch shapes.

The fixed-gold benchmark and the original scores are retained even when a
matcher bug is corrected. `data/gold-refined.json` and
`data/gold-refinement.json` explicitly record the outcome-word refinement;
`extraction-scores-refined.json` records its results. No extraction is regenerated
to make a scoring correction look like model improvement.

Default execution is CPU bfloat16, eight threads, batch size three, greedy
decoding, and a 768-token output cap. There is no JSON repair or hidden retry.
All runs use native Transformers classes and local checkpoint files. No remote
model implementation code is executed.

The NuExtract3 pilot uses four threads. It honors both the end token in the
checkpoint's generation configuration (248044) and its tokenizer/chat-template
end token (248046). Its initial launch was restarted before collecting outputs
to set those explicit stopping IDs.

## Run

From the `model-experiments` repository root, using the existing CPU environment:

```bash
PY=target/manifold-audit/env/bin/python

# Optional: refresh the pilot input snapshots deliberately.
$PY claim-curation/dataset.py --notes ../xdna2-notes --out claim-curation/data

# Public open weights; downloading does not call a paid inference service.
$PY claim-curation/download.py --model nuextract2b --out models/claim-curation/nuextract2b
$PY claim-curation/download.py --model nuextract3 --out models/claim-curation/nuextract3

$PY claim-curation/run_models.py --model nuextract2b \
  --checkpoint models/claim-curation/nuextract2b --data claim-curation/data \
  --out claim-curation/results/nuextract2b --cache claim-curation/cache
$PY claim-curation/run_models.py --model nuextract3 \
  --checkpoint models/claim-curation/nuextract3 --data claim-curation/data \
  --out claim-curation/results/nuextract3 --cache claim-curation/cache --threads 4
$PY claim-curation/run_models.py --model qwen4b \
  --checkpoint models/context-memory/qwen3-4b-stock --data claim-curation/data \
  --out claim-curation/results/qwen4b --cache claim-curation/cache

$PY claim-curation/summarize.py
$PY -m pytest -q claim-curation/test_curator.py
```

The stock Qwen checkpoint is reused from the context-memory experiment. It can
also be downloaded with `download.py --model qwen4b --out NEW_PATH`.

After each model completes, `curator.py --data DATA --out RESULTS` can validate
and score it independently. The output includes `claims.jsonl`,
`quarantine.jsonl`, `claims.sqlite`, and a readable `review.md`. SQLite is a
derived artifact: deleting it and regenerating the report does not lose the
retained inputs or model responses.

Named-anchor scoring checks the statement, then its qualifications and citation.
It does not count words appearing only in the evidence quotation as a successful
claim. Regex misses and compound assertions require qualitative review. The
pair task receives supplied claims and is **not** a test of discovering every
contradiction end to end from the original documents.

## Curate additional documents

The same pipeline accepts new UTF-8 text/Markdown files without gold annotations:

```bash
$PY claim-curation/ingest.py --root ../xdna2-notes \
  --paths raw/phlegm/2026-09-26-aie2p-arithmetic-and-stack-traps.md \
  --out claim-curation/work/notes-batch
$PY claim-curation/run_models.py --model nuextract3 \
  --checkpoint models/claim-curation/nuextract3 --data claim-curation/work/notes-batch \
  --out claim-curation/work/notes-results --cache claim-curation/cache --threads 4
$PY claim-curation/curator.py --data claim-curation/work/notes-batch \
  --out claim-curation/work/notes-results
```

`--chunk-chars` is a soft paragraph/fence-aware size target. Oversize fenced
blocks are retained and marked. This chunker is not a parser for PDFs or an
assurance that dependencies across sections were preserved. Source bytes are
snapshotted unchanged; reingestion keeps any existing annotations. Missing gold
annotations do not count as an expected-empty/no-claims test.

Review proposed statements against their quotations and full source context
before promoting them. The ingestion smoke check used one existing finding;
the complete repository has not been run through the model.

## Model references

- [NuExtract 2.0 2B](https://huggingface.co/numind/NuExtract-2.0-2B), revision
  `5eb4a99746edc2ac44900f56a5c9431d3c03b16d`.
- [NuExtract3](https://huggingface.co/numind/NuExtract3), revision
  `c99dc8f5641b866aa0192b6ea78f84bf9f3535f1`; native structured mode,
  task instructions and developer-example format; thinking disabled.
- [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507),
  revision `cdbee75f17c01a7cc42f958dc650907174af0554`.

Once this extraction/review contract is dependable, the same records can support
an entity/relation index, GraphRAG, cited documentation changes and training data.
The next evaluation should use independently reviewed annotations and whole
source documents, including tables and longer citation chains.

## Teacher-supervised LoRA pilot

`training/run_all.py` runs a separate free, local experiment: a pinned Qwen3.5-27B
Q4_K_M teacher proposes source-grounded claims and critiques its own output;
NuExtract3 trains text-only LoRA adapters on source-reviewed eligible training proposals. Validation and test references are authored separately by the interactive coding agent before student training/inference. All original
weights and vision parameters remain frozen. This tests the value of stronger
supervision; teacher size does not establish annotation correctness.

The fixed split contains 12 training, four validation and four test passages
from ten documents in separate projects. Earlier benchmark documents are
excluded. Each passage includes source metadata, the document title and its
nearest section heading. Upstream references remain unverified. Exact source
snapshots, teacher requests/responses, critiques and rejected attempts are kept.
Whitespace-only evidence alignment must match one unique supplied source span
and is logged; changed words or numbers are rejected. At most one teacher
correction attempt is allowed before quarantine.

The bounded pilot retains nine training passages after source review, one annotated validation passage and all four test passages. Three other validation passages remain unannotated. Teacher self-critique missed scope errors and rejected a supported extraction; explicit agent review decisions preserve both cases. Dense validation attempts exceeded the teacher output cap and are retained. These are agent-authored references, not independent human gold.

The training run uses 24 optimizer steps, rank-8 LoRA in the last eight text
layers, prompt-masked loss, and document-disjoint validation. Test labels never
enter optimization or validation. Base and adapted models then generate claims
on held-out notes with the same native prompt. Citation validity, teacher-quote
overlap and qualifier retention are diagnostics; source review is still needed
to assess semantic support. No claims or source notes are automatically promoted.

See [the fixed protocol](training/protocol.json), [live job status](training/status.json),
and `training/{label,train,evaluate}.log`. The isolated AMD runtime and the
precompiled-operator setting are recorded in the protocol and runtime checks.
The synthetic backward-pass adapter under `training/work/` is a compatibility
fixture and is excluded from the real training/evaluation run.

From this directory, using the existing CPU Python to orchestrate the phases:

```bash
../target/manifold-audit/env/bin/python training/run_all.py
```

To repeat only training and evaluation against the frozen reviewed supervision, add `--from-phase train`. A full replay generates only training proposals, then `finalize_annotations.py` applies retained review decisions and the separate held-out references. Refreshing the document snapshots requires reviewing/replacing those reference annotations; they must not silently transfer to changed inputs.

The runner acquires Dibs GPU/port locks per phase, releases them between phases,
and records failures in `training/status.json`. It calls no hosted inference
service and provisions no paid infrastructure. The local GPU run requires the
isolated `training/rocm-env/`; its package URLs and archive hashes are retained
in `training/rocm-install-plan.json`.

### Completed LoRA pilot result

[Full comparison and source review](training/comparison.json). The 24-step
rank-8 adapter trained on nine reviewed passages, with one validation reference
and four fixed test passages. Validation loss changed from 0.1285 to 0.1318.
Mechanically valid cited outputs fell from 4/4 held-out passages to 3/4. The
adapter recovered fork/runtime details but dropped numeric observations, merged
assertions and emitted a condition absent from the source. Both models still
need review for third-party-report attribution.

This adapter is retained at `training/runs/lora-pilot/final/` as an experimental
artifact. Keep the base model for proposed extraction; improve reviewed
supervision and evaluation coverage before another training run. This small
pilot does not establish that fine-tuning cannot help. Sixteen checks passed;
all compute was local and paid service cost was $0. GPU and port holds released.

The comparison does not use the exact-qualifier diagnostic as scope accuracy:
its requirement to repeat whole reference headlines is too strict here. Quote
overlap also does not establish semantic recall. Held-out annotations are
agent-authored and upstream references remain unverified. Reference provenance
wording in the generation metadata was corrected after inference; original
files are retained and raw generated text was unchanged.
