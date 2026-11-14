# Whole-corpus XDNA2 claim extraction

This run uses the pinned **base NuExtract3** checkpoint, with no LoRA adapter and
no hosted inference. The corpus frozen on 2026-10-25 contains 590 files: 588
Markdown/text/RST documents plus two JSON issue/comment snapshots. Their body
fields produce 632 document units and 937 initial passages.

The run directory is `../work/xdna2-all-20261025/` relative to this directory.
It contains original byte-preserved snapshots under `data/snapshots/`, decoded
JSON body views under `data/units/`, and a manifest recording original paths,
hashes, repository revision and JSON pointers. Markdown headings, frontmatter,
nearby context, code fences and table column labels accompany split passages.
Code, hidden configuration directories and ignored generated artifacts are not
part of the documentation corpus. The JSON snapshots' administrative API fields
are retained in the originals; their issue/comment body text is extracted.

The output files, created once the GPU run begins, are:

- `status.json`: current state, attempted source files and proposal counts.
- `claims.proposed.jsonl`: claims with exact quotations and condition fragments.
- `quarantine.jsonl`: invalid citations and terminal passage-output failures.
- `passages.jsonl`: every passage outcome, including recursive subdivisions.
- `raw.jsonl` and `cache/`: original completions, prompts and cache keys.
- `source-coverage.json`: per-source counts, failures and empty extractions.
- `claims.sqlite`: indexed derived export, created at completion.
- `summary.json`: final counts and limitations.

Every claim remains `review_status: proposed`, with
`semantic_support_verified: false`. Original notes, findings, recommendations
and open-question statuses are not changed. Source metadata is retained as
source metadata; an evidence label on a document does not verify every claim in
it. Upstream links and experiments have not been independently checked.

Plain-text citations use original character/line spans. JSON citations use the
original snapshot hash, a JSON pointer such as `/12/body`, and character/line
spans within that decoded field. They do not pretend a decoded newline is a
verbatim byte sequence in serialized JSON. Separate source files and comments
retain separate provenance even if their text matches.

Capped or malformed outputs are retained and subdivided into smaller target
passages, at most three levels deep. Failed terminal passages and invalid claims
remain visible for review. There is no JSON or quotation repair. Empty output
does not prove that a passage contains no claims. Completion means every frozen
source passage was attempted; it does not certify complete factual recall.

The native structured extraction prompt is versioned in `prompt.json`; it asks
for target-passage assertions, preserved uncertainty and attribution, and exact
condition fragments. Context is supplied to interpret a target, not to create
duplicate metadata observations. These instructions do not guarantee compliance.

Run or resume from the `claim-curation` directory:

```bash
dibs run gpu --preemptible --requeue \
  --label "Extract claims from all 590 XDNA2 documents" --eta 6h -- \
  training/rocm-env/bin/python -u bulk/run.py \
  --out work/xdna2-all-20261025
```

The installed Dibs launcher supplies the chat's identity. The worker checks the
yield file and exits 75 for cooperative requeue. Completed model responses are
cached independently, so yielding/restarting rebuilds exports from retained
responses and continues pending passages. Partial export counts can temporarily
reset during that rebuild. A run-local file lock prevents concurrent writers.

The runtime uses the existing isolated AMD environment and precompiled operator
settings from the training pilot. Greedy extraction batches up to eight inputs;
GPU memory exhaustion reduces the batch size. No adapter is loaded. This is not
a performance benchmark. Paid service cost is $0, and no credentials are needed.

To create a deliberately new frozen corpus, choose a new directory and run
`bulk/corpus.py --root ../../xdna2-notes --out NEW_RUN/data` using a Python with
PyYAML available. Reusing the current manifest extracts the retained snapshots,
even if live notes are subsequently updated. Do not overwrite its snapshots.
