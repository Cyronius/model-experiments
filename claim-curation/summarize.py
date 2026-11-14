#!/usr/bin/env python3
"""Render the pilot measurements without turning extraction into verified truth."""
import argparse
import json
from pathlib import Path

from curator import report


def summarize(root):
    models = []
    for path in sorted((root / "results").iterdir()):
        if not path.is_dir() or not (path / "raw.jsonl").exists():
            continue
        report(root / "data", path)
        score_path = path / "extraction-scores-refined.json"
        if not score_path.exists():
            score_path = path / "extraction-scores.json"
        extraction = json.loads(score_path.read_text())
        relation = json.loads((path / "relation-scores.json").read_text())
        execution = json.loads((path / "execution.json").read_text())
        models.append((path.name, extraction, relation, execution))
    lines = ["# XDNA2 claim-curation pilot", "", "Local CPU inference; no new fine-tuning or paid compute.", "",
             "The fixed dataset contains 12 passages: nine excerpts from five project notes and three synthetic controls. "
             "There are 33 named claim anchors and 10 separately supplied claim pairs. "
             "Annotations were written by the coding agent before benchmark inference; they are not independent human ground truth.", "",
             "| Model | Valid extraction JSON | Named anchors matched | Conditions retained too | Valid citations/records | False claims on questions-only input | Pair labels correct | Capped outputs |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, e, r, execution in models:
        t = e["totals"]
        lines.append(f'| {name} | {t["schema_valid_passages"]}/{t["passages"]} | {t["anchor_matches"]}/{t["gold_anchors"]} | '
                     f'{t["scoped_and_cited_matches"]}/{t["gold_anchors"]} | {t["verbatim_cited_claims"]}/{t["claims"]} | '
                     f'{t["claims_on_empty_control"]} | {r["correct"]}/{r["total"]} | {t["truncated_passages"]} |')
    lines += ["", "A named anchor is a preregistered set of regex checks on the generated statement; source quotations are "
              "not counted as an answer. The conditions column additionally requires the specified qualifiers and exact "
              "source/condition fragments. These are diagnostic coverage checks, not semantic precision or full claim recall. "
              "A source quotation can be accurate while the generated statement misrepresents it. Every exported record stays proposed.", "",
              "The table uses a post-inference matcher refinement: outcome words such as works/worked are bounded so "
              "that 'workers' cannot satisfy a successful-outcome anchor. Original frozen gold and original scores are retained "
              "in gold.json and extraction-scores.json; refined anchors, rationale and scores are separate files. "
              "This fixes a known false positive; it does not turn lexical scoring into semantic adjudication. "
              "See [qualitative review](../qualitative-review.md) for preserved-context and support failures.", "",
              "The relation task receives supplied claims. It measures pair classification separately, not end-to-end discovery "
              "of contradictions from raw documents. Version/workload differences and unknown scope are deliberately included.", "",
              "All records cite exact spans in retained input snapshots. Upstream paths and commits mentioned in project notes "
              "are preserved verbatim, but those upstream sources were not independently fetched or verified. These inputs are "
              "project notes/digests, not uniformly original experiment logs.", "",
              "Prompts, examples, gold anchors and pair labels are fixed in prompt.json and data/. "
              "Only separate synthetic development fixtures were used for prompt-format calibration. "
              "NuExtract3 was added after the older 2B model returned empty lists on the first real passages. "
              "Models use their native chat formats with the same task instructions, templates and development examples.", "",
              "## Per-model details"]
    for name, e, r, execution in models:
        lines += ["", f"### {name}", "", f'Generated {e["totals"]["claims"]} claim records; '
                  f'{e["totals"]["possible_compound_claims"]} records cover multiple named anchors and need atomicity review.', "",
                  "| Passage | Named anchors | Conditions and citations | Records | Schema valid |", "|---|---:|---:|---:|---|"]
        for case in e["passages"]:
            n = len(case["targets"])
            a = sum(bool(t["matched_claims"]) for t in case["targets"])
            s = sum(bool(t["scoped_and_cited_claims"]) for t in case["targets"])
            lines.append(f'| {case["id"]} | {a}/{n} | {s}/{n} | {case["claims"]} | {not case["schema_errors"]} |')
        lines += ["", "| Pair | Expected | Actual | Correct |", "|---|---|---|---|"]
        for pair in r["pairs"]:
            lines.append(f'| {pair["id"]} | {pair["expected"]} | {pair["actual"]} | {pair["correct"]} |')
        inference = sum(b["seconds"] for b in execution["batches"])
        tokens = sum(b["output_tokens"] for b in execution["batches"])
        lines += ["", f'Observed generation batches took {inference:.1f} seconds for {tokens:,} output tokens. '
                  "Runs may overlap and include prefill and padding; these timings are operational observations, not a controlled speed ranking.", "",
                  f'[All raw outputs and rendered prompts]({name}/raw.jsonl) · [All claim records]({name}/claims.jsonl) · '
                  f'[Citation failures]({name}/quarantine.jsonl) · [Execution configuration]({name}/execution.json)']
    (root / "results" / "summary.md").write_text("\n".join(lines) + "\n")
    print("Wrote", root / "results/summary.md")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path(__file__).parent)
    summarize(p.parse_args().root)
