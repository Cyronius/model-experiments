"""Export a small reviewable result bundle; weights and feature caches stay local."""
import argparse
import json
import shutil
from pathlib import Path

from data import BM25, ROOT, read_jsonl


def report(run, data, out, split="test"):
    run, data, out = Path(run), Path(data), Path(out)
    summary = json.loads((run / f"summary-{split}.json").read_text(encoding="utf-8"))
    training = json.loads((run / "training-summary.json").read_text(encoding="utf-8"))
    predictions = read_jsonl(run / f"predictions-{split}.jsonl")
    test_docs = {domain: len({p["doc_id"] for p in predictions if p["domain"] == domain})
                 for domain in summary["metrics"]}
    evaluated_ids = {p["id"] for p in predictions}
    properties = sorted({q["property"] for q in read_jsonl(data / "questions.jsonl")
                         if q["id"] in evaluated_ids and q.get("property")})
    out.mkdir(parents=True, exist_ok=True)
    retriever = BM25(read_jsonl(data / "documents.jsonl"))
    audit = {}
    for question in read_jsonl(data / "questions.jsonl"):
        if question["split"] == "test":
            counts = audit.setdefault(question["domain"], {"hits": 0, "n": 0})
            counts["n"] += 1
            counts["hits"] += retriever.retrieve(question["question"])[0] == question["doc_id"]
    (out / "retrieval-audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    for name in ["summary-test.json", "summary-dev.json", "training-summary.json", "training.jsonl",
                 "predictions-test.jsonl", "predictions-dev.jsonl", "arguments-all.json",
                 "arguments-evaluate-test.json", "arguments-evaluate-dev.json",
                 "interface-summary.json", "interface-control.jsonl", "execution-status.json",
                 "interface-reference-summary.json", "interface-reference-control.jsonl", "bank-diagnostics.json"]:
        if (run / name).exists():
            shutil.copyfile(run / name, out / name)
    shutil.copyfile(data / "manifest.json", out / "data-manifest.json")
    lines = [f"# Encoded-memory pilot: {split} gate", "",
             f"Recipient: `{summary['model']}`. Encoder: {training['encoder_parameters']:,} trainable parameters, "
             f"{summary['slots']} slots. Fixed training budget: {training['steps']} steps.", "",
             f"Memory insertion site: `{training.get('memory_site', summary.get('memory_site', 'prefix'))}`.", "",
             "Meta Llama 3.2 1B was inaccessible to the available credentials (HTTP 403). "
             "This run uses the ungated SmolLM2 checkpoint and the standard Llama implementation. "
             "It is independent of the model-shrinking experiment.", "",
             f"The recipient remained unchanged: **{training['recipient_unchanged']}**. "
             f"Training used {training['train_documents']} fictional records and {training['train_questions']} questions. "
             "All XDNA records and the test device/value bindings were held out from encoder training.", "",
             "| Domain | Condition | Numeric correct | Strict exact |", "|---|---|---:|---:|"]
    for domain, conditions in summary["metrics"].items():
        for condition, metrics in conditions.items():
            n = metrics["n"]
            lines.append(f"| {domain} | {condition} | {round(metrics['numeric_match'] * n)}/{n} "
                         f"({metrics['numeric_match']:.1%}) | {round(metrics['exact_match'] * n)}/{n} |")
    lines += ["", "Numeric scoring accepts units but rejects multiple numbers. Strict exact match accepts only the normalized answer.", ""]
    for domain, recovery in summary["paired_recovery"].items():
        lines.append(f"- {domain}: {recovery['eligible']} questions were missed without memory and answered with original text. "
                     f"Oracle recovery: {recovery['oracle']}; retrieved recovery: {recovery['retrieved']}.")
    if "retrieval_by_domain" in summary:
        lines += ["", "Exact source-ID retrieval:"]
        for domain, values in summary["retrieval_by_domain"].items():
            lines.append(f"- {domain}: {values['hits']}/{values['n']}.")
        lines.append("Another finding may repeat the same evidence, so exact source-ID misses are not always relevance failures.")
    synthetic = summary["metrics"].get("synthetic", {})
    if synthetic:
        oracle = synthetic["oracle"]["numeric_match"]
        control = max(synthetic["none"]["numeric_match"], synthetic["wrong"]["numeric_match"])
        if oracle <= control:
            conclusion = ("This configuration did not demonstrate usable encoding of new facts: correct encoded memories "
                          "did not outperform both the no-memory and wrong-memory controls on the evaluated synthetic questions. "
                          "A retriever cannot repair that encoding/reading failure.")
        else:
            conclusion = ("Correct encoded memories improved synthetic test accuracy over both controls. "
                          "This is preliminary evidence for this particular encoder/recipient pair, with a small "
                          "set of device/value bindings; it does not establish broad domain knowledge or reasoning.")
        lines += ["", "## Interpretation", "", conclusion]
        paired = {}
        for row in predictions:
            if row["condition"] in {"oracle", "wrong"}:
                paired.setdefault(row["id"], {})[row["condition"]] = row["prediction"]
        swaps = [pair for pair in paired.values() if "oracle" in pair and "wrong" in pair]
        changes = sum(pair["oracle"] != pair["wrong"] for pair in swaps)
        lines += ["", f"Swapping correct for deliberately wrong memory changed the generated answer in "
                  f"{changes}/{len(swaps)} evaluated questions."]
    diagnostics_path = run / "bank-diagnostics.json"
    if diagnostics_path.exists():
        diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
        lines += ["", f"The document-dependent component of the stored vectors was "
                  f"{diagnostics['document_variation_relative_to_rms']:.3%} of their overall RMS. "
                  "This measures variation across documents at each slot/coordinate; it does not measure "
                  "how many facts the vectors contain. Memories were stored in fp16."]
    if split == "dev" and synthetic and synthetic["oracle"]["numeric_match"] <= max(synthetic["none"]["numeric_match"], synthetic["wrong"]["numeric_match"]):
        lines += ["", "The first gate failed, so generator answers on the reserved test and the real XDNA probes were not evaluated. "
                  f"The sample contains {synthetic['oracle']['n']} development questions across "
                  f"{test_docs['synthetic']} independent fictional records, querying {', '.join(properties)}. "
                  "No configuration change or additional training was selected using reserved test answers."]
    retrieval_description = "; ".join(f"{domain}: {values['hits']}/{values['n']}" for domain, values in audit.items())
    lines += ["", "A separate, untrained BM25 diagnostic inspected source-ID retrieval on the reserved questions: "
              f"{retrieval_description} matched the designated source. "
              "Other findings may repeat valid evidence. Neither the retriever nor encoder was tuned using that check."]
    for interface_name in ["interface-summary.json", "interface-reference-summary.json"]:
        interface_path = run / interface_name
        if not interface_path.exists():
            continue
        interface = json.loads(interface_path.read_text(encoding="utf-8"))
        lines += ["", f"## Source embeddings at {interface.get('site', 'prefix')}", "",
                  f"Exact source token embeddings at `{interface.get('site', 'prefix')}` answered "
                  f"{interface['numeric_correct']}/{interface['n']} correctly "
                  f"({interface['exact_correct']}/{interface['n']} strict exact match). "
                  "This control uses the full token sequence without learned compression."]
        if interface.get("site", "prefix") == "prefix" and interface["numeric_correct"] < 15:
            lines += ["", "Prefix placement itself damages recall relative to ordinary reference text. "
                      "This run cannot isolate the compressor from the placement problem."]
    lines += ["", "## Limits and artifacts", "",
              f"The evaluated synthetic set has {test_docs.get('synthetic', 0)} independent documents, "
              "with correlated question variants. "
              f"This report includes {summary['metrics'].get('xdna', {}).get('oracle', {}).get('n', 0)} "
              f"numeric XDNA probes from {test_docs.get('xdna', 0)} findings. "
              "No confidence or generalization claim should be inferred from these small counts. "
              "The development and test facts are separate.", "",
              f"The bank contains {summary['bank_documents_encoded']} encoded records "
              f"({summary['memory_bank_bytes']:,} bytes of fp16 values). The source text is also retained in the BM25 index. "
              "Fewer input tokens does not imply fewer storage bytes than plain text. "
              "Timing numbers are local CPU measurements under variable system load, not a speed benchmark.", "",
              f"`predictions-{split}.jsonl` contains every answer and selected record. `summary-{split}.json` contains "
              "the aggregates. `training-summary.json` records the frozen-recipient weight hash. "
              "`data-manifest.json` records source location and generated dataset hashes. "
              "Regenerate data with the recorded seed and document counts; source findings may evolve.", "",
              "Encoder weights and the memory bank remain under the ignored local run folder. "
              "There was no donor model, cloud training, or modification of the XDNA source repository.", ""]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return out / "report.md"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", choices=["dev", "test"], default="test")
    args = parser.parse_args()
    print(report(args.run, args.data, args.out, args.split))
