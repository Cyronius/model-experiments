"""Independently rescore saved manifold IFEval responses and report instruction types."""
import argparse
from collections import defaultdict
import hashlib
import json
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runid", required=True)
    parser.add_argument("--literal-characters", action="store_true", help="Correct random substitution of requested punctuation; preserve native scores")
    args = parser.parse_args()
    out = ROOT / "manifold-compression/results" / args.runid
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    dataset = Path(manifest["dataset"])
    assert hashlib.sha256(dataset.read_bytes()).hexdigest() == manifest["dataset_sha256"]
    docs = [json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines() if line.strip()][:manifest["n_prompts"]]
    assert len(docs) == manifest["n_prompts"]
    assert sum(len(doc["instruction_id_list"]) for doc in docs) == manifest["n_instructions"]
    from langdetect import DetectorFactory
    DetectorFactory.seed = 0
    from lm_eval.tasks.ifeval import utils
    if args.literal_characters:
        from ifeval_scoring import install_literal_character_checker
        install_literal_character_checker()
    audit, breakdown = {}, {}
    corrected, scored_rows = {}, {}
    for model in manifest["models"]:
        work = out / model / "ifeval"
        files = list(work.glob("*.json"))
        assert {f.stem for f in files} == {str(doc["key"]) for doc in docs}, model
        by_type = defaultdict(lambda: {"strict_pass": 0, "loose_pass": 0, "n": 0})
        strict_prompts = loose_prompts = strict_instructions = loose_instructions = caps = 0
        differences = []
        scored_rows[model] = {}
        for doc in docs:
            row = json.loads((work / f"{doc['key']}.json").read_text(encoding="utf-8"))
            assert "error" not in row and row["key"] == doc["key"]
            assert 0 <= row["completion_tokens"] <= manifest["settings"]["max_tokens"]
            actual = utils.process_results(doc, [row["response"]])
            assert actual == utils.process_results(doc, [row["response"]]), (model, doc["key"], "unstable rescore")
            for saved, fresh in (("correct", "prompt_level_strict_acc"), ("loose", "prompt_level_loose_acc"),
                                 ("inst_strict", "inst_level_strict_acc"), ("inst_loose", "inst_level_loose_acc")):
                if row[saved] != actual[fresh]:
                    assert args.literal_characters and doc["key"] in (1122, 1129), (model, doc["key"], saved)
                    differences.append({"key": doc["key"], "field": saved, "original": row[saved], "corrected": actual[fresh]})
                row[saved] = actual[fresh]
            scored_rows[model][doc["key"]] = row
            assert len(row["inst_strict"]) == len(row["inst_loose"]) == len(doc["instruction_id_list"])
            strict_prompts += row["correct"]
            loose_prompts += row["loose"]
            strict_instructions += sum(row["inst_strict"])
            loose_instructions += sum(row["inst_loose"])
            caps += row["hit_cap"]
            for kind, strict, loose in zip(doc["instruction_id_list"], row["inst_strict"], row["inst_loose"]):
                group = by_type[kind]
                group["n"] += 1
                group["strict_pass"] += strict
                group["loose_pass"] += loose
        result = json.loads((out / f"result-{model}.json").read_text(encoding="utf-8"))
        if not args.literal_characters:
            assert result["prompt_pass_count"] == strict_prompts
            assert result["instruction_pass_count"] == strict_instructions
        assert result["instruction_count"] == manifest["n_instructions"]
        assert result["errors"] == 0
        for field, count, denominator in (
            ("ifeval_prompt_strict", strict_prompts, len(docs)),
            ("ifeval_prompt_loose", loose_prompts, len(docs)),
            ("ifeval_inst_strict", strict_instructions, manifest["n_instructions"]),
            ("ifeval_inst_loose", loose_instructions, manifest["n_instructions"]),
            ("cap_hit_rate", caps, len(docs)),
        ):
            if not args.literal_characters:
                assert abs(result[field] - count / denominator) <= 0.000051, (model, field)
        audit[model] = {
            "n_prompts": len(docs), "n_instructions": manifest["n_instructions"],
            "prompt_strict_pass": strict_prompts, "prompt_loose_pass": loose_prompts,
            "instruction_strict_pass": strict_instructions, "instruction_loose_pass": loose_instructions,
            "cap_hits": caps, "all_rescored_results_match": not differences,
            "repeat_rescoring_matches": True, "native_score_differences": differences,
        }
        corrected[model] = {
            "prompt_strict": strict_prompts / len(docs), "instruction_strict": strict_instructions / manifest["n_instructions"],
            "prompt_loose": loose_prompts / len(docs), "instruction_loose": loose_instructions / manifest["n_instructions"],
            "prompt_pass_count": strict_prompts, "instruction_pass_count": strict_instructions,
            "cap_hit_rate": caps / len(docs),
        }
        breakdown[model] = dict(sorted(by_type.items()))
        print(model, json.dumps(audit[model]), flush=True)
    suffix = "-corrected" if args.literal_characters else ""
    for name, value in ((f"scoring-audit{suffix}.json", audit), (f"instruction-breakdown{suffix}.json", breakdown)):
        (out / name).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    if args.literal_characters:
        paired = {}
        for first, second in combinations(scored_rows, 2):
            a, b = scored_rows[first], scored_rows[second]
            paired[f"{first} vs {second}"] = {
                "both_pass": sum(a[k]["correct"] and b[k]["correct"] for k in a),
                "first_only_pass": sum(a[k]["correct"] and not b[k]["correct"] for k in a),
                "second_only_pass": sum(not a[k]["correct"] and b[k]["correct"] for k in a),
                "both_fail": sum(not a[k]["correct"] and not b[k]["correct"] for k in a),
            }
        method = "Local literal-character correction for # and ! counts (keys 1122 and 1129); all other installed lm_eval 0.4.13 checks unchanged. Native results and responses preserved. Every corrected result rescored twice."
        (out / "comparison-corrected.json").write_text(json.dumps({"method": method, "models": corrected, "paired_prompt_strict": paired}, indent=2) + "\n", encoding="utf-8")
        lines = ["# IFEval: stock 0.8B, 300M and 500M", "", method, "",
                 f"{len(docs)} prompts; {manifest['n_instructions']} instruction checks per model. Matching Qwen sampling, thinking off, Q8_0, 1280-token output cap; one sample per prompt.", "",
                 "| Model | Prompt strict | Instruction strict | Prompt loose | Instruction loose |",
                 "|---|---:|---:|---:|---:|"]
        for model, scores in corrected.items():
            lines.append("| " + model + " | " + " | ".join(f"{100*scores[k]:.2f}%" for k in ("prompt_strict", "instruction_strict", "prompt_loose", "instruction_loose")) + " |")
        lines += ["", "Prompt scores require all instructions to pass; instruction scores count checks separately.",
                  "This local scoring correction affects only two prompts and does not change the comparison's ranking. These are not directly comparable to published scores using other decoding or scoring settings.",
                  "Native uncorrected results: summary.md and comparison.json. Exact changes: scoring-audit-corrected.json.",
                  "Upstream checker behavior: https://github.com/google-research/google-research/issues/3427", "",
                  "Paired strict outcomes: " + json.dumps(paired), ""]
        (out / "summary-corrected.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
