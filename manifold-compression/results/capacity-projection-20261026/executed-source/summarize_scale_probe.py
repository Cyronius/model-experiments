"""Compare completed scale-probe arms without treating KL as a quality gate."""
import argparse
import json
from pathlib import Path


def repeated_line_run(text):
    """Supplemental scan: word n-grams miss loops made only of punctuation.

    Added after observing repeated code fences in the Qwen2 75% arm. Report it
    separately and apply it uniformly; never rewrite the preregistered metric.
    """
    best = count = 0
    previous = None
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        count = count + 1 if line == previous else 1
        best = max(best, count)
        previous = line
    return best


def compare_generation(stock, candidate):
    before = {c["id"]: c for c in stock["cases"]}
    after = {c["id"]: c for c in candidate["cases"]}
    if before.keys() != after.keys():
        raise ValueError("Cannot compare incomplete or mismatched generation sets")
    scored = [k for k, c in before.items() if c["check_pass"] is not None]
    correct = [k for k in scored if before[k]["check_pass"]]
    lost = [k for k in correct if not after[k]["check_pass"]]
    gained = [k for k in scored if not before[k]["check_pass"] and after[k]["check_pass"]]
    retained = (len(correct) - len(lost)) / len(correct) if correct else None
    loops = sum(c["loop_flag"] for c in after.values())
    stock_loops = sum(c["loop_flag"] for c in before.values())
    ended = sum(c["ended"] for c in after.values())
    stock_ended = sum(c["ended"] for c in before.values())
    line_loops = [k for k, c in after.items() if repeated_line_run(c.get("answer", "")) >= 8]
    stock_line_loops = [k for k, c in before.items() if repeated_line_run(c.get("answer", "")) >= 8]
    return {"stock_correct": len(correct), "candidate_correct": sum(after[k]["check_pass"] for k in scored),
            "retained_stock_correct_fraction": retained, "lost_cases": lost, "gained_cases": gained,
            "candidate_loop_flags": loops, "stock_loop_flags": stock_loops,
            "candidate_completed": ended, "stock_completed": stock_ended,
            "supplemental_repeated_line_cases": line_loops, "stock_repeated_line_cases": stock_line_loops,
            "screen_pass": retained is not None and retained >= .95 and loops <= stock_loops and ended >= stock_ended - 1,
            "interpretation": "Passing is a reason for broader evaluation, not proof of parity; review long-form answers and loop flags."}


def summarize(folder):
    output = {}
    for name in ("qwen2", "qwen27", "mistral24"):
        root = folder / name
        path = root / "result.json"
        if not path.exists():
            output[name] = {"status": "not run"}
            continue
        run = json.loads(path.read_text())
        if run["status"] != "complete" or run.get("smoke_only"):
            output[name] = {"status": "incomplete or smoke only", "stage": run.get("stage")}
            continue
        stock = json.loads((root / "stock-generation.json").read_text())
        if stock["summary"]["cases"] != 72:
            raise ValueError("Expected all 72 frozen generation cases")
        arms = {}
        for label in ("retain-75", "retain-50"):
            generation = json.loads((root / f"{label}-generation.json").read_text())
            metrics = compare_generation(stock, generation)
            measured = run["arms"][label]
            arms[label] = metrics | {"parameters": measured["parameters"],
                "retained_fraction": measured["retained_fraction"], "validation": measured["validation"]}
        output[name] = {"status": "complete", "data_sha256": run["data_sha256"], "arms": arms}
    # Compare Qwen sizes only on tasks both original models answered correctly.
    if all(output[n]["status"] == "complete" for n in ("qwen2", "qwen27")):
        if output["qwen2"]["data_sha256"] != output["qwen27"]["data_sha256"]:
            raise ValueError("Different data in the size comparison")
        originals = {n: {c["id"] for c in json.loads((folder / n / "stock-generation.json").read_text())["cases"]
                         if c["check_pass"] is True} for n in ("qwen2", "qwen27")}
        common = originals["qwen2"] & originals["qwen27"]
        comparison = {"common_stock_correct_cases": len(common), "arms": {}}
        for label in ("retain-75", "retain-50"):
            counts = {}
            for name in ("qwen2", "qwen27"):
                cases = json.loads((folder / name / f"{label}-generation.json").read_text())["cases"]
                counts[name] = sum(c["id"] in common and c["check_pass"] is True for c in cases)
            comparison["arms"][label] = counts
        comparison["limits"] = "Small screen; Qwen sizes also differ in depth, head counts and embedding tying. Does not isolate a universal scaling law."
        output["qwen_scale_comparison"] = comparison
    return output


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("folder", type=Path)
    args = ap.parse_args()
    result = summarize(args.folder)
    (args.folder / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    for name, entry in result.items():
        print(name, entry.get("status", "comparison"))
        for label, arm in entry.get("arms", {}).items():
            print(" ", label, {k: v for k, v in arm.items() if k != "validation"})
