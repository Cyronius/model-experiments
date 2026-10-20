"""Eval set with 150 held-out rows per hard input type, plus v1.5 titles for them (2026-10-20).

    uv run python scripts/26_build_hard_eval.py
"""

import importlib.util
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from eval import PPL_FALLBACK_THRESHOLD, fallback_title, generate, load_model  # noqa: E402

PER_GROUP = 150
GROUPS = {
    "short": ("very_short", "vague"),
    "code": ("code_paste",),
    "log": ("log_paste",),
    "sql": ("sql",),
    "markdown": ("markdown",),
    "long": ("long",),
}


def read(path: str) -> list[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def main() -> None:
    spec = importlib.util.spec_from_file_location("strat", "scripts/03_stratify.py")
    strat = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(strat)
    held_out = read("data/v5/test.jsonl") + read("data/v5/val.jsonl")
    rng = random.Random(26)
    rows = []
    for group, buckets in GROUPS.items():
        pool = [r for r in held_out if (r.get("bucket") or strat.bucket(r["prompt"])) in buckets]
        rng.shuffle(pool)
        rows += [{"id": r["id"], "bucket": group, "prompt": r["prompt"]} for r in pool[:PER_GROUP]]
    rows += [{"id": r["id"], "bucket": "hard", "prompt": r["prompt"]} for r in read("data/hardcases.jsonl")]
    out = Path("data/eval")
    (out / "hard_eval.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    tokenizer, model = load_model("model/titler-v15-scratch-web")
    titles = {}
    for r in rows:
        title, ppl = generate(tokenizer, model, r["prompt"])
        titles[r["id"]] = fallback_title(r["prompt"]) if ppl > PPL_FALLBACK_THRESHOLD else title
    (out / "hard_titles_v15.json").write_text(json.dumps(titles, ensure_ascii=False), encoding="utf-8")
    print(len(rows), "rows;", {g: sum(r["bucket"] == g for r in rows) for g in [*GROUPS, "hard"]})


if __name__ == "__main__":
    main()
