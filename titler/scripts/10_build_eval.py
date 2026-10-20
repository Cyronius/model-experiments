"""Fix the Phase 1 eval set and the v1.5 baseline titles (2026-10-17).

Usage: python scripts/10_build_eval.py [--model model/titler-v15-scratch-web] [--n 1000]
"""

import argparse
import json
import random
from pathlib import Path

from eval import PPL_FALLBACK_THRESHOLD, fallback_title, generate, load_model


def read(path: str) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model/titler-v15-scratch-web")
    ap.add_argument("--n", type=int, default=1000)
    args = ap.parse_args()

    test = read("data/v3/test.jsonl")
    random.Random(0).shuffle(test)
    rows = test[: args.n] + [{"id": r["id"], "bucket": "hard", "prompt": r["prompt"], "title": None}
                             for r in read("data/hardcases.jsonl")]
    out = Path("data/eval")
    with (out / "eval_set.jsonl").open("w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)

    tokenizer, model = load_model(args.model)
    titles = {}
    for r in rows:
        title, ppl = generate(tokenizer, model, r["prompt"])
        titles[r["id"]] = fallback_title(r["prompt"]) if ppl > PPL_FALLBACK_THRESHOLD else title
    (out / "titles_v15.json").write_text(json.dumps(titles, ensure_ascii=False, indent=0), encoding="utf-8")
    print(f"{len(rows)} eval rows, v1.5 titles -> {out / 'titles_v15.json'}")


if __name__ == "__main__":
    main()
