"""Replace the 27B-MoE teacher's titles with Qwen3-14B-AWQ titles, gpt-6-luna for rows it fails (2026-10-16).

Both titling runs are scripts/04_generate_titles.py (resumable); see the titler-v2 plan for the launch steps.
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

OLD_TEACHER = "qwen36-27b-a2.8b-mtp-iq3exp-q4head"
SPLITS = ("train", "val", "test")


def read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


def verbatim_rates(rows: list[dict], titles: dict[str, str]) -> dict[str, str]:
    """Share of titles copied word-for-word from the prompt, normal bucket, by prompt length."""
    hits: Counter = Counter()
    totals: Counter = Counter()
    for r in rows:
        title = titles.get(r["id"])
        if not title or r.get("bucket") != "normal":
            continue
        n = len(r["prompt"])
        band = "<60" if n < 60 else "60-200" if n < 200 else "200+"
        totals[band] += 1
        hits[band] += title.lower() in r["prompt"].lower()
    return {b: f"{hits[b] / totals[b]:.1%} of {totals[b]}" for b in ("<60", "60-200", "200+") if totals[b]}


def select(args) -> None:
    teacher_of = {r["id"]: r.get("teacher") for r in read(Path(args.pool))}
    rows, old = [], []
    for split in SPLITS:
        for r in read(Path(args.splits) / f"{split}.jsonl"):
            if teacher_of.get(r["id"]) == OLD_TEACHER:
                rows.append({"id": r["id"], "bucket": r["bucket"], "prompt": r["prompt"]})
                old.append({"id": r["id"], "split": split, "title": r["title"]})
    out = Path(args.outdir)
    write(out / "input.jsonl", rows)
    write(out / "old_27b_titles.jsonl", old)
    random.seed(args.seed)
    write(out / "pilot_input.jsonl", random.sample(rows, min(args.pilot, len(rows))))
    print(f"{len(rows):,} rows to re-title -> {out / 'input.jsonl'} (pilot sample: {args.pilot})")


def compare(args) -> None:
    out = Path(args.outdir)
    new = read(Path(args.new))
    old_title = {r["id"]: r["title"] for r in read(out / "old_27b_titles.jsonl")}
    ok = {r["id"]: r["title"] for r in new if r.get("title")}
    print(f"titled {len(ok):,} / {len(new):,} ({len(new) - len(ok):,} failed after retries)")
    print("fail reasons:", Counter(r.get("fail_reason") for r in new if not r.get("title")).most_common(5))
    print("attempts:", dict(Counter(r.get("attempts") for r in new)))
    print("verbatim copies, 27B titles: ", verbatim_rates(new, old_title))
    print("verbatim copies, new titles: ", verbatim_rates(new, ok))
    random.seed(0)
    for r in random.sample([r for r in new if r["id"] in ok], min(args.show, len(ok))):
        print(f"  {old_title[r['id']]}  ->  {ok[r['id']]}")


def failed(args) -> None:
    rows = [{"id": r["id"], "bucket": r.get("bucket"), "prompt": r["prompt"]}
            for r in read(Path(args.new)) if not r.get("title")]
    write(Path(args.output), rows)
    print(f"{len(rows):,} rows failed -> {args.output}")


def merge(args) -> None:
    ok = {}
    for path in [args.new, *args.fallback]:  # earlier files win
        for r in read(Path(path)):
            if r.get("title"):
                ok.setdefault(r["id"], r["title"])
    retitled = {r["id"] for r in read(Path(args.outdir) / "old_27b_titles.jsonl")}
    for split in SPLITS:
        rows = read(Path(args.splits) / f"{split}.jsonl")
        kept, replaced, dropped = [], 0, 0
        for r in rows:
            if r["id"] in retitled:
                # no trusted title for this prompt, so leave it out rather than keep the 27B one
                if r["id"] not in ok:
                    dropped += 1
                    continue
                r["title"] = ok[r["id"]]
                replaced += 1
            kept.append(r)
        write(Path(args.dest) / f"{split}.jsonl", kept)
        print(f"{split}: {len(kept):,} rows ({replaced:,} re-titled, {dropped:,} dropped)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="data/v2", help="current train/val/test")
    ap.add_argument("--outdir", default="data/retitle")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--pool", default="data/16_full_titled_v15.jsonl", help="has the teacher field")
    s.add_argument("--pilot", type=int, default=500)
    s.add_argument("--seed", type=int, default=0)
    c = sub.add_parser("compare")
    c.add_argument("--new", required=True, help="04_generate_titles.py output")
    c.add_argument("--show", type=int, default=20)
    f = sub.add_parser("failed", help="write the rows a titling run failed, as input for the fallback run")
    f.add_argument("--new", required=True, help="04_generate_titles.py output")
    f.add_argument("--output", default="data/retitle/fallback_input.jsonl")
    m = sub.add_parser("merge")
    m.add_argument("--new", required=True, help="04_generate_titles.py output")
    m.add_argument("--fallback", nargs="*", default=[], help="titles for rows --new failed (e.g. gpt-6-luna output)")
    m.add_argument("--dest", default="data/v3")
    args = ap.parse_args()
    {"select": select, "compare": compare, "failed": failed, "merge": merge}[args.cmd](args)


if __name__ == "__main__":
    main()
