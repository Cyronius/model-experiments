"""Replace the 27B-MoE teacher's titles with Qwen3-14B-AWQ titles, Claude Haiku for rows it fails (2026-10-16).

The Qwen titling itself is scripts/04_generate_titles.py (resumable); see the titler-v2 plan for the launch steps.
"""

import argparse
import asyncio
import json
import random
import time
from collections import Counter
from pathlib import Path

import anthropic

from common.lmstudio_client import SYSTEM_PROMPT, user_message
from common.validation import clean_title, validate_title

OLD_TEACHER = "qwen36-27b-a2.8b-mtp-iq3exp-q4head"
HAIKU = "claude-haiku-5-5"
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


async def haiku_title(client, prompt: str) -> tuple[str | None, int, str | None]:
    """Same prompt, validation and 3-try policy as lmstudio_client.generate_title, on Claude Haiku."""
    reason = None
    for attempt in range(1, 4):
        user = user_message(prompt)
        if attempt == 3 and reason:
            user += f"\nYour previous title was rejected: {reason}. Follow the format rules exactly."
        resp = await client.messages.create(
            model=HAIKU,
            max_tokens=100,
            thinking={"type": "disabled"},
            output_config={"effort": "low"},
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user}],
        )
        if resp.stop_reason in ("max_tokens", "refusal"):
            reason = f"stop_reason {resp.stop_reason}"
            continue
        raw = "".join(b.text for b in resp.content if b.type == "text")
        title = clean_title(raw, clamp=False)
        reason = validate_title(title, prompt)
        if reason is None:
            return title, attempt, None
    return None, 3, reason


async def run_haiku(rows: list[dict], out_path: Path, concurrency: int) -> None:
    done = {r["id"] for r in read(out_path)} if out_path.exists() else set()
    todo = [r for r in rows if r["id"] not in done]
    print(f"resuming: {len(done):,} done, {len(todo):,} to title with {HAIKU}")
    sem = asyncio.Semaphore(concurrency)
    stats = Counter()
    start = time.time()
    async with anthropic.AsyncAnthropic() as client:
        with out_path.open("a", encoding="utf-8") as f:
            async def one(r: dict) -> None:
                async with sem:
                    try:
                        title, attempts, reason = await haiku_title(client, r["prompt"])
                    except anthropic.APIStatusError as e:
                        title, attempts, reason = None, 0, f"api error {e.status_code}"
                    except anthropic.APIConnectionError as e:
                        title, attempts, reason = None, 0, f"connection error: {e}"
                f.write(json.dumps({"id": r["id"], "bucket": r.get("bucket"), "prompt": r["prompt"], "title": title,
                                    "teacher": HAIKU, "attempts": attempts, "fail_reason": reason},
                                   ensure_ascii=False) + "\n")
                f.flush()
                stats["ok" if title else "failed"] += 1
                if sum(stats.values()) % 100 == 0:
                    print(f"{sum(stats.values()):,}/{len(todo):,}  {sum(stats.values()) / (time.time() - start):.1f}/s  "
                          f"failed={stats['failed']}", flush=True)

            await asyncio.gather(*(one(r) for r in todo))
    print(f"done: {stats['ok']:,} ok, {stats['failed']:,} failed")


def haiku(args) -> None:
    if args.failed_from:
        rows = [r for r in read(Path(args.failed_from)) if not r.get("title")]
    else:
        rows = read(Path(args.input))
    asyncio.run(run_haiku(rows, Path(args.output), args.concurrency))


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
    h = sub.add_parser("haiku", help="title rows with Claude Haiku (needs ANTHROPIC_API_KEY)")
    src = h.add_mutually_exclusive_group(required=True)
    src.add_argument("--failed-from", help="04_generate_titles.py output; titles only its failed rows")
    src.add_argument("--input", help="any file of id/bucket/prompt rows, e.g. the pilot sample")
    h.add_argument("--output", required=True)
    h.add_argument("--concurrency", type=int, default=16)
    m = sub.add_parser("merge")
    m.add_argument("--new", required=True, help="04_generate_titles.py output")
    m.add_argument("--fallback", nargs="*", default=[], help="titles for rows --new failed (e.g. Haiku output)")
    m.add_argument("--dest", default="data/v3")
    args = ap.parse_args()
    {"select": select, "compare": compare, "haiku": haiku, "merge": merge}[args.cmd](args)


if __name__ == "__main__":
    main()
