"""Build the stage-1 SFT set: shape ports from the local corpus, in the eval's prompt format.

    python3 make_sft.py ../npuasm/results/corpus.jsonl data/sweep_corpus.jsonl [--tasks ../eval/tasks] [--out data] [--max-tokens 24576]

Each example is the stage-0 prompt (`eval/prompt.py`) for a pair of builds of one design (A, B),
and as the answer the exact unified diff from A's npuasm text to B's (difflib, 3 context lines,
which the eval's scorer applies exactly). Pairs come from the same groups as `eval/make_tasks.py`
(same design directory, build names equal once numbers become `#`, same core count and file
set), but every ordered pair of shapes up to 2 apart is a candidate, not one per group.

Held out, so the eval stays clean:
- every build made by a test-split generator (`eval/generators.py`), in any directory, and any
  build with the same name or content as one (the corpus has designs copied between projects);
- the reference build B of every eval task (dev and test), so no answer text leaks; A builds
  may appear, since the eval shows them anyway.

A share of examples (`--example-share`) carry a worked example from the same design, as in
`prompt.build(example=True)`: another pair's diff that never includes B. Families are split
train/val (`--val-share` of families) for loss tracking; the eval's dev tasks remain the
real check. Token counts use the Qwen3.5/3.6 tokenizer (`tokenizer.json`).

Writes data/train.jsonl, data/val.jsonl ({"id", "messages", "meta"}) and data/stats.json.
"""

from __future__ import annotations

import argparse
import collections
import difflib
import hashlib
import json
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "eval"))
sys.path.insert(0, str(HERE.parent / "npuasm"))

from npuasm import build  # noqa: E402
import prompt  # noqa: E402
import score  # noqa: E402
import generators  # noqa: E402
from make_tasks import HASH, numbers, template, diff_lines  # noqa: E402
from add_examples import changed  # noqa: E402

TOKENIZER = next(Path.home().glob(".cache/huggingface/hub/models--Qwen--Qwen3.5-2B/snapshots/*/tokenizer.json"), None)
CHAT_OVERHEAD = 16  # <|im_start|>user ... <|im_end|> etc.


def unified(a: dict[str, str], b: dict[str, str], n: int = 3) -> str:
    return "".join(
        "".join(difflib.unified_diff(a[f].splitlines(True), b[f].splitlines(True), f"a/{f}", f"b/{f}", n=n))
        for f in sorted(a)
    )


def lift(prj: str) -> tuple[str, dict[str, str] | None]:
    try:
        return prj, build.lift(Path(prj), None)[1]
    except Exception:
        return prj, None


def digest(texts: dict[str, str]) -> str:
    return hashlib.sha256(json.dumps(texts, sort_keys=True).encode()).hexdigest()


def bare(prj: str) -> str:
    return HASH.sub("", Path(prj).parent.name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus", type=Path, nargs="+", help="npuasm corpus rows (the local corpus, sweep output)")
    ap.add_argument("--tasks", type=Path, default=HERE.parent / "eval/tasks")
    ap.add_argument("--out", type=Path, default=HERE / "data")
    ap.add_argument("--max-tokens", type=int, default=24_576, help="prompt + answer, Qwen tokens")
    ap.add_argument("--max-diff", type=int, default=600, help="changed lines")
    ap.add_argument("--span", type=int, default=2, help="pair shapes up to this many positions apart")
    ap.add_argument("--per-group", type=int, default=16)
    ap.add_argument("--example-share", type=float, default=0.3)
    ap.add_argument("--max-example-chars", type=int, default=30_000)
    ap.add_argument("--val-share", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=20261108)
    ap.add_argument("--jobs", type=int, default=12)
    a = ap.parse_args()
    rng = random.Random(a.seed)

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(TOKENIZER))

    def ntok(s: str) -> int:
        return len(tok.encode(s, add_special_tokens=False).ids)

    index = json.loads((a.tasks / "index.json").read_text())
    eval_b = {m["b_prj"] for m in index}

    rows = [json.loads(l) for c in a.corpus for l in c.open()]
    rows = [r for r in rows if not r["error"] and all(r["ok"].values())]
    with ProcessPoolExecutor(a.jobs) as pool:
        texts = {p: t for p, t in pool.map(lift, [r["prj"] for r in rows], chunksize=8) if t is not None}

    # What must not appear: test-generator builds and eval B builds, by path, bare name and content.
    def test_gen(p: str) -> bool:
        return generators.split(str(Path(p).parent.parent), template(Path(p).parent.name)) == "test"

    banned_paths = {r["prj"] for r in rows if test_gen(r["prj"])} | eval_b
    banned_digests = {digest(texts[p]) for p in banned_paths if p in texts}
    banned_names = {(template(Path(p).parent.name), bare(p)) for p in banned_paths}
    drop = collections.Counter()

    groups = collections.defaultdict(list)
    for r in rows:
        p = r["prj"]
        if p not in texts:
            drop["lift failed"] += 1
            continue
        b = Path(p).parent
        if "#" not in template(b.name):
            continue
        why = ("test generator or eval B" if p in banned_paths else
               "same content as a held-out build" if digest(texts[p]) in banned_digests else
               "same name as a held-out build" if (template(b.name), bare(p)) in banned_names else None)
        if why:
            drop[why] += 1
            continue
        groups[(str(b.parent), template(b.name), r["cores"])].append(p)

    seen, examples = set(), []
    for (design, tmpl, cores), members in sorted(groups.items()):
        if len(members) < 2:
            continue
        members.sort(key=lambda p: numbers(Path(p).parent.name))
        cands = sorted(pair_candidates(members, a.span).items(), key=lambda kv: (kv[1], rng.random()))
        cands = [ab for ab, _ in cands]  # nearest shapes first, ties in random order
        kept = 0
        for pa, pb in cands:
            if kept >= a.per_group:
                break
            if bare(pa) == bare(pb):
                drop["names differ only by hash"] += 1
                continue
            ta, tb = texts[pa], texts[pb]
            if set(ta) != set(tb):
                drop["file sets differ"] += 1
                continue
            key = hashlib.sha256(json.dumps([tmpl, ta, tb], sort_keys=True).encode()).hexdigest()[:16]
            if key in seen:
                drop["duplicate pair"] += 1
                continue
            d = diff_lines(ta, tb)
            if not 1 <= d <= a.max_diff:
                drop["diff size"] += 1
                continue
            seen.add(key)
            meta = {"design": design.replace(str(Path.home()) + "/code/", ""), "template": tmpl,
                    "a_name": Path(pa).parent.name, "b_name": Path(pb).parent.name, "cores": cores,
                    "diff_lines": d, "difficulty": "easy" if d <= 10 else "medium" if d <= 80 else "hard"}
            answer = unified(ta, tb)
            # The target must apply exactly with the eval's own hunk matcher.
            hunks = score.parse_diff(answer)
            for f in ta:
                got, _ = score.apply(ta[f].splitlines(), hunks.get(f, []))
                if got != tb[f].splitlines():
                    drop["target does not apply"] += 1
                    break
            else:
                ex = None
                if rng.random() < a.example_share:
                    ex = pick_example(members, pa, pb, texts, a.max_example_chars)
                user = prompt.render(meta, ta, ex)
                n = ntok(user) + ntok(answer) + CHAT_OVERHEAD
                if n > a.max_tokens and ex:  # retry without the example
                    ex, user = None, prompt.render(meta, ta)
                    n = ntok(user) + ntok(answer) + CHAT_OVERHEAD
                if n > a.max_tokens:
                    drop["over token budget"] += 1
                    continue
                meta.update(tokens=n, answer_tokens=ntok(answer), example=ex is not None)
                examples.append({"id": f"S{key[:10]}", "messages": [{"role": "user", "content": user},
                                                                     {"role": "assistant", "content": answer}],
                                 "meta": meta})
                kept += 1

    families = sorted({e["meta"]["design"] for e in examples})
    rng.shuffle(families)
    val = set(families[: max(1, round(len(families) * a.val_share))])
    a.out.mkdir(parents=True, exist_ok=True)
    split = {"train": [e for e in examples if e["meta"]["design"] not in val],
             "val": [e for e in examples if e["meta"]["design"] in val]}
    stats = {"held_out": {"test_generators": sorted(generators.TEST), "eval_b_builds": len(eval_b)},
             "generators": dict(collections.Counter(generators.generator(e["meta"]["design"], e["meta"]["template"])
                                                    for e in examples)),
             "dropped": dict(drop)}
    for name, es in split.items():
        rng.shuffle(es)
        with (a.out / f"{name}.jsonl").open("w") as fh:
            for e in es:
                fh.write(json.dumps(e) + "\n")
        toks = sorted(e["meta"]["tokens"] for e in es)
        stats[name] = {
            "examples": len(es), "families": len({e["meta"]["design"] for e in es}),
            "groups": len({(e["meta"]["design"], e["meta"]["template"]) for e in es}),
            "difficulty": dict(collections.Counter(e["meta"]["difficulty"] for e in es)),
            "with_example": sum(e["meta"]["example"] for e in es),
            "tokens_total": sum(toks), "answer_tokens_total": sum(e["meta"]["answer_tokens"] for e in es),
            "tokens_median": toks[len(toks) // 2] if toks else 0, "tokens_max": toks[-1] if toks else 0,
        }
    (a.out / "stats.json").write_text(json.dumps(stats, indent=1) + "\n")
    print(json.dumps(stats, indent=1))


def pair_candidates(members: list[str], span: int) -> dict[tuple[str, str], int]:
    """Ordered pairs of builds whose shapes are near: one name number changed by at most `span`
    steps along its own axis (grid sweeps), or at most `span` apart in sorted order (other groups).
    Value: the distance in steps."""
    nums = [numbers(Path(p).parent.name) for p in members]
    out: dict[tuple[str, str], int] = {}
    for i in range(len(members)):
        for j in range(len(members)):
            if i != j and abs(i - j) <= span:
                out[(members[i], members[j])] = abs(i - j)
    if len({len(n) for n in nums}) == 1 and nums and nums[0]:
        axes = [sorted({n[k] for n in nums}) for k in range(len(nums[0]))]
        for i, x in enumerate(nums):
            for j, y in enumerate(nums):
                diff = [k for k in range(len(x)) if x[k] != y[k]]
                if len(diff) == 1:
                    k = diff[0]
                    d = abs(axes[k].index(x[k]) - axes[k].index(y[k]))
                    if d <= span:
                        key = (members[i], members[j])
                        out[key] = min(out.get(key, d), d)
    return out


def pick_example(members, pa, pb, texts, max_chars, window: int = 4):
    """Another pair of the group, never including B; prefer not A, then the same name positions changing."""
    want = changed(Path(pa).parent.name, Path(pb).parent.name)
    i = members.index(pa)
    near = members[max(0, i - window): i + window + 1]
    best = None
    for c in near:
        for d in near:
            if c == d or pb in (c, d) or bare(c) == bare(d) or set(texts[c]) != set(texts[pa]) or set(texts[d]) != set(texts[pa]):
                continue
            rank = (pa in (c, d), changed(Path(c).parent.name, Path(d).parent.name) != want)
            if best is not None and rank > best[0]:
                continue
            diff = unified(texts[c], texts[d], n=2)
            if not diff or len(diff) > max_chars:
                continue
            if best is None or (rank, len(diff)) < best[:2]:
                best = (rank, len(diff), Path(c).parent.name, Path(d).parent.name, diff)
    if best is None:
        return None
    rank, n, cn, dn, diff = best
    return ({"c_name": cn, "d_name": dn}, diff)


if __name__ == "__main__":
    main()
