"""Score a served model on tasks.py's held-out tasks (exact match, computed answers).

    python3 eval_small.py URL data/tasks/test.jsonl --out runs/NAME/eval-test.json [--per-kind 60] [--jobs 8]

URL is an OpenAI-compatible server (llama-server). Greedy, thinking off. Per kind: exact-match
rate, and line accuracy for the line-per-item kinds (enc, dec, legal). Ports are scored by applying
the answer diff to the source file (the eval's own hunk matcher): exact file, and, split by whether
the template inducer already gets that file exact, so the model's added value shows separately.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "eval"))
import score  # noqa: E402

LINE_KINDS = {"enc", "dec", "legal"}


def ask(url: str, msgs: list[dict], max_tokens: int) -> str:
    body = {"messages": msgs, "temperature": 0, "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        return json.load(r)["choices"][0]["message"]["content"] or ""


def source_of(user: str) -> str:
    return user[user.index("```\n") + 4: user.rindex("```")]


def apply(src: str, diff: str, f: str) -> str | None:
    hunks = score.parse_diff(diff).get(f, [])
    try:
        got, bad = score.apply(src.splitlines(), hunks)
    except Exception:  # noqa: BLE001
        return None
    return None if bad else "\n".join(got)


def grade(e: dict, ans: str) -> dict:
    gold = e["messages"][-1]["content"]
    ans = ans.strip().removeprefix("```").removesuffix("```").strip() if ans.strip().startswith("```") else ans.strip()
    r = {"id": e["id"], "kind": e["kind"], "exact": ans == gold.strip()}
    if e["kind"] in LINE_KINDS:
        g, p = gold.strip().splitlines(), ans.splitlines()
        r["lines"] = sum(x.strip() == y.strip() for x, y in zip(g, p)) / len(g)
    if e["kind"].startswith("port"):
        src, f = source_of(e["messages"][1]["content"]), e["meta"]["file"]
        want = apply(src, gold, f)
        got = apply(src, ans, f)
        r.update(exact=got is not None and got == want, applies=got is not None,
                 template_exact=e["meta"]["template_exact"], file=f)
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("tasks", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-kind", type=int, default=60)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    rows = [json.loads(l) for l in a.tasks.open()]
    by = collections.defaultdict(list)
    for e in rows:
        by[e["kind"]].append(e)
    rng = random.Random(a.seed)
    pick = [e for k in sorted(by) for e in rng.sample(by[k], min(a.per_kind, len(by[k])))]

    def run(e):
        gold = e["messages"][-1]["content"]
        try:
            ans = ask(a.url, e["messages"][:-1], max(256, int(len(gold) / 2) + 256))
        except Exception as x:  # noqa: BLE001
            ans = f"<error {x!r}>"
        return {**grade(e, ans), "answer": ans}

    with ThreadPoolExecutor(a.jobs) as pool:
        res = list(pool.map(run, pick))
    summ = {}
    for k in sorted(by):
        rs = [r for r in res if r["kind"] == k]
        if not rs:
            continue
        s = {"n": len(rs), "exact": round(sum(r["exact"] for r in rs) / len(rs), 3)}
        if k in LINE_KINDS:
            s["lines"] = round(sum(r["lines"] for r in rs) / len(rs), 3)
        if k.startswith("port"):
            hard = [r for r in rs if not r["template_exact"]]
            easy = [r for r in rs if r["template_exact"]]
            s.update(applies=round(sum(r["applies"] for r in rs) / len(rs), 3),
                     hard_n=len(hard), hard_exact=round(sum(r["exact"] for r in hard) / max(1, len(hard)), 3),
                     easy_n=len(easy), easy_exact=round(sum(r["exact"] for r in easy) / max(1, len(easy)), 3))
        summ[k] = s
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"summary": summ, "results": res}, indent=1) + "\n")
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
