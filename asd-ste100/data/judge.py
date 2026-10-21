"""Score meaning preservation of source -> STE pairs with a judge model.

  python judge.py --url http://127.0.0.1:18081 --input out/seed.jsonl --out out/judged.jsonl
Input rows need `id`, `src` and `ste`. Output adds `score` (1-5) and `changed` (true if facts differ).
Rows are written as they finish, so a stopped run resumes: ids already in --out are skipped.
"""
import argparse, json, re, sys, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from llm import chat

PROMPT = """Compare ORIGINAL and REWRITE. Rate how well REWRITE keeps the full meaning of ORIGINAL.
5 = same meaning. 3 = minor loss. 1 = meaning changed or facts added or dropped.
Reply with JSON only: {{"score": <1-5>, "changed": <true|false>}}

ORIGINAL:
{src}

REWRITE:
{ste}"""


def one(r, url):
    txt, _ = chat(url, [{"role": "user", "content": PROMPT.format(src=r["src"], ste=r["ste"])}], temperature=0.0, max_tokens=400)
    m = re.search(r"\{.*?\}", txt, re.S)
    try: j = json.loads(m.group(0))
    except Exception: j = {"score": None, "changed": None}
    return {**r, **j}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--url", required=True); ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--limit", type=int, default=0, help="0 = judge every row"); ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    out = Path(a.out)
    done = {json.loads(l)["id"] for l in out.open(encoding="utf-8")} if out.exists() else set()
    rows = [json.loads(l) for l in open(a.input, encoding="utf-8")][: a.limit or None]
    rows = [r for r in rows if r["id"] not in done]
    lock = threading.Lock(); res = []
    with ThreadPoolExecutor(a.workers) as ex, out.open("a", encoding="utf-8") as fo:
        for f in as_completed([ex.submit(one, r, a.url) for r in rows]):
            try: r = f.result()
            except Exception as e:
                print("error:", e, file=sys.stderr); continue
            with lock:
                fo.write(json.dumps(r, ensure_ascii=False) + "\n"); fo.flush()
            res.append(r)
    sc = [r["score"] for r in res if isinstance(r.get("score"), (int, float))]
    ch = sum(1 for r in res if r.get("changed") is True)
    print(json.dumps({"judged": len(res), "scored": len(sc), "mean_score": round(sum(sc) / max(len(sc), 1), 2), "changed": ch}))
