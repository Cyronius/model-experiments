"""Build sources.jsonl: messy English of mixed length, from instruction data and this repo's docs.

  python fetch_sources.py --out out/sources.jsonl --n 20000 --md-dir ../..

Needs `datasets` and network for the dolly part; skips it with a warning otherwise.
Each row: id, text (one sentence up to six), mode (PROC for instructions, DESC otherwise).
"""
import argparse, hashlib, json, random, re
from pathlib import Path

SENT = re.compile(r"(?<=[.!?])\s+")


def chunk(text: str, rng: random.Random):
    sents = [s.strip() for s in SENT.split(re.sub(r"\s+", " ", text)) if len(s.split()) >= 4]
    i = 0
    while i < len(sents):
        k = rng.choice([1, 1, 2, 3, 4, 6])
        yield " ".join(sents[i:i + k]); i += k


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--md-dir", default=None); ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args(); rng = random.Random(a.seed); rows = []
    def add(text, mode):
        text = text.strip()
        if 5 <= len(text.split()) <= 160:
            rows.append({"id": hashlib.md5(text.encode()).hexdigest()[:12], "text": text, "mode": mode})
    try:
        from datasets import load_dataset
        for r in load_dataset("databricks/databricks-dolly-15k", split="train"):
            for c in chunk(r["instruction"], rng): add(c, "PROC")
            for c in chunk(r["context"] or r["response"], rng): add(c, "DESC")
    except Exception as e:
        print("warning: dolly skipped:", e)
    if a.md_dir:
        for p in Path(a.md_dir).rglob("*.md"):
            if "node_modules" in p.parts or ".git" in p.parts: continue
            body = re.sub(r"```.*?```", " ", p.read_text(encoding="utf-8", errors="ignore"), flags=re.S)
            body = re.sub(r"^[#|>\-*].*$", " ", body, flags=re.M)
            for c in chunk(body, rng): add(c, "DESC")
    uniq = list({r["id"]: r for r in rows}.values()); rng.shuffle(uniq); uniq = uniq[: a.n]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in uniq: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(uniq)} sources -> {a.out}")


if __name__ == "__main__":
    main()
