"""Generate STE rewrites and messy back-translations with a local teacher.

  python gen.py rewrite --url http://127.0.0.1:8080 --input sources.jsonl --out out/seed.jsonl
  python gen.py messy   --url ... --input out/seed.jsonl --out out/messy.jsonl

rewrite: ask the teacher for an STE version, run the checker, and on error-level
violations send the list back and retry. Only checker-clean outputs are written to --out;
the rest go to --out.failed. Both modes resume: ids already written are skipped.
"""
import argparse, hashlib, json, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent)); sys.path.insert(0, str(Path(__file__).parent.parent / "checker"))
from checker import Lexicon, check, errors, describe  # noqa: E402
from llm import chat  # noqa: E402

SYSTEM = """You rewrite English into Simplified Technical English (ASD-STE100).
Rules:
- Use short, common words with one clear meaning. Do not use: ensure, prior to, utilize, in order to, perform, commence, approximately, and similar formal words. Use: make sure, before, use, to, do, start, about.
- Use the active voice. Do not use passive, perfect tenses (has done), or -ing forms of verbs.
- Instructions (mode PROC): imperative mood, one instruction per sentence, at most 20 words per sentence.
- Descriptions (mode DESC): at most 25 words per sentence, at most 6 sentences per paragraph.
- Keep articles (the, a, an). Do not use contractions or semicolons.
- Keep code, file names, identifiers, URLs, numbers, units, and proper names exactly as written.
- Keep the full meaning. Do not add or remove facts.
Output only the rewritten text."""

STYLES = ["verbose and formal, with passive voice", "casual and chatty, with contractions", "terse, with jargon and abbreviations",
          "bureaucratic, with long nested sentences", "a quick message from a busy engineer"]

lock = threading.Lock()


def done_ids(p: Path) -> set:
    return {json.loads(l)["id"] for l in p.open(encoding="utf-8")} if p.exists() else set()


def rewrite_one(row, a, lex):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": f"Mode: {row['mode']}\n\n{row['text']}"}]
    toks, out, errs = 0, "", []
    for attempt in range(1, a.retries + 1):
        out, n = chat(a.url, msgs, temperature=a.temperature)
        toks += n
        errs = errors(check(out, lex, row["mode"]))
        if not errs:
            return {"id": row["id"], "src": row["text"], "ste": out, "mode": row["mode"], "attempts": attempt}, toks, True
        msgs += [{"role": "assistant", "content": out},
                 {"role": "user", "content": "Fix these problems and output only the corrected text: " + describe(errs)}]
    return {"id": row["id"], "src": row["text"], "ste": out, "mode": row["mode"], "attempts": a.retries, "errors": describe(errs)}, toks, False


def messy_one(row, a, lex):
    style = STYLES[int(hashlib.md5(row["id"].encode()).hexdigest(), 16) % len(STYLES)]
    msgs = [{"role": "user", "content": f"Rewrite this text the way a person would normally write it: {style}. "
             f"Keep every fact, name, number, and code term. Do not simplify. Output only the rewritten text.\n\n{row['ste']}"}]
    out, n = chat(a.url, msgs, temperature=0.9)
    return {"id": row["id"], "ste": row["ste"], "messy": out, "style": style, "mode": row.get("mode", "DESC")}, n, True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["rewrite", "messy"]); ap.add_argument("--url", required=True)
    ap.add_argument("--input", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16); ap.add_argument("--retries", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.5); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    lex = Lexicon(); out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    failed = Path(str(out) + ".failed")
    seen = done_ids(out) | done_ids(failed)
    rows = [json.loads(l) for l in open(a.input, encoding="utf-8")]
    rows = [r for r in rows if r["id"] not in seen][: a.limit or None]
    fn = rewrite_one if a.cmd == "rewrite" else messy_one
    ok = bad = toks = 0; t0 = time.time()
    with ThreadPoolExecutor(a.workers) as ex, out.open("a", encoding="utf-8") as fo, failed.open("a", encoding="utf-8") as ff:
        futs = [ex.submit(fn, r, a, lex) for r in rows]
        for k, f in enumerate(as_completed(futs), 1):
            try: rec, n, good = f.result()
            except Exception as e:
                print("error:", e, file=sys.stderr); bad += 1; continue
            toks += n
            with lock:
                (fo if good else ff).write(json.dumps(rec, ensure_ascii=False) + "\n")
                fo.flush(); ff.flush()
            ok += good; bad += not good
            if k % 50 == 0 or k == len(futs):
                el = time.time() - t0
                print(f"[{k}/{len(futs)}] ok={ok} failed={bad} gen_tok={toks} {toks/max(el,1):.0f} tok/s {el:.0f}s", flush=True)
    print(json.dumps({"ok": ok, "failed": bad, "gen_tokens": toks, "seconds": round(time.time() - t0)}))


if __name__ == "__main__":
    main()
