"""Lift every build that may reach training into one pool, for the task generators (tasks.py).

    python3 pool.py ../npuasm/results/corpus.jsonl data/sweep_corpus.jsonl [--out data/pool]

Same hold-out rules as make_sft.py: no build made by a test-split generator (eval/generators.py),
no eval reference build B, and nothing with the same content or name as one of those.

Writes:
- <out>/builds.jsonl.gz: {prj, design, name, template, cores, texts} per allowed build (texts are
  the npuasm files, exactly as `npuasm lift` writes them);
- <out>/bundles.tsv.gz: every distinct core bundle, `hex <TAB> text`, with numeric branch targets
  (no labels), and in how many distinct programs it appears;
- <out>/stats.json.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "eval"))
sys.path.insert(0, str(HERE.parent / "npuasm"))

from npuasm import build, core  # noqa: E402
import generators  # noqa: E402
from make_tasks import HASH, template  # noqa: E402


def lift(prj: str):
    try:
        _, texts, _ = build.lift(Path(prj), None)
        f = build.find(Path(prj))
        progs = {}
        for _, elf in sorted(f.cores.items()):
            seg = core.code_segment(core.read_elf(elf.read_bytes()))
            h = hashlib.sha256(seg.data).hexdigest()[:16]
            if h not in progs:
                progs[h] = [(i.raw.hex(), i.text) for i in core.lift(elf).insns]
        return prj, texts, progs
    except Exception as e:  # noqa: BLE001
        return prj, None, repr(e)


def digest(texts: dict[str, str]) -> str:
    return hashlib.sha256(json.dumps(texts, sort_keys=True).encode()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus", type=Path, nargs="+")
    ap.add_argument("--tasks", type=Path, default=HERE.parent / "eval/tasks")
    ap.add_argument("--out", type=Path, default=HERE / "data/pool")
    ap.add_argument("--jobs", type=int, default=12)
    a = ap.parse_args()

    index = json.loads((a.tasks / "index.json").read_text())
    eval_b = {m["b_prj"] for m in index}
    rows = [json.loads(l) for c in a.corpus for l in c.open()]
    rows = [r for r in rows if not r["error"] and all(r["ok"].values())]
    rows = list({r["prj"]: r for r in rows}.values())

    def test_gen(p: str) -> bool:
        return generators.split(str(Path(p).parent.parent), template(Path(p).parent.name)) == "test"

    texts, progs, drop = {}, {}, collections.Counter()
    with ProcessPoolExecutor(a.jobs) as pool:
        for p, t, pr in pool.map(lift, [r["prj"] for r in rows], chunksize=4):
            if t is None:
                drop["lift failed"] += 1
                continue
            texts[p], progs[p] = t, pr

    banned = {p for p in texts if test_gen(p)} | (eval_b & set(texts))
    banned_digests = {digest(texts[p]) for p in banned}
    bare = lambda p: HASH.sub("", Path(p).parent.name)  # noqa: E731
    banned_names = {(template(Path(p).parent.name), bare(p)) for p in banned | eval_b}
    # A core program seen in a held-out build is held out too, wherever else it appears.
    banned_progs = {h for p in banned for h in progs[p]}

    a.out.mkdir(parents=True, exist_ok=True)
    bundles: dict[str, list] = {}
    kept = 0
    seen_digest = set()
    with gzip.open(a.out / "builds.jsonl.gz", "wt") as fh:
        for r in rows:
            p = r["prj"]
            if p not in texts:
                continue
            b = Path(p).parent
            why = ("test generator or eval B" if p in banned else
                   "same content as a held-out build" if digest(texts[p]) in banned_digests else
                   "same name as a held-out build" if (template(b.name), bare(p)) in banned_names else
                   "duplicate content" if digest(texts[p]) in seen_digest else None)
            if why:
                drop[why] += 1
                continue
            seen_digest.add(digest(texts[p]))
            fh.write(json.dumps({"prj": p, "design": str(b.parent).replace(str(Path.home()) + "/code/", ""),
                                 "name": b.name, "template": template(b.name), "cores": r["cores"],
                                 "texts": texts[p]}) + "\n")
            kept += 1
            for h, lst in progs[p].items():
                if h in banned_progs:
                    continue
                for hx, t in lst:
                    e = bundles.setdefault(hx, [t, set()])
                    e[1].add(h)
    with gzip.open(a.out / "bundles.tsv.gz", "wt") as fh:
        for hx, (t, hs) in sorted(bundles.items()):
            fh.write(f"{hx}\t{t}\t{len(hs)}\n")
    stats = {"builds": kept, "bundles": len(bundles), "dropped": dict(drop),
             "designs": len({str(Path(p).parent.parent) for p in texts})}
    (a.out / "stats.json").write_text(json.dumps(stats, indent=1) + "\n")
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
