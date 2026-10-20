"""Remove rows whose prompt or title pairs sexual content with a minor, from the training data (2026-10-18).

Keyword heuristic, tuned to over-remove: sexual term within 250 characters of a minor term, either order.
Covers every JSONL under data/. Prints counts only. Dry run unless --apply; --apply rewrites each file in place and logs removed ids.

    python scripts/17b_scrub_minors_sexual.py [--apply]
"""

import argparse
import json
import re
from pathlib import Path

SEXUAL = (r"sex\w*|porn\w*|nude\w*|naked|erotic\w*|rape\w*|molest\w*|incest\w*|fuck\w*|cock|penis|vagina\w*|pussy|"
          r"\bcum\b|horny|nsfw|lewd|masturbat\w*|blowjob|handjob|orgasm\w*|genital\w*|lingerie|fetish\w*|hentai|"
          r"strip(?:ping|per)?|breast\w*|boob\w*|sexy|seduc\w*|arous\w*|bdsm|bondage|spank\w*|undress\w*|panties|bikini")
MINOR = (r"child\w*|kids?|minors?|underage|under-age|toddlers?|infants?|pre-?teens?|teen\w*|schoolgirls?|schoolboys?|loli\w*|"
         r"shota|little (?:girl|boy)s?|young (?:girl|boy)s?|(?:[0-9]|1[0-7])\s*[- ]?(?:yo|y/o|y\.o\.?|yrs?|years?|y)(?![a-z])|"
         r"age[d]?\s*:?\s*(?:[0-9]|1[0-7])|(?:6th|7th|8th|9th|10th|11th) grade\w*|middle school\w*|"
         r"elementary school\w*|daughter|stepdaughter|niece|nephew")
NEAR = r"(?:.{0,250})"
PATTERN = re.compile(rf"\b(?:{SEXUAL}){NEAR}\b(?:{MINOR})|\b(?:{MINOR}){NEAR}\b(?:{SEXUAL})", re.I | re.S)

EXCLUDE = ("data/eval/human_read",)


def flagged(rec: dict) -> bool:
    return bool(PATTERN.search(rec.get("prompt") or "") or PATTERN.search(rec.get("title") or ""))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--only", nargs="*", help="scrub just these files instead of everything under data/")
    args = ap.parse_args()
    log = Path("data/scrub_removed_ids.json")
    removed: dict[str, list] = json.loads(log.read_text(encoding="utf-8")) if log.exists() else {}
    for path in [Path(p) for p in args.only] if args.only else sorted(Path("data").rglob("*.jsonl")):
        name = path.as_posix()
        if name.startswith(EXCLUDE):
            continue
        dropped_ids, total = [], 0
        tmp = path.with_suffix(".jsonl.tmp")
        out = tmp.open("w", encoding="utf-8") if args.apply else None
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                total += 1
                if flagged(rec):
                    dropped_ids.append(rec.get("id"))
                elif out:
                    out.write(line.rstrip("\r\n") + "\n")
        if out:
            out.close()
            if dropped_ids:
                tmp.replace(path)
            else:
                tmp.unlink()
        if dropped_ids:
            removed[name] = removed.get(name, []) + dropped_ids
        print(f"{name}: {len(dropped_ids):,} of {total:,} flagged")
    if args.apply:
        log.write_text(json.dumps(removed), encoding="utf-8")


if __name__ == "__main__":
    main()
