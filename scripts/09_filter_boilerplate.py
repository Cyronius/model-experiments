"""Drop v3 rows whose title narrates the user ("User Asks...", "User Expresses...") instead of naming the topic (2026-10-17)."""

import json
import re
from pathlib import Path

NARRATION = re.compile(
    r"^(the )?users?(['’]s)? (asks?|requests?|expresses|expressing|thanks|corrects?|frustrat\w*|criticiz\w*|dissatisf\w*|"
    r"clarif\w+|complain\w*|reacts?|wants?|needs?|seeks?|instructs?|instructions|inquir\w+|confirms?|praises?|"
    r"apologi[sz]\w*|rejects?|questions?|compliments?|demands?|disagrees?|challenges?|repeats?|indicates?|"
    r"provides?|shares?|states?|says?|responds?|reports?|greets?|insists?|notes?|comments?|feedback|query|prompt)\b",
    re.I,
)


def main() -> None:
    root, dropped = Path("data/v3"), []
    for split in ("train", "val", "test"):
        path = root / f"{split}.jsonl"
        rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
        keep = [r for r in rows if not NARRATION.search(r["title"])]
        dropped += [{"split": split, **r} for r in rows if NARRATION.search(r["title"])]
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in keep), encoding="utf-8")
        print(f"{split}: {len(keep):,} kept, {len(rows) - len(keep):,} dropped")
    (root / "dropped_narration.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in dropped), encoding="utf-8")


if __name__ == "__main__":
    main()
