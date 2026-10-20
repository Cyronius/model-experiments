"""Stage 1: download source datasets and extract first-turn English user prompts.

Sources:
  - allenai/WildChat-1M (primary; gated click-through, ODC-BY — accept terms on HF first)
  - lmsys/chatbot_arena_conversations (filler)
  - OpenAssistant/oasst2 (filler, Apache-2.0)
  - lmsys/lmsys-chat-1m (added for titler v1.5 -- gated click-through; barely overlaps the
    sources above (Chatbot Arena/Vicuna demo traffic, not WildChat), and has ~2x the hit rate
    for very_short/vague first-turn prompts per the titler-v1.5 investigation. See
    .claude/plans/ for the sampling that motivated pulling this one in.)

Usage:
    python scripts/01_download.py --output data/01_raw_prompts.jsonl [--source wildchat arena oasst2 lmsys]
"""

import argparse
import json
from pathlib import Path

from datasets import load_dataset


def wildchat():
    ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
    for i, row in enumerate(ds):
        if row.get("language") != "English":
            continue
        conv = row.get("conversation") or []
        if not conv or conv[0].get("role") != "user":
            continue
        text = (conv[0].get("content") or "").strip()
        if text and not conv[0].get("redacted", False):
            yield {"id": f"wc_{i}", "source": "wildchat", "prompt": text}


def arena():
    ds = load_dataset("lmsys/chatbot_arena_conversations", split="train", streaming=True)
    for i, row in enumerate(ds):
        if row.get("language") != "English":
            continue
        conv = row.get("conversation_a") or []
        if not conv or conv[0].get("role") != "user":
            continue
        text = (conv[0].get("content") or "").strip()
        if text:
            yield {"id": f"ar_{i}", "source": "arena", "prompt": text}


def oasst2():
    ds = load_dataset("OpenAssistant/oasst2", split="train", streaming=True)
    for i, row in enumerate(ds):
        # root user prompts only
        if row.get("parent_id") is not None or row.get("role") != "prompter":
            continue
        if row.get("lang") != "en":
            continue
        text = (row.get("text") or "").strip()
        if text:
            yield {"id": f"oa_{i}", "source": "oasst2", "prompt": text}


def lmsys_chat_1m():
    ds = load_dataset("lmsys/lmsys-chat-1m", split="train", streaming=True)
    for i, row in enumerate(ds):
        if row.get("language") != "English":
            continue
        if row.get("redacted"):
            continue
        conv = row.get("conversation") or []
        if not conv or conv[0].get("role") != "user":
            continue
        text = (conv[0].get("content") or "").strip()
        if text:
            yield {"id": f"ls_{i}", "source": "lmsys", "prompt": text}


SOURCES = {"wildchat": wildchat, "arena": arena, "oasst2": oasst2, "lmsys": lmsys_chat_1m}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--source", nargs="+", default=list(SOURCES), choices=list(SOURCES))
    ap.add_argument("--limit", type=int, default=None, help="max prompts per source (for smoke tests)")
    args = ap.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    with out_path.open("w", encoding="utf-8") as f:
        for name in args.source:
            n = 0
            for rec in SOURCES[name]():
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
                if args.limit and n >= args.limit:
                    break
            counts[name] = n
            print(f"{name}: {n} prompts")
    print(f"total: {sum(counts.values())} -> {out_path}")


if __name__ == "__main__":
    main()
