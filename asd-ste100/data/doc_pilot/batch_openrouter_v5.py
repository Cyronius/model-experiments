"""Many passages in one request to an OpenRouter model, then the v4 gates on each rewrite.

One request carries up to --batch passages, each with its id, mode, checker findings
and masked text, after the v5 system prompt (`--variant`, default traps) and a batch
instruction; the reply is one JSON object {"rewrites": [{"id", "rewrite", "uncertain"}]}.
Each rewrite gets the v4 marker check and gates. There are no retries and no meaning
check here: the output is a first-attempt measure to compare with the first attempts
of a local run. Free OpenRouter models allow 50 requests a day on an account that has
bought no credits, so a batch is the unit of cost. The key is read from
OPENROUTER_API_KEY. Outputs go to data/out/doc-teacher-v5/<run>/.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

sys.path[:0] = [str(Path(__file__).parent)]
from gates_v4 import marker_problems, restore_any_order, verify_v4  # noqa: E402
from ste_context_v4 import source_hints_v4  # noqa: E402
from ste_context_v5 import rewrite_prompt  # noqa: E402
from synthesize import mask_literals, sha  # noqa: E402
from checker import Lexicon  # noqa: E402
from gen import proper_names  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "data/out/doc-teacher-v5"
BATCH_NOTE = """BATCH. The user message gives many passages. Each starts with a line "### id: <id>" and has its own mode, checker findings and text. Rewrite each passage by itself, with the rules above; the literal markers of a passage ([[L0]], [[L1]], ...) belong to that passage only. Reply with one JSON object only, with one entry for each passage, in the same order: {"rewrites": [{"id": "<id>", "rewrite": "...", "uncertain": false}, ...]}"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--model", default="nvidia/nemotron-3-ultra-550b-a55b:free")
    parser.add_argument("--inputs", default=str(ROOT / "data/out/doc-pool-v5/fresh48.jsonl"))
    parser.add_argument("--batch", type=int, default=48)
    parser.add_argument("--variant", default="traps")
    parser.add_argument("--max-tokens", type=int, default=30000)
    parser.add_argument("--reasoning", default="low", help="reasoning effort, 'off' (disable thinking) or 'none' (send nothing)")
    args = parser.parse_args()
    key = os.environ["OPENROUTER_API_KEY"]
    lex = Lexicon()
    rows = [json.loads(l) for l in Path(args.inputs).read_text().splitlines()][:args.batch]
    out = BASE / args.run
    out.mkdir(parents=True, exist_ok=True)
    system = rewrite_prompt(lex, args.variant) + "\n\n" + BATCH_NOTE
    blocks, masked = [], {}
    for row in rows:
        marked, literals = mask_literals(row["text"])
        masked[row["id"][:12]] = (marked, literals)
        hints = source_hints_v4(row["text"], row["mode"], lex, proper_names(row["text"]))
        blocks.append(f"### id: {row['id'][:12]}\nMode: {row['mode']}\nDocument family: {row['family']}\n"
                      f"Checker findings on this passage:\n{hints}\nPassage:\n{marked}")
    user = "\n\n".join(blocks) + "\n\nRewrite each passage in STE. Keep the meaning exactly. Reply with the JSON object only."
    body = {"model": args.model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": 0.0, "seed": 42, "max_tokens": args.max_tokens}
    if args.reasoning == "off":
        body["reasoning"] = {"enabled": False}
    elif args.reasoning != "none":
        body["reasoning"] = {"effort": args.reasoning, "exclude": True}
    req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=3600) as r:
        reply = json.loads(r.read())
    seconds = time.time() - t0
    (out / "reply.json").write_text(json.dumps(reply, indent=1, ensure_ascii=False) + "\n")
    content = reply["choices"][0]["message"].get("content") or ""
    text = content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
    items = {i["id"]: i for i in json.loads(text[text.index("{"):text.rindex("}") + 1])["rewrites"]}
    results = []
    for row in rows:
        rid = row["id"][:12]
        marked, literals = masked[rid]
        item = items.get(rid)
        record = {**row, "batch_id": rid, "rewrite_with_markers": None, "ste": None, "issues": []}
        if not item or not isinstance(item.get("rewrite"), str):
            record["issues"] = ["missing_from_batch"]
        else:
            rewrite = item["rewrite"].strip()
            record["rewrite_with_markers"] = rewrite
            if marker_problems(rewrite, len(literals)):
                record["issues"] = ["marker_change"]
            else:
                target = restore_any_order(rewrite, literals)
                record["ste"] = target
                record["issues"] = verify_v4(row["text"], target, row["mode"], lex, bool(item.get("uncertain")),
                                             proper_names(row["text"]))
        record["gates_only"] = {"attempt": 1, "ste": record["ste"]} if record["ste"] and not record["issues"] else None
        results.append(record)
    for name, keep in (("accepted.jsonl", True), ("rejected.jsonl", False)):
        with (out / name).open("w", encoding="utf-8") as f:
            for r in results:
                if (not r["issues"]) == keep:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = {"run": args.run, "model": args.model, "variant": args.variant, "batch": len(rows), "seconds": round(seconds, 1),
               "usage": reply.get("usage"), "passed_gates_first_attempt": sum(1 for r in results if not r["issues"]),
               "missing": sum(1 for r in results if r["issues"] == ["missing_from_batch"]),
               "system_prompt_sha256": sha(system.encode())}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
