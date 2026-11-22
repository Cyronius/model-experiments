"""Curate append-form prompt groups (short prompt -> many appended details) with an OpenAI-style LLM server (2026-11-11).

`groups` runs locally (needs pyarrow). `run` needs only the standard library, so it can run on a worker host.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import random
import re
import sys
import time
import urllib.request
from pathlib import Path

SYSTEM = """You prepare training data for an image-prompt enricher.
You get a SHORT prompt and raw TAILS that real users appended to it after a comma.
Return JSON only.

Step 1, keep. keep=true when SHORT names something to draw: an object, creature, character, person type, place or scene.
Fictional characters and famous fictional places are fine. Spelling mistakes are fine.
keep=false only when SHORT is mostly a style, camera, film, lens, quality or artist tag with no subject, when it names a
real person, or when it is sexual or hateful.
Examples:
- "darth vader rising his beer in a tavern" -> keep (a scene)
- "ancient overgrown statue in a clearing" -> keep
- "a future urban city" -> keep
- "ektachrome, 50 mm portrait lens" -> drop (camera tags only)
- "hyperrealistic photography of a highly intricate" -> drop (no subject)
- "anya taylor-joy" -> drop (real person)
If keep=false, return empty lists.

Step 2, base: SHORT with spelling and spacing fixed ("1 0" -> "10", "dessert" -> "desert"). Do not add or remove content.

Step 3, details: at most {target} short natural phrases (3 to 12 words) taken from the TAILS, each adding visual detail:
setting, time, light, weather, mood, colour, material, pose, camera view or art style.
Remove artist names, website names, camera brands and quality tags ("8k", "highly detailed", "trending on artstation").
Lower case, no trailing period. Each must make sense appended to base after ", ". Merge near-duplicates.

Step 4, extra: new details so that details + extra has {target} items. Vary the kind of detail."""

SCHEMA = {
    "type": "object",
    "properties": {
        "keep": {"type": "boolean"},
        "base": {"type": "string"},
        "details": {"type": "array", "items": {"type": "string"}},
        "extra": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["keep", "base", "details", "extra"],
}
TAG = re.compile(r"\b(artstation|unreal engine|octane|cgsociety|8k|4k|trending|by [a-z]+ [a-z]+)\b")


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower()).strip(" ,")


def groups(args) -> None:
    import pyarrow.parquet as pq

    prompts = {norm(p) for p in pq.read_table(args.parquet, columns=["prompt"]).column(0).to_pylist() if p}
    tails: dict[str, set[str]] = {}
    for p in prompts:
        segs = [s.strip() for s in p.split(",") if s.strip()]
        for i in range(1, len(segs)):
            prefix = ", ".join(segs[:i])
            words = len(prefix.split())
            if words > args.max_words:
                break
            # Cheap pre-filter; the LLM makes the real keep/drop decision.
            if words >= args.min_words and not TAG.search(prefix):
                tails.setdefault(prefix, set()).add(", ".join(segs[i:]))
    rng = random.Random(args.seed)
    out = []
    for prefix, t in tails.items():
        if len(t) >= args.min_tails:
            sample = sorted(t)
            rng.shuffle(sample)
            out.append({"id": hashlib.sha1(prefix.encode()).hexdigest()[:12], "short": prefix,
                        "n_tails": len(t), "tails": sample[: args.max_tails]})
    rng.shuffle(out)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for g in out:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")
    print(f"{len(out)} groups from {len(prompts)} unique prompts -> {args.output}")


def call(url: str, model: str, group: dict, target: int, timeout: float, key_env: str = "",
         reasoning: str = "off", max_tokens: int = 600) -> dict:
    user = json.dumps({"SHORT": group["short"], "TAILS": group["tails"]}, ensure_ascii=False)
    body = {"model": model, "temperature": 0.7, "max_tokens": max_tokens,
            "messages": [{"role": "system", "content": SYSTEM.replace("{target}", str(target))},
                         {"role": "user", "content": user}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "curation", "schema": SCHEMA}},
            "chat_template_kwargs": {"enable_thinking": False}}
    # OpenRouter ignores chat_template_kwargs; some endpoints refuse to disable reasoning, so it is selectable.
    if reasoning == "off":
        body["reasoning"] = {"enabled": False}
    elif reasoning != "default":
        body["reasoning"] = {"effort": reasoning, "exclude": True}
    headers = {"Content-Type": "application/json"}
    if key_env:
        headers["Authorization"] = "Bearer " + os.environ[key_env]
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        resp = json.loads(r.read())
    data = json.loads(resp["choices"][0]["message"]["content"])
    return {"usage": resp.get("usage", {}), "provider": resp.get("provider"), **data}


def clean(phrases: list[str], base: str) -> list[str]:
    seen, out = set(), []
    for p in phrases:
        p = norm(p).rstrip(".")
        if 2 <= len(p.split()) <= 20 and p not in seen and p != norm(base):
            seen.add(p)
            out.append(p)
    return out


def run(args) -> None:
    todo = [json.loads(line) for line in args.groups.open(encoding="utf-8")][args.start:args.start + args.limit]
    done = set()
    if args.output.exists():
        done = {json.loads(line)["id"] for line in args.output.open(encoding="utf-8")}
    todo = [g for g in todo if g["id"] not in done]
    print(f"{len(todo)} groups to do, {len(done)} already done", flush=True)
    t0, n, tokens = time.time(), 0, 0
    with args.output.open("a", encoding="utf-8") as f, cf.ThreadPoolExecutor(args.parallel) as pool:
        futures = {pool.submit(call, args.url, args.model, g, args.target, args.timeout, args.key_env,
                               args.reasoning, args.max_tokens): g for g in todo}
        for fut in cf.as_completed(futures):
            g = futures[fut]
            try:
                r = fut.result()
                row = {"id": g["id"], "short": g["short"], "keep": bool(r["keep"]), "base": norm(r["base"]),
                       "details": clean(r["details"], r["base"]), "extra": clean(r["extra"], r["base"]),
                       "completion_tokens": r["usage"].get("completion_tokens"),
                       "prompt_tokens": r["usage"].get("prompt_tokens"), "cost": r["usage"].get("cost")}
            except Exception as exc:  # noqa: BLE001 - keep the batch moving; failures are retried on rerun
                print(f"{g['id']} failed: {exc}", file=sys.stderr, flush=True)
                continue
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            n += 1
            tokens += row["completion_tokens"] or 0
            if n % args.every == 0:
                dt = time.time() - t0
                print(f"{n} done, {dt / n:.1f} s/group, {tokens / dt:.1f} completion tok/s", flush=True)
    dt = time.time() - t0
    print(f"finished {n} groups in {dt:.0f} s ({dt / max(n, 1):.1f} s/group)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    g = sub.add_parser("groups")
    g.add_argument("--parquet", type=Path, required=True)
    g.add_argument("--output", type=Path, required=True)
    g.add_argument("--min-words", type=int, default=2)
    g.add_argument("--max-words", type=int, default=12)
    g.add_argument("--min-tails", type=int, default=5)
    g.add_argument("--max-tails", type=int, default=24)
    g.add_argument("--seed", type=int, default=0)
    r = sub.add_parser("run")
    r.add_argument("--groups", type=Path, required=True)
    r.add_argument("--output", type=Path, required=True)
    r.add_argument("--url", default="http://127.0.0.1:8080")
    r.add_argument("--model", default="local")
    r.add_argument("--key-env", default="", help="Environment variable that holds a bearer API key")
    r.add_argument("--reasoning", default="off", help="off, default, or an effort level such as low")
    r.add_argument("--max-tokens", type=int, default=600)
    r.add_argument("--target", type=int, default=8)
    r.add_argument("--parallel", type=int, default=2)
    r.add_argument("--start", type=int, default=0)
    r.add_argument("--limit", type=int, default=50)
    r.add_argument("--timeout", type=float, default=600)
    r.add_argument("--every", type=int, default=10)
    args = parser.parse_args()
    groups(args) if args.command == "groups" else run(args)


if __name__ == "__main__":
    main()
