"""Run a model on shape-port tasks and score the answers.

    python3 run.py --backend codex --model gpt-6.1-sol --effort medium --tasks pilot
    python3 run.py --backend llama --model qwen3.5-27b-q4km --url http://127.0.0.1:8091 --tasks pilot

`codex` uses the Codex CLI signed in with the user's ChatGPT account (no API
billing): one `codex exec` per task in an empty read-only sandbox, prompt on
stdin, told to use no tools. `llama` posts to an OpenAI-compatible
llama-server with thinking off. Results go to runs/<tag>/<task>/ and a
summary to runs/<tag>/summary.json. Finished tasks are skipped on rerun.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

import prompt
import score

HERE = Path(__file__).parent


def pilot(index: list[dict], n_per: int = 4) -> list[dict]:
    """A fixed stratified subset: n easy, n medium, n hard, test split first."""
    out = []
    for diff in ("easy", "medium", "hard"):
        pool = sorted((m for m in index if m["difficulty"] == diff), key=lambda m: (m["split"] != "test", m["id"]))
        out += pool[:n_per]
    return out


def run_codex(text: str, model: str, effort: str, timeout: int) -> dict:
    with tempfile.TemporaryDirectory(prefix="nkc-codex-") as d:
        out = Path(d, "answer.txt")
        t0 = time.time()
        r = subprocess.run(
            ["codex", "exec", "-m", model, "-c", f'model_reasoning_effort="{effort}"', "--sandbox", "read-only",
             "--skip-git-repo-check", "--ephemeral", "-C", d, "--json", "-o", str(out), "-"],
            input=text, capture_output=True, text=True, timeout=timeout,
        )
        usage, tools = {}, 0
        for line in r.stdout.splitlines():
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get("type") == "turn.completed":
                usage = ev.get("usage", {})
            if ev.get("type") == "item.completed" and ev.get("item", {}).get("type") not in (None, "agent_message", "reasoning"):
                tools += 1
        return {"answer": out.read_text() if out.exists() else "", "seconds": round(time.time() - t0, 1),
                "exit": r.returncode, "usage": usage, "tool_items": tools, "stderr_tail": r.stderr[-400:] if r.returncode else ""}


def run_llama(text: str, url: str, model: str, max_tokens: int, timeout: int) -> dict:
    body = {"model": model, "messages": [{"role": "user", "content": text}], "temperature": 0,
            "max_tokens": max_tokens, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        j = json.loads(resp.read())
    return {"answer": j["choices"][0]["message"]["content"], "seconds": round(time.time() - t0, 1),
            "finish": j["choices"][0].get("finish_reason"), "usage": j.get("usage", {}), "timings": j.get("timings", {})}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["codex", "llama"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--url", default="http://127.0.0.1:8091")
    ap.add_argument("--max-tokens", type=int, default=12000)
    ap.add_argument("--tasks", default="pilot", help="pilot | all | dev | test | with-example | comma-separated ids")
    ap.add_argument("--variant", choices=["base", "ex1"], default="base", help="ex1: add a worked example pair from the same design")
    ap.add_argument("--tag")
    ap.add_argument("--timeout", type=int, default=3600)
    a = ap.parse_args()
    index = json.loads((HERE / "tasks/index.json").read_text())
    if a.tasks == "pilot":
        chosen = pilot(index)
    elif a.tasks == "all":
        chosen = index
    elif a.tasks == "with-example":
        chosen = [m for m in index if (HERE / "tasks" / m["id"] / "example.json").exists()]
    elif a.tasks in ("dev", "test"):
        chosen = [m for m in index if m["split"] == a.tasks]
    else:
        ids = set(a.tasks.split(","))
        chosen = [m for m in index if m["id"] in ids]
    if a.variant == "ex1":
        chosen = [m for m in chosen if (HERE / "tasks" / m["id"] / "example.json").exists()]
    tag = a.tag or (f"{a.model}-{a.effort}" if a.backend == "codex" else a.model)
    if a.variant != "base":
        tag += f"-{a.variant}"
    root = HERE / "runs" / tag
    root.mkdir(parents=True, exist_ok=True)
    for k, m in enumerate(chosen):
        td = HERE / "tasks" / m["id"]
        rd = root / m["id"]
        if (rd / "result.json").exists():
            continue
        rd.mkdir(parents=True, exist_ok=True)
        text = prompt.build(td, example=a.variant == "ex1")
        (rd / "prompt.txt").write_text(text)
        try:
            if a.backend == "codex":
                out = run_codex(text, a.model, a.effort, a.timeout)
            else:
                out = run_llama(text, a.url, a.model, a.max_tokens, a.timeout)
        except Exception as e:
            out = {"answer": "", "error": f"{type(e).__name__}: {e}"}
        (rd / "answer.txt").write_text(out.get("answer", ""))
        res = {**m, "run": {k2: v for k2, v in out.items() if k2 != "answer"}, "score": score.score(td, out.get("answer", ""))}
        (rd / "result.json").write_text(json.dumps(res, indent=1) + "\n")
        s = res["score"]
        print(f"[{k + 1}/{len(chosen)}] {m['id']} {m['difficulty']:6s} exact={s['exact']} gap={s['gap_closed']} "
              f"hunks={s['hunks_ok']}/{s['hunks_total']} builds={s['builds']} rules={s['rules_pass']} "
              f"bytes={s['bytes_match']} {out.get('seconds', '?')}s", flush=True)
    summarize(root)


def summarize(root: Path) -> dict:
    rs = [json.loads(p.read_text()) for p in sorted(root.glob("*/result.json"))]
    def agg(sel):
        if not sel:
            return {}
        g = [r["score"]["gap_closed"] for r in sel if r["score"]["gap_closed"] is not None]
        return {"n": len(sel), "exact": sum(r["score"]["exact"] for r in sel), "bytes_match": sum(r["score"]["bytes_match"] for r in sel),
                "builds": sum(r["score"]["builds"] for r in sel), "rules_pass": sum(r["score"]["rules_pass"] for r in sel),
                "gap_closed_median": round(statistics.median(g), 3) if g else None,
                "gap_closed_mean": round(statistics.mean(g), 3) if g else None,
                "seconds_median": round(statistics.median(r["run"].get("seconds", 0) or 0 for r in sel), 1)}
    summary = {"all": agg(rs), **{d: agg([r for r in rs if r["difficulty"] == d]) for d in ("easy", "medium", "hard")}}
    (root / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))
    return summary


if __name__ == "__main__":
    main()
