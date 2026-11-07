"""Run a model on shape-port tasks and score the answers.

    python3 run.py --backend codex --model gpt-6.1-sol --effort medium --tasks pilot
    python3 run.py --backend claude --model claude-opus-5-5 --effort xhigh --tasks all --jobs 4
    python3 run.py --backend llama --model qwen3.5-27b-q4km --url http://127.0.0.1:8091 --tasks pilot

`claude` uses the Claude Code CLI signed in with the user's claude.ai
subscription (no API billing; --bare is avoided because it only takes an API
key): one `claude -p` per task in an empty temp directory, prompt on stdin,
all tools off (`--tools ""`), no MCP servers, no settings files, a one-line
system prompt. ~/.claude/CLAUDE.md still loads (a few hundred tokens of
unrelated notes), as ~/.codex/AGENTS.md does for codex.
`codex` uses the Codex CLI signed in with the user's ChatGPT account (no API
billing): one `codex exec` per task in an empty read-only sandbox, prompt on
stdin, told to use no tools. `null` answers every task with no edit (A unchanged): the control for
`npu_check.py`, since a task where A itself matches B can't score answers.
`llama` posts to an OpenAI-compatible
llama-server with thinking off. Results go to runs/<tag>/<task>/ and a
summary to runs/<tag>/summary.json. Finished tasks are skipped on rerun.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import tempfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
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


CLAUDE_SYSTEM = "You are an expert assistant. Answer the user's request directly in your reply."


def run_claude(text: str, model: str, effort: str, timeout: int) -> dict:
    env = {k: v for k, v in os.environ.items() if not (k == "CLAUDECODE" or k.startswith("CLAUDE_CODE_"))}
    with tempfile.TemporaryDirectory(prefix="nkc-claude-") as d:
        t0 = time.time()
        r = subprocess.run(
            ["claude", "-p", "--model", model, "--effort", effort, "--tools", "", "--strict-mcp-config",
             "--setting-sources", "", "--no-session-persistence", "--system-prompt", CLAUDE_SYSTEM,
             "--output-format", "json"],
            input=text, capture_output=True, text=True, timeout=timeout, cwd=d, env=env,
        )
        try:
            j = json.loads(r.stdout)
        except json.JSONDecodeError:
            j = {}
        u = j.get("usage", {})
        usage = {k: u.get(k) for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens", "output_tokens")}
        usage["thinking_tokens"] = (u.get("output_tokens_details") or {}).get("thinking_tokens")
        return {"answer": j.get("result", "") if not j.get("is_error") else "", "seconds": round(time.time() - t0, 1),
                "exit": r.returncode, "usage": usage, "stop_reason": j.get("stop_reason"), "num_turns": j.get("num_turns"),
                "list_cost_usd": j.get("total_cost_usd"), "is_error": j.get("is_error"),
                "stderr_tail": (r.stderr[-400:] or r.stdout[-400:]) if (r.returncode or not j) else ""}


THINK = re.compile(r"<think>.*?(</think>|$)", re.S)


def run_llama(text: str | list, url: str, model: str, max_tokens: int, timeout: int, thinking: bool = False,
              temperature: float | None = 0.0) -> dict:
    messages = text if isinstance(text, list) else [{"role": "user", "content": text}]
    body = {"model": model, "messages": messages,
            "max_tokens": max_tokens, "chat_template_kwargs": {"enable_thinking": thinking}, "seed": 20261105}
    if temperature is not None:  # None: the GGUF's own sampling defaults
        body["temperature"] = temperature
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        j = json.loads(resp.read())
    msg = j["choices"][0]["message"]
    content = msg.get("content") or ""
    stripped = THINK.sub("", content)  # reasoning the server didn't split out; drafts inside would confuse the scorer
    return {"answer": stripped, "seconds": round(time.time() - t0, 1), "finish": j["choices"][0].get("finish_reason"),
            "usage": j.get("usage", {}), "timings": j.get("timings", {}),
            "reasoning_chars": len(msg.get("reasoning_content") or "") + len(content) - len(stripped)}


def feedback(s: dict, meta: dict, files: list[str]) -> str | None:
    """What the toolchain alone says about an answer (nothing from build B):
    None when the diff applies cleanly, assembles and passes the rule gate."""
    issues = []
    if not s["parsed"]:
        issues.append("No unified diff was found in your reply.")
    if s["unknown_files"]:
        issues.append(f"These files are not in the build: {', '.join(s['unknown_files'])}. The build's files are: {', '.join(files)}.")
    if s.get("failed_hunks"):
        listed = "\n".join(f"  {h}" for h in s["failed_hunks"])
        issues.append(f"{s['hunks_total'] - s['hunks_ok']} of {s['hunks_total']} hunks could not be placed: their context and "
                      f"removed lines do not occur in build `{meta['a_name']}`. First lines of those hunks:\n{listed}")
    if not s["builds"]:
        issues.append(f"The patched build does not assemble: {s.get('build_error', '?')}")
    elif not (s["rules_pass"] or s.get("rules_pass_aligned")):
        listed = "\n".join(f"  {v}" for v in s.get("rule_violations", []))
        issues.append(f"The patched build assembles but breaks hardware rules:\n{listed}")
    if not issues:
        return None
    return ("Your diff was checked with the npuasm toolchain.\n\n" + "\n\n".join(issues) +
            f"\n\nReply with a corrected unified diff from `{meta['a_name']}` to `{meta['b_name']}`: the complete diff "
            "against the original files of the first build, not a diff on top of your previous reply. ONLY the diff.")


ROUND_KEYS = ("parsed", "hunks_ok", "hunks_total", "builds", "rules_pass", "rules_pass_aligned", "exact", "gap_closed", "bytes_match")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["codex", "claude", "llama", "null"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--url", default="http://127.0.0.1:8091")
    ap.add_argument("--max-tokens", type=int, default=12000)
    ap.add_argument("--thinking", choices=["off", "on"], default="off", help="llama: chat template enable_thinking")
    ap.add_argument("--temperature", default="0", help="llama: a number, or 'default' for the GGUF's sampling defaults")
    ap.add_argument("--tasks", default="pilot", help="pilot | all | dev | test | with-example | comma-separated ids")
    ap.add_argument("--variant", choices=["base", "ex1"], default="base", help="ex1: add a worked example pair from the same design")
    ap.add_argument("--tag")
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--jobs", type=int, default=1, help="tasks in flight at once")
    ap.add_argument("--rounds", type=int, default=1, help="llama: up to N attempts; after each, the toolchain's own "
                    "feedback (hunks that don't apply, assembler error, rule violations; nothing from build B) goes back to the model")
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
    tag = a.tag or (f"{a.model}-{a.effort}" if a.backend in ("codex", "claude") else a.model)
    if a.variant != "base":
        tag += f"-{a.variant}"
    if a.rounds > 1:
        tag += f"-r{a.rounds}"
    root = HERE / "runs" / tag
    root.mkdir(parents=True, exist_ok=True)
    todo = [(k, m) for k, m in enumerate(chosen) if not (root / m["id"] / "result.json").exists()]

    def one(km):
        k, m = km
        td = HERE / "tasks" / m["id"]
        rd = root / m["id"]
        rd.mkdir(parents=True, exist_ok=True)
        text = prompt.build(td, example=a.variant == "ex1")
        (rd / "prompt.txt").write_text(text)
        try:
            if a.backend == "codex":
                out = run_codex(text, a.model, a.effort, a.timeout)
            elif a.backend == "claude":
                out = run_claude(text, a.model, a.effort, a.timeout)
            elif a.backend == "null":  # control: answer with no edits, i.e. A unchanged
                out = {"answer": "", "seconds": 0}
            else:
                temp = None if a.temperature == "default" else float(a.temperature)
                files = sorted(p.name for p in (td / "a").iterdir())
                msgs, rounds, out = text, [], {}
                # Round 1 of an rN run is the plain run's request (same prompt, temperature 0, fixed
                # seed), so its answer is reused when the plain run (tag without -rN) has the task.
                first = HERE / "runs" / tag.removesuffix(f"-r{a.rounds}") / m["id"]
                for r in range(1, a.rounds + 1):
                    if r == 1 and a.rounds > 1 and (first / "result.json").exists() and (first / "prompt.txt").read_text() == text:
                        o = {"answer": (first / "answer.txt").read_text(), "reused_from": str(first.relative_to(HERE)),
                             **json.loads((first / "result.json").read_text())["run"]}
                    else:
                        o = run_llama(msgs, a.url, a.model, a.max_tokens, a.timeout, thinking=a.thinking == "on", temperature=temp)
                    s = score.score(td, o["answer"])
                    fb = feedback(s, m, files) if r < a.rounds else None
                    if a.rounds > 1:
                        (rd / f"answer.r{r}.txt").write_text(o["answer"])
                        rounds.append({"round": r, **{k2: v for k2, v in o.items() if k2 != "answer"},
                                       "score": {k2: s.get(k2) for k2 in ROUND_KEYS}, "feedback": fb})
                    out = o
                    if fb is None:
                        break
                    msgs = [{"role": "user", "content": text}, {"role": "assistant", "content": o["answer"]},
                            {"role": "user", "content": fb}]
                if rounds:
                    out = {**out, "seconds": round(sum(x.get("seconds", 0) for x in rounds), 1), "rounds": rounds}
        except Exception as e:
            out = {"answer": "", "error": f"{type(e).__name__}: {e}"}
        (rd / "answer.txt").write_text(out.get("answer", ""))
        res = {**m, "run": {k2: v for k2, v in out.items() if k2 != "answer"}, "score": score.score(td, out.get("answer", ""))}
        (rd / "result.json").write_text(json.dumps(res, indent=1) + "\n")
        s = res["score"]
        print(f"[{k + 1}/{len(chosen)}] {m['id']} {m['difficulty']:6s} exact={s['exact']} gap={s['gap_closed']} "
              f"hunks={s['hunks_ok']}/{s['hunks_total']} builds={s['builds']} rules={s['rules_pass']} "
              f"bytes={s['bytes_match']} {out.get('seconds', '?')}s{' ERR ' + str(out.get('error') or out.get('stderr_tail'))[:120] if not out.get('answer') else ''}",
              flush=True)

    with ThreadPoolExecutor(max(1, a.jobs)) as ex:
        list(ex.map(one, todo))
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
