"""Batched OpenRouter calls for the v5 teacher: rewrites and meaning checks, many items a request.

`ask_batch` sends one request with a system prompt and numbered item blocks and returns
the parsed {"id": ...} entries. `rewrite_blocks` and `judge_blocks` build the blocks; a
retry block carries the previous rewrite and the checks it failed, so a batch can mix
first attempts and retries. Costs come back in the usage of each reply.
The key is read from OPENROUTER_API_KEY. TEACHER_URL points the calls at another OpenAI-compatible
server instead (a local llama-server: no key needed, and OpenRouter-only fields are left out).
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

URL = os.environ.get("TEACHER_URL", "https://openrouter.ai/api/v1/chat/completions")
LOCAL = "openrouter.ai" not in URL

REWRITE_BATCH = """BATCH. The user message gives many passages. Each starts with a line "### id: <id>" and has its own mode, checker findings and text. Some also give your previous rewrite of that passage and the checks it failed: fix every point for that passage. Rewrite each passage by itself, with the rules above; the literal markers of a passage ([[L0]], [[L1]], ...) belong to that passage only. Reply with one JSON object only, with one entry for each passage, in the same order: {"rewrites": [{"id": "<id>", "rewrite": "...", "uncertain": false}, ...]}"""

JUDGE_BATCH = """BATCH. The user message gives many MEANING CHECKs. Each starts with a line "### id: <id>". Do each check by itself, with the rules above. Reply with one JSON object only, with one entry for each check, in the same order: {"checks": [{"id": "<id>", "differences": [...]}, ...]}"""


def ask_batch(model: str, system: str, blocks: list[tuple[str, str]], key_name: str, reasoning: str = "off",
              max_tokens: int = 30000, retries: int = 4) -> tuple[dict, dict, float]:
    """blocks: (id, text). Returns ({id: entry}, usage, seconds). Raises after `retries` failed tries."""
    user = "\n\n".join(f"### id: {i}\n{text}" for i, text in blocks) + "\n\nReply with the JSON object only."
    body = {"model": model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": 0.0, "seed": 42, "max_tokens": max_tokens, "usage": {"include": True}}
    if LOCAL:
        del body["usage"]
        if reasoning == "off":
            body["chat_template_kwargs"] = {"enable_thinking": False}
    elif reasoning == "off":
        body["reasoning"] = {"enabled": False}
    elif reasoning != "none":
        body["reasoning"] = {"effort": reasoning, "exclude": True}
    headers = {"Content-Type": "application/json"}
    if not LOCAL:
        headers["Authorization"] = f"Bearer {os.environ['OPENROUTER_API_KEY']}"
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers=headers)
    last = None
    for attempt in range(retries):
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=3600) as r:
                reply = json.loads(r.read())
            text = (reply["choices"][0]["message"].get("content") or "").strip()
            entries = json.loads(text[text.index("{"):text.rindex("}") + 1])[key_name]
            return {str(e.get("id")): e for e in entries if isinstance(e, dict)}, reply.get("usage") or {}, time.time() - t0
        except (urllib.error.URLError, ConnectionError, TimeoutError, ValueError, KeyError, IndexError) as exc:
            last = exc
            time.sleep(min(60, 5 * 2 ** attempt))
    raise RuntimeError(f"batch failed after {retries} tries: {type(last).__name__}: {last}")
