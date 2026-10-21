"""Tiny client for an OpenAI-compatible chat server (vLLM or llama-server)."""
import json, time, urllib.request, urllib.error


def chat(url: str, messages: list[dict], temperature: float = 0.6, max_tokens: int = 700,
         timeout: int = 600, tries: int = 3) -> tuple[str, int]:
    # enable_thinking is for the Qwen teacher; reasoning_effort is for the gpt-oss judge. Each template ignores the other.
    body = json.dumps({"model": "x", "messages": messages, "temperature": temperature, "max_tokens": max_tokens,
                       "chat_template_kwargs": {"enable_thinking": False, "reasoning_effort": "low"}}).encode()
    for k in range(tries):
        try:
            req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=body,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                d = json.load(r)
            return (d["choices"][0]["message"].get("content") or "").strip(), d.get("usage", {}).get("completion_tokens", 0)
        except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError):
            if k == tries - 1: raise
            time.sleep(2 ** k)
