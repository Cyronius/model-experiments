"""Async client for any OpenAI-compatible API (LM Studio locally, vLLM on RunPod,
OpenRouter, ...), with title prompt template, validation, and retry policy shared
by the bake-off and bulk generation.

Set TITLER_API_BASE (and TITLER_API_KEY if the server needs auth) to target a
remote server; defaults to the local LM Studio endpoint."""

import asyncio
import os

import httpx

from .validation import clean_title, validate_title

BASE_URL = os.environ.get("TITLER_API_BASE", "http://localhost:1234/v1").rstrip("/")
API_KEY = os.environ.get("TITLER_API_KEY")
HEADERS = {"Authorization": f"Bearer {API_KEY}"} if API_KEY else {}

SYSTEM_PROMPT = (
    "You write short titles for chat conversations. Output ONLY the title: "
    "3-8 words, Title Case, no quotes, no emoji, no trailing punctuation, no markdown. "
    "Describe the user's request, not your answer to it."
)

# tight truncation: prompt processing dominates on iGPU/CPU boxes, and a title
# rarely needs more than the opening of the message
HEAD_CHARS, TAIL_CHARS = 900, 300


def truncate_prompt(prompt: str) -> str:
    if len(prompt) <= HEAD_CHARS + TAIL_CHARS:
        return prompt
    return prompt[:HEAD_CHARS] + "\n[...]\n" + prompt[-TAIL_CHARS:]


def user_message(prompt: str) -> str:
    return (
        "Write a title for a conversation that starts with this message:\n"
        f"<message>\n{truncate_prompt(prompt)}\n</message>"
    )


async def _post_with_retry(
    client: httpx.AsyncClient, body: dict, tries: int = 5, timeout: float = 180
) -> httpx.Response:
    """Retry 5xx/connect errors with backoff — LM Studio 500s while JIT-loading a model."""
    for i in range(tries):
        try:
            resp = await client.post(
                f"{BASE_URL}/chat/completions", json=body, headers=HEADERS, timeout=timeout
            )
            if resp.status_code >= 500 and i < tries - 1:
                await asyncio.sleep(2**i)
                continue
            resp.raise_for_status()
            return resp
        except (httpx.ConnectError, httpx.ReadTimeout):
            if i == tries - 1:
                raise
            await asyncio.sleep(2**i)
    raise RuntimeError("unreachable")


async def warmup(client: httpx.AsyncClient, model: str) -> None:
    """Force-load the model with one serial request before any concurrent traffic."""
    await _post_with_retry(
        client,
        {"model": model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 4},
    )


async def generate_title(
    client: httpx.AsyncClient,
    prompt: str,
    model: str,
    max_attempts: int = 3,
    no_think: bool = False,
    max_tokens: int | None = None,
) -> tuple[str | None, int, str | None]:
    """Returns (title, attempts_used, last_failure_reason)."""
    reason = None
    for attempt in range(1, max_attempts + 1):
        user = user_message(prompt)
        if no_think:
            user += " /no_think"
        if attempt == 3 and reason:
            user += f"\nYour previous title was rejected: {reason}. Follow the format rules exactly."
        # default max_tokens assumes chat_template_kwargs below might get ignored (true for
        # LM Studio) so leaves room for a think block anyway; a backend confirmed to honor the
        # kwarg (llama-server/vLLM) can pass an explicit small max_tokens instead -- the large
        # default costs real throughput on a server that occasionally doesn't cut reasoning off
        # cleanly even with the flag set. See titler-v1.5 data-pull notes.
        if max_tokens is None:
            max_tokens = 1024 if no_think else 24
        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            "temperature": 0.3 if attempt == 1 else 0.7,
            "max_tokens": max_tokens,
        }
        if no_think:
            # vLLM and llama-server honor this for Qwen3 hybrid-thinking models; LM Studio ignores it
            body["chat_template_kwargs"] = {"enable_thinking": False}
        resp = await _post_with_retry(client, body)
        choice = resp.json()["choices"][0]
        # a reply cut off by max_tokens can still look like a valid short title ("...pyridazinem")
        if choice.get("finish_reason") == "length":
            reason = "hit token limit"
            continue
        title = clean_title(choice["message"]["content"] or "", clamp=False)
        reason = validate_title(title, prompt)
        if reason is None:
            return title, attempt, None
    return None, max_attempts, reason


async def list_models(client: httpx.AsyncClient) -> list[str]:
    resp = await client.get(f"{BASE_URL}/models", headers=HEADERS, timeout=10)
    resp.raise_for_status()
    return [m["id"] for m in resp.json()["data"]]


async def main_smoke() -> None:
    async with httpx.AsyncClient() as client:
        models = await list_models(client)
        print("loaded models:", models)


if __name__ == "__main__":
    asyncio.run(main_smoke())
