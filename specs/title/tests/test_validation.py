# Traces: TITLE-MAX-WORDS, TITLE-PRODUCT-CLAMP, TITLE-APOSTROPHE, TITLE-CASING
import asyncio
import json

import httpx

from common.lmstudio_client import generate_title
from common.validation import clean_title, validate_title

LONG = "Debugging Python Memory Leaks Inside Production Flask Apps Under Heavy Load"
PROMPT = "my flask app leaks memory in production, how do I find where it comes from?"


def teacher(*replies: tuple[str, str]):
    """Fake OpenAI-compatible server answering each request with the next (content, finish_reason)."""
    calls = []

    def handle(request: httpx.Request) -> httpx.Response:
        content, finish = replies[len(calls)]
        calls.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": content}, "finish_reason": finish}]})

    return httpx.MockTransport(handle), calls


def run_teacher(*replies):
    transport, calls = teacher(*replies)

    async def go():
        async with httpx.AsyncClient(transport=transport) as client:
            return await generate_title(client, PROMPT, "teacher-model")

    return asyncio.run(go()), calls


def test_overlong_teacher_title_is_retried_not_cut():
    (title, attempts, reason), calls = run_teacher((LONG, "stop"), ("Finding Flask Memory Leaks", "stop"))
    assert (title, attempts, reason) == ("Finding Flask Memory Leaks", 2, None)
    assert len(calls) == 2


def test_token_limit_reply_is_retried():
    (title, attempts, _), _ = run_teacher(
        ("Production Process of 6-Chloro-N-cyclopropyl-3-pyridazinem", "length"),
        ("Production Process of a Pyridazine Compound", "stop"),
    )
    assert (title, attempts) == ("Production Process of a Pyridazine Compound", 2)


def test_overlong_title_fails_validation():
    assert validate_title(clean_title(LONG, clamp=False), PROMPT) == "too long (11 words)"


def test_product_output_is_clamped_to_eight_words():
    assert clean_title(LONG) == "Debugging Python Memory Leaks Inside Production Flask Apps"


def test_apostrophes_inside_words_are_kept():
    assert clean_title("“Fixing My Children’s Bike”") == "Fixing My Children's Bike"
    assert clean_title("'Don't Panic' Guide") == "Don't Panic Guide"
    assert validate_title("Fixing My Children's Bike", PROMPT) is None
    assert validate_title('Fixing "My" Bike', PROMPT) == "contains quotes"


def test_acronyms_and_inner_capitals_keep_their_spelling():
    assert clean_title("iphone vs iPhone mRNA SQL v2 setup") == "Iphone vs iPhone mRNA SQL v2 Setup"
