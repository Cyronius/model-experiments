import concurrent.futures
import os
from pathlib import Path

import pytest

from titler import Titler, TitlerError, version

MODEL = Path(__file__).parents[2] / "model" / "titler-v3-q4_k_m.gguf"


@pytest.fixture(scope="module")
def titler():
    with Titler(MODEL) as t:
        yield t


def test_titles_a_prompt(titler):
    title = titler.title("how do I sort a list of dicts by a key in python")
    assert "python" in title.lower() and "sort" in title.lower()
    assert len(title.split()) <= 8


def test_non_ascii_prompt_round_trips(titler):
    assert titler.title("日本語のテキストを要約してください。東京の天気について教えて").strip() != ""


def test_threads_agree_with_one_thread(titler):
    prompts = ["write a poem about autumn rain", "what is the capital of australia", "fix this rust borrow checker error"]
    alone = [titler.title(p) for p in prompts]
    with concurrent.futures.ThreadPoolExecutor(3) as pool:
        together = list(pool.map(titler.title, prompts))
    assert alone == together


def test_model_from_bytes():
    with Titler(MODEL.read_bytes()) as t:
        assert t.title("hello there, how are you today").strip() != ""


def test_errors_are_exceptions():
    with pytest.raises(TitlerError):
        Titler("no-such-file.gguf")
    with pytest.raises(TitlerError):
        Titler(b"not a gguf")


def test_closed_titler_raises():
    t = Titler(MODEL)
    t.close()
    with pytest.raises(TitlerError):
        t.title("x")


def test_version():
    assert version() == "0.1.0"
