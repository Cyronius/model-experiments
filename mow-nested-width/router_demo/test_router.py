"""Tests for router_server.py. No models, no network, no GPU required.

Run from the repo root:
    python -m pytest router_demo/test_router.py
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from router_server import build_server, classify_message, route


def u(content: str) -> dict:
    return {"role": "user", "content": content}


def a(content: str) -> dict:
    return {"role": "assistant", "content": content}


# ---------------------------------------------------------------------------
# Unit tests: route()
# ---------------------------------------------------------------------------

def test_empty_messages_go_half():
    assert route([]) == ("half", "short-simple")


def test_plain_short_message_goes_half():
    tier, reason = route([u("what's the weather like today?")])
    assert tier == "half"
    assert reason == "short-simple"


def test_code_fence_triggers_full():
    tier, reason = route([u("explain this:\n```python\nprint('hi')\n```")])
    assert tier == "full"
    assert reason == "code"


def test_code_token_def_triggers_full():
    tier, reason = route([u("what does def foo(x) do in python")])
    assert (tier, reason) == ("full", "code")


def test_code_token_class_triggers_full():
    tier, reason = route([u("write a class Animal with a speak method")])
    assert (tier, reason) == ("full", "code")


def test_code_token_brace_triggers_full():
    tier, reason = route([u("fix this: if (x) { return 1; }")])
    assert (tier, reason) == ("full", "code")


def test_math_equals_triggers_full():
    tier, reason = route([u("solve for x: 3x + 2 = 11")])
    assert (tier, reason) == ("full", "math")


def test_math_caret_triggers_full():
    tier, reason = route([u("what is 2^10")])
    assert (tier, reason) == ("full", "math")


def test_math_word_derivative_triggers_full():
    tier, reason = route([u("what's the derivative of sin(x)")])
    assert (tier, reason) == ("full", "math")


def test_long_digit_run_triggers_full():
    tier, reason = route([u("is 1234567 prime")])
    assert (tier, reason) == ("full", "math")


def test_short_digit_run_does_not_trigger_math():
    tier, reason = route([u("I have 123456 dollars")])
    assert (tier, reason) == ("half", "short-simple")


def test_reasoning_word_prove_triggers_full():
    tier, reason = route([u("prove that the square root of 2 is irrational")])
    assert (tier, reason) == ("full", "reasoning-keyword")


def test_reasoning_word_debug_triggers_full():
    tier, reason = route([u("can you debug why my loop never exits")])
    assert (tier, reason) == ("full", "reasoning-keyword")


def test_reasoning_step_by_step_triggers_full():
    tier, reason = route([u("walk me through this step by step")])
    assert (tier, reason) == ("full", "reasoning-keyword")


def test_long_message_triggers_full():
    long_text = " ".join(["word"] * 130)
    tier, reason = route([u(long_text)])
    assert (tier, reason) == ("full", "long-message")


def test_message_at_length_boundary_does_not_trigger():
    text = " ".join(["word"] * 120)
    tier, reason = route([u(text)])
    assert (tier, reason) == ("half", "short-simple")


def test_stickiness_keeps_full_after_earlier_code_message():
    messages = [
        u("write a class Foo with a bar method"),
        a("here's an implementation ..."),
        u("thanks, now say hi"),
    ]
    tier, reason = route(messages)
    assert tier == "full"
    assert reason.startswith("stickiness:")
    assert "code" in reason


def test_stickiness_does_not_apply_downward_within_a_turn():
    # latest message itself is easy and no earlier user message was hard ->
    # half, proving stickiness only fires when an earlier turn was hard.
    messages = [u("hi there"), a("hello!"), u("how are you")]
    tier, reason = route(messages)
    assert (tier, reason) == ("half", "short-simple")


def test_stickiness_ignores_assistant_messages():
    # assistant reply happens to contain code-like tokens; only user turns
    # should ever move the tier.
    messages = [
        u("hi"),
        a("here is a class Foo { }"),
        u("say hi again"),
    ]
    tier, reason = route(messages)
    assert (tier, reason) == ("half", "short-simple")


@pytest.mark.parametrize("text,expected", [
    ("hello", None),
    ("```code```", "code"),
    ("class Foo", "code"),
    ("a = 1", "math"),
    ("2^2", "math"),
    ("the integral of x", "math"),
    ("please analyze this", "reasoning-keyword"),
    (" ".join(["w"] * 121), "long-message"),
])
def test_classify_message_cases(text, expected):
    assert classify_message(text) == expected


# ---------------------------------------------------------------------------
# Integration test: router_server with mock backends, no real models
# ---------------------------------------------------------------------------

class MockBackendHandler(BaseHTTPRequestHandler):
    """Stands in for llama-server's OpenAI-compatible endpoint."""

    reply_text = "mock reply"
    completion_tokens = 7

    def _send_json(self, code, obj):
        data = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, {"status": "ok"})
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path == "/v1/chat/completions":
            self._send_json(200, {
                "choices": [{"message": {"role": "assistant", "content": self.reply_text}}],
                "usage": {"completion_tokens": self.completion_tokens, "prompt_tokens": 3,
                          "total_tokens": self.completion_tokens + 3},
            })
        else:
            self._send_json(404, {"error": "not found"})

    def log_message(self, fmt, *args):
        pass  # silence request logging during tests


def _wait_until_accepting(url, timeout_s=5):
    """Poll until a socket connection to `url` succeeds. ThreadingHTTPServer's
    listen socket is already bound by the time its constructor returns, but
    on Windows loopback the very first connection right after thread start
    can occasionally be refused; this removes that flakiness from tests
    without touching production connection-handling code."""
    import time
    import urllib.error
    import urllib.request

    deadline = time.monotonic() + timeout_s
    last_exc = None
    while time.monotonic() < deadline:
        try:
            urllib.request.urlopen(url, timeout=1)
            return
        except urllib.error.HTTPError:
            return  # any HTTP response at all means the socket is accepting
        except (urllib.error.URLError, ConnectionError, TimeoutError) as exc:
            last_exc = exc
            time.sleep(0.05)
    raise RuntimeError(f"{url} never started accepting connections: {last_exc}")


@pytest.fixture
def mock_backend():
    server = ThreadingHTTPServer(("127.0.0.1", 0), MockBackendHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    url = f"http://127.0.0.1:{port}"
    _wait_until_accepting(f"{url}/health")
    yield url
    server.shutdown()
    server.server_close()


@pytest.fixture
def router(mock_backend):
    backends = {"full": mock_backend, "half": mock_backend}
    server = build_server(backends, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    url = f"http://127.0.0.1:{port}"
    _wait_until_accepting(f"{url}/chat")
    yield url
    server.shutdown()
    server.server_close()


def _post_chat(router_url, messages):
    import urllib.request

    body = json.dumps({"messages": messages}).encode("utf-8")
    req = urllib.request.Request(
        f"{router_url}/chat", data=body,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return resp.status, json.loads(resp.read())


def test_router_end_to_end_easy_request(router):
    status, result = _post_chat(router, [u("hello there")])
    assert status == 200
    assert result["tier"] == "half"
    assert result["reason"] == "short-simple"
    assert result["reply"] == "mock reply"
    assert result["tokens"] == 7
    assert result["tokens_per_sec"] >= 0
    assert result["ms"] >= 0


def test_router_end_to_end_hard_request(router):
    status, result = _post_chat(router, [u("please debug this ``` code ``` block")])
    assert status == 200
    assert result["tier"] == "full"
    assert result["reason"] == "code"


def test_router_rejects_missing_messages_field(router):
    import urllib.error
    import urllib.request

    req = urllib.request.Request(
        f"{router}/chat", data=b"{}",
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        urllib.request.urlopen(req, timeout=10)
    assert excinfo.value.code == 400
