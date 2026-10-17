"""Per-request router demo for width-sliced Gemma 3n E4B.

Launches two llama-server backends -- the full checkpoint and a 50%-FFN-sliced
half checkpoint (see setup_models.py) -- and puts a small HTTP router in front
of them. Each incoming chat request is classified as "easy" or "hard" by a
pure rule function and forwarded to the cheaper or the fuller backend
accordingly. This is the phase-4 demo described in
.claude/plans/dynamic-width-gemma4.md: a 1.44x measured decode speedup on the
half model (llama.cpp, bf16, A40) traded against quality on a per-request
basis, using only Google's own MatFormer-validated slice.

Stdlib only (http.server + urllib) -- the router itself needs no installs.
The backends it launches are separate llama-server.exe processes; those do
need setup_models.py to have produced the GGUF files first.

Usage:
    python router_server.py            # launches both backends, then serves
    python router_server.py --dry-run  # skips subprocess launch (see below)

--dry-run does not start any backend and does not by itself make the router
usable -- it exists for tests and for manually driving the router against
backends started some other way. Tests instead call build_server() directly
with a dict of mock backend URLs; that is the "mock backends injected for
tests" path referenced in the project brief.
"""

import argparse
import json
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent

CONFIG = {
    "router_port": 8090,
    "full": {
        "port": 8091,
        "model": ROOT / "models" / "full-Q4_K_M.gguf",
    },
    "half": {
        "port": 8092,
        "model": ROOT / "models" / "half-Q4_K_M.gguf",
    },
    "llama_server_bin": ROOT / "bin" / "llama-server.exe",
    "ngl": 99,        # offload all layers to GPU; set to 0 for CPU-only serving
    "ctx_size": 4096,
    "health_timeout_s": 120,
    "request_timeout_s": 180,
    "log_dir": ROOT / "logs",
}


# ---------------------------------------------------------------------------
# Routing policy
# ---------------------------------------------------------------------------
# route() is a pure function: (messages) -> (tier, reason). No I/O, no global
# state, so it's directly unit-testable and the policy is auditable in one
# place. "Stickiness" (see below) is implemented by re-applying the same
# content rules to earlier messages in the same list, rather than by tracking
# session state server-side -- that keeps the function pure while still
# giving upward-only stickiness across a conversation.

_CODE_TOKENS = re.compile(r"\bdef\b|\bclass\b|[{};]")
_MATH_SYMBOLS = re.compile(r"[=^]")
_MATH_WORDS = re.compile(r"\b(integral|derivative)\b", re.IGNORECASE)
_LONG_DIGIT_RUN = re.compile(r"\d{7,}")
_REASONING_WORDS = re.compile(
    r"\b(prove|derive|debug|analyze|analyse)\b|step[- ]by[- ]step", re.IGNORECASE
)
_LONG_MESSAGE_WORDS = 120


def classify_message(text: str) -> str | None:
    """Return a short reason string if `text` alone warrants the full model,
    else None. Order follows the brief: code, then math, then reasoning
    keywords, then raw length."""
    if "```" in text or _CODE_TOKENS.search(text):
        return "code"
    if _MATH_SYMBOLS.search(text) or _MATH_WORDS.search(text) or _LONG_DIGIT_RUN.search(text):
        return "math"
    if _REASONING_WORDS.search(text):
        return "reasoning-keyword"
    if len(text.split()) > _LONG_MESSAGE_WORDS:
        return "long-message"
    return None


def route(messages: list[dict]) -> tuple[str, str]:
    """Decide which backend tier serves this conversation.

    Rule order: classify the latest user message; if it alone triggers a
    full-model rule, route full with that reason. Otherwise, check whether
    any *earlier* user message in the conversation triggered a full-model
    rule -- if so, stay on full. This is stickiness upward only: once a
    conversation touched something hard, later easy-looking turns keep
    riding the full model. Upgrades reuse KV cache/context well and cost is
    the only downside of over-routing; downgrading mid-conversation would
    only ever save money, and risks a visible quality drop partway through
    a reply the user is already reading, so we never downgrade automatically.
    Everything that never triggered a rule goes to half.
    """
    user_messages = [m for m in messages if m.get("role") == "user"]
    if not user_messages:
        return "half", "short-simple"

    latest = user_messages[-1].get("content", "")
    reason = classify_message(latest)
    if reason:
        return "full", reason

    for earlier in user_messages[:-1]:
        earlier_reason = classify_message(earlier.get("content", ""))
        if earlier_reason:
            return "full", f"stickiness:{earlier_reason}"

    return "half", "short-simple"


# ---------------------------------------------------------------------------
# Backend process management
# ---------------------------------------------------------------------------

def launch_backend(tier: str, tier_cfg: dict) -> subprocess.Popen:
    log_dir = CONFIG["log_dir"]
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"llama-server-{tier}.log"
    cmd = [
        str(CONFIG["llama_server_bin"]),
        "-m", str(tier_cfg["model"]),
        "--port", str(tier_cfg["port"]),
        "-ngl", str(CONFIG["ngl"]),
        "--ctx-size", str(CONFIG["ctx_size"]),
    ]
    print(f"launching {tier} backend on port {tier_cfg['port']} (log: {log_path})")
    log_file = open(log_path, "w", encoding="utf-8")
    return subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)


def wait_for_health(base_url: str, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=5) as resp:
                if resp.status == 200:
                    return
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        time.sleep(1)
    raise RuntimeError(f"backend at {base_url} did not become healthy within {timeout_s}s")


# ---------------------------------------------------------------------------
# HTTP router
# ---------------------------------------------------------------------------

def call_backend(base_url: str, messages: list[dict], timeout_s: float,
                  retries: int = 2) -> dict:
    """POST to a llama-server's OpenAI-compatible /v1/chat/completions.

    Retries a couple of times on a dropped connection. Local HTTP servers
    (both llama-server and the loopback sockets this backs onto during
    testing) occasionally abort a fresh connection under load; one retry
    resolves it without masking a genuinely unreachable backend, which still
    fails after `retries` attempts."""
    body = json.dumps({"model": "router-demo", "messages": messages}).encode("utf-8")
    last_exc: Exception | None = None
    for attempt in range(retries):
        req = urllib.request.Request(
            f"{base_url}/v1/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                return json.loads(resp.read())
        except ConnectionError as exc:
            last_exc = exc
            if attempt + 1 < retries:
                time.sleep(0.1)
    raise last_exc


class RouterHandler(BaseHTTPRequestHandler):
    # Set on the class before the server is constructed. Maps tier name
    # ("full" / "half") to a backend base URL. This indirection is what lets
    # tests inject mock backends without touching subprocess launch code.
    backends: dict[str, str] = {}
    request_timeout_s: float = CONFIG["request_timeout_s"]

    def _send_json(self, code: int, obj: dict) -> None:
        data = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self) -> None:
        if self.path != "/chat":
            self._send_json(404, {"error": "not found"})
            return

        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b""
        try:
            payload = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            self._send_json(400, {"error": "invalid JSON body"})
            return

        messages = payload.get("messages")
        if not isinstance(messages, list):
            self._send_json(400, {"error": "'messages' must be a list"})
            return

        tier, reason = route(messages)
        backend = self.backends.get(tier)
        if backend is None:
            self._send_json(500, {"error": f"no backend configured for tier '{tier}'"})
            return

        start = time.monotonic()
        try:
            result = call_backend(backend, messages, self.request_timeout_s)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            self._send_json(502, {"error": f"backend '{tier}' request failed: {exc}"})
            return
        elapsed_s = time.monotonic() - start

        try:
            reply = result["choices"][0]["message"]["content"]
            completion_tokens = result["usage"]["completion_tokens"]
        except (KeyError, IndexError) as exc:
            self._send_json(502, {"error": f"unexpected backend response shape: {exc}"})
            return

        tokens_per_sec = completion_tokens / elapsed_s if elapsed_s > 0 else 0.0
        self._send_json(200, {
            "reply": reply,
            "tier": tier,
            "reason": reason,
            "tokens": completion_tokens,
            "tokens_per_sec": round(tokens_per_sec, 2),
            "ms": round(elapsed_s * 1000, 1),
        })

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write(f"[router] {self.address_string()} {fmt % args}\n")


def build_server(backends: dict[str, str], host: str = "127.0.0.1",
                  port: int = CONFIG["router_port"]) -> ThreadingHTTPServer:
    handler = RouterHandler
    handler.backends = backends
    return ThreadingHTTPServer((host, port), handler)


# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                         help="skip launching backend subprocesses")
    args = parser.parse_args(argv)

    procs: list[subprocess.Popen] = []
    try:
        if args.dry_run:
            print("dry-run: not launching backends; expecting them (or mocks) to already be up")
            backends = {
                "full": f"http://127.0.0.1:{CONFIG['full']['port']}",
                "half": f"http://127.0.0.1:{CONFIG['half']['port']}",
            }
        else:
            for tier in ("full", "half"):
                procs.append(launch_backend(tier, CONFIG[tier]))
            backends = {}
            for tier in ("full", "half"):
                base_url = f"http://127.0.0.1:{CONFIG[tier]['port']}"
                print(f"waiting for {tier} backend to become healthy ...")
                wait_for_health(base_url, CONFIG["health_timeout_s"])
                backends[tier] = base_url
                print(f"{tier} backend ready at {base_url}")

        server = build_server(backends, port=CONFIG["router_port"])
        print(f"router listening on http://127.0.0.1:{CONFIG['router_port']}/chat")
        server.serve_forever()
    except KeyboardInterrupt:
        print("shutting down")
    finally:
        for p in procs:
            p.terminate()
        for p in procs:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()


if __name__ == "__main__":
    main()
