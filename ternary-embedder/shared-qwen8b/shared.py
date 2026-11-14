"""Shared code for Plan A (qwen-space) and Plan B (ngram-space).

Interface fixed by shared-qwen8b/README.md section 4. Built by Plan A's agent,
2026-11-12. Change only in ways that keep these calls working.

    embed(texts, instruction="", plan="A")  -> [n, 4096] float16, via the shared cache
    spend(plan)                              -> {"calls", "tokens", "dollars"} paid by that plan
    is_validation(text)                      -> bool, the text-defined validation split
    leak_key(text)                           -> normaliser behind the leakage rule
    cut_to_contract(text, content_tokens)    -> the document contract (504 or 126 tokens)
    dev_task(name)                           -> BEIR dev task (wraps comparisons/qwen06b/compare.py::data)

The API key is read from DEEPINFRA_API_KEY and is never written anywhere.
Query protocol (endpoint check 2026-11-12, confirmed in Stage 1 of Plan A):
write the instruction into the input string as
``Instruct: {instruction}\\nQuery:{query}`` (see ``query_input``) and leave the
``instruction`` argument empty. ``instruction`` maps to DeepInfra's
``custom_instruction`` field and is part of the cache key.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
SHARED = PROJECT / "training-v2/work/shared-qwen8b"
CACHE_DB = SHARED / "api-cache/cache.sqlite"
TOKENIZER = PROJECT / "tokenizer/tokenizer.json"

PROVIDER = "deepinfra"
MODEL = "Qwen/Qwen3-Embedding-8B"
URL = "https://api.deepinfra.com/v1/inference/" + MODEL
DIM = 4096
CAP_DOLLARS = {"A": 5.0, "B": 5.0}
PRICE_PER_TOKEN = 0.01 / 1e6
BATCH = 32
WORKERS = 4

DEV_PROMPTS = {
    "SciFact": "Given a scientific claim, retrieve documents that support or refute the claim",
    "NFCorpus": "Given a question, retrieve relevant documents that answer the question",
    "ArguAna": "Given a claim, find documents that refute the claim",
}


class KeyRevoked(RuntimeError):
    """HTTP 401/403: the owner has deleted the key. Stop all hosted calls."""


class CapReached(RuntimeError):
    pass


def query_input(query: str, instruction: str) -> str:
    """The query protocol: instruction written into the input string."""
    return f"Instruct: {instruction}\nQuery:{query}"


# ---------------------------------------------------------------- cache

_local = threading.local()


def _db() -> sqlite3.Connection:
    con = getattr(_local, "con", None)
    if con is None:
        CACHE_DB.parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(CACHE_DB, timeout=120, isolation_level=None)
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA busy_timeout=120000")
        con.execute("CREATE TABLE IF NOT EXISTS vec (key BLOB PRIMARY KEY, provider TEXT, model TEXT,"
                    " instruction TEXT, vec BLOB, plan TEXT, utc TEXT)")
        con.execute("CREATE TABLE IF NOT EXISTS spend (plan TEXT PRIMARY KEY, calls INTEGER, tokens INTEGER,"
                    " dollars REAL)")
        con.execute("CREATE TABLE IF NOT EXISTS calls (utc TEXT, plan TEXT, n INTEGER, tokens INTEGER,"
                    " dollars REAL, request_id TEXT)")
        _local.con = con
    return con


def cache_key(text: str, instruction: str = "", provider: str = PROVIDER, model: str = MODEL) -> bytes:
    payload = json.dumps([provider, model, instruction, text], ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).digest()


def spend(plan: str) -> dict:
    row = _db().execute("SELECT calls, tokens, dollars FROM spend WHERE plan=?", (plan,)).fetchone()
    return {"calls": 0, "tokens": 0, "dollars": 0.0} if row is None else \
        {"calls": row[0], "tokens": row[1], "dollars": row[2]}


def _lookup(keys: list[bytes]) -> dict[bytes, np.ndarray]:
    out = {}
    con = _db()
    for i in range(0, len(keys), 500):
        part = keys[i:i + 500]
        q = "SELECT key, vec FROM vec WHERE key IN (%s)" % ",".join("?" * len(part))
        for k, v in con.execute(q, part):
            out[bytes(k)] = np.frombuffer(v, dtype=np.float16)
    return out


def _store(rows, plan, tokens, dollars, request_id, instruction):
    con = _db()
    utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    for attempt in range(20):
        try:
            con.execute("BEGIN IMMEDIATE")
            con.executemany("INSERT OR IGNORE INTO vec VALUES (?,?,?,?,?,?,?)",
                            [(k, PROVIDER, MODEL, instruction, v.astype(np.float16).tobytes(), plan, utc)
                             for k, v in rows])
            con.execute("INSERT INTO spend VALUES (?,1,?,?) ON CONFLICT(plan) DO UPDATE SET"
                        " calls=calls+1, tokens=tokens+excluded.tokens, dollars=dollars+excluded.dollars",
                        (plan, tokens, dollars))
            con.execute("INSERT INTO calls VALUES (?,?,?,?,?,?)", (utc, plan, len(rows), tokens, dollars,
                                                                  request_id))
            con.execute("COMMIT")
            return
        except sqlite3.OperationalError:
            try:
                con.execute("ROLLBACK")
            except sqlite3.OperationalError:
                pass
            time.sleep(0.5 * (attempt + 1))
    raise RuntimeError("cache write failed")


# ---------------------------------------------------------------- client

_lock = threading.Lock()
_revoked = False


def _call(inputs: list[str], instruction: str) -> tuple[np.ndarray, int, float, str]:
    global _revoked
    key = os.environ.get("DEEPINFRA_API_KEY", "").strip()
    if not key:
        raise RuntimeError("DEEPINFRA_API_KEY is not set")
    payload = {"inputs": inputs}
    if instruction:
        payload["custom_instruction"] = instruction
    data = json.dumps(payload).encode()
    delay = 2.0
    for attempt in range(10):
        if _revoked:
            raise KeyRevoked("key rejected earlier in this process")
        req = urllib.request.Request(URL, data=data, headers={"Authorization": "Bearer " + key,
                                                              "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                body = json.loads(r.read())
            break
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                _revoked = True
                raise KeyRevoked(f"HTTP {e.code}: the DeepInfra key was rejected (deleted by the owner?)")
            if e.code == 429 or e.code >= 500:
                time.sleep(delay); delay = min(delay * 2, 60); continue
            msg = e.read().decode("utf-8", "replace")[:300].replace(key, "<key>")
            raise RuntimeError(f"HTTP {e.code}: {msg}")
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            time.sleep(delay); delay = min(delay * 2, 60)
    else:
        raise RuntimeError("hosted call failed after retries")
    emb = np.asarray(body["embeddings"], dtype=np.float32)
    if emb.shape != (len(inputs), DIM):
        raise RuntimeError(f"unexpected shape {emb.shape}")
    status = body.get("inference_status") or {}
    tokens = int(body.get("input_tokens") or status.get("tokens_input") or 0)
    dollars = float(status.get("cost") if status.get("cost") is not None else tokens * PRICE_PER_TOKEN)
    return emb, tokens, dollars, str(body.get("request_id", ""))


def embed(texts, instruction: str = "", plan: str = "A", progress: bool = False) -> np.ndarray:
    """Hosted vectors [n, 4096] float16 exactly as returned, through the shared cache."""
    if plan not in CAP_DOLLARS:
        raise ValueError("plan must be 'A' or 'B'")
    texts = list(texts)
    keys = [cache_key(t, instruction) for t in texts]
    have = _lookup(list(dict.fromkeys(keys)))
    missing = list(dict.fromkeys(i for i, k in enumerate(keys) if k not in have))
    # one representative index per missing key
    seen, todo = set(), []
    for i in missing:
        if keys[i] not in seen:
            seen.add(keys[i]); todo.append(i)
    batches = [todo[i:i + BATCH] for i in range(0, len(todo), BATCH)]
    done = [0]

    def run(batch):
        with _lock:
            s = spend(plan)
            if s["dollars"] >= CAP_DOLLARS[plan]:
                raise CapReached(f"plan {plan} reached its ${CAP_DOLLARS[plan]} cap (${s['dollars']:.4f})")
        emb, tokens, dollars, rid = _call([texts[i] for i in batch], instruction)
        _store([(keys[i], v) for i, v in zip(batch, emb)], plan, tokens, dollars, rid, instruction)
        for i, v in zip(batch, emb):
            have[keys[i]] = v.astype(np.float16)
        done[0] += 1
        if progress and done[0] % 50 == 0:
            print(f"embed: {done[0]}/{len(batches)} batches, plan {plan} spend {spend(plan)}",
                  file=sys.stderr, flush=True)

    if batches:
        with ThreadPoolExecutor(WORKERS) as ex:
            for f in [ex.submit(run, b) for b in batches]:
                f.result()
    out = np.empty((len(texts), DIM), dtype=np.float16)
    for i, k in enumerate(keys):
        out[i] = have[k]
    return out


# ---------------------------------------------------------------- data rules

def is_validation(text: str) -> bool:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16) % 128 == 0


_ws = re.compile(r"\s+")


def leak_key(text: str) -> str:
    return _ws.sub(" ", text.lower()).strip()[:200]


_tok = None


def _tokenizer():
    global _tok
    if _tok is None:
        from tokenizers import Tokenizer
        _tok = Tokenizer.from_file(str(TOKENIZER))
        _tok.no_truncation()
        _tok.no_padding()
    return _tok


def cut_to_contract(text: str, content_tokens: int = 504) -> str:
    """The document contract (qwen-space/PLAN.md section 4).

    Keep the text covered by the first `content_tokens` WordPiece content tokens
    of the student tokenizer: cut the original string at the character offset of
    token `content_tokens`. If that token continues a word (a `##` piece), the
    cut moves back to the start of that word, so the kept string re-tokenizes to
    at most `content_tokens` tokens and the student and teacher see exactly the
    same text. Trailing whitespace is stripped. Texts within the limit are
    returned unchanged.
    """
    enc = _tokenizer().encode(text, add_special_tokens=False)
    if len(enc.ids) <= content_tokens:
        return text
    words = enc.word_ids
    n = content_tokens
    while n > 0 and words[n] is not None and words[n] == words[n - 1]:
        n -= 1
    return text[:enc.offsets[n][0]].rstrip()


def n_content_tokens(text: str) -> int:
    return len(_tokenizer().encode(text, add_special_tokens=False).ids)


def dev_task(name: str) -> dict:
    """SciFact, NFCorpus or ArguAna as comparisons/qwen06b/compare.py::data returns it, plus `instruction`."""
    sys.path.insert(0, str(PROJECT / "comparisons/qwen06b"))
    from compare import data
    d = data(name)
    d["instruction"] = DEV_PROMPTS.get(name)
    return d
