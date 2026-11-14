"""Shared paths, prompts and scoring for the cross-attention memory experiment."""
from __future__ import annotations

import collections
import hashlib
import json
import re
import string
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RULES_PATH = HERE / "rules.json"
MEMORY_MARK = "[supplied as memory]"

QA_INSTRUCTION = (
    "Answer the question using the supplied notes. Give only the short answer, "
    "with no explanation. Preserve exact numbers, units, identifiers, and codes. "
    "If the requested fact is absent from the notes, answer UNKNOWN.\n\nNotes:\n"
)
RECON_INSTRUCTION = "Repeat the supplied notes exactly.\n\nNotes:\n"
CONTINUE_INSTRUCTION = "Continue the text of the supplied notes.\n\nNotes:\n"


def rules():
    return json.loads(RULES_PATH.read_text())


def repo_path(p: str) -> Path:
    return REPO / p


def work_dir() -> Path:
    return repo_path(rules()["data"]["work"])


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def user_prompt(kind: str, question: str | None = None, notes: str | None = None) -> str:
    """notes=None means the notes arrive through cross-attention memory."""
    body = MEMORY_MARK if notes is None else notes
    if kind in ("qa", "qa_removed", "xdna"):
        return QA_INSTRUCTION + body + f"\n\nQuestion: {question}\n"
    if kind == "recon":
        return RECON_INSTRUCTION + body
    if kind == "continue":
        return CONTINUE_INSTRUCTION + body
    raise ValueError(kind)


def chat_ids(tokenizer, prompt: str) -> list[int]:
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    return tokenizer.encode(text, add_special_tokens=False)


def target_ids(tokenizer, target: str) -> list[int]:
    return tokenizer.encode(target + "<|im_end|>", add_special_tokens=False)


# SQuAD v2 official normalization.
def normalize(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def f1(pred: str, gold: str) -> float:
    p, g = normalize(pred).split(), normalize(gold).split()
    common = collections.Counter(p) & collections.Counter(g)
    same = sum(common.values())
    if not p or not g:
        return float(p == g)
    if same == 0:
        return 0.0
    precision, recall = same / len(p), same / len(g)
    return 2 * precision * recall / (precision + recall)


def is_unknown(pred: str) -> bool:
    return normalize(pred) == "unknown"


def score_qa(pred: str, answers: list[str]) -> dict:
    """answers == [] means the correct response is UNKNOWN."""
    pred = pred.strip().split("\n")[0].strip()
    if not answers:
        ok = float(is_unknown(pred))
        return {"em": ok, "f1": ok, "abstained": is_unknown(pred)}
    if is_unknown(pred):
        return {"em": 0.0, "f1": 0.0, "abstained": True}
    return {"em": float(max(normalize(pred) == normalize(a) for a in answers)),
            "f1": max(f1(pred, a) for a in answers), "abstained": False}
