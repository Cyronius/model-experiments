"""Synthetic task generators: task family (arithmetic | brackets | recall)
x difficulty.

Format is `<task><problem>=<answer><eos>`; loss covers the answer tokens and the
eos only. Every example carries {cell, task, difficulty} metadata that never
reaches the model -- it exists so eval can correlate router choices with
difficulty (figure 3) and with task type (figure 4).

Addition answers are emitted least-significant-digit first. Carries propagate
right-to-left, so LSB-first ordering lets a small causal model compute each
output digit from the digits it has already seen; left-to-right forces it to
look ahead and multi-digit addition stops being learnable at this scale.

Operands are also zero-padded to a fixed width (DataConfig.arith_pad_digits).
Reversal alone is not enough: it fixes the offset of operand a's digits but not
operand b's, which start at len(a)+2. Reaching the i-th digit of a variable-length
span then requires counting positions, and learned absolute embeddings cannot
express that. Measured cost of leaving it out: 5-8 digit addition sits at 0%
exact-match at every width (16-512), depth (4-8), step budget (12k-24k) and
d_model (128-256) tried, with middle digits at chance. Padding alone takes the
same model to 100%.

Bracket examples give a prefix with unclosed brackets and ask for the closing
sequence. Answer length equals the open-stack depth, so the difficulty axis and
the amount of signal in the loss scale together.
"""

from dataclasses import dataclass
from typing import List, Tuple
import math
import random
import sys

import numpy as np

from .config import (DataConfig, PAD, EOS, STOI, ITOS,
                     RECALL_CHARS, TABLE_IDS, TABLE_M)

PAIRS = {"(": ")", "[": "]", "{": "}", "<": ">"}
OPENERS = list(PAIRS)

CELLS = ("arith_easy", "arith_hard", "dyck_easy", "dyck_hard",
         "recall_easy", "recall_hard")
EVAL_SEED = 1234          # fixed: every run in the grid evaluates on identical data
MAX_ANSWER_LEN = 10       # 9 digits, or 8 closers; +1 decode step for eos

_TABLES = {}


def recall_table(m: int, seed: int):
    """The lookup table is part of the task definition, not the run: it is built
    from a fixed seed so every condition and every run seed memorizes the same
    associations."""
    key = (m, seed)
    if key not in _TABLES:
        rng = random.Random(seed * 1000003 + m)
        _TABLES[key] = [[(rng.randrange(m), rng.randrange(m)) for _ in range(m)]
                        for _ in range(m)]
    return _TABLES[key]


# ---------------------------------------------------------------- generators

def _int_with_digits(rng: random.Random, d: int) -> int:
    return rng.randint(0, 9) if d == 1 else rng.randint(10 ** (d - 1), 10 ** d - 1)


def gen_arith(rng: random.Random, lo: int, hi: int,
              reverse_operands: bool = True, pad_digits: int = 0) -> Tuple[str, str, int]:
    da, db = rng.randint(lo, hi), rng.randint(lo, hi)
    a, b = _int_with_digits(rng, da), _int_with_digits(rng, db)
    sa, sb = str(a), str(b)
    if pad_digits:
        # Diagnostic: leading zeros make both operands a fixed width, so after
        # reversal the i-th least-significant digit of each is at a constant
        # absolute index. Difficulty still reports the true digit count.
        sa, sb = sa.zfill(pad_digits), sb.zfill(pad_digits)
    if reverse_operands:
        # Reverse operands as well as the answer. With operands left-to-right and
        # only the answer reversed, producing answer digit i means locating digit
        # i-from-the-end of each operand -- index arithmetic over variable-length
        # spans, which learned absolute positions handle badly. Reversing all three
        # fields puts the i-th least-significant digit of every field at offset i
        # from that field's start, so alignment is a fixed offset.
        sa, sb = sa[::-1], sb[::-1]
    prompt = f"A{sa}+{sb}="
    answer = str(a + b)[::-1]          # least-significant digit first
    return prompt, answer, max(da, db)


def gen_dyck(rng: random.Random, lo: int, hi: int, max_extra: int) -> Tuple[str, str, int]:
    depth = rng.randint(lo, hi)
    extra = rng.randint(0, max_extra)
    n_open, n_close = depth + extra, extra
    seq: List[str] = []
    stack: List[str] = []
    while n_open or n_close:
        can_open = n_open > 0
        can_close = n_close > 0 and stack
        # closing is only legal with a non-empty stack, so the final stack depth
        # is always exactly n_open - n_close = depth
        open_now = can_open and (not can_close or rng.random() < 0.5)
        if open_now:
            ch = rng.choice(OPENERS)
            seq.append(ch)
            stack.append(ch)
            n_open -= 1
        else:
            seq.append(PAIRS[stack.pop()])
            n_close -= 1
    prompt = "B" + "".join(seq) + "="
    answer = "".join(PAIRS[c] for c in reversed(stack))
    return prompt, answer, depth


def gen_recall(rng: random.Random, sizes, cfg: DataConfig) -> Tuple[str, str, int]:
    """Memorize T[a][b] -> (c, d). The table id is required: the same (a, b) key
    means different things in different tables, so without it the task is
    ambiguous. Difficulty is log2(table size) = how many associations must be
    stored, which is what MLP width actually buys."""
    m = sizes[rng.randrange(len(sizes))]
    tid = TABLE_IDS[TABLE_M.index(m)]
    tab = recall_table(m, cfg.recall_table_seed)
    i, j = rng.randrange(m), rng.randrange(m)
    c, d = tab[i][j]
    prompt = "R" + tid + RECALL_CHARS[i] + RECALL_CHARS[j] + "="
    answer = RECALL_CHARS[c] + RECALL_CHARS[d]
    return prompt, answer, int(math.log2(m))


def gen_cell(rng: random.Random, cell: str, cfg: DataConfig) -> Tuple[str, str, int]:
    if cell == "arith_easy":
        return gen_arith(rng, *cfg.easy_digits, pad_digits=cfg.arith_pad_digits)
    if cell == "arith_hard":
        return gen_arith(rng, *cfg.hard_digits, pad_digits=cfg.arith_pad_digits)
    if cell == "dyck_easy":
        return gen_dyck(rng, *cfg.easy_depth, cfg.max_extra_pairs)
    if cell == "dyck_hard":
        return gen_dyck(rng, *cfg.hard_depth, cfg.max_extra_pairs)
    if cell == "recall_easy":
        return gen_recall(rng, cfg.recall_easy_m, cfg)
    if cell == "recall_hard":
        return gen_recall(rng, cfg.recall_hard_m, cfg)
    raise ValueError(f"unknown cell {cell}")


# ---------------------------------------------------------------- encoding

def encode(prompt: str, answer: str, block_size: int):
    """-> (x, y, loss_mask, prompt_len). Mask covers answer tokens and the eos."""
    full = [STOI[c] for c in prompt] + [STOI[c] for c in answer] + [EOS]
    if len(full) - 1 > block_size:
        raise ValueError(f"example of length {len(full)} exceeds block_size {block_size}")
    x = np.full(block_size, PAD, dtype=np.int64)
    y = np.full(block_size, PAD, dtype=np.int64)
    m = np.zeros(block_size, dtype=np.float32)
    n = len(full) - 1
    x[:n] = full[:-1]
    y[:n] = full[1:]
    m[len(prompt) - 1: n] = 1.0        # y[i] == full[i+1] lands in the answer region
    return x, y, m, len(prompt)


def decode_tokens(toks) -> str:
    out = []
    for t in toks:
        t = int(t)
        if t in (PAD, EOS):
            break
        out.append(ITOS[t])
    return "".join(out)


# ---------------------------------------------------------------- batches

@dataclass
class Batch:
    x: np.ndarray
    y: np.ndarray
    mask: np.ndarray
    cell: np.ndarray          # index into CELLS
    difficulty: np.ndarray
    prompt_len: np.ndarray
    answers: List[str]

    def __len__(self) -> int:
        return len(self.x)


def _build(examples, cells, cfg: DataConfig) -> Batch:
    xs, ys, ms, pls, ans, diffs = [], [], [], [], [], []
    for (p, a, d) in examples:
        x, y, m, pl = encode(p, a, cfg.block_size)
        xs.append(x); ys.append(y); ms.append(m); pls.append(pl); ans.append(a); diffs.append(d)
    return Batch(
        x=np.stack(xs), y=np.stack(ys), mask=np.stack(ms),
        cell=np.array(cells, dtype=np.int64), difficulty=np.array(diffs, dtype=np.int64),
        prompt_len=np.array(pls, dtype=np.int64), answers=ans,
    )


def sample_batch(rng: random.Random, n: int, cfg: DataConfig) -> Batch:
    """Uniform over the four cells -- the mixed training distribution."""
    cells = [rng.randrange(len(CELLS)) for _ in range(n)]
    ex = [gen_cell(rng, CELLS[c], cfg) for c in cells]
    return _build(ex, cells, cfg)


def make_eval_sets(cfg: DataConfig) -> dict:
    """Per-cell held-out sets, identical across every run in the grid."""
    rng = random.Random(EVAL_SEED)
    out = {}
    for ci, cell in enumerate(CELLS):
        ex = [gen_cell(rng, cell, cfg) for _ in range(cfg.n_eval_per_cell)]
        out[cell] = _build(ex, [ci] * len(ex), cfg)
    return out


# ---------------------------------------------------------------- checking

def exact_match(pred: str, target: str) -> bool:
    return pred == target


def _show(s: str) -> str:
    """Recall symbols are private code points; render them readably."""
    out = []
    for c in s:
        if c in RECALL_CHARS:
            out.append(f"s{RECALL_CHARS.index(c)}")
        elif c in TABLE_IDS:
            out.append(f"T{TABLE_IDS.index(c)}|")
        else:
            out.append(c)
    return "".join(out)


def pretty(prompt: str, answer: str) -> str:
    """Answers are stored LSB-first for addition; un-reverse them for display."""
    shown = answer[::-1] if prompt.startswith("A") else answer
    if prompt.startswith("R"):
        return f"{_show(prompt)}{_show(answer)}"
    return f"{prompt}{answer}   (reads as {prompt}{shown})"


# ---------------------------------------------------------------- sanity check

def _sanity() -> None:
    try:                       # recall symbols are non-ASCII; cp1252 consoles choke
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    cfg = DataConfig()
    rng = random.Random(0)
    print("vocab:", ITOS, f"({len(ITOS)} tokens)\n")

    for cell in CELLS:
        print(f"--- {cell} ---")
        for _ in range(4):
            p, a, d = gen_cell(rng, cell, cfg)
            print(f"  difficulty={d:<2} {pretty(p, a)}")
        print()

    # correctness of the generators themselves
    for _ in range(20000):
        p, a, d = gen_arith(rng, 1, 8)
        lhs, rhs = p[1:-1].split("+")
        # operands are stored least-significant-digit first, like the answer
        assert int(a[::-1]) == int(lhs[::-1]) + int(rhs[::-1]), (p, a)
        assert d == max(len(lhs), len(rhs))
    print("arithmetic: 20000 sums verified")

    for _ in range(5000):                  # diagnostic zero-padding path
        p, a, d = gen_arith(rng, 1, 8, pad_digits=8)
        lhs, rhs = p[1:-1].split("+")
        assert len(lhs) == len(rhs) == 8, p
        assert int(a[::-1]) == int(lhs[::-1]) + int(rhs[::-1]), (p, a)
        assert 1 <= d <= 8
        assert len(p) + len(a) + 1 - 1 <= cfg.block_size, (p, a)
    print("arithmetic (pad_digits=8): 5000 sums verified, operands fixed-width")

    for _ in range(20000):
        p, a, d = gen_dyck(rng, 1, 8, cfg.max_extra_pairs)
        stack = []
        for ch in p[1:-1]:
            if ch in PAIRS:
                stack.append(ch)
            else:
                assert stack and PAIRS[stack[-1]] == ch, p
                stack.pop()
        assert "".join(PAIRS[c] for c in reversed(stack)) == a, (p, a)
        assert len(a) == d
    print("brackets: 20000 prefixes verified (prompt valid, closers correct, len == depth)")

    # recall: the table must be deterministic, self-consistent, and unambiguous
    for m in TABLE_M:
        assert recall_table(m, cfg.recall_table_seed) == recall_table(m, cfg.recall_table_seed)
    seen = {}
    for _ in range(20000):
        p, a, d = gen_recall(rng, TABLE_M, cfg)
        assert 4 <= d <= 6 and len(a) == 2, (p, a, d)
        if p in seen:
            assert seen[p] == a, f"same key mapped to two answers: {_show(p)}"
        seen[p] = a
    print(f"recall: 20000 lookups verified over {len(seen)} distinct keys, "
          f"deterministic and single-valued")

    # exact-match checker on known cases
    assert exact_match("55", "55") and not exact_match("55", "56")
    assert exact_match("])", "])") and not exact_match("])", ")]")
    assert not exact_match("", "5") and not exact_match("55", "5")
    print("exact_match: known cases pass")

    # encoding round-trip and mask placement
    p, a, _ = gen_arith(rng, 2, 2)
    x, y, m, pl = encode(p, a, cfg.block_size)
    assert decode_tokens(x).startswith(p)
    assert decode_tokens(y[pl - 1:]) == a
    assert int(m.sum()) == len(a) + 1, (m.sum(), len(a))
    assert m[:pl - 1].sum() == 0
    print(f"encoding: mask covers exactly the {len(a)} answer tokens + eos")

    # longest example must fit block_size
    worst = 0
    for _ in range(50000):
        cell = CELLS[rng.randrange(4)]
        p, a, _ = gen_cell(rng, cell, cfg)
        worst = max(worst, len(p) + len(a) + 1 - 1)
    print(f"longest x/y seen over 50000 samples: {worst} (block_size={cfg.block_size})")
    assert worst <= cfg.block_size

    ev = make_eval_sets(cfg)
    ev2 = make_eval_sets(cfg)
    assert all(np.array_equal(ev[c].x, ev2[c].x) for c in CELLS)
    print(f"eval sets: {cfg.n_eval_per_cell}/cell, reproducible across calls")

    b = sample_batch(random.Random(7), 4096, cfg)
    counts = np.bincount(b.cell, minlength=4)
    print("mixed batch cell counts (4096 draws):", dict(zip(CELLS, counts.tolist())))
    print("\nall data sanity checks passed")


if __name__ == "__main__":
    _sanity()
