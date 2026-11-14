"""Checks ngram_study's fast paths against slow, obvious versions on small made-up data.

    python -m pytest memdraft/test_ngram_study.py
"""
from collections import Counter, defaultdict

import numpy as np
import pytest

import ngram_study as ns

K, N_MAX = 4, 5


def make_rows(seed, n_rows=40, vocab=30):
    """Requests whose tokens follow a few fixed habits, so short contexts predict the next token.
    The stored top choice equals the written token about 85% of the time."""
    rng = np.random.default_rng(seed)
    habit = np.random.default_rng(0).integers(0, vocab, (vocab, vocab))    # the same habits for every seed
    rows = []
    for i in range(n_rows):
        length = int(rng.integers(30, 120))
        seq = list(rng.integers(0, vocab, 2))
        while len(seq) < length:
            follow = habit[seq[-2], seq[-1]]
            seq.append(int(follow if rng.random() < 0.7 else rng.integers(0, vocab)))
        n_prompt = int(rng.integers(3, 12))
        answer = seq[n_prompt:]
        top = np.array([t if rng.random() < 0.85 else int(rng.integers(0, vocab)) for t in answer], np.int32)
        if i % 5 == 0:
            top[-1] = -1                    # a stop token added without a teacher row
        rows.append({'id': f'p{i // 2}', 'source': 'toy', 'thinking': bool(i % 2), 'finished': True,
                     'prompt_ids': seq[:n_prompt], 'answer_ids': answer, 'amax': top})
    return rows


@pytest.fixture(scope='module')
def world():
    train, held = ns.Data(make_rows(1, 120)), ns.Data(make_rows(2, 40))
    idx = ns.labelled(train)
    static = ns.Static.build(train, idx, train.amax[idx], K)
    return train, held, static, ns.InRequest(held, K)


def context_of(data, j, k):
    return tuple(int(t) for t in data.tok[j - k:j][::-1])


def test_static_counts_match_a_plain_counter(world):
    train, _, static, _ = world
    idx = ns.labelled(train)
    for k in range(1, K + 1):
        counts = defaultdict(Counter)
        for j in idx:
            if train.pos[j] >= k:
                counts[context_of(train, j, k)][int(train.amax[j])] += 1
        assert len(static.keys[k - 1]) == len(counts)
        C, _ = train.context(idx, K)
        h = np.zeros(len(idx), np.uint64)
        for i in range(k):
            h = ns.mix(h, C[:, i])
        hit, token, cbest, ctot = static.lookup(k, h)
        for n, j in enumerate(idx):
            if train.pos[j] < k:
                continue
            seen = counts[context_of(train, j, k)]
            assert hit[n]
            assert ctot[n] == sum(seen.values())
            assert cbest[n] == max(seen.values()) == seen[int(token[n])]


def test_in_request_lookup_matches_a_backward_scan(world):
    _, held, _, inreq = world
    idx = ns.labelled(held)
    C, avail = held.context(idx, K)
    h = np.zeros(len(idx), np.uint64)
    for k in range(1, K + 1):
        h = ns.mix(h, C[:, k - 1])
        hit, token, seen = inreq.lookup(k, h, held.seq[idx], held.pos[idx])
        for n, j in enumerate(idx):
            if held.pos[j] < k:
                continue
            start = held.start[held.seq[j]]
            earlier = [i for i in range(start + k, j) if context_of(held, i, k) == context_of(held, j, k)]
            assert bool(hit[n]) == bool(earlier)
            if earlier:
                assert token[n] == held.tok[earlier[-1]]
                assert seen[n] == min(len(earlier), ns.CC_MAX)


def slow_decode(data, static, inreq, calib, mode, rule, value):
    """One request at a time, one step at a time: the counts of ns.walk per request.
    rule 'tau' drafts while the chain's chance is at least value; 'pad' always drafts value tokens."""
    out = {}
    truth = data.amax if mode == 'greedy' else data.tok
    for s in range(len(data.start)):
        positions = [j for j in range(data.ans[s], data.end[s]) if data.amax[j] >= 0]
        j, last = positions[0], positions[-1]
        steps = drafted = accepted = rejected = reshaped = 0
        shape = -1
        while j <= last:
            context = [int(t) for t in data.tok[data.start[s]:j]]
            chain, running = [], np.float32(1.0)
            for _ in range(value if rule == 'pad' else N_MAX):
                recent = context[::-1][:K]
                C = np.array([recent + [0] * (K - len(recent))], np.int32)
                cands = ns.candidates(C, np.array([len(recent)], np.int32), np.array([s], np.int32),
                                      np.array([data.pos[j]], np.int32), static, inreq, K)
                token, conf = ns.choose(cands, calib)
                running = running * conf[0]
                guess = int(token[0]) if conf[0] > 0 else context[-1]
                if rule == 'tau' and (running < ns.TAUS[0] or running < value):
                    break
                chain.append(guess)
                context.append(guess)
            kept = right = 0
            still_right = True
            for d, token in enumerate(chain):
                at = j + d
                if at >= data.end[s] or truth[at] < 0:
                    break
                kept += 1
                if still_right and token == truth[at]:
                    right += 1
                else:
                    still_right = False
                if mode == 'greedy' and data.tok[at] != data.amax[at]:
                    break
            steps, drafted, accepted = steps + 1, drafted + kept, accepted + right
            rejected, reshaped, shape = rejected + (right < kept), reshaped + (kept != shape), kept
            j += right + 1
        out[s] = (last - positions[0] + 1, steps, drafted, accepted, rejected, reshaped)
    return out


@pytest.mark.parametrize('mode', ['greedy', 'sampled'])
@pytest.mark.parametrize('policy', ['static', 'context', 'both'])
def test_fast_decode_matches_slow_decode(world, mode, policy):
    train, held, static, inreq = world
    use_static = static if policy != 'context' else None
    use_inreq = inreq if policy != 'static' else None
    idx = ns.labelled(held)
    calib = ns.Calibration(K)
    C, avail = held.context(idx, K)
    calib.add(ns.candidates(C, avail, held.seq[idx], held.pos[idx], use_static, use_inreq, K),
              (held.amax if mode == 'greedy' else held.tok)[idx])
    tokens, chance = ns.draft(held, idx, use_static, use_inreq, calib, K, N_MAX, ns.TAUS[0], pad_to=3)
    runs = ns.sweep(held, idx, tokens, chance, mode, pad_to=3)
    for rule in (('tau', 0.05), ('tau', 0.3), ('tau', 0.6), ('pad', 1), ('pad', 3)):
        slow = slow_decode(held, use_static, use_inreq, calib, mode, *rule)
        run = runs[rule]
        fast = {int(s): tuple(int(run[k][n]) for k in ns.COUNTS) for n, s in enumerate(run['seqs'])}
        assert fast == slow, rule
        if rule == ('tau', 0.05):           # the comparison must not pass on empty drafts
            assert run['accepted'].sum() > 50 and run['drafted'].sum() > run['accepted'].sum()
            assert (tokens[:, 1] >= 0).sum() > 50
        if rule == ('pad', 3):
            assert run['drafted'].sum() > 2 * run['steps'].sum()


def test_cpp_drafter_matches_numpy_on_a_shipped_table(world, tmp_path):
    """memdraft.h (through replay.cpp) and ngram_study.Exported must count the same steps."""
    import json
    import shutil
    import subprocess

    import build_table

    if not shutil.which('g++'):
        pytest.skip('no g++')
    train, held, static, inreq = world
    idx = ns.labelled(held)
    calib = ns.Calibration(K)
    C, avail = held.context(idx, K)
    calib.add(ns.candidates(C, avail, held.seq[idx], held.pos[idx], static, inreq, K), held.amax[idx])
    kept, _ = ns.export_table(static, calib, tmp_path / 't.mdt', floor=0.05)
    assert kept > 100
    ns.write_stream(held, tmp_path / 't.stream')
    binary = tmp_path / 'replay'
    subprocess.run(['g++', '-O2', '-std=c++17', '-o', str(binary), str(build_table.HERE / 'replay.cpp')], check=True)
    table = ns.Exported(tmp_path / 't.mdt')
    want = build_table.reference(table, held, ['greedy', 'sampled'], ['static', 'context', 'both'], [0.05, 0.3, 0.6], N_MAX)
    for (mode, policy, rule, value), expect in want.items():
        got = build_table.replay(binary, tmp_path / 't.mdt', tmp_path / 't.stream', mode, policy, rule, value, N_MAX)
        assert {k: got[k] for k in expect} == expect, (mode, policy, rule, value)
    assert want['greedy', 'both', 'tau', 0.05]['accepted'] > 50
    assert want['greedy', 'both', 'pad', 3]['drafted'] > 2 * want['greedy', 'both', 'pad', 3]['steps']
