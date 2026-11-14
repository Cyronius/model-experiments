import numpy as np
import torch

import slim_bench as sbench
from test_slim_batch import ids
from test_slim_opd import trained_student


def kept_rows(model, step=2):
    kept = torch.arange(0, model.config.vocab_size, step)
    index = torch.full((model.config.vocab_size,), -1)
    index[kept] = torch.arange(len(kept))
    return model.lm_head.weight[kept].float(), index


def row(seed, length, choices):
    """A row in the kept (even) ids; choices are their lengths."""
    even = lambda s, n: [2 * t for t in ids(s, length=n, vocab=48)]
    return {'chars': [3 * n for n in choices],
            '64k': {'context': even(seed, length), 'choices': [even(seed + 10 + j, n) for j, n in enumerate(choices)]}}


def direct(model, head, index, r):
    """Each choice's summed log-probability, one unpadded pass per choice."""
    out = []
    for c in r['64k']['choices']:
        x = torch.tensor([r['64k']['context'] + c])
        h = model.model(input_ids=x, use_cache=False).last_hidden_state[0].float()
        lq = (h @ head.T).log_softmax(-1)
        start = len(r['64k']['context']) - 1
        out.append(sum(float(lq[start + t, index[c[t]]]) for t in range(len(c))))
    return out


def test_batched_right_padded_scores_match_one_pass_per_choice_for_student_and_dense():
    model, slim = trained_student()
    head, index = kept_rows(model)
    rows = [row(1, 9, [3, 1, 5]), row(2, 14, [1, 1, 1, 1]), row(3, 6, [2, 4])]
    for on in (True, False):
        slim.on = on
        with torch.no_grad():
            expected = [direct(model, head, index, r) for r in rows]
            batched = sbench.choice_logprobs(model, rows, '64k', head, index, 0, budget=10 ** 6)
            single = sbench.choice_logprobs(model, rows, '64k', head, index, 0, budget=1)
        flat = lambda v: [x for r in v for x in r]
        np.testing.assert_allclose(flat(batched), flat(expected), atol=1e-4)
        np.testing.assert_allclose(flat(single), flat(expected), atol=1e-4)
    slim.on = False
    # The single-token row ran as one pass over its context.
    assert [s[1] for s in sbench.sequences(rows, '64k')] == [0, 1, 2, None, 0, 1]


def test_batches_keep_to_the_budget_longest_first_and_cover_every_sequence():
    lengths = [5, 30, 7, 30, 12, 1]
    batches = list(sbench.batches(lengths, 40))
    assert sorted(k for b in batches for k in b) == list(range(len(lengths)))
    assert all(len(b) == 1 or lengths[b[0]] * len(b) <= 40 for b in batches)
    assert [lengths[b[0]] for b in batches] == sorted((lengths[b[0]] for b in batches), reverse=True)
    assert batches[0] == [1]


def test_per_character_picks_and_the_paired_comparison():
    r = {'chars': [2, 20], 'answer': 1}
    assert sbench.picks(r, [-1.0, -5.0], 'sum') == 0 and sbench.picks(r, [-1.0, -5.0], 'per_char') == 1
    base = np.array([True] * 6 + [False] * 4)
    arm = np.array([True] * 1 + [False] * 9)
    stats, shares = sbench.compare([str(i) for i in range(10)], np.full(10, 0.25), base, arm, 2000, 0)
    assert stats['base_only'] == 5 and stats['arm_only'] == 0
    assert abs(stats['mcnemar_p'] - 2 / 32) < 1e-12
    assert abs(stats['difference'] + 0.5) < 1e-12
    assert abs(stats['share'] - (0.1 - 0.25) / (0.6 - 0.25)) < 1e-12
    same, _ = sbench.compare([str(i) for i in range(10)], np.full(10, 0.25), base, base, 200, 0)
    assert same['share'] == 1 and same['difference_ci95'] == [0, 0] and same['mcnemar_p'] == 1
    assert shares.shape == (2000,)


def test_overlap_finds_a_shared_window_and_nothing_else():
    seqs = np.array([list(range(100, 140)), list(range(200, 240))], dtype=np.uint32)
    ref = sbench.windows(seqs, 5)
    shared = {'answer': 0, '64k': {'context': [0, 1, 2, 3, 120, 121, 122], 'choices': [[123, 124]]}}
    apart = {'answer': 0, '64k': {'context': [0, 120, 121, 122, 123], 'choices': [[9]]}}
    assert sbench.overlapping([shared, apart], ref, 5) == [True, False]


def test_an_arm_name_can_carry_its_expert_count():
    assert sbench.variant('healed_k16') == ('healed', 16)
    assert sbench.variant('every_neuron') == ('every_neuron', None)
    assert sbench.variant('dense_full') == ('dense_full', None)
