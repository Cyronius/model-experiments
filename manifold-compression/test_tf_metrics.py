import argparse
import copy

import pytest
import torch

from local_projection_models import build
from test_projection_models import tiny_model
from tf_metrics import (choice_logprobs, choice_metrics, compare_scores, document_rows, grams, overlapping,
                        paired_bootstrap, score_row, stage_score, text_metrics)


def projected():
    torch.manual_seed(5)
    p = torch.linalg.qr(torch.randn(32, 32, dtype=torch.float64)).Q.float()[:, :24].contiguous()
    return build(tiny_model('qwen', True), [p] * 2, [1.04] * 5)


def text_row(seed=3, length=21):
    g = torch.Generator().manual_seed(seed)
    return {'id': f't{seed}', 'group': f'g{seed}', 'source': 'text', 'kind': 'text',
            'input_ids': torch.randint(0, 101, (length,), generator=g).tolist(), 'positions': list(range(4, length - 1))}


def test_text_metrics_match_direct_full_vocabulary_computation():
    teacher, student, row = tiny_model('qwen', True), projected(), text_row()
    got = text_metrics(teacher, student, row, chunk=5)
    ids = torch.tensor([row['input_ids']])
    pos = torch.tensor(row['positions'])
    lp = teacher(ids, use_cache=False).logits[0, pos].double().log_softmax(-1)
    lq = student(ids, use_cache=False).logits[0, pos].double().log_softmax(-1)
    targets = ids[0, pos + 1]
    assert abs(got['kl'] - (lp.exp() * (lp - lq)).sum().item()) < 1e-6
    assert got['top1'] == (lp.argmax(-1) == lq.argmax(-1)).sum().item()
    assert abs(got['ce'] + lq.gather(-1, targets[:, None]).sum().item()) < 1e-6
    assert got['kl'] > 0 and got['tokens'] == len(pos)


def test_teacher_against_itself_is_exactly_zero():
    teacher = tiny_model('qwen', True)
    got = text_metrics(teacher, teacher, text_row())
    assert got['kl'] == 0 and got['top1'] == got['tokens'] and got['ce'] == got['teacher_ce']


def test_choice_scores_use_identical_context_and_all_three_rules():
    model = tiny_model('qwen', True)
    row = {'id': 'm', 'group': 'mc/x', 'source': 'mc', 'kind': 'mc', 'context_ids': [5, 9, 13, 2],
           'choice_ids': [[7], [8, 30, 31], [40, 41]], 'choice_chars': [2, 12, 5], 'answer': 1}
    got = choice_metrics(model, row)
    full = model(torch.tensor([[5, 9, 13, 2, 8, 30, 31]]), use_cache=False).logits[0].double().log_softmax(-1)
    direct = full[3, 8] + full[4, 30] + full[5, 31]
    assert abs(got['logprobs'][1] - direct.item()) < 1e-6
    scores = {'sum': got['logprobs'], 'per_token': [got['logprobs'][i] / n for i, n in enumerate((1, 3, 2))],
              'per_char': [got['logprobs'][i] / n for i, n in enumerate((2, 12, 5))]}
    for rule, values in scores.items():
        assert got['picks'][rule] == max(range(3), key=values.__getitem__)
        assert got['correct'][rule] == int(got['picks'][rule] == 1)


def test_batched_and_single_token_choice_paths_match_separate_forwards():
    model = projected()
    context = [5, 9, 13, 2, 44]
    for choices in ([[7, 3, 3, 1], [8], [30, 31]], [[7], [8], [30]]):
        got = choice_logprobs(model, context, choices)
        for c, value in zip(choices, got):
            lp = model(torch.tensor([context + c]), use_cache=False).logits[0].double().log_softmax(-1)
            direct = sum(lp[len(context) - 1 + j, t].item() for j, t in enumerate(c))
            assert abs(value - direct) < 1e-5


def test_document_chunks_score_every_next_token_and_group_by_document():
    class Tok:
        def __call__(self, text, add_special_tokens=False):
            return argparse.Namespace(input_ids=[int(w) for w in text.split()])
    rows = document_rows(Tok(), {'id': 'a', 'text': ' '.join(map(str, range(25)))}, 'prose', 8, 2)
    assert [r['input_ids'] for r in rows] == [list(range(8)), list(range(8, 16))]
    assert all(r['positions'] == list(range(7)) and r['group'] == 'prose/a' for r in rows)
    assert len(document_rows(Tok(), {'id': 'b', 'text': '1 2 3'}, 'prose', 8, 2)) == 0


def test_overlap_audit_flags_planted_training_text():
    row = text_row(length=40)
    seen = grams(list(range(500, 600)))
    assert not overlapping(row, seen)
    seen |= grams(row['input_ids'][10:30])
    assert overlapping(row, seen)


def test_bootstrap_resamples_groups_and_weights_tokens():
    same = paired_bootstrap(['a', 'b', 'c'], [1., 2., 3.], [1., 2., 3.], [1, 2, 3])
    assert same['difference'] == 0 and same['ci95'] == [0, 0]
    # Group 'a' holds every change; its two items move together under resampling.
    got = paired_bootstrap(['a', 'a', 'b', 'c'], [0., 0., 5., 5.], [2., 2., 5., 5.], [1, 1, 2, 2], draws=4000)
    assert abs(got['difference'] - 4 / 6) < 1e-12
    assert got['groups'] == 3 and got['ci95'][0] == 0 and got['ci95'][1] > got['difference']


def test_compare_reports_paired_transitions():
    teacher, student = tiny_model('qwen', True), projected()
    rows = [text_row(s) for s in range(4)]
    rows.append({'id': 'm', 'group': 'mc/x', 'source': 'mc', 'kind': 'mc', 'context_ids': [5, 9],
                 'choice_ids': [[7], [8]], 'choice_chars': [2, 2], 'answer': 0})
    base = {'arm': 'stock', 'manifest_sha256': 'x', 'rows': [score_row(teacher, teacher, r) for r in rows]}
    arm = {'arm': 'p', 'manifest_sha256': 'x', 'rows': [score_row(teacher, student, r) for r in rows]}
    out = compare_scores(base, arm, draws=500)
    assert out['metrics']['text/kl']['base'] == 0 and out['metrics']['text/kl']['difference'] > 0
    mc = out['metrics']['mc/accuracy_sum']
    assert mc['items'] == 1 and mc['base_only_correct'] + mc['arm_only_correct'] <= 1
    unpaired, duplicated, retokenized = copy.deepcopy(arm), copy.deepcopy(arm), copy.deepcopy(arm)
    unpaired['rows'].pop()
    duplicated['rows'][1] = copy.deepcopy(duplicated['rows'][0])
    retokenized['rows'][0]['tokens'] += 1
    for bad in (unpaired, duplicated, retokenized):
        with pytest.raises(ValueError):
            compare_scores(base, bad)


def test_score_rejects_a_custom_loader_named_like_a_registered_arm(tmp_path):
    args = argparse.Namespace(name='stock', arm=None, loader='test_projection_models:tiny_model', out=tmp_path)
    with pytest.raises(ValueError, match='registered arm'):
        stage_score(args)
