import math
import types

import pytest
import torch

import heal_probe as hp
import slim_opd as so
from h0_mlp_probe import token_kl
from test_slim_batch import ids
from test_slim_heal import student

STOP = [94, 95]


def trained_student(seed=0):
    model, slim = student(seed)
    for m in slim.adapted.values():
        if m.b is not None:
            torch.nn.init.normal_(m.b, std=0.1)
    return model, slim


def tiny_vocab(model, kept=96):
    kept = torch.arange(kept)
    removed = torch.full((model.config.vocab_size,), -torch.inf)
    removed[kept] = 0
    return types.SimpleNamespace(kept=kept, removed=removed, stop=STOP, head=model.lm_head.weight[kept].detach())


def prompts(lengths=(5, 9, 12)):
    # Ids below the stop tokens, so a prompt never holds one.
    return [ids(10 + n, length=n, vocab=90) for n in lengths]


def test_loop_end_cuts_after_the_first_copied_token():
    unique = list(range(30))
    assert so.loop_end(unique * 2, 20, 5) == 31
    assert so.loop_end(unique + list(range(100, 160)), 20, 5) == 90
    # A short copy (3 repeated spans) is not a loop.
    assert so.loop_end(unique + unique[:22] + list(range(100, 140)), 20, 5) == 92
    assert so.loop_end([7, 8, 9] * 40, 4, 10) == 4


def test_padding_changes_nothing_once_each_rows_first_real_token_runs_every_neuron():
    model, slim = trained_student()
    drift = so.padding_drift(model, slim, prompts())
    assert drift['student'] < 1e-6 and drift['dense'] < 1e-6 and not slim.on
    assert 'select' not in vars(slim)
    # Without FirstReal the shorter prompts' first tokens are routed like any other.
    x, mask, first = so.left_pad(prompts(), 0)
    slim.on = True
    with torch.no_grad():
        plain = model(input_ids=x, attention_mask=mask, position_ids=(mask.cumsum(-1) - 1).clamp_min(0)).logits
        alone = model(input_ids=torch.tensor([prompts()[0]])).logits[0]
    slim.on = False
    assert token_kl(alone, plain[0, first[0]:]).mean() > 1e-2


def test_first_real_keeps_another_routing():
    model, slim = trained_student()
    hp.route(slim, 'all')
    custom = slim.select
    with so.FirstReal(slim, torch.tensor([0])):
        assert slim.select is not custom
    assert slim.select is custom
    hp.route(slim, 'learned')


@pytest.mark.parametrize('on', [True, False])
def test_batched_greedy_answers_match_each_prompt_alone(on):
    model, slim = trained_student(1)
    vocab = tiny_vocab(model)
    cfg = {'max_new_tokens': 8, 'state': 'fp32'}
    batch = so.generate(model, slim, vocab, prompts(), cfg, on, True)
    for p, row in zip(prompts(), batch, strict=True):
        assert so.generate(model, slim, vocab, [p], cfg, on, True) == [row]
    assert not slim.on and 'select' not in vars(slim)


def test_greedy_rows_stay_greedy_among_sampled_ones():
    model, slim = trained_student(2)
    vocab = tiny_vocab(model)
    cfg = {'max_new_tokens': 8, 'state': 'fp32'}
    sampling = {'temperature': 1.0, 'top_p': 0.95, 'top_k': 20, 'presence_penalty': 0.0}
    alone = so.generate(model, slim, vocab, prompts()[:1], cfg, True, True)
    mixed = so.generate(model, slim, vocab, prompts()[:2], cfg, True, [True, False], sampling, seed=3)
    assert mixed[0] == alone[0]
    assert mixed[1] != so.generate(model, slim, vocab, prompts()[1:2], cfg, True, True)[0]


def test_decode_keeps_the_64k_rows_and_one_token_for_greedy_rows():
    Decode, _ = so.stopping_classes()
    g = torch.Generator().manual_seed(4)
    scores = torch.randn(2, 96, generator=g)
    removed = torch.zeros(96)
    removed[80:] = -torch.inf
    sampling = {'temperature': 1.0, 'top_p': 1.0, 'top_k': 5, 'presence_penalty': 0.0}
    filtered = Decode(removed, 1, torch.tensor([True, False]), sampling)(torch.zeros(2, 3, dtype=torch.long), scores)
    assert torch.isfinite(filtered[0]).sum() == 1 and filtered[0].argmax() == scores[0, :80].argmax()
    assert torch.isfinite(filtered[1]).sum() == 5 and torch.isfinite(filtered[1, :80]).sum() == 5
    assert torch.equal(Decode(removed, 1, torch.tensor([True, True]), None)(None, scores), scores + removed)


def test_loop_stop_ends_running_rows_that_repeat():
    _, LoopStop = so.stopping_classes()
    loop = [1, 2, 3] * 20
    rows = torch.tensor([[9, 9] + loop, [9, 9] + list(range(10, 70)), [9, 9] + [5, STOP[0]] + loop[:-2]])
    stop = LoopStop(2, 3, STOP, {'every': 2, 'window': 4, 'run': 10})
    assert not stop(rows, None).any()
    assert stop(rows, None).tolist() == [True, False, False]


def test_batch_rows_mark_the_positions_that_predict_counted_answer_tokens():
    rows = [{'prompt': [1, 2, 3], 'answer': [4, 5, 6], 'count': 2}, {'prompt': [7], 'answer': [8, 9, 10, 11], 'count': 4}]
    x, mask = so.batch_rows(rows, 0)
    assert x.tolist() == [[1, 2, 3, 4], [7, 8, 9, 10]]
    assert mask.tolist() == [[False, False, True, True], [True, True, True, True]]
    for b, r in enumerate(rows):
        seq = r['prompt'] + r['answer']
        assert [seq[i + 1] for i in mask[b].nonzero().flatten().tolist()] == r['answer'][:r['count']]


@pytest.mark.parametrize('reverse', [True, False])
def test_micro_step_is_the_kl_at_counted_positions_and_trains_only_the_updates(reverse):
    model, slim = trained_student(3)
    vocab = tiny_vocab(model, 80)
    rows = [{'prompt': ids(1, 6, 80), 'answer': ids(2, 5, 80), 'count': 3},
            {'prompt': ids(3, 4, 80), 'answer': ids(4, 7, 80), 'count': 7}]
    total = so.micro_step(model, slim, vocab, rows, reverse, 4, 0.5)
    assert not slim.on and not model.training
    expected = 0.0
    for r in rows:
        x = torch.tensor([r['prompt'] + r['answer'][:r['count'] - 1]])
        with torch.no_grad():
            dense = model(x).logits[0, len(r['prompt']) - 1:, :80]
            slim.on = True
            routed = model(x).logits[0, len(r['prompt']) - 1:, :80]
            slim.on = False
        expected += float((token_kl(routed, dense) if reverse else token_kl(dense, routed)).sum())
    assert total == pytest.approx(expected, rel=1e-4)
    trained = {id(p) for p in slim.lora_parameters()}
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in slim.lora_parameters())
    assert all(p.grad is None for p in model.parameters() if id(p) not in trained)


def test_training_rows_count_the_students_answers_to_their_first_repeat():
    loop = {'window': 4, 'run': 10}
    asked = [{'id': 'a', 'ids': [1, 2]}, {'id': 'b', 'ids': [3]}]
    student = [{'tokens': [5, 6, 7] * 20, 'stop': None, 'loop_stopped': True},
               {'tokens': [8, 9], 'stop': STOP[0], 'loop_stopped': False}]
    rows = so.train_rows(student, asked, True, loop, [True, False])
    assert [r['count'] for r in rows] == [4, 3] and [r['greedy'] for r in rows] == [True, False]
    assert rows[1]['answer'] == [8, 9, STOP[0]] and rows[0]['prompt'] == [1, 2]
    teacher = so.train_rows([{'tokens': [5, 6, 7] * 20, 'stop': STOP[1], 'loop_stopped': False}], asked[:1], False, loop)
    assert teacher[0]['count'] == 61 and not teacher[0]['greedy']


def test_student_rows_alternate_decodings_in_pairs():
    assert so.pattern(8) == [True, True, False, False, True, True, False, False]


def test_learning_rate_warms_up_then_follows_the_time_budget_to_the_floor():
    cfg = {'warmup_steps': 4, 'lr_floor': 0.1}
    assert so.factor(0, 0.0, cfg) == 0.25
    assert so.factor(4, 0.0, cfg) == pytest.approx(1.0)
    assert so.factor(9, 0.5, cfg) == pytest.approx(0.55)
    assert so.factor(9, 2.0, cfg) == pytest.approx(0.1)


def test_fisher_less_is_the_hypergeometric_tail():
    assert so.fisher_less(0, 5, 5, 5) == pytest.approx(1 / math.comb(10, 5))
    assert so.fisher_less(15, 48, 18, 48) == pytest.approx(0.334, abs=1e-3)
    assert so.fisher_less(5, 5, 5, 5) == pytest.approx(1.0)
