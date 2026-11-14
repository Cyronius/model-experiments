import copy
import math
import types
import weakref

import numpy as np
import pytest
import torch

import heal_probe as hp
import moe_carve as mc
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


@pytest.mark.parametrize('on', [True, False])
def test_compact_decoding_answers_as_generate_while_rows_leave_the_batch(on):
    model, slim = trained_student(1)
    vocab = tiny_vocab(model)
    cfg = {'max_new_tokens': 12, 'state': 'fp32'}
    full = so.generate(model, slim, vocab, prompts(), cfg, on, True)
    # Stop tokens that end the rows at different steps, so rows leave the batch one by one.
    vocab.stop = [full[0]['tokens'][2], full[2]['tokens'][6]]
    reference = so.generate(model, slim, vocab, prompts(), cfg, on, True)
    assert sum(r['stop'] is not None for r in reference) >= 2
    assert so.generate(model, slim, vocab, prompts(), dict(cfg, compact=True), on, True) == reference
    assert not slim.on and 'select' not in vars(slim)


def test_compact_decoding_keeps_each_rows_greedy_flag_and_stored_format_after_others_leave():
    model, slim = trained_student(2)
    vocab = tiny_vocab(model)
    cfg = {'max_new_tokens': 10, 'state': 'fp32', 'compact': True}
    sampling = {'temperature': 1.0, 'top_p': 0.95, 'top_k': 20, 'presence_penalty': 0.0}
    # The second stop token also pads.
    vocab.stop = [so.generate(model, slim, vocab, prompts()[:1], cfg, True, True)[0]['tokens'][1], STOP[1]]
    alone = so.generate(model, slim, vocab, prompts()[2:], dict(cfg, compact=False), True, True)
    mixed = so.generate(model, slim, vocab, prompts(), cfg, True, [True, False, True], sampling, seed=3)
    assert mixed[0]['stop'] == vocab.stop[0] and len(mixed[0]['tokens']) == 1
    # Once row 0 has left, row 2 is still decoded greedily: its answer is the one it gives alone.
    assert mixed[2] == alone[0]


def test_out_of_memory_halves_the_batch_once_the_failed_attempt_is_freed(monkeypatch):
    model, slim = trained_student(1)
    vocab = tiny_vocab(model)
    cfg = {'max_new_tokens': 8, 'state': 'fp32'}
    alone = [so.generate(model, slim, vocab, [p], cfg, True, True)[0] for p in prompts()]
    real, failed = model.generate, []

    def generate(**kwargs):
        # A retry inside the except clause would still hold the failed attempt's cache.
        assert all(cache() is None for cache in failed)
        if len(kwargs['input_ids']) > 1:
            cache = torch.zeros(4)
            failed.append(weakref.ref(cache))
            raise torch.cuda.OutOfMemoryError('test')
        return real(**kwargs)
    monkeypatch.setattr(model, 'generate', generate)
    assert so.generate(model, slim, vocab, prompts(), cfg, True, True) == alone
    assert len(failed) == 2


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


def healed_student(seed):
    """A trained student whose adapter, as load_healed keeps it, is the healing run's; then the
    student's own copy moves away from it, as training moves it."""
    model, slim = trained_student(seed)
    slim.adapters = {True: copy.deepcopy(slim.state())}
    with torch.no_grad():
        for p in slim.lora_parameters():
            p.add_(torch.randn_like(p) * 0.05)
    return model, slim


@pytest.mark.parametrize('reverse', [True, False])
@pytest.mark.parametrize('kind', ['dense', 'every_neuron'])
def test_micro_step_is_the_kl_at_counted_positions_and_trains_only_the_updates(reverse, kind):
    model, slim = healed_student(3)
    vocab = tiny_vocab(model, 80)
    rows = [{'prompt': ids(1, 6, 80), 'answer': ids(2, 5, 80), 'count': 3},
            {'prompt': ids(3, 4, 80), 'answer': ids(4, 7, 80), 'count': 7}]
    trained = [p.detach().clone() for p in slim.lora_parameters()]
    total = so.micro_step(model, slim, vocab, so.Teacher(slim, kind), rows, reverse, 4, 0.5)
    assert not slim.on and not model.training and 'select' not in vars(slim)
    assert all(torch.equal(p, t) for p, t in zip(slim.lora_parameters(), trained))
    # The teacher by hand: the dense model, or the healed adapter with every neuron.
    expected = 0.0
    for r in rows:
        x = torch.tensor([r['prompt'] + r['answer'][:r['count'] - 1]])
        with torch.no_grad():
            if kind == 'dense':
                target = model(x).logits[0, len(r['prompt']) - 1:, :80]
            else:
                slim.load(slim.adapters[True])
                hp.route(slim, 'all')
                slim.on = True
                target = model(x).logits[0, len(r['prompt']) - 1:, :80]
                hp.route(slim, 'learned')
                for p, t in zip(slim.lora_parameters(), trained):
                    p.copy_(t)
            slim.on = True
            routed = model(x).logits[0, len(r['prompt']) - 1:, :80]
            slim.on = False
        expected += float((token_kl(routed, target) if reverse else token_kl(target, routed)).sum())
    assert total == pytest.approx(expected, rel=1e-4)
    ids_trained = {id(p) for p in slim.lora_parameters()}
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in slim.lora_parameters())
    assert all(p.grad is None for p in model.parameters() if id(p) not in ids_trained)


def test_the_every_neuron_teacher_answers_as_the_healed_student_with_every_neuron():
    model, slim = healed_student(5)
    vocab = tiny_vocab(model)
    cfg = {'max_new_tokens': 8, 'state': 'fp32'}
    trained = [p.detach().clone() for p in slim.lora_parameters()]
    with so.Teacher(slim, 'every_neuron') as teacher:
        answers = so.generate(model, slim, vocab, prompts(), cfg, teacher.on, True)
    assert all(torch.equal(p, t) for p, t in zip(slim.lora_parameters(), trained)) and 'select' not in vars(slim)
    slim.load(slim.adapters[True])
    hp.route(slim, 'all')
    assert answers == so.generate(model, slim, vocab, prompts(), cfg, True, True)
    hp.route(slim, 'learned')
    assert answers != so.generate(model, slim, vocab, prompts(), cfg, True, True)
    with pytest.raises(ValueError):
        so.Teacher(slim, 'perfect')


def test_text_comes_first_from_the_sequences_healing_never_reached():
    order = so.text_order(40, 7, 32)
    healed = np.random.default_rng(7).permutation(40)
    assert order[:8].tolist() == healed[32:].tolist() and order[8:].tolist() == healed[:32].tolist()


def test_text_rows_count_every_position_after_the_first_and_train_toward_the_dense_model():
    rows = so.text_rows(np.array([[5, 6, 7, 8], [9, 10, 11, 12]], dtype=np.uint32))
    x, mask = so.batch_rows(rows, 0)
    assert x.tolist() == [[5, 6, 7], [9, 10, 11]] and mask.all()
    student = {'prompt': [1], 'answer': [2], 'count': 1, 'reverse': True}
    teacher = {'prompt': [1], 'answer': [2], 'count': 1, 'reverse': False}
    split = so.parts([student, rows[0], teacher])
    assert split == {'forward': [teacher], 'text': [rows[0]], 'reverse': [student]}


def test_the_start_adapter_must_be_its_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(so.mc, 'OUT', tmp_path)
    _, slim = trained_student(6)
    torch.save(slim.state(torch.bfloat16), tmp_path / 'adapter.pt')
    cfg = {'start': {'adapter': 'adapter.pt', 'record': 'train.json'}}
    (tmp_path / 'train.json').write_text(so.json.dumps({'adapter_out_sha256': so.sha(tmp_path / 'adapter.pt')}))
    state = so.start_state(cfg)
    assert all(torch.equal(a, b) for n in state['lora'] for a, b in zip(state['lora'][n], slim.state(torch.bfloat16)['lora'][n]))
    assert so.start_state({}) is None
    (tmp_path / 'train.json').write_text(so.json.dumps({'adapter_out_sha256': '0' * 64}))
    with pytest.raises(ValueError):
        so.start_state(cfg)


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


def test_a_resumed_run_warms_up_again_to_the_cosine():
    cfg = {'warmup_steps': 4, 'lr_floor': 0.1}
    assert so.factor(20, 0.5, cfg, since=20) == pytest.approx(0.25 * 0.55)
    assert so.factor(23, 0.5, cfg, since=20) == pytest.approx(0.55)
    assert so.factor(9, 0.5, cfg, since=0) == so.factor(9, 0.5, cfg)


def test_resuming_keeps_the_rounds_up_to_the_checkpoint_and_the_schedules_place(tmp_path, monkeypatch):
    monkeypatch.setattr(so.mc, 'OUT', tmp_path)
    monkeypatch.setattr(so, 'FOLDER', 'run')
    cfg = {'checkpoint_every': 3, 'warmup_steps': 4, 'lr_floor': 0.1, 'lr': 5e-5}
    # 7 rounds of 2 steps, the time budget's share done at step s s / 20; latest.pt holds 6 rounds.
    steps = [{'step': s + 1, 'round': s // 2, 'lr': 5e-5 * so.factor(s, s / 20, cfg)} for s in range(14)]
    rounds = [{'round': r, 'steps': 2 * r + 2} for r in range(7)]
    (tmp_path / 'run').mkdir()
    write = lambda **more: (tmp_path / 'run/progress.json').write_text(
        so.json.dumps({'rounds': rounds, 'steps': steps, 'mid': None} | more))
    write()
    kept, kept_steps, mid, done = so.resume_point(cfg)
    assert kept == rounds[:6] and kept_steps == steps[:12] and mid is None
    assert done == pytest.approx(11 / 20)
    # A checkpoint the run recorded wins over the rule.
    write(checkpoint=3, mid={'round': 1})
    kept, _, mid, done = so.resume_point(cfg)
    assert len(kept) == 3 and done == pytest.approx(5 / 20) and mid == {'round': 1}
    write(checkpoint=1)
    with pytest.raises(ValueError):
        so.resume_point(cfg)


def test_a_longer_cap_is_compared_prompt_by_prompt_with_the_answers_the_shorter_cap_cut():
    loop = {'window': 2, 'last': 8, 'threshold': 0.45}
    row = lambda i, tokens, finished, correct=False: {'id': i, 'tokens': tokens, 'length': len(tokens),
                                                       'finished': finished, 'correct': correct}
    short = [row('a', [1, 2], True), row('b', [1, 2, 1, 2, 1, 2], False), row('c', [3, 4, 5, 6, 7, 8], False),
             row('d', [1, 2], True)]
    long = [row('a', [1, 2], True), row('b', [1, 2, 1, 2, 1, 2, 1, 2, 1, 2], False),
            row('c', [3, 4, 5, 6, 7, 8, 9], True, True), row('d', [1, 2, 3], True), row('e', [1], True)]
    c = so.against_shorter(long, short, loop)
    # d ended at 2 tokens before and runs on now; e has no shorter answer.
    assert c['prompts'] == 4 and c['same_start'] == 3 and c['cut'] == 2
    assert c['cut_before']['loops'] == 1 and c['cut_before']['finished'] == 0
    assert c['cut_after']['finished'] == 1 and c['cut_after']['correct'] == 1 and c['cut_after']['loops'] == 1
    assert so.against_shorter(short[:1], short[:1], loop)['cut_before'] is None


def test_fisher_less_is_the_hypergeometric_tail():
    assert so.fisher_less(0, 5, 5, 5) == pytest.approx(1 / math.comb(10, 5))
    assert so.fisher_less(15, 48, 18, 48) == pytest.approx(0.334, abs=1e-3)
    assert so.fisher_less(5, 5, 5, 5) == pytest.approx(1.0)


def test_more_experts_run_more_neurons_after_the_first_position():
    model, slim = trained_student()
    x = torch.randn(1, 6, model.config.intermediate_size)
    h = torch.randn(1, 6, model.config.hidden_size)
    k = mc.at(slim.s, 'K', 0)
    counts = lambda: slim.select(0, x, h)[0, 1:].sum(-1)
    base = counts()
    so.experts(slim, k + 1)
    assert (counts() - base == mc.at(slim.s, 'c', 0)).all()


def test_a_smaller_cap_cuts_and_rescores_the_answers_that_did_not_stop_within_it():
    tok = types.SimpleNamespace(decode=lambda t: ' '.join(map(str, t)))
    prompts = [{'id': 'a', 'answer': 7}, {'id': 'b', 'answer': 3}, {'id': 'c', 'answer': 5}]
    rows = [{'id': 'a', 'tokens': [1, 7], 'length': 2, 'finished': True, 'answer': 7, 'correct': True},
            {'id': 'b', 'tokens': [1, 2, 3, 4, 3], 'length': 5, 'finished': True, 'answer': 3, 'correct': True},
            {'id': 'c', 'tokens': [5, 1, 5, 2, 9, 9], 'length': 6, 'finished': False, 'answer': 9, 'correct': False}]
    cut = so.truncated(rows, prompts, tok, 4)
    assert cut[0] is rows[0]
    assert cut[1]['tokens'] == [1, 2, 3, 4] and not cut[1]['finished'] and not cut[1]['correct']
    assert cut[2]['length'] == 4 and cut[2]['answer'] == 2
    # Finishing in exactly cap tokens leaves no room for the stop token.
    assert not so.truncated(rows, prompts, tok, 2)[0]['finished']


def test_compact_decoding_ends_a_row_with_non_finite_logits_and_flags_it():
    """A row whose logits turn non-finite stops there, unfinished and flagged; the others go on."""
    import types
    vocab = types.SimpleNamespace(stop=[9, 8])

    class Model:
        config = None

        def __call__(self, input_ids, attention_mask, position_ids, past_key_values, use_cache, logits_to_keep):
            b = input_ids.shape[0]
            logits = torch.full((b, 1, 10), -10.0)
            logits[:, 0, 1] = 10.0
            if attention_mask.shape[1] == 4 and b == 2:
                logits[1, 0, 0] = float('nan')
            return types.SimpleNamespace(logits=logits)

    class Processor:
        greedy = torch.tensor([True, True])

        def __call__(self, history, logits):
            return logits

    stored = types.SimpleNamespace(variants=['fp32', 'fp32'])
    import slim_opd as so
    from unittest import mock
    with mock.patch('transformers.DynamicCache') as cache:
        cache.return_value.reorder_cache = lambda keep: None
        ids = torch.tensor([[5, 6], [5, 6]])
        rows = so.compact(Model(), vocab, ids, torch.ones_like(ids), Processor(), stored, 4, True)
    assert rows[0] == {'tokens': [1, 1, 1, 1], 'stop': None, 'loop_stopped': False}
    assert rows[1] == {'tokens': [1, 1], 'stop': None, 'loop_stopped': False, 'nonfinite_at': 2}
