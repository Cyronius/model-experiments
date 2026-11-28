import json

import numpy as np
import torch

import loop_breaker as lb
import loop_breaker2 as lb2
from test_loop_breaker import Tok as Ids


class Tok(Ids):
    def decode(self, tokens):
        return ' '.join(str(t) for t in tokens)


def tiny(tmp_path):
    import heal_block as hb
    from test_heal_block import block as tiny_block
    model, slim, first = tiny_block(layers=(6, 7))
    torch.save({'layers': [6, 7], 'copies': {i: c.state_dict() for i, c in zip([6, 7], first.copies)},
                'routers': {i: slim.routers[i].state_dict() for i in (6, 7)}}, tmp_path / 'block.pt')
    return model, slim, hb.Pieces(model, slim, hb.load_pieces(model, slim, [tmp_path / 'block.pt']))


def test_the_first_neurons_answer_and_the_unedited_model_is_captured_over_those_answers(tmp_path):
    import heal_block_check as hbc
    from test_slim_batch import ids
    model, slim, block = tiny(tmp_path)
    h = model.config.hidden_size
    first = {'layer': 7, 'fit': {'batch': 2, 'generate': {'state': 'fp32', 'max_new_tokens': 10, 'compact': True, 'seed': 0}}}
    prompts = [{'id': f'p{k}', 'answer': 1.0, 'ids': ids(k, length=7 + k)} for k in range(3)]
    torch.manual_seed(0)
    saved = {'clamp': 4.0, 'v': torch.randn(h), 'q': torch.randn(2, h)}
    saved['gate'], saved['up'], saved['down'] = lb.neuron_weights(torch.randn(h), torch.randn(h) / 8, 40 * torch.randn(h), -5.0, 4.0, 4.0)
    rows = lb2.answers_and_captures(model, slim, block, Tok(), prompts, first, saved, tmp_path)
    assert len(rows) == 3 and [json.loads(line)['id'] for line in (tmp_path / 'gen_fit.jsonl').open()] == ['p0', 'p1', 'p2']
    piece = dict(block.prefix)[7]
    assert piece.mlp._modules.get('bridge') is None
    # The neurons changed the answers (they are large and always on), and the capture is the unedited model's.
    plain = lb.greedy(model, slim, block, lb.vocab_of(model, Tok()), prompts, first['fit']['generate'], 2, 0)
    assert any(a['tokens'] != b['tokens'] for a, b in zip(rows, plain))
    k = 1
    z = np.load(tmp_path / 'capture' / f'{k}.npz')
    assert z['x'].shape == (len(rows[k]['tokens']) + 1, h)
    with hbc.Student(block):
        logits = model.lm_head(lb.student_pass(model, prompts[k]['ids'] + rows[k]['tokens'])[len(prompts[k]['ids']) - 1:]).float()
    torch.testing.assert_close(logits[:, [3, 4]], torch.from_numpy(z['stop_logits']), atol=0.05, rtol=0.05)
    # A second call reuses the saved answers.
    again = lb2.answers_and_captures(model, slim, block, Tok(), prompts, first, saved, tmp_path)
    assert [r['tokens'] for r in again] == [r['tokens'] for r in rows]


def test_closed_after_marks_the_positions_behind_a_close():
    closed, second = lb2.closed_after([7, 3, 8, 3, 9], 3)
    assert closed.tolist() == [False, False, True, True, True, True] and second.tolist() == [False] * 4 + [True] * 2


def reads(h=32, seed=0):
    torch.manual_seed(seed)
    w, c, n, m = torch.randn(h) / 4, torch.randn(h) / 8, torch.randn(h) / 6, torch.randn(h) / 6
    q = torch.randn(2, h)
    saved = {'w': w, 'c': c, 'b': 0.2, 'k': 4.0, 'clamp': 4.0, 'q': q, 'rms': 2.0, 'v': 5 * lb.stop_write(q[:1], 2.0)}
    saved['gate'], saved['up'], saved['down'] = lb.neuron_weights(w, c, saved['v'], 0.2, 4.0, 4.0, n)
    return saved, m


def test_the_four_neurons_compute_both_activations_and_the_two_writes():
    saved, m = reads()
    v_end1 = lb2.end_write(saved['q'], saved['rms'])
    torch.testing.assert_close(saved['q'] @ v_end1 / saved['rms'], torch.tensor([0.0, 1.0]), atol=1e-4, rtol=1e-4)
    gate, up, down = lb2.four_neurons(saved, m, v_end1, 12.0, 16.0)
    assert gate.shape == (4, 32) and up.shape == (4, 32) and down.shape == (32, 4)
    assert torch.equal(gate[:2], saved['gate']) and torch.equal(down[:, :2], saved['down'])
    neurons = lb.Breaker(32, 4)
    with torch.no_grad():
        neurons.gate.weight.copy_(gate)
        neurons.up.weight.copy_(up)
        neurons.down.weight.copy_(down)
    x = torch.randn(200, 32) * 3
    t = {'s': x @ saved['w'], 'u': x @ saved['c'], 'nx': x @ saved['up'][0], 'mx': x @ m}
    a1, a2 = lb2.activations(t, saved, 16.0)
    assert float(a1.abs().max()) > 0.3 and float(a2.abs().max()) > 0.3
    f = neurons.features(x)
    torch.testing.assert_close((f[:, 0] - f[:, 1]) / 4, a1, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close((f[:, 2] - f[:, 3]) / 4, a2, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(neurons(x), a1[:, None] * saved['v'] + a2[:, None] * (12.0 * v_end1 - saved['v']), atol=1e-3, rtol=1e-3)


def test_judge_names_what_the_four_neurons_do_to_an_answer():
    think, end, word = 3, 4, 9
    # Five positions, candidates [word, </think>, <|im_end|>]; the unedited model always says the word.
    tokens = torch.tensor([[word, think, end]] * 5)
    base = torch.tensor([[10.0, 0.0, 0.0]] * 5)
    t = {'base': base, 'tokens': tokens, 'think': torch.tensor([[0.0, 16.0, 2.0]] * 5), 'end': torch.tensor([[0.0, 0.0, 1.0]] * 5)}
    off, on = torch.zeros(5), torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0])
    opened, behind = torch.zeros(5, dtype=torch.bool), torch.tensor([False, True, True, True, True])
    none = torch.zeros(5, dtype=torch.bool)
    stops = (think, end)
    # The first pair alone closes the thinking at 2; with the second pair silent the four keep that.
    assert lb2.judge('unedited', 'loop', t, on, off, 12.0, opened, none, stops) == ('think', 'kept', 2, think)
    # The second pair fully on while the thinking is open: <|im_end|> inside the thinking.
    assert lb2.judge('unedited', 'loop', t, on, on, 12.0, opened, none, stops) == ('think', 'end_in_think', 2, end)
    # The first pair fires behind a close: the four end the answer there; a weak second pair leaves the second close.
    assert lb2.judge('first', 'loop', t, on, on, 12.0, behind, none, stops) == ('closed', 'ended', 2, end)
    assert lb2.judge('first', 'loop', t, on, 0.2 * on, 12.0, behind, none, stops) == ('closed', 'none', None, None)
    assert lb2.judge('unedited', 'loop', t, on, 0.2 * on, 12.0, behind, none, stops) == ('closed', 'second_close', 2, think)
    # A finished answer no pair touches; and positions behind a second close do not make a 'closed' answer.
    assert lb2.judge('first', 'done', t, off, off, 12.0, behind, none, stops) == ('done', 'none', None, None)
    assert lb2.judge('first', 'cut', t, on, on, 12.0, behind, behind, stops)[0] == 'rest'
    counts = {'done': {'none': 50, 'changed': 1}, 'think': {'kept': 19, 'other': 1}, 'closed': {'ended': 3}}
    assert lb2.allowed(counts, 0.02) and not lb2.allowed(dict(counts, think={'kept': 17, 'other': 3}), 0.02)
    assert not lb2.allowed(dict(counts, closed={'ended': 2, 'end_in_think': 1}), 0.02)
    grid = [{'dose': d, 'counts': {'closed': {'ended': e}}} for d, e in ((12, 2), (16, 4), (20, 4), (28, 4))]
    assert lb2.select(grid, [12, 16, 20, 24, 28])['dose'] == 20          # the smallest with the most, plus one step
    assert lb2.select(grid[:2], [12, 16, 20])['dose'] == 16 and lb2.select(grid[2:], [12, 16, 20, 24, 28])['dose'] == 20   # no allowed step above


def test_build_runs_from_the_two_sets_of_captures_to_a_report(tmp_path):
    model, slim, block = tiny(tmp_path)
    rng = np.random.default_rng(0)
    text = lambda n: rng.integers(6, 96, n).tolist()
    first = {'layer': 7, 'loop': {'window': 5, 'last': 100, 'threshold': 0.45},
             'cycle': {'min_period': 8, 'max_period': 40, 'agree': 0.9}, 'labels': {'margin': 4},
             'probe': {'ridge': 0.01}, 'neurons': {'k': 4, 'clamp': 4, 'up': 'line_end', 'tokens': ['</think>']},
             'grid': {'false_positive_rates': [0.2, 0.05], 'doses': [4, 16]}, 'screen': {'window': 5}}
    one, two = tmp_path / 'one', tmp_path / 'two'
    prompts = [{'ids': text(8)} for _ in range(8)]
    for folder, close in ((one, False), (two, True)):
        folder.mkdir()
        rows = []
        for k in range(8):
            loop = k % 2 == 0
            tokens = (text(20) + ([3] if close else []) + text(12) * 6) if loop else text(20) + [3] + text(20)
            rows.append({'id': f'row-{k}', 'tokens': tokens, 'length': len(tokens), 'finished': not loop, 'correct': False})
        (folder / 'gen_fit.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
        lb.capture(model, block, Tok(), rows, prompts, first, folder)
    (one / 'tokens.json').write_text(json.dumps({'line_ends': list(range(6, 20)), 'think': 3, 'written': [3],
                                                 'prompt_last': [p['ids'][-1] for p in prompts]}))
    import argparse
    lb.stage_build(argparse.Namespace(threads=2), first, one)
    if not (one / 'breaker.pt').exists():          # the tiny first screen allowed no setting: write neurons by hand
        saved, _ = reads(32, 1)
        torch.save(saved, one / 'breaker.pt')
    rules = {'neurons': {'line': 16.0}, 'closed': {'ridge': 0.01}, 'grid': {'doses': [8, 16]}}
    lb2.build(first, one, rules, two)
    report = json.loads((two / 'build.json').read_text())
    assert report['answers']['first'] == {'loop': 4, 'drift': 0, 'done': 4, 'cut': 0} and len(report['grid']) == 2
    assert 0 <= report['closed_read']['held_out']['auroc_line_ends'] <= 1      # a random tiny model: no real read
    assert all(set(g) == {'dose', 'counts', 'allowed'} for g in report['grid'])
    if report['selected']:
        saved2 = torch.load(two / 'breaker2.pt', weights_only=True)
        assert saved2['gate'].shape == (4, 32) and saved2['down'].shape == (32, 4) and saved2['end_dose'] == report['selected']['dose']
        assert len(report['held_out']['answers']) == 8 and isinstance(report['gate'], bool)


def test_answers_of_records_each_pair_and_takes_the_neurons_out_again(tmp_path):
    from test_slim_batch import ids
    model, slim, block = tiny(tmp_path)
    first = {'layer': 7}
    cfg = {'state': 'fp32', 'max_new_tokens': 9, 'compact': None, 'seed': 0}
    prompts = [{'id': f'p{k}', 'answer': 1.0, 'ids': ids(k, length=6 + k)} for k in range(3)]
    plain, none = lb2.answers_of(model, slim, block, Tok(), prompts, cfg, first, None, 'unedited')
    assert none is None and len(plain) == 3
    saved, m = reads(model.config.hidden_size)
    saved['gate'], saved['up'], saved['down'] = lb2.four_neurons(saved, m, lb2.end_write(saved['q'], saved['rms']), 12.0, 16.0)
    rows, acts = lb2.answers_of(model, slim, block, Tok(), prompts, cfg, first, saved, 'breaker2')
    assert acts.shape[0] == 2 and acts.shape[2] == 3 and acts.shape[1] >= max(len(r['tokens']) for r in rows)
    assert dict(block.prefix)[7].mlp._modules.get('bridge') is None
    again, _ = lb2.answers_of(model, slim, block, Tok(), prompts, cfg, first, None, 'unedited')
    assert [r['tokens'] for r in again] == [r['tokens'] for r in plain]


def test_outcome_counts_what_the_predictions_name_and_pairs_the_arms(tmp_path):
    think, loop = 9, {'window': 4, 'last': 40, 'threshold': 0.45}
    row = lambda k, tokens, finished, correct=False: {'id': f'p{k}', 'tokens': tokens, 'length': len(tokens),
                                                      'finished': finished, 'correct': correct}
    cycle = [1, 2, 3, 4, 5] * 12
    unedited = [row(0, cycle, False), row(1, [1, 2, think, 3], True, True), row(2, [5, 6, 7], True), row(3, cycle, False)]
    breaker2 = [row(0, cycle[:20] + [think, 6], True, True), row(1, [1, 2, think, 3], True, True),
                row(2, [5, 6, 7], True), row(3, cycle[:20] + [think, 7, think, think], False)]
    files = {'check': {'unedited': tmp_path / 'u.jsonl', 'breaker': tmp_path / 'none.jsonl', 'breaker2': tmp_path / 'b.jsonl'}}
    for path, rows in ((files['check']['unedited'], unedited), (files['check']['breaker2'], breaker2)):
        path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    report = lb2.outcome(files, think, loop)['check']
    assert report['marks']['unedited'] == {'right': 1, 'right_and_finished': 1, 'finished': 2, 'loops': 2, 'three_or_more_closes': 0,
                                           'ended_inside_thinking': 1, 'cut_inside_thinking': 2}
    assert report['marks']['breaker2'] == {'right': 2, 'right_and_finished': 2, 'finished': 3, 'loops': 1, 'three_or_more_closes': 1,
                                           'ended_inside_thinking': 1, 'cut_inside_thinking': 0}
    assert list(report['pairs']) == ['breaker2_against_unedited']
    pair = report['pairs']['breaker2_against_unedited']
    assert (pair['loops_fixed'], pair['loops_new'], pair['right_gained'], pair['right_lost']) == (1, 0, 1, 0)


def test_sampled_answers_repeat_with_the_seed_and_silent_neurons_leave_them_alone(tmp_path):
    from test_slim_batch import ids
    model, slim, block = tiny(tmp_path)
    first = {'layer': 7}
    cfg = {'state': 'fp32', 'max_new_tokens': 9, 'compact': None, 'seed': 0,
           'sampling': {'temperature': 1.0, 'top_p': 0.95, 'top_k': 20, 'presence_penalty': 1.5}}
    prompts = [{'id': f'p{k}', 'answer': 1.0, 'ids': ids(k, length=6 + k)} for k in range(3)]
    plain, _ = lb2.answers_of(model, slim, block, Tok(), prompts, cfg, first, None, 'unedited', sampled=True)
    again, _ = lb2.answers_of(model, slim, block, Tok(), prompts, cfg, first, None, 'unedited', sampled=True)
    greedy, _ = lb2.answers_of(model, slim, block, Tok(), prompts, cfg, first, None, 'unedited')
    assert [r['tokens'] for r in again] == [r['tokens'] for r in plain] != [r['tokens'] for r in greedy]
    saved, m = reads(model.config.hidden_size)
    saved['gate'], saved['up'], saved['down'] = lb2.four_neurons(saved, m, lb2.end_write(saved['q'], saved['rms']), 12.0, 16.0)
    saved['down'] = torch.zeros_like(saved['down'])                 # neurons that write nothing
    rows, acts = lb2.answers_of(model, slim, block, Tok(), prompts, cfg, first, saved, 'breaker2', sampled=True)
    assert [r['tokens'] for r in rows] == [r['tokens'] for r in plain] and acts.shape[0] == 2


def test_dense_answers_come_from_the_model_with_the_router_off(tmp_path):
    from test_slim_batch import ids
    model, slim, block = tiny(tmp_path)
    cfg = {'state': 'fp32', 'max_new_tokens': 9, 'compact': None, 'seed': 0}
    prompts = [{'id': f'p{k}', 'answer': 1.0, 'ids': ids(k, length=6 + k)} for k in range(3)]
    rows = lb2.dense_answers(model, slim, Tok(), prompts, cfg)
    assert [r['arm'] for r in rows] == ['dense'] * 3 and not slim.on
    assert [r['tokens'] for r in lb2.dense_answers(model, slim, Tok(), prompts, cfg, batch=2)] == [r['tokens'] for r in rows]
