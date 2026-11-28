import numpy as np
import torch

import loop_breaker as lb

CYCLE = {'min_period': 8, 'max_period': 700, 'agree': 0.9}


def test_cycle_finds_the_period_and_where_the_first_copy_starts():
    rng = np.random.default_rng(0)
    body = rng.integers(10, 1000, 37).tolist()
    tokens = rng.integers(1000, 2000, 150).tolist() + body * 5 + body[:11]
    assert lb.cycle(tokens, CYCLE) == (37, 150)
    assert lb.cycle(rng.integers(0, 1000, 400).tolist(), CYCLE) is None
    # A counter inside the body (one token differs each copy) is still a cycle.
    counted = sum(([c] + body for c in range(5000, 5006)), [])
    assert lb.cycle(rng.integers(1000, 2000, 60).tolist() + counted, CYCLE)[0] == 38


def test_classes_and_split():
    pos, neg = lb.classes('loop', (30, 100), 400, 20)
    assert len(pos) == 401 and pos.nonzero()[0][0] == 151 and neg.nonzero()[0][-1] == 69
    pos, neg = lb.classes('done', None, 50, 20)
    assert not pos.any() and neg.all()
    assert not np.any(lb.classes('cut', None, 50, 20))
    found = [('loop', (8, 1)), ('done', None), ('loop', (8, 1)), ('done', None), ('done', None), ('drift', None)]
    assert lb.split(found) == [True, True, False, False, True, True]


def test_the_written_neurons_compute_the_activation_and_write_v():
    torch.manual_seed(0)
    h, k, clamp, b = 32, 4.0, 4.0, 1.5
    w, c, v = torch.randn(h), torch.randn(h) / h, torch.randn(h)
    x = torch.randn(3, 7, h)
    gate, up, down = lb.neuron_weights(w, c, v, b, k, clamp)
    neurons = lb.Breaker(h, 2)
    with torch.no_grad():
        neurons.gate.weight.copy_(gate)
        neurons.up.weight.copy_(up)
        neurons.down.weight.copy_(down)
    neurons.record = []
    act = lb.activation(x @ w, x @ c, b, k, clamp)
    torch.testing.assert_close(neurons(x), act[..., None] * v, atol=1e-4, rtol=1e-4)
    f = neurons.record[0]
    torch.testing.assert_close((f[:, 0] - f[:, 1]) / clamp, act[:, -1], atol=1e-5, rtol=1e-4)
    # Far below the threshold it writes nothing, at the threshold almost nothing, half of v half a
    # unit above it and exactly v far above (with the constant at 1).
    s, u = torch.tensor([-5.0, 1.5, 2.0, 10.0]), torch.ones(4)
    a = lb.activation(s, u, b, k, clamp)
    assert abs(float(a[0])) < 1e-6 and float(a[1]) < 0.02 and abs(float(a[2]) - 0.5) < 1e-3 and abs(float(a[3]) - 1) < 1e-4


def test_the_write_raises_both_stop_logits_by_the_dose_and_the_control_none():
    torch.manual_seed(1)
    h, rms = 64, 3.0
    rows, scale = torch.randn(10, h), 1 + 0.1 * torch.randn(h)
    q = rows[:2] * scale
    v1 = lb.stop_write(q, rms)
    r = torch.randn(h)
    r = r * rms / r.pow(2).mean().sqrt()
    logits = lambda z: rows @ (scale * z / r.pow(2).mean().sqrt())     # the final norm at the unedited RMS
    torch.testing.assert_close((logits(r + 6 * v1) - logits(r))[:2], torch.full((2,), 6.0), atol=1e-3, rtol=1e-3)
    control = lb.random_write(6 * v1, q)
    torch.testing.assert_close(control.norm(), (6 * v1).norm(), atol=1e-5, rtol=1e-4)
    assert float((q @ control).abs().max()) < 1e-3


def test_the_probe_separates_the_classes_and_the_constant_is_one():
    torch.manual_seed(2)
    h = 24
    direction, offset = torch.zeros(h), torch.zeros(h)
    direction[3], offset[5] = 2.0, 4.0

    def batch(n):
        pos = torch.rand(n) < 0.3
        x = torch.randn(n, h) + offset + pos[:, None] * direction
        return x, pos, ~pos, torch.ones(n) * 2

    w, c, rms, n = lb.fit_probe([batch(4000), batch(4000)], 0.01)
    x, pos, neg, _ = batch(4000)
    assert lb.auroc(x[pos] @ w, x[neg] @ w) > 0.9 and rms == 2.0 and n['pos'] + n['neg'] == 8000
    u = x @ c
    assert abs(float(u.mean()) - 1) < 0.05 and float(u.std()) < 0.4


def test_the_screen_names_where_an_edited_answer_leaves_its_text():
    # Sixty positions, tokens [7, 8 | stops 1, 2]: the stop token 1 wins once the dose is in.
    terms = {'tokens': torch.tensor([[7, 8, 1, 2]] * 60), 'base': torch.tensor([[5.0, 1.0, 2.0, 0.0]] * 60),
             'gain': torch.tensor([[0.0, 0.0, 1.0, 1.0]] * 60), 'inside': torch.ones(60, dtype=torch.bool)}
    act = torch.zeros(60)
    assert lb.first_change(terms, act, 8) == (None, None)
    act[41:] = 1.0
    assert lb.first_change(terms, act, 8) == (41, 1) and lb.first_change(terms, act, 2) == (None, None)
    row, tokens = {'tokens': [7] * 59}, {'written': [1], 'think': 1}
    name = lambda kind, cyc=None, tokens=tokens: lb.outcome(kind, cyc, row, terms, act, 8, tokens)[0]
    assert name('loop', (10, 30)) == 'broken' and name('loop', (5, 20)) == 'late'
    assert name('loop', (10, 30), {'written': [2], 'think': 2}) == 'other'
    assert name('done') == 'collateral_repeating'
    terms['inside'][41] = False                                    # not in a repeat there
    assert name('loop', (10, 30)) == 'broken' and name('loop', (5, 47)) == 'collateral'    # within a period of the cycle, or not
    assert name('done') == 'collateral' and name('cut') == 'collateral'
    row = {'tokens': [7] * 50 + [1] + [7] * 8}                     # its own </think> comes 9 tokens later
    assert name('done') == 'benign'
    assert lb.outcome('done', None, row, terms, torch.zeros(60), 8, tokens)[0] == 'none'


def test_repeating_marks_the_positions_whose_last_tokens_occurred_before():
    tokens = [1, 2, 3, 4, 5, 9, 1, 2, 3, 4, 5, 6]
    inside = lb.repeating(tokens, 3)
    assert len(inside) == 13 and [t for t, x in enumerate(inside) if x] == [9, 10, 11]


def test_an_up_projection_of_its_own_makes_the_neurons_an_and():
    s, u = torch.tensor([10.0, 10.0, -5.0]), torch.ones(3)
    a = lb.activation(s, u, 1.5, 4.0, 4.0, up=torch.tensor([1.0, 0.0, 1.0]))
    torch.testing.assert_close(a, torch.tensor([1.0, 0.0, 0.0]), atol=1e-4, rtol=1e-4)
    h = 16
    torch.manual_seed(0)
    w, c, n, v, x = torch.randn(h), torch.randn(h) / h, torch.randn(h), torch.randn(h), torch.randn(5, 3, h)
    gate, up, down = lb.neuron_weights(w, c, v, 0.3, 4.0, 4.0, n)
    neurons = lb.Breaker(h, 2)
    with torch.no_grad():
        neurons.gate.weight.copy_(gate)
        neurons.up.weight.copy_(up)
        neurons.down.weight.copy_(down)
    torch.testing.assert_close(neurons(x), lb.activation(x @ w, x @ c, 0.3, 4.0, 4.0, x @ n)[..., None] * v, atol=1e-4, rtol=1e-4)


class Tok:
    def convert_tokens_to_ids(self, t):
        return {'</think>': 3, '<|im_end|>': 4, '<|endoftext|>': 5}[t]


def test_capture_and_the_installed_neurons_agree_with_the_screen_arithmetic(tmp_path):
    import heal_block as hb
    import heal_block_check as hbc
    from test_heal_block import block as tiny_block
    from test_slim_batch import ids
    model, slim, first = tiny_block(layers=(6, 7))
    torch.save({'layers': [6, 7], 'copies': {i: c.state_dict() for i, c in zip([6, 7], first.copies)},
                'routers': {i: slim.routers[i].state_dict() for i in (6, 7)}}, tmp_path / 'block.pt')
    block = hb.Pieces(model, slim, hb.load_pieces(model, slim, [tmp_path / 'block.pt']))
    rules = {'layer': 7}
    prompts = [{'ids': ids(0, length=12)}, {'ids': ids(1, length=9)}]
    rows = [{'tokens': ids(2, length=20)}, {'tokens': ids(3, length=15)}]
    lb.capture(model, block, Tok(), rows, prompts, rules, tmp_path)
    assert np.load(tmp_path / 'capture' / '0.npz')['r'].shape == (21, model.config.hidden_size)
    z, head = np.load(tmp_path / 'capture' / '1.npz'), torch.load(tmp_path / 'head.pt', weights_only=True)
    h = model.config.hidden_size
    assert z['x'].shape == (16, h) and z['r'].shape == (16, h) and z['top'].shape == (16, 5) and head['stops'] == [3, 4]
    index = torch.full((head['vocab'],), -1, dtype=torch.long)
    index[head['ids']] = torch.arange(len(head['ids']))
    assert (index[torch.from_numpy(z['top'].astype(np.int64))] >= 0).all()
    # The unedited logits rebuilt from the captured residual are the model's.
    r = torch.from_numpy(z['r'])
    q = head['rows'][index[torch.tensor(head['stops'])]] * head['scale']
    rebuilt = (r @ q.T) / r.pow(2).mean(-1, keepdim=True).sqrt()
    torch.testing.assert_close(rebuilt, torch.from_numpy(z['stop_logits']), atol=0.05, rtol=0.05)
    # Hand-written neurons: the model's stop logits move as the screen's arithmetic says.
    torch.manual_seed(3)
    x = torch.from_numpy(z['x'].astype(np.float32))
    w, c = torch.randn(h) / 4, torch.randn(h) / 8
    rms = float(r.pow(2).mean(-1).sqrt().median())
    saved = {'q': q, 'v': 5 * lb.stop_write(q, rms), 'clamp': 4.0}
    saved['gate'], saved['up'], saved['down'] = lb.neuron_weights(w, c, saved['v'], 0.2, 4.0, 4.0)
    act = lb.activation(x @ w, x @ c, 0.2, 4.0, 4.0)
    assert float(act.abs().max()) > 0.3
    edited = r + act[:, None] * saved['v']
    forecast = (edited @ q.T) / edited.pow(2).mean(-1, keepdim=True).sqrt()
    neurons = lb.install(block, rules, saved, 'breaker')
    seq = prompts[1]['ids'] + rows[1]['tokens']
    with hbc.Student(block):
        got = model.lm_head(lb.student_pass(model, seq)[len(prompts[1]['ids']) - 1:]).float()[:, [3, 4]]
    torch.testing.assert_close(got - torch.from_numpy(z['stop_logits']), forecast - rebuilt, atol=0.15, rtol=0.1)
    # The control writes nothing along the stop rows.
    lb.install(block, rules, saved, 'breaker-random-write')
    with hbc.Student(block):
        control = model.lm_head(lb.student_pass(model, seq)[len(prompts[1]['ids']) - 1:]).float()[:, [3, 4]]
    moved, spared = (got - torch.from_numpy(z['stop_logits'])).abs().mean(), (control - torch.from_numpy(z['stop_logits'])).abs().mean()
    assert float(spared) < 0.5 * float(moved)


def test_build_runs_from_captures_to_a_report(tmp_path):
    import argparse
    import json
    import heal_block as hb
    from test_heal_block import block as tiny_block
    model, slim, first = tiny_block(layers=(6, 7))
    torch.save({'layers': [6, 7], 'copies': {i: c.state_dict() for i, c in zip([6, 7], first.copies)},
                'routers': {i: slim.routers[i].state_dict() for i in (6, 7)}}, tmp_path / 'block.pt')
    block = hb.Pieces(model, slim, hb.load_pieces(model, slim, [tmp_path / 'block.pt']))
    rng = np.random.default_rng(0)
    text = lambda n: rng.integers(6, 96, n).tolist()
    rows, prompts = [], []
    for k in range(8):
        loop = k % 2 == 0
        tokens = text(20) + text(12) * 6 if loop else text(40)
        rows.append({'id': f'row-{k}', 'tokens': tokens, 'length': len(tokens), 'finished': not loop, 'correct': False})
        prompts.append({'ids': text(8)})
    rules = {'layer': 7, 'loop': {'window': 5, 'last': 100, 'threshold': 0.45},
             'cycle': {'min_period': 8, 'max_period': 40, 'agree': 0.9}, 'labels': {'margin': 4},
             'probe': {'ridge': 0.01}, 'neurons': {'k': 4, 'clamp': 4, 'up': 'line_end', 'tokens': ['</think>']},
             'grid': {'false_positive_rates': [0.2, 0.05], 'doses': [4, 16]}, 'screen': {'window': 5}}
    assert [k for k, _ in lb.kinds(rows, rules)] == ['loop', 'done'] * 4
    (tmp_path / 'gen_fit.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    lb.capture(model, block, Tok(), rows, prompts, rules, tmp_path)
    (tmp_path / 'tokens.json').write_text(json.dumps({'line_ends': list(range(6, 20)), 'think': 3, 'written': [3],
                                                      'prompt_last': [p['ids'][-1] for p in prompts]}))
    lb.stage_build(argparse.Namespace(threads=2), rules, tmp_path)
    report = json.loads((tmp_path / 'build.json').read_text())
    assert report['answers'] == {'loop': 4, 'drift': 0, 'done': 4, 'cut': 0} and len(report['grid']) == 4
    assert 0 <= report['auroc']['held_out'] <= 1 and 'line_end_read' in report
    if report['selected']:
        saved = torch.load(tmp_path / 'breaker.pt', weights_only=True)
        assert saved['gate'].shape == (2, 32) and saved['down'].shape == (32, 2) and 'gate' in report
        assert len(report['held_out']['answers']) == 4 and not torch.equal(saved['up'][0], saved['c'])


def test_greedy_answers_from_the_pieces_with_and_without_compaction(tmp_path):
    import heal_block as hb
    from test_heal_block import block as tiny_block
    from test_slim_batch import ids
    model, slim, first = tiny_block(layers=(6, 7))
    torch.save({'layers': [6, 7], 'copies': {i: c.state_dict() for i, c in zip([6, 7], first.copies)},
                'routers': {i: slim.routers[i].state_dict() for i in (6, 7)}}, tmp_path / 'block.pt')
    block = hb.Pieces(model, slim, hb.load_pieces(model, slim, [tmp_path / 'block.pt']))
    vocab = lb.vocab_of(model, Tok())
    prompts = [{'ids': ids(k, length=6 + k)} for k in range(3)]
    plain = lb.greedy(model, slim, block, vocab, prompts, {'state': 'fp32', 'max_new_tokens': 12, 'compact': None}, 3, 0)
    packed = lb.greedy(model, slim, block, vocab, prompts, {'state': 'fp32', 'max_new_tokens': 12, 'compact': True}, 2, 0)
    assert len(plain) == len(packed) == 3 and all(len(r['tokens']) <= 12 for r in plain + packed)
    assert sum(a['tokens'] == b['tokens'] for a, b in zip(plain, packed)) >= 2      # the same answers, but for near-ties
    assert not slim.on and model.model.layers[7] is not block.prefix[-1][1]         # the dense model is back


def test_against_pairs_an_arm_with_the_unedited_answers_and_asks_whether_the_neurons_fired():
    loop = {'window': 20, 'last': 1000, 'threshold': 0.45}
    start, turn = list(range(100, 140)), list(range(30))
    row = lambda k, tokens, finished, correct: {'id': k, 'tokens': tokens, 'length': len(tokens), 'finished': finished, 'correct': correct}
    base = [row('fixed', start + turn * 20, False, False),          # loops; the arm ends it after the first copy
            row('same', start, True, True),                         # identical in both
            row('spoiled', start, True, True),                      # the arm changes token 5 with the neurons silent
            row('still', start + turn * 20, False, False)]          # loops in both
    rows = [row('fixed', start + turn + [7, 8], True, True),
            row('same', start, True, True),
            row('spoiled', start[:5] + [9] + start[6:], True, False),
            row('still', start + turn * 20, False, False)]
    acts = np.zeros((700, 4), np.float32)
    acts[len(start) + len(turn) - 3, 0] = 0.9                       # fired three tokens before 'fixed' left its text
    pair = lb.against(base, rows, acts, loop)
    assert (pair['loops_fixed'], pair['loops_new'], pair['right_gained'], pair['right_lost']) == (1, 0, 1, 1)
    assert pair['p_fewer_loops'] == 0.5 and pair['unedited']['loops'] == 2
    answers = {a['id']: a for a in pair['answers']}
    assert answers['fixed'] == {'id': 'fixed', 'left_at': 70, 'fired': True, 'was_loop': True, 'is_loop': False}
    assert answers['same']['left_at'] is None and answers['still']['left_at'] is None
    assert answers['spoiled']['left_at'] == 5 and not answers['spoiled']['fired']
    assert (pair['fixed_after_firing'], pair['changed_without_firing']) == (1, 1)
    assert 'answers' not in lb.against(base, rows, None, loop)
