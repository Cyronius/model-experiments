import copy
import math
import json

import numpy as np
import pytest
import torch
from transformers import Qwen3_5TextConfig, Qwen3_5ForCausalLM

from accounting import count, target_config
from engine import (counted, evaluate, prepare_student, profile_costs, step_cost,
                    stream, training_loss)
from pruning import build_student, slice_parameter
from schedules import STRATEGIES, Schedule, hierarchy, windows


@pytest.fixture(scope='module')
def models():
    torch.set_num_threads(1)
    torch.manual_seed(17)
    cfg = Qwen3_5TextConfig(vocab_size=64, hidden_size=32, intermediate_size=128,
                           num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1,
                           head_dim=16, linear_num_key_heads=2, linear_num_value_heads=2,
                           linear_key_head_dim=8, linear_value_head_dim=8,
                           rope_parameters={'rope_type': 'default', 'rope_theta': 10000.,
                                            'partial_rotary_factor': 0.5, 'mrope_section': [2, 1, 1]},
                           tie_word_embeddings=True)
    teacher = Qwen3_5ForCausalLM(cfg).eval().requires_grad_(False)
    target = cfg.to_dict() | dict(hidden_size=16, intermediate_size=64)
    state = teacher.state_dict()
    student, manifest = build_student(cfg.to_dict(), state.__getitem__, 16, count(target)['total'])
    return teacher, student, manifest


def test_accounting_and_pruning(models, tmp_path):
    teacher, student, manifest = models
    assert count(student.config.to_dict())['total'] == sum(p.numel() for p in student.parameters())
    assert student.lm_head.weight is student.model.embed_tokens.weight
    r = torch.tensor(manifest['residual_indices'])
    keep = [torch.tensor(x) for x in manifest['ffn_indices']]
    for name, p in student.state_dict().items():
        assert torch.equal(p, slice_parameter(name, teacher.state_dict()[name], r, keep))
    student.save_pretrained(tmp_path)
    loaded = Qwen3_5ForCausalLM.from_pretrained(tmp_path)
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    with torch.no_grad():
        torch.testing.assert_close(loaded(ids).logits, student(ids).logits)
    assert loaded.lm_head.weight is loaded.model.embed_tokens.weight


def test_native_stream_matches_hf(models):
    teacher, _, _ = models
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    with torch.no_grad():
        h = stream(teacher, ids)[-1]
        torch.testing.assert_close(teacher.lm_head(teacher.model.norm(h)), teacher(ids).logits)


def test_infeasible_ffn_only():
    from accounting import ROOT
    cfg = json.loads((ROOT / 'metadata/Qwen3.5-2B.json').read_text())['config']
    with pytest.raises(ValueError, match='impossible'):
        target_config(cfg, 800_000_000)


def test_schedules_cover_edges_and_stagger():
    assert hierarchy(24) == [1, 2, 4, 8, 16, 24]
    assert windows(5, 2) == [(0, 2), (2, 4), (4, 5)]
    assert (1, 3) in windows(5, 2, True)
    for strategy in STRATEGIES:
        s = Schedule(strategy, 24, rng=np.random.default_rng(0))
        for progress in (0, .1, .5, .9999, 1):
            a, b = s.choose(progress)
            assert 0 <= a < b <= 24
    assert Schedule('hierarchical', 24).choose(.999) == (0, 24)
    assert Schedule('joint', 24).choose(0) == (0, 24)
    # random: reproducible from its rng, every width equally likely, always a valid window.
    draws = [Schedule('random', 24, rng=np.random.default_rng(5)).choose(0) for _ in range(2)]
    assert draws[0] == draws[1]
    r = Schedule('random', 24, rng=np.random.default_rng(0))
    picks = [r.choose(i / 4000) for i in range(4000)]
    assert all(0 <= a < b <= 24 for a, b in picks)
    widths = np.bincount([b - a for a, b in picks], minlength=25)[1:]
    assert widths.min() > 100 and len(set(picks)) > 200
    # random_dwell: keeps its window until the loss gains, stalls or hits the cap.
    d = Schedule('random_dwell', 24, rng=np.random.default_rng(1))
    first = d.choose(0)
    for _ in range(5):
        d.feedback(1.0, 0)
    assert d.choose(0) == first
    for _ in range(3):
        d.feedback(0.5, 0)  # EMA 1.0 -> 0.95 -> ... reaches the 5% gain on the first step
    assert d.observations[-1]['reason'] == 'gain' and d.flip
    d.choose(0)
    for _ in range(5 + 15):
        d.feedback(1.0, 0)
    assert d.observations[-1]['reason'] == 'stall'
    # coarse_to_fine: hierarchical reversed, whole network first and single layers last.
    c = Schedule('coarse_to_fine', 24)
    assert c.choose(0) == (0, 24) and c.choose(1 / 6 + 1e-9) == (0, 16) and c.choose(.999) == (23, 24)
    # global_fixed: the whole network for the first of 7 equal shares, then blocks bottom to top.
    g = Schedule('global_fixed', 24)
    assert g.choose(0) == (0, 24) and g.choose(1 / 7 + 1e-9) == (0, 4) and g.choose(.9999) == (20, 24)
    # alternating: the last 20% of each of 24 slices is global; otherwise the fixed sweep.
    alt = Schedule('alternating', 24)
    assert alt.choose(0) == (0, 4) and alt.choose(.9 / 24) == (0, 24) and alt.choose(.5) == (12, 16)
    shares = [alt.choose(i / 4800) == (0, 24) for i in range(4800)]
    assert abs(sum(shares) / len(shares) - .2) < .01
    f = Schedule('fixed', 24)
    f.choose(1 / 12)
    assert abs(f.within - .5) < 1e-9


def test_adaptive_controllers_advance_through_blocks():
    s = Schedule('adaptive_error', 8, fixed_width=2)
    # Added error per layer: .1 .1 .0 .6 .0 .0 .1 .0 -> block (2, 4) adds the most.
    s.observe([.1, .2, .2, .8, .8, .8, .9, .9], .1)
    assert s.active == (2, 4)
    s.observe([.1, .2, .2, .3, .3, .3, 1.2, 1.2], .2)
    assert s.active == (6, 8)
    s.observe([.5] * 8, .3)
    assert s.active == (0, 2)
    # Cap: block (0, 2) always adds the most error, but no block may take more than
    # 2 x 1/4 of the budget in total; then the next-largest eligible block trains.
    c = Schedule('adaptive_error', 8, fixed_width=2)
    stuck = [.5, .5, .6, .6, .6, .6, .6, .6]
    picks = []
    for f in [i / 20 for i in range(20)]:
        c.observe(stuck, f)
        picks.append(c.active)
    assert picks[0] == (0, 2) and picks[-1] == (2, 4)
    assert c.block_spent[0] <= 0.5 + 0.05 and c.observations[-1]['reason']['capped'] == [0]
    p = Schedule('adaptive_plateau', 8, fixed_width=2)
    assert p.choose(0) == (0, 2)
    p.observe([.5] * 8, .0)
    p.observe([.4] * 8, .05)
    assert p.active == (0, 2)
    p.observe([.399] * 8, .1)  # 0.25% better: plateau
    assert p.active == (2, 4) and p.last_error is None
    p.observe([.3] * 8, .2)
    p.observe([.2] * 8, .3)
    p.observe([.1] * 8, .61)  # still improving, but held 2 x 1/4 of the budget
    assert p.active == (4, 6)
    for f in (.7, .8, .9, .95):
        p.observe([.1] * 8, f)
    assert p.active[0] in (0, 2, 4, 6)
    assert all(o['new'] != o['old'] for o in p.observations if o['reason']['moved'])


@pytest.mark.parametrize('amp', [False, True])
def test_gradient_locality_and_flop_composition(models, amp):
    teacher, damaged, manifest = models
    student = copy.deepcopy(damaged)
    prepare_student(student)
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    r = torch.tensor(manifest['residual_indices'])
    profile = profile_costs(student, teacher, ids, amp=amp)
    for window in [(0, 1), (1, 3), (0, 4), (3, 4)]:
        student.zero_grad(set_to_none=True)

        def step():
            loss = training_loss(student, teacher, ids, r, window, amp=amp)
            loss.backward()
            return loss

        loss, measured = counted(step)
        assert torch.isfinite(loss)
        assert measured == step_cost(profile, window)
        for i, block in enumerate(student.model.layers):
            has_grad = any(p.grad is not None for p in block.parameters())
            assert has_grad == (window[0] <= i < window[1])
        assert (student.model.embed_tokens.weight.grad is not None) == (window[0] == 0 or window[1] == 4)
        assert (student.model.norm.weight.grad is not None) == (window[1] == 4)
    assert all(p.grad is None for p in teacher.parameters())


def test_teacher_fed_windows_skip_the_student_prefix(models):
    """teacher_input: the window reads the teacher stream, so the student prefix neither runs
    (FLOPs match step_cost(..., True)) nor matters (changing it leaves the loss unchanged)."""
    teacher, damaged, manifest = models
    student = copy.deepcopy(damaged)
    prepare_student(student)
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    r = torch.tensor(manifest['residual_indices'])
    profile = profile_costs(student, teacher, ids, amp=False)
    window = (2, 4)

    def step():
        student.zero_grad(set_to_none=True)
        loss = training_loss(student, teacher, ids, r, window, amp=False, teacher_input=True)
        loss.backward()
        return loss

    loss, measured = counted(step)
    assert measured == step_cost(profile, window, True) < step_cost(profile, window)
    assert all(p.grad is None for i in (0, 1) for p in student.model.layers[i].parameters())
    with torch.no_grad():
        for p in student.model.layers[0].parameters():
            p.add_(0.5)
    again = training_loss(student, teacher, ids, r, window, amp=False, teacher_input=True)
    assert torch.equal(loss.detach(), again.detach())
    assert not torch.equal(again.detach(), training_loss(student, teacher, ids, r, window, amp=False).detach())
    # A window at the bottom reads the student's own embedding either way.
    assert step_cost(profile, (0, 2), True) == step_cost(profile, (0, 2))


@pytest.mark.parametrize('strategy', STRATEGIES)
def test_every_strategy_can_optimize_and_evaluate(models, strategy):
    teacher, damaged, manifest = models
    student = copy.deepcopy(damaged)
    prepare_student(student)
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    r = torch.tensor(manifest['residual_indices'])
    schedule = Schedule(strategy, 4, fixed_width=2, rng=np.random.default_rng(0))
    metrics = evaluate(student, teacher, [ids], r, amp=False)
    if schedule.adaptive:
        schedule.observe(metrics['discrepancy_by_depth'], .5)
    params = [p for p in student.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-3, foreach=False)
    window = schedule.choose(.6)
    before = [p.detach().clone() for p in student.model.layers[window[0]].parameters()]
    loss = training_loss(student, teacher, ids, r, window, amp=False)
    loss.backward()
    optimizer.step()
    assert any(not torch.equal(a, b) for a, b in zip(before, student.model.layers[window[0]].parameters()))
    after = evaluate(student, teacher, [ids], r, amp=False)
    assert all(torch.isfinite(torch.tensor(after[k])) for k in ('ce', 'teacher_kl'))


def test_runner_budget_and_final_artifact(models, tmp_path, monkeypatch):
    import numpy as np
    import run
    from pruning import sha256

    teacher, damaged, manifest = models
    teacher_dir, damaged_dir, data_dir = [tmp_path / x for x in ('teacher', 'damaged', 'data')]
    teacher.save_pretrained(teacher_dir)
    damaged.save_pretrained(damaged_dir)
    record = copy.deepcopy(manifest)
    record['source_sha256'] = {p.name: sha256(p) for p in teacher_dir.glob('*.safetensors')}
    record['artifact_sha256'] = {p.name: sha256(p) for p in damaged_dir.iterdir() if p.is_file()}
    (damaged_dir / 'pruning.json').write_text(json.dumps(record))
    data_dir.mkdir()
    # Distinct toy arrays only test the harness, never model quality.
    arrays = {name: np.arange(1 + i * 8, 9 + i * 8, dtype=np.int32)[None]
              for i, name in enumerate(('train', 'calibration', 'controller', 'validation', 'test'))}
    np.savez(data_dir / 'tokens.npz', **arrays)
    (data_dir / 'manifest.json').write_text(json.dumps({'files_sha256': {'tokens.npz': sha256(data_dir / 'tokens.npz')}}))
    out = tmp_path / 'run'
    monkeypatch.setattr('sys.argv', ['run.py', '--teacher', str(teacher_dir), '--damaged', str(damaged_dir),
                                   '--data', str(data_dir), '--output', str(out), '--strategy', 'joint',
                                   '--device', 'cpu', '--budget-flops', '30000000', '--target-ce', '10',
                                   '--eval-points', '2', '--eval-sequences', '1', '--max-steps', '4'])
    run.main()
    result = json.loads((out / 'summary.json').read_text())
    assert 0 < result['training_flops'] <= 30_000_000
    assert result['unspent_training_flops'] == 30_000_000 - result['training_flops']
    assert result['total_accounted_flops'] > result['training_flops']
    assert result['reached_target']
    loaded = run.load_model(out / 'final', 'cpu', torch.float32)
    assert sum(p.numel() for p in loaded.parameters()) == record['parameter_counts']['total']


def test_corpus_splits_contamination_and_parsing():
    import prepare_data as pd
    import numpy as np
    splits = [pd.bucket('wiki', f'item-{i}') for i in range(5000)]
    assert splits == [pd.bucket('wiki', f'item-{i}') for i in range(5000)]
    share = {s: splits.count(s) / len(splits) for s in set(splits)}
    assert set(share) == {'train', 'controller', 'test_in_domain', 'calibration'}
    assert .9 < share['train'] < .96 and .02 < share['controller'] < .04
    assert {pd.bucket('biomed', f'd{i}') for i in range(100)} == {'validation', 'test'}
    dirty = pd.Contamination(['Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May.',
                              'What is the boiling point of water at sea level?', 'Too short a question'])
    assert dirty.hit('Problem: natalia sold clips to 48 of her friends in April and then she sold half')
    assert dirty.hit('Q: what is the boiling point of water at sea level? A: 100 C')
    assert not dirty.hit('Natalia sold clips to 48 of her friends.')
    assert not dirty.hit('Too short a question')
    assert pd.allocate({'a': 2, 'b': 10, 'c': 10}, 12) == {'a': 2, 'b': 5, 'c': 5}
    assert pd.rows_of(list(range(10)), 4).tolist() == [[0, 1, 2, 3], [4, 5, 6, 7]]
    text = '### Instruction\nExplain recursion briefly.\n### Reasoning\nDefine it, give an example.\n### Response\nA function that calls itself on a smaller input.'
    assert pd.instruction_pair(text) == ('Explain recursion briefly.', 'Define it, give an example.',
                                         'A function that calls itself on a smaller input.')
    assert pd.instruction_pair(text.replace('### Reasoning', '### Thoughts')) is None
    assert pd.looping('a b c d e f g h ' * 4) and not pd.looping('one two three four five six seven eight nine')
    assert pd.last_boxed('so \\boxed{\\frac{1}{2}} done') == '\\frac{1}{2}'


def test_templated_turns_close_their_thinking():
    from pathlib import Path
    import prepare_data as pd
    teacher = Path(__file__).resolve().parents[1] / 'models/manifold-audit/Qwen3.5-2B'
    if not (teacher / 'tokenizer.json').exists():
        pytest.skip('local teacher tokenizer not present')
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(teacher, local_files_only=True)
    text = tok.apply_chat_template(pd.turn('What is 2+3?', 'Add 2 and 3 to get 5.', 'The answer is 5.'), tokenize=False)
    assert '<think>\nAdd 2 and 3 to get 5.\n</think>\n\nThe answer is 5.<|im_end|>' in text


def test_canary_and_domain_metrics(models, tmp_path):
    import numpy as np
    import canary
    from evaluate import corpus_metrics, gsm8k_answer
    from pruning import sha256
    teacher, damaged, _ = models
    teacher_dir, data_dir = tmp_path / 'teacher', tmp_path / 'data'
    teacher.save_pretrained(teacher_dir)
    data_dir.mkdir()
    rows = np.arange(1, 33, dtype=np.int32).reshape(4, 8) % 60 + 1
    np.savez(data_dir / 'tokens.npz', calibration=rows)
    (data_dir / 'manifest.json').write_text('{}')
    reference = tmp_path / 'cpu.json'
    reference.write_text(json.dumps(dict(canary.identity(teacher_dir, data_dir),
                                         ce=canary.sequence_ce(teacher, rows, 'cpu', False))))
    gpu = tmp_path / 'gpu.json'
    first = canary.check(teacher, teacher_dir, data_dir, reference, 'cpu', False, gpu)
    assert first['ok'] and first['wrote_gpu_reference'] and gpu.exists()
    assert canary.check(teacher, teacher_dir, data_dir, reference, 'cpu', False, gpu)['gpu_abs_diff'] == [0.0] * 4
    bad = json.loads(reference.read_text())
    bad['ce'][2] += 0.5
    reference.write_text(json.dumps(bad))
    with pytest.raises(canary.CanaryFailure):
        canary.check(teacher, teacher_dir, data_dir, reference, 'cpu', False)
    metrics = corpus_metrics(damaged, teacher, rows, np.array([0, 1, 1, 0]), ['a', 'b'], 'cpu', False)
    assert set(metrics['domains']) == {'a', 'b'} and metrics['tokens'] == 28
    assert abs(metrics['ce'] - (metrics['domains']['a']['ce'] + metrics['domains']['b']['ce']) / 2) < 1e-9
    assert gsm8k_answer('so the total is \\boxed{1,234}.') == '1234'
    assert gsm8k_answer('She pays 18 dollars, then 20.') == '20'
    assert gsm8k_answer('\\boxed{' + '9' * 400 + '}') == '9' * 400
    assert gsm8k_answer('no number') is None


def test_activation_selection_and_norm_scale(models):
    from pruning import activation_stats, plain_ce
    teacher, _, _ = models
    cfg = teacher.config.to_dict()
    target = count(cfg | dict(hidden_size=16, intermediate_size=64))['total']
    rows = [torch.arange(1, 17) % 60 + 1, torch.arange(5, 21) % 60 + 1]
    stats = activation_stats(teacher, [r.numpy() for r in rows])
    assert stats['tokens'] == 32 and stats['ffn_activation_energy'].shape == (4, 128)
    assert len(stats['norm_names']) == 2 * 4 + 1
    state = teacher.state_dict()
    plain, record = build_student(cfg, state.__getitem__, 16, target, stats, False)
    scaled, scaled_record = build_student(cfg, state.__getitem__, 16, target, stats, True)
    r = torch.tensor(record['residual_indices'])
    assert torch.equal(r, torch.sort(torch.argsort(stats['norm_output_energy'], descending=True, stable=True)[:16]).values)
    assert not record['rescaling'] and scaled_record['rescaling'] and len(scaled_record['norm_scales']) == 9
    energy = stats['norm_input_energy'][-1]
    expected = float(((energy[r].sum() / 16) / (energy.sum() / 32)).sqrt())
    assert abs(scaled_record['norm_scales']['model.norm.weight'] - expected) < 1e-12
    torch.testing.assert_close((1 + scaled.model.norm.weight.double()),
                               (1 + plain.model.norm.weight.double()) * expected, rtol=1e-6, atol=1e-6)
    for name, p in plain.state_dict().items():
        if not name.endswith('layernorm.weight') and name != 'model.norm.weight':
            assert torch.equal(p, scaled.state_dict()[name])
    assert math.isfinite(plain_ce(scaled, [r.numpy() for r in rows]))


def test_adaptive_decisions_count_against_budget(models, tmp_path, monkeypatch):
    import numpy as np
    import run
    from pruning import sha256
    teacher, damaged, manifest = models
    teacher_dir, damaged_dir, data_dir = [tmp_path / x for x in ('teacher', 'damaged', 'data')]
    teacher.save_pretrained(teacher_dir)
    damaged.save_pretrained(damaged_dir)
    record = copy.deepcopy(manifest)
    record['source_sha256'] = {p.name: sha256(p) for p in teacher_dir.glob('*.safetensors')}
    record['artifact_sha256'] = {p.name: sha256(p) for p in damaged_dir.iterdir() if p.is_file()}
    (damaged_dir / 'pruning.json').write_text(json.dumps(record))
    data_dir.mkdir()
    arrays = {name: (np.arange(1, 9 * 3 + 1, dtype=np.int32).reshape(3, 9)[:, :8] + 8 * i) % 60 + 1
              for i, name in enumerate(('train', 'calibration', 'controller', 'validation', 'test'))}
    np.savez(data_dir / 'tokens.npz', **arrays)
    (data_dir / 'manifest.json').write_text(json.dumps({'files_sha256': {'tokens.npz': sha256(data_dir / 'tokens.npz')}}))
    out = tmp_path / 'run'
    monkeypatch.setattr('sys.argv', ['run.py', '--teacher', str(teacher_dir), '--damaged', str(damaged_dir),
                                   '--data', str(data_dir), '--output', str(out), '--strategy', 'adaptive_plateau',
                                   '--device', 'cpu', '--budget-flops', '60000000', '--target-ce', '1',
                                   '--eval-points', '2', '--eval-sequences', '1', '--controller-points', '6',
                                   '--controller-sequences', '2', '--warmup-updates', '4'])
    run.main()
    result = json.loads((out / 'summary.json').read_text())
    decisions = json.loads((out / 'controller-decisions.json').read_text())
    assert result['decision_flops'] > 0 and len(decisions) >= 2
    assert result['training_flops'] + result['decision_flops'] <= 60_000_000
    assert result['unspent_training_flops'] == 60_000_000 - result['training_flops'] - result['decision_flops']
    assert sum(result['parameter_group_updates'].values()) > 0
    assert json.loads((out / 'provenance.json').read_text())['controller_rows'] == [0, 2]


def test_report_gap_ladder():
    from report import gap_closed, ladder
    history = [dict(step=i, tokens=i * 10, training_flops=i * 100, elapsed_seconds=i,
                    validation=dict(ce=ce)) for i, ce in enumerate([9.0, 7.0, 6.0, 5.5, 4.0, 3.0, 2.9])]
    assert gap_closed(9.0, 5.0, 3.0) == 2 / 3
    rungs = ladder(history, 3.0)
    # 25% rung is 7.5: evaluations 1 and 2 are both below, confirmed at 2.
    assert rungs['gap25']['step'] == 2 and rungs['gap50']['step'] == 3
    assert rungs['gap75']['step'] == 5 and rungs['gap100']['step'] == 6
    assert ladder(history[:5], 3.0)['gap100'] is None


def test_branch_resume_matches_uninterrupted_run(models, tmp_path, monkeypatch):
    """A trunk checkpoint plus a branch reproduces an uninterrupted run exactly (CPU, joint)."""
    import numpy as np
    from safetensors.torch import load_file
    import run
    from pruning import sha256
    teacher, damaged, manifest = models
    teacher_dir, damaged_dir, data_dir = [tmp_path / x for x in ('teacher', 'damaged', 'data')]
    teacher.save_pretrained(teacher_dir)
    damaged.save_pretrained(damaged_dir)
    record = copy.deepcopy(manifest)
    record['source_sha256'] = {p.name: sha256(p) for p in teacher_dir.glob('*.safetensors')}
    record['artifact_sha256'] = {p.name: sha256(p) for p in damaged_dir.iterdir() if p.is_file()}
    (damaged_dir / 'pruning.json').write_text(json.dumps(record))
    data_dir.mkdir()
    arrays = {name: (np.arange(1, 9 * 3 + 1, dtype=np.int32).reshape(3, 9)[:, :8] + 8 * i) % 60 + 1
              for i, name in enumerate(('train', 'calibration', 'controller', 'validation', 'test'))}
    np.savez(data_dir / 'tokens.npz', **arrays)
    (data_dir / 'manifest.json').write_text(json.dumps({'files_sha256': {'tokens.npz': sha256(data_dir / 'tokens.npz')}}))

    def launch(out, budget, *extra):
        monkeypatch.setattr('sys.argv', ['run.py', '--teacher', str(teacher_dir), '--damaged', str(damaged_dir),
                                         '--data', str(data_dir), '--output', str(out), '--strategy', 'joint',
                                         '--device', 'cpu', '--budget-flops', str(budget), '--target-ce', '1',
                                         '--eval-points', '2', '--eval-sequences', '1', '--warmup-updates', '3',
                                         *extra])
        run.main()

    probe = tmp_path / 'probe'
    launch(probe, 10 ** 12, '--max-steps', '1')
    cost = json.loads((probe / 'steps.jsonl').read_text().splitlines()[0])['step_flops']
    # Four steps over three training rows, so the data order reshuffles inside the branch.
    full, trunk, branch = tmp_path / 'full', tmp_path / 'trunk', tmp_path / 'branch'
    launch(full, 4 * cost)
    # A fraction this negative is met at the first evaluation after step 0 (here after step 2).
    launch(trunk, 4 * cost, '--checkpoint-gaps', '-100')
    point = trunk / 'checkpoints' / 'gap-10000'
    meta = json.loads((point / 'checkpoint.json').read_text())
    assert meta['trunk']['trunk_step'] == 2 and meta['trunk']['trunk_strategy'] == 'joint'
    launch(branch, 2 * cost, '--init-from', str(point))
    assert json.loads((branch / 'summary.json').read_text())['steps'] == 2
    a, b = [load_file(next((x / 'final').glob('*.safetensors'))) for x in (full, branch)]
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
    provenance = json.loads((branch / 'provenance.json').read_text())
    assert provenance['branch']['gap_fraction'] == -100
    curve = json.loads((branch / 'curve.json').read_text())
    assert curve[0]['validation']['ce'] == pytest.approx(meta['trunk']['checkpoint_validation_ce'])


def test_batch_sequences_accumulate(models, tmp_path, monkeypatch):
    import numpy as np
    import run
    from pruning import sha256
    teacher, damaged, manifest = models
    teacher_dir, damaged_dir, data_dir = [tmp_path / x for x in ('teacher', 'damaged', 'data')]
    teacher.save_pretrained(teacher_dir)
    damaged.save_pretrained(damaged_dir)
    record = copy.deepcopy(manifest)
    record['source_sha256'] = {p.name: sha256(p) for p in teacher_dir.glob('*.safetensors')}
    record['artifact_sha256'] = {p.name: sha256(p) for p in damaged_dir.iterdir() if p.is_file()}
    (damaged_dir / 'pruning.json').write_text(json.dumps(record))
    data_dir.mkdir()
    arrays = {name: (np.arange(1, 9 * 3 + 1, dtype=np.int32).reshape(3, 9)[:, :8] + 8 * i) % 60 + 1
              for i, name in enumerate(('train', 'calibration', 'controller', 'validation', 'test'))}
    np.savez(data_dir / 'tokens.npz', **arrays)
    (data_dir / 'manifest.json').write_text(json.dumps({'files_sha256': {'tokens.npz': sha256(data_dir / 'tokens.npz')}}))
    rows = {}
    for batch in (1, 2):
        out = tmp_path / f'b{batch}'
        monkeypatch.setattr('sys.argv', ['run.py', '--teacher', str(teacher_dir), '--damaged', str(damaged_dir),
                                         '--data', str(data_dir), '--output', str(out), '--strategy', 'joint',
                                         '--device', 'cpu', '--budget-flops', str(10 ** 12), '--target-ce', '1',
                                         '--eval-points', '1', '--eval-sequences', '1', '--max-steps', '2',
                                         '--batch-sequences', str(batch)])
        run.main()
        rows[batch] = [json.loads(x) for x in (out / 'steps.jsonl').read_text().splitlines()]
    assert len(rows[2]) == 2 and rows[2][-1]['tokens'] == 2 * 2 * 8
    assert rows[2][0]['step_flops'] == 2 * rows[1][0]['step_flops']


@pytest.mark.parametrize('inputs,strategy', [('teacher', 'layerwise'), ('ramp', 'layerwise'),
                                             ('student', 'random'), ('student', 'random_dwell')])
def test_window_inputs_run_with_exact_accounting(models, tmp_path, monkeypatch, inputs, strategy):
    """The runner's predicted-vs-observed FLOP check passes for teacher-fed, ramped and random windows."""
    import numpy as np
    import run
    from pruning import sha256
    teacher, damaged, manifest = models
    teacher_dir, damaged_dir, data_dir = [tmp_path / x for x in ('teacher', 'damaged', 'data')]
    teacher.save_pretrained(teacher_dir)
    damaged.save_pretrained(damaged_dir)
    record = copy.deepcopy(manifest)
    record['source_sha256'] = {p.name: sha256(p) for p in teacher_dir.glob('*.safetensors')}
    record['artifact_sha256'] = {p.name: sha256(p) for p in damaged_dir.iterdir() if p.is_file()}
    (damaged_dir / 'pruning.json').write_text(json.dumps(record))
    data_dir.mkdir()
    arrays = {name: (np.arange(1, 9 * 3 + 1, dtype=np.int32).reshape(3, 9)[:, :8] + 8 * i) % 60 + 1
              for i, name in enumerate(('train', 'calibration', 'controller', 'validation', 'test'))}
    np.savez(data_dir / 'tokens.npz', **arrays)
    (data_dir / 'manifest.json').write_text(json.dumps({'files_sha256': {'tokens.npz': sha256(data_dir / 'tokens.npz')}}))
    out = tmp_path / f'run-{inputs}-{strategy}'
    monkeypatch.setattr('sys.argv', ['run.py', '--teacher', str(teacher_dir), '--damaged', str(damaged_dir),
                                     '--data', str(data_dir), '--output', str(out), '--strategy', strategy,
                                     '--device', 'cpu', '--budget-flops', str(4 * 10 ** 8), '--target-ce', '1',
                                     '--eval-points', '1', '--eval-sequences', '1', '--max-steps', '60',
                                     '--window-inputs', inputs]
                        + (['--lr-decay', 'cosine'] if strategy == 'random_dwell' else []))
    run.main()
    rows = [json.loads(x) for x in (out / 'steps.jsonl').read_text().splitlines()]
    if inputs == 'student':
        assert not any(r['teacher_input'] for r in rows)
        if strategy == 'random':
            assert len({tuple(r['window']) for r in rows}) > 1
        factors = [r['lr_factor'] for r in rows]
        assert factors == sorted(factors, reverse=True) and (factors[-1] < 1) == (strategy == 'random_dwell')
        return
    fed = [r['teacher_input'] for r in rows if r['window'][0] > 0]
    assert fed and (all(fed) if inputs == 'teacher' else 0 < sum(fed) < len(fed))
    assert not any(r['teacher_input'] for r in rows if r['window'][0] == 0)
