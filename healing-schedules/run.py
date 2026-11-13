"""One independent healing arm; restart from the verified damaged checkpoint.

Run under Dibs. This script never provisions compute or downloads checkpoints.

Follow-up (branching) experiment: a trunk run saves a checkpoint the first time validation
closes each fraction in --checkpoint-gaps; a branch run (--init-from) restarts from one, with
the trunk's weights, Adam state, per-group update counts and data position, under its own
budget and a fresh schedule. Branches from one checkpoint compare strategies phase by phase.
"""
import argparse
import json
import math
import platform
import resource
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import transformers

from accounting import architecture
from canary import CanaryFailure, check as canary_check
from engine import (counted, evaluate, parameter_groups, prepare_student, profile_costs, step_cost,
                    stream_errors, training_loss)
from pruning import sha256, verify_artifact
from schedules import STRATEGIES, Schedule


def load_model(folder, device, dtype):
    config = json.loads((Path(folder) / 'config.json').read_text())
    cfg, cls = architecture(config)
    model, info = cls.from_pretrained(folder, config=cfg, dtype=dtype,
                                      attn_implementation='eager', output_loading_info=True)
    if info.get('missing_keys') or info.get('mismatched_keys') or info.get('error_msgs'):
        raise ValueError(f'Incomplete model load: {info}')
    # Vision and MTP branches are intentionally excluded from this text-only study.
    extra = [k for k in info.get('unexpected_keys', []) if not
             (k.startswith(('model.visual.', 'visual.', 'mtp.')) or '.mtp.' in k)]
    if extra:
        raise ValueError(f'Unexpected text weights: {extra}')
    return model.to(device).eval().requires_grad_(False)


def save_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def batches(array, device, limit=None):
    for row in array[:limit]:
        yield torch.tensor(row[None], device=device, dtype=torch.long)


def synchronize(device):
    if str(device).startswith('cuda'):
        torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--teacher', type=Path, required=True)
    ap.add_argument('--damaged', type=Path, required=True)
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--strategy', choices=STRATEGIES, required=True)
    ap.add_argument('--seed', type=int, default=17)
    ap.add_argument('--budget-flops', type=int, required=True)
    ap.add_argument('--target-ce', type=float, required=True,
                    help='Freeze from baseline validation before any healing; never use test CE')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--lr-decay', choices=('none', 'cosine'), default='none',
                    help='cosine: multiply the learning rate by (1 + cos(pi * budget fraction spent)) / 2')
    ap.add_argument('--batch-sequences', type=int, default=1,
                    help='Sequences per optimizer update (gradients accumulated; FLOPs scale with it)')
    ap.add_argument('--warmup-updates', type=int, default=32,
                    help='Linear warmup over each parameter group\'s first N updates (groups join at different times)')
    ap.add_argument('--controller-points', type=int, default=24, help='Adaptive decisions per budget')
    ap.add_argument('--controller-sequences', type=int, default=3,
                    help='Controller rows per decision, evenly spaced over the split; their FLOPs count against the budget')
    ap.add_argument('--eval-points', type=int, default=10)
    ap.add_argument('--eval-sequences', type=int, default=16)
    ap.add_argument('--max-steps', type=int, default=100000)
    ap.add_argument('--max-seconds', type=int, default=3600)
    ap.add_argument('--canary', type=Path, help='CPU FP32 canary reference (required on a GPU)')
    ap.add_argument('--canary-gpu', type=Path, help='GPU canary reference; written on the first pass')
    ap.add_argument('--checkpoint-gaps', type=lambda v: [float(x) for x in v.split(',')], default=[],
                    help='Comma list of gap fractions (e.g. 0.25,0.5,0.75): save a branch checkpoint at the first '
                         'evaluation after step 0 whose validation closes at least that fraction')
    ap.add_argument('--data-offset', type=int, default=0,
                    help='Skip this many rows of the seed\'s training order (continuing an arm from its final model on unseen rows)')
    ap.add_argument('--window-inputs', choices=('student', 'teacher', 'ramp'), default='student',
                    help='What a window reads below its first layer: the student prefix (pilot), the teacher stream '
                         '(follow-up 3), or ramp: teacher with probability 1 - (position inside the window\'s own '
                         'FLOP segment), so each block starts teacher-fed and ends on its own prefix')
    ap.add_argument('--init-from', type=Path, help='Branch checkpoint directory (a trunk run\'s checkpoints/gapNN)')
    args = ap.parse_args()
    if str(args.device).startswith('cuda') and args.canary is None:
        raise ValueError('GPU runs need --canary (the local GPU can compute wrong values silently)')
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.data_offset < 0 or (args.data_offset and args.init_from is not None):
        raise ValueError('--data-offset must be >= 0 and cannot be combined with --init-from')
    if args.budget_flops <= 0 or args.eval_points < 1 or args.target_ce <= 0 or args.warmup_updates < 0 \
            or args.controller_points < 1 or args.controller_sequences < 1 or args.batch_sequences < 1:
        raise ValueError('Require positive budget, evaluation count, and target CE')
    manifest = verify_artifact(args.damaged)
    data_meta = json.loads((args.data / 'manifest.json').read_text())
    for name, digest in data_meta['files_sha256'].items():
        if sha256(args.data / name) != digest:
            raise ValueError(f'Data changed: {name}')
    # Require the very same teacher bytes used for pruning.
    for name, digest in manifest['source_sha256'].items():
        if sha256(args.teacher / name) != digest:
            raise ValueError(f'Teacher changed: {name}')
    arrays = np.load(args.data / 'tokens.npz', allow_pickle=False)
    train = arrays['train']
    shape = train.shape[1:]
    if len(train) < 1 or any(arrays[k].shape[1:] != shape or len(arrays[k]) < 1
                             for k in ('calibration', 'controller', 'validation', 'test')):
        raise ValueError('All splits need nonempty equal-length sequences')
    args.output.mkdir(parents=True)
    executed = args.output / 'executed-source'
    executed.mkdir()
    for path in Path(__file__).parent.glob('*.py'):
        shutil.copy2(path, executed / path.name)
    shutil.copy2(Path(__file__).parent / 'protocol.json', executed / 'protocol.json')
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    if str(args.device).startswith('cuda'):
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats()
    amp = str(args.device).startswith('cuda')
    dtype = torch.bfloat16 if amp else torch.float32
    teacher = load_model(args.teacher, args.device, dtype)
    canary = {}

    def run_canary(when):
        if args.canary is not None:
            canary[when] = canary_check(teacher, args.teacher, args.data, args.canary, args.device, amp, args.canary_gpu)

    try:
        run_canary('start')
    except CanaryFailure as failure:
        save_json(args.output / 'canary-failed.json', json.loads(str(failure)))
        raise SystemExit(3)
    student = load_model(args.damaged, args.device, dtype)
    prepare_student(student)
    indices = torch.tensor(manifest['residual_indices'], device=args.device)
    groups = parameter_groups(student)
    params = [p for _, group in groups for p in group]
    optimizer = torch.optim.AdamW([dict(params=group, name=name) for name, group in groups], lr=args.lr,
                                  betas=(.9, .999), eps=1e-8, weight_decay=0., foreach=False)
    updates = {name: 0 for name, _ in groups}
    branch = None
    if args.init_from is not None:
        meta = json.loads((args.init_from / 'checkpoint.json').read_text())
        if meta['damaged_manifest_sha256'] != sha256(args.damaged / 'pruning.json') or \
                meta['data_manifest_sha256'] != sha256(args.data / 'manifest.json'):
            raise ValueError('Branch checkpoint was made from another damaged model or dataset')
        if meta['target_ce'] != args.target_ce:
            raise ValueError('Branch checkpoint was made against another target')
        state = torch.load(args.init_from / 'state.pt', map_location=args.device, weights_only=False)
        if sha256(args.init_from / 'state.pt') != meta['state_sha256']:
            raise ValueError('Branch checkpoint state does not match its record')
        student.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        updates = dict(state['updates'])
        branch = dict(checkpoint=str(args.init_from), checkpoint_sha256=meta['state_sha256'], **meta['trunk'])
    picks = np.unique(np.linspace(0, len(arrays['controller']) - 1, args.controller_sequences).round().astype(int))
    decision_rows = arrays['controller'][picks]
    window_rng = np.random.default_rng([args.seed, 2])  # random-strategy window draws
    schedule = Schedule(args.strategy, student.config.num_hidden_layers, rng=window_rng)
    profile = profile_costs(student, teacher, next(batches(arrays['calibration'], args.device)), amp)
    save_json(args.output / 'profile.json', profile)
    provenance = dict(arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                      damaged_manifest_sha256=sha256(args.damaged / 'pruning.json'),
                      data_manifest_sha256=sha256(args.data / 'manifest.json'),
                      source_sha256={p.name: sha256(p) for p in Path(__file__).parent.glob('*.py')},
                      python=platform.python_version(), torch=torch.__version__,
                      transformers=transformers.__version__, device=str(args.device),
                      device_name=torch.cuda.get_device_name() if amp else platform.processor(),
                      cuda=torch.version.cuda, hip=torch.version.hip,
                      precision='FP32 trainable weights/Adam states; BF16 autocast' if amp else 'FP32',
                      controller_rows=picks.tolist(), branch=branch,
                      profile=profile, target_ce=args.target_ce, canary_start=canary.get('start'),
                      timing_isolation='Use Dibs --measure for timing claims; ordinary runs are observational')
    save_json(args.output / 'provenance.json', provenance)
    consumed = eval_flops = decision_flops = steps = tokens = 0
    history, hits = [], []
    rng = np.random.default_rng(args.seed)
    input_rng = np.random.default_rng([args.seed, 1])  # --window-inputs ramp draws; separate from data order
    order = rng.permutation(len(train))
    cursor = 0  # rows drawn so far, counting the trunk's when branching
    # Replay the loop's reshuffles: it reshuffles before drawing row k*len(train), k >= 1.
    for _ in range(max(0, args.data_offset - 1) // len(train)):
        order = rng.permutation(len(train))
    cursor = args.data_offset
    if branch is not None:
        rng.bit_generator.state = state['data']['rng']
        if 'input_rng' in state['data']:
            input_rng.bit_generator.state = state['data']['input_rng']
        if 'window_rng' in state['data']:
            window_rng.bit_generator.state = state['data']['window_rng']
        order, cursor = np.asarray(state['data']['order']), state['data']['cursor']
        del state
    saved = set()
    synchronize(args.device)
    start = time.monotonic()

    def milestone():
        nonlocal eval_flops
        def assess(split):
            return evaluate(student, teacher, batches(arrays[split], args.device, args.eval_sequences), indices, amp)
        validation, vf = counted(lambda: assess('validation'))
        controller, cf = counted(lambda: assess('controller'))
        eval_flops += vf + cf
        synchronize(args.device)
        row = dict(step=steps, training_flops=consumed, decision_flops=decision_flops, evaluation_flops=eval_flops,
                   total_accounted_flops=consumed + decision_flops + eval_flops + profile['profiling_flops'],
                   tokens=tokens, elapsed_seconds=time.monotonic() - start,
                   validation=validation, controller=controller, next_window=list(schedule.active))
        row['validation_regression'] = bool(history and validation['ce'] >
                                            min(x['validation']['ce'] for x in history) + 0.05)
        history.append(row)
        if validation['ce'] <= args.target_ce:
            hits.append(len(history) - 1)
        save_json(args.output / 'curve.json', history)
        print(json.dumps(row), flush=True)
        start_ce = branch['start_validation_ce'] if branch else history[0]['validation']['ce']
        for fraction in args.checkpoint_gaps:
            if steps and fraction not in saved and validation['ce'] <= start_ce - fraction * (start_ce - args.target_ce):
                save_checkpoint(fraction, row, start_ce)
                saved.add(fraction)

    def save_checkpoint(fraction, row, start_ce):
        folder = args.output / 'checkpoints' / f'gap{round(fraction * 100)}'
        folder.mkdir(parents=True)
        torch.save(dict(model=student.state_dict(), optimizer=optimizer.state_dict(), updates=dict(updates),
                        data=dict(rng=rng.bit_generator.state, order=order.tolist(), cursor=cursor,
                                  input_rng=input_rng.bit_generator.state,
                                  window_rng=window_rng.bit_generator.state)),
                   folder / 'state.pt')
        trunk = dict(trunk_run=str(args.output), trunk_strategy=args.strategy, trunk_seed=args.seed,
                     trunk_branch=branch, gap_fraction=fraction, start_validation_ce=start_ce,
                     checkpoint_validation_ce=row['validation']['ce'], trunk_step=steps, trunk_tokens=tokens,
                     trunk_training_flops=consumed, trunk_decision_flops=decision_flops,
                     trunk_elapsed_seconds=row['elapsed_seconds'])
        save_json(folder / 'checkpoint.json', dict(
            trunk=trunk, target_ce=args.target_ce, state_sha256=sha256(folder / 'state.pt'),
            damaged_manifest_sha256=sha256(args.damaged / 'pruning.json'),
            data_manifest_sha256=sha256(args.data / 'manifest.json'),
            rule='first evaluation after step 0 at or below the gap threshold (one evaluation, not confirmed)'))

    def decide():
        nonlocal decision_flops
        errors, cost = counted(lambda: stream_errors(student, teacher, batches(decision_rows, args.device), indices, amp))
        decision_flops += cost
        schedule.observe(errors, (consumed + decision_flops) / args.budget_flops)

    milestone()
    if schedule.adaptive:
        decide()
    next_eval = args.budget_flops / args.eval_points
    decision_every = args.budget_flops / args.controller_points
    next_decision = decision_every
    stop = 'budget'
    with (args.output / 'steps.jsonl').open('x') as log:
        while consumed + decision_flops < args.budget_flops:
            spent = consumed + decision_flops
            window = schedule.choose(spent / args.budget_flops)
            teacher_input = (args.window_inputs == 'teacher' or
                             (args.window_inputs == 'ramp' and input_rng.random() >= schedule.within))
            teacher_input = teacher_input and window[0] > 0
            cost = args.batch_sequences * step_cost(profile, window, teacher_input)
            if spent + cost > args.budget_flops:
                break
            if steps >= args.max_steps or time.monotonic() - start >= args.max_seconds:
                stop = 'step_limit' if steps >= args.max_steps else 'time_limit'
                break
            batch = []
            for _ in range(args.batch_sequences):
                if cursor and cursor % len(train) == 0:
                    order = rng.permutation(len(train))
                batch.append(torch.tensor(train[order[cursor % len(train)]][None], device=args.device, dtype=torch.long))
                cursor += 1
            optimizer.zero_grad(set_to_none=True)

            def backward():
                total = 0.0
                for ids in batch:
                    loss = training_loss(student, teacher, ids, indices, window, amp=amp,
                                         teacher_input=teacher_input) / len(batch)
                    loss.backward()
                    total = total + loss.detach()
                return total

            loss, actual = counted(backward)
            if actual != cost:
                raise RuntimeError(f'FLOP model mismatch: predicted {cost}, observed {actual}')
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite loss')
            torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
            decay = (1 + math.cos(math.pi * spent / args.budget_flops)) / 2 if args.lr_decay == 'cosine' else 1.0
            for group in optimizer.param_groups:
                if any(q.grad is not None for q in group['params']):
                    updates[group['name']] += 1
                    group['lr'] = args.lr * (min(1.0, updates[group['name']] / args.warmup_updates)
                                             if args.warmup_updates else 1.0) * decay
            optimizer.step()
            schedule.feedback(float(loss), spent / args.budget_flops)
            steps += 1
            consumed += actual
            tokens += sum(ids.numel() for ids in batch)
            log.write(json.dumps(dict(step=steps, window=list(window), teacher_input=teacher_input, lr_factor=decay, loss=float(loss.detach()),
                                      step_flops=actual, cumulative_flops=consumed, tokens=tokens)) + '\n')
            log.flush()
            spent = consumed + decision_flops
            if schedule.adaptive and spent >= next_decision:
                decide()
                next_decision = (math.floor(spent / decision_every) + 1) * decision_every
            if spent >= next_eval:
                milestone()
                next_eval = (math.floor(spent / (args.budget_flops / args.eval_points)) + 1) * args.budget_flops / args.eval_points
    if history[-1]['step'] != steps:
        milestone()
    # Test data is touched only after optimization and scheduling are finished.
    test, test_flops = counted(lambda: evaluate(student, teacher,
                                               batches(arrays['test'], args.device, args.eval_sequences), indices, amp))
    paired_hits = [i for i in hits if i > 0 and i - 1 in hits]
    first_hit = history[paired_hits[0]] if paired_hits else None
    summary = dict(status=stop, steps=steps, tokens=tokens, training_flops=consumed, decision_flops=decision_flops,
                   budget_flops=args.budget_flops, unspent_training_flops=args.budget_flops - consumed - decision_flops,
                   parameter_group_updates=updates,
                   profiling_flops=profile['profiling_flops'], evaluation_flops=eval_flops + test_flops,
                   total_accounted_flops=consumed + decision_flops + profile['profiling_flops'] + eval_flops + test_flops,
                   final_test=test, reached_target=first_hit is not None, first_confirmed_target=first_hit,
                   peak_host_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                   peak_device_allocated_bytes=torch.cuda.max_memory_allocated() if amp else None,
                   peak_device_reserved_bytes=torch.cuda.max_memory_reserved() if amp else None,
                   elapsed_seconds=time.monotonic() - start,
                   note='A time/step-limited run is censored, not an equal-budget result.')
    try:
        run_canary('end')
    except CanaryFailure as failure:
        summary.update(status='canary_failed', canary_end=json.loads(str(failure)))
        save_json(args.output / 'summary.json', summary)
        raise SystemExit(3)
    summary['canary'] = canary
    student.save_pretrained(args.output / 'final', safe_serialization=True, max_shard_size='2GB')
    save_json(args.output / 'controller-decisions.json', schedule.observations)
    save_json(args.output / 'summary.json', summary)


if __name__ == '__main__':
    main()
