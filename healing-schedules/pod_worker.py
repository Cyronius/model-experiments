"""Runs on the rented GPU: iterative pruning against one-shot pruning (plan-iterative-prune.md).

Started by run_pod.py on the pod from /workspace/manifold (a repository-shaped tree). Everything the launcher
copies home is under outputs/: per arm, each stage's curve, summary, provenance and steps, the artifacts'
pruning.json, the quick evaluation and the block probe; status.json for the launcher; the final weights of
the staged (S3) arms. Weights and other bulky files stay in work/.

Arms run in priority order. One whose estimated time would pass the deadline is skipped, not started.

Local smoke (CPU, real 2B, a step per stage):
    POD_SMOKE=1 POD_ROOT=<repo> python healing-schedules/pod_worker.py
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import urllib.request

SMOKE = os.environ.get('POD_SMOKE') == '1'
ROOT = Path(os.environ.get('POD_ROOT', Path.cwd())).resolve()
HS = ROOT / 'healing-schedules'
OUT = Path(os.environ.get('POD_OUT', ROOT / 'outputs'))
WORK = Path(os.environ.get('POD_WORK', ROOT / 'work'))
sys.path.insert(0, str(HS))
PY = sys.executable
TEACHER = ROOT / 'models/manifold-audit/Qwen3.5-2B'
DATA = HS / 'data/mix-2048'
LIGHT = HS / 'artifacts/damaged-light-h1792-n1473244736'
BASE = HS / 'baselines/mix-2048'
DEVICE = os.environ.get('POD_DEVICE', 'cpu' if SMOKE else 'cuda')
STUDY = 6 * 10**15            # the study budget (1x), training FLOPs
LR = '0.00003'
REST, GRAPHQL = 'https://rest.runpod.io/v1/pods/', 'https://api.runpod.io/graphql'

# name, kind, cuts (fraction of the teacher's non-embedding parameters removed after each stage), budget, seed
ARMS = [
    # oneshot-1x-seed17 already ran on the first pod (hs-dec55376b8e8, torch fallback kernel: val 3.1408,
    # test 3.1166); it reruns here on fla only if time allows. oneshot-1x-seed2 is the same-kernel baseline.
    ('S3-1x-seed17', 'staged', [.10, .20, .25], STUDY, 17),
    ('oneshot-1x-seed2', 'oneshot', [.25], STUDY, 2),
    ('S2-1x-seed17', 'staged', [.125, .25], STUDY, 17),
    ('S5-1x-seed17', 'staged', [.05, .10, .15, .20, .25], STUDY, 17),
    ('oneshot-1x-seed17', 'oneshot', [.25], STUDY, 17),
    ('S3-10x-seed17', 'staged', [.10, .20, .25], 10 * STUDY, 17),
    ('oneshot-10x-seed17', 'oneshot', [.25], 10 * STUDY, 17),
]
KEEP_WEIGHTS = {'S3-1x-seed17', 'S3-10x-seed17'}
if SMOKE:
    ARMS = [('S2-smoke', 'staged', [.125, .25], 2 * 10**14, 17)]
    KEEP_WEIGHTS = set()


def save(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, indent=2) + '\n')
    temp.replace(path)


def status(stage, **info):
    save({'stage': stage, 'updated_at': time.time(), **info}, OUT / 'status.json')
    print(time.strftime('%T'), stage, json.dumps(info), flush=True)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(1 << 20), b''):
            value.update(data)
    return value.hexdigest()


def artifacts():
    save({str(p.relative_to(OUT)): sha(p) for p in sorted(OUT.rglob('*'))
          if p.is_file() and p.suffix not in ('.log', '.tmp') and
          p.name not in ('status.json', 'artifact-manifest.json')}, OUT / 'artifact-manifest.json')


# --- the pod's own key, used only to delete this pod if the launcher and its watchdog have both died ---

def pod_identity():
    try:
        items = Path('/proc/1/environ').read_bytes().decode(errors='replace').split('\0')
    except OSError:
        return None, None
    env = dict(item.split('=', 1) for item in items if '=' in item)
    return env.get('RUNPOD_POD_ID'), env.get('RUNPOD_API_KEY')


def provider(method, url, key, data=None):
    request = urllib.request.Request(url, data=json.dumps(data).encode() if data is not None else None, method=method,
        headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json', 'Accept': 'application/json',
                 'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(request, timeout=20) as response:
        body = response.read()
        return json.loads(body) if body else None


def delete_self_at(when):
    time.sleep(max(0, when - time.time()))
    pod, key = pod_identity()
    if not pod or not key:
        print('no pod id or key in the environment: nothing to do', flush=True)
        return
    for _ in range(120):
        for name, call in (('rest', lambda: provider('DELETE', REST + pod, key)),
                           ('graphql', lambda: provider('POST', GRAPHQL, key,
                                                        {'query': 'mutation { podTerminate(input: {podId: "%s"}) }' % pod}))):
            try:
                print(time.strftime('%T'), name, 'delete:', call(), flush=True)
            except Exception as error:
                print(time.strftime('%T'), name, 'delete failed:', type(error).__name__, flush=True)
        time.sleep(60)


# --- setup ----------------------------------------------------------------------------------------

def fetch_teacher():
    manifest = json.loads((TEACHER / 'download_manifest.json').read_text())
    for name, info in manifest['files'].items():
        path = TEACHER / name
        if not path.exists():
            from huggingface_hub import hf_hub_download
            got = hf_hub_download(manifest['repo'], name, revision=manifest['revision'], local_dir=str(WORK / 'hf'))
            shutil.move(got, path)
        if sha(path) != info['sha256']:
            raise ValueError(f'Teacher file {name} does not match its pinned hash')


def geometry(cut):
    """Residual width and exact total parameters after removing `cut` of the teacher's non-embedding parameters.
    The residual shrinks along the light student's slope (256 coordinates per 25%, aligned to 64)."""
    from accounting import count, target_config
    config = json.loads((HS / 'metadata/Qwen3.5-2B.json').read_text())['config']
    full = count(config)
    teacher = full['total'] - full['embedding_and_tied_head']
    hidden = int(round((2048 - 256 * cut / .25) / 64) * 64)
    cfg = dict(config.get('text_config', config), hidden_size=hidden)
    target = int(count(cfg)['embedding_and_tied_head'] + (1 - cut) * teacher)
    total = count(target_config(config, target, hidden))['total']
    return hidden, total


# --- steps ----------------------------------------------------------------------------------------

def call(command, log, attempts=2):
    """Run a step; a canary failure (exit 3) is retried in a fresh process."""
    for attempt in range(attempts):
        with log.open('a') as stream:
            stream.write(f'--- {time.strftime("%T")} {" ".join(map(str, command))}\n')
            stream.flush()
            code = subprocess.run([str(c) for c in command], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT).returncode
        if code == 0:
            return
        if code != 3:
            raise RuntimeError(f'{Path(str(command[1])).name} failed with exit {code}; see {log.name}')
    raise RuntimeError(f'{Path(str(command[1])).name}: canary failed {attempts} times')


def canary():
    return [] if DEVICE == 'cpu' else ['--canary', BASE / 'canary-cpu.json', '--canary-gpu', OUT / 'canary-gpu.json']


def prune(cut, output, parent=None, healed=None, log=None):
    hidden, total = geometry(cut)
    command = [PY, HS / 'pruning.py', '--output', output, '--hidden', hidden, '--target', total, '--data', DATA,
               '--device', DEVICE, '--threads', 8]
    command += ['--source', healed, '--from-artifact', parent] if parent else ['--source', TEACHER]
    if SMOKE:
        command += ['--probe-rows', 1, '--train-rows', 1]
    call(command, log, attempts=1)
    return hidden, total


def heal(artifact, output, budget, seed, offset, evals, log):
    command = [PY, HS / 'run.py', '--teacher', TEACHER, '--damaged', artifact, '--data', DATA, '--output', output,
               '--strategy', 'joint', '--lr-decay', 'cosine', '--seed', seed, '--budget-flops', budget,
               '--target-ce', TARGET, '--eval-points', evals, '--eval-sequences', 1 if SMOKE else 32,
               '--max-seconds', 40000, '--batch-sequences', 1, '--lr', LR, '--data-offset', offset,
               '--device', DEVICE] + canary()
    if SMOKE:
        command += ['--max-steps', 1]
    for attempt in range(2):
        try:
            call(command, log, attempts=1)
            return json.loads((output / 'summary.json').read_text())
        except RuntimeError:
            failed = output / 'canary-failed.json'
            if attempt == 0 and failed.exists():
                shutil.move(output, output.with_name(output.name + f'.canary-failed-{int(time.time())}'))
                continue
            raise


def evaluate(final, artifact, arm_out, log):
    quick = ['--splits', 'test', '--mc-examples', 200, '--gsm8k-examples', 32, '--gsm8k-tokens', 1024]
    if SMOKE:
        quick = ['--splits', 'test', '--sequences', 1, '--mc-examples', 2, '--gsm8k-examples', 1, '--gsm8k-tokens', 8]
    call([PY, HS / 'evaluate.py', '--model', final, '--teacher', TEACHER, '--tokenizer', TEACHER, '--data', DATA,
          '--output', arm_out / 'eval-quick.json', '--device', DEVICE] + quick + canary(), log)
    call([PY, HS / 'block_probe.py', '--model', f'final={final}', '--teacher', TEACHER,
          '--pruning', artifact / 'pruning.json', '--data', DATA, '--output', arm_out / 'block-probe.json',
          '--sequences', 1 if SMOKE else 32, '--device', DEVICE] + canary(), log)


def keep_small(run_dir, dest):
    dest.mkdir(parents=True, exist_ok=True)
    for name in ('curve.json', 'summary.json', 'provenance.json', 'steps.jsonl', 'controller-decisions.json'):
        if (run_dir / name).exists():
            shutil.copy2(run_dir / name, dest / name)


def run_arm(name, kind, cuts, budget, seed):
    arm_work, arm_out = WORK / name, OUT / name
    arm_work.mkdir(parents=True, exist_ok=True)
    arm_out.mkdir(parents=True, exist_ok=True)
    log = arm_out / 'arm.log'
    record = {'name': name, 'kind': kind, 'cuts': cuts, 'budget_flops': budget, 'seed': seed, 'stages': []}
    began = time.time()
    share = budget // len(cuts)
    # About 24 evaluations per 1x arm and 40 per 10x arm, split over the stages.
    evals = max(2, round((24 if budget <= STUDY else 40) / len(cuts)))
    offset, artifact, final = 0, None, None
    for k, cut in enumerate(cuts, 1):
        stage = {'stage': k, 'cut': cut}
        if kind == 'oneshot':
            artifact = LIGHT
        else:
            t = time.time()
            new = arm_work / f'artifact{k}'
            stage['hidden'], stage['parameters'] = prune(cut, new, artifact, final, log)
            stage['prune_seconds'] = time.time() - t
            artifact = new
            shutil.copy2(artifact / 'pruning.json', arm_out / f'stage{k}-pruning.json')
        run_dir = arm_work / f'stage{k}'
        summary = heal(artifact, run_dir, share, seed, offset, evals, log)
        curve = json.loads((run_dir / 'curve.json').read_text())
        stage.update(start_val_ce=curve[0]['validation']['ce'], end_val_ce=curve[-1]['validation']['ce'],
                     steps=summary['steps'], tokens=summary['tokens'], training_flops=summary['training_flops'],
                     data_offset=offset, final_test=summary.get('final_test'), seconds=summary['elapsed_seconds'])
        keep_small(run_dir, arm_out / f'stage{k}')
        record['stages'].append(stage)
        save(record, arm_out / 'arm.json')
        offset += summary['steps']
        final = run_dir / 'final'
        status('running', arm=name, stage_index=k, of=len(cuts), start_val_ce=stage['start_val_ce'], end_val_ce=stage['end_val_ce'])
    t = time.time()
    evaluate(final, artifact, arm_out, log)
    record.update(eval_seconds=time.time() - t, seconds=time.time() - began,
                  training_flops=sum(s['training_flops'] for s in record['stages']),
                  tokens=sum(s['tokens'] for s in record['stages']),
                  final_val_ce=record['stages'][-1]['end_val_ce'],
                  final_test_ce=json.loads((arm_out / 'eval-quick.json').read_text())['test']['ce'])
    if name in KEEP_WEIGHTS:
        shutil.copytree(final, OUT / 'weights' / name)
        shutil.copy2(artifact / 'pruning.json', OUT / 'weights' / name / 'pruning.json')
    save(record, arm_out / 'arm.json')
    shutil.rmtree(arm_work)       # weights of finished arms are not needed again
    return record


def estimate(kind, cuts, budget, rates):
    """Seconds an arm should take, from the arms finished so far (None before the first)."""
    if not rates:
        return None
    flops_per_second = sum(r['flops_per_second'] for r in rates) / len(rates)
    overhead = max(r['overhead'] for r in rates)
    prunes = 0 if kind == 'oneshot' else len(cuts) * max([r['prune'] for r in rates] + [180])
    return budget / flops_per_second + overhead + prunes


def main():
    global TARGET
    deadline = float(os.environ.get('MANIFOLD_JOB_DEADLINE', time.time() + 86400))
    OUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    if not SMOKE:
        subprocess.Popen([PY, __file__, '--delete-self-at', str(deadline + 600 + 900)],
                         stdout=(OUT / 'self-delete.log').open('a'), stderr=subprocess.STDOUT, start_new_session=True)
    try:
        status('setup')
        fetch_teacher()
        TARGET = json.loads((BASE / 'target-light.json').read_text())['target_ce']
        if (geometry(.25)) != (1792, 1473244736):
            raise ValueError('Stage geometry does not land on the light student')
        if not SMOKE:
            import torch
            status('setup', gpu=torch.cuda.get_device_name(), torch=torch.__version__)
            try:          # a record of the pod's software, not a gate: the arms below check their own inputs
                call([PY, '-m', 'pytest', '-q', HS / 'test_experiment.py'], OUT / 'tests.log', attempts=1)
                status('setup', tests='passed')
            except RuntimeError:
                status('setup', tests='FAILED, see tests.log')
        done, skipped, rates, failed = [], [], [], []
        for name, kind, cuts, budget, seed in ARMS:
            need = estimate(kind, cuts, budget, rates)
            if need is not None and time.time() + 1.25 * need > deadline - 900:
                skipped.append({'arm': name, 'estimated_seconds': need})
                status('running', skipped=name, estimated_seconds=need)
                continue
            status('running', arm=name, done=[d['name'] for d in done])
            try:
                record = run_arm(name, kind, cuts, budget, seed)
            except Exception as error:     # one arm's failure does not cost the others
                traceback.print_exc()
                failed.append({'arm': name, 'error': f'{type(error).__name__}: {error}'[:2000]})
                save({'done': done, 'skipped': skipped, 'failed': failed}, OUT / 'results.json')
                shutil.rmtree(WORK / name, ignore_errors=True)
                continue
            train = sum(s['seconds'] for s in record['stages'])
            prune_seconds = [s.get('prune_seconds', 0) for s in record['stages']]
            rates.append({'flops_per_second': record['training_flops'] / max(1, train),
                          'overhead': record['eval_seconds'] + 120,
                          'prune': max(prune_seconds) if kind == 'staged' else 0})
            done.append(record)
            save({'done': done, 'skipped': skipped, 'failed': failed}, OUT / 'results.json')
            artifacts()
        save({'done': done, 'skipped': skipped, 'failed': failed}, OUT / 'results.json')
        artifacts()
        status('complete', outcome={d['name']: round(d['final_val_ce'], 4) for d in done},
               skipped=[s['arm'] for s in skipped], failed=[f['arm'] for f in failed])
    except Exception as error:
        traceback.print_exc()
        artifacts()
        status('failed', error=f'{type(error).__name__}: {error}'[:2000])
        raise


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--delete-self-at':
        delete_self_at(float(sys.argv[2]))
    else:
        main()
