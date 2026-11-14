"""One coordinated CPU/network reservation for the current fixed comparisons.

Run under dibs run cpu network runpod-ternary. Does not allocate paid resources.
"""
import json
import concurrent.futures
import subprocess
import sys
import time
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent


def recovered():
    out = ROOT / 'recovered-depth/results'
    for name in ('float', 'ternary'):
        if (out / (name + '-quality.json')).exists():
            continue
        with (out / (name + '-quality.log')).open('a') as log:
            subprocess.run([sys.executable, str(ROOT / 'evaluate_data_first.py'),
                            '--model', name, '--freeze', str(out / 'frozen.json'),
                            '--output', str(out)], check=True, stdout=log, stderr=subprocess.STDOUT)
    for name in ('float', 'ternary'):
        with (out / (name + '-mteb.log')).open('a') as log:
            subprocess.run([str(PROJECT / '.eval-venv/bin/python'), str(ROOT / 'evaluate_round2_mteb.py'),
                            '--freeze', str(out / 'frozen.json'), '--branch', name,
                            '--output', str(out / 'mteb' / name)], check=True, stdout=log, stderr=subprocess.STDOUT)
    from score_campaign import mteb_scores
    report = {'scope': 'Historical fixed three-layer checkpoints; observed development tasks, not a matched pretraining-depth experiment.', 'models': {}}
    for name in ('float', 'ternary'):
        quality = json.loads((out / (name + '-quality.json')).read_text())
        wide = mteb_scores(out / 'mteb' / name / 'candidate')
        assert len(wide) == 12
        assert abs(wide['STSBenchmark'] - quality['stsbenchmark_spearman']) < .02
        report['models'][name] = dict(selection=quality['selection'], core_retrieval=quality['mean_core_retrieval'],
                                     stsbenchmark=quality['stsbenchmark_spearman'], tasks=quality['tasks'],
                                     mteb_12=wide, mteb_12_mean=statistics.mean(wide.values()))
    (out / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print('Recovered-depth quality complete', flush=True)


def main():
    recovered()
    arms = {
        'hardneg': ['float-pairs', 'float-hard', 'ternary-pairs', 'ternary-hard'],
        'nli-aux': ['float-0.0', 'float-0.1', 'float-0.25', 'ternary-0.0', 'ternary-0.1', 'ternary-0.25'],
    }
    def score(profile, names):
        state = ROOT / 'work' / (profile + '-run') / 'pod.json'
        while True:
            s = json.loads(state.read_text())
            if s.get('deletion_verified_http_status') == 404:
                break
            if time.time() > s['deadline'] + 600:
                raise RuntimeError('Deletion verification overdue: ' + profile)
            time.sleep(10)
        with (ROOT / profile / 'results-scoring.log').open('a') as log:
            subprocess.run([sys.executable, str(ROOT / 'score_campaign.py'), '--profile', profile,
                            '--arms', *names, '--mteb', '--dibs-inner'], check=True,
                           stdout=log, stderr=subprocess.STDOUT)
        print(profile, 'quality complete', flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(score, profile, names) for profile, names in arms.items()]
        for future in futures:
            future.result()
    subprocess.run([sys.executable, str(ROOT / 'compare_campaign.py')], check=True)


if __name__ == '__main__':
    main()
