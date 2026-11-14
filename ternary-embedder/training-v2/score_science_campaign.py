"""Wait for both owned science jobs to close, freeze all six, then score.

The CPU/network reservation begins after paid pods are deleted, so waiting for
a shared CPU cannot extend GPU billing. No paid resources are allocated here.
"""
import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROFILES = ('science-float', 'science-ternary')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dibs-inner', action='store_true')
    args = parser.parse_args()
    if not args.dibs_inner:
        while True:
            states = []
            for profile in PROFILES:
                path = ROOT / 'work' / (profile + '-run') / 'pod.json'
                if path.exists():
                    states.append(json.loads(path.read_text()))
            if len(states) == 2 and all(s.get('deletion_verified_http_status') == 404 for s in states):
                break
            if states and time.time() > max(s['deadline'] for s in states) + 600:
                raise RuntimeError('Science pod deletion verification overdue')
            time.sleep(10)
        env = dict(os.environ, HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', DIBS_AGENT='science-quality')
        subprocess.run(['dibs', 'run', 'cpu', '--label', 'Six frozen science cached quality comparisons',
                        '--eta', '35m', '--', sys.executable, str(Path(__file__).resolve()), '--dibs-inner'], check=True, env=env)
        return
    arms = {}
    for profile in PROFILES:
        plan = json.loads((ROOT / profile / 'runpod-job.json').read_text())
        arms[profile] = [v['id'] for v in plan['training']['arms']]
        freeze = ROOT / profile / 'results/frozen.json'
        freeze.parent.mkdir(exist_ok=True)
        if not freeze.exists():
            with (freeze.parent / 'freeze.log').open('w') as log:
                subprocess.run([sys.executable, str(ROOT / 'freeze_campaign.py'), '--profile', profile,
                                '--arms', *arms[profile]], check=True, stdout=log, stderr=subprocess.STDOUT)
    # No external labels are read until both three-arm freezes exist.
    def score(profile):
        with (ROOT / profile / 'results-scoring.log').open('a') as log:
            subprocess.run([sys.executable, str(ROOT / 'score_campaign.py'), '--profile', profile,
                            '--arms', *arms[profile], '--mteb', '--dibs-inner'], check=True,
                           stdout=log, stderr=subprocess.STDOUT)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(score, p) for p in PROFILES]
        for future in futures:
            future.result()
    summaries = {p: json.loads((ROOT / p / 'results/summary.json').read_text()) for p in PROFILES}
    models = {name: model for report in summaries.values() for name, model in report['models'].items()}
    assert len(models) == 6
    from summarize_pair_results import interval
    contrasts = {}
    for precision in ('float', 'ternary'):
        profile = 'science-' + precision
        out = ROOT / profile / 'results'
        left, right = precision + '-fresh', '3-' + precision + '-fresh'
        contrasts[precision] = interval(json.loads((out / (left + '-rows.json')).read_text()),
                                        json.loads((out / (right + '-rows.json')).read_text()))
    result = {'scope': 'Observed development tasks; fixed12-task subset, not full MTEB. Both profiles frozen before scoring.',
              'resources': [v['resources'] for v in summaries.values()], 'models': models,
              'matched_recipe_contrasts': {k: v for report in summaries.values() for k, v in report['matched_recipe_contrasts'].items()},
              'candidate_depth_contrasts_different_pretraining': contrasts,
              'depth_comparison_limit': 'Different historical pretraining and token budgets; candidate comparison, not a controlled causal depth experiment.',
              'reserved_fiqa_scidocs_scored': False}
    dest = ROOT / 'science-mix/results'
    dest.mkdir(exist_ok=True)
    (dest / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    subprocess.run([sys.executable, str(ROOT / 'compare_campaign.py')], check=True)


if __name__ == '__main__':
    main()
