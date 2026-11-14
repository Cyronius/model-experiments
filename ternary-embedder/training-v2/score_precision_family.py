"""Score a frozen float/ternary/Q4/Q8 family on the fixed observed tasks.

Optional source_summary/source_arm fields reuse already scored, SHA-identical
float/ternary checkpoints. Packed variants use the real packed loader, then
dense reference arithmetic; this does not establish native low-bit speed.
"""
import argparse
import concurrent.futures
import json
import os
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--freeze', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--dibs-inner', action='store_true')
    a = p.parse_args()
    if not a.dibs_inner:
        env = dict(os.environ, HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', DIBS_AGENT='precision-quality')
        subprocess.run(['dibs', 'run', 'cpu', '--label', 'Frozen precision family cached quality', '--eta', '25m', '--',
                        sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], '--dibs-inner'], check=True, env=env)
        return
    import torch
    from pair_training_state import digest, atomic_json
    from score_campaign import mteb_scores
    from summarize_pair_results import interval
    frozen = json.loads(a.freeze.read_text())
    assert set(frozen['models']) == {'float', 'ternary', 'q4', 'q8'}
    a.output.mkdir(parents=True, exist_ok=True)
    logs = a.output / 'logs'
    logs.mkdir(exist_ok=True)
    models, pending = {}, []
    for name, selection in frozen['models'].items():
        assert digest(Path(selection['path'])) == selection['sha256']
        if name in ('q4', 'q8'):
            blob = torch.load(selection['path'], map_location='cpu', weights_only=False)
            assert blob['format'] == 'ternary-embedder-dense-lowbit-v1'
            assert blob['metadata']['source_sha256'] == frozen['models']['float']['sha256']
            assert blob['metadata']['dense_weight_bits'] == int(name[1:]) and blob['metadata']['activation_bits'] == 8
            assert blob['dims']['layers'] == frozen['models']['float']['layers']
        if selection.get('source_summary'):
            source = Path(selection['source_summary'])
            if selection.get('source_summary_sha256'):
                assert digest(source) == selection['source_summary_sha256']
            arm = selection['source_arm']
            previous = json.loads(source.read_text())['models'][arm]
            assert previous['selection']['sha256'] == selection['sha256']
            assert len(previous['mteb_12']) == 12
            models[name] = dict(previous, selection=selection, reused_sha_identical_result=str(source))
            rows = source.parent / (arm + '-rows.json')
            target = a.output / (name + '-rows.json')
            if rows.resolve() != target.resolve():
                shutil.copyfile(rows, target)
        else:
            pending.append(name)

    def worker(name):
        quality = a.output / (name + '-quality.json')
        if quality.exists():
            assert json.loads(quality.read_text())['selection']['sha256'] == frozen['models'][name]['sha256']
        else:
            with (logs / (name + '-quality.log')).open('w') as log:
                subprocess.run([sys.executable, str(ROOT / 'evaluate_data_first.py'), '--model', name,
                                '--freeze', str(a.freeze), '--output', str(a.output)], check=True,
                               stdout=log, stderr=subprocess.STDOUT)
        with (logs / (name + '-mteb.log')).open('a') as log:
            subprocess.run([str(PROJECT / '.eval-venv/bin/python'), str(ROOT / 'evaluate_round2_mteb.py'),
                            '--freeze', str(a.freeze), '--branch', name, '--output', str(a.output / 'mteb' / name)],
                           check=True, stdout=log, stderr=subprocess.STDOUT)
        result = json.loads(quality.read_text())
        wide = mteb_scores(a.output / 'mteb' / name / 'candidate')
        assert len(wide) == 12 and abs(wide['STSBenchmark'] - result['stsbenchmark_spearman']) < .02
        for task in ('SciFact', 'NFCorpus'):
            assert abs(wide[task] - result['tasks'][task]['ndcg10']) < .02
        return name, dict(selection=frozen['models'][name], core_retrieval=result['mean_core_retrieval'],
                          stsbenchmark=result['stsbenchmark_spearman'], tasks=result['tasks'], gates=result['gates'],
                          mteb_12=wide, mteb_12_mean=statistics.mean(wide.values()))
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        for name, result in pool.map(worker, pending):
            models[name] = result
    original = json.loads((a.output / 'float-rows.json').read_text())
    contrasts = {}
    for name in ('q4', 'q8'):
        contrasts[name + ' minus float'] = dict(
            core=interval(original, json.loads((a.output / (name + '-rows.json')).read_text())),
            stsbenchmark_delta=models[name]['stsbenchmark'] - models['float']['stsbenchmark'],
            mean12_delta=models[name]['mteb_12_mean'] - models['float']['mteb_12_mean'])
    report = dict(scope='Frozen family; fixed12 observed development tasks, not full MTEB. Independent float/ternary histories; only Q4/Q8-minus-float contrasts isolate precision export.',
                  reserved_fiqa_scidocs_scored=False, packed_kernel_speed_established=False,
                  models=models, matched_precision_contrasts=contrasts)
    atomic_json(a.output / 'summary.json', report)
    print(json.dumps({name: dict(core=m['core_retrieval'], sts=m['stsbenchmark'], mean12=m['mteb_12_mean'])
                      for name, m in models.items()}, indent=2), flush=True)


if __name__ == '__main__':
    main()
