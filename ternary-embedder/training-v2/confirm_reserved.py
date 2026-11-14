"""One frozen Q8 confirmation; never tune or select using these judgments."""
import argparse
import concurrent.futures
import datetime
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
OUT = ROOT / 'delivery/reserved'
TASKS = ('FiQA', 'SCIDOCS')
REFS = {
    'bge_micro': PROJECT / 'cache/hub/models--TaylorAI--bge-micro-v2/snapshots/3edf6d7de0faa426b09780416fe61009f26ae589',
    'minilm': PROJECT / 'cache/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/1110a243fdf4706b3f48f1d95db1a4f5529b4d41',
}
from pair_training_state import atomic_json, digest


def read(path):
    return json.loads(path.read_text())


def activate():
    from compare_campaign import wide
    OUT.mkdir(parents=True, exist_ok=True)
    frozen_path = PROJECT / 'models/campaign-20261025/frozen.json'
    quality_path = ROOT / 'delivery/family-results/summary.json'
    reference_path = ROOT / 'round2/quality.json'
    frozen, quality, references = map(read, (frozen_path, quality_path, reference_path))
    candidate = frozen['models']['q8']
    assert digest(Path(candidate['path'])) == candidate['sha256'] == '38e400de8c984ec7e7dd9aa33e2163096b8dc3725b1e89fdf416fda0f6117d70'
    assert quality['models']['q8']['selection']['sha256'] == candidate['sha256']
    gates = {}
    q = quality['models']['q8']
    for name in REFS:
        r = references['models'][name]
        scores = wide(ROOT / 'campaign/mteb-references' / name)
        assert len(scores) == 12
        gates[name] = dict(
            each_core=all(q['tasks'][t]['ndcg10'] > r['tasks'][t]['ndcg10'] for t in ('SciFact', 'NFCorpus')),
            sts=q['stsbenchmark'] > r['stsbenchmark_spearman'],
            mean12=q['mteb_12_mean'] > statistics.mean(scores.values()))
        assert all(gates[name].values()), gates
    activation_path = OUT / 'activation.json'
    if activation_path.exists():
        receipt = read(activation_path)
        assert receipt['candidate'] == candidate
        assert receipt['frozen_sha256'] == digest(frozen_path)
        assert receipt['quality_sha256'] == digest(quality_path)
    else:
        for task in TASKS:
            assert not (PROJECT / 'comparisons/qwen06b/data' / task.lower() / 'qrels/test.tsv').exists(), 'Judgments unexpectedly already extracted'
        receipt = dict(activated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                       status='Frozen before first extraction/read of reserved judgments',
                       candidate=candidate, frozen_sha256=digest(frozen_path), quality_sha256=digest(quality_path),
                       references_sha256=digest(reference_path), gates=gates,
                       candidate_development_metrics={k:q[k] for k in ('core_retrieval','stsbenchmark','mteb_12_mean')},
                       reference_revisions={k:str(v) for k,v in REFS.items()}, tasks=TASKS,
                       protocol_sha256=digest(ROOT/'delivery/RESERVED_PROTOCOL.md'),
                       no_further_training_or_selection=True, token_cap=128, query_instructions=False)
        atomic_json(activation_path, receipt)
    # Extraction occurs only after the durable frozen-candidate receipt exists.
    for task in TASKS:
        base = PROJECT / 'comparisons/qwen06b/data'
        target = base / task.lower() / 'qrels/test.tsv'
        with zipfile.ZipFile(base / (task.lower()+'.zip')) as archive:
            members = [n for n in archive.namelist() if n.endswith('/qrels/test.tsv')]
            assert len(members) == 1
            content = archive.read(members[0])
        if target.exists():
            assert target.read_bytes() == content
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
    return receipt


def worker(name):
    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer
    sys.path.insert(0, str(PROJECT / 'comparisons/qwen06b'))
    from compare import data
    from round2_metrics import evaluate
    from diagnose import cached_encode
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    receipt = read(OUT / 'activation.json')
    if name == 'q8':
        from export_pair_precision import load_reference
        candidate = receipt['candidate']
        assert digest(Path(candidate['path'])) == candidate['sha256']
        model = load_reference(Path(candidate['path'])).eval()
        tokenizer = AutoTokenizer.from_pretrained(PROJECT/'tokenizer', local_files_only=True)
        identity = dict(name=name, sha256=candidate['sha256'], cap=128)
    else:
        path = REFS[name]
        model = AutoModel.from_pretrained(path, local_files_only=True).eval()
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        weights = sorted(path.glob('*.safetensors')) or sorted(path.glob('pytorch_model*.bin'))
        assert weights
        identity = dict(name=name, revision=path.name, weights={p.name:digest(p) for p in weights}, pooling='masked-mean-unit', cap=128)
    result = dict(identity=identity, activation_sha256=digest(OUT/'activation.json'), tasks={}, token_cap=128)
    rows = {}
    for task in TASKS:
        d = data(task)
        vectors = cached_encode(model, tokenizer, d['queries']+d['texts'], name!='q8', 64, identity,
                                OUT/'vectors'/f'{name}-{task}.npy')
        n = len(d['queries'])
        metrics, rr = evaluate(vectors[:n] @ vectors[n:].T, d)
        result['tasks'][task] = dict(metrics, source_sha256=d['source_sha256'])
        rows[task] = rr
        atomic_json(OUT/(name+'-rows.json'), rows)
        atomic_json(OUT/(name+'-quality.json'), result)
        print(name, task, metrics['ndcg10'], flush=True)
    return name


def contrast(left, right):
    import numpy as np
    rng = np.random.default_rng(20261025)
    per_task, bootstraps, means = {}, [], []
    for task in TASKS:
        a,b = left[task],right[task]
        assert [r['query_id'] for r in a] == [r['query_id'] for r in b]
        delta = np.array([y['ndcg10']-x['ndcg10'] for x,y in zip(a,b)])
        samples = delta[rng.integers(0,len(delta),(5000,len(delta)))].mean(1)*100
        means.append(float(delta.mean()*100)); bootstraps.append(samples)
        per_task[task] = dict(delta_ndcg10=means[-1], paired_query_bootstrap_95_ci=np.percentile(samples,[2.5,97.5]).tolist())
    return dict(tasks=per_task, mean_delta_ndcg10=statistics.mean(means),
                paired_query_bootstrap_95_ci=np.percentile(np.mean(bootstraps,axis=0),[2.5,97.5]).tolist(),
                seed=20261025, resamples=5000,
                scope='Query sampling within these two fixed datasets only; not domain or training-seed uncertainty')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--worker',choices=('q8',*REFS));args=parser.parse_args()
    if args.worker:
        worker(args.worker);return
    receipt=activate()
    def launch(name):
        env=dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
                 HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
        with (OUT/(name+'.log')).open('a') as log:
            subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker',name],check=True,
                           stdout=log,stderr=subprocess.STDOUT,env=env)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(launch,('q8',*REFS)))
    models={name:read(OUT/(name+'-quality.json')) for name in ('q8',*REFS)}
    for m in models.values():
        assert set(m['tasks']) == set(TASKS)
        assert m['activation_sha256'] == digest(OUT/'activation.json')
        m['mean_retrieval']=statistics.mean(t['ndcg10'] for t in m['tasks'].values())
    text_audit=read(ROOT/'data-first/audit.json')
    text_files={}
    for task in TASKS:
        for filename in ('corpus.jsonl','queries.jsonl'):
            relative=f'comparisons/qwen06b/data/{task.lower()}/{filename}'
            expected=text_audit['reference_text_files'][relative]
            assert digest(PROJECT/relative)==expected
            assert all(m['tasks'][task]['source_sha256'][filename]==expected for m in models.values())
            text_files[relative]=expected
    candidate_rows=read(OUT/'q8-rows.json')
    contrasts={name:contrast(read(OUT/(name+'-rows.json')),candidate_rows) for name in REFS}
    report=dict(scope='Frozen-candidate reserved retrieval confirmation; not full MTEB', reserved_judgments_opened=True,
                activation=receipt, activation_sha256=digest(OUT/'activation.json'), models=models,
                q8_minus_references=contrasts,
                beats_each_task={name:all(models['q8']['tasks'][t]['ndcg10']>models[name]['tasks'][t]['ndcg10'] for t in TASKS) for name in REFS},
                beats_mean={name:models['q8']['mean_retrieval']>models[name]['mean_retrieval'] for name in REFS},
                original_training_exclusion_texts_verified=text_files,
                default_deployment_changed=False, further_selection_performed=False)
    atomic_json(ROOT/'delivery/reserved-confirmation.json',report)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()
