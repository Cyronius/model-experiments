"""Freeze the complete paid comparison, then score locally with no new charges."""
import argparse
import concurrent.futures
import datetime
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

from pair_training_state import atomic_json, digest

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
PROFILE = ROOT / 'broad-data'
WORK = ROOT / 'work/broad-data-run'
OUT = PROFILE / 'results'


def read(path):
    return json.loads(path.read_text())


def status(message):
    atomic_json(PROFILE / 'quality-work.json', dict(
        status=message, updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        paid_allocation_performed_by_this_script=False))
    print(message, flush=True)


def run(command, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('a') as output:
        subprocess.run(command, check=True, stdout=output, stderr=subprocess.STDOUT)


def write_report(report):
    resource=report['resource']
    lines=['# Broad-data controlled comparison','',
        f"All four training arms completed. The paid pod is deleted with HTTP404 verification. Estimated total cost: **${resource['estimated_total_usd']:.2f}**, within the approved **$10** cap.",'',
        'Architecture: three layers, 384 dimensions, 17,241,984 parameters. Each arm saw 3,771,503 pair presentations. Q4/Q8 are fixed exports from the independent float branch.','',
        '| Candidate | SciFact | NFCorpus | FiQA | SCIDOCS | STSBenchmark | Fixed12 mean |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name,entry in sorted(report['models'].items()):
        values=[entry['tasks'][t]['ndcg10'] for t in ('SciFact','NFCorpus','FiQA','SCIDOCS')]+[entry['stsbenchmark'],entry['mteb_12_mean']]
        lines.append('| '+name+' | '+' | '.join(f'{v:.2f}' for v in values)+' |')
    for name,entry in report['references'].items():
        values=[entry['tasks'][t]['ndcg10'] for t in ('SciFact','NFCorpus','FiQA','SCIDOCS')]+[entry['stsbenchmark'],entry['mteb_12_mean']]
        lines.append('| '+name+' | '+' | '.join(f'{v:.2f}' for v in values)+' |')
    lines+=['','## Broader data versus repeated old data','',
        '| Precision | Core retrieval delta (95% query CI) | FiQA/SCIDOCS delta (95% query CI) | STS delta | Fixed12 delta |',
        '|---|---:|---:|---:|---:|']
    for precision,c in report['broad_minus_repeat'].items():
        core=c['core'];extra=c['fiqa_scidocs']
        ci=core['paired_query_bootstrap_95_ci']
        core_text=f"{core['mean_delta_ndcg10']:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]"
        extra_ci=extra['paired_query_bootstrap_95_ci']
        lines.append(f"| {precision} | {core_text} | {extra['mean_delta_ndcg10']:+.3f} [{extra_ci[0]:+.3f}, {extra_ci[1]:+.3f}] | {c['stsbenchmark_delta']:+.3f} | {c['mean12_delta']:+.3f} |")
    lines+=['','These contrasts compare complete data recipes at matched pair presentations. Source composition, token lengths and compute differ. Query intervals do not establish training-seed stability or full-MTEB superiority.','']
    selected=report['reserved_gate_candidate']
    confirmation=OUT/'reserved-confirmation.json'
    if selected and confirmation.exists():
        result=read(confirmation)
        lines += [f"The predeclared development gate selected **{selected}** before full reserved judgments were opened. Confirmation used this one frozen candidate and the two pinned references. Touche query1 was excluded because its judgments appeared in a recorded search preview before evaluation.",'',
            '| Reserved query subset | Candidate nDCG@10 | BGE-micro | MiniLM |','|---|---:|---:|---:|']
        for task in ('webis-touche2020','trec-covid'):
            scores=[result['models'][name]['tasks'][task]['ndcg10'] for name in ('candidate','bge_micro','minilm')]
            lines.append('| '+task+' | '+' | '.join(f'{v:.2f}' for v in scores)+' |')
        lines+=['','Paired confirmation intervals and query counts are in [reserved-confirmation.json](results/reserved-confirmation.json). No further training or selection used those judgments.','']
    else:
        lines+=['No trained candidate passed every predeclared development gate. Full new reserved judgments were left unopened. The recorded Touche query1 preview was excluded from the planned confirmation.','']
    lines+=['All artifacts remain experimental; deployment was not changed. Detailed selections, scores and intervals: [development-summary.json](results/development-summary.json).','']
    (PROFILE/'RESULTS.md').write_text('\n'.join(lines))


def observed_retrieval_worker(name):
    # FiQA/SCIDOCS were opened in the PREVIOUS campaign and are now development.
    import numpy as np
    import torch
    from transformers import AutoTokenizer
    sys.path.insert(0, str(PROJECT / 'comparisons/qwen06b'))
    sys.path.insert(0, str(PROJECT / 'retrieval-diagnostics'))
    from compare import data
    from diagnose import cached_encode, student
    from round2_metrics import evaluate
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    pool = read(OUT / 'candidate-pool.json')
    entry = pool[name]
    selection = entry['selection']
    path = Path(selection['path'])
    assert digest(path) == selection['sha256']
    precision = selection['precision']
    if precision in ('q4', 'q8'):
        from export_pair_precision import load_reference
        model = load_reference(path)
    else:
        model = student(path, 'quant' if precision == 'ternary' else 'float')
    tokenizer = AutoTokenizer.from_pretrained(PROJECT / 'tokenizer', local_files_only=True)
    identity = dict(sha256=selection['sha256'], precision=precision, cap=128,
                    encoder_sha256=digest(PROJECT / 'retrieval-diagnostics/diagnose.py'))
    result = dict(selection=selection, token_cap=128, query_instructions=False,
                  scope='Previously opened FiQA/SCIDOCS development judgments', tasks={})
    rows = {}
    destination = OUT / 'observed-retrieval'
    destination.mkdir(exist_ok=True)
    with torch.inference_mode():
        for task in ('FiQA', 'SCIDOCS'):
            dataset = data(task)
            cache = ROOT / 'work/broad-data-evaluation/vectors' / (selection['sha256'] + '-' + task + '.npy')
            vectors = cached_encode(model, tokenizer, dataset['queries'] + dataset['texts'], False, 64, identity, cache)
            n = len(dataset['queries'])
            metrics, per_query = evaluate(vectors[:n] @ vectors[n:].T, dataset)
            result['tasks'][task] = dict(metrics, source_sha256=dataset['source_sha256'])
            rows[task] = per_query
            atomic_json(destination / (name + '-rows.json'), rows)
            atomic_json(destination / (name + '-quality.json'), result)
            print(name, task, metrics['ndcg10'], flush=True)


def score():
    from export_pair_precision import export
    from score_campaign import mteb_scores
    from confirm_broad_reserved import contrast
    plan = read(PROFILE / 'runpod-job.json')
    arms = [arm['id'] for arm in plan['training']['arms']]
    OUT.mkdir(exist_ok=True)
    status('Freezing all four completed arms before external development scoring')
    if not (OUT / 'frozen.json').exists():
        run([sys.executable, str(ROOT / 'freeze_campaign.py'), '--profile', 'broad-data', '--arms', *arms], OUT / 'freeze.log')
    frozen = read(OUT / 'frozen.json')
    assert set(frozen['models']) == set(arms)
    if not (OUT / 'summary.json').exists():
        status('Scoring four frozen arms on the fixed core and 12-task development suite')
        run([sys.executable, str(ROOT / 'score_campaign.py'), '--profile', 'broad-data', '--arms', *arms,
             '--mteb', '--dibs-inner'], OUT / 'scoring.log')
    base = read(OUT / 'summary.json')
    pool = {}
    for recipe in ('repeat', 'broad'):
        folder = OUT / 'families' / recipe
        folder.mkdir(parents=True, exist_ok=True)
        family_path = folder / 'frozen.json'
        if not family_path.exists():
            models = {}
            for precision in ('float', 'ternary'):
                arm = precision + '-' + recipe
                models[precision] = dict(frozen['models'][arm], source_summary=str(OUT / 'summary.json'),
                                         source_summary_sha256=digest(OUT / 'summary.json'), source_arm=arm)
            for bits, group in ((4, 16), (8, 64)):
                name = 'q' + str(bits)
                entry = export(Path(models['float']['path']), folder / (name + '.pt'), bits, 'maxabs', group)
                models[name] = dict(entry, precision=name, layers=3, eligible_for_promotion=models['float']['eligible_for_promotion'])
            atomic_json(family_path, dict(models=models, frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                protocol_sha256=digest(PROFILE / 'evaluation-protocol.json'),
                scope='Predeclared fixed Q4 group16/Q8 group64 maxabs exports; no scale search'))
        if not (folder / 'quality/summary.json').exists():
            status('Scoring fixed precision exports for ' + recipe + ' data')
            run([sys.executable, str(ROOT / 'score_precision_family.py'), '--freeze', str(family_path),
                 '--output', str(folder / 'quality'), '--dibs-inner'], folder / 'scoring.log')
        report = read(folder / 'quality/summary.json')
        for name, result in report['models'].items():
            pool[recipe + '-' + name] = dict(result, recipe=recipe, precision=name,
                                            core_rows=str(folder / 'quality' / (name + '-rows.json')))
    parent_report = read(ROOT / 'delivery/family-results/summary.json')
    for precision in ('float', 'ternary'):
        pool['parent-' + precision] = dict(parent_report['models'][precision], recipe='parent', precision=precision,
            core_rows=str(ROOT / 'delivery/family-results' / (precision + '-rows.json')))
    atomic_json(OUT / 'candidate-pool.json', pool)
    status('Scoring previously opened FiQA/SCIDOCS development tasks; unexposed reserved query subsets remain sealed')
    def worker(name):
        target = OUT / 'observed-retrieval' / (name + '-quality.json')
        if target.exists() and set(read(target)['tasks']) == {'FiQA', 'SCIDOCS'}:
            assert read(target)['selection']['sha256'] == pool[name]['selection']['sha256']
            return
        run([sys.executable, str(Path(__file__).resolve()), '--retrieval-worker', name],
            OUT / 'observed-retrieval' / (name + '.log'))
    # Step0 selections can be SHA-identical across both recipes and the parent.
    # Score each identity once so two processes never publish the same cache.
    canonical={};aliases={}
    for name in sorted(pool):
        entry=pool[name]['selection'];key=(entry['sha256'],entry['precision'])
        if key in canonical:aliases[name]=canonical[key]
        else:canonical[key]=name
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(worker, canonical.values()))
    for name,source in aliases.items():
        result=read(OUT/'observed-retrieval'/(source+'-quality.json'))
        result=dict(result,selection=pool[name]['selection'],reused_sha_identical_result=source)
        atomic_json(OUT/'observed-retrieval'/(name+'-quality.json'),result)
        atomic_json(OUT/'observed-retrieval'/(name+'-rows.json'),read(OUT/'observed-retrieval'/(source+'-rows.json')))
    refs = read(ROOT / 'round2/quality.json')['models']
    reference = {}
    for name in ('bge_micro', 'minilm'):
        old = read(ROOT / 'delivery/reserved' / (name + '-quality.json'))
        wide = mteb_scores(ROOT / 'campaign/mteb-references' / name)
        assert len(wide) == 12
        reference[name] = dict(tasks=dict(refs[name]['tasks'], **old['tasks']),
            stsbenchmark=refs[name]['stsbenchmark_spearman'], mteb_12_mean=statistics.mean(wide.values()))
    candidates = {}
    for name, entry in pool.items():
        extra = read(OUT / 'observed-retrieval' / (name + '-quality.json'))
        tasks = dict(entry['tasks'], **extra['tasks'])
        for task in ('FiQA', 'SCIDOCS'):
            for ref in reference.values():assert tasks[task]['source_sha256'] == ref['tasks'][task]['source_sha256']
        gates = {}
        for ref_name, ref in reference.items():
            gates[ref_name] = dict(
                **{task:tasks[task]['ndcg10'] > ref['tasks'][task]['ndcg10'] for task in ('SciFact','NFCorpus','FiQA','SCIDOCS')},
                stsbenchmark=entry['stsbenchmark'] > ref['stsbenchmark'],
                mean12=entry['mteb_12_mean'] > ref['mteb_12_mean'])
        candidates[name] = dict(entry, tasks=tasks, development_gates=gates,
            passes_all_gates=all(all(v.values()) for v in gates.values()),
            four_task_mean=statistics.mean(tasks[t]['ndcg10'] for t in ('SciFact','NFCorpus','FiQA','SCIDOCS')))
    comparisons = {}
    for precision in ('float','ternary','q4','q8'):
        left, right = 'repeat-' + precision, 'broad-' + precision
        a,b = candidates[left],candidates[right]
        comparisons[precision] = dict(
            core=contrast(read(Path(a['core_rows'])), read(Path(b['core_rows'])),tasks=('SciFact','NFCorpus')),
            fiqa_scidocs=contrast(read(OUT/'observed-retrieval'/(left+'-rows.json')), read(OUT/'observed-retrieval'/(right+'-rows.json')),tasks=('FiQA','SCIDOCS')),
            four_task_mean_delta=b['four_task_mean']-a['four_task_mean'],
            stsbenchmark_delta=b['stsbenchmark']-a['stsbenchmark'], mean12_delta=b['mteb_12_mean']-a['mteb_12_mean'])
    passing = [name for name, result in candidates.items() if not name.startswith('parent-') and result['passes_all_gates']]
    selected = min(passing, key=lambda n:(-candidates[n]['four_task_mean'],candidates[n]['selection']['sha256'])) if passing else None
    report = dict(scope='Fixed observed development tasks; recipe comparison at matched pair presentations, not equal FLOPs or full MTEB',
        resource=read(WORK/'pod.json'), models=candidates, references=reference, broad_minus_repeat=comparisons,
        reserved_gate_candidate=selected, new_reserved_full_qrels_read=False,
        reserved_preview_exposure=read(PROFILE/'reserved-preview-exposure.json'),
        report_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),protocol_sha256=digest(PROFILE/'evaluation-protocol.json'))
    atomic_json(OUT/'development-summary.json', report)
    atomic_json(OUT/'development-gate.json',dict(passed=selected is not None,candidate=selected,
        summary_sha256=digest(OUT/'development-summary.json'),new_reserved_full_qrels_read=False,
        preview_excluded_query_ids={'webis-touche2020':['1']}))
    if selected:
        status('Development gate passed; confirming one frozen candidate on unexposed reserved query subsets')
        run([sys.executable,str(ROOT/'confirm_broad_reserved.py')],OUT/'reserved-confirmation.log')
    write_report(report)
    status('Comparison and report complete; paid pod deleted; reserved gate ' + ('passed and confirmed' if selected else 'did not pass'))
    atomic_json(PROFILE/'active-work.json',dict(status='Training and evaluation complete; see RESULTS.md',paid_pod_active=False,
        approved_cap_usd=10,estimated_total_usd=report['resource']['estimated_total_usd'],
        updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),results=str(PROFILE/'RESULTS.md')))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--inner',action='store_true')
    parser.add_argument('--retrieval-worker')
    args=parser.parse_args()
    if args.retrieval_worker:
        observed_retrieval_worker(args.retrieval_worker)
        return
    if args.inner:
        score()
        return
    while True:
        state=read(WORK/'pod.json')
        if state.get('deletion_verified_http_status')==404:
            break
        if time.time()>state['deadline']+600:raise RuntimeError('Owned pod deletion verification overdue')
        time.sleep(20)
    atomic_json(PROFILE/'active-work.json',dict(status='All paid training closed; local evaluation queued',paid_pod_active=False,
        approved_cap_usd=10,estimated_total_usd=state['estimated_total_usd'],updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    plan=read(PROFILE/'runpod-job.json')
    for arm in plan['training']['arms']:
        if not (WORK/'backups'/('broad-data-'+arm['id'])/'completed.json').exists():
            raise RuntimeError('Paid job closed with an incomplete arm; do not expose development labels')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',
             TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',DIBS_AGENT='broad_quality')
    command=['dibs','run','cpu','network','--label','Frozen broad-data quality comparison','--eta','3h','--',
             sys.executable,str(Path(__file__).resolve()),'--inner']
    try:
        while True:
            output=[]
            process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,env=env)
            for line in process.stdout:
                print(line,end='',flush=True);output.append(line)
            code=process.wait()
            if code==0:break
            text=''.join(output)
            if 'WAIT' not in text:raise subprocess.CalledProcessError(code,command)
            wait=['dibs','wait','--pressure' if '--pressure' in text else '--measure']
            subprocess.run(wait,check=True,env=env)
            # Retry the exact resource command after Dibs reports FREE.
    finally:
        subprocess.run(['dibs','disk','release'],check=True)


if __name__=='__main__':main()
