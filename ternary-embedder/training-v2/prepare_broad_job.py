"""Prepare an independently bounded four-arm job; never grant spending approval."""
import collections
import copy
import json
import shutil
from pathlib import Path
from broad_data_fetch import atomic,digest
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent


def main():
    ready=ROOT/'work/broad-data/ready';audit=json.loads((ready/'audit.json').read_text())
    assert not (ROOT/'work/broad-data-run/pod.json').exists()
    for filename,expected in audit['output_sha256'].items():assert digest(ready/filename)==expected
    counts=collections.Counter()
    for line in (ready/'legacy-anchor.jsonl').open():counts[json.loads(line)['source']]+=1
    n=audit['train_rows'];total=sum(counts.values());quotas={s:n*c//total for s,c in counts.items()}
    for s in sorted(counts,key=lambda s:(-(n*counts[s]%total),s))[:n-sum(quotas.values())]:quotas[s]+=1
    control=ready/'train-repeat.jsonl';position=collections.Counter();written=0
    with control.open('w') as output:
        for line in (ready/'legacy-anchor.jsonl').open():
            source=json.loads(line)['source'];copies=quotas[source]//counts[source]+(position[source]<quotas[source]%counts[source]);position[source]+=1
            for _ in range(copies):output.write(line);written+=1
    assert written==n
    atomic(ready/'repeat-audit.json',dict(rows=written,unique_source_pairs=total,source_presentations=quotas,
                source_sha256=digest(ready/'legacy-anchor.jsonl'),sha256=digest(control),
                policy='Same total pair presentations as broad data, replaying only retained old training positives; source counts preserve the old mixture to rounding. This compares whole data recipes, not isolated novelty at matched domain weights.'))
    family=PROJECT/'models/campaign-20261025';frozen=json.loads((family/'frozen.json').read_text());starts=ROOT/'work/broad-data/starts';starts.mkdir(exist_ok=True)
    for name in ('ternary','float'):
        entry=frozen['models'][name];source=Path(entry['path']);assert digest(source)==entry['sha256']
        target=starts/(name+'.pt');shutil.copy2(source,target);assert digest(target)==entry['sha256']
    plan=copy.deepcopy(json.loads((ROOT/'human-mix/runpod-job.json').read_text()))
    for key in ('campaign_cap_usd','resource_change','parallel_campaign'):plan.pop(key,None)
    plan.update(profile='broad-data',status='Prepared only; a NEW explicit spending approval is required. Previous $10 campaign is closed.',
                resource='One RTX 4090 24GB RunPod Secure Cloud pod, at least32GB host RAM',minimum_host_ram_gb=32,
                listed_rate_usd_per_hour=.89,maximum_gpu_rate_usd_per_hour=.90,maximum_elapsed_hours=6,
                expected_elapsed_hours=[4,5.5],expected_total_usd=[3.61,4.96],proposed_total_cap_usd=6,
                disk_gb=80,retained_paid_volumes=False,work_duration_minutes=320,work_deadline_minutes=335,watchdog_delete_minutes=350,
                package_transfer_timeout_seconds=1800,independent_of_closed_campaign=True)
    inputs={key:dict(path=str(path.relative_to(PROJECT)),sha256=digest(path)) for key,path in
            [('train',ready/'train.jsonl'),('train_repeat',control),('dev',ready/'dev.jsonl')]}
    arms=[]
    for precision in ('ternary','float'):
        for mixture in ('repeat','broad'):
            arms.append(dict(id=precision+'-'+mixture,checkpoint=str((starts/(precision+'.pt')).relative_to(PROJECT)),layers=3,batch=1024,micro_batch=128,lr=1e-5,
                             use_negatives=False,data_key='train_repeat' if mixture=='repeat' else 'train'))
    plan['data_inputs']=inputs
    plan['training']=dict(arms=arms,epochs_each=1,seed=71,eval_every=500,save_every=100,source_floor=2,
                teacher_targets=False,auxiliary_nli=False,validation_negatives=False,train_rows=n,control_rows=n,dev_rows=audit['validation_rows'],
                comparison='Two precisions x broad positives versus repeated old positives. Same fixed3-layer starts, batch, optimizer settings, seed, validation and pair presentations; source composition and token lengths differ and will be logged.',
                selection='Highest held-source macro nDCG with every source within2 points of initialization; unchanged step0 remains eligible. Freeze all four arms before external development scoring.',
                external_protocol='FiQA, SCIDOCS and prior fixed12 tasks are observed development. New Touche/TREC-COVID labels stay sealed until one candidate passes the predeclared development gate. No full-MTEB or seed-replication claim.')
    plan['data_audit_files']=[str(p.relative_to(PROJECT)) for p in [ready/'audit.json',ready/'repeat-audit.json',ROOT/'broad-data/sources.lock.json',ROOT/'broad-data/evaluation.lock.json',ROOT/'broad-data/quarantine.json']]
    assert (plan['maximum_gpu_rate_usd_per_hour']+plan['disk_gb']*.1/720)*plan['maximum_elapsed_hours']<plan['proposed_total_cap_usd']
    atomic(ROOT/'broad-data/runpod-job.json',plan)
    print('Prepared four arms,',n,'pair presentations each. New $6 approval required; nothing allocated.',flush=True)


if __name__=='__main__':main()
