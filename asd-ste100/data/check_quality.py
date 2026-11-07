"""Scan or filter manifest-listed corpora using versioned deterministic quality gates."""
from __future__ import annotations
import argparse
from collections import Counter
import gzip
import hashlib
import heapq
import io
import json
from pathlib import Path
import sys

from quality_gates import inspect_pair
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tokenizer'))
from prepare import input_paths, records
from ste_tokenizer import sha256


def run(pairs, output, filtering=False, shard_size=25000):
    fingerprints = {n:sha256(Path(__file__).parent/n) for n in ('quality_gates.py','check_quality.py')}
    paths = list(input_paths(pairs))
    original = json.loads((pairs / 'summary.json').read_text())
    if output.exists() and any(output.iterdir()):
        raise ValueError('use a new or empty output directory')
    output.mkdir(parents=True, exist_ok=True)
    scanned, excluded, flagged = 0, 0, 0
    by_rule, exclusions, by_kind, by_split, by_bucket, reviews = Counter(), Counter(), Counter(), Counter(), Counter(), Counter()
    streams, written, samples = {}, [], {}
    def close(split):
        text, raw = streams.pop(split)
        text.close(); raw.close()
    with gzip.open(output / 'findings.jsonl.gz', 'wt', encoding='utf-8') as findings, (output/'rejected.jsonl').open('w',encoding='utf-8') as failures:
        try:
            for path, _ in paths:
                for row in records(path):
                    scanned += 1
                    issues = inspect_pair(row)
                    hard = any(i['action'] == 'exclude' for i in issues)
                    if issues:
                        flagged += 1
                        entry = {k: row[k] for k in ('id','src','ste','mode','kind','noise','split','meaning_review')}
                        entry.update(issues=issues, excluded=hard)
                        findings.write(json.dumps(entry, ensure_ascii=False) + '\n')
                        for rule in {i['rule'] for i in issues}:
                            by_rule[rule] += 1
                            if any(i['rule']==rule and i['action']=='exclude' for i in issues):
                                exclusions[rule] += 1
                            heap = samples.setdefault(rule, [])
                            priority = int(hashlib.sha256((rule+'\0'+row['id']).encode()).hexdigest(),16)
                            item = (-priority, row['id'], entry)
                            if len(heap) < 8: heapq.heappush(heap,item)
                            elif item[:2] > heap[0][:2]: heapq.heapreplace(heap,item)
                    if hard:
                        failures.write(json.dumps({'id':row['id'],'split':row['split'],'kind':row['kind'],
                            'reason':'deterministic_quality','quality_rules':sorted({i['rule'] for i in issues if i['action']=='exclude'})})+'\n')
                        excluded += 1
                        continue
                    by_kind[row['kind']] += 1; by_split[row['split']] += 1
                    by_bucket[str(row['bucket'])] += 1; reviews[row['meaning_review']] += 1
                    if filtering:
                        split = row['split']
                        if (by_split[split] - 1) % shard_size == 0:
                            if split in streams: close(split)
                            target = output / 'shards' / f'{split}-{(by_split[split]-1)//shard_size:05d}.jsonl.gz'
                            target.parent.mkdir(exist_ok=True)
                            raw = target.open('wb')
                            streams[split] = io.TextIOWrapper(gzip.GzipFile(filename='',fileobj=raw,mode='wb',mtime=0,compresslevel=1),encoding='utf-8',newline='\n'),raw
                            written.append(target)
                        streams[split][0].write(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n')
                    if scanned % 50000 == 0: print(f'scanned {scanned:,}; excluded {excluded:,}; flagged {flagged:,}',flush=True)
        finally:
            for split in list(streams): close(split)
    if scanned != original['pairs']:
        raise ValueError('scanned count differs from parent corpus summary')
    sample_rows = [dict(entry,example_rule=rule) for rule,heap in sorted(samples.items()) for _,_,entry in sorted(heap,key=lambda x: -x[0])]
    (output/'examples.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in sample_rows))
    if fingerprints != {n:sha256(Path(__file__).parent/n) for n in fingerprints}:
        raise RuntimeError('quality code changed during run; discard this incomplete output')
    report = {'scanned':scanned,'excluded':excluded,'kept':scanned-excluded,'flagged_including_excluded':flagged,
              'rule_matches':dict(by_rule),'exclusion_rules':dict(exclusions),'kept_kinds':dict(by_kind),
              'kept_splits':dict(by_split),'kept_meaning_review':dict(reviews),
              'parent_summary_sha256':sha256(pairs/'summary.json'),
              'parent_shards':{str(p):h for p,h in paths},
              'code_sha256':fingerprints,
              'filtered':filtering,'limits':['Rule match counts overlap; excluded is the unique removed count.',
                                          'Review flags alone do not remove rows. This is not a meaning judgment.',
                                          'Kept rows retain their original IDs, splits, groups, attribution and review labels.']}
    (output/'quality-report.json').write_text(json.dumps(report,indent=2)+'\n')
    if filtering:
        summary = dict(original)
        summary.update(scanned=scanned,pairs=scanned-excluded,splits=dict(by_split),buckets=dict(by_bucket),
                       rejected={'deterministic_quality':excluded} if excluded else {},inputs={str(p):h for p,h in paths},
                       shards=[{'path':p.relative_to(output).as_posix(),'sha256':sha256(p),'bytes':p.stat().st_size} for p in written],
                       quality_report_sha256=sha256(output/'quality-report.json'),parent_summary_sha256=report['parent_summary_sha256'])
        # Old export fields describe upstream strict tokenization, not this filtering pass.
        for key in ('bucket_transitions','limit','workers','code_sha256'):
            summary.pop(key,None)
        summary['code_sha256'] = report['code_sha256']
        (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return report


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('scan','filter'))
    parser.add_argument('--pairs',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(run(args.pairs,args.out,args.command=='filter'),indent=2))
