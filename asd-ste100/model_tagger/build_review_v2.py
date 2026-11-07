"""Freeze an expanded assistant-reviewed benchmark before scoring either model."""
import json
from pathlib import Path
import re
from tagger import ROOT
from data_revision_v4 import instantiate

BENCH=ROOT/'model_tagger/benchmarks'

def main():
    rows=[json.loads(line) for line in (BENCH/'software-review-v1.jsonl').read_text().splitlines()]
    # Preserve original benchmark unchanged. V2 explicitly distinguishes optional
    # article/style preferences from mandatory corrections; no predictions used.
    optional={19,20,21,22,23,24,25,29,30}
    for i,row in enumerate(rows,1):
        row['supersedes_v1_id']=row['id'];row['id']='software-review-v3-'+str(i).zfill(3)
        if i in optional:
            row['category']='style_optional';row['acceptable_outputs']=list(dict.fromkeys([row['source'],*row['acceptable_outputs']]))
            row['reference_revision']='Accept unchanged telegraphic/uncountable technical wording as well as the explicit-article preference. Not a mandatory meaning repair.'
    targets={
      'purpose':['Open `analysis_log` to check the result.','Read `checklist` to find the next step.','Copy `snapshot` to keep a copy.','Use `settings` to start the test.'],
      'before':['Check `signature` before you delete the file.','Read `notes` before you start the update.','Copy `report` before you stop the worker.','Close `socket` before you open the next file.'],
      'after':['Check `digest` after you save the file.','Read `status` after you close the pipe.','Open `report` after you stop the test.','Use `copy` after you delete the file.'],
      'conditional':['If `verified` is false, do not start the transfer.','If `result` is missing, stop the test.','If `ready` is true, read the file.','If `count` is missing, keep the log.'],
      'because':['Do not start the test because `verified` is false.','Keep the file because `ready` is false.','Stop the transfer because `result` is missing.','Read the report because `status` is false.'],
      'now':['Read `report` at this time.','Check `manifest` at this time.','Close `socket` at this time.','Copy `snapshot` at this time.']}
    swaps={'purpose':(' to ',' in order to '),'before':(' before ',' prior to '),'after':(' after ',' subsequent to '),'conditional':('If ','In the event that '),'because':(' because ',' due to the fact that '),'now':(' now.',' at this point in time.')}
    def add(category,source,accepted,family):
        index=len(rows)+1
        source=instantiate(source,'eval_quality4_'+str(index))
        accepted=[instantiate(text,'eval_quality4_'+str(index)) for text in accepted]
        literals=re.findall(r'`[^`]+`',source)
        rows.append(dict(id=f'software-review-v3-{index:03}',category=category,mode='PROC',source=source,acceptable_outputs=accepted,
                         must_preserve=list(dict.fromkeys(literals)),review_status='assistant_reviewed_owner_pending',family=family,
                         rationale='Reviewed text preserves literal values, referents, negation, modal strength and temporal/conditional relations. Phrase simplification is an explicit requested style normalization.' if category=='repair' else 'Keep the original facts, scope and literal surfaces; do not add an unrequested assertion or change modal/temporal force.'))
    for family,frames in targets.items():
        old,new=swaps[family]
        for target in frames:add('repair',target.replace(old,new,1),[target],family)
    clean=[
      'Do not delete `audit.log` before you save the backup.',
      'You should stop `worker` if `status` is false.',
      'You must stop `worker` if `status` is false.',
      'You may delete `cache` after the update.',
      'You might need `backup` after the update.',
      'Use `prior to` as the literal label.',
      'Print "in order to" in `status_log`.',
      'If `ready` is false, keep `backup`. Otherwise, delete `backup`.',
      'Delete `cache` only after `archive` is complete.',
      'Do not change `mode` from 275 to 572.',
      'Read `src` before `dst`.',
      'Read `dst` before `src`.',
      '`get_status()` returns false when `ready` is false.',
      'The file is empty. Do not delete `path`.',
      'Keep `release.zip` unless `verified` is true.',
      'Run `sync()` without `--force`.',
      'Stop `worker` after 275 ms.',
      'Set `file_mode` to "read-only-v4".',
      'Use nfs://review-v4-host/releases/test9.',
      'Write a C++ function for `summary.csv`.',
      'Use ``print(`ready-v4`)``.',
      'Keep **do not delete** in `warning`.',
      'Set `minimum` to -275 kPa.',
      'Do not replace "must" with "should" in `policy`.']
    for source in clean:add('clean',source,[source],'meaning_control')
    path=BENCH/'software-review-v3.jsonl'
    if path.exists():raise ValueError('benchmark already frozen')
    path.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows))
    print('Frozen benchmark cases:',len(rows))

if __name__=='__main__':main()
