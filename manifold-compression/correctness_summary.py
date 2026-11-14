"""Paired correctness and behavior comparisons for the completed pilot."""
import json
from pathlib import Path

from scale_probe import save
from summarize_scale_probe import compare_generation


def paired(before,after):
    left={c['id']:c for c in before['cases']};right={c['id']:c for c in after['cases']}
    if left.keys()!=right.keys():raise ValueError('Mismatched evaluation cases')
    scored=[k for k,c in left.items() if c['grade'] is not None]
    wins=[k for k in scored if not left[k]['grade']['correct'] and right[k]['grade']['correct']]
    losses=[k for k in scored if left[k]['grade']['correct'] and not right[k]['grade']['correct']]
    original=sum(left[k]['grade']['correct'] for k in scored)
    return {'scored':len(scored),'baseline_correct':original,
        'candidate_correct':sum(right[k]['grade']['correct'] for k in scored),
        'gained':wins,'lost':losses,'loss_fraction_of_baseline_correct':len(losses)/original if original else None,
        'baseline_behavior':before['summary'],'candidate_behavior':after['summary'],
        'new_loop_cases':[k for k in left if not left[k]['combined_loop_flag'] and right[k]['combined_loop_flag']],
        'new_unparseable':[k for k in scored if left[k]['grade']['parsed'] and not right[k]['grade']['parsed']]}


def summarize(folder):
    def read(name):return json.loads((folder/name).read_text())
    runs=read('result.json')
    fresh={name:read(name+'-test.json') for name in ['stock','full_width','unhealed_75','healed-selected']}
    legacy={name:read(name+'-legacy72.json') for name in fresh}
    pairs={'full_width_vs_stock':('stock','full_width'),
           'healed_vs_unhealed':('unhealed_75','healed-selected'),
           'healed_vs_stock_reference':('stock','healed-selected')}
    result={'status':runs['status'],'selected_step':runs['selected_step'],'updates_completed':len(runs['steps']),
        'fresh_test':{name:paired(fresh[a],fresh[b]) for name,(a,b) in pairs.items()},
        'legacy72':{name:compare_generation(legacy[a],legacy[b]) for name,(a,b) in pairs.items()},
        'interpretation':'Fresh correctness is primary; stock is a reference, never a training target. Legacy checks include formatting. Review unparseable answers and long-form generations. This small structured screen cannot establish general parity.'}
    save(result,folder/'comparison.json')
    return result


if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('folder',type=Path)
    args=ap.parse_args();print(json.dumps(summarize(args.folder),indent=2))
