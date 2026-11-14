"""Summarize the frozen human-feature-scale probe against exact source models."""
import datetime
import json
from pathlib import Path
from pair_training_state import atomic_json, digest
ROOT = Path(__file__).resolve().parent

def main():
    out = ROOT / 'human-similarity-head/results'
    summary = json.loads((out / 'summary.json').read_text())
    parents = json.loads((ROOT / 'broad-data/results/development-summary.json').read_text())['models']
    plan = json.loads((out.parent / 'plan.json').read_text())
    results = {}
    for name, model in summary['models'].items():
        source = parents[name]
        assert plan['source_models'][name]['sha256'] == source['selection']['sha256']
        assert len(model['mteb_12']) == 12
        fit = json.loads((out / (name + '-fit.json')).read_text())
        assert fit['selection']['sha256'] == model['selection']['sha256']
        assert fit['full_encoder_unchanged'] and fit['materialization_max_abs_error'] < 1e-5
        results[name] = dict(core=model['core_retrieval'], sts=model['stsbenchmark'],
            mean12=model['mteb_12_mean'], selected_step=fit['selected_step'],
            source_sha256=source['selection']['sha256'],
            core_delta=model['core_retrieval']-source['core_retrieval'],
            sts_delta=model['stsbenchmark']-source['stsbenchmark'],
            mean12_delta=model['mteb_12_mean']-source['mteb_12_mean'],
            task_deltas={task: value-source['mteb_12'][task] for task,value in model['mteb_12'].items()})
    report = dict(scope='Exploratory observed-development results; no reserved labels opened and no paid allocation.',
        summary_sha256=digest(out/'summary.json'), models=results, deployment_changed=False,
        conclusion='No variant improves core retrieval over its own source. Do not promote; skip additional FiQA/SCIDOCS scoring after this failed requirement.')
    assert all(m['core_delta'] < 0 for m in results.values())
    atomic_json(out/'diagnostic-summary.json', report)
    table = '\n'.join('| '+name+' | '+ ' | '.join(f'{v[k]:.3f}' for k in ('core','sts','mean12','core_delta','sts_delta','mean12_delta'))+' |' for name,v in results.items())
    analysis = ROOT/'broad-data/ANALYSIS.md'
    text=analysis.read_text()
    old='Core retrieval declined for all four adjusted models relative to their own starting checkpoints. Wider evaluation is being completed after an offline dataset-cache failure; the training and materialization checks completed successfully. This limited output-only experiment cannot establish whether full-encoder human-supervised training would work.'
    new='Core retrieval declined for all four adjusted models relative to their own starting checkpoints. The complete fixed 12-task evaluation and materialization checks passed. Additional FiQA/SCIDOCS scoring was skipped because every variant already failed retrieval retention. This limited output-only experiment cannot establish whether full-encoder human-supervised training would work.\n\n| Adjusted source | Core | STS | Mean12 | Core delta | STS delta | Mean12 delta |\n|---|---:|---:|---:|---:|---:|---:|\n'+table+'\n\n[Full source-matched diagnostics](../human-similarity-head/results/diagnostic-summary.json). Deltas are relative to each named source, not all relative to the original parent.'
    if old in text: analysis.write_text(text.replace(old,new))
    state=ROOT/'broad-data/session-notes.json'
    data=json.loads(state.read_text());data.update(updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),current='Paid campaign and free retention/human-similarity diagnostics complete; previous model family retained',paid_pod_active=False,next=['Prepare a new controlled data/objective experiment before requesting a fresh bounded paid-job approval.','No remaining paid budget is standing authorization.','Keep new reserved labels unopened for exploratory development.'])
    atomic_json(state,data)
    print(json.dumps(results,indent=2))

if __name__ == '__main__': main()
