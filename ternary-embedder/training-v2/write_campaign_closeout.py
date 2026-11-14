"""Refresh mutable campaign status after all frozen verification is complete."""
import datetime
import json
from pathlib import Path
from pair_training_state import atomic_json,digest

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parent


def main():
    result=json.loads((ROOT/'delivery/summary.json').read_text())
    reserved=result['reserved_confirmation']
    cost=result['cost']
    assert cost['active_owned_pods']==0 and cost['estimated_spent_usd']<=10
    assert result['final_latency_clean'] and result['reserved_judgments_opened']
    now=datetime.datetime.now(datetime.timezone.utc).isoformat()
    means=reserved['models']
    held_sentence=(f"Frozen Q8's FiQA/SCIDOCS retrieval mean is **{means['q8']['mean_retrieval']:.2f}**, versus **{means['bge_micro']['mean_retrieval']:.2f}** for BGE-micro and **{means['minilm']['mean_retrieval']:.2f}** for MiniLM.")
    outcome=('The broader goal of beating both references remains unmet.' if not all(reserved['beats_mean'].values()) else 'These are two fixed confirmation datasets, not proof of general superiority.')
    status=dict(updated_utc=now,status='complete',paid_pod_active=False,active_pods=[],
                user_authorization='Cumulative $10 experiment campaign; retain ternary and independent float/Q4/Q8.',
                approved_campaign_usd=10,actual_estimated_closed_jobs_usd=cost['estimated_spent_usd'],
                committed_upper_bound_usd=cost['estimated_spent_usd'],remaining_budget_usd=10-cost['estimated_spent_usd'],
                no_further_paid_jobs_planned=True,completed_owned_allocations=9,completed_paid_training_arms=34,
                completed_profiles=['depth3','batch1024','nli-aux','hardneg','science-float','science-ternary','scales','human-mix'],
                final_family_path=str(PROJECT/'models/campaign-20261025'),final_report=str(ROOT/'delivery/RESULTS.md'),
                final_summary_sha256=digest(ROOT/'delivery/summary.json'),reserved_judgments_opened=True,
                reserved_confirmation_path=str(ROOT/'delivery/reserved-confirmation.json'),
                reserved_beats_each_task=reserved['beats_each_task'],reserved_beats_mean=reserved['beats_mean'],
                no_training_or_selection_after_reserved_labels=True,default_deployment_changed=False,
                native_ternary_quality='Verified final checkpoint vectors at caps32/128; full-corpus native core45.1637 vs CPU45.2270.',
                final_latency='Initial and final Dibs CLEAN; same pinned two-core CPU, single actual text per native dispatch.',
                role_projection='Four candidates selected identity/null; no optional head in final family. Both query-head tests passed.',
                checkpoint_recovery='Periodic SHA-verified off-pod complete-state backups; scales continuation resumed step68 and finished.',
                scope='Fixed12 development tasks and two frozen-candidate retrieval confirmations, not full MTEB or seed replication.',
                disk_claim_release_pending=True)
    atomic_json(ROOT/'campaign/active-work.json',status)
    atomic_json(ROOT/'campaign/pending-role-work.json',dict(status='Complete; superseded by campaign/active-work.json and delivery/RESULTS.md',
                updated_utc=now,role_candidates=4,selected_identity_null=4,query_head_tests_passed=2,
                final_family_has_query_head=False,paid_pod_active=False,waits_resolved=True))
    lines=['# Final campaign findings','',f"Updated {now}. Training is complete. Estimated spend **${cost['estimated_spent_usd']:.4f} of $10**; all nine owned RunPod allocations were deleted with GET404 evidence.",'',held_sentence+' '+outcome,'',
           'The strongest ternary model improved core retrieval from **42.128 to45.227**. Final float/Q4/Q8 reach **47.022/46.669/47.475**, versus BGE-micro46.214 and MiniLM46.920. Q8 exceeds both reference point estimates on each development core task, STSBenchmark, and the fixed12-task mean (**72.023**). This is not full-MTEB superiority; selection used many development trials and full-pipeline seed replication remains outstanding.','',
           '## Reserved confirmation','',
           'Q8 was frozen by physical SHA before opening previously reserved FiQA/SCIDOCS judgments. No training or selection followed. These labels are now opened. Public historical pretraining exposure remains unknown.','',
           '| Model | FiQA | SCIDOCS | Mean nDCG@10 |','|---|---:|---:|---:|']
    for name in ('q8','bge_micro','minilm'):
        m=reserved['models'][name]
        lines.append(f"| {name} | {m['tasks']['FiQA']['ndcg10']:.3f} | {m['tasks']['SCIDOCS']['ndcg10']:.3f} | {m['mean_retrieval']:.3f} |")
    lines += ['', '## What the experiments support','',
              'Fresh science data outperformed a matched-size repeat-data control more clearly for float (+0.808 retrieval points, query95% interval[0.305,1.331]) than ternary (+0.265, interval[-0.721,1.192]). The best final models retain pretrained behavior through checkpoint blending.','',
              'Simply cloning and appending a layer hurt. Recovering a properly pretrained third-layer checkpoint helped, but its different pretraining history prevents an equal-compute depth claim. Unjudged BM25 negatives hurt both precisions. Human NLI classification/ranking restored broader behavior at a retrieval cost. More continuation, learned scales, and output/query projections did not beat the selected ternary blend. Learned-scale two-layer gains were uncertain. Lower weight-reconstruction error did not imply better Q4 retrieval.','',
              'The ternary model remains useful and fast, but still trails both compact references on development retrieval. Better judged relevance data, protected pretrained behavior, and a new untouched confirmation suite are more defensible next steps than assuming another layer or more identical epochs will solve the gap. Further paid work needs a new bounded authorization.','',
              'The four experimental artifacts have three layers and17,241,984 parameters. Native final ternary quality was verified; initial/final CLEAN timings include8.61ms CPU and1.79ms NPU at128tokens. Packed Q4/Q8 formats are real storage artifacts, while optimized packed execution is unimplemented.','',
              'See [complete results](../delivery/RESULTS.md), [experiment interpretation](../delivery/EXPERIMENT_NOTES.md), [all development comparisons](COMPARISON.md), [billing](../delivery/cost-closeout.json), and [final model files](../../models/campaign-20261025/README.md).','']
    (ROOT/'campaign/FINDINGS.md').write_text('\n'.join(lines))
    status_path=ROOT/'delivery/CURRENT_STATUS.md'
    text=status_path.read_text()
    replacement=(f"The final Q8 weights were frozen before opening the FiQA and SCIDOCS judgments. Their confirmation retrieval mean is **{means['q8']['mean_retrieval']:.2f}**, versus **{means['bge_micro']['mean_retrieval']:.2f}** for BGE-micro and **{means['minilm']['mean_retrieval']:.2f}** for MiniLM. No further training or model selection used these judgments. See [full results](RESULTS.md) for both tasks and paired intervals.")
    text=text.replace('The final Q8 weights are frozen. A confirmation on previously reserved FiQA and SCIDOCS judgments is pending. No further training or model selection will use those judgments.',replacement)
    text=text.replace('The largest remaining gap is to Qwen 0.6B.', 'Qwen 0.6B retains a 4.14-point advantage on the development retrieval mean.')
    status_path.write_text(text)
    for path,marker in ((ROOT/'README.md','\n\nHistorical starting direction'),(PROJECT/'README.md','\nThe entries below record earlier experiments;')):
        text=path.read_text()
        if held_sentence not in text:
            text=text.replace(marker,'\n\n'+held_sentence+' '+outcome+'\n'+marker,1)
        path.write_text(text)
    print('Campaign mutable status closed; release the local Dibs disk claim next.',flush=True)


if __name__=='__main__':
    main()
