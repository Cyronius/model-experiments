# Final campaign findings

Updated 2026-10-25T18:45:30.040209+00:00. Training is complete. Estimated spend **$9.5843 of $10**; all nine owned RunPod allocations were deleted with GET404 evidence.

Frozen Q8's FiQA/SCIDOCS retrieval mean is **22.25**, versus **20.52** for BGE-micro and **29.17** for MiniLM. The broader goal of beating both references remains unmet.

The strongest ternary model improved core retrieval from **42.128 to45.227**. Final float/Q4/Q8 reach **47.022/46.669/47.475**, versus BGE-micro46.214 and MiniLM46.920. Q8 exceeds both reference point estimates on each development core task, STSBenchmark, and the fixed12-task mean (**72.023**). This is not full-MTEB superiority; selection used many development trials and full-pipeline seed replication remains outstanding.

## Reserved confirmation

Q8 was frozen by physical SHA before opening previously reserved FiQA/SCIDOCS judgments. No training or selection followed. These labels are now opened. Public historical pretraining exposure remains unknown.

| Model | FiQA | SCIDOCS | Mean nDCG@10 |
|---|---:|---:|---:|
| q8 | 26.805 | 17.697 | 22.251 |
| bge_micro | 23.523 | 17.521 | 20.522 |
| minilm | 36.626 | 21.716 | 29.171 |

## What the experiments support

Fresh science data outperformed a matched-size repeat-data control more clearly for float (+0.808 retrieval points, query95% interval[0.305,1.331]) than ternary (+0.265, interval[-0.721,1.192]). The best final models retain pretrained behavior through checkpoint blending.

Simply cloning and appending a layer hurt. Recovering a properly pretrained third-layer checkpoint helped, but its different pretraining history prevents an equal-compute depth claim. Unjudged BM25 negatives hurt both precisions. Human NLI classification/ranking restored broader behavior at a retrieval cost. More continuation, learned scales, and output/query projections did not beat the selected ternary blend. Learned-scale two-layer gains were uncertain. Lower weight-reconstruction error did not imply better Q4 retrieval.

The ternary model remains useful and fast, but still trails both compact references on development retrieval. Better judged relevance data, protected pretrained behavior, and a new untouched confirmation suite are more defensible next steps than assuming another layer or more identical epochs will solve the gap. Further paid work needs a new bounded authorization.

The four experimental artifacts have three layers and17,241,984 parameters. Native final ternary quality was verified; initial/final CLEAN timings include8.61ms CPU and1.79ms NPU at128tokens. Packed Q4/Q8 formats are real storage artifacts, while optimized packed execution is unimplemented.

See [complete results](../delivery/RESULTS.md), [experiment interpretation](../delivery/EXPERIMENT_NOTES.md), [all development comparisons](COMPARISON.md), [billing](../delivery/cost-closeout.json), and [final model files](../../models/campaign-20261025/README.md).
