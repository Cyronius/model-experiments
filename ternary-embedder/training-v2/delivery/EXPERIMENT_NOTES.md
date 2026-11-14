# What improved quality

The campaign supports shipping separate ternary and higher-precision artifacts. The strongest ternary model reaches **45.23** on the two-task retrieval mean, compared with **42.13** for the original. The final Q8 artifact reaches **47.48**. Qwen 0.6B remains ahead at **51.62** under the same retrieval protocol.

These conclusions come from development experiments. The reserved FiQA/SCIDOCS confirmation must be reported separately, and must not influence further model selection in this campaign.

The completed frozen-Q8 confirmation did not meet the broader MiniLM target. Q8's FiQA/SCIDOCS mean was **22.25**, versus **20.52 BGE-micro** and **29.17 MiniLM**. Q8 minus BGE-micro was **+1.73 points**, with paired query 95% interval **[1.07, 2.38]**; Q8 minus MiniLM was **-6.92**, interval **[-7.94, -5.92]**. Q8's advantage over BGE-micro on SCIDOCS alone was uncertain. MiniLM won both tasks, including a **9.82-point** lead on finance. This supports seeking broader relevance coverage in a future campaign, rather than treating the development-suite win as general superiority. These judgments are now opened; no model was tuned or selected afterward.

## Depth and data

Simply copying and appending another transformer layer hurt retrieval. Recovering the project's genuinely pretrained three-layer checkpoint was more useful. Its pretraining history differs from the two-layer model, so this is not an equal-compute demonstration that a third layer alone caused the improvement.

Fresh science relevance data was the clearest controlled data improvement. Against a matched-size control that repeated existing science data, it improved float retrieval by **0.81 points**, with a paired query-bootstrap 95% interval of **[0.30, 1.33]**. The corresponding ternary gain was **0.26 points**, with interval **[-0.72, 1.19]**. The float result supports better domain coverage; the ternary evidence is less conclusive. These intervals measure query sampling uncertainty, not full training-seed variation.

Blending checkpoints retained useful general-language behavior while preserving most of the retrieval gain. The final float backbone blends the earlier retrieval-preserving checkpoint with the human-data continuation. The selected ternary backbone uses an earlier three-layer blend. They are separate selected models, rather than the same checkpoint quantized into every format. Q4 and Q8 are quantized from the selected float backbone.

## Changes that did not solve the ternary gap

- A larger contrastive batch helped retrieval but weakened the broader task average.
- Unjudged BM25 hard negatives reduced retrieval in both precisions. They may contain valid answers; treating them as known negatives was not beneficial here.
- Adding human NLI supervision recovered broader language-task quality, but it did not produce the strongest retrieval model. A logical contradiction is not equivalent to an independently judged irrelevant search result.
- Learned ternary row scales produced only a small, uncertain improvement in the matched two-layer comparison: **+0.25 retrieval points**, with interval **[-0.13, 0.64]**. They did not beat the selected three-layer blend.
- Frozen output projections and separate query projections did not improve the selected ternary backbone. The final artifacts do not contain those experimental heads.
- Minimizing Q4 weight reconstruction error did not yield the best embedding quality. The final Q4 format uses the better development setting, with group size 16; Q8 uses group size 64.

## Implication for the next campaign

The architecture can improve, but the current results do not justify claiming that another layer or more identical training will close the remaining ternary gap. The most defensible next step would be a new, independently evaluated corpus with stronger relevance judgments and domain coverage, including negatives checked for false-negative contamination. Keep a fixed broader retention set and compare against a matched-size repeated-data control.

A future depth experiment should initialize from properly pretrained checkpoints with comparable token budgets, and should replicate the final training recipe across seeds. Any further paid training needs its own bounded spending approval; this campaign's approved work is complete and its paid allocations are deleted.

## Operational outcome

All nine owned RunPod allocations were deleted. Estimated spend was **$9.5843**, within the **$10** campaign cap. Periodic off-pod checkpoint copies preserved model, optimizer, RNG, and sampler state; one interrupted arm was resumed from a saved checkpoint and finished. Cost and deletion evidence are recorded in `cost-closeout.json`.

The four frozen model artifacts contain **17,241,984 parameters** each. Packed Q4/Q8 files save storage, but their reference loader currently reconstructs dense float tensors. Those files do not yet establish a packed-kernel latency advantage. Native ternary execution was verified separately for both quality and isolated latency.
