# IFEval: stock 0.8B, 300M and 500M

Local literal-character correction for # and ! counts (keys 1122 and 1129); all other installed lm_eval 0.4.13 checks unchanged. Native results and responses preserved. Every corrected result rescored twice.

541 prompts; 834 instruction checks per model. Matching Qwen sampling, thinking off, Q8_0, 1280-token output cap; one sample per prompt.

| Model | Prompt strict | Instruction strict | Prompt loose | Instruction loose |
|---|---:|---:|---:|---:|
| stock-0.8B | 46.03% | 56.71% | 52.68% | 63.07% |
| ours-300M | 15.71% | 27.34% | 17.56% | 29.98% |
| ours-500M | 14.23% | 26.50% | 16.82% | 29.50% |

Prompt scores require all instructions to pass; instruction scores count checks separately.
This local scoring correction affects only two prompts and does not change the comparison's ranking. These are not directly comparable to published scores using other decoding or scoring settings.
Native uncorrected results: summary.md and comparison.json. Exact changes: scoring-audit-corrected.json.
Upstream checker behavior: https://github.com/google-research/google-research/issues/3427

Paired strict outcomes: {"stock-0.8B vs ours-300M": {"both_pass": 62, "first_only_pass": 187, "second_only_pass": 23, "both_fail": 269}, "stock-0.8B vs ours-500M": {"both_pass": 63, "first_only_pass": 186, "second_only_pass": 14, "both_fail": 278}, "ours-300M vs ours-500M": {"both_pass": 50, "first_only_pass": 35, "second_only_pass": 27, "both_fail": 429}}
