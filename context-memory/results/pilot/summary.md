# Qwen 4B XDNA2 encoded-memory recall pilot

Local CPU inference, no training, $0 paid compute.

| Condition | Real note facts | Synthetic canaries | Absent facts (abstention) |
|---|---:|---:|---:|
| no_memory | 0/14 | 0/2 | 2/2 |
| full_text | 0/14 | 0/2 | 0/2 |
| latent | 13/14 | 2/2 | 2/2 |
| zero_memory | 0/14 | 0/2 | 2/2 |
| counterfactual | — | 2/2 | — |
| stock_no_memory | 0/14 | 0/2 | 2/2 |
| stock_full_text | 12/14 | 2/2 | 1/2 |

Encoded 1,929 source tokens into 483 continuous memory slots (2,472,960 bytes), once before questions; decoder width 2560.

The latent and zero-memory conditions read only the saved memory file and question text. Original notes are read by the encoder and the full-text control. The first five conditions use the same released LCLM Qwen3-4B decoder, which was trained by the authors. Stock controls, when included, use the original Qwen3-4B-Instruct-2507 weights. This is not a frozen-stock-Qwen surgery experiment. No query-time retrieval or tool access is enabled.

The LCLM full-text control ended immediately in prompt preflight, with and without document markers. Stock controls were added afterward to check that the corpus and questions are answerable. A comparison across stock and LCLM weights does not isolate compression cost. The no-memory, zero-memory, and changed-bank controls establish whether encoded content is being used.

Scores use preregistered answer patterns. Check the raw answers below: matching is a small-pilot diagnostic, not a broad benchmark. The six notes are hand-selected, and some contain disputed or stack-specific findings. Questions ask what those notes report, not universal hardware truth. Canary strings are synthetic controls.

When review.json is supplied, scores include explicit human-readable adjudications. The raw regex result remains in scored.json. The latent uint8-view response is rejected for contradicting itself; the firmware response correctly says the version is unspecified despite missing the regex wording. The latter response and the contradictory response also hit the output cap.

## Raw answers

| Condition | Question | Answer | Pass |
|---|---|---|---|
| no_memory | program-size | UNKNOWN | no |
| no_memory | overflow-error | UNKNOWN | no |
| no_memory | compile-flag | UNKNOWN | no |
| no_memory | stack-working | UNKNOWN | no |
| no_memory | stack-granite | UNKNOWN | no |
| no_memory | float-bug | UNKNOWN | no |
| no_memory | peano-symbol | UNKNOWN | no |
| no_memory | mask-view | UNKNOWN | no |
| no_memory | classic-time | UNKNOWN | no |
| no_memory | elf-time | UNKNOWN | no |
| no_memory | completion-state | UNKNOWN | no |
| no_memory | softmax-mask | UNKNOWN | no |
| no_memory | same-context | UNKNOWN | no |
| no_memory | switched-context | UNKNOWN | no |
| no_memory | pilot-code | UNKNOWN | no |
| no_memory | recovery-phrase | UNKNOWN | no |
| no_memory | missing-driver | UNKNOWN | yes |
| no_memory | missing-canary | UNKNOWN | yes |
| full_text | program-size |  | no |
| full_text | overflow-error |  | no |
| full_text | compile-flag |  | no |
| full_text | stack-working |  | no |
| full_text | stack-granite |  | no |
| full_text | float-bug |  | no |
| full_text | peano-symbol |  | no |
| full_text | mask-view |  | no |
| full_text | classic-time |  | no |
| full_text | elf-time |  | no |
| full_text | completion-state |  | no |
| full_text | softmax-mask |  | no |
| full_text | same-context |  | no |
| full_text | switched-context |  | no |
| full_text | pilot-code |  | no |
| full_text | recovery-phrase |  | no |
| full_text | missing-driver |  | no |
| full_text | missing-canary |  | no |
| latent | program-size | 16 KB | yes |
| latent | overflow-error | XAIE_INVALID_ELF | yes |
| latent | compile-flag | The answer is: `-Os` | yes |
| latent | stack-working | The original phlegm fp32-vector observation in F-0009 (before its disputed section) states that a worker stack of 0x1800 works. | yes |
| latent | stack-granite | The answer is: 0x3000. | yes |
| latent | float-bug | The 16-lane int32-to-float `to_float` conversion produced NaN on device. | yes |
| latent | peano-symbol | The exact cycle-counter function reported as unlinkable under Peano in F-0014 is `get_cycles()`. | yes |
| latent | mask-view | The provided notes do not specify an integer view for masking in F-0014. The recommendation is to "Mask through a uint8 view," but no specific integer view is mentioned. Therefore | no |
| latent | classic-time | The classic mlir-aie flow took 0.602 ms. | yes |
| latent | elf-time | The xrt::elf load-path time for ROT13 in F-0038 was 0.447 milliseconds. | yes |
| latent | completion-state | The completion-state number reported by both ROT13 load paths in F-0038 is 4. | yes |
| latent | softmax-mask | bf16 `lowest` | yes |
| latent | same-context | The back-to-back dispatch time range within one context was 0.23-0.25 ms. | yes |
| latent | switched-context | The dispatch time range measured when the previous dispatch used another xclbin context was 0.65-0.79 ms. | yes |
| latent | pilot-code | The code for the memory recall pilot is velvet-heron-7316. | yes |
| latent | recovery-phrase | The recovery phrase for the pilot memory bank is mosaic-lantern-4829. | yes |
| latent | missing-driver | The notes for F-0038 do not specify the exact NPU firmware version used for the ROT13 load-path comparison. The sources mention "XRT runtime 32.0 | yes |
| latent | missing-canary | The emergency override password for the memory recall pilot is UNKNOWN. | yes |
| zero_memory | program-size |  | no |
| zero_memory | overflow-error | UNKNOWN | no |
| zero_memory | compile-flag | UNKNOWN | no |
| zero_memory | stack-working | UNKNOWN | no |
| zero_memory | stack-granite | UNKNOWN | no |
| zero_memory | float-bug | UNKNOWN | no |
| zero_memory | peano-symbol |  | no |
| zero_memory | mask-view | UNKNOWN | no |
| zero_memory | classic-time |  | no |
| zero_memory | elf-time | The provided notes do not contain any information about ROT13 in F-0038 or the xrt::elf load-path time. Therefore, the requested fact is absent.  Answer | no |
| zero_memory | completion-state | UNKNOWN | no |
| zero_memory | softmax-mask |  | no |
| zero_memory | same-context | UNKNOWN | no |
| zero_memory | switched-context |  | no |
| zero_memory | pilot-code | The code for the memory recall pilot is UNKNOWN. | no |
| zero_memory | recovery-phrase | The recovery phrase for the pilot memory bank is UNKNOWN. | no |
| zero_memory | missing-driver | UNKNOWN | yes |
| zero_memory | missing-canary | UNKNOWN | yes |
| counterfactual | pilot-code | The code for the memory recall pilot is copper-badger-9527. | yes |
| counterfactual | recovery-phrase | The recovery phrase for the pilot memory bank is amber-compass-1638. | yes |
| stock_no_memory | program-size | UNKNOWN | no |
| stock_no_memory | overflow-error | UNKNOWN | no |
| stock_no_memory | compile-flag | UNKNOWN | no |
| stock_no_memory | stack-working | UNKNOWN | no |
| stock_no_memory | stack-granite | UNKNOWN | no |
| stock_no_memory | float-bug | UNKNOWN | no |
| stock_no_memory | peano-symbol | UNKNOWN | no |
| stock_no_memory | mask-view | UNKNOWN | no |
| stock_no_memory | classic-time | UNKNOWN | no |
| stock_no_memory | elf-time | UNKNOWN | no |
| stock_no_memory | completion-state | UNKNOWN | no |
| stock_no_memory | softmax-mask | UNKNOWN | no |
| stock_no_memory | same-context | UNKNOWN | no |
| stock_no_memory | switched-context | UNKNOWN | no |
| stock_no_memory | pilot-code | UNKNOWN | no |
| stock_no_memory | recovery-phrase | UNKNOWN | no |
| stock_no_memory | missing-driver | UNKNOWN | yes |
| stock_no_memory | missing-canary | UNKNOWN | yes |
| stock_full_text | program-size | 16 KB | yes |
| stock_full_text | overflow-error |  | no |
| stock_full_text | compile-flag | - Os | no |
| stock_full_text | stack-working | 0x1800 | yes |
| stock_full_text | stack-granite | 0x3000 | yes |
| stock_full_text | float-bug | The 16-lane int32-to-float to_float conversion produced NaN on device. | yes |
| stock_full_text | peano-symbol | get_cycles() | yes |
| stock_full_text | mask-view | The question asks for the integer view recommended in F-0014 for masking. In the provided notes under F-0014, the recommendation is to "Mask through a uint8 | yes |
| stock_full_text | classic-time | 0.602 | yes |
| stock_full_text | elf-time | 0.447 | yes |
| stock_full_text | completion-state | 4 | yes |
| stock_full_text | softmax-mask | bf16 `lowest` | yes |
| stock_full_text | same-context | 0.23-0.25 | yes |
| stock_full_text | switched-context | 0.65-0.79 ms | yes |
| stock_full_text | pilot-code | The code for the memory recall pilot is velvet-heron-7316. | yes |
| stock_full_text | recovery-phrase | mosaic-lantern-4829 | yes |
| stock_full_text | missing-driver | UNKNOWN | yes |
| stock_full_text | missing-canary | The emergency override password for the memory recall pilot is mosaic-lantern-4829. | no |
