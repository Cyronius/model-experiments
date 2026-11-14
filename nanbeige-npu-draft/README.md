# Nanbeige tiny NPU drafter

Experimental rank-16 Q4 vocabulary head for a block-diffusion drafter conditioned on Nanbeige4.3-33B. The target checkpoint is the local `Nanbeige4.3-33B-Q4pure-NPU2` GGUF. The model, teacher capture, acceptance evaluation, and NPU export are tracked here as separate gates.

The initial gate is a tied, row-scaled Q4 token table with 166,912 rows and 16 columns. Its packed size is 1,335,296 bytes of nibbles plus 333,824 bytes of FP16 row scales, or 1.59 MiB total. An NPU residency claim requires counting the body, buffers, and the layout of the 4 MiB memory-tile L2 separately from the 2 MiB of distributed compute-tile RAM.

The first model uses the GPU target's normal next-token choice as a seed and drafts six subsequent tokens in parallel from the target's 7,168-dimensional hidden vector. It has two 96-wide attention blocks and a tied Q4 rank-16 token table. The tensor payload is **3.225 MiB** with an FP16 body (1.592 MiB table + 1.633 MiB body). This is a size estimate, not a claim that the NPU kernels or their staging fit in memory. A streaming argmax is required so six full-vocabulary logit vectors are never materialized on the NPU.

The local Vulkan llama-server accepts `--embedding --pooling none --embd-normalize -1` and returns one 7,168-dimensional vector per input token from the exact Nanbeige Q4 GGUF. `capture.py` first obtains greedy continuations and their raw token IDs from a normal server (`return_tokens: true`), then obtains aligned hidden vectors using an embedding-mode server loaded with the same GGUF. Exact token IDs avoid a decode/re-tokenize mismatch at prompt boundaries. Each dataset file is independent; use different prompt sources for train and holdout.

Example free local workflow:

```bash
dibs run gpu port-8400 --label "Nanbeige microdrafter capture" --eta 45m -- \
  python run_capture.py
dibs run cpu --label "Factor Nanbeige output head" --eta 20m -- \
  /home/cyrus/code/model-experiments/target/manifold-rocm/env/bin/python -B init_head.py \
  --output data/pilot/head-rank16.npy
dibs run gpu --label "Train Nanbeige rank-16 Q4 microdrafter" --eta 30m -- \
  /home/cyrus/code/model-experiments/target/manifold-rocm/env/bin/python -B train.py \
  --train data/pilot/train.npz --eval data/pilot/eval.npz \
  --init-factors data/pilot/head-rank16.npy --output runs/q4-pilot --steps 1000
```

The training metric is the exact prefix match against held-out greedy target tokens, counting the GPU seed separately. This is an offline acceptance diagnostic. Request-level speed requires a real NPU drafter and GPU verifier integration, which this first pilot does not yet provide.

The released `ProCreations/Nanbeige4.3-33B-DFlash2` is the GPU drafting baseline and possible teacher. It borrows the target's full embedding and output head, so its reported acceptance does not establish the accuracy of this tiny standalone head.

If Q4 acceptance is promising but SRAM placement is tight, a rank-16 2-bit ternary table with the same per-row FP16 scales would be 1,001,472 bytes (0.955 MiB), saving 667,648 bytes (0.637 MiB). Alternatively, rank-32 ternary uses **exactly the same 1,669,120 table bytes** as rank-16 Q4 and may be the better accuracy tradeoff. The Bonsai 2 ternary DFlash drafter in `openflowlm-next/utilities/dflash-compare` shows that post-quantization tuning can preserve much of a larger drafter's acceptance, but it borrows the target's full head and therefore does not validate ternary factors here.

Do not report a speedup from acceptance alone. Evaluate accepted tokens per target verification and complete request latency against plain Nanbeige with the same quantized target and prompts. For a ternary follow-up, train or fine-tune after quantization and compare with this Q4 model at matched prompts and draft depth.

## First Q4 screening result

The original target-generated set has 64 IFEval prompts / 825 nonoverlapping training blocks and 20 separate DFlash comparison prompts / 260 validation blocks. A direct Q8 output-head probe reproduced the greedy seed on 16/16 sampled blocks, confirming that the saved FP16 hidden vectors are aligned with the target's head. A rank-16 randomized SVD of the target output matrix captured about 1.76% of its weight energy and matched 0/16 greedy seeds by itself; this is a weight-space diagnostic, not the trained drafter's accuracy.

The 300-step warm-start run reached 0.227 accepted future tokens per validation block. The 300-step cold-start control reached 0.215. A 5,000-step warm-start run reached 0.262 at its best checkpoint (1.262 tokens per verification if the guaranteed GPU seed is counted); two-pass refinement did not improve acceptance, and no block had all six future tokens accepted. Its training probe rose to 1.844 accepted future tokens while validation remained at 0.262, so it overfit this small set. A per-position newline majority baseline from the same training data reaches 0.219 accepted future tokens on validation. These are offline prefix-match diagnostics, not latency or NPU results.

A CPU reload of the best checkpoint scored 0.254 rather than the GPU's 0.262 on the same 260 blocks, a small backend numerical difference near argmax ties. A prompt-level paired bootstrap against the newline baseline gave a mean improvement of 0.035 future tokens with a 95% interval of [0.000, 0.081] across 20 prompts. This does not establish a useful acceptance gain.

The expanded 192-prompt target-generated set has 2,479 nonoverlapping blocks. A 1,000-step Q4 run on CPU reached 0.235 accepted future tokens at its best checkpoint, again close to the 0.219 newline baseline and with zero six-token exact blocks. Prefix-weighted training (`6,3,2,1,1,1`) reached at most 0.238 by step 700; that optional run was stopped when Dibs reported host CPU pressure. An overlapping-window version and its Q4 rerun are pending the shared GPU reservation. Preserve the original `data/pilot` and `runs/q4-*` files when continuing.

## MTP-1101 conditioning experiment

The external `applepie69/Nanbeige4.3-33B-MTP-1101` head is pinned to revision `376c8341b19e72754624df1d28de89c99ac8dbab` under ignored `data/mtp-1101/`. Its base config came from `Nanbeige/Nanbeige4.3-33B` revision `df919e1e2cf077a22204c78b6c826bf3db18bc08`. The head's 693,668,864 parameters stay off the NPU. `mtp_experiment.py` feeds the exact local GGUF's final hidden states and Q4_0 token embeddings into that head, using the causal KV/RoPE calculation in the publisher's `speed_test.py`. The extracted MTP output replaces the original hidden vector in the same 3.225 MiB tiny drafter; this first comparison does not add a second condition projection.

The new evaluation uses the GGUF's own chat template, 192 IFEval training prompts, 64 disjoint IFEval prompts for checkpoint selection, and 20 untouched chat/code/docs/reasoning test prompts. Both tiny drafters will train on identical target-generated continuations with identical seed, model, budget, and initialization. `mtp_head_probe.py` measures the standalone MTP top-1 acceptance and top-16 coverage against the exact Q8_0 GGUF output head. `rerank_mtp_prior.py` uses the 64 validation prompts to choose two scalar weights for a sparse top-16 correction to the first draft token, then evaluates that correction once on the 20 test prompts. `evaluate_checkpoint.py` and `compare_mtp_runs.py` report offline accepted-prefix length and a paired prompt bootstrap. This remains an accuracy gate; NPU residency and end-to-end latency require subsequent implementation and measurement.

The checkpoint loaded with the local base config, its Q4 embedding path produced finite outputs, and a small-model check matched the publisher's stepwise causal head computation exactly. Full target-generated capture and tiny-drafter acceptance are pending the shared GPU reservation.

The offline head calculation includes the full prompt and generated-token history. A live integration must keep the MTP layer's KV cache aligned with the base verifier, including rollback after a rejected draft; feeding only the current 7,168-vector to a stateless head would not reproduce this experiment.

### Live MTP head diagnostic

`llama_bridge.cpp` exposes cached GGUF decode, verifier logits, final hidden states, and KV rollback to `mtp_speed.py`. The script compares plain greedy decoding with one-token speculation from the published MTP head on the same Q4 target and prompts. It can also verify six-token blocks from a trained tiny checkpoint on the GPU. The NPU drafter runtime is not implemented.

A CPU-only bridge (`llama_bridge_cpu.so`, linked against the CPU llama.cpp build) completed the 20 untouched test prompts at 32 output tokens each. The MTP head proposed 459 tokens and the verifier accepted 168 (**36.6%**). Speculation used 459 target verification calls versus 620 plain decode calls, and all **20/20** greedy output sequences matched exactly. The result is saved in `data/mtp-chat/cpu-only-acceptance-20x32.json`. A separate Vulkan-linked `-ngl 0` check produced one mismatch in 20 prompts and allocated a Vulkan compute buffer despite zero offloaded transformer layers; it is not used as the acceptance result. CPU throughput during the clean run was affected by another heavy build, so its tokens/sec are diagnostic only.

The 20-prompt GPU run at 64 output tokens per prompt accepted **324/938 drafts (34.5%)**. Raw decode rates were **8.27 tokens/s plain** and **8.56 tokens/s with MTP**; full-request rates were **7.66** and **7.67 tokens/s**. Four of 20 output sequences diverged, and Dibs reported `NOISY` at completion due to unrelated processes, so these are diagnostic measurements, not a validated lossless speedup. The result and full token traces are saved in `data/mtp-chat/gpu-mtp-20x64.json`. One mismatch occurred at output token zero. A targeted fresh-context check of the four divergent prompts found the same first-token argmax with both prefill output settings, so the prefill flag alone does not explain that mismatch; the chat prompt's leading logit margin was only 0.124. Cross-prompt context reuse, verification batch numerics, and KV rollback need separate isolation. The published MTP head did not reach an acceptance rate that makes this prototype compelling.

The publisher's reference speed script says the 1101 head was trained with an NF4 base, while this test uses a Q4_0 GGUF converted from the same BF16 checkpoint. The bundled `mtp_config.json` reports **55.5% top-1 agreement** on teacher-forced held-out text. That metric differs from our **36.6% acceptance during greedy generation** in both data distribution and target quantization. The GGUF bridge returns the final normalized hidden state that the head expects, and the token/position pairing follows the publisher's reference code. Quantization alone has not been shown to explain the gap.

### Execution after Dibs releases the GPU

Use `/home/cyrus/code/model-experiments/target/manifold-rocm/env/bin/python` for these scripts. Rendered train, validation, and test prompts already exist in ignored `data/mtp-chat/`. Run `generate_mtp_chat.py` under `dibs run gpu port-8400` with those three prompt files and `--output data/mtp-chat-gpu`. Run `mtp_experiment.py capture` on each generated JSONL under the same GPU and port reservation, writing separate `train-sequences`, `validation-sequences`, and `eval-sequences` directories. Then run `mtp_experiment.py features` under `dibs run gpu` on each directory, using `--stride 1` for training and `--stride 7` for validation and test.

The feature command writes paired `*-base.npz` and `*-mtp.npz` files. Run `mtp_head_probe.py` on validation and test MTP features under `dibs run cpu`, with `--rows 1024` and the matching generated JSONL. Train `train.py` once for each condition with the same `--init-factors data/pilot192/head-rank16.npy`, `--seed 17`, `--steps 1000`, `--batch 16`, and `--position-weights 6 3 2 1 1 1`; select checkpoints on the 64-prompt validation split. Evaluate those checkpoints on the 20-prompt test split with `evaluate_checkpoint.py`, then run `compare_mtp_runs.py`. Finally, `rerank_mtp_prior.py` can select sparse top-16 correction weights on validation and apply them once to test. Keep all GPU or CPU-heavy commands wrapped in the corresponding Dibs run.
