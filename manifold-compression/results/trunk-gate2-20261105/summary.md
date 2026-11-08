# 9B experts grafted onto a small Qwen3.5 trunk: results and lessons (2026-11-05 to 11-08)

Status: **closed** (owner, 2026-11-07). Grafting one model's experts onto another model failed. The goal is still turning a dense model into an MoE cheaply, and the work goes back to carving the dense model directly (the S0_K24 line).

Plan with the pre-registered gates: `../.claude/plans/9b-experts-on-small-trunk.md` (ignored by git). Code: `trunk_gate1.py`, `trunk_gate2.py`. Results: `results/trunk-gate1-20261105/gate1.json`, this folder. Logs: `logs/trunk_gate*_*.log` (ignored by git).

## The question

Can a stock Qwen3.5-0.8B or 2B take routed experts made from the Qwen3.5-9B's MLPs and score above its own size class? The trunk keeps everything it has, including its own MLP. The added experts only write to the residual stream, through an output gate that starts at 0, so step 0 is exactly the stock trunk.

The target (stretch): the 0.8B with 9B experts reaches the stock 2B, or the 2B with 9B experts reaches the stock 4B.

## Gate 1: how much of each 9B MLP the small trunk's stream can drive (no training)

`trunk_gate1.py`. 64 rows of 512 tokens (wikitext, UltraChat with thinking off, MBPP); fit on 3 of every 4 rows, score on the 4th (8,176 tokens). For each 9B layer, a ridge map from the best trunk layer (within ±3 of the proportional position) to the 9B MLP input. Then the full 9B MLP runs on the mapped input and is compared with its true output.

| trunk | median R², MLP output | median R², MLP input | middle layers 7-15, MLP output | pre-registered band |
|---|---|---|---|---|
| 2B | 0.381 | 0.48 | 0.18-0.26 | mixed (0.3-0.6) |
| 0.8B | 0.299 | 0.43 | 0.11-0.20 | dead by 0.001; owner passed it as mixed |

- U-shaped by depth: the first two 9B layers (0.52-0.64) and the last (0.67-0.73) carry over best; the middle of the 9B carries over worst.
- Single neurons carry over poorly: median neuron-activation R² 0.20-0.40 for the 2B. The MLP output does better only because it sums many neurons.
- The mixed band meant: go to gate 2 only with a random-expert control. That rule decided the outcome.

## Gate 2: frozen trunk, 9B experts trained briefly, against random experts

`trunk_gate2.py`.

- 24 of the 32 9B layers paired one-to-one with the 24 trunk layers (strictly increasing, maximizing held-out input R²). Mean input R² 0.429 (0.8B), 0.482 (2B).
- Experts: the S0_K24 partition and learned routers (96 experts × 128 neurons per layer, top-24). Gate, up and router rows folded through the affine ridge map; down folded through its pseudo-inverse, then a per-channel output gate starting at 0.
- Trained, trunk frozen: LoRA rank 64 on the folded weights, the routers, the gates. Loss: KL to the 9B's stored top-20 log-probs plus 0.5 × CE, answer tokens only.
- Control: random experts and router with the same row and column norms, same steps, same data order.
- Budget: about 1.1M answer tokens per arm (0.8B 92 steps, 1.3 h; 2B 90 steps, 1.7 h on the local 8060S). The plan's budget was 20M. The output gates ended near |g| ≈ 0.005.

### Held-out KL to the 9B (160 answers, 214,246 tokens, recomputed on CPU)

| trunk | stock | + 9B experts | + random experts | share of the gain the control also gets |
|---|---|---|---|---|
| 0.8B | 0.5061 | 0.4891 | 0.4901 | 95% |
| 2B | 0.3218 | 0.3120 | 0.3128 | 91% |

### Multiple-choice bench (CPU; choice log-probs, no generation; chance 25%)

MMLU 1,135 items, ARC-Challenge 600, ARC-Easy 400, HellaSwag 600.

| model | MMLU | ARC-C | ARC-E | HellaSwag | all |
|---|---|---|---|---|---|
| stock 0.8B | 47.1 | 32.2 | 53.0 | 49.3 | 45.2 |
| 0.8B + 9B experts | 47.7 | 32.3 | 52.8 | 50.5 | 45.7 |
| 0.8B + random experts | 48.1 | 32.3 | 55.0 | 52.0 | 46.5 |
| stock 2B | 53.7 | 39.7 | 60.3 | 60.2 | 53.0 |
| stock 9B (GPU run of 2026-11-01, same items) | 67.1 | 54.3 | 76.2 | 70.5 | 66.4 |

Paired on the same items (z = difference / its standard error): 9B experts vs stock +0.5 points (z 1.2); random experts vs stock +1.3 (z 3.4); 9B experts vs random experts −0.8 (z −2.3). The stock 2B on CPU matches its 2026-11-01 GPU run (53.4 / 39.5 / 60.5 / 59.7), so the 9B row is comparable. The bench of the 2B arms with experts was stopped before it finished, to free the machine.

### GSM8K, 96 prompts, model-card sampler (local GPU, no dense canary)

| model | thinking off | thinking on |
|---|---|---|
| 0.8B + 9B experts | 43 | 35 |
| 0.8B + random experts | 43 | 31 |
| 2B + 9B experts | 73 | 57 |
| 2B + random experts | 77 | 52 (9 loops) |
| stock 2B | 71 | 38 |
| stock 4B | 83 | |

One standard deviation for the difference between two arms is about 7. The stock 0.8B has no clean score.

### Verdict

Fail, on the pre-registered rule "not better than the random-expert control". The 9B experts beat random experts by 0.0009 KL on both trunks, tied or lost on GSM8K within noise, and lost on the multiple-choice bench. Stretch: no (0.8B: 43 against the stock 2B's 71 on GSM8K; 45.7 against 53.0 on the bench).

## Lessons

About dense-to-MoE conversion:

1. **Expert weights do not carry over between models through a linear map.** Read from another model's stream, the 9B experts were worth no more than random weights of the same shape. An expert works only in the stream of the model it was carved from. This supports carving a model's own MLPs (S0_K24), not grafting.
2. **Gate 1's neuron-level R² was the right warning.** The MLP-output R² (0.30-0.38) looked half-good because it sums many neurons, but each expert holds only 128 neurons, and single neurons were readable at R² 0.2-0.4. Judge a fold by what one expert sees, not by the summed output.
3. **The gain came from capacity plus distillation, not from the donor weights.** Random experts with 1M tokens of distillation from the 9B lifted the stock 0.8B by 1.3 points (z 3.4). Distillation from a strong teacher is cheap and works; the weights you start the experts from must already fit the stream.
4. **Carving down is cheaper than upcycling up.** A carve keeps the knowledge in the weights and training only repairs damage. Adding experts to a small model means learning the knowledge through distillation, at a much larger token cost.

About method:

5. **The random-weight control is what made the result honest.** Without it, gate 2 reads as a win: held-out KL down 2-3%, bench up, GSM8K at or above the stock 2B. Every one of those gains was matched by random experts. Any "add capacity from model X" experiment needs a same-shape random control trained identically.
6. **Small KL gains can be pure capacity.** A 3% held-out KL drop said nothing about whether the donor weights mattered. Score with something that can separate arms: paired multiple-choice accuracy (2,735 items gave z 2.3 for a 0.8-point gap) and sampled GSM8K.
7. **A zero-init output gate that stays near zero means the added part barely trained.** The gates ended at |g| ≈ 0.005 after 90 steps. Watch the gate size during training; it shows whether the budget was large enough to test the idea at all.

About the local machine:

8. **The local GPU (gfx1151) has computed wrong values silently since 2026-11-05.** The GPU multiple-choice bench gave NaN log-probs (391 rows for the stock 0.8B; CPU runs of the same rows were clean). Held-out KL was recomputed on CPU for all six arms and matched the GPU values within 0.0001, so training was not affected. Re-check any local GPU eval on CPU, or against the dense canary, before trusting it.
9. **CPU is a usable eval device for small models.** bf16 on 16 threads: held-out KL 18 min (0.8B stock) to 88 min (2B with experts, sharing cores); the bench 45-75 min per 0.8B arm.
10. **Process hygiene for detached jobs:** `pkill -f` with a pattern that also appears in your own command kills your own shell; killing a dibs wrapper leaves its child shell running, so two drivers ran at once until it was found. Kill by PID, check the process tree, and give every step a skip-if-output-exists guard.

## Files

- `Qwen3.5-*/fold.json`: the layer pairing and input R² per pair.
- `Qwen3.5-*/<arm>/train.json`: config, steps, tokens, training-time held-out KL.
- `Qwen3.5-*/<arm>/held_out_cpu.json`: CPU held-out KL and CE.
- `Qwen3.5-*/<arm>/check_thinking_{off,on}.json`: GSM8K summaries.
- `Qwen3.5-*/<arm>/bench.json`: CPU multiple-choice bench with per-item log-probs.
- `run_*.sh`: the detached drivers, in the order they ran.

Deleted on 2026-11-08, when the line closed: the trained adapters (`state.pt`), the folded experts (`fold.pt`), the GSM8K generations (`gen_*.jsonl`), the discarded GPU bench outputs, and the gate 1 scratch folder `trunk-gate1-20261105/` at the repo root (activation captures, data, a copy of the Qwen3.5-0.8B snapshot). A rerun needs gate 1's captures again, and `trunk_gate1.py` still has Windows paths.
