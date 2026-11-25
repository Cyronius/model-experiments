# adaptive-loop: a looped model that picks an operation each step

Idea (owner, 2026-11-18): a small transformer that loops. At each step a controller looks at the state, picks one operation (an expert) from a shared library, applies it, and decides whether it is done. Weights are not tied to a depth: the experts are learned operations, used in whatever order and as many times as each token needs.

This replaces the earlier plan in `../looped-moe-pool/`, which tried to build this by pooling a pretrained MoE's experts. That failed (stages 0 and 1, see its findings.md) because experts pretrained for one depth do not work at others. Here everything is trained from scratch, so the operations learn to be order-free from the start.

## What has to be true for the idea to work

1. **Adaptive operation choice.** Different tokens and inputs follow different sequences of operations. If every token runs the same fixed sequence, it is just a deep network.
2. **Adaptive depth.** Harder tokens take more steps, without being told how hard they are.
3. **Better than the baselines at matched compute:** a dense stack of distinct layers, and a single looped block with the same halting.

## Testbeds

Three tasks, so one task's quirks do not decide the verdict.

1. **Circuit simulator** (first). Random Boolean circuits (AND, OR, XOR, NAND, NOR, NOT). One position per node, nodes shuffled, wired by random labels, so the model has to find each gate's inputs by attention. The target is every gate's output. A gate at level *l* needs at least *l* propagation steps, so the right number of steps per token is known exactly. This is a direct check of point 2. It also tests depth extrapolation: train on depth ≤ 12 and test on depth 13–24 with more steps.
2. **Big Muff-family pedals** (added 2026-11-19, owner's request). Analog circuits: resistors, capacitors, NPN transistors, clipping diodes, tone stacks; 1–4 stages, 25% real Muff layout. One token per component and node; each node predicts its voltage at 16 phases of the last of 4 input-sine cycles. Ground truth from our own MNA/Newton solver (`analog.py`, checked against exact cases). Out of distribution: 5–6 stages. Possible standalone use: a schematic-conditioned real-time pedal emulator. The full waveform in one shot proved unlearnable (findings.md); the DC bias point is learnable. Current form (2026-11-23, `gen_analog_step.py`): one 48 kHz sample at a time. Each node gets its voltage now and the input at the next sample, and predicts its change; tested closed loop over 480 samples (`eval_rollout.py`), which is how a real-time emulator would run.
3. **Post-OCR cleanup** (third). wikitext-2 windows of ~100 characters, corrupted with OCR-style errors (character confusions, rn/m and cl/d, dropped and stray characters, merged and split words, line-break hyphens); the target is the clean text. Each input character predicts 0–2 output characters, so an encoder can fix insertions and deletions. Test on wikitext-2 test, plus a heavy-noise set (15% of characters damaged vs 3%). Most characters only need copying and a few need real repair, so per-token adaptivity should show up as steps spent only on the damaged spans. Scored by character error rate.

## Models (same code; only the per-step block differs)

- `dense`: T distinct blocks (attention + MLP), with early exit.
- `loop`: one shared block, looped, with halting. This is `oploop` with K = 1.
- `oploop`: shared attention each step, then a router picks 1 of K expert MLPs per token. The router sees the state plus a step embedding.

All three use input re-injection each step, an output head after every step, and the same halting.

## Training plan for adaptivity

- **Answer at every step.** Loss on every step's prediction, weighted by the halting distribution (PonderNet style), so stopping early always gives a usable answer.
- **Charge for thinking.** A cost on the expected number of steps, ramped in after the task starts being learned, so the model spends steps only where they lower the loss.
- **Hard choice that matches inference.** Straight-through Gumbel top-1 routing: the forward pass uses exactly one expert per token (as at inference), the backward pass uses the soft probabilities. Temperature anneals.
- **Variety.** Switch-style balance loss across experts. Track H(op | step) and H(op | gate type, step) to catch collapse to a fixed sequence.
- **Train and inference halting agree.** At evaluation a token halts at its first step with halting probability > 0.5. Report two modes: `readout` (all tokens keep computing; each token's answer is read at its halting step) and `freeze` (a halted token's state stops changing, which is what saves compute).

## Measurements

- Accuracy by gate level, in distribution and on deeper circuits.
- Mean halting step by level, and its correlation with level.
- Router statistics: usage per expert, entropy per step, and whether choices depend on the token.
- Wall time and active FLOPs per token.
