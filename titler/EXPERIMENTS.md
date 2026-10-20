# titler: experiment history

Full methodology and results for every released (and rejected) version of titler. The READMEs
stay short; this is where the "why" lives — what was tried, what broke, what the data actually
showed, and why each decision was made.

## Summary

| version | architecture | params | ROUGE-L vs teacher | format compliance | key result |
|---|---|---|---|---|---|
| v1.0 | from-scratch, 3+3/d256 | 7.61M | — | ~96.7% | shipped with a label-truncation bug (see below) |
| v1.1 | from-scratch, 3+3/d256 | 7.61M | 0.456 | 98.6% | fixed the truncation bug; +24k synthetic hard cases |
| v1.2 | from-scratch, 3+3/d256 | 7.61M | 0.484 | 98.0% raw | full 340k data pool; rejected a 21M-param scale-up (+0.004 ROUGE-L only) |
| v1.3 | pretrained-init (`t5-efficient-tiny`) | 15.6M | 0.478 | 96.0% raw | closed the ROUGE gap but introduced confident generic-collapse; not shipped |
| v1.4 | both, compared | 7.6M / 15.6M | 0.473 / 0.472 | 97.8% / 97.0% raw | root-caused the collapse; label smoothing + synthetic data partially fixed it; not shipped |
| **v1.5** | from-scratch, 3+3/d256 | 7.61M | **0.511** (0.513 after v1.6) | 95.0% raw (99.8% after v1.6) | LMSYS-Chat-1M data expansion actually resolved the collapse — **current model** |
| v1.6 | (no model change) | — | 0.513 | **99.8%** | format-compliance was mostly a measurement bug, not a model defect |

## v1.0: the label-truncation bug

Title labels were capped at 16 tokens during preprocessing, on the assumption that 8-word titles
≈ 11 tokens. Wrong: with an 8k custom vocab, Title Case English runs ~2.5 tokens/word — the
median title is ~20 tokens — so **75% of the labels were truncated mid-word with EOS appended**,
and the model faithfully learned to end titles mid-word ("Centering a Division with Flexbo"). No
decoding trick fixes a learned behavior; required a full retrain with the cap raised to 32.
Lesson: **when distilling with a small custom vocab, measure label lengths in tokens, not words.**

## v1.1: the synthetic hard-case buckets

From-scratch retrain on an augmented 209k set (200k wild-prompt subsample + 24k synthetic:
SQL, log pastes, vague messages, markdown documents). Seq2SeqTrainer, fp16, LR 5e-4 cosine,
batch 128, label smoothing 0.1, 10 epochs, input 512 / target 32 tokens. ROUGE-L 0.456, format
compliance 98.6%. The synthetic buckets transformed their categories: log pastes went from word
salad ("Pool Connection PottpClientConnected Error") to correct topical titles ("Database
Connection Pool Timeout Error"); SQL and markdown went from unseen to reliably good.

## v1.2: data over depth, round one

Identical architecture/recipe, retrained on the full 339,674-pair pool (all 313,876 deduped wild
prompts instead of the 200k subsample). ROUGE-L 0.484 (+0.028 over v1.1).

**Also tried and rejected: a 21M-parameter, 2.75×-larger architecture** (8 encoder + 4 decoder
layers, d_model 320, gated-GELU FFN) trained on the *same* 209k set v1.1 used. It improved
ROUGE-L by only **+0.004** — seven times less than what the full-data retrain bought, at 2.75x
the size and cost. First clean evidence in this project that data mattered far more than
capacity.

## v1.3: pretrained-init, and a new failure mode

**Motivation.** Titles were "extractive" — rearranging prompt words rather than proposing a real
title. Diagnosis: abstractive/compressive title generation is a language-understanding skill that
comes from pretraining at scale, not from fine-tuning on ~340k narrow examples. A from-scratch
model's only fallback under uncertainty is to lean on the input tokens themselves.

**Approach.** Fine-tune `google/t5-efficient-tiny` (15.6M params, pretrained on C4, not
instruction-tuned) instead of training from random init.

**Run 1** (`lr=1e-4, epochs=5`) hit an fp16-overflow bug first (silent `grad_norm: nan` from step
1 — known T5-in-fp16 issue, fixed with `bf16=True` for the pretrained path). Once fixed: ROUGE-L
0.437 (below v1.2's 0.484), but genuine abstractive wins started appearing ("Who was Julius
Evola?" → v1.2 parrots it verbatim; v1.3 compresses to "Julius Evola Inquiry"). Also introduced
two *new* failure modes never seen in the from-scratch lineage: **hallucination** ("inflammatory
markers." → "Inflammatory Markers in a Neurology," an invented and wrong field) and
**self-repetition** ("Hello and Hello").

**Run 2** (`lr=2e-4, epochs=8`, a stronger recipe) closed almost the entire ROUGE-L gap to v1.2
(0.478) — but the qualitative picture, the actual thing being tested, got *worse*, not better:
- **Generic-collapse**: "User Asks for Assistance" became the exact, character-for-character
  output for 5 different prompts spanning opposite sentiment — "hi", "are you there", "hello can
  you hear me ok good", "you suck", and "thanks that worked!". A friendly close, an insult, and
  three greetings, all collapsed into one identical string.
- **Self-repetition became systematic**: "Help with Help", "Test Test Discussion", "Cost
  Calculation and Cost Inquiry" — six separate hard-cases, not an isolated glitch.
- **Hallucination got more concerning**: "what is 15% of 2400" → "Calculating a **10%** of 2400"
  (wrong number — misstates the input itself). A plain C program titled "Convert **C++** Code to
  **C#**" (invents two languages absent from the prompt).

**Root-cause investigation** (not guessed — checked against data):
- **Not a decode bug.** `repetition_penalty=1.3` changed zero of 8 collapsed outputs. Temperature
  sampling (15 draws across 3 prompts) landed in the same "User Asks for X" template family every
  time, varying only the tail phrase — a genuine, robust collapse in the learned distribution.
- **Not memorized from the teacher.** Titles matching `^(User Asks|Request for|User Requests)`
  are only 1.74% of the 346,598-row training corpus, and "User Asks for Assistance" itself
  appears only 5 times total. "Hi" and "Are you there" both appear in training data with
  *distinct*, non-template teacher titles. The student generalized a rare style family into a
  far more aggressive policy than the teacher's own label frequency would predict.
- **Not present in the from-scratch lineage.** The same 5 prompts run through v1.2 (no
  pretraining) produced 5 distinct titles — no collapse. But 2 of the 5 kept the *original*
  extractive-copying problem this whole experiment existed to fix ("you suck" → "User Asks for
  Suck"). v1.2 has no collapse mode but has extraction; v1.3 has less extraction but a new,
  more severe collapse mode.

**Mechanism (best-supported reading):** pretraining gives the base model a general "produce
plausible boilerplate under uncertainty" habit a from-scratch model never acquires. Cross-entropy
training rewards the population-common answer for inputs whose "correct" title is inherently
underdetermined by 4-8 words — and pushing training harder (higher LR, more epochs) sharpens this
into a *stronger* pull toward the population mode, not toward per-input specificity. This is a
standard mode-seeking-under-optimization-pressure story, not a data-leakage or decoding bug.
**Not shipped.**

## v1.4: fixing the collapse directly, on both architectures

Two levers, tested together (not ablated separately), on **both** the pretrained (v1.4-tiny) and
from-scratch (v1.4-scratch) architectures — to answer whether fixing the collapse actually needs
the pretrained model's extra capacity, or not:

1. **Label smoothing 0.1 → 0.2** (nothing in training previously penalized concentrating
   probability on one population-mode answer).
2. **New synthetic buckets**: ~1,500 hostile/negative reactions ("you suck", "this is useless")
   and ~1,500 gratitude/closing messages ("thanks that worked!", "perfect, thanks") — the
   existing "vague" bucket had neither. Titled by `qwen36-27b-a2.8b-mtp-iq3exp-q4head` after two
   smaller teacher candidates were rejected: a 350M model mostly failed to produce valid output
   (8/10 empty completions), and a 1.2B model collapsed every sample into an "Understanding Your
   X" template regardless of input sentiment — i.e. it reproduced the exact bug this data exists
   to fix, which would have baked the bug into the fix.

**Results**: both fixes measurably weakened the collapse but did not eliminate it, on either
architecture. Gratitude fully resolved on both ("thanks that worked!" → "Thanks for the Help").
The one shared 5-way collapse basin split into two smaller basins on both variants — fewer
prompts sharing an answer, but still sharing. Hostile sentiment didn't transfer into v1.4-tiny's
generation at all despite 1,471 new training examples; v1.4-scratch handled it by falling back to
verbatim extraction ("You Suck" — sentiment-correct in a degenerate way, not a real fix).

**v1.4-scratch answered the "do we need the pretrained oomph" question: no.** Half the
parameters, less than half the artifact size, ROUGE-L and format compliance matching or beating
v1.4-tiny, no self-repetition anywhere in the 68-case suite (v1.4-tiny still showed it), no
invented-number hallucination (v1.4-tiny still fabricated "10%"; v1.4-scratch backed off to a
vaguer but not-wrong answer instead). **Neither shipped** — but v1.4-scratch was identified as the
architecture to build on, since it left only one failure mode (the collapse) to solve instead of
juggling collapse + hallucination + self-repetition across two architectures.

## v1.5: the data expansion that actually worked

**The lever left unpulled**: the wild-prompt pool was thin on real short/vague/casual first-turn
messages — exactly the population the collapse was a symptom of not having enough real examples
for. Added **114,581 new prompts from LMSYS-Chat-1M**, chosen specifically because it barely
overlaps the existing sources (Chatbot Arena/Vicuna demo traffic, not WildChat) and skews toward
that underrepresented distribution. Same architecture and recipe as v1.4-scratch — no changes —
retrained on the enlarged 481,488-pair pool (455,382 train / 4,641 val / 4,641 test).

**Result: ROUGE-L 0.473 → 0.511, and the collapse pattern is resolved** across every hard-case
that previously exhibited it. "hi" → "Casual Greeting and Conversation Start" instead of the
stuck "User Asks for Assistance" template. "you suck" → the sentiment-appropriate "You Suck"
instead of collapsing into generic help-boilerplate. No self-repetition anywhere in the 68-case
suite. This is the clearest evidence in the project's history that data coverage, not model
capacity or training duration, was the actual bottleneck — confirmed a third time, now on the
specific failure mode that first looked like it might need architecture, not data.

One real miss found: a relationship/anniversary prompt was mistitled "Married 6-Year-Old Crime
Podcasts" — a genuine hallucination, isolated rather than a repeated pattern.

**A licensing tradeoff, disclosed rather than glossed over**: LMSYS-Chat-1M's license includes a
no-redistribute-the-dataset clause, a deletion-on-request obligation, and a clause allowing
termination "at any time, for any reason" requiring destruction of all copies. Earlier versions
(through v1.4) deliberately excluded this source for exactly that reason. The decision for v1.5:
ship anyway — publishing weights trained on gated data is not the same as redistributing the
dataset itself (the raw text is nowhere in the released artifacts, and the license explicitly
permits both research and commercial use), which is the standard most released models operate
under. But the deletion/termination obligation is real: if LMSYS ever requests deletion of
specific conversations or terminates the agreement, the honest response is to retrain without
that data and republish.

**Getting the 116,612 new prompts titled was the actual bottleneck of this round**, not the model
training. LM Studio topped out around 0.6-0.7 req/s regardless of concurrency tuning; a local AMD
iGPU hit a genuine memory-bandwidth ceiling around 0.9 req/s. The fix was renting a RunPod RTX
4090 and building llama.cpp from source with CUDA (no official prebuilt Linux+CUDA `llama-server`
binary exists), which got throughput to ~4.5 req/s — the full job in ~7 hours instead of ~38.

## v1.6: fixing the format-compliance gate

Every release from v1.0 through v1.5 missed its own ≥99% format-compliance gate (v1.0 96.7%
through v1.5's 95.0%), always framed as an open model-quality target. Turned out to be **mostly a
measurement bug, not a model defect** — confirmed by logging *why* titles failed instead of just
counting pass/fail, something `eval.py` had never done.

Of 19 failures on a 500-sample run:
- **13 (68%)** were titles exactly 1-2 words over the 8-word cap — already-correct titles. Every
  real consumer of the model (the web/Node/Python examples, `usability_judge.py`) already
  truncated to 8 words downstream; `eval.py` was the one place that didn't, and the shared
  `clean_title()` function didn't enforce the cap either — it existed only as five separately
  duplicated `[:8]` implementations scattered across consumers.
- **5 (26%)** were a "verbatim prompt substring" check flagging legitimately good titles that
  happened to echo phrasing the user themselves used (e.g. a prompt opening "I want you to act as
  an aspect-based sentiment analysis engine..." got titled almost exactly that — correct, not
  lazy copying). Traced back to the check's original spec (`.claude/plans/archive/titler-v1.md`:
  "not >80% verbatim substring" — a ratio test) versus its shipped implementation (a blunt
  substring test with no ratio). Left the shared validator untouched (it also filters
  training-data quality and gates a teacher retry loop — a bigger, riskier change than an eval
  number), and split `eval.py`'s reporting instead: the objective checks (length, quotes,
  newlines, emoji, banned prefixes) are the gate; verbatim-extraction rate is reported separately,
  informationally.
- **1 (5%)** was a genuine short-output case ("hdey", a garbled 4-char input, → 2-word title).
  Tried `min_new_tokens` as a decode-time fix; it didn't even fix the target case (the model was
  already past the token floor) while visibly degrading good short titles elsewhere ("Write a
  Story" → "Write a Story with a Title"). Rejected; left as a documented, accepted 0.2% edge case.

**Result: 95.0% → 99.8%, with zero retraining** — same v1.5 weights. ROUGE-L moved slightly *up*
(0.511 → 0.513), since the clamp only ever removes trailing words from over-length titles, and
several of those were already garbled mid-word cutoffs from hitting the generation length limit,
not real content. Verified no regression to training-data filtering: re-ran the split script
against the existing data pool and diffed the output — byte-identical.

## Lessons that generalized across versions

- **Data beat depth, every time it was tested.** A 2.75× architecture scale-up bought +0.004
  ROUGE-L (v1.1→v1.2); a full-data retrain on the unchanged architecture bought +0.028 in the same
  comparison. Pretrained-init (2× the parameters) never beat the from-scratch model on any axis
  once both had the same fixes applied (v1.4). The LMSYS-Chat-1M data expansion resolved a failure
  mode that first looked architecture-shaped (v1.3's collapse) — the fourth confirmation of the
  same pattern.
- **Aggregate metrics can move in the wrong direction relative to the real question.** v1.3's
  ROUGE-L improved run-over-run while the qualitative failure (collapse, hallucination,
  self-repetition) got worse. Never ship on an aggregate number alone without a qualitative,
  human-read regression check on the specific thing being fixed.
- **When a metric misses its gate for the entire project's history, check what's actually being
  measured before assuming the model needs to improve.** v1.6's gate had never passed once, and
  the fix was two mechanical code bugs, not a data or training change.
- **LLM judges that reason before answering can silently return an empty verdict** when the token
  budget is consumed by hidden `<think>` content — hit this twice, independently, at different
  points in the project (once judging, once during v1.5's data-titling pass). Always inspect raw
  judge/teacher responses before trusting aggregate numbers.
