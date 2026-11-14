# Prepared fresh-science comparison

Active comparison: six arms on two separately bounded Secure RTX4090 pods,
science-float and science-ternary. The combined cap is$3.25 within the explicitly
approved cumulative$10 campaign. Both were allocated at$0.89/hour. Each deletes
as soon as its three arms finish; local quality scoring starts after both close.

The additional sample reads16 random1000-row groups from each of62 pinned
S2ORC Parquet shards. Global hash selection from435053 eligible unique titles
retains250000 pairs:244901 train,5099 dev. Both title and abstract must fit below
128 tokens; titles need3 words and abstracts12. This is weak title/abstract
association, not human relevance judgments. Exact model-visible exclusions and
punctuation-normalized title checks cover existing relevance/NLI inputs and
known benchmark inputs. New connected components stay in one split.

The current clean mixture has1023999 training pairs and21011 held pairs. It
removes the1645 training and17 dev pairs that overlapped newly observed legacy
test inputs. The repeat control substitutes244901 repeated old science pairs
for the fresh training pairs. Both conditions have identical total pair counts,
source weights and expanded held corpus. Root clean-data-audit.json pins these
inputs; older data-audit.json describes the superseded mixture.

Four arms compare repeat/fresh with the original two-layer float and ternary
parents. Two additional fresh-data arms use recovered pretrained three-layer
r0 checkpoints. These test candidate recipes, not a controlled estimate of
pretraining depth: the three-layer float checkpoint saw75M tokens, versus150M
for the two-layer float control. Three-layer candidates have17241984 parameters;
existing NPU graphs support only two layers. All checkpoints are fixed original
parents, so the new clean-data fine-tune does not inherit the earlier campaign's
fine-tuning overlaps. Historical public-text pretraining overlap remains unknown.

Fixed loss: NLI encoder weight.1 float/.25 ternary; no tested BM25-negative loss.
Auxiliary batch192 exposes the encoder to287456 distinct human pairs across
385440 exposures; the classifier sees the whole300k pool. Frozen-encoder warmup
advances the sampler, so the encoder does not see all300k distinct examples.
The remote-runtime count is verified in../campaign/science-exposure-audit.json.

Float cap$1.50/max1.6h, work80min/watchdog90min; ternary cap$1.75/max1.9h,
work100min/watchdog110min. Each uses temporary60GB disk. Regular checkpoints
are copied off-pod and SHA-verified, retaining full optimizer/RNG/sampler/head
state for recovery. No new teacher targets. Freeze all six completed arms using
the fixed internal source guards before any external development scoring.
