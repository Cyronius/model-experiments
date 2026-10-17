# What Gemma 3n's Nesting Actually Buys, and Why Gemma 4 Dropped It

*Measurements taken in August 2026 on a rented RTX A40 GPU (48 GB) from RunPod.
Models: Gemma 3n E4B-it and Gemma 4 E4B-it, both run in bf16 unless stated
otherwise. Software: transformers 5.15.1 and PyTorch 2.9.1+cu128 for the
HuggingFace-based tests (August 20-21), and llama.cpp built from its master
branch with CUDA support on August 22 for the decode-speed test. Code and raw
logs: `gemma_phase1/` in this repo.*

Gemma 3n is a 4-billion-parameter model. But a working 2-billion-parameter
model is nested inside its weights. Google can cut Gemma 3n E4B down to that
smaller size, using a method called Mix-n-Match, without retraining anything.
The result still works. This is called nesting, and Gemma 3n is the only
shipped, production model that does it.

Gemma 4 does not. Its two sizes, E2B and E4B, are trained completely
separately. Gemma 4's technical report never mentions nesting, and the model
shapes don't line up the way a nested model's would: E4B has a hidden size of
2560 and 42 layers, and E2B is not a clean slice of those numbers. Some blog
posts online still claim Gemma 4 nests the same way Gemma 3n does. The actual
config files say otherwise.

Before the idea disappears from Google's lineup, we wanted to measure what
nesting is worth. Three questions: What does nested training buy over a
normal model cut to the same size? Can you switch which size you're running
mid-conversation without losing the model's memory of what was already said?
And is nesting actually faster in a real inference engine, not just smaller
on paper?

Short version: yes to all three. Nesting buys a usable 2x range of model
sizes from one downloaded checkpoint. Switching sizes mid-conversation costs
almost nothing. And in a fast inference engine, cutting the model in half
also makes it 44% faster to generate text.

## What a nested model actually is

Most models are trained at one fixed size. A nested model, sometimes called a
MatFormer (short for Matryoshka Transformer, after Russian nesting dolls), is
trained differently: several widths of the same layer are trained at once,
all sharing one block of weights. Think of a set of measuring cups nested
inside each other. The smallest cup has to hold a full, correct amount on its
own, and the largest cup contains all the smaller ones.

In practice, each training batch runs through the model more than once, at
different widths, and the losses from every width get added together before
the weights update. That forces the first slice of each weight matrix to
work as a complete, usable model by itself, not just as a fragment of a
bigger one. Once training finishes, the weight matrix can be cut down to any
of the trained widths and it still runs.

## 1. What nested training buys: the collapse test

Every transformer layer has two parts. Attention lets tokens look at each
other. A feed-forward network (FFN, also called an MLP) processes each token
on its own, usually with more parameters than the attention part. The FFN is
the piece that gets nested. Cutting a model "to width h" means keeping only
the first h columns of the FFN's weight matrices and throwing the rest away.

To see what nested training buys, we ran the same cutting script on two
models: Gemma 4 E4B, trained at one fixed size, and Gemma 3n E4B, trained
nested. Same script, same cut widths, same text.

We scored quality with perplexity, a standard measure of how well a model
predicts text: roughly, the number of words the model is still confused
between when guessing what comes next. Lower is better; a small rise is
normal, a 10x rise means the model is falling apart. We used the
wikitext-2-raw-v1 test set, took its first 40,960 tokens, and split that into
windows of 2048 tokens each. We scored each window on its own, and gave each
one its own BOS token (short for "beginning of sequence") stitched onto the
front before scoring. Gemma models expect a BOS token at the very start of
any input. Skip it on every window and the score gets much worse, even on
identical text. We don't have the exact before-and-after numbers saved (we
remember it being something like 445 versus 61 perplexity on the same text),
but the fix is applied to every number below.

| FFN width | Gemma 4 E4B (not nested) | Gemma 3n E4B (nested) |
|---|---|---|
| 100% | ppl 60.8 | ppl 34.5 |
| 75% | ppl 171 (2.8x worse) | ppl 37.0 (7% worse) |
| 50% | ppl 5,996 (99x worse) | ppl 44.5 (29% worse) |
| 25% | ppl 49,000,000 | ppl 3,541 (collapsed) |

At half width, the non-nested model stops producing real text. Its actual
sample output at that setting is number salad: `* 121-13 (21-6-0-212)-21:2-
23-22-0-20 (The City of`. The nested model at the same width still answers
correctly: "The capital of France is **Paris**... the **Seine**." Nesting
comes from training, not from the architecture. An un-nested model sliced in
half breaks. A nested model sliced in half barely notices.

Nesting has a floor, though. Gemma 3n's official 2B size has an FFN width of
8192, exactly 50% of E4B's 16384. Google never trained anything narrower as
a standalone target, and at 25% width the nested model collapses too. So the
usable range is 50% to 100% of full width: a 2x range of compute to dial,
not an open-ended ladder down to nothing.

## 2. The quality-vs-cost ladder, with real sliced weights

Section 1 zeroed out FFN weight rows instead of physically removing them, to
check the collapse idea quickly. Zeroing the rows of a unit and actually
deleting it produce the exact same output, because a unit whose weights are
all zero contributes nothing downstream. Keeping it around costs nothing in
accuracy. Zeroing doesn't save memory or make the model faster, though: the
zeroed rows are still sitting in GPU memory, and the GPU still multiplies
them.

For real memory and speed numbers, we did the cut for real: built new,
smaller weight matrices with the extra rows and columns physically removed.
An FFN computes roughly `gate = W_gate @ x`, `up = W_up @ x`, then
`activation(gate) * up`, then `out = W_down @ (that)`. `gate_proj` and
`up_proj` produce h-dimensional outputs, so slicing them means keeping their
first h output rows; `down_proj` consumes that h-dimensional vector as
input, so slicing it means keeping its first h input columns. We ran this on
Gemma 3n E4B at five widths, on the same A40, using Hugging Face's
`transformers` library (HF for short from here on):

| FFN width | intermediate size | wikitext ppl | HellaSwag-200 | decode tok/s | prefill ktok/s | VRAM |
|---|---|---|---|---|---|---|
| 100% | 16384 | 34.46 | 0.590 | 12.4 | 5.1 | 13.7 GB |
| 87.5% | 14336 | 35.99 | 0.585 | 12.5 | 5.3 | 12.8 GB |
| 75% | 12288 | 36.94 | 0.590 | 12.8 | 5.6 | 11.9 GB |
| 62.5% | 10240 | 39.96 | 0.560 | 12.8 | 5.9 | 11.0 GB |
| 50% | 8192 | 46.16 | 0.560 | 12.9 | 6.2 | 10.2 GB |

HellaSwag is a multiple-choice common-sense benchmark. We used the first 200
validation examples: for each one, the model scores every candidate ending by
its mean per-token loss, and picks the ending with the lowest loss as its
answer. Accuracy holds within 3 points across the whole ladder.

Memory drops 26% at half width. Prefill (processing a batch of input tokens
all at once, before generating anything new) gets 22% faster. Both are real,
physical savings. Decode speed (generating new tokens one at a time) barely
moves: 12.4 to 12.9 tokens per second across the whole ladder. That flat
line is a limit of how HF runs the model, not a sign that cutting the FFN
doesn't help. At batch size 1, HF's standard generation loop spends most of
its time on Python overhead and GPU kernel launches, not the FFN math
itself, so the weight savings mostly disappear into that overhead. Section 4
runs the same comparison in an engine built to avoid that problem, and gets
a very different result.

One quirk: zeroing rows and truly slicing them give slightly different
scores on Gemma 3n, even though they should be identical by the logic above.
Gemma 3n tracks statistics about which of its FFN units are active for a
given token. This is called activation sparsity, and some of the model's
internal routing depends on it. Those statistics are computed over however
many units the FFN currently has. Zero out rows and the FFN still has all
16384 units, just with some silenced; slice the matrix down and it genuinely
has 8192, so the statistics come out slightly different. In practice,
zero-masking scored a bit better at half width (ppl 44.5) than true slicing
(ppl 46.2). If memory isn't the constraint, masking is the better way to run
a narrower Gemma 3n.

## 3. Switching width mid-conversation doesn't break the cache

During generation, a transformer keeps a cache of every past token's
attention data, called the KV cache (key/value cache), so it doesn't have to
reprocess the whole conversation from scratch on every new token. Attention
is identical at every width in a nested model (only the FFN is sliced), so a
conversation's KV cache is the same shape no matter which width produced it.
That raises a real question: cache the context at one width, switch to a
different width for the answer, and is the leftover cache still
trustworthy? The residual stream (the running total each layer adds to as a
token passes through the network) was computed at the old width when that
cache was built. It's not obvious reusing it at a new width is safe.

We measured this on our own toy nested transformers rather than on Gemma
directly, since it lets us test every width pair, not just one. These are
small models, about 0.36 million parameters, trained with the same nested
setup as Gemma's (shared attention, sliced FFNs) across 260 training runs
covering different width tiers, 10 random seeds, and different routing
setups. The relevant piece: a transformer is causal, meaning a token can only
attend to earlier tokens, never later ones. Because of that, and because
only the FFN differs between widths, we could simulate "prompt processed at
width A, answer generated at width B on the old cache" inside a single
normal forward pass: run the FFN at width A for the early token positions
and width B for the later ones, instead of building two separate caches and
swapping between them. That's mathematically the same thing a real
deployment would do, just cheaper to test.

Results, averaged over the 10 seeds for the loss numbers and 3 seeds for
exact-match and refresh:

- Upgrading from a narrow prompt to a wide answer, on the stale cache,
  captures 71-76% of the benefit you'd get from fully reprocessing the
  context at the new width, and is never worse than just staying narrow.
- Downgrading from a wide prompt to a narrow answer is free: quality is at or
  slightly better than running narrow the whole time.
- Recomputing just the last 4 tokens of context at the new width, before
  generating the answer, closes 89% of whatever gap upgrading left behind.
- The cost of a stale cache is concentrated exactly where extra width would
  have helped anyway. Easy parts of the data are unaffected either way.

The deployment policy this points to: switch width freely in either
direction, refresh a handful of trailing tokens on an upgrade, and never
bother reprocessing the whole context from scratch. That gives a single
resident checkpoint a runtime quality/cost dial, with the conversation
intact across switches. Two separate models (a real E2B and a real E4B
running side by side) cannot do this at all, because their caches aren't
compatible.

## 4. The real speedup: decode in llama.cpp

Section 2's flat decode numbers came from HF transformers' standard
generation loop, which isn't built to expose memory-bandwidth savings at
batch size 1. To get a real answer, we exported both the full model and the
50%-width slice to GGUF, the file format llama.cpp uses, built for fast
loading and fast execution on both CPU and GPU.

The slicing itself moved below the PyTorch level this time, straight onto
the raw safetensors files the model ships in. A script opened each weight
shard, matched tensors by shape (`(16384, 2048)` for gate and up,
`(2048, 16384)` for down), and kept only the first 8192 rows or columns of
each match: 105 tensors total, 35 layers times 3 matrices per layer. The
multimodal vision and audio weights, which don't match those shapes, passed
through untouched. We edited `intermediate_size` in `config.json` to match,
then converted the result with llama.cpp's `convert_hf_to_gguf.py`. Cutting
the safetensors files directly kept the checkpoint's layout exactly as the
converter expects.

We benchmarked both GGUF files with `llama-bench` on the same A40, in bf16,
with all layers offloaded to the GPU (`-ngl 99`), measuring a 512-token
prefill (`pp512`) and a 128-token generation (`tg128`), each averaged over 3
repetitions:

| model | size | params | prefill (pp512) | decode (tg128) |
|---|---|---|---|---|
| E4B full | 12.80 GiB | 6.87 B | 3,928 tok/s | 53.4 tok/s |
| E4B @ 50% FFN | 9.51 GiB | 5.11 B | 4,953 tok/s | **76.9 tok/s** |

Decode is 1.44x faster at half width, and prefill is 1.26x faster. That's a
very different result from section 2, because generating one token at a
time is bandwidth-bound: the GPU spends most of its time reading weights
from memory, not doing math. Cutting FFN weight bytes by about 1.34x should
save close to that much decode time in a bandwidth-bound runtime, and a
well-built engine like llama.cpp delivers it, coming in slightly above that
floor. That's likely because Gemma 3n's activation-sparsity computation also
gets cheaper at the smaller width. HF's own loop showed only 1.04x on the
exact same weights, so the slow speed in section 2 came from the software,
not the model or the hardware.

By parameter count, the 5.11-billion-parameter half-width model is
essentially an E2B, extracted straight from the E4B checkpoint by cutting
those 105 tensors, no retraining involved, and llama.cpp loads and runs it
today.

One limit on the "dynamic" version of this idea: llama.cpp currently loads
each width as a separate model file. A single resident checkpoint that
switches widths on the fly (the scenario section 3's KV cache result is
built for) needs changes inside llama.cpp itself: teaching its compute graph
to run FFN matmuls against only the first h rows or columns of one
already-loaded weight buffer, instead of loading a second file. A 44% decode
speedup makes that worth doing. At the 4% speedup section 2 first
suggested, it would not have been.

## 5. What didn't work

We tried two more ideas and dropped both.

**Per-token width routing.** Instead of picking one width for a whole
request, a learned router could pick a different FFN width for every single
token, spending more compute on hard tokens and less on easy ones. Tested on
the 260-run toy grid from section 3: a router trained on top of an
already-trained nested model lost about 0.06 nats to simply picking one
fixed width for the whole request, at matched average compute. Nothing
measured at production scale suggests this result would flip.

**Nested-width speculative decoding.** Speculative decoding speeds up
generation by having a cheap "draft" model guess several tokens ahead, then
having the full model check all those guesses in one pass. That's cheaper
than generating one token at a time if the guesses are usually right. The
idea was to use a narrow slice of the same nested model as the draft.
Problem: even a 50%-width slice still costs roughly half of a full forward
pass. Gemma 4 ships its own multi-token prediction (MTP) heads: small extra
output heads that guess several tokens ahead in the same pass the main
model already runs, at roughly 2% of a full pass's cost. MTP wins by a wide
margin, and there's no case for nested-width drafting instead.

What still stands, from sections 1 through 4:

- A real, roughly 2x range of quality-vs-compute inside one downloaded
  checkpoint, worth a measured 44% faster decode at the cheap end, in a
  runtime built to expose it.
- Switching that width mid-conversation for close to free, without losing
  the conversation's cache. Nobody else offers this today: Google's own
  tooling only lets you pick a width once, before serving starts.
- Real memory and prefill savings for a model deployed at one fixed width,
  including a working E2B-sized model with no retraining, that llama.cpp
  runs right now.
- Direct evidence, from the collapse test in section 1, of what nested
  training specifically buys over a same-size model trained the ordinary
  way. That's a case for bringing the idea back in a future model family,
  not a reason to treat it as settled history.

*The toy-scale groundwork behind this (260 training runs across width
tiers, seeds, and routing setups, on nested transformers with about 0.36
million parameters) ran on CPU at no GPU cost, and lives in `mow/` in this
repo, along with the mid-conversation switch tests from section 3. Only the
production-model tests on Gemma, sections 1 through 4 above, used rented GPU
time: roughly $20-25 total.*
