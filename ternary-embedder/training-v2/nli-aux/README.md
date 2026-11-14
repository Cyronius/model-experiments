# Human NLI supervision without a larger inference model

The wider baseline checks found similarity forgetting during relevance-only
training. Test additional human entailment/neutral/contradiction labels with a
training-only classifier, following the classification objective in
[Sentence-BERT](https://aclanthology.org/D19-1410.pdf). No teacher vectors or
model-generated labels are used in this phase.

The audited data contains300,000 official AllNLI training pairs,100,000 per
label, sampled after globally shuffling the complete memory-mapped upstream
training table. A short streaming buffer would only sample an early prefix.
The official dev split retains18,624 pairs after exclusions. Auxiliary dev has
no exact model-visible inputs in retrieval training, auxiliary train has none
in either holdout, and known benchmark inputs are excluded. Source revisions,
checksums, label mappings and rejections are in`data-audit.json`. Semantic near
duplicates are not exhaustively excluded.

Six arms start from the fixed original two-layer float/ternary parents, chosen
for their stronger wider quality. This comparison is independent of the ongoing
larger-batch job. Both precisions test encoder NLI-gradient weights0,0.1,0.25;
all arms train the same classifier on the same labels. Weight0 is a probe-only
control. Other settings are batch1024, microbatch128, LR4e-5, temperature0.05,
two full relevance-data passes, seed29, BF16 encoder and FP32 losses. Explicit
BM25 negatives are disabled here so that the human-label effect is isolated.

The classifier is a linear3-class head over`[u,v,abs(u-v)] *10`. Warm it for100
updates with the encoder frozen. Then train it every relevance update, passing
the configured fraction of its gradient to the encoder. Its LR is25 times the
encoder LR; clip head and encoder separately. These fixed training choices are
experiments, not established improvements.

Select the highest mean of held-source macro nDCG and held-NLI macro accuracy,
subject to each retrieval source staying within2 points and NLI accuracy within
1 point of post-warmup initialization. Include the initial encoder null control.
Freeze all six choices before observed external scoring. The probe's accuracy
is an internal selection signal; the full12-task suite checks real embedding
quality. FiQA/SCIDOCS labels retain the existing conditional reservation.

The classifier is excluded from all inference checkpoints. The encoder remains
15,467,520 parameters with the ordinary ternary format. Resumable checkpoints
retain the classifier, its AdamW state and the separate sampling cursor/RNG.
Actual384-wide CPU interruption tests reproduce the model, optimizer, both
samplers, head and RNG exactly, and verify that head warmup leaves encoder
weights unchanged. Float/ternary CUDA BF16 warmup, joint-gradient and state
checks passed on the active4090 before launch. They run again on the new pod.

One additional Secure RTX4090 at$0.89/hour, maximum accepted$0.90/hour, expected
1.8–2.2 hours/$1.65–$2.05. Hard maximum3 hours/**$3.00 including60GB temporary
disk**. Work deadline160 minutes, independent deletion watchdog170 minutes,
no retained paid volume. Together with the active larger-batch reservation and
closed depth cost, the worst-case campaign commitment is$6.3017 of the explicitly
authorized$10. Both controllers save and verify local recovery copies; finish
by deleting each owned pod and verifying HTTP404.

Exposure detail: there are1,534 relevance updates per arm. Auxiliary batch64
therefore exposes the head to104,576 label pairs including6,400 warmup pairs;
the encoder receives98,176 auxiliary pairs when its NLI gradient weight is
nonzero. The300,000-row audit describes the available dataset, not a claim that
each arm completes a full auxiliary epoch. Exact counters are saved in logs
and resumable states.
