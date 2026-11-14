# Larger contrastive batches

Prepared during the depth experiment. Launch only after all depth candidates
are frozen and scored, if quality still warrants this comparison. This is a
follow-up within the user's explicitly authorized cumulative$10 campaign.

Start all arms from the selected two-layer data-first parents, exactly as the
two-layer depth-job controls. Retain the same780,743 positive pairs, full two
passes, temperature0.05, seed29, BF16, token cap128 and held-source selection.
No teacher targets or extra inference parameters. Batch1024 has eight times
fewer updates than batch128 at equal pair exposure: test LR1e-5 and4e-5 for both
float and ternary, rather than attributing LR/update-count effects to batch alone.
The completed depth two-layer arms are the small-batch controls.

Representation-gradient replay uses microbatches128 to evaluate one shared
contrastive loss over the large batch. It reproduces full-graph gradients in
float/ternary CPU tests; a CUDA BF16 test passes on the active4090 with matching per-microbatch autocast boundaries.
A full-width ternary interruption test reproduces weights, optimizer, sampling
and RNG state exactly. This is not gradient accumulation over separate losses.
See Gao et al., https://aclanthology.org/2021.repl4nlp-1.31/ .

Select the best held-source macro nDCG checkpoint with every source within two
points of initialization. Initialization is an eligible null control. Freeze
all six choices before observed external development scoring. The existing
conditional FiQA/SCIDOCS reservation still applies. No broad superiority claim
follows from tuning on observed tasks.

Proposed resource: one Secure RTX4090, currently$0.89/hour, accepted maximum
$0.90/hour, expected1.5–2 hours/$1.40–$1.90. Reserve$2.50 with a2.5-hour maximum
including60GB temporary disk. Work stops at130 minutes, deletion watchdog at140.
Recover SHA256-verified resumable checkpoints locally at least every few minutes;
stop if no off-pod backup succeeds for ten minutes. No retained paid volume.

The broader baseline checks found data-first forgetting: float12-task mean fell
from71.403 to70.668, ternary70.156 to69.491, despite better retrieval. Add two
predata-parent arms at LR4e-5, one per precision, under the same large-batch/data
exposure. These test restarting from the stronger general-purpose geometry.
The reference means are bge-micro70.904 and MiniLM71.481. These are averages over
a fixed12-task subset, not the full MTEB. No reserved FiQA/SCIDOCS labels were used.

Numerical detail: the cached representation loss computes dot-product logits in
FP32 outside autocast. The earlier small-batch depth controls computed that loss
inside BF16 autocast. Thus this is a larger-batch training-recipe comparison;
it cannot attribute every difference to batch size alone. A promising recipe
should also get matched FP32 small-batch controls in its confirmation run.

LR clarification:4e-5 is four times1e-5; it is an intermediate scaling choice,
not exactly sqrt(8), which is about2.83. The actual training arguments were
1e-5 and4e-5 throughout.
