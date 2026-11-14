# Learned ternary scales and low-rate continuation

All five arms completed, were backed up locally, and were frozen before any
external scoring. The first bounded pod completed three arms and saved the
fourth at step68; a separately bounded allocation resumed that full optimizer,
sampler, classifier, and RNG state and finished the remaining two arms. Both
pods were deleted with GET404 proof. Estimated combined cost **$1.8722**;
cumulative completed campaign cost **$9.3226** before the focused human-data job.

| Arm | Retrieval mean | STSBenchmark | 12-task mean |
|---|---:|---:|---:|
| Two-layer ternary, ordinary continuation | 44.722 | 81.365 | 70.061 |
| Two-layer ternary, learned row scales | 44.974 | 81.339 | 70.022 |
| Three-layer ternary, learned scales/all weights | 44.215 | 81.447 | 70.058 |
| Three-layer ternary, only row scales trainable | 44.739 | 81.512 | 70.071 |
| Three-layer float, low-rate continuation | 46.434 | 82.610 | 71.085 |

Learning row scales adds **0.253 retrieval points** against the matched two-layer
ordinary continuation, with query95% interval **[-0.126,+0.643]**. The wider mean
changes by **-0.038**. This setting does not close the reference gap or establish
a statistically convincing retrieval improvement.

Freezing the three-layer master weights improves retrieval by **0.524** against
training scales and all weights, interval **[-0.045,+1.103]**, with only **+0.013**
on the wider mean. Both remain below the stronger three-layer blend. The
three-layer float continuation also trails the float blend. More training and
this particular scale-learning recipe are not the quality breakthrough sought.
This does not rule out other scale schedules, data, or optimization methods.

Each arm performs one full epoch of1,023,999 relevance pairs,1004 updates,
batch1024/micro128, encoderLR1e-5, seed43. Learned scales and the disposable
NLI classifier use LR1e-3 before cosine decay. NLI encoder weight is0.25 for
ternary and0.1 for float. Scale-only training updates10,368 row scales in the
three-layer encoder; all inference matrices still have three-valued codes and
one BF16 scale per row. The canonical export uses the existing native format.

The NLI encoder sees192,768 pair exposures in this stage; head warmup adds19,200
head-only examples. All five use identical data orders and the prescribed held
source/NLI floors. The two-layer fixed/learned comparison isolates the scale
recipe. Three-layer learned-versus-frozen also changes which other weights can
train; parent contrasts include additional training and do not isolate scales.

Scores cover observed development tasks and the fixed12-task subset. Query
intervals exclude training-seed variability. Reserved judgments remain unopened.
[Frozen cohort and continuation proof](results/frozen.json),
[all scores, matched contrasts, and parent comparisons](results/summary.json).
