# Matched hard-negative comparison

Four arms start from the fixed original two-layer float/ternary parents at
batch1024, microbatch128, LR4e-5, temperature0.05, seed29, BF16 encoder and FP32
cached contrastive logits. Each precision compares all original positive pairs
against those same pairs plus their filtered original BM25 training negatives.
All arms see two full passes over the same780,743 queries/positive pairs. This
comparison is independent of the ongoing batch and human-NLI experiments.

The data audit restored182,553 original MS MARCO BM25 negatives, rejecting
known positives, near-duplicate positives, incomplete inputs, held inputs and
benchmark overlaps. Every known positive for a query is accepted in the loss,
including passages introduced as another query's negative. Retain all positive
pairs even when no candidate negative passes the filters. Unjudged candidates
are not guaranteed irrelevant.

Both conditions validate against the same held corpus, augmented with2,847
original BM25 candidates strictly disjoint from all training inputs. Select the
maximum held-source macro nDCG with each source within2 points of initialization;
the initial null control is eligible. Freeze all four choices before external
scoring, then evaluate the observed retrieval tasks and fixed12-task suite.
FiQA/SCIDOCS relevance judgments remain conditionally reserved.

Full-width ternary cached-gradient training, augmented validation and an
interruption/resume reproduce weights, optimizer, sampling and RNG exactly on
CPU. CUDA BF16 cached-gradient checks passed on the active4090 and run again
before this workload. Architecture and ternary inference format are unchanged;
the hard negatives are training data, not inference-time retrieval machinery.

One additional Secure RTX4090 at$0.89/hour, maximum accepted$0.90/hour; expected
1–1.7 hours/$0.90–$1.60. Maximum2 hours/**$2.00 including60GB temporary disk**.
Work stops at100 minutes, independent deletion watchdog at110. Checkpoints are
saved every100 updates or120 seconds, with SHA256-verified local recovery copies.
No retained paid volumes. The full campaign remains bounded by the user's
explicit$10 authorization; current full reservations plus the completed depth
cost total$8.3017. Durable reservation records and file locks prevent concurrent
controllers from losing a new spending reservation.
