# Data-first RunPod job — completed, pod deleted

One RTX A5000 24 GB, listed at $0.27/hour, maximum accepted GPU rate $0.30/hour.
Proposed maximum elapsed time four hours, expected two to three hours, expected
$0.60–$1.00 total including temporary disk, approved hard total cap **$1.50**.
60 GB container disk, no retained paid volume. Work stops at 220 minutes;
an independent local watchdog deletes the pod at 230 minutes, leaving retry
margin before four hours. Setup, transfer and recovery share that time budget.
Pricing checked at https://www.runpod.io/pricing on 2026-10-25.

The user approved this bounded job on 2026-10-25. Pod `pjeduelfgo0u00` is
now deleted; it ran on Secure Cloud at $0.27/hour. The controller refuses launch without a matching explicit
job-specific approval record. Credentials remain local in their existing
0600 file/0700 directory; the upload contains no API key or SSH private key.

## Training

Two matched continuation branches: independent float checkpoint and current
ternary checkpoint. Each sees two complete shuffled passes over the 780,743
training pairs, with batch 128, learning rate 1e-5, temperature 0.05 and BF16
CUDA autocast. Every source is covered each epoch; source batches are shuffled
and interleaved. A final singleton is folded into the preceding source batch.
The architecture and 128-token limit remain unchanged. No teacher loss.

Validation ranks each held source against every unique positive document in
that source. Save validation results at initialization, every 1500 updates,
and completion. Select highest equal-source mean nDCG@10 among checkpoints
with no source dropping more than two points below its initialization. Include
the initialization as a null control. This is training-domain validation;
external development tasks and reserved tests remain separate local checks.
Int4/int8 conversion and accuracy checks follow locally from the float branch.

## Checkpoints and off-pod recovery

- Save every **500 updates or 120 seconds**, and on graceful interruption/end.
- Save model, optimizer, sampling permutations/cursors, CPU/CUDA RNG state,
  learning-rate position, counters and validation/selection progress.
- Write to a temporary file, flush/fsync, rename atomically, then publish its
  SHA256 manifest. Keep the latest two resumable checkpoints on the pod.
- Attempt a local backup every **60 seconds**. Pin the checkpoint with an
  immutable remote hardlink before transfer, so training rotation cannot
  remove it mid-download. Download to a local temporary file, verify SHA256,
  and rename only after verification. A failed/corrupt transfer preserves the
  previous local copy. Keep the latest two local resume files per branch,
  along with best/final weights and validation milestone weights.
- If off-pod checkpoint backup remains unavailable for ten minutes, stop the
  job and retain the verified local copies. The independent budget watchdog
  remains in force even if training or SSH hangs.
- Recover artifacts on exit, delete only this job's pod, and verify HTTP 404.
  A provider failure before the next completed transfer can still lose recent
  updates; completed local backups survive pod/container loss.

Local preflight: six data/loss tests, three sampler/atomic-checkpoint tests and
three backup-transfer/snapshot tests pass. A six-step ternary run interrupted at step three
and resumed exactly reproduces uninterrupted weights, optimizer and RNG state.
A float smoke exercises the full validation, floor guard and checkpoint path.
These are CPU correctness checks, not CUDA throughput or quality measurements.
`recovery-preflight.json` records the interruption/resume result.

Controller: `../pair_runpod.py`; workload: `../run_data_first.sh`; training:
`../pair_train.py`. The package includes audited train/dev files, both starting
checkpoints, tokenizer, code and input hashes. Its local path and SHA256 are
under ignored `training-v2/work/data-first-run/package.json`.


Allocation note: the legacy v1 API's CUDA enum ends at 13.0 while the available
host reports 13.2. The controller now uses REST v2 and an explicit official
PyTorch image, avoiding obsolete template constraints. Failed allocation calls
created no pods; the successful pod keeps the original approved resource and cap.


Completed in 40.81 minutes, estimated $0.1893 including temporary disk. All final artifacts were recovered and SHA256-verified. See [external quality and results](RESULTS.md).
