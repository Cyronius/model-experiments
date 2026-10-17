# pod/: the $5 data session

Makes checker-clean STE rewrites, messy back-translations, and a judge sample. It does
not train a model: the student code is not written yet. Training is a later, cheaper session.

## What you do

1. Rent one GPU pod with an image that has the CUDA toolkit (`nvcc` must exist). A 24 GB
   card is probably enough for the 16.5 GB teacher at Q4_K_M. A 48 GB card is safe. Read
   the price per hour off the pod page.
2. Open a shell on the pod and run:

   ```
   git clone --depth 1 --branch ccr-822a92e7-k01v1x https://github.com/Cyronius/qwen-prune-heal-pipeline /workspace/repo
   cd /workspace/repo/asd-ste100/pod
   RATE=<price per hour> BUDGET=2 bash pod_smoke.sh      # about 20 to 30 minutes
   ```

3. Read the last lines. They give measured tokens per second, success rate, and an hours
   and dollars estimate for 12,000 and 30,000 sources.
4. If the estimate fits, run the real session:

   ```
   RATE=<price per hour> BUDGET=5 N=12000 bash pod_run.sh
   ```

5. Copy `/workspace/ste-out.tgz` home, then stop the pod. Set `HF_TOKEN` and `OUT_REPO`
   to upload it to a private Hugging Face dataset instead. Set `AUTOSTOP=1` on RunPod to
   have the script stop the pod itself.

## How the budget is enforced

- `RATE` is required. The scripts refuse to start without it.
- Cost is wall-clock time since the first run times `RATE`. The clock survives a rerun.
- Before every phase the script checks the spend. At 85 percent of `BUDGET` it stops,
  packs what exists, and tells you to stop the pod. Generation phases also run under a
  `timeout` for the time left. The remaining 15 percent is slack for downloads and packing.
- Phases are flag-gated and the generators resume, so a stopped run can continue later.
- The pod bills until you stop it. The script cannot do that for you unless `AUTOSTOP=1`.

## Not verified

- The model files are found by glob (`*Q4_K_M*`, `*mxfp4*`). The Hugging Face repos were
  unreachable from where this was written. If a glob matches nothing, the script says so and
  stops. Fix the glob in `common.sh`.
- llama.cpp tag `b10673` is copied from `bench/pod_bench.sh`. Newer tags may be needed.
- Every time estimate is a guess until the smoke session runs.
