# pod/: the $5 data session

Makes checker-clean STE rewrites, messy back-translations, and a judge sample. It does
not train a model: the student code is not written yet. Training is a later, cheaper session.

## What you do

1. Rent one GPU pod with an image that has the CUDA toolkit (`nvcc` must exist), for example
   `runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404`. The teacher is Qwen3.6-35B-A3B-FP8 (35 GB)
   on vLLM, so use a 48 GB card with FP8 support (L40S, RTX 6000 Ada) or an H100. Ask for a
   driver with CUDA 12.8 or later (`--min-cuda-version 12.8`). Use a volume of 100 GB. Read the
   price per hour off the pod page.
2. Open a shell on the pod and run:

   ```
   git clone --depth 1 --branch main https://github.com/Cyronius/model-experiments /workspace/repo
   cd /workspace/repo/asd-ste100/pod
   RATE=<price per hour> BUDGET=2 bash pod_smoke.sh      # about 25 minutes, about $0.45 on an L40S
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

## Known problems

- RunPod's nginx listens on port 8081, so the judge uses port 18081.
- vLLM needs `ninja` and the CUDA toolkit at startup. `phase_vllm` installs ninja into the venv.
- vLLM fails to start if the judge's `llama-server` still holds GPU memory.
- `pod_run.sh` has not had a full run yet. Its time estimates come from the smoke session.
