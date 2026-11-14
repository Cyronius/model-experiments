# Plan: run whole‑network distillation on mc‑distill‑08b

## Goal
Repair the 0.8 B student by distilling the 2 B teacher on a RunPod pod and download checkpoints for local evaluation.

## Steps
1. **Prepare pod** – SSH to the pod, create a work directory, clone the repository.
2. **Re‑build student** – Run `manifold-compression/build_projected.py` with default args, output to `student` directory.
3. **Run distillation** – Execute `manifold-compression/distill_whole.py`:
   - `--teacher Qwen/Qwen3.5-2B`
   - `--student <student_path>`
   - `--out distill_out`
   - `--tokens 100e6`
   - default checkpoints (`5e6,20e6,50e6,100e6`).
4. **Monitor** – Stream pod logs with `runpodctl pod logs <id> --follow`.
5. **Download checkpoints** – SCP the `distill_out` folder to `C:\code\model-experiments\distill_checkpoints`.
6. **Local evaluation** – Convert each checkpoint to GGUF, run smoke test (40 prompts) with `llama.cpp`, judge with Haiku.
7. **Cleanup** – Stop the pod and remove remote work files.

## Cost estimate
- Pod time (≈ 2 h on A40): $0.80 – $1.60
- Data transfer: negligible
- Local evaluation: none

## Decision rules
- If coherence ≥ 3.25 at 100 M tokens → consider larger‑scale runs.
- If coherence ≤ 2.2 → stop distillation line.

## Risks
- GPU memory for KL over 248 K vocab; script uses 4 K‑token chunks.
- Pod may terminate early; script will stop pod manually after download.

## Output
Checkpoints `ckpt_5M`, `ckpt_20M`, `ckpt_50M`, `ckpt_100M` saved locally, plus `history.json` with KL curve.
