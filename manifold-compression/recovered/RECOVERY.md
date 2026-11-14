# Manifold checkpoint recovery

Source: user-provided FTP share on the former Windows machine, 2026-10-24.

The Windows working copy is newer than this Linux checkout. Its findings include completed whole-network distillation to 500,006,912 tokens and final evaluation through 2026-10-24. Recovered notes and scripts are kept here without replacing checkout files.

Selected model artifacts:

- `models/manifold-compression/distill-08b/continuation-500m/ckpt_500M/`: final Hugging Face export; expected safetensors size 2,013,387,376 bytes.
- `models/manifold-compression/mc-proj-0.8b-tail.gguf`: original polished BF16 model; expected size 1,516,744,352 bytes.

Original remote paths have prefix `/code/model-experiments/` followed by these same model paths. Each download is written to `.partial`, checked against the FTP directory size, then renamed on completion. A full optimizer resume checkpoint is not included; the experiment notes say it remained on the stopped RunPod persistent volume.

Other models found on FTP but not selected for download include distillation exports at 5M–450M tokens and the 9B-to-4B projection (`proj-4b`, `proj-4b-tail`, and `mc-9b-4b-tail-q8.gguf`).

See `findings-windows.md` for the updated experiment status. Final held-out KL was 0.2688; final sampled chat coherence was 2.35/5 versus stock 0.8B at 3.075/5. Corrected full IFEval strict prompt accuracy was 77/541 (14.23%) versus stock 249/541 (46.03%).
