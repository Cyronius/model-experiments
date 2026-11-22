"""One serialized per-op profile of the stock Klein native run, to read per-dispatch floors (2026-11-21)."""
import json
import subprocess
from pathlib import Path

CFG = json.loads(Path(r"C:\dev\klein-eval\resident-20261118\nodibs.json").read_text())
OUT = Path(r"C:\code\model-experiments\artifacts\flux-conditioning\npu-profile")

cmd = (f". '{CFG['env_script']}'; & '{CFG['python']}' -m open_kernels.flux2_native_generate --model '{CFG['model']}' "
       f"--kernels '{CFG['kernels']}' --transformer-gguf '{CFG['transformer_gguf']}' --gguf-modulation {CFG['gguf_modulation']} "
       f"--prompt 'three red apples on a wooden table' --out '{OUT}' --profile")
subprocess.run(["powershell.exe", "-NoProfile", "-Command", cmd], cwd=Path(CFG["repo_root"]) / "open-kernels", check=True)
s = json.loads((OUT / "summary.json").read_text())
print("phases", s["npu_phases_s"], "dispatches", s["dispatches"])
for k, v in list(s["profile"].items())[:60]:
    print(f"{v['s']:8.4f} s {v['ops']:5d} ops {1000 * v['s'] / v['ops']:7.3f} ms/op  {k}")
