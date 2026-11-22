"""Stock Klein 4B on the NPU at 512x512 with 4, 16, 24, 32, 48 and 64 steps: one contact sheet per prompt (2026-11-21)."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw

CFG = json.loads(Path(r"C:\dev\klein-eval\resident-20261118\nodibs.json").read_text())
WT = Path(r"C:\code\phlegm-wt-serve")          # a worktree of the branch with --serve
OUT = Path(__file__).resolve().parents[1] / "artifacts" / "flux-conditioning" / "npu-steps"
STEPS = [4, 16, 24, 32, 48, 64]
PROMPTS = ["three red apples on a wooden table",
           "a lighthouse on a rocky coast at sunset, waves crashing",
           "a busy street market with detailed stalls, people and hanging lanterns",
           "portrait of an old fisherman with a weathered face, harbor in the background"]


def main() -> None:
    cmd = (f". '{CFG['env_script']}'; & '{CFG['python']}' -m open_kernels.flux2_native_generate --serve "
           f"--model '{CFG['model']}' --kernels '{CFG['kernels']}' --transformer-gguf '{CFG['transformer_gguf']}' "
           f"--gguf-modulation {CFG['gguf_modulation']}")
    p = subprocess.Popen(["powershell.exe", "-NoProfile", "-Command", cmd], cwd=WT / "open-kernels" if (WT / "open-kernels").is_dir() else WT,
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, encoding="utf-8")
    assert json.loads(p.stdout.readline()).get("ok")
    times = {}
    try:
        for s in STEPS:
            for i, prompt in enumerate(PROMPTS):
                d = OUT / f"p{i}" / f"s{s}"
                if (d / "image.png").is_file():
                    continue
                p.stdin.write(json.dumps({"prompt": prompt, "size": 512, "steps": s, "seed": 1, "out": str(d)}) + "\n")
                p.stdin.flush()
                r = json.loads(p.stdout.readline())
                if not r.get("ok"):
                    raise RuntimeError(r)
                times.setdefault(s, []).append(r.get("npu_s"))
                print(f"steps {s} prompt {i}: {r.get('npu_s')} s", flush=True)
    finally:
        p.stdin.close()
        p.wait(timeout=120)
    for i, prompt in enumerate(PROMPTS):
        sheet = Image.new("RGB", (512 * len(STEPS), 512 + 40), "white")
        draw = ImageDraw.Draw(sheet)
        for j, s in enumerate(STEPS):
            sheet.paste(Image.open(OUT / f"p{i}" / f"s{s}" / "image.png").convert("RGB"), (512 * j, 40))
            draw.text((512 * j + 8, 12), f"{s} steps", fill="black")
        sheet.save(OUT / f"sheet-p{i}.png")
    (OUT / f"times-{min(times, default=0)}.json").write_text(json.dumps({"prompts": PROMPTS, "npu_s": times}, indent=1))


if __name__ == "__main__":
    main()
