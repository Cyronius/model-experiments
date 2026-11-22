"""Step 0 of the compact-conditioning NPU plan: projected conditioning files through the native Klein runner,
one resident Runner, exact noise; pixel MAD against the CPU diffusers images of the same rows (2026-11-20)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

CFG = json.loads(Path(r"C:\dev\klein-eval\resident-20261118\nodibs.json").read_text())
A = Path(r"C:\code\model-experiments\artifacts\flux-conditioning")
OUT = A / "npu-projected"


def jobs() -> list[dict]:
    cases = json.loads((A / "encoder-probe/run1/targets/manifest.json").read_text(encoding="utf-8"))["cases"]
    out = []
    for c in cases:
        for tag in ("a-teacher", "c-probe"):
            out.append({"name": f"{c['id']}-{tag}", "prompt": c["prompt"],
                        "cond": str(A / f"encoder-probe/run1/cond/{c['id']}-{tag}.safetensors"),
                        "cpu": str(A / f"encoder-probe/run1/png/{c['id']}-{tag}__noise-a.png")})
    en = A / "enricher-v2/run2-cfg/images"
    for c in cases:
        cid = c["id"].removeprefix("b-")
        for tag in ("0base", "g2-s0-sample"):
            if (en / "cond" / f"{cid}-{tag}.safetensors").exists():
                out.append({"name": f"enr-{cid}-{tag}", "prompt": c["prompt"],
                            "cond": str(en / "cond" / f"{cid}-{tag}.safetensors"),
                            "cpu": str(en / "png" / f"{cid}-{tag}__noise-a.png")})
    return out


def worker_side(job_file: str) -> None:
    sys.path.insert(0, str(Path(CFG["repo_root"]) / "open-kernels" / "open_kernels"))
    import flux2_native_generate as g
    from klein_conditioning import load_projected_conditioning
    js = json.loads(Path(job_file).read_text(encoding="utf-8"))
    noise = np.load(OUT / "noise-a.npy").astype(np.float32)
    gguf = (CFG["transformer_gguf"], g.gguf_modulation(Path(CFG["transformer_gguf"]), Path(CFG["model"]), CFG["gguf_modulation"]))
    r = g.Runner(Path(CFG["model"]), Path(CFG["kernels"]), 512, 4, False, 4, projected=True, gguf=gguf, elf=True)
    for j in js:
        r.set_projected_conditioning(load_projected_conditioning(Path(j["cond"])), j["prompt"])
        r.set_noise(g.pack_noise(noise))
        ph = r.run()
        Image.fromarray(r.image()).save(OUT / f"{j['name']}.png")
        print(json.dumps({"name": j["name"], "phases": {k: round(v, 4) for k, v in ph.items()}}), flush=True)


def main() -> None:
    if len(sys.argv) > 1:
        return worker_side(sys.argv[1])
    from safetensors.torch import load_file
    OUT.mkdir(exist_ok=True)
    np.save(OUT / "noise-a.npy", load_file(str(A / "enricher-smoke-20261110/noise-a.safetensors"))["latents"][0].float().numpy())
    js = jobs()
    (OUT / "jobs.json").write_text(json.dumps(js, indent=1), encoding="utf-8")
    cmd = (f". '{CFG['env_script']}'; & '{CFG['python']}' '{Path(__file__).resolve()}' '{OUT / 'jobs.json'}'")
    p = subprocess.run(["powershell.exe", "-NoProfile", "-Command", cmd], cwd=Path(CFG["repo_root"]) / "open-kernels",
                       capture_output=True, text=True, encoding="utf-8")
    lines = [json.loads(l) for l in p.stdout.splitlines() if l.startswith("{")]
    if p.returncode or len(lines) != len(js):
        print(p.stdout[-3000:], p.stderr[-3000:])
        raise SystemExit("worker failed")
    rows = []
    for j, l in zip(js, lines):
        a = np.asarray(Image.open(OUT / f"{j['name']}.png").convert("RGB"), dtype=np.float32)
        b = np.asarray(Image.open(j["cpu"]).convert("RGB").resize((512, 512)), dtype=np.float32)
        rows.append({"name": j["name"], "mad_npu_vs_cpu": round(float(np.abs(a - b).mean()), 2), "phases": l["phases"]})
        print(rows[-1]["name"], rows[-1]["mad_npu_vs_cpu"], l["phases"].get("total"))
    (OUT / "results.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    for kind in ("a-teacher", "c-probe", "0base", "g2-s0"):
        v = [r["mad_npu_vs_cpu"] for r in rows if kind in r["name"]]
        print(f"{kind}: NPU vs CPU MAD mean {np.mean(v):.1f} (n {len(v)})")
    t = [r["phases"] for r in rows]
    print("median phases:", {k: round(float(np.median([x[k] for x in t if k in x])), 3) for k in t[0]})


if __name__ == "__main__":
    main()
