"""Which generator script produced a build, and the generator-level train/test split.

The first split (make_tasks.py, 2026-11-03) held out design *directories*. That leaks: the
same generator writes into many directories (openflowlm-next vs openflowlm-pr209 `attn_block`,
gemma4 bundle v1 vs v2, `kernel-builds/llama32-1b-gguf-*`), so test tasks had near-twins in dev.
Since 2026-11-08 the split is by generator: a test generator's builds never enter training,
in any directory. Dev generators are trained on at other shapes, so dev measures new shapes of
known generators and test measures generators never seen in training.

Generators (scripts under ~/code/phlegm/open_kernels and ~/code/openflowlm-next/open_kernels):
  gemma4_text_kernels    build_gemma4_text_kernels.py: attention, head/hidden norm, rope,
                         PLE input, GELU MLP (and the older gemma4-* builds of the same ops)
  gemma4_multimodal_ops  build_gemma4_multimodal_ops.py: audio, vision, CLIP linear ops
  gemma4_projections     build_gemma4_text_projections.py: n#_k# / bf#_n#_k# GEMV projections
  lm_head                designs/lm_head_q4, lm_head_q8 (and gemma4 lm-head builds)
  attn_gemm              designs/attn_block/attn_gemm.py (build_pv#_m#, kernel-builds ag_pv#)
  ln                     designs/ln/ln.py
  gptoss                 build_gptoss_kernels.py
"""

from __future__ import annotations

import re

TEST = frozenset({"gemma4_text_kernels", "gemma4_multimodal_ops", "lm_head"})

MULTIMODAL = ("audio", "vision", "linear_clip", "pos_add", "pool-norm", "pool_norm")


def generator(design: str, template: str) -> str:
    """design: build directory's parent (any prefix); template: build name with numbers as #."""
    t = template
    if "lm_head" in design or "lm-head" in t or "lm_head" in t:
        return "lm_head"
    if "attn_block" in design or t.startswith("ag_pv"):
        return "attn_gemm"
    if re.search(r"designs/ln$", design):
        return "ln"
    if t.startswith("gptoss"):
        return "gptoss"
    if "projection" in design or re.fullmatch(r"(gemma#_)?(bf#_)?(proj_)?n#_k#(_v#)?", t):
        return "gemma4_projections"
    if "multimodal" in design or any(x in t for x in MULTIMODAL):
        return "gemma4_multimodal_ops"
    if "gemma" in design or t.startswith("gemma"):
        return "gemma4_text_kernels"
    return "other:" + design


def split(design: str, template: str) -> str:
    return "test" if generator(design, template) in TEST else "dev"


def resplit(tasks_dir) -> None:
    """Relabel an existing task set (index.json and each task.json) with the generator split."""
    import collections
    import json
    from pathlib import Path

    root = Path(tasks_dir)
    index = json.loads((root / "index.json").read_text())
    for m in index:
        m["generator"] = generator(m["design"], m["template"])
        m["split"] = split(m["design"], m["template"])
        tj = root / m["id"] / "task.json"
        meta = json.loads(tj.read_text())
        meta.update(generator=m["generator"], split=m["split"])
        tj.write_text(json.dumps(meta, indent=1) + "\n")
    (root / "index.json").write_text(json.dumps(index, indent=1) + "\n")
    print(dict(sorted(collections.Counter((m["split"], m["difficulty"]) for m in index).items())))


if __name__ == "__main__":
    import sys

    resplit(sys.argv[1])
