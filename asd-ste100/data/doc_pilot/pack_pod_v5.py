"""Pack what a pod needs for pod/doc_teacher_run.sh into data/out/doc-pool-v5/doc-teacher.tgz.

The doc pilot code, and the local modules it imports that are not on GitHub main
(found by importing synthesize_v5 and listing the modules loaded from this
repository), the lexicon files the prompts are built from, the pod script and the
passages (`inputs.jsonl`, from build_pool_v5.py). The lexicon is AGPL-derived:
the tarball is for our own pod, not for publication.
"""

from __future__ import annotations

import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/out/doc-pool-v5"


def modules() -> list[Path]:
    code = ("import sys; sys.argv = ['x']; import synthesize_v5, judge_eval, build_pool_v5; "
            "print('\\n'.join(sorted({m.__file__ for m in list(sys.modules.values()) if getattr(m, '__file__', None)})))")
    env = {"PYTHONPATH": "data:checker:tokenizer:data/doc_pilot", "PATH": "/usr/bin:/bin"}
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, check=True).stdout
    return sorted({Path(p).resolve() for p in out.split() if Path(p).resolve().is_relative_to(ROOT)})


def main():
    files = set(modules()) | set((ROOT / "data/doc_pilot").glob("*.py")) | {ROOT / "pod/doc_teacher_run.sh"}
    files |= {ROOT / "lexicon" / n for n in ("lexicon.jsonl", "rules.json", "tech_words.txt")}
    pool = OUT / "pool.jsonl"
    with tarfile.open(OUT / "doc-teacher.tgz", "w:gz") as tar:
        for f in sorted(files):
            tar.add(f, arcname=str(f.relative_to(ROOT)))
        data = pool.read_bytes()
        info = tarfile.TarInfo("inputs.jsonl")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    print(json.dumps({"tarball": str(OUT / "doc-teacher.tgz"), "files": len(files) + 1,
                      "passages": sum(1 for _ in pool.open()), "mb": round((OUT / "doc-teacher.tgz").stat().st_size / 1e6, 1)}))


if __name__ == "__main__":
    main()
