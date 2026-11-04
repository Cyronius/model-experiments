"""Recompute the score of every saved answer (after a scorer change).

    python3 rescore.py runs/*/
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import run
import score

HERE = Path(__file__).parent


def one(p: str) -> str:
    rp = Path(p)
    res = json.loads(rp.read_text())
    res["score"] = score.score(HERE / "tasks" / res["id"], (rp.parent / "answer.txt").read_text())
    rp.write_text(json.dumps(res, indent=1) + "\n")
    return p


if __name__ == "__main__":
    for d in sys.argv[1:]:
        files = [str(p) for p in Path(d).glob("*/result.json")]
        with ProcessPoolExecutor(12) as pool:
            list(pool.map(one, files))
        run.summarize(Path(d))
