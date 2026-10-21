# MTEB subset for the embed-pipeline quality floor (plan step 0b), 2026-10-16.
#   python run_mteb.py run [model ...]
#   python run_mteb.py summary
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
RESULTS = HERE / "results"
TASKS = {
    "STSBenchmark": "STS", "SICK-R": "STS", "STS12": "STS", "STS13": "STS",
    "STS14": "STS", "STS15": "STS", "STS16": "STS",
    "SciFact": "Retrieval", "NFCorpus": "Retrieval",
    "Banking77Classification": "Classification",
    "AskUbuntuDupQuestions": "Reranking",
    "TwitterSemEval2015": "PairClassification",
}
MODELS = {
    "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    "bge-small-en-v1.5": "BAAI/bge-small-en-v1.5",
    "ternlight-mini": "mini",
    "ternlight-base": "base",
    "potion-base-8M": "minishlab/potion-base-8M",
    "paraphrase-MiniLM-L3-v2": "sentence-transformers/paraphrase-MiniLM-L3-v2",
    "paraphrase-MiniLM-L6-v2": "sentence-transformers/paraphrase-MiniLM-L6-v2",
    "bge-micro-v2": "TaylorAI/bge-micro-v2",
}


class Ternlight:
    def __init__(self, tier):
        self.tier = tier

    def encode(self, sentences, **kwargs):
        with tempfile.TemporaryDirectory() as d:
            src, dst = Path(d, "in.json"), Path(d, "out.npy")
            src.write_text(json.dumps(list(sentences)), encoding="utf-8")
            subprocess.run(["node", str(HERE / "ternlight_embed.js"), self.tier, str(src), str(dst)],
                           check=True)
            return np.load(dst)


def run(names):
    import mteb
    import torch
    from sentence_transformers import SentenceTransformer

    # Leave cores for the user's own work.
    torch.set_num_threads(6)
    tasks = mteb.get_tasks(tasks=list(TASKS), languages=["eng"])
    for name in names:
        ref = MODELS[name]
        if name.startswith("ternlight"):
            model = Ternlight(ref)
        else:
            model = SentenceTransformer(ref, device="cpu")
            # Ternlight truncates at 128 tokens; compare at the same length (static models have no limit).
            try:
                model.max_seq_length = 128
            except AttributeError:
                pass
        mteb.MTEB(tasks=tasks).run(model, output_folder=str(RESULTS / name), eval_splits=["test"],
                                   encode_kwargs={"batch_size": 64}, verbosity=1)


def score(name, task):
    hits = list((RESULTS / name).rglob(f"{task}.json"))
    if not hits:
        return None
    rows = json.loads(hits[0].read_text(encoding="utf-8"))["scores"]["test"]
    return 100 * float(np.mean([r["main_score"] for r in rows]))


def summary():
    names = [n for n in MODELS if (RESULTS / n).exists()]
    print("| task | type | " + " | ".join(names) + " |")
    print("|---|---|" + "---:|" * len(names))
    for task, kind in TASKS.items():
        cells = [score(n, task) for n in names]
        print(f"| {task} | {kind} | " + " | ".join("—" if c is None else f"{c:.1f}" for c in cells) + " |")
    for kind in dict.fromkeys(TASKS.values()):
        cells = []
        for n in names:
            vals = [score(n, t) for t, k in TASKS.items() if k == kind]
            cells.append("—" if None in vals else f"{np.mean(vals):.1f}")
        print(f"| **{kind} mean** | | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2:] or list(MODELS))
    else:
        summary()
