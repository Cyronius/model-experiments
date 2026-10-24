"""One prespecified, label-free geometry ablation: subtract the corpus centroid.

Use only corpus embeddings to fit one shared transform for queries and docs.
This is a diagnostic on these test collections, not a selected production fix.
"""
import json
from pathlib import Path

import numpy as np

from diagnose import ROOT, dataset, evaluate, paired


if __name__ == "__main__":
    report = {}
    for name in ("SciFact", "NFCorpus"):
        data = dataset(name, Path("C:/Users/josha/.cache/huggingface/datasets"))
        nq = len(data["qids"])
        report[name] = {}
        for model in ("minilm", "float75_float", "qat150_quant", "float150_float"):
            path = ROOT / f"cache/embeddings/{name}_{model}.npy"
            if not path.is_file():
                continue
            v = np.load(path)
            centroid = v[nq:].mean(0)
            centered = v - centroid
            centered /= np.maximum(np.linalg.norm(centered, axis=1, keepdims=True), 1e-12)
            original, original_rows = evaluate(v[:nq] @ v[nq:].T, data)
            transformed, transformed_rows = evaluate(centered[:nq] @ centered[nq:].T, data)
            report[name][model] = {"original_ndcg10": original["ndcg10"],
                                   "centered_ndcg10": transformed["ndcg10"],
                                   "paired": paired(original_rows, transformed_rows)}
    (ROOT / "centering-probe.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
