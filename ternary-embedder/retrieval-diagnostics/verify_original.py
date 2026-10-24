"""Cross-check the custom diagnostic with the unchanged original StudentEncoder."""
import json
from pathlib import Path
import sys

import numpy as np
import torch

from diagnose import ROOT, RUN, dataset, evaluate
sys.path.insert(0, str(ROOT.parent / "tern-embed"))
from evaluate import StudentEncoder


if __name__ == "__main__":
    torch.set_num_threads(6)
    encoder = StudentEncoder(str(RUN / "model.pt"), "cpu")
    results = {}
    for name in ("SciFact", "NFCorpus"):
        data = dataset(name, Path("C:/Users/josha/.cache/huggingface/datasets"))
        q = encoder.encode(data["queries"], batch_size=256)
        d = encoder.encode(data["texts"], batch_size=256)
        summary, rows = evaluate(q @ d.T, data)
        cached = np.load(ROOT / f"cache/embeddings/{name}_qat150_quant.npy")
        ref = np.concatenate([q, d])
        results[name] = {"original_encoder": summary,
                         "max_vector_element_difference_vs_diagnostic": float(np.abs(ref - cached).max()),
                         "mean_vector_cosine_vs_diagnostic": float((ref * cached).sum(-1).mean())}
        np.save(ROOT / f"cache/embeddings/{name}_original_encoder.npy", ref)
        print(json.dumps({"dataset": name, **results[name]}), flush=True)
    (ROOT / "original-encoder-check.json").write_text(json.dumps(results, indent=2) + "\n")
