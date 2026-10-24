"""Inspect embedding geometry and preserve concrete paired ranking failures."""
import json
from pathlib import Path

import numpy as np

from diagnose import ROOT, dataset


def geometry(vectors):
    v = vectors.astype(np.float64)
    n = len(v)
    total = v.sum(0)
    mean_pair = (total @ total - np.square(v).sum()) / (n * (n - 1))
    centered = v - v.mean(0)
    eigenvalues = np.maximum(np.linalg.eigvalsh(centered.T @ centered / n), 0)
    total_variance = eigenvalues.sum()
    if total_variance > 1e-20:
        probabilities = eigenvalues / total_variance
        positive = probabilities[probabilities > 0]
        effective_rank = float(np.exp(-(positive * np.log(positive)).sum()))
    else:
        effective_rank = 0.0
    return {"mean_pairwise_cosine": float(mean_pair), "centered_effective_rank": effective_rank,
            "centered_variance": float(total_variance), "norm_min": float(np.linalg.norm(v, axis=1).min()),
            "norm_max": float(np.linalg.norm(v, axis=1).max())}


def main():
    report = json.loads((ROOT / "diagnosis.json").read_text())
    result = {"geometry": {}, "examples": {}, "embedding_agreement": {}, "mean_comparisons": {}}
    task_rows = {}
    for name in ("SciFact", "NFCorpus"):
        data = dataset(name, Path("C:/Users/josha/.cache/huggingface/datasets"))
        rows = json.loads((ROOT / f"outputs/{name}_per_query.json").read_text())
        task_rows[name] = rows
        result["geometry"][name] = {}
        vectors = {}
        for model in report["models"]:
            v = np.load(ROOT / f"cache/embeddings/{name}_{model}.npy")
            vectors[model] = v
            result["geometry"][name][model] = geometry(v[len(data["qids"]):])
        result["embedding_agreement"][name] = {}
        for left, right in (("float75_float", "float75_quant"), ("qat150_float", "qat150_quant")):
            if left not in vectors or right not in vectors:
                continue
            agreement = (vectors[left] * vectors[right]).sum(-1)
            nq = len(data["qids"])
            result["embedding_agreement"][name][f"{left} -> {right}"] = {
                "mean_query_cosine": float(agreement[:nq].mean()),
                "mean_document_cosine": float(agreement[nq:].mean()),
                "p10_document_cosine": float(np.percentile(agreement[nq:], 10))}
        indexes = {cid: i for i, cid in enumerate(data["cids"])}
        examples = []
        for qi, (baseline, student) in enumerate(zip(rows["minilm"], rows["qat150_quant"])):
            if baseline["first_relevant_rank"] != 1 or student["first_relevant_rank"] == 1:
                continue
            qid = baseline["query_id"]
            relevant = baseline["best_relevant_id"]
            competitor = student["best_competitor_id"]
            cases = {model: {key: row[qi][key] for key in ("ndcg10", "first_relevant_rank", "best_relevant_margin")}
                     for model, row in rows.items()}
            examples.append({"query_id": qid, "query": data["queries"][qi],
                             "baseline_relevant_id": relevant,
                             "baseline_relevant_preview": " ".join(data["texts"][indexes[relevant]].split()[:20]),
                             "student_competitor_id": competitor,
                             "student_competitor_preview": " ".join(data["texts"][indexes[competitor]].split()[:20]),
                             "models": cases})
        examples.sort(key=lambda ex: ex["models"]["qat150_quant"]["first_relevant_rank"], reverse=True)
        result["examples"][name] = examples[:8]
    for left, right in (("minilm", "qat150_quant"), ("float75_float", "float75_quant"),
                        ("qat150_float", "qat150_quant"), ("float75_float", "qat150_quant"),
                        ("qat150_float", "qat150_weights"), ("qat150_float", "qat150_activations"),
                        ("qat150_weights", "qat150_quant"), ("float150_float", "qat150_quant"),
                        ("minilm", "float150_float"), ("float75_float", "float150_float")):
        if left not in report["models"] or right not in report["models"]:
            continue
        rng = np.random.default_rng(0)
        samples, means = [], []
        for rows in task_rows.values():
            delta = np.array([b["ndcg10"] - a["ndcg10"] for a, b in zip(rows[left], rows[right])])
            samples.append(delta[rng.integers(0, len(delta), size=(4000, len(delta)))].mean(1) * 100)
            means.append(float(delta.mean() * 100))
        result["mean_comparisons"][f"{left} -> {right}"] = {
            "mean_ndcg10_delta_right_minus_left": float(np.mean(means)),
            "paired_bootstrap_95_ci": np.percentile(np.mean(samples, axis=0), [2.5, 97.5]).tolist()}
    (ROOT / "geometry-and-examples.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["geometry"], indent=2))


if __name__ == "__main__":
    main()
