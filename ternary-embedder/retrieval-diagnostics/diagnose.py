"""Paired retrieval/quantization diagnosis using cached MTEB test sets.

No training: compare identical saved master weights under four arithmetic modes.
The 75M checkpoint is float-trained; the 150M checkpoint is QAT-trained. Turning
QAT master weights back to float is an inference ablation, NOT a float-only run.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(ROOT / "cache/huggingface"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np
import pyarrow.ipc as ipc
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(ROOT.parent / "tern-embed"))
from model import Student, TernLinear, quant_act, tern_codes  # noqa: E402

RUN = ROOT.parent / "tern-embed/pulls/grid1/runs/2a_384"
HUB = ROOT / "cache/huggingface/hub"
MINILM = HUB / "models--sentence-transformers--all-MiniLM-L6-v2/snapshots/1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
TOKENIZER = HUB / "models--BAAI--bge-small-en-v1.5/snapshots/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
MODES = {"float": (False, False), "weights": (True, False),
         "activations": (False, True), "quant": (True, True)}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def emit(value):
    print(json.dumps(value), flush=True)


def dataset(name, cache):
    low = name.lower()
    root = cache / f"mteb___{low}"
    files = {}
    for kind, filename in (("corpus", f"{low}-corpus.arrow"),
                           ("queries", f"{low}-queries.arrow"), ("qrels", f"{low}-test.arrow")):
        hits = sorted(root.rglob(filename))
        if len(hits) != 1:
            raise ValueError(f"Expected exactly one {filename} in {root}; got {hits}")
        files[kind] = hits[0]
    data = {key: ipc.open_stream(str(path)).read_all().to_pylist() for key, path in files.items()}
    corpus = sorted(data["corpus"], key=lambda row: row["_id"])
    cids = [str(row["_id"]) for row in corpus]
    texts = [(row.get("title", "") + " " + row["text"]).strip() for row in corpus]
    qrels = {}
    for row in data["qrels"]:
        qrels.setdefault(str(row["query-id"]), {})[str(row["corpus-id"])] = int(row["score"])
    queries = sorted((r for r in data["queries"] if str(r["_id"]) in qrels), key=lambda r: r["_id"])
    return {"name": name, "cids": cids, "texts": texts,
            "qids": [str(r["_id"]) for r in queries], "queries": [r["text"] for r in queries],
            "qrels": qrels, "source_sha256": {key: digest(path) for key, path in files.items()}}


class FrozenLinear(nn.Module):
    def __init__(self, module, weights, activations):
        super().__init__()
        effective = module.weight.detach()
        if weights:
            codes, scale = tern_codes(effective)
            effective = codes * scale
        self.register_buffer("weight", effective.clone())
        self.register_buffer("bias", module.bias.detach().clone())
        self.activations = activations

    def forward(self, x):
        return F.linear(quant_act(x) if self.activations else x, self.weight, self.bias)


def student(path, mode):
    model = Student.load(path).eval()
    for layer in model.layers:
        for name in ("q", "k", "v", "o", "fc1", "fc2"):
            original = getattr(layer, name)
            assert isinstance(original, TernLinear)
            setattr(layer, name, FrozenLinear(original, *MODES[mode]))
    return model


@torch.inference_mode()
def encode(model, tokenizer, texts, baseline, batch):
    tokens = tokenizer(texts, truncation=True, max_length=128)["input_ids"]
    order = sorted(range(len(tokens)), key=lambda i: len(tokens[i]))
    result = np.empty((len(tokens), 384), dtype=np.float32)
    start = time.perf_counter()
    for offset in range(0, len(order), batch):
        indexes = order[offset:offset + batch]
        length = max(len(tokens[i]) for i in indexes)
        ids = torch.zeros((len(indexes), length), dtype=torch.long)
        mask = torch.zeros_like(ids, dtype=torch.bool)
        for row, index in enumerate(indexes):
            ids[row, :len(tokens[index])] = torch.tensor(tokens[index])
            mask[row, :len(tokens[index])] = True
        if baseline:
            hidden = model(input_ids=ids, attention_mask=mask.long()).last_hidden_state
            vector = F.normalize((hidden * mask[..., None]).sum(1) / mask.sum(1, keepdim=True), dim=-1)
        else:
            vector = model(ids, mask)[2]
        result[indexes] = vector.numpy()
        if offset % (batch * 20) == 0:
            emit({"encoded": min(offset + batch, len(order)), "total": len(order),
                  "seconds": round(time.perf_counter() - start, 1)})
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite embeddings")
    return result


def cached_encode(model, tokenizer, texts, baseline, batch, identity, path):
    key = hashlib.sha256(json.dumps({"identity": identity, "texts": texts, "max_length": 128,
                                     "encoder_version": 1}, sort_keys=True).encode()).hexdigest()
    meta = path.with_suffix(".json")
    if path.is_file() and meta.is_file() and json.loads(meta.read_text())["key"] == key:
        return np.load(path)
    vectors = encode(model, tokenizer, texts, baseline, batch)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, vectors)
    meta.write_text(json.dumps({"key": key, "identity": identity}, indent=2))
    return vectors


def evaluate(scores, data):
    """TREC nDCG uses raw relevance grades, not 2**grade-1."""
    cids = data["cids"]
    lookup = {cid: i for i, cid in enumerate(cids)}
    # TREC resolves equal scores by descending document ID. Exact ties are rare.
    tie_order = np.array(sorted(range(len(cids)), key=lambda i: cids[i], reverse=True))
    rows = []
    for qi, qid in enumerate(data["qids"]):
        score = scores[qi].copy()
        if qid in lookup:
            score[lookup[qid]] = -np.inf
        grades = data["qrels"][qid]
        positive = [lookup[cid] for cid, grade in grades.items() if grade > 0 and cid != qid]
        if not positive:
            raise ValueError(f"No positive judgments for {qid}")
        ranked = tie_order[np.argsort(-score[tie_order], kind="stable")]
        positive_set = set(positive)
        ranks = [rank + 1 for rank, index in enumerate(ranked) if index in positive_set]
        top10 = ranked[:10]
        dcg = sum(max(grades.get(cids[index], 0), 0) / np.log2(rank + 2) for rank, index in enumerate(top10))
        ideal = sorted([grade for cid, grade in grades.items() if grade > 0 and cid != qid], reverse=True)[:10]
        idcg = sum(grade / np.log2(rank + 2) for rank, grade in enumerate(ideal))
        competitors = [index for index in ranked if index not in positive_set and cids[index] != qid]
        best = max(positive, key=lambda i: score[i])
        competitor = competitors[0]
        rows.append({"query_id": qid, "ndcg10": float(dcg / idcg), "first_relevant_rank": ranks[0],
                     "recall10": sum(rank <= 10 for rank in ranks) / len(positive),
                     "relevant_count": len(positive), "best_relevant_id": cids[best],
                     "best_competitor_id": cids[competitor], "best_relevant_score": float(score[best]),
                     "best_competitor_score": float(score[competitor]),
                     "best_relevant_margin": float(score[best] - score[competitor]),
                     "top10": [cids[i] for i in top10]})
    summary = {"ndcg10": 100 * float(np.mean([r["ndcg10"] for r in rows])),
               "recall10": 100 * float(np.mean([r["recall10"] for r in rows])),
               "top1_relevant_queries": sum(r["first_relevant_rank"] == 1 for r in rows),
               "median_first_relevant_rank": float(np.median([r["first_relevant_rank"] for r in rows])),
               "median_best_relevant_margin": float(np.median([r["best_relevant_margin"] for r in rows])),
               "queries": len(rows)}
    return summary, rows


def paired(left, right):
    assert [r["query_id"] for r in left] == [r["query_id"] for r in right]
    delta = np.array([b["ndcg10"] - a["ndcg10"] for a, b in zip(left, right)])
    rng = np.random.default_rng(0)
    samples = delta[rng.integers(0, len(delta), size=(4000, len(delta)))].mean(1) * 100
    return {"ndcg10_delta_right_minus_left": float(delta.mean() * 100),
            "paired_bootstrap_95_ci": np.percentile(samples, [2.5, 97.5]).tolist(),
            "queries_improved": int((delta > 1e-12).sum()), "queries_worsened": int((delta < -1e-12).sum()),
            "top1_gained": sum(a["first_relevant_rank"] > 1 and b["first_relevant_rank"] == 1 for a, b in zip(left, right)),
            "top1_lost": sum(a["first_relevant_rank"] == 1 and b["first_relevant_rank"] > 1 for a, b in zip(left, right))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-cache", type=Path, default=Path("C:/Users/josha/.cache/huggingface/datasets"))
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--models", nargs="*", default=["minilm", "float75_float", "float75_quant", "qat150_quant",
                                                       "qat150_float", "qat150_weights", "qat150_activations"]
                        + (["float150_float"] if (ROOT / "float-control/runs/float150/model.pt").is_file() else []))
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    data = [dataset(name, args.dataset_cache) for name in ("SciFact", "NFCorpus")]
    for d in data:
        emit({"dataset": d["name"], "corpus": len(d["cids"]), "queries": len(d["qids"])})
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    baseline_tokenizer = AutoTokenizer.from_pretrained(str(MINILM), local_files_only=True)
    report = {"utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "scope": "Paired inference quantization ablations at fixed checkpoint weights; no new training",
              "max_length": 128, "query_prompts": False, "models": {}, "datasets": {},
              "limitations": ["75M float checkpoint vs 150M QAT checkpoint are NOT equal-budget training controls",
                              "150M float mode uses QAT-trained master weights, not a float-only-trained model",
                              "Competitors have no positive qrel; unjudged documents may still be relevant"]}
    all_rows, all_scores, vectors = {}, {}, {}
    for name in args.models:
        if name == "minilm":
            model = AutoModel.from_pretrained(str(MINILM), local_files_only=True, attn_implementation="eager").eval()
            tk = baseline_tokenizer
            identity = {"model": str(MINILM), "weights_sha256": digest(MINILM / "model.safetensors"),
                        "pooling": "mean", "tokenizer_sha256": digest(MINILM / "tokenizer.json")}
        else:
            checkpoint_name, mode = name.split("_", 1)
            checkpoint = (ROOT / "float-control/runs/float150/model.pt" if checkpoint_name == "float150"
                          else RUN / ("float.pt" if checkpoint_name == "float75" else "model.pt"))
            if checkpoint_name not in ("float75", "float150", "qat150") or mode not in MODES:
                raise ValueError(name)
            model, tk = student(checkpoint, mode), tokenizer
            identity = {"checkpoint": str(checkpoint), "sha256": digest(checkpoint), "mode": mode,
                        "model_source_sha256": digest(ROOT.parent / "tern-embed/model.py"),
                        "tokenizer_sha256": digest(TOKENIZER / "tokenizer.json")}
        report["models"][name] = identity
        for d in data:
            emit({"model": name, "dataset": d["name"]})
            combined = d["queries"] + d["texts"]
            v = cached_encode(model, tk, combined, name == "minilm", args.batch, identity,
                              ROOT / f"cache/embeddings/{d['name']}_{name}.npy")
            nq = len(d["queries"])
            scores = v[:nq] @ v[nq:].T
            summary, rows = evaluate(scores, d)
            report["datasets"].setdefault(d["name"], {"source_sha256": d["source_sha256"], "models": {}})["models"][name] = summary
            all_rows.setdefault(d["name"], {})[name] = rows
            all_scores.setdefault(d["name"], {})[name] = scores
            vectors.setdefault(d["name"], {})[name] = v
            emit({"model": name, "dataset": d["name"], **summary})
        del model
    for d in data:
        name = d["name"]
        rows = all_rows[name]
        pairs = [("float75_float", "float75_quant"), ("qat150_float", "qat150_quant"),
                 ("float75_float", "qat150_quant"), ("minilm", "qat150_quant"),
                 ("qat150_float", "qat150_weights"), ("qat150_float", "qat150_activations"),
                 ("qat150_weights", "qat150_quant"), ("float150_float", "qat150_quant"),
                 ("minilm", "float150_float"), ("float75_float", "float150_float")]
        report["datasets"][name]["comparisons"] = {f"{a} -> {b}": paired(rows[a], rows[b])
                                                   for a, b in pairs if a in rows and b in rows}
        # Same concrete judged-relevant / competitor pairs under every model.
        if "minilm" in rows and "qat150_quant" in rows:
            lookup = {cid: i for i, cid in enumerate(d["cids"])}
            fixed = []
            for qi, qid in enumerate(d["qids"]):
                positive = rows["minilm"][qi]["best_relevant_id"]
                competitor = rows["qat150_quant"][qi]["best_competitor_id"]
                margins = {m: float(s[qi, lookup[positive]] - s[qi, lookup[competitor]])
                           for m, s in all_scores[name].items()}
                fixed.append({"query_id": qid, "relevant_id": positive, "competitor_id": competitor,
                              "margins": margins})
            report["datasets"][name]["fixed_pair_positive_margins"] = {
                m: {"positive_pairs": sum(r["margins"][m] > 0 for r in fixed), "pairs": len(fixed),
                    "median_margin": float(np.median([r["margins"][m] for r in fixed]))} for m in rows}
            (ROOT / "outputs").mkdir(exist_ok=True)
            (ROOT / f"outputs/{name}_fixed_pairs.json").write_text(json.dumps(fixed, indent=2))
        (ROOT / "outputs").mkdir(exist_ok=True)
        (ROOT / f"outputs/{name}_per_query.json").write_text(json.dumps(rows, indent=2))
    report["mean_ndcg10"] = {m: float(np.mean([report["datasets"][d["name"]]["models"][m]["ndcg10"] for d in data]))
                             for m in args.models}
    if "float150_float" in args.models:
        old = json.loads((RUN / "config.json").read_text())
        control = json.loads((ROOT / "float-control/runs/float150/config.json").read_text())
        matched = ("init", "layer_ids", "width", "loops", "pool", "layer_teacher", "pairs", "tokens",
                   "batch", "lr", "seed", "steps", "mean_len", "dims")
        differences = {key: {"original": old[key], "control": control[key]}
                       for key in matched if old[key] != control[key]}
        if differences or control["float_frac"] != 1 or old["float_frac"] != 0.5:
            raise ValueError(f"Training control mismatch: {differences}")
        provenance = json.loads((ROOT / "control-run.json").read_text())
        if report["models"]["float150_float"]["sha256"] != provenance["checkpoint_sha256"]:
            raise ValueError("Float control checkpoint differs from verified download")
        report["scope"] = "Paired inference ablations plus independent equal-budget float-only training control"
        report["training_control"] = {"matched_settings": {key: control[key] for key in matched},
                                      "float_fraction": {"original": 0.5, "control": 1.0},
                                      "run": provenance}
        def logged_metrics(path):
            rows = {}
            for line in path.read_text(errors="replace").splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict) and "stsb" in row and "held_cos" in row:
                    rows[row["step"]] = row
            return rows
        old_metrics = logged_metrics(RUN / "log.jsonl")
        new_metrics = logged_metrics(ROOT / "float-control/runs/float150/log.jsonl")
        report["training_control"]["float_prefix_agreement"] = {
            str(step): {"stsb_delta": new_metrics[step]["stsb"] - old_metrics[step]["stsb"],
                        "held_cos_delta": new_metrics[step]["held_cos"] - old_metrics[step]["held_cos"]}
            for step in sorted(old_metrics.keys() & new_metrics.keys()) if step <= 8000}
        report["limitations"].append("Training control uses the same recipe and seed on a different GPU/runtime; single seed")
    (ROOT / "diagnosis.json").write_text(json.dumps(report, indent=2) + "\n")
    emit({"mean_ndcg10": report["mean_ndcg10"]})


if __name__ == "__main__":
    main()
