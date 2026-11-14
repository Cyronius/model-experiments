#!/usr/bin/env python3
"""Fetch public text, build train/eval examples, and encode every passage once.

Stages: fetch -> build -> encode.  Questions and answers never reach the
encoder; each passage is encoded alone, so a memory bank is a concatenation of
independently encoded passages.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

from common import REPO, file_sha256, repo_path, rules, sha256, work_dir


def enc_tokenizer():
    from transformers import AutoTokenizer
    r = rules()["encoder"]
    return AutoTokenizer.from_pretrained(repo_path(r["checkpoint"]) / "encoder", local_files_only=True)


def clean(text: str) -> str:
    return text.replace("�", "").strip()


# ---------------------------------------------------------------- fetch

def fetch(args):
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem, hf_hub_download
    r = rules()["data"]
    raw = work_dir() / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation"):
        path = hf_hub_download(r["squad"]["repo"], f"squad_v2/{split}-00000-of-00001.parquet",
                               repo_type="dataset", revision=r["squad"]["revision"], local_dir=raw)
        print("SQUAD", split, file_sha256(Path(path)), flush=True)
    fw = r["fineweb"]
    out = raw / "fineweb.jsonl"
    if out.exists():
        print("FINEWEB exists", out)
        return
    tok = enc_tokenizer()
    fs = HfFileSystem()
    handle = fs.open(f"datasets/{fw['repo']}@{fw['revision']}/{fw['file']}", "rb", block_size=8 << 20)
    pf = pq.ParquetFile(handle)
    kept = []
    for g in range(pf.num_row_groups):
        table = pf.read_row_group(g, columns=["id", "text"])
        for row in table.to_pylist():
            n = len(tok.encode(row["text"], add_special_tokens=False))
            if n >= fw["min_tokens"]:
                kept.append({"id": row["id"], "text": row["text"], "tokens": n})
                if len(kept) == fw["docs"]:
                    break
        print("FINEWEB row group", g, len(kept), flush=True)
        if len(kept) == fw["docs"]:
            break
    if len(kept) < fw["docs"]:
        raise ValueError("Not enough FineWeb-Edu documents")
    with out.open("w") as f:
        for row in kept:
            f.write(json.dumps(row) + "\n")
    print("FINEWEB", len(kept), file_sha256(out), flush=True)


# ---------------------------------------------------------------- build

def squad_rows(split):
    import pyarrow.parquet as pq
    path = work_dir() / "raw" / "squad_v2" / f"{split}-00000-of-00001.parquet"
    return pq.read_table(path).to_pylist()


def paragraphs(rows, split):
    """Unique contexts -> passage ids; returns passages and context->pid map."""
    passages, pid_of = {}, {}
    for r in rows:
        c = r["context"]
        if c not in pid_of:
            pid = f"sq-{split}-{len(pid_of)}"
            pid_of[c] = pid
            passages[pid] = {"text": c, "title": r["title"]}
    return passages, pid_of


def pick_distractors(rng, pids, titles, avoid_title, k):
    out = set()
    while len(out) < k:
        p = rng.choice(pids)
        if titles[p] != avoid_title:
            out.add(p)
    return sorted(out)


def qa_example(rng, kind, q, gold_pid, pids, titles, lo, hi, answers):
    k = rng.randint(lo, hi)
    chosen = pick_distractors(rng, pids, titles, q["title"], k)
    if gold_pid is not None:
        chosen.append(gold_pid)
    rng.shuffle(chosen)
    return {"kind": kind, "id": q["id"], "title": q["title"], "passages": chosen,
            "question": q["question"], "answers": answers}


def unique(seq):
    return list(dict.fromkeys(seq))


def build(args):
    R = rules()
    r, ev = R["data"], R["eval"]
    rng = random.Random(R["seed"])
    tok = enc_tokenizer()
    out = work_dir() / "examples"
    out.mkdir(parents=True, exist_ok=True)
    passages = {"train": {}, "eval": {}}

    # SQuAD v2: train split for training, validation split (disjoint articles) for eval.
    tr_rows, va_rows = squad_rows("train"), squad_rows("validation")
    if {r_["title"] for r_ in tr_rows} & {r_["title"] for r_ in va_rows}:
        raise ValueError("SQuAD train/validation articles overlap")
    tr_pass, tr_pid = paragraphs(tr_rows, "train")
    va_pass, va_pid = paragraphs(va_rows, "validation")
    passages["train"].update(tr_pass)
    passages["eval"].update(va_pass)
    lo, hi = r["distractors"]
    train = []
    tr_pids = sorted(tr_pass, key=lambda p: int(p.split("-")[-1]))
    tr_titles = {p: tr_pass[p]["title"] for p in tr_pids}
    answerable = []
    for q in tr_rows:
        ans = unique(q["answers"]["text"])
        train.append(qa_example(rng, "qa", q, tr_pid[q["context"]], tr_pids, tr_titles, lo, hi, ans))
        if ans:
            answerable.append(q)
    for q in rng.sample(answerable, r["qa_removed"]):
        train.append(qa_example(rng, "qa_removed", q, None, tr_pids, tr_titles, max(lo, 1), hi + 1, []))
    for pid in rng.sample(tr_pids, r["squad_recon_paragraphs"]):
        train.append({"kind": "recon", "id": pid, "passages": [pid], "target": tr_pass[pid]["text"]})

    # FineWeb-Edu: reconstruction and continuation; the last documents are held out.
    docs = [json.loads(l) for l in (work_dir() / "raw" / "fineweb.jsonl").read_text().splitlines()]
    held = r["fineweb"]["heldout_docs"]
    fw_eval = {"recon": [], "continue": []}
    for i, d in enumerate(docs):
        split = "eval" if i >= len(docs) - held else "train"
        ids = tok.encode(d["text"], add_special_tokens=False)
        pid = f"fw-{i}"
        if i % 2 == 0:
            n = rng.randint(*r["recon_tokens"])
            text = clean(tok.decode(ids[:n]))
            ex = {"kind": "recon", "id": pid, "passages": [pid], "target": text}
        else:
            t = rng.randint(*r["continue_target_tokens"])
            m = rng.randint(r["continue_memory_tokens"][0], min(r["continue_memory_tokens"][1], len(ids) - t))
            text = clean(tok.decode(ids[:m]))
            ex = {"kind": "continue", "id": pid, "passages": [pid], "target": clean(tok.decode(ids[m:m + t]))}
        passages[split][pid] = {"text": text, "title": f"fineweb:{d['id']}"}
        (train.append(ex) if split == "train" else fw_eval[ex["kind"]].append(ex))
    rng.shuffle(train)

    # Held-out evaluation sets.
    va_pids = sorted(va_pass, key=lambda p: int(p.split("-")[-1]))
    va_titles = {p: va_pass[p]["title"] for p in va_pids}
    ans_rows = [q for q in va_rows if q["answers"]["text"]]
    imp_rows = [q for q in va_rows if not q["answers"]["text"]]
    S = ev["squad_qa"]
    lo, hi = S["distractors"]
    picked = rng.sample(ans_rows, S["answerable"] + S["removed"])
    qa_eval = [qa_example(rng, "qa", q, va_pid[q["context"]], va_pids, va_titles, lo, hi,
                          unique(q["answers"]["text"])) for q in picked[:S["answerable"]]]
    qa_eval += [qa_example(rng, "qa", q, va_pid[q["context"]], va_pids, va_titles, lo, hi, [])
                for q in rng.sample(imp_rows, S["unanswerable"])]
    for ex in qa_eval[S["answerable"]:]:
        ex["kind"] = "qa_unanswerable"
    qa_eval += [qa_example(rng, "qa_removed", q, None, va_pids, va_titles, max(lo, 1), hi + 1, [])
                for q in picked[S["answerable"]:]]
    # Wrong-memory control: each item gets another item's memory bank, chosen so
    # that no paragraph in it comes from the question's own article.
    n = len(qa_eval)
    for i, ex in enumerate(qa_eval):
        for step in range(n // 2, n // 2 + n):
            other = qa_eval[(i + step) % n]["passages"]
            if all(va_titles[p] != ex["title"] for p in other):
                ex["wrong_passages"] = other
                break
        else:
            raise ValueError("No wrong-memory bank without the question's article")
    recon_eval = rng.sample(fw_eval["recon"], ev["recon"]["fineweb"])
    recon_eval += [{"kind": "recon", "id": p, "passages": [p], "target": va_pass[p]["text"]}
                   for p in rng.sample(va_pids, ev["recon"]["squad"])]
    cont_eval = rng.sample(fw_eval["continue"], ev["continue"]["fineweb"])
    for group in (recon_eval, cont_eval):
        for i, ex in enumerate(group):
            ex["wrong_passages"] = group[(i + len(group) // 2) % len(group)]["passages"]

    # XDNA2 pilot banks, chunked into encoder windows exactly as LCLM windows them.
    W = R["encoder"]["window_tokens"]
    for bank in ("original", "counterfactual"):
        text = (REPO / "context-memory" / "data" / f"{bank}.txt").read_text()
        ids = tok.encode(text, add_special_tokens=False)
        chunks = [ids[o:o + W] for o in range(0, len(ids), W)]
        for j, c in enumerate(chunks):
            passages["eval"][f"xdna-{bank}-{j}"] = {"token_ids": c, "title": f"xdna:{bank}"}

    def dump(name, rows):
        path = out / name
        with path.open("w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        print("WROTE", name, len(rows), file_sha256(path), flush=True)

    dump("train.jsonl", train)
    dump("eval_qa.jsonl", qa_eval)
    dump("eval_recon.jsonl", recon_eval)
    dump("eval_continue.jsonl", cont_eval)
    for split, p in passages.items():
        (out / f"passages_{split}.json").write_text(json.dumps(p))
        print("PASSAGES", split, len(p), flush=True)
    counts = {}
    for ex in train:
        counts[ex["kind"]] = counts.get(ex["kind"], 0) + 1
    print("TRAIN KINDS", counts, flush=True)


# ---------------------------------------------------------------- encode

def encode(args):
    import torch
    from safetensors.torch import save_file
    from transformers import AutoModel
    R = rules()
    E = R["encoder"]
    torch.set_num_threads(args.threads)
    upstream = repo_path(E["upstream"])
    sys.path.insert(0, str(upstream))
    from latent_context import Encoder
    ckpt = repo_path(E["checkpoint"])
    config = json.loads((ckpt / "model_config.json").read_text())
    tok = enc_tokenizer()
    device = torch.device(args.device)
    model = AutoModel.from_pretrained(ckpt / "encoder", dtype=torch.bfloat16, attn_implementation="sdpa",
                                      local_files_only=True).to(device).eval()
    encoder = Encoder(compression_ratio=config["compression_ratio"], max_length=32768,
                      encoder_model=model, encoder_tokenizer=tok, train_encoder=False,
                      max_encode_batch_size=args.batch, pooling=config["pooling"],
                      encoder_mask_type=config["encoder_mask_type"],
                      encoder_window_size=config["encoder_window_size"],
                      boundary_overlap=config["boundary_overlap"]).eval()
    mem_dir = work_dir() / "memory"
    mem_dir.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        passages = json.loads((work_dir() / "examples" / f"passages_{split}.json").read_text())
        ids = {p: v["token_ids"] if "token_ids" in v else tok.encode(v["text"], add_special_tokens=False)
               for p, v in passages.items()}
        order = sorted(ids, key=lambda p: len(ids[p]))
        chunks, index, offset, done = [], {}, 0, 0
        start = time.perf_counter()
        total = sum(len(v) for v in ids.values())
        i = 0
        while i < len(order):
            # Group passages of similar length; keep each call near args.tokens tokens.
            group, budget = [], 0
            while i < len(order) and (not group or budget + len(ids[order[i]]) <= args.tokens):
                group.append(order[i])
                budget += max(1, len(ids[order[i]]))
                i += 1
            with torch.inference_mode():
                outs = encoder([ids[p] for p in group])
            for p, m in zip(group, outs):
                if not torch.isfinite(m).all():
                    raise ValueError(f"Non-finite memory for {p}")
                chunks.append(m.to("cpu", torch.bfloat16))
                index[p] = [offset, m.shape[0], len(ids[p])]
                offset += m.shape[0]
            done += budget
            if len(index) % 2000 < len(group):
                rate = done / (time.perf_counter() - start)
                print(f"ENCODE {split} {len(index)}/{len(order)} passages, {done}/{total} tokens, "
                      f"{rate:.0f} tok/s", flush=True)
        memory = torch.cat(chunks)
        path = mem_dir / f"{split}.safetensors"
        save_file({"memory": memory.contiguous()}, path, metadata={
            "encoder_revision": E["revision"], "split": split})
        manifest = {"split": split, "passages": len(index), "slots": memory.shape[0],
                    "source_tokens": total, "width": memory.shape[1], "seconds": time.perf_counter() - start,
                    "device": args.device, "file_sha256": file_sha256(path),
                    "passages_sha256": file_sha256(work_dir() / "examples" / f"passages_{split}.json")}
        (mem_dir / f"{split}_index.json").write_text(json.dumps({"manifest": manifest, "index": index}))
        print("ENCODED", json.dumps(manifest), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="stage", required=True)
    sub.add_parser("fetch")
    sub.add_parser("build")
    e = sub.add_parser("encode")
    e.add_argument("--splits", nargs="+", default=["eval", "train"])
    e.add_argument("--device", default="cpu")
    e.add_argument("--threads", type=int, default=16)
    e.add_argument("--batch", type=int, default=16)
    e.add_argument("--tokens", type=int, default=8192)
    args = ap.parse_args()
    {"fetch": fetch, "build": build, "encode": encode}[args.stage](args)


if __name__ == "__main__":
    main()
