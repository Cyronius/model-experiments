#!/usr/bin/env python3
"""Resumable base NuExtract3 corpus extraction. Run inside dibs run gpu."""
from __future__ import annotations
import argparse
import collections
import datetime as dt
import fcntl
import json
import os
import signal
import sqlite3
import sys
from pathlib import Path

from corpus import digest, write_json, passage, divide, cite

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
sys.path[:0] = [str(PROJECT), str(PROJECT / "training")]
from curator import parse_output, schema_errors
from run_models import render, MODELS


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def exported_db(path, manifest, records):
    temporary = path.with_suffix(".tmp.sqlite")
    if temporary.exists():
        temporary.unlink()
    with sqlite3.connect(temporary) as db:
        db.executescript("""
        CREATE TABLE sources (id TEXT PRIMARY KEY, path TEXT, data_json TEXT);
        CREATE TABLE units (id TEXT PRIMARY KEY, source_id TEXT, json_pointer TEXT, data_json TEXT);
        CREATE TABLE claims (id TEXT PRIMARY KEY, source_id TEXT, source_path TEXT,
          unit_id TEXT, statement TEXT, quote TEXT, review_status TEXT, data_json TEXT);
        CREATE INDEX claims_source ON claims(source_id);
        CREATE INDEX claims_unit ON claims(unit_id);
        """)
        db.executemany("INSERT INTO sources VALUES (?,?,?)", [(k, v["path"], json.dumps(v)) for k,v in manifest["sources"].items()])
        db.executemany("INSERT INTO units VALUES (?,?,?,?)", [(k,v["source_key"],v["json_pointer"],json.dumps(v)) for k,v in manifest["units"].items()])
        db.executemany("INSERT INTO claims VALUES (?,?,?,?,?,?,?,?)", [
            (r["id"],r["source_key"],r["source_path"],r["unit_key"],r["statement"],r["quote"],"proposed",json.dumps(r)) for r in records])
    temporary.replace(path)


def run(args):
    import torch
    import transformers
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration, StoppingCriteria, StoppingCriteriaList
    from runtime import configure
    runtime = configure(torch)
    torch.set_num_threads(4)
    torch.manual_seed(2718)
    args.out.mkdir(parents=True, exist_ok=True)
    lock = (args.out / "runner.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest_path = args.out / "data/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    metadata = args.checkpoint / ".cache/huggingface/download/config.json.metadata"
    if metadata.read_text().splitlines()[0] != MODELS["nuextract3"]["revision"]:
        raise ValueError("Student checkpoint revision differs from pinned base NuExtract3")
    # Snapshots and decoded field views must still match the frozen corpus.
    for source in manifest["sources"].values():
        if digest((args.out / "data" / source["snapshot"]).read_bytes()) != source["sha256"]:
            raise ValueError("Original source snapshot changed")
    texts = {k: (args.out / "data" / u["text_path"]).read_bytes().decode("utf-8") for k,u in manifest["units"].items()}
    if any(digest(texts[k]) != u["sha256"] for k,u in manifest["units"].items()):
        raise ValueError("Decoded document view changed")
    prompt = json.loads(args.prompt.read_text())
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    config = {"model": MODELS["nuextract3"], "adapter": None, "dtype": "bfloat16", "device": "cuda",
              "torch": torch.__version__, "transformers": transformers.__version__, "runtime": runtime,
              "prompt_sha256": digest(prompt), "max_new_tokens": args.max_tokens,
              "max_batch_size": args.batch_size, "eos_token_ids": [248044,248046], "greedy": True,
              "policy": "No JSON or evidence repair; malformed/capped output subdivides at most three times; uncited claims quarantined"}
    cache = args.out / "cache"
    cache.mkdir(exist_ok=True)
    write_json(args.out / "execution.json", {"config": config, "manifest_sha256": digest(manifest_path.read_bytes()),
               "started_utc": now(), "pid": os.getpid(), "paid_compute_cost_usd": 0,
               "source_code_sha256": {p.name:digest(p.read_bytes()) for p in ROOT.glob("*.py")}})
    queue = collections.deque(manifest["passages"])
    remaining = collections.Counter(c["unit_key"] for c in queue)
    initial_by_unit = remaining.copy()
    coverage = {k: {"source_path":s["path"],"units":0,"completed_units":0,"initial_passages":0,
                        "leaf_passages":0,"cited_claims":0,"quarantined_claims":0,"failed_passages":0,"empty_passages":0}
                for k,s in manifest["sources"].items()}
    for key,u in manifest["units"].items():
        coverage[u["source_key"]]["units"] += 1
        coverage[u["source_key"]]["initial_passages"] += initial_by_unit[key]
        if not remaining[key]:
            coverage[u["source_key"]]["completed_units"] += 1
    stopped = False
    yield_path = os.environ.get("DIBS_YIELD_FILE")
    def yielding():
        return bool(yield_path and Path(yield_path).exists())
    def interrupt(signum, frame):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    class YieldAtBoundary(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            requested = (stopped or yielding()) if input_ids.shape[1] % 16 == 0 else False
            return torch.full((input_ids.shape[0],), requested, device=input_ids.device, dtype=torch.bool)
    counts = collections.Counter()
    records, record_ids = [], set()
    model = None
    batch_limit = args.batch_size
    state = {"state":"running", "started_utc":now(), "source_files":len(coverage),
             "document_units":len(manifest["units"]), "initial_passages":len(manifest["passages"]),
             "model":config["model"], "adapter":None, "paid_compute_cost_usd":0,
             "review_status":"All claims are proposed; citation validity does not establish semantic support."}
    def checkpoint(status="running"):
        state.update(state=status, updated_utc=now(), pending_passages=len(queue),
                     completed_source_files=sum(v["completed_units"] == v["units"] for v in coverage.values()),
                     valid_proposed_claims=len(records), counts=dict(counts))
        write_json(args.out / "status.json", state)
        write_json(args.out / "source-coverage.json", coverage)
    def finish_case(case, raw, key, handles):
        counts["attempted_passages"] += 1
        unit = manifest["units"][case["unit_key"]]
        source = manifest["sources"][unit["source_key"]]
        text = texts[case["unit_key"]]
        try:
            payload = parse_output(raw["output"])
            errors = schema_errors(payload)
        except (TypeError, ValueError) as exc:
            payload, errors = None, [str(exc)]
        if raw["truncated"]:
            errors.append("Completion did not reach a native end token")
        children = divide(case, unit, text) if errors else []
        event = {"id":case["id"],"unit_key":case["unit_key"],"source_path":source["path"],
                 "source_start":case["source_start"],"source_end":case["source_end"],
                 "depth":case["depth"],"parent":case["parent"],"cache_key":key,
                 "errors":errors,"status":"subdivided" if children else "needs_review" if errors else "processed"}
        handles["passages"].write(json.dumps(event)+"\n")
        handles["raw"].write(json.dumps({**event,**raw})+"\n")
        if children:
            queue.extend(children)
            remaining[case["unit_key"]] += len(children)-1
            counts["subdivided_passages"] += 1
            return
        c = coverage[unit["source_key"]]
        c["leaf_passages"] += 1
        if errors:
            c["failed_passages"] += 1
            handles["quarantine"].write(json.dumps({"kind":"passage_output_failure",**event})+"\n")
        else:
            if not payload["claims"]:
                c["empty_passages"] += 1
            for claim in payload["claims"]:
                record = cite(claim, case, unit, source, text, key)
                if record["citation_valid"] and record["id"] in record_ids:
                    counts["duplicate_records"] += 1
                    continue
                if record["citation_valid"]:
                    record_ids.add(record["id"])
                    records.append(record)
                    c["cited_claims"] += 1
                    handles["claims"].write(json.dumps(record,ensure_ascii=False)+"\n")
                else:
                    c["quarantined_claims"] += 1
                    handles["quarantine"].write(json.dumps({"kind":"claim_citation_failure",**record},ensure_ascii=False)+"\n")
        remaining[case["unit_key"]] -= 1
        if not remaining[case["unit_key"]]:
            c["completed_units"] += 1

    names = {"claims":"claims.proposed.jsonl","quarantine":"quarantine.jsonl","passages":"passages.jsonl","raw":"raw.jsonl"}
    handles = {k:(args.out/v).open("w",buffering=1) for k,v in names.items()}
    checkpoint()
    try:
        while queue:
            if stopped or yielding():
                checkpoint("yielding" if yielding() else "interrupted")
                return 75 if yielding() else 130
            batch = []
            while queue and len(batch) < batch_limit:
                case = queue.popleft()
                rendered = render(tokenizer,{"task":"extraction","text":case["text"]},prompt,"nuextract3")
                key = digest({"config":config,"case":case,"rendered_prompt":rendered})
                saved = cache/(key+".json")
                if saved.exists():
                    finish_case(case,json.loads(saved.read_text()),key,handles)
                    counts["cache_hits"] += 1
                else:
                    batch.append((case,rendered,key))
            if batch:
                if model is None:
                    if not torch.cuda.is_available():
                        raise RuntimeError("Local AMD GPU unavailable; no paid/remote fallback is configured")
                    model = Qwen3_5ForConditionalGeneration.from_pretrained(args.checkpoint,dtype=torch.bfloat16,
                        attn_implementation="sdpa",local_files_only=True).to("cuda").eval()
                    model.requires_grad_(False)
                inputs = tokenizer([x[1] for x in batch],padding=True,return_tensors="pt").to("cuda")
                try:
                    with torch.inference_mode():
                        output = model.generate(**inputs,do_sample=False,max_new_tokens=args.max_tokens,
                            eos_token_id=[248044,248046],pad_token_id=tokenizer.pad_token_id,
                            stopping_criteria=StoppingCriteriaList([YieldAtBoundary()]))
                except torch.OutOfMemoryError:
                    if len(batch) == 1:
                        raise
                    del inputs
                    torch.cuda.empty_cache()
                    batch_limit = max(1,len(batch)//2)
                    queue.extendleft(x[0] for x in reversed(batch))
                    counts["oom_batch_reductions"] += 1
                    checkpoint()
                    continue
                if stopped or yielding():
                    queue.extendleft(x[0] for x in reversed(batch))
                    checkpoint("yielding" if yielding() else "interrupted")
                    return 75 if yielding() else 130
                for (case,rendered,key),ids in zip(batch,output[:,inputs.input_ids.shape[1]:]):
                    tokens = ids.tolist()
                    stop = next((i for i,t in enumerate(tokens) if t in (248044,248046)),len(tokens))
                    raw = {"output":tokenizer.decode(tokens[:stop],skip_special_tokens=True),"output_tokens":stop,
                           "truncated":stop == len(tokens),"rendered_prompt":rendered,
                           "input_tokens":len(tokenizer.encode(rendered,add_special_tokens=False)),
                           "batch_ids":[c[0]["id"] for c in batch]}
                    write_json(cache/(key+".json"),raw)
                    finish_case(case,raw,key,handles)
                    counts["new_generations"] += 1
                del inputs,output
            checkpoint()
            print(json.dumps({"completed_files":state["completed_source_files"],"source_files":len(coverage),
                   "pending":len(queue),"cited_claims":len(records),"attempted_passages":counts["attempted_passages"]}),flush=True)
        exported_db(args.out/"claims.sqlite",manifest,records)
        checkpoint("complete")
        write_json(args.out/"summary.json",{**state,"source_files_without_cited_claims":sum(not c["cited_claims"] for c in coverage.values()),
                   "failed_leaf_passages":sum(c["failed_passages"] for c in coverage.values()),
                   "quarantined_claims":sum(c["quarantined_claims"] for c in coverage.values()),
                   "empty_leaf_passages":sum(c["empty_passages"] for c in coverage.values()),
                   "coverage_note":"Completed means all frozen source text was attempted. Claim recall and semantic truth are not certified. Empty and failed passages require review."})
        return 0
    except BaseException as exc:
        state["error"] = repr(exc)
        checkpoint("failed")
        raise
    finally:
        for handle in handles.values():
            handle.close()


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",required=True,type=Path)
    p.add_argument("--checkpoint",type=Path,default=PROJECT.parent/"models/claim-curation/nuextract3")
    p.add_argument("--prompt",type=Path,default=ROOT/"prompt.json")
    p.add_argument("--batch-size",type=int,default=8)
    p.add_argument("--max-tokens",type=int,default=1536)
    args=p.parse_args()
    sys.exit(run(args))
