#!/usr/bin/env python3
"""Cached source-grounded distillation through an explicitly local teacher.

This client deliberately cannot call a remote paid model endpoint. Teacher
labels passing the two checks are eligible supervision, not certified facts.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import copy
import re
import urllib.parse
import urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

INSTRUCTION = """Extract atomic, self-contained assertions from the supplied body.
Preserve all workload, device, version, configuration, numeric-unit and uncertainty
qualifiers needed to interpret each assertion. Resolve 'it', 'both' and 'the array'
using the supplied heading/context only when the reference is explicit. Otherwise
retain the uncertainty rather than inventing a scope. Do not extract metadata as
standalone observations: metadata belongs in conditions. Keep observations,
recommendations, hypotheses and limitations distinct in the statement wording.
A recommendation to do X is not evidence that X guarantees success. Not tested
does not mean failed. Return JSON with exactly one claims array; each claim has
statement (string), quote (an exact contiguous supplied passage), and conditions
(an array of exact supplied fragments). Do not invent missing facts. Questions
and future plans are not assertions. An empty claims array is allowed.
Copy quotation and condition characters literally, including Markdown emphasis,
backticks, punctuation and capitalization. Encode line breaks as JSON \\n escapes;
never rewrite a quote into a cleaned-up paraphrase."""

CRITIC = """Audit the proposed extraction against the supplied source, not against
your background knowledge. Check missing/changed qualifiers, numbers, negation,
unsupported causal guarantees, recommendation vs observation, unsupported scope,
missing important assertions, compound claims and quotation support. Return JSON:
{"verdict":"accept"|"reject"|"uncertain","issues":["short specific issue"]}.
Accept only if every statement is faithful and self-contained and the extraction
does not omit a substantive assertion. Exact quotations alone are insufficient.
Do not silently repair the candidate. Uncertain cases must not enter training."""


def fp(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def request(endpoint, teacher_id, revision, instruction, text, cache, max_tokens):
    parsed = urllib.parse.urlparse(endpoint)
    if parsed.hostname not in ("127.0.0.1", "localhost", "::1") or parsed.scheme != "http":
        raise ValueError("Only an explicit local HTTP teacher endpoint is supported")
    payload = {"model": teacher_id, "messages": [{"role": "system", "content": instruction},
               {"role": "user", "content": text}], "temperature": 0, "max_tokens": max_tokens,
               "chat_template_kwargs": {"enable_thinking": False}, "response_format": {"type": "json_object"}}
    key = fp({"endpoint": endpoint, "teacher_revision": revision, "request": payload})
    file = cache / f"{key}.json"
    if file.exists():
        return json.loads(file.read_text()), key
    req = urllib.request.Request(endpoint.rstrip("/") + "/chat/completions",
                                 data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as response:
        raw = json.load(response)
    result = {"teacher_revision": revision, "request": payload, "response": raw}
    file.write_text(json.dumps(result, indent=2) + "\n")
    return result, key


def mechanically_validate(payload, document):
    if not isinstance(payload, dict) or set(payload) != {"claims"} or not isinstance(payload["claims"], list):
        return ["Schema: expected exactly one claims array"]
    errors = []
    for i, c in enumerate(payload["claims"]):
        if not isinstance(c, dict) or set(c) != {"statement", "quote", "conditions"}:
            errors.append(f"claim {i}: wrong fields")
            continue
        if not isinstance(c["statement"], str) or not c["statement"].strip():
            errors.append(f"claim {i}: missing statement")
        if not isinstance(c["quote"], str) or not c["quote"] or c["quote"] not in document:
            errors.append(f"claim {i}: quotation not verbatim")
        conditions = c["conditions"]
        if not isinstance(conditions, list) or any(not isinstance(x, str) or not x or x not in document for x in conditions):
            errors.append(f"claim {i}: condition not verbatim")
    return errors


def align_evidence(payload, document):
    """Restore whitespace only, and only from one unique original source span.

    Never repair words/numbers/negation. Preserve every proposed fragment in
    the alignment log; semantic checks still run against the aligned candidate.
    """
    aligned, changes = copy.deepcopy(payload), []
    if not isinstance(aligned, dict) or not isinstance(aligned.get("claims"), list):
        return aligned, changes
    for i, claim in enumerate(aligned["claims"]):
        if not isinstance(claim, dict):
            continue
        fragments = [("quote", None, claim.get("quote"))]
        if isinstance(claim.get("conditions"), list):
            fragments += [("conditions", j, x) for j, x in enumerate(claim["conditions"])]
        for field, index, fragment in fragments:
            if not isinstance(fragment, str) or not fragment.strip() or fragment in document:
                continue
            pattern = r"\s+".join(re.escape(x) for x in fragment.strip().split())
            matches = list(re.finditer(pattern, document))
            if len(matches) != 1:
                continue
            restored = matches[0].group()
            if index is None:
                claim[field] = restored
            else:
                claim[field][index] = restored
            changes.append({"claim_index": i, "field": field, "condition_index": index,
                            "proposed": fragment, "restored": restored,
                            "reason": "Unique whitespace-only match in supplied source"})
    return aligned, changes


def annotate(args, case, original, cache):
    context = "SOURCE CONTEXT:\n" + case["document"] + "\n\nBODY TO EXTRACT:\n" + case["body"]
    feedback, attempts = "", []
    for attempt in range(2):
        raw, key = request(args.endpoint, args.teacher_id, args.teacher_revision, INSTRUCTION,
                           context + feedback, cache, args.max_tokens)
        choice = raw["response"]["choices"][0]
        content = choice["message"]["content"]
        unaligned, alignments = None, []
        try:
            unaligned = json.loads(content)
            labels, alignments = align_evidence(unaligned, case["document"])
            errors = mechanically_validate(labels, case["document"])
            errors += mechanically_validate(labels, original)
        except (TypeError, ValueError) as exc:
            labels, errors = None, [str(exc)]
        if choice.get("finish_reason") != "stop":
            errors.append("Teacher completion did not stop normally")
        audit, audit_key = None, None
        if not errors:
            review, audit_key = request(args.endpoint, args.teacher_id, args.teacher_revision, CRITIC,
                                       context + "\n\nCANDIDATE:\n" + json.dumps(labels), cache, 500)
            try:
                review_choice = review["response"]["choices"][0]
                audit = json.loads(review_choice["message"]["content"])
                if review_choice.get("finish_reason") != "stop" or audit.get("verdict") not in ("accept", "reject", "uncertain"):
                    errors.append("Invalid critic response")
            except (TypeError, ValueError, AttributeError) as exc:
                errors.append("Critic parse: " + str(exc))
        eligible = not errors and isinstance(audit, dict) and audit.get("verdict") == "accept" and audit.get("issues") == []
        attempts.append({"teacher_cache_key": key, "critic_cache_key": audit_key,
                         "audit": audit, "validation_errors": errors, "eligible": eligible})
        if eligible:
            break
        feedback = "\n\nPREVIOUS CANDIDATE (failed validation):\n" + content
        feedback += "\nCHECKER FEEDBACK:\n" + json.dumps({"citation_errors": errors, "source_audit": audit})
        feedback += "\nProduce a corrected complete extraction. Use only the supplied source; preserve uncertainty."
    return {**case, "target": labels, "teacher_id": args.teacher_id, "teacher_revision": args.teacher_revision,
            "teacher_cache_key": key, "critic_cache_key": audit_key, "audit": audit,
            "unaligned_teacher_target": unaligned, "evidence_alignment": alignments,
            "validation_errors": errors, "training_eligible": eligible, "attempts": attempts,
            "provenance": "Local teacher generation and self-audit; not independent human certification"}


def generate(args):
    args.out.mkdir(parents=True, exist_ok=True)
    cache = args.out / "cache"
    cache.mkdir(exist_ok=True)
    manifest = json.loads((args.data / "manifest.json").read_text())
    snapshots = {}
    for key, source in manifest["sources"].items():
        original = (args.data / source["snapshot"]).read_bytes().decode("utf-8")
        if hashlib.sha256(original.encode()).hexdigest() != source["sha256"]:
            raise ValueError("Source snapshot hash changed")
        snapshots[key] = original
    rows = []
    cases = [c for c in manifest["examples"] if c["split"] in args.splits]
    pool = ThreadPoolExecutor(max_workers=getattr(args, "workers", 2))
    try:
        for row in pool.map(lambda case: annotate(args, case, snapshots[case["source_key"]], cache), cases):
            rows.append(row)
            (args.out / "labels.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            print(row["id"], row["split"], "eligible" if row["training_eligible"] else "quarantined", flush=True)
    finally:
        # Let the supervisor stop its server promptly on interruption; cancelling
        # queued work avoids generating more labels after the user stops a job.
        pool.shutdown(wait=False, cancel_futures=True)
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--endpoint", required=True)
    p.add_argument("--teacher-id", required=True)
    p.add_argument("--teacher-revision", required=True)
    p.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    p.add_argument("--max-tokens", type=int, default=3072)
    p.add_argument("--workers", type=int, default=2)
    a = p.parse_args()
    rows = generate(a)
    print(json.dumps({s: {"accepted": sum(r["split"] == s and r["training_eligible"] for r in rows),
                          "total": sum(r["split"] == s for r in rows)} for s in a.splits}, indent=2))
