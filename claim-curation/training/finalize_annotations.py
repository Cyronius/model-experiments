#!/usr/bin/env python3
"""Retained agent-authored reference annotations and explicit teacher review.

These annotations were authored before student document training or generation.
They assess assertions in the supplied notes, not independently verified truth.
"""
from __future__ import annotations
import hashlib
import json
import re
from pathlib import Path
from teacher import mechanically_validate

ROOT = Path(__file__).resolve().parent


def finalize():
    manifest = json.loads((ROOT / "data/manifest.json").read_text())
    cases = {c["id"]: c for c in manifest["examples"]}
    original = ROOT / "labels/teacher-pass-20261025.jsonl"
    if not original.exists():
        original.write_bytes((ROOT / "labels/labels.jsonl").read_bytes())
    teachers = {r["id"]: r for r in map(json.loads, original.read_text().splitlines())}
    reviews = json.loads((ROOT / "labels/review-decisions.json").read_text())["reviews"]
    references = {}

    def add(key, bullet, statement, conditions=()):
        c = cases[key]
        bullets = re.findall(r"^- .*?(?=\n- |\Z)", c["body"], re.M | re.S)
        references.setdefault(key, {"claims": []})["claims"].append(
            {"statement": statement, "quote": bullets[bullet].rstrip(),
             "conditions": [c["title"], *conditions]})

    # Validation: the first document in the fixed validation order. The other
    # three dense validation passages remain explicitly unannotated in this pilot.
    v = "D-be58f86360bdf45c1e7b"
    reported = ("Third-party results, not rerun here.", "Strix Halo, NOT Strix Point.", "DRM driver 0.7.0")
    facts = [
        (0, "Hipfire reports that its libc-only Linux/Strix Halo path ran mlir-aie PDIs and instruction streams bit-exact against XRT, including after context destroy/recreate."),
        (0, "The reported raw-dispatch setup uses a 64 MiB DEV_HEAP BO at a 64 MiB-aligned host virtual address."),
        (0, "The reported CREATE_HWCTX configuration uses all-zero QoS, max_opc 2048 and num_tiles 32."),
        (0, "The reported raw-dispatch sequence uses DEV BOs for the PDI and instructions, SHMEM BOs for data and one CMD BO."),
        (0, "After EXEC_CMD, the reported path waits on the hardware-context syncobj timeline at the EXEC_CMD sequence with WAIT_FOR_SUBMIT and checks ERT state 4."),
        (1, "Hipfire reports an ERT_START_CU packet header of 0x30010001 with cu_mask 1."),
        (1, "The reported ERT_START_CU payload uses opcode 3 and encodes instruction size in bytes, not words."),
        (1, "In the reported ERT_START_CU packet, bo0 is at payload offset 0x14, making the u64 slots 4-mod-8 unaligned."),
        (2, "Hipfire reports that kernel 7.0 has no WAIT_CMD ioctl and that a syncobj timeline wait works instead."),
        (3, "On the reported driver, SYNC_BO FROM_DEVICE returns EINVAL for SHMEM BOs."),
        (3, "Hipfire reports that SYNC_BO TO works for SHMEM BOs on the reported driver."),
        (3, "The reported readback path uses CPU clflush on each 64 B line with mfence before and after."),
        (3, "The hipfire source says its readback flush matches the installed XRT shim's default non-coherent clflush_data sync."),
        (4, "The hipfire report requires flushing the output buffer before a run to prevent a later dirty-line eviction from overwriting device output."),
        (5, "Hipfire reports a readback-flush average of 1.475 ms for a 35.65 MB GEMM output over all iterations."),
        (5, "Hipfire reports a readback-flush average of 17.345 microseconds for a 524,288 B output over all iterations."),
        (6, "Hipfire reports that only bo0 through bo4 reach the firmware."),
        (6, "The reported designs consequently use at most five DDR-patched buffers."),
    ]
    for bullet, statement in facts:
        add(v, bullet, statement, reported)

    k = "D-4b57f363a78d5d55d512"
    scope = ("WSL's Ubuntu 24.04", "Measured by ggml-xdna (WSL, 2026-10-12)")
    add(k, 0, "In ggml-xdna's reported Ubuntu 24.04 WSL environment, the Linux NPU device node /dev/accel is absent.", scope)
    add(k, 0, "In ggml-xdna's reported Ubuntu 24.04 WSL environment, the GPU passes through /dev/dxg but the NPU does not pass through.", scope)
    add(k, 0, "In that reported WSL environment, NPU-side execution and timing are unavailable; only builds and host-only tests can be run.", scope)

    k = "D-94becbcc50724f7f2cc3"
    hearsay = ("Reported by the fork (hearsay)",)
    add(k, 0, "The note identifies Nulifyer/llama-xdna-hybrid as a fork of ggml-xdna v0.1.0 (f5fe5b4), with ten commits from 2026-10-06.", hearsay)
    add(k, 1, "The third-party fork reports passing its NPU tests on a Ryzen AI 9 HX 370 under Ubuntu 24.04 with XRT 2.25 and Ubuntu 26.04 with XRT 2.21.", hearsay)
    add(k, 1, "The third-party fork reports that an 8,192-token Qwen3.5-2B prompt took 7-20% less time than the GPU alone in two sittings on the reported Linux/HX 370 setup.", (*hearsay, "Qwen3.5-2B, two sittings", "Ryzen AI 9 HX 370"))
    add(k, 2, "The third-party fork reports that its build against ggml-xdna's vendored XRT 2.20 headers ran with XRT 2.25.", hearsay)
    add(k, 3, "The third-party fork reports that its NPU code needed almost no change from Windows on the same chip, with most changes in platform code.", hearsay)
    add(k, 3, "The reported Linux platform changes included locating the add-on folder with dladdr.", hearsay)
    add(k, 3, "The reported Linux platform changes included obtaining memory information from /proc/meminfo and a container memory limit.", hearsay)

    k = "D-67113b2f7eeb685c2042"
    unknowns = ("run date not stated", "NPU power mode not stated", "version or commit not stated")
    add(k, 0, "For Qwen3.6 35B-A3B on the reported HX 370 setup with OpenFlowLM running the whole model on the NPU, q4_1 prompt-reading rates were 134, 136, 139 and 137 tok/s.", unknowns)
    add(k, 1, "For the reported Qwen3.6 35B-A3B OpenFlowLM q4_1 NPU runs on the HX 370, 32-token reply-writing rates were 12.4, 12.0, 10.8 and 8.9 tok/s.", unknowns)
    add(k, 2, "For the reported Qwen3.6 35B-A3B OpenFlowLM NPU runs with 8-bit weights and 4-bit experts, prompt-reading rates were 112, 116, 118 and 114 tok/s.", unknowns)
    add(k, 3, "For the reported Qwen3.6 35B-A3B OpenFlowLM NPU runs with 8-bit weights and 4-bit experts on the HX 370, reply-writing rates were 7.0, 8.2, 7.8 and 6.6 tok/s.", unknowns)
    add(k, 4, "The note reports that OpenFlowLM reply speed dropped from 12.4 to 8.9 tok/s as the prompt grew from 1k to 8k, while GPU-engine speeds held steady.", unknowns)

    k = "D-ac65a301dc94089c12ef"
    scope = ("llama.cpp b10944 Vulkan", "run date not stated", "NPU power mode not stated")
    add(k, 0, "For the reported Qwen3.6 35B-A3B Q4_K_M prompt-reading runs on the HX 370, llama.cpp with the add-on achieved 223, 216, 259 and 235 tok/s versus GPU-alone rates of 223, 202, 197 and 202 tok/s.", scope)
    add(k, 0, "The reported Qwen3.6 35B-A3B Q4_K_M prompt-reading lead from the add-on was up to 1.31x over the GPU alone.", scope)
    add(k, 1, "For the reported Qwen3.6 35B-A3B Q8_0 prompt-reading runs on the HX 370, the add-on achieved 170, 196, 199 and 198 tok/s versus GPU-alone rates of 181, 168, 199 and 187 tok/s.", scope)
    add(k, 1, "The reported Qwen3.6 35B-A3B Q8_0 prompt-reading lead from the add-on was up to 1.17x over the GPU alone.", scope)
    add(k, 2, "The add-on leaves reply generation to the GPU, so the note expects its reply rows to equal the GPU-alone rows.", scope)
    add(k, 2, "The note reports reply-row differences of up to 14% and interprets 14% as the noise on figures in these tables.", scope)
    add(k, 3, "For the reported Qwen3.6 35B-A3B model on the HX 370, the add-on cannot send the expert step to the NPU; only attention and shared weights reach the NPU.", scope)

    rows = []
    for case in manifest["examples"]:
        key = case["id"]
        if key in references:
            row = {**case, "target": references[key], "training_eligible": True,
                   "teacher_id": "Interactive Codex annotation agent", "teacher_revision": "Deployment revision unavailable; exact retained annotations are the reproducible artifact",
                   "provenance": "Agent-authored source-assertion reference, written before student training/inference; not independent human gold",
                   "audit": {"verdict": "agent_source_review", "issues": []}, "validation_errors": []}
        elif case["split"] == "train":
            row = teachers[key]
            if row["document"] != case["document"]:
                raise ValueError("Teacher proposal and frozen input differ")
            review = reviews.get(key, {"decision": "retain_proposal", "issues": []})
            row["agent_review"] = review
            if review["decision"] == "reject":
                row["training_eligible"] = False
            elif review["decision"] == "accept_over_teacher_critic":
                row["training_eligible"] = True
        else:
            row = {**case, "target": None, "training_eligible": False,
                   "provenance": "Unannotated validation passage; excluded from this bounded pilot",
                   "validation_errors": ["No accepted reference annotation"]}
        if row["training_eligible"]:
            source = manifest["sources"][row["source_key"]]
            original_text = (ROOT / "data" / source["snapshot"]).read_bytes().decode()
            errors = mechanically_validate(row["target"], row["document"])
            errors += mechanically_validate(row["target"], original_text)
            if errors:
                raise ValueError(f"{key}: {errors}")
        rows.append(row)
    destination = ROOT / "labels/labels.jsonl"
    destination.write_text("".join(json.dumps(r) + "\n" for r in rows))
    summary = {"annotation_policy": "Source-reviewed teacher training proposals; agent-authored validation and test references",
               "independent_human_gold": False, "labels_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
               "splits": {s: {"eligible": sum(r["split"] == s and r["training_eligible"] for r in rows),
                                "total": sum(r["split"] == s for r in rows)} for s in ("train", "validation", "test")}}
    (ROOT / "labels/final-supervision.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    finalize()
