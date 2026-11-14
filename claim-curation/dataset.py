"""Fixed pilot passages and gold anchors, authored before held-out inference.

Gold is an auditable pilot annotation, not an independently adjudicated dataset.
Synthetic fixtures are explicitly separated from observations about hardware.
"""
import hashlib
import json
import subprocess
from pathlib import Path

PATHS = {
    "array": "raw/phlegm/2026-09-02-array-resource-limits-dma-l1-program-memory.md",
    "arithmetic": "raw/phlegm/2026-09-26-aie2p-arithmetic-and-stack-traps.md",
    "abi": "raw/phlegm/2026-09-02-instruction-stream-ddr-patch-and-kernel-abi.md",
    "llm": "raw/LLMNpuTest/2026-09-01-dma-l1-program-memory-limits-and-silent-faults.md",
    "granite": "raw/OpenFlowLM-Next3/2026-10-19-oflm-iron-peano-aiecc-dma-traps-and-hangs.md",
}


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def gold(name, patterns, scope=()):
    return {"id": name, "patterns": patterns, "scope_patterns": list(scope)}


def build(notes: Path, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    (out / "sources").mkdir(exist_ok=True)
    docs = {key: (notes / path).read_text() for key, path in PATHS.items()}
    source_manifest = {}
    for key, text in docs.items():
        (out / "sources" / f"{key}.md").write_text(text)
        header = text.split("---", 2)[1]
        context = dict(line.split(":", 1) for line in header.splitlines() if ":" in line)
        refs = next((line for line in text.splitlines() if line.startswith("Sources")), "")
        source_manifest[key] = {
            "path": PATHS[key], "snapshot": f"sources/{key}.md", "sha256": digest(text),
            "context": {k.strip(): v.strip() for k, v in context.items()},
            "upstream_reference_text": refs, "upstream_references_verified": False,
            "source_level": "project_note_or_digest",
        }

    cases, golds = [], {}

    def add(name, key, text, expected):
        original = docs[key]
        start = original.index(text)
        header = original[:original.index("---", 3) + 3]
        cases.append({"id": name, "source_key": key, "synthetic": False,
                      "text": header + "\n\n" + text, "excerpt": text,
                      "source_start": start, "source_end": start + len(text)})
        golds[name] = expected

    def line(key, needle):
        return next(l for l in docs[key].splitlines() if needle in l)

    add("core-dma", "array", line("array", "- Every core:"), [
        gold("two-inputs", [r"(?:core|compute.?tile)", r"(?:2|two).{0,15}input"]),
        gold("two-outputs", [r"(?:core|compute.?tile)", r"(?:2|two).{0,15}output"]),
        gold("third-input", [r"(?:third|3).{0,25}input|input.{0,25}(?:third|3)", r"cannot|can't|not |unavailable|only"]),
    ])
    add("shim-wrap", "array", line("array", "outermost dimension wraps silently"), [
        gold("wrap-64", [r"(?:outermost|outer).{0,30}dimension|dimension.{0,20}(?:outermost|outer)", r"wrap", r"64"]),
        gold("split-hang", [r"split|several fills", r"hung|hang"]),
        gold("middle-dimension", [r"(?:large|count)", r"middle", r"dimension"]),
    ])
    add("granite-stack", "granite", line("granite", "- Granite attention workers:"), [
        gold("granite-nan", [r"0x1800", r"nan"], [r"granite", r"(?:pos|position).{0,12}(?:>=|≥|at least|4|four)"]),
        gold("granite-working", [r"0x3000", r"work|succeed|correct|resolve"], [r"granite"]),
        gold("tree-spill", [r"unrolled", r"tree", r"spill"]),
    ])
    soft = line("granite", "softmax with four interleaved exponentials")
    soft = soft[:soft.index(" Read the frame")]
    add("softmax-stack", "granite", soft, [
        gold("five-kb-frame", [r"5\s*(?:kb|kilobyte)", r"frame"], [r"four|4", r"exponential"]),
        gold("four-kb-fault", [r"4\s*(?:kb|kilobyte)", r"hang|hung|nan"], [r"softmax|attn_fa"]),
        gold("softmax-working", [r"0x2000", r"work|succeed|resolve"], [r"softmax|attn_fa"]),
    ])
    transfer = line("llm", "- A 128-byte shim transfer") + "\n" + line("llm", "- The GEMV designs have 128-byte")
    add("transfer-size", "llm", transfer, [
        gold("small-transfer-zero", [r"128", r"(?:transfer|shim)", r"zero|empty"], [r"attention|128"]),
        gold("padding-fix", [r"1024", r"pad", r"fix|resolve|correct"]),
        gold("gemv-exception", [r"gemv", r"exact|correct|work"], [r"(?:whole|entire|full).{0,20}drain"]),
        gold("transfer-not-element", [r"transfer", r"(?:not|rather than).{0,40}(?:element|objectfifo)"]),
    ])
    add("l1-budget", "llm", line("llm", "- Budget used: 64 KB"), [
        gold("budget", [r"62208", r"budget|available|memory|64\s*kb"], [r"0xd00"]),
        gold("batch-fit", [r"61952", r"built|ran|fit|work|succeed"], [r"batch.{0,5}2"]),
        gold("wrong-rule", [r"61184|63\s*kb", r"reject|incorrect|wrong|would"], [r"rule|budget"]),
    ])
    rot = line("abi", "Both paths loaded ROT13")
    rot = rot[rot.index("Both paths loaded ROT13"):]
    add("rot13", "abi", rot, [
        gold("classic-time", [r"classic", r"0[.]602"]),
        gold("elf-time", [r"elf", r"0[.]447"]),
        gold("state-complete", [r"state.{0,6}4|4.{0,12}state", r"complet"]),
        gold("byte-exact", [r"byte.exact"]),
    ])
    add("softmax-mask", "arithmetic", line("arithmetic", "- Masked softmax scores"), [
        gold("lowest", [r"softmax|mask", r"bf16|bfloat16", r"lowest"], [r"mask"]),
        gold("avoid-infinity", [r"-inf|negative infinity", r"finite|not|avoid|cannot|should"]),
    ])
    rounding = docs["arithmetic"].split("## Core rounding mode\n\n", 1)[1].split("\n\n##", 1)[0]
    add("rounding", "arithmetic", rounding, [
        gold("default-floor", [r"floor", r"start|default"]),
        gold("first-dispatch", [r"first.{0,35}dispatch", r"(?:2x|twice|2 times)|1[.]8e-4"], [r"context|load"]),
        gold("rounding-fix", [r"rounding", r"first", r"every|each"], [r"kernel|dispatch"]),
    ])

    synthetic = [
        ("synthetic-negation", "Synthetic test fixture, not a hardware finding.\nOn device Cedar with compiler 1.2, the ash kernel took 7 ms at batch 2. The firmware version was not recorded. This measurement does not establish the latency at other batch sizes.", [
            gold("ash-latency", [r"ash", r"7\s*ms"], [r"cedar", r"1[.]2", r"batch.{0,5}2"]),
            gold("firmware-unknown", [r"firmware", r"not.{0,20}recorded|unknown|unspecified"]),
            gold("batch-limitation", [r"(?:not|cannot|unknown|unestablished)", r"other.{0,15}batch|batch.{0,15}(?:other|sizes)"]),
        ]),
        ("synthetic-questions", "Synthetic test fixture.\nQuestions for future work:\n- How much stack would the birch kernel require?\n- Could compiler 9.9 change that requirement?", []),
        ("synthetic-version-scope", "Synthetic test fixture, not a hardware finding.\nFor the birch kernel, stack 0x1800 was sufficient only with compiler 1.2. Compiler 1.3 was not tested. The result must not be generalized to compiler 1.3.", [
            gold("version-only", [r"0x1800", r"sufficient|work"], [r"birch", r"only", r"1[.]2"]),
            gold("version-untested", [r"1[.]3", r"not.{0,20}test|untested"]),
        ]),
    ]
    for name, text, expected in synthetic:
        key = name
        docs[key] = text
        (out / "sources" / f"{key}.md").write_text(text)
        source_manifest[key] = {"path": f"synthetic://{key}", "snapshot": f"sources/{key}.md",
                                "sha256": digest(text), "context": {}, "source_level": "synthetic_fixture",
                                "upstream_reference_text": "", "upstream_references_verified": False}
        cases.append({"id": name, "source_key": key, "synthetic": True, "text": text,
                      "excerpt": text, "source_start": 0, "source_end": len(text)})
        golds[name] = expected

    pairs = []

    def pair(name, a, b, relation, reason, synthetic=True):
        pairs.append({"id": name, "claim_a": a, "claim_b": b,
                      "synthetic": synthetic, "gold_relation": relation, "gold_reason": reason})

    pair("same-run-different-result", "Run R17: device Cedar, compiler 1.2, ash kernel, batch 2. All 20 repetitions completed successfully.",
         "Run R17: device Cedar, compiler 1.2, ash kernel, batch 2. None of the 20 repetitions completed successfully.", "contradicts", "Same run and quantifier, opposite outcomes.")
    pair("different-workload", "For the Granite attention workers, 0x1800 read back NaN; 0x3000 worked.",
         "For LLMNpuTest GEMV and elementwise workers, stack_size=0xD00 ran successfully.", "different_scope", "Different kernel workloads; stack requirements may differ.", False)
    pair("different-versions", "With compiler 1.2, the birch kernel worked at stack 0x1800.",
         "With compiler 1.3, the birch kernel hung at stack 0x1800.", "different_scope", "Explicitly different compiler versions.")
    pair("unit-equivalence", "On the same device, core memory capacity is 64 KiB.", "On the same device, core memory capacity is 65536 bytes.", "supports", "64 KiB equals 65536 bytes.")
    pair("unknown-scope", "A worker with stack 0x1800 ran successfully; compiler version and workload were not recorded.",
         "Granite attention workers with stack 0x1800 produced NaN.", "insufficient_information", "Cannot establish matching scope from missing workload and version.")
    pair("same-run-timing", "Run T4 measured the classic ROT13 path at exactly 0.602 ms.",
         "Run T4 measured the classic ROT13 path at exactly 0.447 ms.", "contradicts", "Same named measurement and metric have different exact values.")
    pair("different-paths", "The classic ROT13 path completed in 0.602 ms.", "The ELF ROT13 path completed in 0.447 ms.", "different_scope", "Different load paths, not competing values for one path.", False)
    pair("partial-versus-universal", "Every tested input produced a finite result in run V9.",
         "One tested input produced NaN in run V9.", "contradicts", "A counterexample refutes the universal statement for the same run.")
    pair("untested-not-failed", "Compiler 1.3 was not tested for the birch kernel.",
         "Compiler 1.3 fails for the birch kernel.", "insufficient_information", "Absence of testing does not establish a failure or its negation.")
    pair("table-paraphrase", "Granite attention workers: 0x1800 returns NaN from position 4 onward; 0x3000 works.",
         "The Granite attention test succeeded with stack 0x3000, but stack 0x1800 gave NaN at pos >= 4.", "supports", "Same scoped observations, paraphrased.", False)

    manifest = {"version": "pilot-v1", "notes_commit": subprocess.check_output(["git", "-C", str(notes), "rev-parse", "HEAD"], text=True).strip(),
                "sources": source_manifest, "passages": cases,
                "annotation": "Agent-authored pilot gold, auditable against retained sources; not independent human ground truth."}
    for name, content in (("manifest.json", manifest), ("gold.json", golds), ("pairs.json", pairs)):
        (out / name).write_text(json.dumps(content, indent=2) + "\n")
    print(f"Prepared {len(cases)} passages, {sum(map(len, golds.values()))} gold claim anchors, {len(pairs)} relation pairs.")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--notes", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    build(a.notes, a.out)
