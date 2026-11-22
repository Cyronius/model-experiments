"""Question-independent documents, fact-disjoint splits, and a lexical retriever."""
import argparse
import hashlib
import json
import math
import random
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def synthetic_records(n_train=256, n_dev=32, n_test=64, seed=42):
    """Random bindings, deliberately labelled fictional; no XDNA training leakage."""
    rng = random.Random(seed)
    docs, questions = [], []
    properties = {
        "alignment": (["16", "32", "64", "128", "256", "512"], "Required alignment", "bytes"),
        "queue": (["2", "3", "4", "6", "8", "12"], "Queue capacity", "entries"),
    }
    for split, count in [("train", n_train), ("dev", n_dev), ("test", n_test)]:
        for i in range(count):
            # Entity IDs have no relation to values. Test facts are encoded after fitting.
            entity = f"unit-{rng.getrandbits(64):016x}"
            doc_id = f"synthetic-{split}-{i:04d}"
            values = {key: rng.choice(spec[0]) for key, spec in properties.items()}
            text = (f"Fictional engineering test record. These are invented values, not XDNA guidance.\n"
                    f"Device: {entity}.\nRequired alignment: {values['alignment']} bytes.\n"
                    f"Queue capacity: {values['queue']} entries.")
            docs.append({"id": doc_id, "split": split, "domain": "synthetic", "text": text,
                         "source": None, "entity": entity, "values": values})
            for key, (_, label, unit) in properties.items():
                forms = [f"For {entity}, what is the {label.lower()} in {unit}?",
                         f"What {label.lower()} ({unit}) is specified for {entity}?"]
                for variant, question in enumerate(forms):
                    questions.append({"id": f"{doc_id}-{key}-{variant}", "doc_id": doc_id,
                                      "split": split, "domain": "synthetic", "question": question,
                                      "answers": [values[key]], "property": key,
                                      "instruction": "Answer with only the number."})
    return docs, questions


# Each answer is an explicit span from a finding, not an LLM-generated answer key.
# All real findings remain outside training. Scope is supplied to the model in the record.
XDNA_PROBES = [
    ("F-0002", "How many buffer descriptors does one column shim have in total?", ["16"], "16 BDs"),
    ("F-0006", "How many KB of program memory does a core have?", ["16"], "16 KB"),
    ("F-0046", "How many input DMA channels does each core have?", ["2"], "2 input"),
    ("F-0046", "How many S2MM DMA channels does each memtile have?", ["6"], "6 S2MM"),
    ("F-0053", "What is the maximum shim BD d0, in words?", ["1023"], "1023 words"),
    ("F-0238", "How many bytes should each core buffer's size be a multiple of to avoid the 512-bit alignment fault?", ["64"], "64 B"),
    ("F-0239", "How many bytes of total core data memory must accommodate both placed buffers and the stack?", ["65536"], "65,536 B"),
    ("F-0239", "In the cited L1 test, how many total bytes failed to fit?", ["66320"], "66,320 B"),
    ("F-0245", "How many bytes per core did the __udivsi3 helper add?", ["192"], "192 B"),
]


def load_findings(notes_root):
    docs = []
    for path in sorted((Path(notes_root) / "findings").glob("F-*.md")):
        original = path.read_text(encoding="utf-8-sig")
        match = re.match(r"\A---\s*\n(.*?)\n---\s*\n(.*)", original, re.S)
        if not match:
            raise ValueError(f"Missing frontmatter: {path}")
        # The source repo uses flat frontmatter with unquoted ': ' in stack
        # descriptions. Preserve that dialect instead of requiring valid YAML.
        meta = dict(re.findall(r"(?m)^([a-z_]+):[ \t]*(.*)$", match[1]))
        body = match[2]
        if not meta.get("stack") or not meta.get("evidence"):
            raise ValueError(f"Missing evidence/stack: {path}")
        # Claim and advice precede long stack metadata so the central content survives
        # a reasonable token cap. All metadata is retained in the exported record.
        text = body.split("**Where it was found:**")[0].strip()
        text += f"\nEvidence: {meta['evidence']}.\nVerified: {meta.get('last_verified')}.\nStack: {meta['stack']}"
        docs.append({"id": meta["id"], "split": "test", "domain": "xdna", "text": text,
                     "source": path.relative_to(notes_root).as_posix(), "source_sha256": digest(original),
                     "metadata": json.loads(json.dumps(meta, default=str))})
    return docs


def build_dataset(notes_root, output, n_train=256, n_dev=32, n_test=64, seed=42):
    docs, questions = synthetic_records(n_train, n_dev, n_test, seed)
    findings = load_findings(notes_root)
    by_id = {doc["id"]: doc for doc in findings}
    for i, (doc_id, question, answers, evidence_span) in enumerate(XDNA_PROBES):
        if doc_id not in by_id or evidence_span not in by_id[doc_id]["text"]:
            raise ValueError(f"Probe {doc_id} no longer supported by its source: {evidence_span}")
        questions.append({"id": f"xdna-{i:03d}", "doc_id": doc_id, "split": "test", "domain": "xdna",
                          "question": question, "answers": answers, "evidence_span": evidence_span,
                          "instruction": "Answer with only the number. Use the scope of the supplied record."})
    docs += findings
    output = Path(output)
    write_jsonl(output / "documents.jsonl", docs)
    write_jsonl(output / "questions.jsonl", questions)
    manifest = {"seed": seed, "notes_root": str(Path(notes_root).resolve()), "documents": len(docs),
                "xdna_documents": len(findings), "questions": len(questions),
                "training": "synthetic only; all XDNA documents are held out",
                "documents_sha256": digest((output / "documents.jsonl").read_text(encoding="utf-8")),
                "questions_sha256": digest((output / "questions.jsonl").read_text(encoding="utf-8"))}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def words(text):
    return re.findall(r"[a-z0-9_]+", text.lower())


class BM25:
    """Untrained selector; its text index is separate from the encoded value bank."""
    def __init__(self, documents):
        self.ids = [doc["id"] for doc in documents]
        self.counts = [Counter(words(doc["text"])) for doc in documents]
        self.lengths = [sum(count.values()) for count in self.counts]
        self.average = sum(self.lengths) / max(1, len(self.ids))
        df = Counter(word for count in self.counts for word in count)
        self.idf = {word: math.log(1 + (len(self.ids) - n + 0.5) / (n + 0.5)) for word, n in df.items()}

    def retrieve(self, question, k=1):
        terms = set(words(question))
        scores = []
        for i, count in enumerate(self.counts):
            norm = 1.2 * (0.25 + 0.75 * self.lengths[i] / max(self.average, 1))
            score = sum(self.idf.get(word, 0) * count[word] * 2.2 / (count[word] + norm)
                        for word in terms if count[word])
            scores.append((score, self.ids[i]))
        return [doc_id for _, doc_id in sorted(scores, key=lambda item: (-item[0], item[1]))[:k]]


def normalize_answer(text):
    text = re.sub(r"(?<=\d),(?=\d)", "", text.strip().lower())
    return re.sub(r"\s+", " ", text).strip(" .`\n\t")


def exact_match(prediction, answers):
    return normalize_answer(prediction) in {normalize_answer(answer) for answer in answers}


def numeric_match(prediction, answers):
    """Tolerates units/prose, but rejects multiple/conflicting numbers."""
    nums = re.findall(r"(?<![a-z0-9_])[+-]?\d+(?:\.\d+)?", normalize_answer(prediction))
    return len(nums) == 1 and nums[0] in {normalize_answer(answer) for answer in answers}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--notes", type=Path, default=Path("C:/code/xdna2-notes"))
    parser.add_argument("--out", type=Path, default=ROOT / "data")
    parser.add_argument("--train-docs", type=int, default=256)
    parser.add_argument("--dev-docs", type=int, default=32)
    parser.add_argument("--test-docs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(build_dataset(args.notes, args.out, args.train_docs, args.dev_docs, args.test_docs, args.seed), indent=2))
