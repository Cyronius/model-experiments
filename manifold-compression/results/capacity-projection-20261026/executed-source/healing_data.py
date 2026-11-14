"""Complete-response data and deterministic checks for a small healing pilot."""
import argparse
import hashlib
import json
import random
import re
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoTokenizer

REVISION = "8049631c405ae6576f93f445c6b8166f76f5505a"


def chat_ids(tokenizer, messages, generation=False):
    text = tokenizer.apply_chat_template(messages, tokenize=False,
                                        add_generation_prompt=generation, enable_thinking=False)
    return tokenizer(text, add_special_tokens=False).input_ids


def encode_complete(tokenizer, messages, max_length=384):
    """Score the final assistant response, including EOS; never truncate it.

    Earlier turns supply context. Scoring the last assistant turn avoids
    misaligned masks when a chat template renders past reasoning differently.
    """
    if not messages or messages[-1]["role"] != "assistant":
        raise ValueError("A complete final assistant response is required")
    prefix = chat_ids(tokenizer, messages[:-1], generation=True)
    full = chat_ids(tokenizer, messages)
    if full[:len(prefix)] != prefix:
        raise ValueError("Chat template prefix does not match the completed conversation")
    stop = tokenizer.eos_token_id
    tail = full[len(prefix):]
    if stop not in tail:
        raise ValueError("Completed response has no ending token")
    end = len(prefix) + tail.index(stop) + 1
    # Reject embedded special endings followed by substantive text.
    if tokenizer.decode(full[end:], skip_special_tokens=True).strip():
        raise ValueError("Text occurs after the assistant ending token")
    full = full[:end]
    if len(full) > max_length:
        return None
    if len(full) <= len(prefix) + 1:
        raise ValueError("Empty assistant response")
    return {"input_ids": full, "answer_start": len(prefix),
            "positions": list(range(len(prefix) - 1, len(full) - 1)),
            "targets": full[len(prefix):], "eos_token_id": stop}


def pair(prompt, answer):
    return [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}]


def curated_train():
    examples = [
        ("What does Python print for print([2, 4, 6, 8][1:3])? Explain briefly.",
         "It prints [4, 6]. The slice includes index 1 and stops before index 3."),
        ("Explain what dict.get('key', 0) does in Python in two sentences.",
         "It returns the value associated with 'key' if that key exists. Otherwise, it returns 0 without raising KeyError."),
        ("Give a Python expression that squares every number in a list called values.",
         "[value ** 2 for value in values]"),
        ("What does Python print for print(len(set([3, 3, 7])))? Answer briefly.",
         "It prints 2, because the set contains two distinct values: 3 and 7."),
        ("Return only a JSON object with fruit set to pear and quantity set to 4.",
         '{"fruit": "pear", "quantity": 4}'),
        ("List exactly three ways to save water, one per numbered line.",
         "1. Repair leaking taps.\n2. Take shorter showers.\n3. Run the dishwasher only when full."),
        ("Respond with only the word ready in lowercase.", "ready"),
        ("Summarize in one sentence: Lina planted basil on Monday. She watered it every morning. New leaves appeared on Friday.",
         "Lina planted basil on Monday, watered it daily, and saw new leaves by Friday."),
        ("What is 14 times 16? Show one short calculation.", "14 × 16 = 14 × (10 + 6) = 140 + 84 = 224."),
        ("A book costs 18 dollars and a pen costs 3 dollars. What do two books and one pen cost?",
         "Two books cost 2 × 18 = 36 dollars. Adding the pen gives 36 + 3 = 39 dollars."),
        ("Why do shadows change length during a day? Use two sentences.",
         "The Sun's angle in the sky changes as Earth rotates. A low Sun makes long shadows, while a higher Sun makes shorter shadows."),
        ("Explain why a metal spoon feels colder than a wooden spoon at the same room temperature.",
         "Metal conducts heat away from your hand faster than wood. Your skin cools more quickly, making the metal feel colder even though both spoons started at the same temperature."),
    ]
    rows = [{"id": f"curated-{i}", "source": "locally reviewed pilot example", "messages": pair(p, a)}
            for i, (p, a) in enumerate(examples)]
    conversations = [
        pair("Remember that the parcel code is R8T2.", "The parcel code is R8T2.") + pair("What is the code? Return only the code.", "R8T2"),
        pair("Start a shopping list with rice and beans.", "Rice, beans.") + pair("Add carrots and return just a JSON array.", '["rice", "beans", "carrots"]'),
        pair("We have 12 chairs.", "There are 12 chairs.") + pair("We lend 5 chairs to a neighbor. How many remain?", "12 − 5 = 7 chairs remain."),
        pair("Suggest a short title for a story about a lighthouse.", "The Last Beacon") + pair("Make that title uppercase. Output only the title.", "THE LAST BEACON"),
    ]
    rows += [{"id": f"curated-multi-{i}", "source": "locally reviewed pilot example", "messages": m}
             for i, m in enumerate(conversations)]
    return rows


def eval_cases():
    def case(name, prompt, kind, expected=None, **extra):
        return {"id": name, "messages": [{"role": "user", "content": prompt}],
                "check": {"kind": kind, "expected": expected, **extra}}
    cases = [
        case("json", 'Return only a JSON object with city set to Oslo and count set to 3.', "json", {"city": "Oslo", "count": 3}),
        case("copy", "Repeat exactly this code and nothing else: A7x-P4q", "exact", "A7x-P4q"),
        case("multiply", "What is 29 times 17? Return only the number.", "exact", "493"),
        case("sort", "Sort these numbers in ascending order: 9, -2, 4, 0. Return only a comma-separated list.", "numbers", [-2, 0, 4, 9]),
        case("units", "How many milliliters are in 2.5 liters? Return only the number.", "exact", "2500"),
        case("logic", "All fens are lums. All lums are peks. Are all fens peks? Answer only yes or no.", "exact", "yes"),
        case("five-lines", "List these five colors, one per line and nothing else: red, blue, green, yellow, purple.", "lines", ["red", "blue", "green", "yellow", "purple"]),
        case("poem", "Write a four-line poem about rain. Output only the poem, with no title.", "line_count", 4),
        case("summary", "Summarize in one sentence: The Redwood library will reopen on Friday. It was closed for repairs. Fifteen new desks were installed.", "summary", None),
        case("alias", "What does this Python code print, and why? a = [5, 8]; b = a; b.append(13); print(a). Answer in at most 60 words.", "alias", None),
        case("ice", "Why does ice float on water? Answer in at most 80 words.", "ice", None),
        case("interview", "How should I prepare for a job interview? Give five numbered tips in at most 150 words.", "numbered", 5, max_words=150),
        case("dog", "What should I consider when adopting a dog? Give a concise checklist in at most 150 words.", "word_limit", 150),
        case("hash-table", "Explain what a hash table is to a beginner programmer. Keep your answer under 150 words.", "word_limit", 149),
    ]
    cases += [
        {"id": "multi-memory", "messages": pair("Remember: my preferred color is teal.", "Your preferred color is teal.") + [{"role": "user", "content": "What is my preferred color? Output only the color."}], "check": {"kind": "exact", "expected": "teal"}},
        {"id": "multi-edit", "messages": pair("My list is apples, bread, and tea.", "Your list is apples, bread, and tea.") + [{"role": "user", "content": "Remove bread and add cheese. Return only the updated list as a JSON array, keeping the remaining order."}], "check": {"kind": "json", "expected": ["apples", "tea", "cheese"]}},
    ]
    return cases


def check_answer(text, spec):
    """Transparent mechanical/rubric checks, not a comprehensive quality judge."""
    text = text.strip()
    lower = text.lower()
    kind, expected = spec["kind"], spec.get("expected")
    words = re.findall(r"\b\w+\b", text)
    lines = [s.strip() for s in text.splitlines() if s.strip()]
    if kind == "exact":
        return text.casefold() == expected.casefold() if spec.get("case_insensitive") else text == expected
    if kind == "json":
        try:
            return json.loads(text) == expected
        except (ValueError, TypeError):
            return False
    if kind == "numbers":
        try:
            return [int(s.strip()) for s in text.split(",")] == expected
        except ValueError:
            return False
    if kind == "lines":
        return lines == expected
    if kind == "line_count":
        return len(lines) == expected
    if kind == "word_limit":
        return 5 <= len(words) <= expected
    if kind == "numbered":
        return len(re.findall(r"(?m)^\s*\d+[.)]\s", text)) == expected and len(words) <= spec["max_words"]
    if kind == "summary":
        return all(s in lower for s in ("redwood", "friday")) and ("15" in lower or "fifteen" in lower) and len(re.findall(r"[.!?](?:\s|$)", text)) == 1
    if kind == "alias":
        return bool(re.search(r"\[\s*5\s*,\s*8\s*,\s*13\s*\]", text)) and ("same list" in lower or "same object" in lower) and len(words) <= 60
    if kind == "ice":
        return ("less dense" in lower or "lower density" in lower) and len(words) <= 80
    raise ValueError(kind)


def scoring_spec(case):
    spec = dict(case["check"])
    # These prompts request a yes/no answer or a color, not lowercase spelling.
    # The code-copy case remains case sensitive. This correction was identified
    # in the stock baseline, before viewing candidate generations.
    if case["id"] in ("logic", "multi-memory"):
        spec["case_insensitive"] = True
    return spec


def generation_metrics(text, token_ids, eos):
    words = re.findall(r"\w+", text.lower())
    grams = [tuple(words[i:i + 4]) for i in range(len(words) - 3)]
    repeat = 1 - len(set(grams)) / len(grams) if grams else 0.
    # Repetition is a diagnostic flag; repeated phrases can be legitimate.
    return {"ended": eos in token_ids, "generated_tokens": len(token_ids),
            "repeated_4gram_fraction": repeat, "loop_flag": len(words) >= 40 and repeat > .20}


def build_data(out, source, max_length=384):
    tokenizers = [AutoTokenizer.from_pretrained(p, local_files_only=True) for p in
                  ("models/manifold-audit/Qwen3.5-2B", "models/manifold-audit/SmolLM2-1.7B-Instruct")]
    used_prompts = set()
    def acceptable(messages):
        return all(encode_complete(t, messages, max_length) is not None for t in tokenizers)
    train = curated_train()
    assert all(acceptable(r["messages"]) for r in train)
    for row in train:
        used_prompts.add(row["messages"][0]["content"].strip().casefold())
    splits = {}
    for split, needed in [("train_sft", 48), ("test_sft", 12)]:
        file = next((source / "data").glob(f"{split}-00000-*.parquet"))
        table = pq.read_table(file, columns=["prompt_id", "messages"])
        order = list(range(len(table)))
        random.Random(137).shuffle(order)
        selected = []
        for idx in order:
            raw = table.slice(idx, 1).to_pylist()[0]
            conversation = raw["messages"]
            if len(conversation) < 2 or conversation[0]["role"] != "user":
                continue
            prompt = conversation[0]["content"].strip().casefold()
            if prompt in used_prompts:
                continue
            # Retain whole turns, preferring two-turn context where it fits.
            candidates = [conversation[:4], conversation[:2]] if len(conversation) >= 4 else [conversation[:2]]
            for messages in candidates:
                if [m["role"] for m in messages] != ["user", "assistant"] * (len(messages) // 2):
                    continue
                if any(len(m["content"]) > 4000 for m in messages):
                    continue
                if acceptable(messages):
                    selected.append({"id": f"{split}-{raw['prompt_id']}", "source": "HuggingFaceH4/ultrachat_200k", "messages": messages})
                    used_prompts.add(prompt)
                    break
            if len(selected) == needed:
                break
        if len(selected) != needed:
            raise ValueError("Not enough complete short conversations")
        splits[split] = selected
    train += splits["train_sft"]
    random.Random(137).shuffle(train)
    cases = eval_cases()
    for c in cases:
        assert c["messages"][0]["content"].strip().casefold() not in used_prompts
    result = {"revision": REVISION, "seed": 137, "max_length": max_length,
              "train": train, "validation": splits["test_sft"], "generation_eval": cases}
    encoded_counts = {}
    for name, tok in zip(("qwen35", "smollm2"), tokenizers):
        encoded_counts[name] = {}
        for split, rows in (("train", train), ("validation", splits["test_sft"])):
            rows = [encode_complete(tok, r["messages"], max_length) for r in rows]
            encoded_counts[name][split] = {"rows": len(rows), "input_tokens": sum(len(r["input_ids"]) for r in rows),
                                           "assistant_tokens": sum(len(r["targets"]) for r in rows), "ending_tokens": len(rows)}
    result["counts"] = encoded_counts
    result["source_sha256"] = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in (source / "data").glob("*.parquet")}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(encoded_counts, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--source", type=Path, default=Path("target/manifold-audit/ultrachat"))
    args = ap.parse_args()
    build_data(args.out, args.source)
