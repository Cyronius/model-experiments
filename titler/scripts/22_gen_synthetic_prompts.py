"""Generate varied short / vague / messy first messages with an OpenAI-compatible server (2026-10-18).

The wild pools have almost no context-free one-liners, which is where the v2 titler loses. Output is
unlabeled prompts; the teacher titles them in the next step.

    TITLER_API_BASE=... TITLER_API_KEY=... python scripts/22_gen_synthetic_prompts.py --model Qwen/Qwen3-14B-AWQ --calls 3600 --output data/24_synth2_raw.jsonl
"""

import argparse
import asyncio
import json
import random
import re
from pathlib import Path

import httpx

from common.lmstudio_client import BASE_URL, HEADERS

TOPICS = """cooking, personal finance, python, javascript, sql, excel, fitness, travel, relationships, parenting, job search,
resume writing, math homework, chemistry, physics, biology, history, philosophy, law, taxes, real estate, gardening, pets,
cars, home repair, electronics, gaming, minecraft, anime, movies, music, guitar, writing fiction, poetry, essays, marketing,
startups, product management, customer support emails, social media, seo, photography, drawing, fashion, skincare, sleep,
mental health, productivity, studying, college applications, languages, translation, grammar, trivia, geography, space,
climate, medicine symptoms, nutrition, diy crafts, wedding planning, birthday gifts, linux, docker, git, networking,
cybersecurity, machine learning, statistics, data analysis, spreadsheets, powerpoint, public speaking, negotiation,
interview prep, board games, chess, sports, running, cycling, camping, fishing, cooking for kids, baking, coffee, wine,
religion, ethics, politics explainers, economics, crypto, stocks, insurance, retirement, car buying, moving abroad, visas,
roleplay, jokes, riddles, naming things, business names, slogans, lyrics, screenplays, dnd campaigns, worldbuilding""".replace("\n", " ").split(", ")

STYLES = {
    "context_free": "very short context-free messages of 1-6 words that a user might send with no topic at all, like someone who forgot to say what they want (examples of the SHAPE only: 'is it worth it', 'how much does it cost', 'why isn't this working', 'can you explain that', 'what about the second one')",
    "vague_on_topic": "short underspecified requests (3-12 words) that only hint at the topic of {topic}, with the actual need left unclear",
    "casual_typos": "casual lowercase messages with typos, slang and no punctuation about {topic}, 5-20 words",
    "bare_commands": "bare one or two word commands or fragments like 'translate', 'summarize this', 'rewrite', 'continue', 'more', 'shorter', 'explain like im 5'",
    "instruction_plus_paste": "a short instruction followed by a few lines of pasted content related to {topic} (a snippet of text, code, a list, a table, an email); 2-8 lines total",
    "paste_only": "a few lines of pasted content related to {topic} with no instruction at all (an email, a paragraph, a code snippet, a list of items)",
    "personal_short": "short personal or emotional messages (4-15 words) related to {topic}, as someone starting to talk about a problem",
    "short_questions": "short direct questions (4-12 words) about {topic}",
}
WEIGHTS = {"context_free": 2, "vague_on_topic": 3, "casual_typos": 3, "bare_commands": 1, "instruction_plus_paste": 3,
           "paste_only": 2, "personal_short": 2, "short_questions": 2}


async def one(client, sem, model, rng, call_id):
    style = rng.choices(list(WEIGHTS), weights=list(WEIGHTS.values()))[0]
    topic = rng.choice(TOPICS)
    spec = STYLES[style].format(topic=topic)
    prompt = (f"Write 12 different realistic first messages a person might send to an AI chat assistant: {spec}. "
              "Make them varied in wording and intent. Output ONLY a JSON array of 12 strings. /no_think")
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 1.0, "top_p": 0.95, "max_tokens": 1500,
            "chat_template_kwargs": {"enable_thinking": False}}
    async with sem:
        try:
            r = await client.post(f"{BASE_URL}/chat/completions", json=body, headers=HEADERS, timeout=180)
            text = r.json()["choices"][0]["message"]["content"]
        except (httpx.HTTPError, KeyError, ValueError):
            return []
    m = re.search(r"\[.*\]", text, re.S)
    try:
        items = json.loads(m.group(0)) if m else []
    except ValueError:
        return []
    return [{"id": f"syn2_{call_id}_{k}", "source": "synthetic2", "style": style, "prompt": s.strip()}
            for k, s in enumerate(items) if isinstance(s, str) and 1 <= len(s.strip()) <= 1500]


async def main_async(args) -> None:
    rng = random.Random(args.seed)
    sem = asyncio.Semaphore(args.concurrency)
    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(*(one(client, sem, args.model, random.Random(rng.random()), i) for i in range(args.calls)))
    rows = [r for batch in results for r in batch]
    Path(args.output).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"{len(rows):,} prompts from {args.calls} calls -> {args.output}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--calls", type=int, default=3600)
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--output", required=True)
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
