"""Synthesize hard-bucket prompts the wild data under-represents: SQL statements
(title-a-query use case), log/stack-trace pastes, and vague messages.
Programmatic templates with randomized vocab pools; the teacher titles them later.

    python scripts/06_synthesize_hard.py --output data/06_synth_prompts.jsonl --sql 8000 --logs 8000 --vague 4000
"""

import argparse
import json
import random
from pathlib import Path

# ---------------------------------------------------------------- vocab pools
DOMAINS = {
    "commerce": (["orders", "customers", "products", "order_items", "payments", "refunds", "carts", "inventory"],
                 ["total", "status", "created_at", "price", "quantity", "sku", "email", "region", "discount"]),
    "saas": (["users", "accounts", "subscriptions", "invoices", "events", "features", "plans", "sessions"],
             ["plan_tier", "mrr", "churned_at", "trial_ends", "seats", "usage", "login_at", "role"]),
    "health": (["patients", "appointments", "providers", "claims", "prescriptions", "visits"],
               ["dob", "diagnosis_code", "scheduled_at", "npi", "amount", "status", "refills"]),
    "logistics": (["shipments", "warehouses", "drivers", "routes", "packages", "stops"],
                  ["eta", "weight_kg", "origin", "destination", "delivered_at", "carrier", "zone"]),
    "media": (["videos", "channels", "views", "comments", "playlists", "creators"],
              ["duration", "watch_time", "published_at", "likes", "category", "monetized"]),
    "hr": (["employees", "departments", "salaries", "reviews", "time_off", "candidates"],
           ["hire_date", "manager_id", "base_salary", "rating", "pto_days", "stage"]),
}
AGGS = ["COUNT(*)", "SUM({c})", "AVG({c})", "MAX({c})", "MIN({c})", "COUNT(DISTINCT {c})"]
INTERVALS = ["7 days", "30 days", "90 days", "1 year", "24 hours", "6 months"]

SQL_ASKS = [
    "", "", "",  # often the SQL arrives bare
    "what does this query do\n\n", "explain this:\n", "optimize this query:\n",
    "why is this slow?\n", "is there anything wrong with this\n", "rewrite this to be more readable\n",
    "I ran this and got unexpected results:\n", "convert this to use a CTE\n",
]

LOG_ASKS = ["", "", "", "what does this mean\n\n", "getting this error:\n", "help\n\n",
            "this just started happening in production\n", "how do I fix this\n"]

VAGUE_POOL = [
    "hey", "yo", "hello?", "u there", "quick question", "need some help",
    "can you do something for me", "i need advice", "got a sec", "help me out",
    "this isn't working", "it broke again", "same problem as before", "why though",
    "what do you think", "is that good or bad", "should I do it", "does that make sense",
    "idk what to do", "im stuck", "make it better", "try again", "not that one",
    "the other thing", "like I said before", "you know what I mean", "do your thing",
    "surprise me", "whatever works", "just fix it", "hmm", "ok but why", "wait what",
    "that's not what I asked", "can you be more specific", "never mind", "one more thing",
    "before I forget", "real quick", "so about that", "any ideas", "thoughts?",
    "what would you do", "is this a bad idea", "tell me something interesting",
    "i'm bored", "feeling stuck on my project", "big decision to make", "wish me luck",
]
VAGUE_SUFFIX = ["", "", " please", " thanks", " asap", " lol", "...", "??", " 🙏", " real quick"]

# added for titler v1.4 -- the wild data and the original vague bucket had zero coverage of
# hostile/negative reactions or gratitude/closing messages, both of which the v1.3 hard-case
# suite showed collapsing to the same generic "User Asks for Assistance"-style title regardless
# of actual sentiment. see .claude/plans/titler-v1.4-collapse-fix-dual-arch.md
HOSTILE_POOL = [
    "you suck", "this is useless", "worst answer ever", "that's wrong", "not helpful at all",
    "you're not listening", "this is garbage", "totally useless", "you got it wrong again",
    "terrible answer", "you don't understand", "this makes no sense", "you're useless",
    "absolutely not what I asked", "way off base", "nope wrong again", "you missed the point",
    "this is a joke right", "unbelievably bad", "you're kidding me", "come on that's not right",
    "this is so frustrating", "i give up", "waste of time", "you never get it right",
]

GRATITUDE_POOL = [
    "thanks that worked!", "perfect, thanks", "got it thanks", "you're the best", "that fixed it",
    "nice one thanks", "exactly what I needed", "thank you so much", "that did it",
    "appreciate it", "awesome thanks", "you're a lifesaver", "worked like a charm",
    "much appreciated", "great, thanks", "perfect solution thanks", "that's exactly right",
    "thanks a ton", "cheers that helped", "brilliant thank you", "yes! that's it", "nailed it thanks",
    "amazing, thank you", "you fixed it thanks", "finally, thanks",
]


def make_sql(rng: random.Random) -> str:
    domain = rng.choice(list(DOMAINS))
    tables, cols = DOMAINS[domain]
    t1, t2 = rng.sample(tables, 2)
    c1, c2 = rng.sample(cols, 2)
    agg = rng.choice(AGGS).format(c=c1)
    style = rng.random()
    if style < 0.25:  # aggregate report
        q = (f"SELECT {t1}.{c2}, {agg} AS metric\nFROM {t1}\n"
             f"JOIN {t2} ON {t2}.{t1[:-1]}_id = {t1}.id\n"
             f"WHERE {t1}.{c1} IS NOT NULL\nGROUP BY {t1}.{c2}\n"
             f"ORDER BY metric DESC\nLIMIT {rng.choice([5, 10, 20, 100])};")
    elif style < 0.45:  # time-window filter
        q = (f"SELECT * FROM {t1}\nWHERE {rng.choice(['created_at', c1])} >= NOW() - INTERVAL '{rng.choice(INTERVALS)}'\n"
             f"  AND {c2} = '{rng.choice(['active', 'pending', 'failed', 'completed'])}'\n"
             f"ORDER BY created_at DESC;")
    elif style < 0.6:  # CTE
        q = (f"WITH recent AS (\n  SELECT {t1[:-1]}_id, {agg} AS n\n  FROM {t2}\n"
             f"  WHERE created_at > CURRENT_DATE - {rng.choice([7, 30, 90])}\n  GROUP BY {t1[:-1]}_id\n)\n"
             f"SELECT {t1}.*, recent.n FROM {t1}\nJOIN recent ON recent.{t1[:-1]}_id = {t1}.id\n"
             f"WHERE recent.n > {rng.choice([1, 5, 10])};")
    elif style < 0.72:  # update
        q = (f"UPDATE {t1}\nSET {c1} = {rng.choice(['NULL', '0', f'{c1} * 1.05', chr(39) + 'archived' + chr(39)])}\n"
             f"WHERE {c2} < '{rng.choice(['2024-01-01', '2025-06-30'])}'\n"
             f"  AND id IN (SELECT {t1[:-1]}_id FROM {t2} WHERE {rng.choice(cols)} IS NULL);")
    elif style < 0.82:  # delete / cleanup
        q = (f"DELETE FROM {t2}\nWHERE {t1[:-1]}_id NOT IN (SELECT id FROM {t1})\n"
             f"  OR created_at < NOW() - INTERVAL '{rng.choice(INTERVALS)}';")
    elif style < 0.92:  # ddl
        q = (f"CREATE TABLE {t1}_audit (\n  id BIGSERIAL PRIMARY KEY,\n  {t1[:-1]}_id BIGINT REFERENCES {t1}(id),\n"
             f"  {c1} TEXT,\n  changed_by TEXT,\n  changed_at TIMESTAMPTZ DEFAULT NOW()\n);\n"
             f"CREATE INDEX idx_{t1}_audit_{c1} ON {t1}_audit({c1});")
    else:  # window function
        q = (f"SELECT id, {c1},\n  ROW_NUMBER() OVER (PARTITION BY {c2} ORDER BY {c1} DESC) AS rn\n"
             f"FROM {t1}\nQUALIFY rn <= {rng.choice([1, 3, 5])};")
    return rng.choice(SQL_ASKS) + q


ERR_TEMPLATES = [
    lambda r, m: (f"Traceback (most recent call last):\n  File \"{m}.py\", line {r.randint(5, 400)}, in {r.choice(['main', 'run', 'handler', 'process', '<module>'])}\n"
                  f"    {r.choice(['result = data[key]', 'conn = pool.acquire()', 'obj.save()', 'value = int(raw)', 'resp.raise_for_status()'])}\n"
                  + r.choice([f"KeyError: '{r.choice(['user_id', 'config', 'token', 'items'])}'",
                              "ValueError: invalid literal for int() with base 10: 'N/A'",
                              f"AttributeError: 'NoneType' object has no attribute '{r.choice(['save', 'get', 'id', 'split'])}'",
                              "ConnectionResetError: [Errno 104] Connection reset by peer"])),
    lambda r, m: (f"java.lang.{r.choice(['NullPointerException', 'IllegalStateException', 'OutOfMemoryError: Java heap space', 'ClassCastException'])}\n"
                  f"\tat com.{m}.service.{m.capitalize()}Service.{r.choice(['process', 'execute', 'load', 'sync'])}({m.capitalize()}Service.java:{r.randint(20, 900)})\n"
                  f"\tat com.{m}.controller.{m.capitalize()}Controller.handle({m.capitalize()}Controller.java:{r.randint(20, 200)})"),
    lambda r, m: (f"{r.choice(['2026-07-14T02:11:09Z', '2026-08-02 14:33:51', '2026-06-30T23:59:01Z'])} ERROR [{m}-worker-{r.randint(1, 16)}] "
                  + r.choice(["Timeout waiting for connection from pool after 30000ms",
                              f"Failed to publish message to topic '{m}.events': Broker not available",
                              f"Deadlock found when trying to get lock; try restarting transaction",
                              f"OOMKilled: container exceeded memory limit {r.choice(['512Mi', '1Gi', '2Gi'])}"])),
    lambda r, m: (f"npm ERR! code {r.choice(['ERESOLVE', 'ELIFECYCLE', 'ENOENT', 'ETARGET'])}\n"
                  f"npm ERR! {r.choice(['unable to resolve dependency tree', f'missing script: {m}', f'No matching version found for {m}@^{r.randint(2, 9)}.0.0'])}\n"
                  f"npm ERR! A complete log of this run can be found in ~/.npm/_logs/"),
    lambda r, m: (f"Error: cannot find module '{m}'\nRequire stack:\n- /app/src/{r.choice(['index', 'server', 'worker'])}.js\n"
                  f"    at Function.Module._resolveFilename (node:internal/modules/cjs/loader:{r.randint(900, 1200)}:15)"),
    lambda r, m: (f"panic: runtime error: {r.choice(['index out of range [3] with length 2', 'invalid memory address or nil pointer dereference'])}\n"
                  f"goroutine {r.randint(1, 200)} [running]:\nmain.(*{m.capitalize()}).{r.choice(['Start', 'Run', 'Flush'])}(0x0)\n\t/app/{m}.go:{r.randint(10, 300)}"),
    lambda r, m: (f"Warning  {r.choice(['FailedScheduling', 'BackOff', 'Unhealthy', 'FailedMount'])}  "
                  + r.choice([f"0/{r.randint(3, 12)} nodes are available: insufficient {r.choice(['memory', 'cpu'])}.",
                              f"Back-off restarting failed container {m}",
                              f"Liveness probe failed: HTTP probe failed with statuscode: {r.choice([500, 502, 503])}",
                              f"MountVolume.SetUp failed for volume \"{m}-secrets\": secret \"{m}\" not found"])),
]
MODULES = ["auth", "billing", "ingest", "scheduler", "notifier", "gateway", "sync", "report", "cache", "upload", "search", "webhook"]


def make_log(rng: random.Random) -> str:
    return rng.choice(LOG_ASKS) + rng.choice(ERR_TEMPLATES)(rng, rng.choice(MODULES))


MD_ASKS = ["", "", "summarize this\n\n", "clean this up\n\n", "turn this into an email\n\n",
           "review these notes\n\n", "make this more concise\n\n"]
MD_TOPICS = ["migration", "onboarding", "launch", "incident", "roadmap", "redesign", "hiring", "budget", "sprint", "security audit"]
NAMES = ["Alex", "Sam", "Jordan", "Priya", "Chen", "Maria", "Devon", "Kim"]


def make_markdown(rng: random.Random) -> str:
    topic = rng.choice(MD_TOPICS)
    domain = rng.choice(list(DOMAINS))
    tables, cols = DOMAINS[domain]
    style = rng.random()
    if style < 0.35:  # meeting notes
        who = rng.sample(NAMES, 3)
        body = (f"# {topic.title()} Sync — {rng.choice(['Mon', 'Tue', 'Wed', 'Thu'])}\n\n"
                f"Attendees: {', '.join(who)}\n\n## Decisions\n"
                f"- {rng.choice(['Ship behind a feature flag', 'Delay to next quarter', 'Go with option B', 'Hire two more engineers'])}\n"
                f"- {who[0]} owns the {rng.choice(tables)} cleanup\n\n## Action items\n"
                f"- [ ] {who[1]}: draft the {topic} doc\n- [ ] {who[2]}: audit {rng.choice(cols)} usage\n"
                f"- [x] schedule follow-up")
    elif style < 0.6:  # readme / howto
        body = (f"# {topic.replace(' ', '-')}-tool\n\n{rng.choice(['CLI for', 'Service that handles', 'Library for'])} "
                f"{topic} workflows.\n\n## Install\n```\nnpm install {topic.split()[0]}-tool\n```\n\n## Usage\n"
                f"```\n{topic.split()[0]}-tool --{rng.choice(cols)} <value> [--dry-run]\n```\n\n"
                f"## Config\n| key | default |\n|---|---|\n| {rng.choice(cols)} | {rng.choice(['true', '30s', 'null'])} |\n"
                f"| retries | {rng.randint(0, 5)} |")
    elif style < 0.8:  # prd / plan
        body = (f"## {topic.title()} Plan\n\n**Goal:** {rng.choice(['reduce churn', 'cut infra cost 20%', 'improve p95 latency', 'unblock enterprise deals'])}\n\n"
                f"**Scope:**\n1. Migrate {rng.choice(tables)} to the new schema\n2. Backfill {rng.choice(cols)}\n"
                f"3. {rng.choice(['Dashboards', 'Alerts', 'Docs'])}\n\n**Non-goals:** touching {rng.choice(tables)}\n\n"
                f"**Risks:** {rng.choice(['data loss during backfill', 'third-party API limits', 'timeline slips past Q3'])}")
    else:  # changelog
        body = (f"## v{rng.randint(1, 4)}.{rng.randint(0, 9)}.{rng.randint(0, 9)}\n\n### Added\n- {topic} support\n"
                f"- `--{rng.choice(cols)}` flag\n\n### Fixed\n- crash when {rng.choice(tables)} is empty\n"
                f"- {rng.choice(['off-by-one in pagination', 'race on shutdown', 'unicode handling in exports'])}\n\n"
                f"### Deprecated\n- legacy {rng.choice(tables)} endpoint")
    return rng.choice(MD_ASKS) + body


def make_vague(rng: random.Random) -> str:
    base = rng.choice(VAGUE_POOL)
    if rng.random() < 0.3:
        base = base + rng.choice([",", " -", " and"]) + " " + rng.choice(VAGUE_POOL)
    return base + rng.choice(VAGUE_SUFFIX)


def make_hostile(rng: random.Random) -> str:
    base = rng.choice(HOSTILE_POOL)
    if rng.random() < 0.3:
        base = base + rng.choice([",", " -", " and"]) + " " + rng.choice(HOSTILE_POOL)
    return base + rng.choice(VAGUE_SUFFIX)


def make_gratitude(rng: random.Random) -> str:
    base = rng.choice(GRATITUDE_POOL)
    if rng.random() < 0.3:
        base = base + rng.choice([",", " -", " and"]) + " " + rng.choice(GRATITUDE_POOL)
    return base + rng.choice(VAGUE_SUFFIX)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--sql", type=int, default=8000)
    ap.add_argument("--logs", type=int, default=8000)
    ap.add_argument("--vague", type=int, default=4000)
    ap.add_argument("--hostile", type=int, default=1500,
                    help="negative/hostile short reactions -- see titler-v1.4 plan")
    ap.add_argument("--gratitude", type=int, default=1500,
                    help="gratitude/closing short messages -- see titler-v1.4 plan")
    ap.add_argument("--md", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    seen: set[str] = set()
    rows = []
    for bucket, n, fn in (("sql", args.sql, make_sql), ("log_paste", args.logs, make_log),
                          ("vague", args.vague, make_vague), ("markdown", args.md, make_markdown),
                          ("hostile", args.hostile, make_hostile), ("gratitude", args.gratitude, make_gratitude)):
        made = tries = 0
        while made < n and tries < n * 30:
            tries += 1
            p = fn(rng)
            if p in seen:
                continue
            seen.add(p)
            rows.append({"id": f"syn_{bucket}_{made}", "source": "synthetic", "bucket": bucket, "prompt": p})
            made += 1
        print(f"{bucket}: {made} (wanted {n})")

    rng.shuffle(rows)
    with Path(args.output).open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"total {len(rows)} -> {args.output}")


if __name__ == "__main__":
    main()
