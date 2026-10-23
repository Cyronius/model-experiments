# Crawls TSR for the self-training set: a seeded random sample of games per platform, and
# for each game up to --max-sheets sprite sheets that carry a category seed (round-robin
# across seeds so creatures don't crowd out items, weapons, food and effects).
# Pause: stop the process. Resume: run the same command. The game order is fixed by --seed
# and every request is cached, so a rerun replays from disk and carries on.
# Writes DATA/selftrain/sheets.jsonl (one line per downloaded sheet).
import argparse, json, os, random, re, sys
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spritekit import DATA
import tsrkit

ap = argparse.ArgumentParser()
ap.add_argument('--quota', default='snes:35,nes:25,game_boy_advance:20,sega_genesis:10,game_boy_gbc:10')
ap.add_argument('--max-sheets', type=int, default=40)
ap.add_argument('--min-sheets', type=int, default=8, help='skip games with fewer seeded sheets')
ap.add_argument('--max-creature', type=int, default=20, help='cap on creature sheets per game')
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--unseeded', action='store_true',
                help='also take sheets no rule labels (unlabelled volume); seeded sheets are picked first')
args = ap.parse_args()

OUT = f'{DATA}/selftrain'
os.makedirs(OUT, exist_ok=True)
man = f'{OUT}/sheets.jsonl'
have = {json.loads(l)['asset'] for l in open(man)} if os.path.exists(man) else set()
SKIP_GAME = re.compile(r'\((prototype|hack|beta|demo|unreleased|bootleg)\)|\bunpublished\b|\bhack\b', re.I)
TRAIN_SEEDS = {'creature', 'item', 'weapon', 'food', 'effect'}

with open(man, 'a') as fh:
    for spec in args.quota.split(','):
        plat, quota = spec.split(':'); quota = int(quota)
        games = sorted(g for g in tsrkit.platform_games(plat) if not SKIP_GAME.search(g[0]))
        random.Random(f'{args.seed}:{plat}').shuffle(games)
        accepted = 0
        for title, gp in games:
            if accepted >= quota:
                break
            by_seed, unseeded = defaultdict(list), []
            for section, stitle, asset in tsrkit.sheets(gp):
                cat, role = tsrkit.seed(stitle, section)
                if cat in TRAIN_SEEDS:
                    by_seed[cat].append((section, stitle, asset, cat, role))
                elif cat is None and args.unseeded:
                    unseeded.append((section, stitle, asset, None, None))
            n = sum(len(v) for v in by_seed.values()) + len(unseeded)
            if n < args.min_sheets:
                continue
            accepted += 1
            queues = [list(v)[:args.max_creature] if k == 'creature' else list(v) for k, v in by_seed.items()]
            picked = []
            while len(picked) < args.max_sheets and any(queues):
                for q in queues:
                    if q and len(picked) < args.max_sheets:
                        picked.append(q.pop(0))
            picked += unseeded[:args.max_sheets - len(picked)]
            got = 0
            for section, stitle, asset, cat, role in picked:
                if asset in have:
                    got += 1
                    continue
                if tsrkit.sheet_png(asset) is None:
                    continue
                fh.write(json.dumps(dict(platform=plat, game=title, game_path=gp, section=section, title=stitle,
                                         asset=asset, seed=cat, role=role)) + '\n'); fh.flush()
                have.add(asset); got += 1
            print(f'{plat:17s} {accepted:3d}/{quota} {title[:45]:45s} seeded {n:3d} took {got:2d} '
                  f'{dict((k, len(v)) for k, v in by_seed.items())} unseeded {len(unseeded)}', flush=True)
print('crawl complete')
