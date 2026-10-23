# Bake-off entrant: an instruction-following VLM served by `flm serve <tag>` (NPU),
# asked for the label fields as JSON. Usage:
#   bakeoff_vlm.py <model tag> <eval|tsr> [--hints] [--limit N] [--stride N]
# Packs are run without hints (their folder names are the answer key); TSR runs pass the
# game / section / sheet title, as production will. Writes DATA/bakeoff/out/<tag>-<set>.jsonl
# with the parsed fields, the raw reply and the per-request latency.
import argparse, base64, json, os, re, time, requests
from spritekit import DATA

ap = argparse.ArgumentParser()
ap.add_argument('model'); ap.add_argument('set'); ap.add_argument('--hints', action='store_true')
ap.add_argument('--limit', type=int, default=0); ap.add_argument('--stride', type=int, default=1); ap.add_argument('--port', type=int, default=52625)
args = ap.parse_args()

CATS = ['character', 'monster', 'boss', 'item', 'weapon', 'food', 'terrain', 'structure', 'nature', 'effect', 'ui']
PROMPT = """This is a pixel-art game sprite, shown enlarged on a grey background.
Reply with JSON only, no other text:
{"category": "...", "subject": "...", "facing": "...", "pose": "...", "caption": "..."}

category: one of character, monster, boss, item, weapon, food, terrain, structure, nature, effect, ui
  character = a person or humanoid hero/NPC; monster = a creature, animal or enemy;
  item = an object you pick up (potion, key, coin, armour, tool, book); weapon = sword, axe,
  bow, spear, arrow, bomb; food = something edible; terrain = a ground, floor or water tile;
  structure = wall, door, building part, fence, furniture; nature = tree, bush, plant, rock;
  effect = magic, fire, explosion, slash or particles; ui = text, icon or interface element
subject: a short noun phrase naming exactly what it is (e.g. "skeleton", "red potion", "oak tree")
facing: front, back, left, right or none (front = towards the viewer; left = towards the viewer's left)
pose: idle, walk, attack, hurt or other
caption: at most 12 words describing it"""

recs = [json.loads(l) for l in open(f'{DATA}/bakeoff/{"corpus" if args.set == "eval" else "tsr"}.jsonl')]
if args.set == 'eval':
    recs = [r for r in recs if r['eval']]
recs = recs[::args.stride]   # every Nth sprite: a cheap stratified subset (ids are ordered by source)
if args.limit:
    recs = recs[:args.limit]
os.makedirs(f'{DATA}/bakeoff/out', exist_ok=True)
out_path = f'{DATA}/bakeoff/out/{args.model.replace(":", "_")}-{args.set}{"-hints" if args.hints else ""}.jsonl'
done = set()
if os.path.exists(out_path):
    done = {json.loads(l)['id'] for l in open(out_path)}
url = f'http://127.0.0.1:{args.port}/v1/chat/completions'


def parse(txt):
    m = re.search(r'\{.*\}', txt, re.S)
    if not m:
        return {}
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        d = dict(re.findall(r'"(\w+)"\s*:\s*"([^"]*)"', m.group(0)))
    return {k: (str(v).strip().lower() if k != 'caption' else str(v).strip()) for k, v in d.items()}


with open(out_path, 'a') as fh:
    for i, r in enumerate(recs):
        if r['id'] in done:
            continue
        b64 = base64.b64encode(open(f"{DATA}/bakeoff/render/{r['id']}.png", 'rb').read()).decode()
        text = PROMPT
        if args.hints:
            text += f"\n\nContext: from the game \"{r['game']}\", sprite sheet \"{r['title']}\" (section \"{r['section']}\")."
        body = dict(model=args.model, temperature=0, max_tokens=160, stream=False,
                    messages=[{'role': 'user', 'content': [
                        {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{b64}'}},
                        {'type': 'text', 'text': text}]}])
        t0 = time.time()
        reply = requests.post(url, json=body, timeout=600).json()
        dt = time.time() - t0
        txt = reply['choices'][0]['message']['content'] if 'choices' in reply else json.dumps(reply)
        fh.write(json.dumps(dict(id=r['id'], pred=parse(txt), raw=txt, sec=round(dt, 3))) + '\n'); fh.flush()
        if i % 25 == 0:
            print(f'{i}/{len(recs)} {dt:.2f}s {txt[:100]!r}', flush=True)
print('wrote', out_path)
