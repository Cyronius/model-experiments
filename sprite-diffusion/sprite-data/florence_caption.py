# Bake-off entrant: Florence-2 zero-shot <CAPTION> on a bake-off set (greedy, batched).
# Florence cannot answer the label fields; the scorer maps its caption to a category and
# a subject by text similarity. Run with C:\dev\igpu-venv (transformers 5.x has Florence2).
#   florence_caption.py <eval|tsr> [--model florence-community/Florence-2-large] [--bs 16]
import argparse, json, os, sys, time, torch
from PIL import Image
from transformers import AutoProcessor, Florence2ForConditionalGeneration
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spritekit import DATA

ap = argparse.ArgumentParser()
ap.add_argument('set'); ap.add_argument('--model', default='florence-community/Florence-2-large')
ap.add_argument('--bs', type=int, default=16)
args = ap.parse_args()
B = f'{DATA}/bakeoff'
recs = [json.loads(l) for l in open(f'{B}/{"corpus" if args.set == "eval" else "tsr"}.jsonl')]
if args.set == 'eval':
    recs = [r for r in recs if r['eval']]
proc = AutoProcessor.from_pretrained(args.model)
model = Florence2ForConditionalGeneration.from_pretrained(args.model, torch_dtype=torch.float16).to('cuda').eval()
out = []
t0 = time.time()
for i in range(0, len(recs), args.bs):
    batch = recs[i:i + args.bs]
    ims = [Image.open(f"{B}/render/{r['id']}.png").convert('RGB') for r in batch]
    x = proc(text=['<CAPTION>'] * len(ims), images=ims, return_tensors='pt').to('cuda', torch.float16)
    with torch.no_grad():
        ids = model.generate(**x, max_new_tokens=40, num_beams=1, do_sample=False)
    for r, t in zip(batch, proc.batch_decode(ids, skip_special_tokens=True)):
        out.append(dict(id=r['id'], caption=t.strip()))
dt = time.time() - t0
os.makedirs(f'{B}/out', exist_ok=True)
name = args.model.split('/')[-1].lower()
with open(f'{B}/out/{name}-{args.set}.jsonl', 'w') as fh:
    for o in out:
        o['sec'] = round(dt / len(out), 4)
        fh.write(json.dumps(o) + '\n')
print(f'{len(out)} captions in {dt:.0f}s = {1000 * dt / len(out):.0f} ms/sprite (batch {args.bs})')
