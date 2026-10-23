# SigLIP 2 helpers shared by the self-training scripts: the model, chunk-cached image
# embeddings (pause/resume safe), and text embeddings. Run under C:\dev\igpu-venv.
import glob, os, subprocess, sys, numpy as np, torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

MID = os.environ.get('SIGLIP', 'google/siglip2-base-patch16-256')   # so400m-patch14-384: 13x slower
TAG = MID.split('/')[-1]
dev = 'cuda'
# Same prompts as the bake-off (bakeoff_siglip.py), so its accuracy figures carry over.
CAT_PROMPTS = {
    'character': ['a person', 'a human character', 'a hero', 'a villager'],
    'monster': ['a monster', 'a creature', 'an animal', 'an enemy'],
    'item': ['an item', 'a key', 'a coin', 'a piece of armor', 'a tool'],
    'weapon': ['a weapon', 'a sword', 'an axe', 'a bow', 'a spear'],
    'food': ['food', 'a fruit', 'a piece of meat', 'a vegetable', 'a drink', 'a potion', 'a bottle of liquid'],   # food = food and drink, potions included (user, 2026-10-18)
    'terrain': ['a ground tile', 'a floor tile', 'grass', 'water'],
    'structure': ['a wall', 'a door', 'part of a building', 'a fence', 'furniture'],
    'nature': ['a tree', 'a bush', 'a plant', 'a rock'],
    'effect': ['a magic effect', 'an explosion', 'fire', 'a slash effect'],
}
MERGE = {'character': 'creature', 'monster': 'creature'}
_m = {}


def model():
    if not _m:
        # ROCm's SDPA path is ~1.4x slower than eager here (so400m: 245 vs 170 ms/img).
        _m['model'] = AutoModel.from_pretrained(MID, dtype=torch.float16, attn_implementation='eager').to(dev).eval()
        _m['proc'] = AutoProcessor.from_pretrained(MID)
    return _m['model'], _m['proc']


def _feat(o):   # transformers 5 returns an output object; 4.x returned the tensor
    return o if torch.is_tensor(o) else o.pooler_output


def gpu_wanted():
    # Another session waits on the gpu lock we hold: exit 75 after a saved chunk and let the wrapper requeue.
    try:
        out = subprocess.run([os.environ.get('DIBS_EXE', 'dibs'), 'status'], capture_output=True, text=True, timeout=60).stdout
    except Exception:
        return False
    return any(l.startswith('gpu') and 'waiting:' in l for l in out.splitlines())


def embed_cached(ids, paths, cdir, chunk=512):
    """Normalised image embeddings for ids. Each finished chunk of 512 is saved to cdir, so
    stopping and re-running resumes; ids already cached are not recomputed."""
    os.makedirs(cdir, exist_ok=True)
    have = {}
    for f in sorted(glob.glob(f'{cdir}/*.npz')):
        z = np.load(f); have.update(zip(z['ids'].tolist(), z['E']))
    todo = [(i, p) for i, p in zip(ids, paths) if i not in have]
    m, proc = model() if todo else (None, None)
    for k in range(0, len(todo), chunk):
        part = todo[k:k + chunk]; out = []
        for j in range(0, len(part), 32):
            x = proc(images=[Image.open(p).convert('RGB') for _, p in part[j:j + 32]], return_tensors='pt').to(dev)
            with torch.no_grad():
                out.append(torch.nn.functional.normalize(_feat(m.get_image_features(
                    pixel_values=x['pixel_values'].half())).float(), dim=-1).cpu().numpy())
        E = np.concatenate(out)
        np.savez(f'{cdir}/{len(glob.glob(cdir + "/*.npz")):05d}.npz', ids=np.array([i for i, _ in part]), E=E)
        have.update(zip([i for i, _ in part], E))
        print(f'  {os.path.basename(cdir)}: {len(have)}/{len(ids)} embedded', flush=True)
        if os.environ.get('YIELD_GPU') and gpu_wanted():
            print('yielding the gpu', flush=True)
            sys.exit(75)
    return np.stack([have[i] for i in ids])


def embed_text(texts):
    m, proc = model()
    x = proc(text=texts, padding='max_length', max_length=64, truncation=True, return_tensors='pt').to(dev)
    with torch.no_grad():
        return torch.nn.functional.normalize(_feat(m.get_text_features(**x)).float(), dim=-1).cpu().numpy()
