"""Stage 2 arm 5 extension: the data curve was still rising at 300,000, so gather the rest of the
eligible training pool (same filters, never validation) and its native-512 Q8 features.
Outputs in training-v2/work/qwen-space/stage2/: texts_rest.json, targets_rest.npy, F_q8_native_rest.npy."""
import json, sys
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
from common import WORK, features, load_student, Timer
from prep_data import OUT, select
from prep_index import FILES


def main():
    torch.set_num_threads(int(sys.argv[1]) if len(sys.argv) > 1 else 16)
    if not (OUT / 'targets_rest.npy').exists():
        ix, train, val, stats = select()
        prev = np.load(OUT / 'data.npz')
        assert np.array_equal(prev['train'], train), 'selection changed'
        sha_hex = np.array([b.tobytes().hex() for b in ix['sha']])
        from prep_data import leaked_hashes
        leaked = np.isin(sha_hex, list(leaked_hashes()))
        ok = (ix['ntok'] <= 504) & ~leaked & (ix['ntok'] > 0) & ~ix['val']
        rest = np.setdiff1d(np.where(ok)[0], train)
        print('rest', len(rest), flush=True)
        texts = [None] * len(rest)
        tg = np.lib.format.open_memmap(OUT / 'targets_rest.tmp.npy', mode='w+', dtype=np.float16, shape=(len(rest), 4096))
        by_file = {}
        fidx, rowi = np.asarray(ix['fileidx']), np.asarray(ix['row'])
        for k, g in enumerate(rest):
            by_file.setdefault(int(fidx[g]), []).append((int(rowi[g]), k))
        for fi, items in sorted(by_file.items()):
            t = pq.read_table(FILES[fi])
            rr = np.array([r for r, _ in items]); kk = np.array([k for _, k in items])
            sub = t.take(rr)
            emb = sub.column('embedding').combine_chunks()
            codes = emb.flatten().to_numpy().astype(np.float32).reshape(len(rr), 4096)
            amax = sub.column('abs_max_val').to_numpy().astype(np.float32)
            v = codes * amax[:, None] / 127
            v /= np.linalg.norm(v, axis=1, keepdims=True)
            tg[kk] = v.astype(np.float16)
            for k, s in zip(kk, sub.column('text').to_pylist()):
                texts[k] = s
            print('gather', fi, flush=True)
        tg.flush(); del tg
        (OUT / 'texts_rest.json').write_text(json.dumps(texts))
        np.save(OUT / 'rest_ids.npy', rest)
        (OUT / 'targets_rest.tmp.npy').rename(OUT / 'targets_rest.npy')
    p = OUT / 'F_q8_native_rest.npy'
    if not p.exists():
        texts = json.loads((OUT / 'texts_rest.json').read_text())
        m = load_student('q8', native512=True)
        with Timer('features q8 native rest'):
            np.save(p, features(m, texts, 'native'))


if __name__ == '__main__':
    main()
