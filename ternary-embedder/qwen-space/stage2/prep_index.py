"""Stage 2 data step 1: an index over every Jaswanth row (file, row, content tokens,
validation flag, text sha256). Saved to training-v2/work/qwen-space/jaswanth-index.npz."""
import glob, hashlib, re, sys
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import WORK, SHARED_WORK
import shared


def key(p):
    m = re.search(r'train_part_([\d_]+)-(\d+)-of', p)
    return tuple(int(x) for x in m[1].split('_')), int(m[2])


FILES = sorted(glob.glob(str(SHARED_WORK / 'jaswanth/data/*.parquet')), key=key)


def main():
    tk = shared._tokenizer()
    fi, ri, nt, val, sha = [], [], [], [], []
    for i, f in enumerate(FILES):
        texts = pq.read_table(f, columns=['text']).column('text').to_pylist()
        enc = tk.encode_batch(texts, add_special_tokens=False)
        for r, (t, e) in enumerate(zip(texts, enc)):
            fi.append(i); ri.append(r); nt.append(len(e.ids))
            h = hashlib.sha256(t.encode('utf-8'))
            val.append(int(h.hexdigest()[:8], 16) % 128 == 0)
            sha.append(h.digest())
        print(i, len(fi), flush=True)
    np.savez(WORK / 'jaswanth-index.npz', fileidx=np.array(fi, np.int16), row=np.array(ri, np.int32),
             ntok=np.array(nt, np.int32), val=np.array(val), sha=np.frombuffer(b''.join(sha), 'u1').reshape(-1, 32),
             files=np.array([Path(f).name for f in FILES]))


if __name__ == '__main__':
    main()
