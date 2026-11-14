"""Stage 0 step 4: verify the Jaswanth download and compare its first rows with RyanMarcus."""
import glob, hashlib, json, re, sys
from pathlib import Path
import numpy as np, pyarrow.parquet as pq
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import SHARED_WORK, dump, sha256
from manifest import record, done
J = SHARED_WORK / 'jaswanth'
def key(p):
    m = re.search(r'train_part_([\d_]+)-(\d+)-of', p)
    return tuple(int(x) for x in m[1].split('_')), int(m[2])
files = sorted(glob.glob(str(J / 'data/*.parquet')), key=key)
rows, hashes = 0, {}
for f in files:
    rows += pq.ParquetFile(f).metadata.num_rows
    hashes[Path(f).name] = sha256(f)
t = pq.read_table(files[0]).slice(0, 2000)
codes = np.array(t.column('embedding').to_pylist(), dtype=np.float32)
amax = np.array(t.column('abs_max_val').to_pylist(), dtype=np.float32)
jt = t.column('text').to_pylist()
assert codes.shape[1] == 4096, codes.shape
print('code range', codes.min(), codes.max())
v = codes * amax[:, None] / 127
raw_norm = np.linalg.norm(v, axis=1)
v /= raw_norm[:, None]
from huggingface_hub import HfFileSystem
fs = HfFileSystem()
REV = 'e39015f788e59a66a03538d3945bac57f976bd61'
rp = pq.ParquetFile(fs.open(f'datasets/RyanMarcus/qwen-8B-embedding-distill@{REV}/qwen-8b-distill-data.parquet', 'rb', block_size=1 << 22))
rg = rp.read_row_group(0)
rt = rg.column('text').to_pylist()[:2000]
re_ = np.stack(rg.column('embedding').to_numpy(zero_copy_only=False)[:2000]).astype(np.float32)
same = [a == b for a, b in zip(jt, rt)]
cos = (v * re_ / np.linalg.norm(re_, axis=1, keepdims=True)).sum(1)
res = {'source': 'Jaswanth-0821/qwen3_embedding_8b_distill', 'revision': (SHARED_WORK / 'jaswanth.revision').read_text().strip(),
       'files': len(files), 'rows': rows, 'dims': int(codes.shape[1]),
       'row_order': 'files sorted by (part, shard) from the name train_part_{part}-{shard}-of-00002.parquet; global row id = running index in that order',
       'dequant_raw_norm_first2000': {'min': float(raw_norm.min()), 'median': float(np.median(raw_norm)), 'max': float(raw_norm.max())},
       'ryanmarcus_revision': REV,
       'first2000_text_identical_to_ryanmarcus_rows_0_1999': int(sum(same)),
       'first2000_cos_vs_ryanmarcus': {'min': float(cos.min()), 'p5': float(np.percentile(cos, 5)), 'median': float(np.median(cos))}}
print(json.dumps(res, indent=2))
dump(Path(__file__).parent / 'jaswanth-verify.json', res)
record('jaswanth', dict(res, path='training-v2/work/shared-qwen8b/jaswanth/', built_by='Plan A', file_sha256=hashes))
done('jaswanth')
