"""Download robro612 NQ vectors, keep only the reserved subsample rows (+ all queries), delete the full file."""
import json, sys
from pathlib import Path
import numpy as np
from huggingface_hub import hf_hub_download
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import SHARED_WORK, sha256
REV = '985ee9aebdbb6aa21035b2b93d35f6634d243456'; REPO = 'robro612/nq_qwen3_embedding_8b'
d = SHARED_WORK / 'reserved/nq'; d.mkdir(parents=True, exist_ok=True)
tmp = SHARED_WORK / 'reserved/nq-full-tmp'
for f in ('queries.npy', 'queries_ids.npy', 'doc_ids.npy', 'qrels.test.tsv'):
    hf_hub_download(REPO, f, repo_type='dataset', revision=REV, local_dir=d)
ids = np.load(d / 'doc_ids.npy', allow_pickle=False)
want = (SHARED_WORK / 'reserved/nq-subsample-ids.txt').read_text().split('\n')[:-1]
pos = {str(x): i for i, x in enumerate(ids)}
rows = np.array(sorted(pos[w] for w in want))
full = hf_hub_download(REPO, 'documents.npy', repo_type='dataset', revision=REV, local_dir=tmp)
D = np.load(full, mmap_mode='r')
sub = np.asarray(D[rows])
np.save(d / 'subsample-documents.npy', sub)
np.save(d / 'subsample-doc-ids.npy', ids[rows])
info = {'rows': len(rows), 'source_revision': REV, 'documents_npy_sha256_of_subsample': sha256(d / 'subsample-documents.npy')}
(d / 'subsample.json').write_text(json.dumps(info, indent=2) + '\n')
Path(full).unlink()
import shutil; shutil.rmtree(tmp, ignore_errors=True)
print(json.dumps(info))
