"""Losslessly factor paired teacher caches to avoid transferring duplicate passages."""
import hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
work=ROOT/'work'
a=np.load(work/'refined-prompted.npy',mmap_mode='r');b=np.load(work/'refined-plain.npy',mmap_mode='r')
assert a.shape==b.shape
for start in range(0,len(a),256):
 assert np.array_equal(a[start:start+256,1:],b[start:start+256,1:]),'Passage targets differ'
out=work/'teacher-export';out.mkdir(exist_ok=True)
for name,values in [('passages.npy',a[:,1:]),('prompted-queries.npy',a[:,0]),('plain-queries.npy',b[:,0])]:
 np.save(out/name,values)
files=[out/name for name in ('passages.npy','prompted-queries.npy','plain-queries.npy')]
checks={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
meta={'shape':list(a.shape),'passage_vectors_identical':True,'lossless_float32':True,'sha256':checks,
      'prompted_metadata':json.loads((work/'refined-prompted.json').read_text()),
      'plain_metadata':json.loads((work/'refined-plain.json').read_text())}
(out/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
print('Exported',sum(p.stat().st_size for p in files),'bytes; shared passages verified exact.',flush=True)
