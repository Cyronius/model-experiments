"""Non-pickle cache for exact benchmark exclusion identities, keyed by provenance."""
import hashlib
import json
import numpy as np
from broad_data_fetch import atomic,digest


def key_for(identity):
    return hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()


def load_cache(path,identity):
    meta=path.with_suffix('.json')
    if not path.exists() or not meta.exists():return None
    record=json.loads(meta.read_text())
    if record['identity_key']!=key_for(identity):return None
    if digest(path)!=record['sha256']:raise RuntimeError('Reference exclusion cache checksum mismatch')
    with np.load(path,allow_pickle=False) as data:
        assert data['visible'].dtype==np.dtype('V16') and data['lexical'].dtype==np.dtype('V16') and data['raw'].dtype==np.dtype('S64')
        return ({v.tobytes() for v in data['visible']},{bytes(v).decode('ascii') for v in data['raw']},{v.tobytes() for v in data['lexical']})


def save_cache(path,identity,visible,raw,lexical):
    assert all(len(v)==16 for v in visible|lexical) and all(len(v)==64 for v in raw)
    temporary=path.with_suffix('.tmp.npz')
    np.savez(temporary,visible=np.frombuffer(b''.join(sorted(visible)),dtype='V16'),
             raw=np.asarray(sorted(raw),dtype='S64'),lexical=np.frombuffer(b''.join(sorted(lexical)),dtype='V16'))
    temporary.replace(path)
    atomic(path.with_suffix('.json'),dict(identity_key=key_for(identity),identity=identity,sha256=digest(path),counts=dict(visible=len(visible),raw=len(raw),lexical=len(lexical))))
