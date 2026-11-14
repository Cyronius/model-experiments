"""Append/replace an item in the shared MANIFEST.json under an flock (Plan A helper)."""
import fcntl, json, sys, time
from pathlib import Path
SHARED = Path(__file__).resolve().parent.parent / 'training-v2/work/shared-qwen8b'
def record(item, entry):
    SHARED.mkdir(parents=True, exist_ok=True)
    path = SHARED / 'MANIFEST.json'
    with open(SHARED / '.manifest.lock', 'w') as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        m = json.loads(path.read_text()) if path.exists() else {'items': {}}
        m['items'][item] = dict(entry, item=item, recorded_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
        tmp = path.with_suffix('.tmp'); tmp.write_text(json.dumps(m, indent=2) + '\n'); tmp.replace(path)
def done(item, folder=None):
    f = SHARED / (folder or item); f.mkdir(parents=True, exist_ok=True)
    (f / 'DONE').write_text(time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()) + ' Plan A\n')
