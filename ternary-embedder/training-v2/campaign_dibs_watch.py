"""Track already-running remote work and its pending local quality evaluations."""
import json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
profiles=('batch1024','nli-aux','hardneg')
while True:
 states=[json.loads((ROOT/'work'/(p+'-run')/'pod.json').read_text()) for p in profiles]
 closed=all(s.get('deletion_verified_http_status')==404 for s in states)
 if closed:
  print('All owned comparison pods deleted; releasing remote/network coordination.',flush=True);break
 time.sleep(15)
