"""Wait for quality completion before requesting the whole-machine measurement."""
import os,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent
while not (ROOT/'delivery/quality-complete.json').exists():time.sleep(5)
env=dict(os.environ,DIBS_AGENT='delivery_latency')
with (ROOT/'delivery/latency-dibs.log').open('w') as log:
 subprocess.run(['dibs','run','--measure','--label','Final embedding family matched CPU Qwen and single-text NPU latency','--eta','5m','--','bash',str(ROOT/'measure_delivery.sh')],cwd=PROJECT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
print('Final matched timing series complete; inspect initial/final CLEAN audit.',flush=True)
