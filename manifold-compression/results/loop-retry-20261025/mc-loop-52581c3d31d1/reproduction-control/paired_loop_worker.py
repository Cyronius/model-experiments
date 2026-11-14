"""Wait for the initial diagnostic, then run its same-pod original-limit control."""
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from correctness_worker import ROOT,status,artifacts


def main():
    deadline=float(os.environ['SCALE_JOB_DEADLINE'])
    def expired(signum,frame):raise TimeoutError('Original approved deadline reached')
    signal.signal(signal.SIGALRM,expired);signal.alarm(max(1,int(deadline-time.time())))
    while True:
        p=ROOT/'results/status.json'
        if p.exists():
            state=json.loads(p.read_text())
            if state['stage']=='failed':raise RuntimeError('Initial diagnostic failed')
            if state['stage']=='complete':break
        time.sleep(1)
    status('same-pod original-limit control')
    dest=ROOT/'results/reproduction-control';dest.mkdir()
    for name in ['paired_loop_worker.py','paired_loop_control.py']:shutil.copy2(ROOT/name,dest/name)
    with (ROOT/'results/control.log').open('w') as log:
        subprocess.run([sys.executable,'-u',str(ROOT/'paired_loop_control.py'),
            '--source',str(ROOT/'models/qwen27'),'--recipe',str(ROOT/'recipe'),
            '--data',str(ROOT/'data.json'),'--request',str(ROOT/'loop-retry.json'),
            '--adapter',str(ROOT/'adapter'),'--extended',str(ROOT/'results/loop-retry'),
            '--out',str(ROOT/'results/original-limit-control'),'--deadline',str(deadline-60)],
            stdout=log,stderr=subprocess.STDOUT,check=True,timeout=max(1,int(deadline-time.time())-30))
    artifacts();status('complete',paired_control=True)


if __name__=='__main__':
    try:main()
    except BaseException as error:
        signal.alarm(0);artifacts();status('failed',paired_control=True,error=f'{type(error).__name__}: {error}');raise
