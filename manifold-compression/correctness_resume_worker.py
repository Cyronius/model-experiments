"""Resume on the existing approved pod after the first segment completes."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import shutil
import time

from scale_worker import status,artifacts,ROOT


def main():
    deadline=float(os.environ['SCALE_JOB_DEADLINE'])
    def expired(signum,frame):raise TimeoutError('Extended job deadline reached')
    signal.signal(signal.SIGALRM,expired);signal.alarm(max(1,int(deadline-time.time())))
    prior=Path(os.environ.get('CORRECTNESS_PRIOR',str(ROOT/'results/healing')))
    previous=json.loads((prior/'result.json').read_text())
    if previous.get('stop_reason')=='development regression gate':
        status('complete',continuation_skipped='Respecting the development regression gate');return
    if len(previous['steps'])>=32:status('complete',continuation_skipped='All updates already complete');return
    status('continuation starting')
    destination=ROOT/'results/reproduction-extension';destination.mkdir(exist_ok=True)
    for name in ['resume_correctness.py','correctness_resume_worker.py']:
        shutil.copy2(ROOT/name,destination/name)
    with (ROOT/'results/continuation.log').open('w') as log:
        subprocess.run([sys.executable,'-u',str(ROOT/'resume_correctness.py'),
            '--source',str(ROOT/'models/qwen27'),'--recipe',str(ROOT/'recipe'),
            '--data',str(ROOT/'data.json'),'--legacy-data',str(ROOT/'legacy-data.json'),
            '--prior',str(prior),'--out',str(ROOT/'results/healing-extended'),'--deadline',str(deadline-60)],
            stdout=log,stderr=subprocess.STDOUT,check=True,timeout=max(1,int(deadline-time.time())-30))
    from correctness_summary import summarize
    report=summarize(ROOT/'results/healing-extended')
    artifacts();status('complete',selected_step=report['selected_step'],updates=report['updates_completed'],continuation=True)


if __name__=='__main__':
    try:main()
    except BaseException as error:
        signal.alarm(0);artifacts();status('failed',error=f'{type(error).__name__}: {error}');raise
