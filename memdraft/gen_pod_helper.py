"""A second generator beside gen_pod_worker.py's, on the same pod and GPU.

One llama-server keeps one CPU thread busy and leaves the A40 about 40% idle, whatever the number
of answers at a time (measured 2026-11-11: 501, 550 and 567 tokens/s with 32, 64 and 128). A second
server on the same GPU brought the two to 791 tokens/s together.

The worker goes through the pieces of the prompt file from the first one up. This takes them from
the last one down, with its own server, and stops when the worker has reached the piece it is on
(or is not in its main run). The worker finds the finished pieces done and passes over them. In the
piece where the two meet, a few prompts can get two answers: whoever reads the rows keeps one.

On the pod, from /workspace/manifold/memdraft, with the worker's deadline (MANIFOLD_JOB_DEADLINE):
    nohup ../env/bin/python -u gen_pod_helper.py DEADLINE >> ../outputs/helper.log 2>&1 < /dev/null &
"""
import json
import re
import subprocess
import sys
import time

import gen_pod_worker as w

PARALLEL, PORT = 32, 8456


def worker_at(first):
    """The line of the prompt file where the worker's piece starts; None when it is not in its main run."""
    try:
        stage = json.loads((w.OUT / 'status.json').read_text())['stage']
    except (OSError, ValueError, KeyError):
        return None
    m = re.match(r'rows (\d+)-\d+ of ', stage)
    return int(m[1]) - first if m else None


def help_out(job, count, piece, deadline):
    for a in reversed(range(0, count, piece)):
        b = min(count, a + piece)
        at = worker_at(job['first'])
        if at is None or at >= a:
            print(time.strftime('%T'), f'lines {a}..{b}: the worker is at {at}, stopping', flush=True)
            return
        limit = deadline - w.STOP - time.time()
        if limit < 120:
            print(time.strftime('%T'), 'no time left', flush=True)
            return
        out = w.OUT / 'rows' / f"rows-{job['first'] + a:06d}-{job['first'] + b:06d}.jsonl"
        command = [sys.executable, '-u', 'gen_code.py', '--server', str(job['server']), '--model', str(job['model']),
                   '--prompts', str(job['prompts']), '--skip', str(a), '--count', str(b), '--parallel', str(PARALLEL),
                   '--max-tokens', str(job['max_tokens']), '--ctx', str(job['ctx']), '--out', str(out),
                   '--time-limit', str(int(limit)), '--port', str(job.get('port', PORT)), '--'] + job['server_args']
        print(time.strftime('%T'), f'lines {a}..{b}', flush=True)
        process = subprocess.Popen(command, cwd=w.ROOT, env=job['env'])
        while process.poll() is None:
            time.sleep(5)
            at = worker_at(job['first'])
            if at is None or at >= a:
                print(time.strftime('%T'), f'lines {a}..{b}: the worker is at {at}, stopping', flush=True)
                process.terminate()          # gen_code.py stops its server and drops the answers it has open;
                # the server would take a minute to go, writing answers the worker also writes: end it now
                subprocess.run(['pkill', '-KILL', '-f', f"llama-server -m .* --port {job.get('port', PORT)} "])
                process.wait()
                return
        rows = w.Rows(out).read()
        print(time.strftime('%T'), f'lines {a}..{b}: exit {process.returncode}, {rows.n} answers in the file', flush=True)
        if process.returncode != 0 or rows.n < 0.98 * (b - a):
            return                           # leave the rest to the worker rather than loop on a failure


if __name__ == '__main__':
    import os
    manifest = json.loads((w.REPO / 'bundle-manifest.json').read_text())
    settings = manifest['settings']
    env = dict(os.environ, PATH=f"{w.CUDA}/bin:{os.environ.get('PATH', '/usr/local/bin:/usr/bin:/bin')}",
               LD_LIBRARY_PATH=f'{w.CUDA}/lib64')
    help_out({'first': manifest['prompts']['first'], 'server': w.WORK / 'build/bin/llama-server',
              'model': w.WORK / 'model' / manifest['model']['file'], 'prompts': w.WORK / 'prompts.jsonl',
              'max_tokens': settings['max_tokens'], 'ctx': settings['ctx'], 'env': env,
              'server_args': ['-ngl', '99', '-t', '4']},
             manifest['prompts']['count'], settings['piece'], float(sys.argv[1]))
