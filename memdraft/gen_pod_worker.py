"""Disposable CUDA worker for run_gen_pod.py: the 9B (Q4_K_M) answers coding prompts under llama.cpp.
Has no account credential.

Checks the bundle. Downloads the pinned GGUF, the pinned prompt set and the pinned llama.cpp
source, each checked against the hashes in the bundle manifest. Builds llama-server with CUDA.
Rebuilds the prompt order and checks it against the launcher's. Then runs gen_code.py over the
prompts in pieces: outputs/rows/rows-<from>-<to>.jsonl, named by place in the order, written as
the answers finish and copied home by the launcher every half minute.

The first minutes try several numbers of answers at a time and keep the fastest. If all the
prompts would then cost more than the cap, it stops there (stage 'complete', outcome 'stopped').

Last resort against a launcher that has died at home: a detached process deletes this pod, with
the key the provider puts in every pod's own environment, 15 minutes after the launcher's deadline.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tarfile
import time
import urllib.request

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
OUT = REPO / 'outputs'
WORK = Path('/workspace/gen')          # model, prompt set, llama.cpp: not copied home
CUDA = '/usr/local/cuda'
STOP = 420                             # take no new prompt this long before the worker's deadline: the open answers still finish
REST, GRAPHQL = 'https://rest.runpod.io/v1/pods/', 'https://api.runpod.io/graphql'


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            value.update(chunk)
    return value.hexdigest()


def tree_sha(root):
    """One hash for a source tree: every file's path and content hash, in path order."""
    root = Path(root)
    files = sorted(p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and not p.is_symlink())
    value = hashlib.sha256()
    for name in files:
        value.update(f'{name}\0{sha(root / name)}\n'.encode())
    return value.hexdigest()


def ids_sha(prompts):
    return hashlib.sha256('\n'.join(p['id'] for p in prompts).encode()).hexdigest()


def save(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, indent=2) + '\n')
    temp.replace(path)


def status(stage, **info):
    save({'stage': stage, 'updated_at': time.time(), **info}, OUT / 'status.json')
    print(time.strftime('%T'), stage, json.dumps(info), flush=True)


def artifacts():
    save({str(p.relative_to(OUT)): sha(p) for p in sorted(OUT.rglob('*'))
          if p.is_file() and p.suffix not in ('.log', '.tmp') and
          p.name not in ('status.json', 'artifact-manifest.json')}, OUT / 'artifact-manifest.json')


# --- the pod's own key: only ever used on this pod, for this pod -------------------------------

def pod_identity():
    """This pod's id and its own key. They are in the container's environment, which an ssh session does not inherit."""
    try:
        items = Path('/proc/1/environ').read_bytes().decode(errors='replace').split('\0')
    except OSError:
        return None, None
    env = dict(item.split('=', 1) for item in items if '=' in item)
    return env.get('RUNPOD_POD_ID'), env.get('RUNPOD_API_KEY')


def provider(method, url, key, data=None):
    request = urllib.request.Request(url, data=json.dumps(data).encode() if data is not None else None, method=method,
        headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json', 'Accept': 'application/json',
                 'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(request, timeout=20) as response:
        body = response.read()
        return json.loads(body) if body else None


def sees_itself():
    """Which of the provider's interfaces answers this pod's key about this pod: 'rest', 'graphql' or None."""
    pod, key = pod_identity()
    if not pod or not key:
        return None
    try:
        if provider('GET', REST + pod, key)['id'] == pod:
            return 'rest'
    except Exception:
        pass
    try:
        if provider('POST', GRAPHQL, key, {'query': 'query { pod(input: {podId: "%s"}) { id } }' % pod})['data']['pod']['id'] == pod:
            return 'graphql'
    except Exception:
        pass
    return None


def delete_self_at(when):
    """Runs detached. By `when` the launcher and its watchdog should have deleted this pod long ago."""
    print('waiting until', when, flush=True)
    time.sleep(max(0, when - time.time()))
    pod, key = pod_identity()
    if not pod or not key:
        print('no pod id or key in the environment: nothing to do', flush=True)
        return
    for _ in range(120):
        for name, call in (('rest', lambda: provider('DELETE', REST + pod, key)),
                           ('graphql', lambda: provider('POST', GRAPHQL, key,
                                                        {'query': 'mutation { podTerminate(input: {podId: "%s"}) }' % pod}))):
            try:
                print(time.strftime('%T'), name, 'delete:', call(), flush=True)
            except Exception as error:
                print(time.strftime('%T'), name, 'delete failed:', type(error).__name__, flush=True)
        time.sleep(60)


# --- generation --------------------------------------------------------------------------------

class Rows:
    """Counts the answers in a rows file as it grows."""

    def __init__(self, path):
        self.path, self.at, self.n, self.tokens, self.seconds = Path(path), 0, 0, 0, 0.0

    def read(self):
        if self.path.exists():
            with self.path.open('rb') as stream:
                stream.seek(self.at)
                data = stream.read()
            end = data.rfind(b'\n') + 1
            for line in data[:end].splitlines():
                row = json.loads(line)
                self.n += 1
                self.tokens += len(row['answer_ids'])
                self.seconds += row['seconds']
            self.at += end
        return self


def rate(parallel, tokens, seconds):
    """Tokens per second with `parallel` answers written at once, from the answers' own times."""
    return parallel * tokens / seconds if seconds > 0 else None


def generate(job, a, b, parallel, limit, report):
    """gen_code.py over lines a..b of the prompt file, `parallel` answers at a time, taking no new prompt after
    `limit` seconds. Returns its exit code, the answers now in the piece's file, and the tokens per second
    of the answers that finished before the limit (None if there were fewer than `parallel`)."""
    out = OUT / 'rows' / f"rows-{job['first'] + a:06d}-{job['first'] + b:06d}.jsonl"
    out.parent.mkdir(exist_ok=True)
    rows = Rows(out).read()
    had = (rows.n, rows.tokens, rows.seconds)
    command = [sys.executable, '-u', 'gen_code.py', '--server', str(job['server']), '--model', str(job['model']),
               '--prompts', str(job['prompts']), '--skip', str(a), '--count', str(b), '--parallel', str(parallel),
               '--max-tokens', str(job['max_tokens']), '--ctx', str(job['ctx']), '--out', str(out),
               '--time-limit', str(int(limit))] + job.get('gen_args', []) + ['--'] + job['server_args']
    began, told, busy = time.time(), 0, (0, 0, 0.0)      # busy: answers, tokens, seconds of this run before the limit
    with (OUT / 'gen.log').open('a') as stream:
        stream.write(f"--- {time.strftime('%T')} lines {a}..{b}, {parallel} at a time, limit {int(limit)} s\n")
        stream.flush()
        process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, env=job['env'])
        while process.poll() is None:
            time.sleep(5)
            rows.read()
            if time.time() < began + limit:
                busy = (rows.n - had[0], rows.tokens - had[1], rows.seconds - had[2])
            if time.time() - told >= 60:
                told = time.time()
                report(rows.n, rows.tokens, rate(parallel, *busy[1:]))
    rows.read()
    return process.returncode, rows.n, rate(parallel, *busy[1:]) if busy[0] >= parallel else None


def main():
    deadline = float(os.environ['MANIFOLD_JOB_DEADLINE'])       # the launcher's own deadline less 10 minutes

    def expired(signum, frame):
        raise TimeoutError('Authorized worker deadline reached')
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(max(1, int(deadline - time.time())))
    OUT.mkdir(exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((REPO / 'bundle-manifest.json').read_text())
    for relative, expected in manifest['files'].items():
        if sha(REPO / relative) != expected:
            raise ValueError('Bundle checksum mismatch: ' + relative)
        dest = OUT / 'executed-source' / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / relative, dest)
    shutil.copy2(REPO / 'bundle-manifest.json', OUT / 'bundle-manifest.json')
    (OUT / 'gpu.txt').write_text(subprocess.check_output(['nvidia-smi'], text=True))
    with (OUT / 'delete-self.log').open('a') as stream:
        subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--delete-self-at', str(deadline + 600 + 900)],
                         stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    known = {'self_delete': sees_itself()}

    status('downloads', outcome=known)
    from huggingface_hub import hf_hub_download

    def fetch(what, **more):
        spec = manifest[what]
        path = Path(hf_hub_download(spec['repo'], filename=spec['file'], revision=spec['revision'],
                                    local_dir=WORK / what, token=False, **more))
        if sha(path) != spec['sha256']:
            raise ValueError(f'Pinned {what} mismatch')
        return path
    model = fetch('model')
    dataset = fetch('dataset', repo_type='dataset')
    spec = manifest['llama_cpp']
    archive = WORK / 'llama.cpp.tar.gz'
    for attempt in range(3):
        try:
            urllib.request.urlretrieve(spec['url'], archive)
            break
        except OSError:
            if attempt == 2:
                raise
            time.sleep(10)
    with tarfile.open(archive) as tar:
        tar.extractall(WORK / 'src', filter='data')
    source = WORK / 'src' / ('llama.cpp-' + spec['commit'])
    if tree_sha(source) != spec['tree_sha256']:
        raise ValueError('Pinned llama.cpp source mismatch')

    status('build', outcome=known)
    env = dict(os.environ, PATH=f"{CUDA}/bin:{os.environ.get('PATH', '/usr/local/bin:/usr/bin:/bin')}",
               LD_LIBRARY_PATH=f'{CUDA}/lib64', DEBIAN_FRONTEND='noninteractive')
    build = WORK / 'build'
    with (OUT / 'build.log').open('w') as stream:
        subprocess.run(['bash', '-c', 'set -eu; apt-get update -qq; apt-get install -y -qq cmake build-essential; '
                        f'cmake -S {source} -B {build} ' + ' '.join(spec['cmake']) + '; '
                        f'cmake --build {build} --target llama-server -j 12'],
                       stdout=stream, stderr=subprocess.STDOUT, check=True, env=env, timeout=max(60, deadline - time.time()))

    status('prompts', outcome=known)
    sys.path.insert(0, str(ROOT))
    import gen_code
    want = manifest['prompts']
    order = gen_code.pick(str(dataset), want['loop'], gen_code.SEED, gen_code.MAX_CHARS)[want['first']:]
    if len(order) != want['count'] or ids_sha(order) != want['ids_sha256']:
        raise ValueError("The prompt order differs from the launcher's")
    prompts = WORK / 'prompts.jsonl'
    with prompts.open('w') as stream:
        for p in order:
            stream.write(json.dumps(p) + '\n')

    settings = manifest['settings']
    job = {'first': want['first'], 'server': build / 'bin/llama-server', 'model': model, 'prompts': prompts,
           'max_tokens': settings['max_tokens'], 'ctx': settings['ctx'], 'env': env,
           'server_args': ['-ngl', '99', '-t', '4']}
    outcome = work(job, len(order), settings, manifest['budget'], deadline, known)
    artifacts()
    status('complete', outcome=outcome)


def work(job, count, settings, budget, deadline, known):
    """Answer the `count` prompts of job['prompts'], as far as the time goes. Returns the outcome to report."""
    piece = settings['piece']
    created = deadline + 600 - budget['hours'] * 3600
    done = {'answers': 0, 'tokens': 0}           # in the pieces that are closed

    def report(stage, extra=None):
        def tell(answers, tokens, speed):
            status(stage, outcome=dict(known, **(extra or {}), answers=done['answers'] + answers,
                                       tokens=done['tokens'] + tokens, now_tok_s=speed and round(speed)))
        return tell

    # Which number of answers at a time is fastest? Each try is real work on the first piece.
    tried, best = {}, None
    first_piece = min(piece, count)
    for i, parallel in enumerate(settings['try_parallel']):
        # the first try also loads the model from a cold disk
        code, n, speed = generate(job, 0, first_piece, parallel, settings['try_seconds'][min(i, 1)],
                                  report(f'try {parallel} at a time', {'tried': tried}))
        tried[str(parallel)] = speed and round(speed)
        if code != 0 or speed is None:
            if best is None:
                raise RuntimeError(f'gen_code.py failed with {parallel} answers at a time (exit {code}): see gen.log')
            break
        if best is not None and speed < 1.08 * tried[str(best)]:
            best = parallel if speed > tried[str(best)] else best
            break
        best = parallel
    speed = tried[str(best)]
    so_far = Rows(OUT / 'rows' / f"rows-{job['first']:06d}-{job['first'] + first_piece:06d}.jsonl").read()
    hours_left = (count - so_far.n) * (so_far.tokens / so_far.n) / speed / 3600
    spent_hours = (time.time() - created) / 3600
    projected = (spent_hours + hours_left) * budget['usd_per_hour']
    known.update(tried=tried, parallel=best, tok_s=speed, tokens_per_answer=round(so_far.tokens / so_far.n),
                 hours_left_at_check=round(hours_left, 2), projected_usd=round(projected, 2))
    if projected > budget['cap_usd']:
        return dict(known, stopped='all the prompts would cost more than the cap', answers=so_far.n, tokens=so_far.tokens)

    for a in range(0, count, piece):
        b = min(count, a + piece)
        stage = f"rows {job['first'] + a}-{job['first'] + b} of {job['first'] + count}"
        n = 0
        for attempt in range(2):
            limit = deadline - STOP - time.time()
            if limit < 120:
                break
            code, n, speed = generate(job, a, b, best, limit, report(stage))
            if n >= 0.98 * (b - a) or time.time() > deadline - STOP:
                break
        else:
            if n < 0.9 * (b - a):
                raise RuntimeError(f'{stage}: only {n} answers after two passes (last exit {code}): see gen.log')
        closed = Rows(OUT / 'rows' / f"rows-{job['first'] + a:06d}-{job['first'] + b:06d}.jsonl").read()
        done['answers'] += closed.n
        done['tokens'] += closed.tokens
        if deadline - STOP - time.time() < 120:
            break
    return dict(known, answers=done['answers'], tokens=done['tokens'], prompts_left=count - done['answers'])


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--delete-self-at':
        delete_self_at(float(sys.argv[2]))
        sys.exit()
    try:
        main()
    except BaseException as error:
        signal.alarm(0)
        artifacts()
        status('failed', error=f'{type(error).__name__}: {error}')
        raise
