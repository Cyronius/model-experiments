#!/usr/bin/env python3
"""Have the target model write answers to coding prompts, and keep them as token ids.

The table's text has no code, and neither does the offline study. This makes some: the target
(through llama-server, the same build and GGUF the live runs use) answers prompts from the
OSS-Instruct set at temperature 0 with thinking off, several at a time. Each answer is stored with
its prompt and answer token ids, as a row ngram_study.Data reads. At temperature 0 the token the
model wrote is its top choice, so the rows need no stored log-probs (code_study.py fills them in).

    dibs run cpu port-8455 --label "MemDraft: the 9B writes code" --eta 3h -- \
        python memdraft/gen_code.py --dataset D/data-oss_instruct-decontaminated.jsonl --count 600 \
        --out target/memdraft/code/rows.jsonl -- -ngl 0 -t 8

The prompts are half Python and half the set's eight other languages, in a fixed order, so a
larger --count later adds prompts and keeps the ones already answered. Safe to stop and start
again: prompts with a row in --out are skipped.

--gpu-guard and --time-limit are for a borrowed GPU turn, as in live_bench.py: start only when
nothing else is on the GPU, stop our server if something arrives, and take no new prompt after
the limit.

--remote uses a llama-server that is already running, on this machine or another, and starts and
stops nothing:

    python memdraft/gen_code.py --remote http://HOST:PORT --skip 2000 --count 12000 --parallel 4 \
        --out target/memdraft/code/rows-9b-remote.jsonl

--prompts takes the prompts from a file and not from the set (gen_pod_worker.py, on a rented GPU,
answers one such file in pieces).
"""
import argparse
import json
import os
import random
import shlex
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import gpu_alone
from live_bench import Watch, log

SEED, MAX_CHARS = 20261110, 4000     # the prompt order; code_study.py holds prompts out by it

def call(where, path, body=None, timeout=1200):
    """`where` is the port of our own server, or the address of one that is already running."""
    base = where if isinstance(where, str) else f'http://127.0.0.1:{where}'
    request = urllib.request.Request(base + path, method='POST' if body is not None else 'GET',
                                     data=json.dumps(body).encode() if body is not None else None,
                                     headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def pick(dataset, count, seed, max_chars):
    """The first `count` prompts of a fixed order: Python and the other languages alternate."""
    by_lang = {}
    for n, line in enumerate(open(dataset)):       # the set's own `index` repeats; the line number does not
        r = json.loads(line)
        if len(r['problem']) <= max_chars:
            by_lang.setdefault(r['lang'], []).append({'id': f'oss-{n}', 'lang': r['lang'], 'prompt': r['problem']})
    rng = random.Random(seed)
    for lang in sorted(by_lang):
        rng.shuffle(by_lang[lang])
    others = sorted(l for l in by_lang if l != 'python')
    out = []
    for i in range(count):
        lang = 'python' if i % 2 == 0 else others[(i // 2) % len(others)]
        if by_lang[lang]:
            out.append(by_lang[lang].pop())
    return out


class Server:
    def __init__(self, binary, model, port, parallel, ctx, common, logfile):
        self.port = port
        self.log = open(logfile, 'ab')
        command = [binary, '-m', model, '--host', '127.0.0.1', '--port', str(port), '-np', str(parallel),
                   '-c', str(parallel * ctx), '--jinja', '--cache-ram', '0', '--no-warmup'] + common
        self.log.write((' '.join(shlex.quote(c) for c in command) + '\n').encode())
        self.log.flush()
        self.process = subprocess.Popen(command, stdout=self.log, stderr=subprocess.STDOUT, start_new_session=True)

    def wait_ready(self, seconds=900):
        deadline = time.time() + seconds
        while time.time() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f'server exited with {self.process.returncode}')
            try:
                if call(self.port, '/health', timeout=5).get('status') == 'ok':
                    return
            except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
                pass
            time.sleep(1)
        raise RuntimeError('server did not become ready')

    def stop(self):
        if self.process.poll() is None:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
                self.process.wait(20)                   # with answers open it does not go by itself
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait()
            except ProcessLookupError:
                pass
        self.log.close()


def prompt_ids(port, prompt):
    """The token ids /v1/chat/completions would decode for this prompt, thinking off."""
    text = call(port, '/apply-template', {'messages': [{'role': 'user', 'content': prompt}],
                                         'chat_template_kwargs': {'enable_thinking': False}})['prompt']
    return call(port, '/tokenize', {'content': text, 'add_special': True, 'parse_special': True})['tokens']


def answer(port, p, max_tokens):
    began = time.time()
    ids = prompt_ids(port, p['prompt'])
    got = call(port, '/completion', {'prompt': ids, 'n_predict': max_tokens, 'temperature': 0.0, 'seed': 1,
                                     'cache_prompt': False, 'return_tokens': True}, timeout=3600)
    return {'id': p['id'], 'source': 'code', 'lang': p['lang'], 'thinking': False, 'prompt_ids': ids,
            'answer_ids': got['tokens'], 'finished': got.get('stop_type') == 'eos', 'stop': got.get('stop_type'),
            'text': got['content'], 'seconds': round(time.time() - began, 1),
            'decode_ms': got['timings']['predicted_ms']}


def same_as_chat(port, prompt):
    """Our three calls must give what the chat endpoint gives: same prompt length, same first tokens."""
    chat = call(port, '/v1/chat/completions', {'messages': [{'role': 'user', 'content': prompt}], 'max_tokens': 12,
                                               'temperature': 0.0, 'seed': 1, 'cache_prompt': False,
                                               'chat_template_kwargs': {'enable_thinking': False}})
    mine = answer(port, {'id': 'check', 'lang': '', 'prompt': prompt}, 12)
    message = chat['choices'][0]['message']
    theirs = (message.get('reasoning_content') or '') + (message.get('content') or '')
    return chat['timings']['prompt_n'] == len(mine['prompt_ids']) and theirs.strip() == mine['text'].strip(), \
        f"chat: {chat['timings']['prompt_n']} prompt tokens, {theirs!r}; ours: {len(mine['prompt_ids'])}, {mine['text']!r}"


def main():
    argv = sys.argv[1:]
    common = []
    if '--' in argv:
        at = argv.index('--')
        common, argv = argv[at + 1:], argv[:at]
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    S = '~/.cache/model-experiments/memdraft'
    ap.add_argument('--server', default=f'{S}/llama.cpp/build-cpu/bin/llama-server')
    ap.add_argument('--model', default=f'{S}/models/Qwen3.5-9B-Q4_K_M.gguf')
    ap.add_argument('--dataset', default=f'{S}/code-prompts-src/data-oss_instruct-decontaminated.jsonl')
    ap.add_argument('--out', required=True, help='rows are appended here, one JSON object per line')
    ap.add_argument('--count', type=int, required=True, help='answer the first N prompts of the fixed order')
    ap.add_argument('--seed', type=int, default=SEED)
    ap.add_argument('--max-chars', type=int, default=MAX_CHARS, help='leave out prompts longer than this')
    ap.add_argument('--max-tokens', type=int, default=1024)
    ap.add_argument('--parallel', type=int, default=8, help='answers written at the same time')
    ap.add_argument('--ctx', type=int, default=3072, help='context per answer, prompt included')
    ap.add_argument('--port', type=int, default=8455)
    ap.add_argument('--remote', metavar='URL', help='use the llama-server already running at this address (http://host:port) '
                    'and start none; keep --parallel within its slots')
    ap.add_argument('--skip', type=int, default=0, help='leave out the first N prompts of the order')
    ap.add_argument('--prompts', metavar='FILE', help='answer the prompts of this file in its order (JSON lines with id, '
                    'lang, prompt) and pick none from --dataset; --count and --skip then count its lines')
    ap.add_argument('--gpu-guard', action='store_true')
    ap.add_argument('--time-limit', type=int, metavar='SECONDS', help='take no new prompt after this long')
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(line)['id'] for line in out.open()} if out.exists() else set()
    if args.prompts:
        order = [json.loads(line) for line in open(args.prompts)][:args.count]
    else:
        order = pick(os.path.expanduser(args.dataset), args.count, args.seed, args.max_chars)
    todo = [p for p in order[args.skip:] if p['id'] not in done]
    log(f'{len(done)} rows in {out}, {len(todo)} prompts to answer')
    if not todo:
        return
    if args.remote and (args.gpu_guard or common):
        ap.error("--remote starts no server: --gpu-guard and server options after -- do not apply")
    if args.gpu_guard and not gpu_alone.wait(2, 5, report=log, until=args.time_limit and time.time() + args.time_limit / 2):
        log('the GPU was not ours in time')
        sys.exit(75)
    limit = time.time() + args.time_limit if args.time_limit else None
    where, server, served = args.port, None, None
    if args.remote:
        where = args.remote.rstrip('/')
        props = call(where, '/props', timeout=30)
        served = os.path.basename(props.get('model_alias') or props.get('model_path') or '') or where
        slots, ctx = props.get('total_slots'), props.get('default_generation_settings', {}).get('n_ctx')
        log(f'remote server {where}: {served}, {slots} slots, context {ctx} per slot')
        if slots and args.parallel > slots:
            log(f'--parallel {args.parallel} is more than its {slots} slots: the extra requests will wait in its queue')
        if ctx and ctx < args.ctx:
            ap.error(f'its context per slot ({ctx}) is under --ctx {args.ctx}: long prompts would be cut')
    else:
        server = Server(os.path.expanduser(args.server), os.path.expanduser(args.model), args.port, args.parallel, args.ctx,
                        common, out.with_suffix('.server.log'))
    watch = Watch(server) if args.gpu_guard else None
    lock, began, written, cancelled = threading.Lock(), time.time(), [0, 0], threading.Event()

    def work(p):
        if cancelled.is_set() or (limit and time.time() > limit):
            return
        try:
            row = answer(where, p, args.max_tokens)
            if served:
                row['server'] = served
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError, KeyError) as error:
            if not (watch and watch.company):
                log(f"{p['id']}: {type(error).__name__}: {error}")
            return
        with lock:
            with out.open('a') as f:
                f.write(json.dumps(row) + '\n')
            written[0] += 1
            written[1] += len(row['answer_ids'])
            if written[0] % 20 == 0:
                log(f'{written[0]} answers, {written[1]:,} tokens, {written[1] / (time.time() - began):.1f} tok/s')

    def terminated(*_):
        # A cancel must not leave our server running, and must drop the prompts still waiting: the pool
        # would go on through all of them, and send them to whatever server is on the port by then.
        cancelled.set()
        if server:
            server.stop()
        sys.exit(143)

    signal.signal(signal.SIGTERM, terminated)
    try:
        if watch:
            watch.start()
        if server:
            server.wait_ready()
        ok, what = same_as_chat(where, todo[0]['prompt'])
        log(('matches the chat endpoint: ' if ok else 'DOES NOT MATCH the chat endpoint: ') + what)
        if not ok:
            sys.exit(2)
        began = time.time()
        with ThreadPoolExecutor(args.parallel) as pool:
            list(pool.map(work, todo))
    finally:
        if watch:
            watch.done.set()
        if server:
            server.stop()
    if watch and watch.company:
        log(f'another process came onto the GPU, stopped: {watch.company[0][1]}')
    log(f'{written[0]} answers, {written[1]:,} tokens in {time.time() - began:.0f} s, '
        f'{written[1] / max(1.0, time.time() - began):.1f} tok/s')


if __name__ == '__main__':
    main()
