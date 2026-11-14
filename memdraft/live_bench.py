#!/usr/bin/env python3
"""Tokens per second of a llama-server, one arm per set of speculative-decoding options.

Every arm gets its own server process on the same model and answers the same prompts with the same
sampling. Arms take turns round by round (none, a, b, none, a, b, ...), or after every few prompts
with --chunk, so drift on a shared machine lands on all of them. The number that counts is decode tokens per second from the server's
own timings: tokens generated / time spent generating, prompt processing excluded.

    dibs run --measure gpu --eta 30m -- python live_bench.py \
        --server ~/.cache/model-experiments/memdraft/llama.cpp/build-vk/bin/llama-server \
        --model MODEL.gguf --prompts prompts/live-v1.jsonl --out results/live-NAME.json -- -ngl 99 \
        --arm none= \
        --arm table="--spec-type ngram-table --spec-table-file T.mdt --spec-table-tau 0.3"

Options after a bare `--` and before the first --arm go to every server (-ngl, -t, -c ...).
Timing is only meaningful under `dibs run --measure`; say so with --clean when it was.

--gpu-guard is for GPU runs: every arm waits until no other process has the GPU (gpu_alone.py). If
another process shows up on the GPU while an arm runs, our server is stopped within seconds and
the arm is marked shared and left out of the summary.

--time-limit is for a borrowed GPU turn: no arm starts unless it can finish within that many
seconds of the start, so the GPU goes back on time. The arms that got all their answers are
listed as complete_arms in the result; run the rest on a later turn, with the baseline again.
"""
import argparse
import http.client
import json
import os
import shlex
import signal
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import gpu_alone


def log(msg):
    print(time.strftime('%H:%M:%S'), msg, flush=True)


def call(port, path, body=None, timeout=1200):
    request = urllib.request.Request(f'http://127.0.0.1:{port}{path}', method='POST' if body is not None else 'GET',
                                     data=json.dumps(body).encode() if body is not None else None,
                                     headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


class Server:
    def __init__(self, binary, model, port, common, arm_args, logfile):
        self.port = port
        self.log = open(logfile, 'ab')
        command = [binary, '-m', model, '--host', '127.0.0.1', '--port', str(port), '-np', '1', '--jinja',
                   '--cache-ram', '0', '--no-warmup'] + common + arm_args
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
        """Safe to call twice: the GPU watch and the main thread may both stop the server."""
        if self.process.poll() is None:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
                self.process.wait(60)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait()
            except ProcessLookupError:
                pass
        self.log.close()


class Watch(threading.Thread):
    """While an arm runs, look at the GPU every few seconds; if another process shows up on it,
    stop our server at once so the two never share it for long."""

    def __init__(self, server, every=5):
        super().__init__(daemon=True)
        self.server, self.every, self.company, self.done = server, every, [], threading.Event()

    def run(self):
        while not self.done.wait(self.every):
            others = gpu_alone.users(ignore={self.server.process.pid})
            if others:
                self.company = others
                self.server.stop()
                return


def ask(port, prompt, max_tokens, temperature, thinking):
    body = {'messages': [{'role': 'user', 'content': prompt}], 'max_tokens': max_tokens, 'temperature': temperature,
            'cache_prompt': False, 'seed': 1, 'chat_template_kwargs': {'enable_thinking': thinking}}
    answer = call(port, '/v1/chat/completions', body)
    t = answer['timings']
    message = answer['choices'][0]['message']
    return {'text': (message.get('reasoning_content') or '') + (message.get('content') or ''),
            'tokens': t['predicted_n'], 'ms': t['predicted_ms'], 'prompt_tokens': t['prompt_n'], 'prompt_ms': t['prompt_ms'],
            'drafted': t.get('draft_n', 0), 'accepted': t.get('draft_n_accepted', 0)}


def summarize(arms, rows, base):
    """Per arm: total decode tokens/s, and the median over prompts of its speed against `base`."""
    out = {}
    rows = [r for r in rows if not r.get('shared_gpu')]
    for arm in arms:
        mine = [r for r in rows if r['arm'] == arm]
        tokens, ms = sum(r['tokens'] for r in mine), sum(r['ms'] for r in mine)
        drafted, accepted = sum(r['drafted'] for r in mine), sum(r['accepted'] for r in mine)
        ratios = []
        for r in mine:
            ref = [b for b in rows if b['arm'] == base and b['prompt'] == r['prompt'] and b['round'] == r['round']]
            if ref and r['ms'] > 0 and ref[0]['ms'] > 0:
                ratios.append((r['tokens'] / r['ms']) / (ref[0]['tokens'] / ref[0]['ms']))
        same = [r['text'] == b['text'] for r in mine for b in rows
                if b['arm'] == base and b['prompt'] == r['prompt'] and b['round'] == r['round']]
        steps = tokens - accepted
        out[arm] = {'tokens_per_second': 1000 * tokens / ms if ms else None, 'tokens': tokens, 'requests': len(mine),
                    'tokens_per_step': tokens / steps if steps else None, 'ms_per_step': ms / steps if steps else None,
                    'median_speed_vs_base': statistics.median(ratios) if ratios else None,
                    'min_speed_vs_base': min(ratios) if ratios else None, 'max_speed_vs_base': max(ratios) if ratios else None,
                    'drafted': drafted, 'accepted': accepted,
                    'same_text_as_base': f'{sum(same)}/{len(same)}'}
    return out


def main():
    argv = sys.argv[1:]
    common = []
    if '--' in argv:
        at = argv.index('--')
        rest = argv[at + 1:]
        first_arm = next((i for i, a in enumerate(rest) if a == '--arm'), len(rest))
        common, argv = rest[:first_arm], argv[:at] + rest[first_arm:]
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--server', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--prompts', required=True, help='jsonl with id, domain, prompt')
    ap.add_argument('--out', required=True)
    ap.add_argument('--arm', action='append', required=True, metavar='NAME=OPTIONS',
                    help='an arm and its server options; the first arm is the baseline')
    ap.add_argument('--rounds', type=int, default=1)
    ap.add_argument('--max-tokens', type=int, default=256)
    ap.add_argument('--temperature', type=float, default=0.0)
    ap.add_argument('--thinking', action='store_true')
    ap.add_argument('--limit', type=int, help='use only the first N prompts')
    ap.add_argument('--per-domain', type=int, help='use only the first N prompts of each domain')
    ap.add_argument('--port', type=int, default=8455)
    ap.add_argument('--gpu-guard', action='store_true',
                    help='start each arm only when nothing else is on the GPU, and stop it if something arrives')
    ap.add_argument('--guard-checks', type=int, default=3, help='looks in a row that must find the GPU alone')
    ap.add_argument('--guard-gap', type=int, default=20, help='seconds between those looks')
    ap.add_argument('--chunk', type=int, help='prompts per server start. Arms take turns after every chunk, so --chunk 1 puts '
                    'all arms on one prompt back to back; use it where speed drifts within minutes (a shared CPU)')
    ap.add_argument('--retries', type=int, default=0, help='with --gpu-guard: run an interrupted arm again, up to this many times')
    ap.add_argument('--time-limit', type=int, metavar='SECONDS', help='start no arm that could not finish within this long of the '
                    'start, judged by the longest arm so far; arms are tried in the order given, so put the wanted ones first')
    ap.add_argument('--arm-seconds', type=int, default=90, help='with --time-limit: what to allow for an arm before one has been timed')
    ap.add_argument('--clean', action='store_true', help='this run is inside `dibs run --measure`')
    args = ap.parse_args(argv)

    arms = {}
    for spec in args.arm:
        name, _, options = spec.partition('=')
        arms[name] = shlex.split(options)
    prompts = [json.loads(line) for line in open(args.prompts)][:args.limit]
    if args.per_domain:
        seen = {}
        prompts = [p for p in prompts if seen.setdefault(p.get('domain'), []).append(p) or len(seen[p.get('domain')]) <= args.per_domain]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    server_log = out.with_suffix('.server.log')
    record = {'args': vars(args), 'common': common, 'arms': arms, 'started': time.strftime('%Y-%m-%d %H:%M:%S'),
              'load_average_start': os.getloadavg()[0], 'rows': []}
    base = next(iter(arms))
    chunk = args.chunk or len(prompts)
    turns = [(rnd, prompts[i:i + chunk]) for rnd in range(args.rounds) for i in range(0, len(prompts), chunk)]
    limit = time.time() + args.time_limit if args.time_limit else None
    longest, stopped = 0, None

    def answered(arm, turn):
        return sum(1 for r in record['rows'] if r['arm'] == arm and r['turn'] == turn and r['prompt'] is not None and not r.get('shared_gpu'))

    try:
        for turn, (rnd, some) in enumerate(turns):
            for arm, options, attempt in [(a, o, t) for a, o in arms.items() for t in range(args.retries + 1)]:
                if stopped:
                    break
                if attempt and not any(r.get('shared_gpu') and r['arm'] == arm and r['turn'] == turn and r['attempt'] == attempt - 1
                                       for r in record['rows']):
                    continue                                              # the attempt before this one was not interrupted
                need = 1.25 * longest or args.arm_seconds
                if limit and arm != base and answered(base, turn) < len(some):
                    stopped = f'{base} was interrupted, so nothing after it would have a baseline'
                elif limit and time.time() + need > limit:
                    stopped = f'{arm} needs about {need:.0f} s and {max(0, limit - time.time()):.0f} s are left'
                elif args.gpu_guard and not gpu_alone.wait(args.guard_checks, args.guard_gap, report=log, until=limit and limit - need):
                    stopped = f'the GPU was not ours in time for {arm} (about {need:.0f} s)'
                if stopped:
                    log(f'stopping here: {stopped}')
                    break
                began = time.time()
                server = Server(os.path.expanduser(args.server), os.path.expanduser(args.model), args.port, common, options, server_log)
                watch = Watch(server) if args.gpu_guard else None
                try:
                    if watch:
                        watch.start()
                    server.wait_ready()
                    ask(args.port, 'Say hello.', 8, 0.0, False)           # first-touch costs stay out of the timings
                    for p in some:
                        row = ask(args.port, p['prompt'], args.max_tokens, args.temperature, args.thinking)
                        record['rows'].append(dict(row, arm=arm, round=rnd, turn=turn, attempt=attempt, prompt=p['id'], domain=p.get('domain')))
                except (urllib.error.URLError, http.client.HTTPException, ConnectionError, RuntimeError, OSError) as error:
                    if watch and watch.company:
                        log(f'round {rnd} {arm}: stopped our server, {type(error).__name__}')
                    elif limit:                                           # a borrowed turn goes on with the other arms
                        log(f'round {rnd} {arm}: failed, skipped: {type(error).__name__}: {error}')
                    else:
                        raise
                finally:
                    if watch:
                        watch.done.set()
                    server.stop()
                mine = [r for r in record['rows'] if r['arm'] == arm and r['turn'] == turn and r['attempt'] == attempt]
                if watch and watch.company:
                    if not mine:                                          # interrupted before the first answer: leave a marker so a retry follows
                        record['rows'].append({'arm': arm, 'round': rnd, 'turn': turn, 'attempt': attempt, 'prompt': None, 'domain': None})
                        mine = record['rows'][-1:]
                    for r in mine:
                        r['shared_gpu'] = True
                    log(f'round {rnd} {arm}: another process came onto the GPU, this arm is not counted: {watch.company[0][1]}')
                mine = [r for r in mine if r['prompt'] is not None]
                if len(mine) == len(some) and not (watch and watch.company):
                    longest = max(longest, time.time() - began)
                if mine:
                    log(f"round {rnd} {some[0]['id'] + ' ' if args.chunk else ''}{arm}: {1000 * sum(r['tokens'] for r in mine) / sum(r['ms'] for r in mine):.2f} tok/s, "
                        f"{sum(r['accepted'] for r in mine)}/{sum(r['drafted'] for r in mine)} draft tokens accepted")
                record['load_average_end'] = os.getloadavg()[0]
                record['summary'] = summarize(list(arms), record['rows'], base)
                record['by_domain'] = {d: summarize(list(arms), [r for r in record['rows'] if r['domain'] == d], base)
                                       for d in sorted({p.get('domain') or 'all' for p in prompts})}
                out.write_text(json.dumps(record, indent=1))
    finally:                                                          # also after an error: the next turn reads complete_arms
        record['stopped_early'] = stopped
        record['complete_arms'] = [a for a in arms if all(answered(a, t) == len(some) for t, (_, some) in enumerate(turns))]
        out.write_text(json.dumps(record, indent=1))
    for arm, s in record.get('summary', {}).items():
        if s['tokens_per_second'] is None or s['median_speed_vs_base'] is None:
            log(f'{arm}: no countable requests')
            continue
        log(f"{arm}: {s['tokens_per_second']:.2f} tok/s, median {s['median_speed_vs_base']:.3f}x of {base} "
            f"({s['min_speed_vs_base']:.2f} to {s['max_speed_vs_base']:.2f}), same text {s['same_text_as_base']}")


if __name__ == '__main__':
    main()
