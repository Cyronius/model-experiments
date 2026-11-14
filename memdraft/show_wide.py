#!/usr/bin/env python3
"""Sum the turns of run_wide_live.sh: per prompt set and arm, decode tokens/s against plain decoding.

Every arm is paired with plain decoding of the same turn on the same prompts (each turn starts a
fresh plain server), so slow and fast turns weigh on both. Per set and arm:
  speed     the arm's tokens / time over the plain arm's tokens / time on the same requests,
            with a 95% bootstrap interval over prompts (both rounds of a prompt stay together)
  r1, r2    the same per round
  median    of per-request speeds; faster: on how many requests it beat plain
  tok/step  tokens per decode step; step: what one of its steps costs in plain steps
  same      requests whose text equals plain decoding's
Requests cut short by another process on the GPU are left out.

    memdraft/show_wide.py memdraft/results/wide-9b-vk-STAMP
"""
import collections
import json
import random
import statistics
import sys
from pathlib import Path

SETS = {'oss': ('python', 'other-languages'), 'edit': ('edit',), 'write': ('write',), 'edit+write': ('edit', 'write')}


def main(folder):
    pairs = collections.defaultdict(list)     # arm -> [(set row, base row, round)]
    arms, loads = [], []
    for path in sorted(Path(folder).glob('*.json')):
        record = json.load(open(path))
        rnd = int(path.stem.split('-r')[1].split('-')[0])
        loads.append((record.get('load_average_start', 0), record.get('load_average_end', 0)))
        rows = [r for r in record['rows'] if not r.get('shared_gpu') and r.get('prompt')]
        base = next(iter(record['arms']))
        plain = {r['prompt']: r for r in rows if r['arm'] == base}
        for r in rows:
            if r['prompt'] in plain:
                if r['arm'] not in arms:
                    arms.append(r['arm'])
                pairs[r['arm']].append((r, plain[r['prompt']], rnd))
    print(f'{folder}: {len(loads)} turn files, load average at turn starts '
          f'{min(l[0] for l in loads):.1f} to {max(l[0] for l in loads):.1f}')
    rng = random.Random(0)
    for name, domains in SETS.items():
        print(f'\n{name}  ({"/".join(domains)})')
        print(f'{"arm":14s} {"tok/s":>6s} {"plain":>6s} {"speed":>6s} {"95% interval":>13s} {"r1":>6s} {"r2":>6s} '
              f'{"median":>6s} {"faster":>7s} {"tok/step":>8s} {"step":>5s} {"same":>6s}')
        for arm in arms:
            mine = [p for p in pairs[arm] if p[0]['domain'] in domains]
            if not mine:
                continue

            def speed(sel):
                t, ms = sum(a['tokens'] for a, _, _ in sel), sum(a['ms'] for a, _, _ in sel)
                bt, bms = sum(b['tokens'] for _, b, _ in sel), sum(b['ms'] for _, b, _ in sel)
                return (t / ms) / (bt / bms) if ms and bms else float('nan')
            by_prompt = collections.defaultdict(list)
            for p in mine:
                by_prompt[p[0]['prompt']].append(p)
            keys = list(by_prompt)
            boot = sorted(speed([p for k in (rng.choice(keys) for _ in keys) for p in by_prompt[k]]) for _ in range(2000))
            per = [(a['tokens'] / a['ms']) / (b['tokens'] / b['ms']) for a, b, _ in mine if a['ms'] and b['ms']]
            t, ms = sum(a['tokens'] for a, _, _ in mine), sum(a['ms'] for a, _, _ in mine)
            bt, bms = sum(b['tokens'] for _, b, _ in mine), sum(b['ms'] for _, b, _ in mine)
            steps = t - sum(a['accepted'] for a, _, _ in mine)
            step_cost = (ms / steps) / (bms / bt)
            rounds = [speed([p for p in mine if p[2] == r]) if any(p[2] == r for p in mine) else float('nan') for r in (1, 2)]
            same = sum(a['text'] == b['text'] for a, b, _ in mine)
            print(f'{arm:14s} {1000 * t / ms:6.1f} {1000 * bt / bms:6.1f} {speed(mine):6.3f} '
                  f'{boot[50]:6.3f}-{boot[1949]:.3f} {rounds[0]:6.3f} {rounds[1]:6.3f} {statistics.median(per):6.3f} '
                  f'{sum(x > 1 for x in per):3d}/{len(per):<3d} {t / steps:8.2f} {step_cost:5.2f} {same:3d}/{len(mine):<3d}')


if __name__ == '__main__':
    main(sys.argv[1])
