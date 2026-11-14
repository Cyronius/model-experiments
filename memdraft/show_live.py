#!/usr/bin/env python3
"""Print a live_bench.py result: every prompt under every arm, then one line per arm.

    python show_live.py results/lookup-9b-vk-*.json

Per prompt: tokens/s, draft tokens accepted/drafted, and = when the text equals the first arm's.
Per arm: tokens/s over all its prompts, against the first arm on the same prompts, and paired per
prompt (median, range, on how many prompts it was faster). `step` is what one of its decode steps
costs in plain steps of the same prompt (median over prompts). Requests cut short by another
process on the GPU are left out.
"""
import collections
import json
import statistics
import sys


def speed(rows):
    return 1000 * sum(r['tokens'] for r in rows) / sum(r['ms'] for r in rows)


def main(path):
    record = json.load(open(path))
    rows = [r for r in record['rows'] if not r.get('shared_gpu') and r.get('prompt')]
    arms = [a for a in record['arms'] if any(r['arm'] == a for r in rows)]
    base = arms[0]
    print(path, '| load average', round(record.get('load_average_start', 0), 1), 'to', round(record.get('load_average_end', 0), 1))
    cell = collections.defaultdict(dict)
    for r in rows:
        cell[r['prompt']].setdefault(r['arm'], r)
    width = max(15, max(len(a) for a in arms))
    print('%-8s' % 'prompt', *[a.rjust(width) for a in arms])
    for prompt, by_arm in cell.items():
        texts = []
        for a in arms:
            r = by_arm.get(a)
            same = '' if r is None or base not in by_arm else '=' if r['text'] == by_arm[base]['text'] else '~'
            texts.append(('' if r is None else '%5.1f %3d/%-4d%s' % (1000 * r['tokens'] / r['ms'], r['accepted'], r['drafted'], same)).rjust(width))
        print('%-8s' % prompt, *texts)
    print()
    domains = sorted({r['domain'] for r in rows})
    print('%-*s %6s %6s  %-22s %9s %8s %5s  %s' % (width, 'arm', 'tok/s', 'x', 'per prompt: median (range) faster', 'accepted', 'tok/step', 'step', 'x by domain'))
    for a in arms:
        mine = [r for p in cell.values() for r in [p.get(a)] if r is not None and base in p]
        ref = [p[base] for p in cell.values() if a in p and base in p]
        if not mine:
            continue
        ratios = [(r['tokens'] / r['ms']) / (b['tokens'] / b['ms']) for r, b in zip(mine, ref)]
        steps = [(r['ms'] / max(1, r['tokens'] - r['accepted'])) / (b['ms'] / b['tokens']) for r, b in zip(mine, ref)]
        tokens, accepted, drafted = (sum(r[k] for r in mine) for k in ('tokens', 'accepted', 'drafted'))
        by_domain = ' '.join('%s %.2f' % (d, speed([r for r in mine if r['domain'] == d]) / speed([b for b in ref if b['domain'] == d]))
                             for d in domains if any(r['domain'] == d for r in mine))
        paired = '%.2f (%.2f-%.2f) %d/%d' % (statistics.median(ratios), min(ratios), max(ratios), sum(x > 1 for x in ratios), len(ratios))
        print('%-*s %6.2f %6.3f  %-22s %4d/%-4d %8.2f %5.2f  %s' % (width, a, speed(mine), speed(mine) / speed(ref), paired, accepted, drafted,
                                                                  tokens / (tokens - accepted), statistics.median(steps), by_domain))


if __name__ == '__main__':
    for path in sys.argv[1:]:
        main(path)
