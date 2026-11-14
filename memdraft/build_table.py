#!/usr/bin/env python3
"""Build the table the drafter ships with, and check the C++ drafter against the study code.

The table counts the model's top choice after the last 1..K tokens over the training shards. Its
confidences come from held-out requests of the last shard (a quarter of its problems); the rest of
that shard is written out as a replay stream.

    python build_table.py --shards DIR0 DIR1 DIR2 --out target/memdraft/9b --check

--check compiles replay.cpp and requires its counts (steps, drafted, accepted) to equal the numpy
drafter's on the stream, for both scoring modes, all three policies and three thresholds.
--all-shards also counts the held-out shard into the table. Use that for a table that will meet
new prompts in a live run; its replay stream is then no longer held out.
"""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

import ngram_study as ns

HERE = Path(__file__).resolve().parent


def split(shards, cache=None):
    """(training rows, calibration Data, test Data), split as the study splits them."""
    loaded = [ns.load_shard(s, cache=cache) for s in shards]
    held = [r for r in loaded[-1] if r['finished']]
    order = {p: i for i, p in enumerate(dict.fromkeys(r['id'] for r in held))}
    cal = ns.Data([r for r in held if order[r['id']] % 4 == 0])
    test = ns.Data([r for r in held if order[r['id']] % 4 != 0])
    return [r for s in loaded[:-1] for r in s], loaded[-1], cal, test


def counted(rows, K):
    train = ns.Data([r for r in rows if r['finished']])
    idx = ns.labelled(train)
    return ns.Static.build(train, idx, train.amax[idx], K), len(idx)


def reference(table, data, modes, policies, taus, n_max, pads=(1, 3)):
    """The numpy drafter's totals on `data` with the shipped table, keyed (mode, policy, rule, value)."""
    idx = ns.labelled(data)
    out = {}
    for policy in policies:
        inreq = ns.InRequest(data, table.K) if policy != 'static' else None
        tokens, chance = ns.draft(data, idx, table if policy != 'context' else None, inreq, table, table.K, n_max,
                                  min(taus), max(pads, default=0))
        for mode in modes:
            accepted_all, judged = ns.score(data, idx, tokens, chance, mode)
            rules = [('tau', t, np.minimum((chance >= t).sum(1).astype(np.int32), judged)) for t in taus]
            rules += [('pad', n, np.minimum(np.int32(n), judged)) for n in pads]
            for rule, value, drafted in rules:
                run = ns.walk(data, idx, np.minimum(accepted_all, drafted), drafted)
                out[mode, policy, rule, value] = {k: int(run[k].sum()) for k in ns.COUNTS}
    return out


def replay(binary, table, stream, mode, policy, rule, value, n_max):
    """replay.cpp's totals for one rule: ('tau', t) or ('pad', n)."""
    setting = ['--tau', str(value), '--n-max', str(n_max)] if rule == 'tau' else ['--pad', '1', '--n-max', str(value)]
    done = subprocess.run([str(binary), str(table), str(stream), '--mode', mode, '--policy', policy] + setting,
                          check=True, capture_output=True, text=True)
    return json.loads(done.stdout)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--shards', nargs='+', required=True, help='shard folders; the last one is held out')
    ap.add_argument('--out', required=True, help='output prefix: writes PREFIX.mdt, PREFIX.stream, PREFIX.json')
    ap.add_argument('--K', type=int, default=6)
    ap.add_argument('--floor', type=float, default=0.1, help='drop entries whose confidence is below this')
    ap.add_argument('--min-count', type=int, default=1, help='drop contexts seen fewer times than this')
    ap.add_argument('--all-shards', action='store_true')
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--cache')
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    train_rows, held_rows, cal, test = split(args.shards, args.cache)
    static, n_train = counted(train_rows, args.K)
    static.min_count = args.min_count
    calib = ns.Calibration(args.K)
    idx = ns.labelled(cal)
    C, avail = cal.context(idx, args.K)
    calib.add(ns.candidates(C, avail, cal.seq[idx], cal.pos[idx], static, ns.InRequest(cal, args.K), args.K), cal.amax[idx])
    if args.all_shards:
        static, n_train = counted(train_rows + held_rows, args.K)
        static.min_count = args.min_count
    kept, dropped = ns.export_table(static, calib, f'{out}.mdt', args.floor)
    ns.write_stream(test, f'{out}.stream')
    size = Path(f'{out}.mdt').stat().st_size
    ns.log(f'{out}.mdt: {kept:,} entries from {n_train:,} tokens, {dropped:,} dropped for space, {size / 2**20:.1f} MiB')
    record = {'args': vars(args), 'train_tokens': n_train, 'entries': kept, 'dropped': dropped, 'bytes': size}

    if args.check:
        table = ns.Exported(f'{out}.mdt')
        modes, policies, taus, n_max = ['greedy', 'sampled'], ['static', 'context', 'both'], [0.1, 0.3, 0.6], 8
        want = reference(table, test, modes, policies, taus, n_max)
        binary = out.parent / 'replay'
        subprocess.run(['g++', '-O2', '-std=c++17', '-Wall', '-Wextra', '-o', str(binary), str(HERE / 'replay.cpp')], check=True)
        record['replay'] = []
        for (mode, policy, rule, value), expect in want.items():
            got = replay(binary, f'{out}.mdt', f'{out}.stream', mode, policy, rule, value, n_max)
            record['replay'].append(dict(got, rule=rule, value=value))
            same = all(got[k] == expect[k] for k in expect)
            ns.log(f"{mode:8s}{policy:8s}{rule} {value}: {got['tokens'] / got['steps']:.3f} tokens/step, "
                   f"{got['draft_ns_per_step']:.0f} ns per draft call, C++ {'=' if same else '!='} numpy")
            if not same:
                raise SystemExit(f'replay.cpp disagrees with ngram_study: {got} vs {expect}')
    Path(f'{out}.json').write_text(json.dumps(record, indent=1))


if __name__ == '__main__':
    main()
