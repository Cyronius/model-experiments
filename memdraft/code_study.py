#!/usr/bin/env python3
"""Offline study on code: what would the table drafter get on code the target wrote?

ngram_study.py's text is math, reasoning and chat. Here the held-out text is answers the 9B wrote
to coding prompts (gen_code.py: temperature 0, thinking off), and the question is where a table
for code should come from. Tables are counted from:

  shipped   the text the shipped table was counted from (the heal shards): no code in it
  own       code the 9B wrote for other prompts
  other     code another model wrote: the OSS-Instruct set's own solutions (gpt-3.5), tokenized
            here. "other, same prompts" answers exactly the prompts of `own`, so the two differ
            only in who wrote the text; "other, all" is every solution outside the held-out prompts.

Each table is scored alone and together with the in-request lookup, and the in-request lookup is
scored alone. Scoring, calibration and the cost model are ngram_study's. A prompt's crc decides
its part: 2 in 8 set the confidences and pick the drafting rule, 2 in 8 are the test, 4 in 8 are
`own` table text. Only finished answers are tested (as in ngram_study); table text uses all.
--more-rows adds answers to prompts past the pool (run_gen_more.sh on this machine, run_gen_pod.py
on a rented GPU) to the `own` table text.
The calibration and test prompts among the first --pool prompts of gen_code.py's order are kept
out of every table, answered yet or not, so a table written today stays clean as answers arrive.

    python memdraft/code_study.py --rows target/memdraft/code/rows-9b.jsonl --out memdraft/results/code-study.json

For a live run (run_code_live.sh): --export writes one of the tables for the llama.cpp drafter,
and --prompts-out writes test prompts for live_bench.py.

    python memdraft/code_study.py --rows ROWS --tables 'shipped + own + other all' --min-count 3 \
        --export 'shipped + own + other all=target/memdraft/code/9b-code' --prompts-out memdraft/prompts/code-v1.jsonl
"""
import argparse
import json
import os
import time
import zlib
from pathlib import Path

import numpy as np

import gen_code
import ngram_study as ns

S = 'manifold-compression/results/heal-shards-pod-20261105'
SHARDS = [f'{S}/mc-shard-ec420510e5b4/shards/shard_000', f'{S}/mc-shard-d65446e1cd6b/shards/shard_001',
          f'{S}/mc-shard-1135ab4d74a4/shards/shard_002']
COSTS = [{'name': 'gpu9b', 'beta': 0.065, 'reject': 0.0, 'reshape': 0.185, 'accept': 0.021},
         {'name': 'gpu9b-graph-per-size', 'beta': 0.065, 'reject': 0.0, 'reshape': 0.0, 'accept': 0.021},
         {'beta': 0.1, 'reject': 0.0, 'reshape': 0.0}]


def part(row_id):
    g = zlib.crc32(row_id.encode()) % 8
    return 'cal' if g < 2 else 'test' if g < 4 else 'table'


def own_rows(path):
    rows = []
    for line in open(path):
        if not line.endswith('\n'):
            break                                       # a row gen_code.py is still writing
        r = json.loads(line)
        r['amax'] = np.array(r['answer_ids'], np.int32)     # temperature 0: what it wrote is its top choice
        r['source'] = 'python' if r['lang'] == 'python' else 'other-languages'
        rows.append(r)
    return rows


def other_rows(dataset, tokenizer, like, skip):
    """The set's own solutions as rows, in the chat format of `like` (a generated row). Prompts in `skip` are left out."""
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(tokenizer)
    pre, suf, eos = like['prompt_ids'][:3], like['prompt_ids'][-9:], like['prompt_ids'][-9]
    raw = [(f'oss-{n}', json.loads(line)) for n, line in enumerate(open(dataset))]
    raw = [(i, r) for i, r in raw if i not in skip]
    problems = tok.encode_batch([r['problem'] for _, r in raw], add_special_tokens=False)
    solutions = tok.encode_batch([r['solution'] for _, r in raw], add_special_tokens=False)
    rows = []
    for (i, r), p, s in zip(raw, problems, solutions):
        if len(p.ids) + len(s.ids) > 6000:
            continue
        answer = np.array(s.ids + [eos], np.int32)
        rows.append({'id': i, 'source': 'other', 'thinking': False, 'prompt_ids': pre + p.ids + suf,
                     'answer_ids': answer, 'amax': answer})
    return rows


def fenced(rows, tokenizer):
    """Per answer, per token: is it inside a ``` block? The fence tokens themselves count as outside."""
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(tokenizer)
    out = []
    for r in rows:
        text, inside = '', []
        for t in r['answer_ids']:
            before = text.count('```') % 2
            text += tok.decode([int(t)], skip_special_tokens=False)
            inside.append(before == 1 and text.count('```') % 2 == 1)
        out.append(np.array(inside, bool))
    return out


def first_guess(static, calib, test, inreq, K, where=None):
    """Share of test positions where the first guess is the model's token: table, request, both, either.
    `where` names masks over the test positions; each gets the same shares for its positions."""
    ti = ns.labelled(test)
    C, avail = test.context(ti, K)
    cands = ns.candidates(C, avail, test.seq[ti], test.pos[ti], static, inreq, K)
    truth, right = test.amax[ti], {}
    for name, kinds in (('table', ('static',)), ('request', ('context',)), ('both', ('static', 'context'))):
        some = [c for c in cands if c[0] in kinds]
        if some:
            token, conf = ns.choose(some, calib)
            right[name] = (conf > 0) & (token == truth)
    if 'table' in right and 'request' in right:
        right['either'] = right['table'] | right['request']
    out = {k: float(v.mean()) for k, v in right.items()}
    for name, mask in (where or {}).items():
        out[name] = dict({k: float(v[mask].mean()) for k, v in right.items()}, share_of_tokens=float(mask.mean()))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    C = '~/.cache/model-experiments/memdraft'
    ap.add_argument('--rows', required=True, help="gen_code.py's output")
    ap.add_argument('--more-rows', nargs='+', default=[], metavar='FILE',
                    help="more of gen_code.py's output, for prompts past --pool (run_gen_more.sh, run_gen_pod.py): all of it is own table text")
    ap.add_argument('--dataset', default=f'{C}/code-prompts-src/data-oss_instruct-decontaminated.jsonl')
    ap.add_argument('--tokenizer', default='models/manifold-audit/Qwen3.5-9B/tokenizer.json')
    ap.add_argument('--cache', default='target/memdraft/cache')
    ap.add_argument('--out')
    ap.add_argument('--K', type=int, default=6)
    ap.add_argument('--n-max', type=int, default=8)
    ap.add_argument('--pad-to', type=int, default=4)
    ap.add_argument('--skip-shards', action='store_true', help='leave out the tables that need the heal shards')
    ap.add_argument('--pool', type=int, default=2000, help="the first POOL prompts of gen_code.py's order can be held out")
    ap.add_argument('--min-count', type=int, default=1, help='tables propose only contexts seen at least this often')
    ap.add_argument('--tables', nargs='+', metavar='NAME', help='only these tables')
    ap.add_argument('--export', metavar='NAME=PREFIX', help='write table NAME as PREFIX.mdt for the llama.cpp drafter, '
                    'with the confidences the calibration answers give it')
    ap.add_argument('--floor', type=float, default=0.1, help='with --export: drop entries whose confidence is below this')
    ap.add_argument('--prompts-out', help='write test prompts for live_bench.py here')
    ap.add_argument('--prompts-count', type=int, default=24)
    args = ap.parse_args()
    K = args.K

    dataset = os.path.expanduser(args.dataset)
    pool = gen_code.pick(dataset, args.pool, gen_code.SEED, gen_code.MAX_CHARS)
    held_ids = {p['id'] for p in pool if part(p['id']) != 'table'}
    if args.prompts_out:
        some = [p for p in pool if part(p['id']) == 'test'][:args.prompts_count]
        Path(args.prompts_out).write_text(''.join(json.dumps(
            {'id': p['id'], 'domain': 'python' if p['lang'] == 'python' else 'other-languages', 'lang': p['lang'],
             'prompt': p['prompt']}) + '\n' for p in some))
        ns.log(f'{args.prompts_out}: {len(some)} test prompts')

    own = own_rows(args.rows)
    held = [r for r in own if part(r['id']) != 'table']
    if any(r['id'] not in held_ids for r in held):
        raise SystemExit('answers reach past --pool: raise it, or their prompts are not held out of the other-code tables')
    cal = ns.Data([r for r in held if part(r['id']) == 'cal' and r['finished']])
    test = ns.Data([r for r in held if part(r['id']) == 'test' and r['finished']])
    more = [r for path in args.more_rows for r in own_rows(path)]
    if len({r['id'] for r in more}) < len(more):
        raise SystemExit('--more-rows has two answers to the same prompt')
    if {r['id'] for r in more} & {p['id'] for p in pool}:
        raise SystemExit('--more-rows has answers to prompts inside --pool: those belong in --rows')
    own_table = [r for r in own if part(r['id']) == 'table'] + more
    tokens = lambda rows: int(sum(len(r['answer_ids']) for r in rows))
    ns.log(f'{len(own)} answers, {tokens(own):,} tokens; {sum(not r["finished"] for r in own)} hit the length limit. '
           + (f'{len(more)} more answers past the pool, {tokens(more):,} tokens. ' if more else '') +
           f'calibration {len(cal.start)} answers / {len(ns.labelled(cal)):,} tokens, test {len(test.start)} / '
           f'{len(ns.labelled(test)):,}, own table text {len(own_table)} / {tokens(own_table):,}')

    other = other_rows(dataset, args.tokenizer, own[0], held_ids)
    by_id = {r['id']: r for r in other}
    rebuilt = sum(by_id[r['id']]['prompt_ids'] == r['prompt_ids'] for r in own_table if r['id'] in by_id)
    ns.log(f'other: {len(other):,} solutions, {tokens(other):,} tokens; prompts rebuilt here equal the generated ones for '
           f'{rebuilt} of {len(own_table)}')
    same = [by_id[r['id']] for r in own_table if r['id'] in by_id]
    shards = [] if args.skip_shards else [r for s in SHARDS for r in ns.load_shard(s, None, args.cache) if r['finished']]

    tables = [('own code', own_table), ('own code, half', own_table[::2]), ('own code, quarter', own_table[::4]),
              ('other code, same prompts', same), ('other code, all', other)]
    if shards:
        tables = [('shipped text (no code)', shards)] + tables + [
            ('shipped + own', shards + own_table), ('shipped + other all', shards + other),
            ('shipped + own + other all', shards + own_table + other)]
    else:
        tables.append(('own + other all', own_table + other))
    if args.tables:
        tables = [t for t in tables if t[0] in args.tables]
    export_name, _, export_prefix = (args.export or '').partition('=')
    if args.export and export_name not in [t[0] for t in tables]:
        raise SystemExit(f'--export: no table named {export_name!r}')

    test_rows = [r for r in held if part(r['id']) == 'test' and r['finished']]       # the order ns.Data(test) has them in
    ti = ns.labelled(test)
    seq = test.seq[ti]
    first = np.r_[0, np.cumsum([len(r['answer_ids']) for r in test_rows])]
    in_code = np.concatenate(fenced(test_rows, args.tokenizer))[first[seq] + ti - test.ans[seq]]
    where = {'in code blocks': in_code, 'outside code blocks': ~in_code}
    ns.log(f'{in_code.mean():.0%} of the test tokens are inside code blocks')
    result = {'args': vars(args), 'answers': len(own), 'more_answers': len(more), 'held_out_prompts': len(held_ids), 'test_tokens': int(len(ns.labelled(test))), 'configs': {}}
    inreq_test = ns.InRequest(test, K)

    def record(name, policy, static, table_tokens):
        rows, calib = ns.evaluate(static, cal, test, 'greedy', policy, K, args.n_max, COSTS, args.pad_to)
        guess = first_guess(static, calib, test, inreq_test if policy != 'static' else None, K, where)
        result['configs'][f'{name} | {policy}'] = {'table': name, 'policy': policy, 'table_tokens': table_tokens,
                                                    'first_guess_right': guess, 'by_cost': rows}
        g = rows['gpu9b']
        ns.log(f"{name:28s} {policy:8s} table {table_tokens / 1e6:6.2f}M  first guess "
               + ' '.join(f'{k} {v:.3f}' for k, v in guess.items() if k not in where)
               + f"  | gpu9b {g['all']['speedup']:.3f}x ({g['rule'][0]} {g['rule'][1]}, {g['all']['tokens_per_step']:.2f} tok/step, "
                 f"acc {g['all']['acceptance']:.0%})  graph-per-size {rows['gpu9b-graph-per-size']['all']['speedup']:.3f}x  "
                 f"b0.1 {rows['b0.1']['all']['speedup']:.3f}x  "
               + ' '.join(f"{d.split('/')[0]} {v['speedup']:.3f}x" for d, v in g['domains'].items()))
        for name in where:
            ns.log(f"{'':28s} {name} ({guess[name]['share_of_tokens']:.0%} of tokens): first guess "
                   + ' '.join(f'{k} {v:.3f}' for k, v in guess[name].items() if k != 'share_of_tokens'))
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(result, indent=1))
        return calib

    record('no table', 'context', None, 0)
    for name, rows in tables:
        t0 = time.time()
        train = ns.Data(rows)
        idx = ns.labelled(train)
        static = ns.Static.build(train, idx, train.amax[idx], K)
        static.min_count = args.min_count
        del train
        for policy in ('static', 'both'):
            calib = record(name, policy, static, len(idx))
        if name == export_name:                      # confidences as for 'both': the table beside the in-request lookup
            kept, dropped = ns.export_table(static, calib, f'{export_prefix}.mdt', args.floor)
            size = Path(f'{export_prefix}.mdt').stat().st_size
            ns.log(f'{export_prefix}.mdt: {kept:,} entries from {len(idx):,} tokens, {dropped:,} dropped for space, {size / 2**20:.1f} MiB')
            Path(f'{export_prefix}.json').write_text(json.dumps(
                {'args': vars(args), 'table': name, 'train_tokens': int(len(idx)), 'entries': kept, 'dropped': dropped, 'bytes': size}, indent=1))
        ns.log(f'   ({name}: {time.time() - t0:.0f} s)')


if __name__ == '__main__':
    main()
