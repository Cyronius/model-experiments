#!/usr/bin/env python3
"""Ship the code table and the reranker to the llama.cpp drafter, and check the C++ against numpy.

The table is the largest one of rerank7.py: the shipped text, the 9B's own code (yesterday's
answers outside the test part, the 4,200 written on this machine, the pod's 68,985) and the
gpt-3.5 solutions. The reranker is one of rerank7.py's (default: fine-tuned with all of the pod's
answers). Writes, for a minimum count M (contexts seen fewer times are left out of the file):
  PREFIX.mdt     the table, version 2 (memdraft.h: confidences as floats, and the count cells the
                 reranker reads)
  PREFIX.mdr     the reranker: its weights, and per token its class, frequency and the two
                 embedding projections (so the drafter needs no embedding matrix)
  PREFIX.json    offline scores on the code test answers (pad 2 and 3, the 9B's GPU step costs):
                 the drafter and the reranker with the full table, with tables cut at counts 2, 3
                 and 5, and read back from the files; and the C++ check
--check compiles replay.cpp, has it draft at every test position with the files, and requires
its drafts to equal the numpy reranker's at nearly every position (float sums in another order
can tip a near-tie), then counts steps with both.

Run from the repo root with target/manifold-audit/env/bin/python under a memory limit
(systemd-run --user --scope -p MemoryMax=30G; peak about 20 GB):
    export_reranker.py --pod-rows target/memdraft/code/rows-9b-pod.jsonl --out target/memdraft/code/9b-code-pod-c3 --check
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
SCRATCH = HERE.parent / 'target/memdraft/code/rerank-scratch'
sys.path.insert(0, str(SCRATCH))
sys.path.insert(0, str(HERE))
import code_study as cs           # noqa: E402
import gen_code                   # noqa: E402
import ngram_study as ns          # noqa: E402
import rerank as rr               # noqa: E402
import rerank2 as r2              # noqa: E402
import rerank7 as r7              # noqa: E402

K = rr.K
BEST = 'embeddings, learned from 512'
HIDDEN = (512, 256, 128)
DEFAULT_MODEL = 'full table, reranker fine-tuned with all 68985 of the pod\'s answers too'


def chain_tokens(data, idx, inreq, static, pick, depth):
    """rr.chain's drafts at every position: [positions, depth]."""
    C, avail = data.context(idx, K)
    seq, pos = data.seq[idx], data.pos[idx]
    tokens = np.full((len(idx), depth), -1, np.int32)
    for d in range(depth):
        token, conf = pick(ns.candidates(C, avail, seq, pos, static, inreq, K), C, d)
        token = np.where(conf > 0, token, C[:, 0])
        tokens[:, d] = token
        C = np.concatenate([token[:, None], C[:, :-1]], 1)
        avail = np.minimum(avail + 1, K)
    return tokens


def write_reranker(model, emb, cls, logfreq, path):
    sd = {k: v.detach().float() for k, v in model.state_dict().items()}
    inp, out = (torch.from_numpy(e) for e in emb)
    with torch.no_grad():
        p_in, p_out = inp @ sd['p_in.weight'].T, out @ sd['p_out.weight'].T
    V, d = p_in.shape
    n_x, n_cls = model.n_in, int(cls.max()) + 1
    assert n_x == 8 * rr.S + 5 + 3 * rr.NC + 4 and n_cls <= rr.NC and len(cls) == len(logfreq) == V
    h1, h2, h3 = HIDDEN
    assert sd['net.0.weight'].shape == (h1, n_x + 7 * d)
    blobs = [cls.astype(np.uint8).tobytes(), logfreq.astype(np.float32).tobytes(),
             p_out.numpy().astype(np.float32).tobytes(), p_in.numpy().astype(np.float32).tobytes(),
             sd['net.0.weight'].T.contiguous().numpy().tobytes(), sd['net.0.bias'].numpy().tobytes(),
             sd['net.2.weight'].T.contiguous().numpy().tobytes(), sd['net.2.bias'].numpy().tobytes(),
             sd['net.4.weight'].T.contiguous().numpy().tobytes(), sd['net.4.bias'].numpy().tobytes(),
             sd['net.6.weight'][0].numpy().tobytes(), sd['net.6.bias'].numpy().tobytes()]
    import struct
    with open(path, 'wb') as f:
        f.write(struct.pack('<8s10I16x', b'MDRERANK', 1, V, rr.S, rr.NC, 6, d, h1, h2, h3, n_x))
        f.write(blobs[0] + b'\0' * (-len(blobs[0]) % 8))
        for b in blobs[1:]:
            f.write(b)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--pod-rows', required=True)
    ap.add_argument('--model', default='target/memdraft/code/rerank-more/pod.pt')
    ap.add_argument('--key', default=DEFAULT_MODEL, help='which reranker in --model')
    ap.add_argument('--min-count', type=int, default=3)
    ap.add_argument('--out', required=True, help='output prefix')
    ap.add_argument('--check', action='store_true')
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    record = {'args': vars(args)}

    def save():
        Path(f'{out}.json').write_text(json.dumps(record, indent=1))

    pool = gen_code.pick(rr.dataset, 2000, gen_code.SEED, gen_code.MAX_CHARS)
    held_ids = {p['id'] for p in pool if cs.part(p['id']) != 'table'}
    own = cs.own_rows('target/memdraft/code/rows-9b.jsonl')
    more = cs.own_rows('target/memdraft/code/rows-9b-more.jsonl')
    pod = r7.light(args.pod_rows)
    test = ns.Data([r for r in own if cs.part(r['id']) == 'test' and r['finished']])
    own_table = [r for r in own if cs.part(r['id']) == 'table']
    cal_rows = [r for r in own if cs.part(r['id']) == 'cal']
    other = cs.other_rows(rr.dataset, rr.tokenizer, own[0], held_ids)
    chat = [r for s in cs.SHARDS for r in ns.load_shard(s, None, 'target/memdraft/cache') if r['finished']]
    cls = rr.token_classes()
    emb512 = rr.embeddings_pca(512)
    emb = {'pca': None, 'full': None, 'p512': tuple(torch.from_numpy(e) for e in emb512)}

    cal = ns.Data([r for r in cal_rows if r['finished']])
    ci = ns.labelled(cal)
    inreq_cal = ns.InRequest(cal, K)
    ti = ns.labelled(test)
    inreq_test = ns.InRequest(test, K)

    static, counts, n = r2.table(chat + own_table + more + pod + other)
    calib = ns.Calibration(K)
    calib.add(r2.proposals(cal, ci, static, inreq_cal)[1], cal.amax[ci])
    logfreq = (np.log1p(counts) / 16).astype(np.float32)
    n_in = rr.build(r2.proposals(cal, ci[:1000], static, inreq_cal)[1], calib, cal.context(ci[:1000], K)[0], logfreq, cls)[2].shape[1]
    model = r2.Net(n_in, 'p512', emb, HIDDEN)
    model.load_state_dict(torch.load(args.model)[args.key])
    model.eval()
    record['table_tokens'] = n
    ns.log(f'table: {n:,} tokens; reranker {args.key!r}')

    def reranked(conf_of):
        def pick(cands, C, d):
            ii, tok, X, ids = rr.build(cands, conf_of, C, logfreq, cls)
            with torch.no_grad():
                p = np.concatenate([torch.sigmoid(model(torch.from_numpy(X[b:b + 16384]), torch.from_numpy(ids[b:b + 16384])).double()).numpy()
                                    for b in range(0, len(X), 16384)]) if len(X) else np.zeros(0)
            token, conf = np.full(len(C), -1, np.int32), np.zeros(len(C), np.float64)
            np.maximum.at(conf, ii, p)
            win = p >= conf[ii]
            token[ii[win]] = tok[win]
            has = np.zeros(len(C), bool)
            has[ii] = True
            return token, np.where(has, np.maximum(conf, 1e-30), 0)    # nothing proposed: memdraft.h repeats the last token
        return pick

    def scored(name, table, conf_of):
        row = {}
        for who, pick in (('drafter', lambda cands, C, d: ns.choose(cands, conf_of)), ('reranker', reranked(conf_of))):
            r = rr.chain(test, ti, inreq_test, table, pick)
            row[who] = {p: r[p] for p in ('pad 2', 'pad 3')}
            row[who]['first_guess_right'] = r['first_guess_right']
        record.setdefault('offline', {})[name] = row
        ns.log(f"{name:44s} drafter pad 2 {row['drafter']['pad 2']['speedup']:.3f}x pad 3 {row['drafter']['pad 3']['speedup']:.3f}x | "
               f"reranker pad 2 {row['reranker']['pad 2']['speedup']:.3f}x pad 3 {row['reranker']['pad 3']['speedup']:.3f}x")
        save()

    for m in sorted({1, 2, 3, 5, args.min_count}):
        static.min_count = m
        scored(f'contexts seen at least {m} times', static, calib)
    static.min_count = args.min_count
    kept, dropped = ns.export_table2(static, calib, f'{out}.mdt')
    write_reranker(model, emb512, cls, logfreq, f'{out}.mdr')
    record['files'] = {'entries': kept, 'dropped_for_space': dropped,
                       'table_bytes': Path(f'{out}.mdt').stat().st_size, 'reranker_bytes': Path(f'{out}.mdr').stat().st_size}
    ns.log(f"{out}.mdt: {kept:,} entries ({dropped:,} dropped for space), {record['files']['table_bytes'] / 2**20:.0f} MiB; "
           f"{out}.mdr: {record['files']['reranker_bytes'] / 2**20:.0f} MiB")
    del static
    shipped = ns.Exported2(f'{out}.mdt')
    scored('the files, read back', shipped, shipped)

    if args.check:
        ns.write_stream(test, f'{out}.stream')
        binary = out.parent / 'replay'
        subprocess.run(['g++', '-O2', '-std=c++17', '-Wall', '-Wextra', '-o', str(binary), str(HERE / 'replay.cpp')], check=True)
        depth = 3
        want = chain_tokens(test, ti, inreq_test, shipped, reranked(shipped), depth)
        subprocess.run([str(binary), f'{out}.mdt', f'{out}.stream', '--pad', '1', '--n-max', str(depth),
                        '--reranker', f'{out}.mdr', '--dump', f'{out}.dump'], check=True, capture_output=True)
        got = np.fromfile(f'{out}.dump', np.int32).reshape(-1, depth)
        assert got.shape == want.shape, (got.shape, want.shape)
        same = (got == want).mean(0)
        whole = float((got == want).all(1).mean())
        record['check'] = {'positions': len(got), 'same_by_depth': same.tolist(), 'same_chain': whole}
        ns.log(f"C++ against numpy at {len(got):,} positions: same token at depth 1..{depth}: "
               + ', '.join(f'{s:.5f}' for s in same) + f'; whole chain {whole:.5f}')
        record['replay'] = {}
        for label, extra in (('drafter', []), ('reranker', ['--reranker', f'{out}.mdr'])):
            for pad in (2, 3):
                done = subprocess.run([str(binary), f'{out}.mdt', f'{out}.stream', '--pad', '1', '--n-max', str(pad)] + extra,
                                      check=True, capture_output=True, text=True)
                r = json.loads(done.stdout)
                record['replay'][f'{label}, pad {pad}'] = r
                ns.log(f"replay {label:8s} pad {pad}: {r['tokens'] / r['steps']:.3f} tokens/step, "
                       f"{r['draft_ns_per_step'] / 1000:.1f} us per draft call")
        save()
        if whole < 0.99:
            raise SystemExit('replay.cpp disagrees with the numpy reranker at more than 1% of positions')
    save()
    ns.log('done')


if __name__ == '__main__':
    main()
