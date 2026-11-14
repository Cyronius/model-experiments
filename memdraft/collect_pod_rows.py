#!/usr/bin/env python3
"""Put the rows a pod wrote (run_gen_pod.py) into one file, one answer per prompt.

The pod writes pieces, rows-<from>-<to>.jsonl, and where its two generators met, a few prompts
have two answers: the first one written is kept. Prompts without an answer are counted.

    python memdraft/collect_pod_rows.py target/memdraft/code/pod/gen-XXXX --out target/memdraft/code/rows-9b-pod.jsonl
"""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('job', help="the job's directory under target/memdraft/code/pod/")
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    job = Path(args.job)
    want = json.loads((job / 'bundle-manifest.json').read_text())['prompts']
    seen, twice, tokens, unfinished, lines = set(), 0, 0, 0, 0
    with open(args.out + '.tmp', 'w') as out:
        for piece in sorted((job / 'rows').glob('rows-*.jsonl')):
            for line in piece.open():
                if not line.endswith('\n'):
                    break                                   # a row that was still being copied
                row = json.loads(line)
                if row['id'] in seen:
                    twice += 1
                    continue
                seen.add(row['id'])
                tokens += len(row['answer_ids'])
                unfinished += not row['finished']
                out.write(line)
                lines += 1
    Path(args.out + '.tmp').replace(args.out)
    print(f"{lines} answers, {tokens:,} tokens ({unfinished} hit the length limit) in {args.out}; "
          f"{twice} second answers left out; {want['count'] - lines} of the pod's {want['count']} prompts have no answer")


if __name__ == '__main__':
    main()
