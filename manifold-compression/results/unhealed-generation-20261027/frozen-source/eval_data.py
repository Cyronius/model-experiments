"""Fetch pinned public evaluation data and convert it to tf_metrics JSONL.

Each dataset is downloaded at a fixed commit into its own directory under
models/manifold-compression/eval-data/ (outside git). Conversion writes the
normalized files that tf_metrics.py prepare reads with --mc / --chat, plus a
provenance record of commits, file hashes, selection seeds and counts.

    dibs run network -- python eval_data.py fetch
    python eval_data.py convert
"""
import argparse
import hashlib
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
RESULTS = ROOT / 'results'
DATA = REPO / 'models/manifold-compression/eval-data'
FILES = {
    'mmlu': ('cais/mmlu', 'c30699e8356da336a370243923dbaf21066bb9fe', ['all/test-00000-of-00001.parquet']),
    'arc': ('allenai/ai2_arc', '210d026faf9955653af8916fad021475a3f00453',
            ['ARC-Challenge/test-00000-of-00001.parquet', 'ARC-Easy/test-00000-of-00001.parquet']),
    'hellaswag': ('Rowan/hellaswag', '218ec52e09a7e7462a5400043bb9a69a41d06b76',
                  ['data/validation-00000-of-00001.parquet']),
    # The revision every UltraChat calibration and validation row in this project came from.
    'ultrachat': ('HuggingFaceH4/ultrachat_200k', '8049631c405ae6576f93f445c6b8166f76f5505a',
                  ['data/test_sft-00000-of-00001-f7dfac4afe5b93f4.parquet']),
    'humaneval': ('openai/openai_humaneval', '7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544',
                  ['openai_humaneval/test-00000-of-00001.parquet']),
    'wikitext': ('Salesforce/wikitext', 'b08601e04326c79dfdd32d625aee71d232d685c3',
                 ['wikitext-2-raw-v1/test-00000-of-00001.parquet']),
}
# Every file that holds UltraChat rows used for calibration, validation or generation so far.
USED_CHAT = ['scale-probe-20261024/data.json', 'local-projection-20261025/data.json',
             'local-projection-20261025/extended-calibration-source.json', 'broad-more-20261025/data.json']
SEED = 26102701


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_jsonl(rows, path):
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    return {'file': path.name, 'rows': len(rows), 'sha256': sha(path)}


def fetch():
    from huggingface_hub import hf_hub_download
    record = {}
    for name, (repo, revision, files) in FILES.items():
        target = DATA / name
        target.mkdir(parents=True, exist_ok=True)
        record[name] = {'repo': repo, 'revision': revision, 'files': {}}
        for file in files:
            path = hf_hub_download(repo, file, repo_type='dataset', revision=revision, local_dir=target)
            record[name]['files'][file] = sha(path)
    (DATA / 'fetch.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))


def table(name, file):
    import pyarrow.parquet as pq
    path = DATA / name / file
    expected = json.loads((DATA / 'fetch.json').read_text())[name]['files'][file]
    if sha(path) != expected:
        raise ValueError(f'{path} differs from the fetched file')
    return pq.read_table(path).to_pylist()


def sample(rows, count, seed, key=None):
    """Deterministic sample; with key, an equal share from every stratum."""
    rng = random.Random(seed)
    if key is None:
        rows = list(rows)
        rng.shuffle(rows)
        return rows[:count]
    strata = {}
    for r in rows:
        strata.setdefault(key(r), []).append(r)
    out = []
    for k in sorted(strata):
        group = strata[k]
        rng.shuffle(group)
        out += group[:count]
    return out


def hellaswag_text(text):
    # The preprocessing used by the common evaluation harnesses.
    text = text.strip().replace(' [title]', '. ')
    return re.sub(r'\[.*?\]', '', text).replace('  ', ' ')


def used_chat_keys():
    ids, prompts = set(), set()
    for path in USED_CHAT:
        data = json.loads((RESULTS / path).read_text())
        for value in data.values():
            if not isinstance(value, list):
                continue
            for row in value:
                if isinstance(row, dict) and 'messages' in row:
                    ids.add(row.get('id', ''))
                    prompts.add(row['messages'][0]['content'].strip().casefold())
    return ids, prompts


def convert(args):
    out = DATA / 'converted'
    out.mkdir(exist_ok=True)
    record = {'fetch': json.loads((DATA / 'fetch.json').read_text()), 'seed': SEED, 'outputs': {}}

    mmlu = sample(table('mmlu', 'all/test-00000-of-00001.parquet'), args.mmlu_per_subject, SEED, lambda r: r['subject'])
    record['outputs']['mmlu'] = write_jsonl(
        [{'question': r['question'], 'choices': list(r['choices']), 'answer': int(r['answer']), 'subject': r['subject']}
         for r in mmlu], out / 'mmlu.jsonl')

    for split, count in (('ARC-Challenge', args.arc_challenge), ('ARC-Easy', args.arc_easy)):
        rows = sample(table('arc', f'{split}/test-00000-of-00001.parquet'), count, SEED)
        items = [{'question': r['question'], 'choices': list(r['choices']['text']),
                  'answer': list(r['choices']['label']).index(r['answerKey']), 'subject': r['id']} for r in rows]
        record['outputs'][split] = write_jsonl(items, out / f'{split.lower()}.jsonl')

    rows = sample(table('hellaswag', 'data/validation-00000-of-00001.parquet'), args.hellaswag, SEED)
    items = [{'question': hellaswag_text(f"{r['activity_label']}: {r['ctx_a']} {r['ctx_b'].capitalize()}"),
              'choices': [hellaswag_text(e) for e in r['endings']], 'answer': int(r['label']),
              'subject': r['activity_label']} for r in rows]
    record['outputs']['hellaswag'] = write_jsonl(items, out / 'hellaswag.jsonl')

    ids, prompts = used_chat_keys()
    chats = []
    for r in sample(table('ultrachat', FILES['ultrachat'][2][0]), 10 ** 9, SEED):
        m = r['messages']
        key = f"test_sft-{r['prompt_id']}"
        if len(m) < 2 or m[0]['role'] != 'user' or key in ids or m[0]['content'].strip().casefold() in prompts:
            continue
        # One or two complete exchanges, ending on an assistant turn, as in scale_data.py.
        chats.append({'id': key, 'messages': [{'role': x['role'], 'content': x['content']} for x in m[:4 if len(m) >= 4 else 2]]})
        if len(chats) == args.ultrachat:
            break
    record['outputs']['ultrachat'] = write_jsonl(chats, out / 'ultrachat-heldout.jsonl')

    code = [{'id': r['task_id'].replace('/', '-'),
             'messages': [{'role': 'user', 'content': 'Complete this Python function.\n\n```python\n' + r['prompt'] + '```'},
                          {'role': 'assistant', 'content': '```python\n' + r['prompt'] + r['canonical_solution'] + '```'}]}
            for r in table('humaneval', 'openai_humaneval/test-00000-of-00001.parquet')]
    record['outputs']['humaneval'] = write_jsonl(code, out / 'humaneval.jsonl')

    # WikiText rows are lines; an article starts at a single-'=' heading.
    articles, current = [], []
    for line in (r['text'] for r in table('wikitext', 'wikitext-2-raw-v1/test-00000-of-00001.parquet')):
        if re.fullmatch(r' = [^=].* = \n', line) and current:
            articles.append(''.join(current))
            current = []
        current.append(line)
    articles.append(''.join(current))
    prose = [{'id': f'wikitext-{i}', 'text': a.strip()} for i, a in enumerate(articles) if a.strip()]
    record['outputs']['wikitext'] = write_jsonl(prose, out / 'wikitext.jsonl')
    record['excluded_chat'] = {'files': USED_CHAT, 'ids': len(ids), 'prompts': len(prompts)}
    record['counts'] = {k: v for k, v in vars(args).items() if k != 'stage'}
    (out / 'convert.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record['outputs'], indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('stage', choices=['fetch', 'convert'])
    ap.add_argument('--mmlu-per-subject', type=int, default=20)
    ap.add_argument('--arc-challenge', type=int, default=600)
    ap.add_argument('--arc-easy', type=int, default=400)
    ap.add_argument('--hellaswag', type=int, default=600)
    ap.add_argument('--ultrachat', type=int, default=320)
    args = ap.parse_args()
    fetch() if args.stage == 'fetch' else convert(args)


if __name__ == '__main__':
    main()
