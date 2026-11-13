"""Build the frozen healing corpus: nine in-mix domains, two held-out domains, benchmark-clean.

In-mix (trained on): chat (UltraChat), instruction (K2-Horizon SFT-Reasoning sample), math (MATH
train), math_word (GSM8K train), reasoning (OpenThoughts-114k), wiki (Wikipedia 20231101.en),
science (arXiv abstracts), code (the local Python standard library), code_cpp (mlir-aie C++).
The chat-shaped domains go through the teacher's chat template. instruction, math, math_word and
reasoning carry non-empty <think> blocks that close into an answer and <|im_end|>; a stream with
only empty think blocks taught an earlier Qwen3.5 student to loop instead of stopping.

Held out (never trained on): biomed (TREC-COVID corpus) and argument (Touche-2020 corpus). They
supply the stopping-rule validation split and the final test split.

Items are assigned to splits by a salted hash of their identity, so an item (whole problem,
article, file or conversation) never straddles splits. Items sharing a word n-gram with a
benchmark question (GSM8K test, ARC-Easy/Challenge validation+test, MMLU test) are dropped
before tokenization. Hub sources are pinned by revision and read in bounded windows; local
sources are hashed file by file in the manifest.

  python prepare_data.py --tokenizer <teacher> --output data/mix-2048          # fetch + build
  python prepare_data.py --tokenizer <teacher> --output data/mix-2048b --raw data/mix-2048/raw
"""
import argparse
import gzip
import hashlib
import io
import json
import re
import unicodedata
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
SALT = 'healing-schedules-mix-v1'

HUB = {
    'gsm8k': ('openai/gsm8k', '740312add88f781978c0658806c59bc2815b9866'),
    'math': ('EleutherAI/hendrycks_math', '21a5633873b6a120296cce3e2df9d5550074f4a3'),
    'openthoughts': ('open-thoughts/OpenThoughts-114k', 'bd093c3994fd54d2390985b66988ddf282a55eb6'),
    'wiki': ('wikimedia/wikipedia', 'b04c8d1ceb2f5cd4588862100d08de323dccfbaa'),
    'arxiv': ('gfissore/arxiv-abstracts-2021', 'e4c5fbd4dec8e46a5dc869216fe1c94cc585757a'),
    'biomed': ('BeIR/trec-covid', '7e16fde3016c639c7f856e803f4bab92645562c4'),
    'argument': ('BeIR/webis-touche2020', '7ebed360c2743a0024cb96a404eb89e27c931ebc'),
    'arc': ('allenai/ai2_arc', '210d026faf9955653af8916fad021475a3f00453'),
    'mmlu': ('cais/mmlu', 'c30699e8356da336a370243923dbaf21066bb9fe'),
}
MATH_CONFIGS = ('algebra', 'counting_and_probability', 'geometry', 'intermediate_algebra',
                'number_theory', 'prealgebra', 'precalculus')
# Whole small files; bounded leading row groups (or lines) of large ones.
WHOLE = {
    'gsm8k-train.parquet': ('gsm8k', 'main/train-00000-of-00001.parquet'),
    'gsm8k-test.parquet': ('gsm8k', 'main/test-00000-of-00001.parquet'),
    'arc-easy-validation.parquet': ('arc', 'ARC-Easy/validation-00000-of-00001.parquet'),
    'arc-easy-test.parquet': ('arc', 'ARC-Easy/test-00000-of-00001.parquet'),
    'arc-challenge-validation.parquet': ('arc', 'ARC-Challenge/validation-00000-of-00001.parquet'),
    'arc-challenge-test.parquet': ('arc', 'ARC-Challenge/test-00000-of-00001.parquet'),
    'mmlu-test.parquet': ('mmlu', 'all/test-00000-of-00001.parquet'),
    **{f'math-{c}-train.parquet': ('math', f'{c}/train-00000-of-00001.parquet') for c in MATH_CONFIGS},
}
WINDOWS = {  # file: (source, path, columns, rows)
    'openthoughts.jsonl': ('openthoughts', 'metadata/train-00000-of-00012.parquet',
                           ['problem', 'deepseek_reasoning', 'deepseek_solution', 'domain'], 4000),
    'wiki.jsonl': ('wiki', '20231101.en/train-00020-of-00041.parquet', ['id', 'title', 'text'], 8000),
    'biomed.jsonl': ('biomed', 'corpus/corpus-00000-of-00001.parquet', ['_id', 'title', 'text'], 8000),
    'argument.jsonl': ('argument', 'corpus/corpus-00000-of-00001.parquet', ['_id', 'title', 'text'], 8000),
}
ARXIV = ('arxiv.jsonl', 'arxiv-abstracts.jsonl.gz', 250_000, 10_000)  # file, path, skip lines, lines

LOCAL = {
    'chat': REPO / 'target/manifold-audit/ultrachat/data/train_sft-00000-of-00003-a3ecf92756993583.parquet',
    'instruction': REPO / 'target/manifold-healing/horizon-raw',
    'code': Path('/usr/lib64/python3.14'),
    'code_cpp': REPO.parent / 'openflowlm-next/third_party/mlir-aie',
}
CPP_DIRS = ('lib', 'aie_kernels', 'include', 'tools', 'runtime_lib', 'aie_runtime_lib')
SKIP_PY = {'site-packages', 'test', 'tests', 'idle_test', '__pycache__', 'encodings', 'pydoc_data'}

IN_MIX = ('chat', 'instruction', 'math', 'math_word', 'reasoning', 'wiki', 'science', 'code', 'code_cpp')
HELD_OUT = ('biomed', 'argument')
DOMAINS = IN_MIX + HELD_OUT
CHAT_KINDS = {'chat', 'instruction', 'math', 'math_word', 'reasoning'}
# Chat items longer than their cap are skipped (a cut think block would teach not stopping);
# plain text is truncated to its cap.
CAP = {'reasoning': 8192, 'instruction': 6144, 'code': 16384, 'code_cpp': 16384}
DEFAULT_CAP = 4096
MIN_TOKENS = 32
MAX_FILE_BYTES = 200_000
# Percent buckets of the salted item hash, per in-mix domain; the rest is train.
IN_MIX_BUCKETS = (('controller', 3), ('test_in_domain', 3), ('calibration', 1))
NGRAM = 13
SHORT_NGRAM = 8


def sha(data):
    return hashlib.sha256(data if isinstance(data, bytes) else data.encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def bucket(domain, ident):
    n = int(sha(f'{SALT}:split:{domain}:{ident}')[:8], 16) % 100
    if domain in HELD_OUT:
        return 'validation' if n < 50 else 'test'
    edge = 0
    for name, share in IN_MIX_BUCKETS:
        edge += share
        if n < edge:
            return name
    return 'train'


def order_key(domain, ident):
    return sha(f'{SALT}:order:{domain}:{ident}')


# ---------------------------------------------------------------- fetching (network)

def fetch(raw):
    from huggingface_hub import HfFileSystem
    import pyarrow.parquet as pq
    fs = HfFileSystem(token=False)
    raw.mkdir(parents=True, exist_ok=True)

    def remote(source, path):
        repo, revision = HUB[source]
        return fs.open(f'datasets/{repo}@{revision}/{path}', 'rb', block_size=1 << 20, cache_type='readahead')

    for name, (source, path) in WHOLE.items():
        if not (raw / name).exists():
            with remote(source, path) as f:
                data = f.read()
            (raw / f'{name}.tmp').write_bytes(data)
            (raw / f'{name}.tmp').replace(raw / name)
    for name, (source, path, columns, rows) in WINDOWS.items():
        if (raw / name).exists():
            continue
        out = []
        with remote(source, path) as f:
            parquet = pq.ParquetFile(f)
            for group in range(parquet.num_row_groups):
                out += parquet.read_row_group(group, columns=columns).to_pylist()
                if len(out) >= rows:
                    break
        (raw / f'{name}.tmp').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in out[:rows]))
        (raw / f'{name}.tmp').replace(raw / name)
    name, path, skip, lines = ARXIV
    if not (raw / name).exists():
        out = []
        with remote('arxiv', path) as f, gzip.open(f, 'rt') as text:
            for i, line in enumerate(text):
                if i >= skip:
                    out.append(line if line.endswith('\n') else line + '\n')
                if len(out) >= lines:
                    break
        (raw / f'{name}.tmp').write_text(''.join(out))
        (raw / f'{name}.tmp').replace(raw / name)
    record = dict(hub={k: dict(repo=r, revision=v) for k, (r, v) in HUB.items()},
                  whole={k: dict(source=s, path=p) for k, (s, p) in WHOLE.items()},
                  windows={k: dict(source=s, path=p, columns=c, rows=n) for k, (s, p, c, n) in WINDOWS.items()},
                  arxiv=dict(file=ARXIV[0], path=ARXIV[1], skip_lines=ARXIV[2], lines=ARXIV[3]),
                  sha256={p.name: file_sha(p) for p in sorted(raw.iterdir()) if p.suffix in ('.parquet', '.jsonl')})
    (raw / 'fetch.json').write_text(json.dumps(record, indent=1) + '\n')
    return record


# ---------------------------------------------------------------- items

def read_parquet(path):
    import pyarrow.parquet as pq
    return pq.read_table(path).to_pylist()


def read_jsonl(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f]


def turn(question, reasoning, content):
    return [{'role': 'user', 'content': question.strip()},
            {'role': 'assistant', 'reasoning_content': reasoning.strip(), 'content': content.strip()}]


def last_boxed(text):
    """The contents of the last \\boxed{...} in text (braces matched), or None."""
    start = text.rfind('\\boxed{')
    if start < 0:
        return None
    depth, i = 0, start + len('\\boxed')
    for j in range(i, len(text)):
        depth += {'{': 1, '}': -1}.get(text[j], 0)
        if depth == 0:
            return text[i + 1:j]
    return None


def instruction_pair(text):
    """(prompt, reasoning, response) of a Horizon instruction sample with unambiguous sections."""
    for a, b, c in (('### Instruction', '### Reasoning', '### Response'), ('# Query', '# Analysis', '# Response')):
        if not text.startswith(a + '\n'):
            continue
        if text.count('\n' + b + '\n') != 1 or text.count('\n' + c + '\n') != 1:
            return None
        prompt, rest = text[len(a):].split('\n' + b + '\n')
        reasoning, answer = rest.split('\n' + c + '\n')
        prompt, reasoning, answer = prompt.strip(), reasoning.strip(), answer.strip()
        if re.search(r'NAME_\d|<\|im_|</?think>', prompt + reasoning + answer):
            return None
        if not (10 <= len(prompt) <= 6000 and 20 <= len(answer) <= 9000 and reasoning):
            return None
        if looping(answer) or looping(reasoning):
            return None
        return prompt, reasoning, answer
    return None


def looping(text, n=8, limit=4):
    """True when some word n-gram repeats limit or more times (a degenerate generation)."""
    words = text.split()
    counts = {}
    for i in range(len(words) - n + 1):
        gram = tuple(words[i:i + n])
        counts[gram] = counts.get(gram, 0) + 1
        if counts[gram] >= limit:
            return True
    return False


def after_header(text, language):
    """Python from the first top-level definition, C++ after the last #include; None without one."""
    if language == 'python':
        start = min((i for i in (text.find('\ndef '), text.find('\nclass ')) if i >= 0), default=-1)
        return text[start + 1:] if start >= 0 else None
    lines = text.splitlines(keepends=True)
    last = max((i for i, line in enumerate(lines) if line.startswith('#include')), default=-1)
    return ''.join(lines[last + 1:]).lstrip() or None


def local_files(roots, patterns):
    found = {}
    for folder, prefix in roots:
        for pattern in patterns:
            for path in folder.rglob(pattern):
                rel = path.relative_to(folder)
                if SKIP_PY & set(rel.parts[:-1]) or path.stat().st_size > MAX_FILE_BYTES:
                    continue
                found[prefix + str(rel)] = path
    return found


def items(domain, raw, local_hashes):
    """(ident, payload) per item; payload is plain text or a list of chat messages."""
    if domain == 'chat':
        import pyarrow.parquet as pq
        for row in pq.read_table(LOCAL['chat'], columns=['prompt_id', 'messages']).to_pylist():
            roles = [m['role'] for m in row['messages']]
            if roles and roles == ['user', 'assistant'] * (len(roles) // 2):
                yield row['prompt_id'], [{'role': m['role'], 'content': m['content']} for m in row['messages']]
    elif domain == 'instruction':
        for path in sorted(LOCAL['instruction'].glob('instruction-*.jsonl')):
            for row in read_jsonl(path):
                parts = instruction_pair(row['text'])
                if parts:
                    yield f"{row['source_file']}:{row['row']}", turn(*parts)
    elif domain == 'math':
        for c in MATH_CONFIGS:
            for n, r in enumerate(read_parquet(raw / f'math-{c}-train.parquet')):
                boxed = last_boxed(r['solution'])
                if boxed is not None:
                    yield f'{c}-{n}', turn(r['problem'], r['solution'], f'The answer is $\\boxed{{{boxed}}}$.')
    elif domain == 'math_word':
        for n, r in enumerate(read_parquet(raw / 'gsm8k-train.parquet')):
            work, _, answer = r['answer'].rpartition('####')
            yield f'gsm8k-train-{n}', turn(r['question'], re.sub(r'<<[^>]*>>', '', work),
                                           f'The answer is {answer.strip()}.')
    elif domain == 'reasoning':
        for n, r in enumerate(read_jsonl(raw / 'openthoughts.jsonl')):
            if r['deepseek_reasoning'] and r['deepseek_solution']:
                yield f'openthoughts-{n}', turn(r['problem'], r['deepseek_reasoning'], r['deepseek_solution'])
    elif domain == 'wiki':
        for r in read_jsonl(raw / 'wiki.jsonl'):
            yield f"en-{r['id']}", f"{r['title']}\n\n{r['text']}"
    elif domain == 'science':
        for r in read_jsonl(raw / 'arxiv.jsonl'):
            yield f"arxiv-{r['id']}", f"{' '.join(r['title'].split())}\n\n{r['abstract'].strip()}"
    elif domain in ('code', 'code_cpp'):
        if domain == 'code':
            files, language = local_files([(LOCAL['code'], '')], ['*.py']), 'python'
        else:
            files = local_files([(LOCAL['code_cpp'] / d, f'{d}/') for d in CPP_DIRS],
                                ['*.cpp', '*.cc', '*.h', '*.hpp'])
            language = 'cpp'
        for ident, path in sorted(files.items()):
            data = path.read_bytes()
            local_hashes[f'{domain}:{ident}'] = sha(data)
            text = after_header(data.decode(errors='replace'), language)
            if text:
                yield ident, text
    elif domain in HELD_OUT:
        for r in read_jsonl(raw / f'{domain}.jsonl'):
            text = f"{r['title'].strip()}\n\n{r['text'].strip()}" if r['title'] and r['title'].strip() else r['text'].strip()
            yield r['_id'], text
    else:
        raise ValueError(domain)


# ---------------------------------------------------------------- benchmarks and contamination

def words(text):
    return re.findall(r'[a-z0-9]+', text.lower())


def benchmark_rows(raw):
    """Evaluation rows (task, id, question, choices/answer) for downstream.jsonl."""
    rows = []
    for task, split in (('arc_easy', 'validation'), ('arc_challenge', 'validation'),
                        ('arc_easy', 'test'), ('arc_challenge', 'test')):
        for r in read_parquet(raw / f"{task.replace('_', '-')}-{split}.parquet"):
            labels = list(r['choices']['label'])
            if r['answerKey'] not in labels:
                continue
            rows.append(dict(task=task, split=split, id=r['id'], question=r['question'],
                             choices=list(r['choices']['text']), answer=labels.index(r['answerKey'])))
    for n, r in enumerate(read_parquet(raw / 'mmlu-test.parquet')):
        rows.append(dict(task='mmlu', split='test', id=f"mmlu-{r['subject']}-{n}", subject=r['subject'],
                         question=r['question'], choices=list(r['choices']), answer=int(r['answer'])))
    for n, r in enumerate(read_parquet(raw / 'gsm8k-test.parquet')):
        rows.append(dict(task='gsm8k', split='test', id=f'gsm8k-test-{n}', question=r['question'],
                         answer=r['answer'].rpartition('####')[2].strip().replace(',', '')))
    return rows


class Contamination:
    """Word n-grams of benchmark questions: every 13-gram of long questions, and the whole
    question for those of 8-12 words. Shorter questions are too generic to match on."""

    def __init__(self, questions):
        self.grams = {NGRAM: set()}
        for q in questions:
            w = words(q)
            if len(w) >= NGRAM:
                self.grams[NGRAM].update(' '.join(w[i:i + NGRAM]) for i in range(len(w) - NGRAM + 1))
            elif len(w) >= SHORT_NGRAM:
                self.grams.setdefault(len(w), set()).add(' '.join(w))

    def hit(self, text):
        w = words(text)
        for n, grams in self.grams.items():
            if any(' '.join(w[i:i + n]) in grams for i in range(len(w) - n + 1)):
                return True
        return False


# ---------------------------------------------------------------- packing

def allocate(available, total):
    """Sequences per domain: equal shares, a short domain's shortfall spread over the others."""
    counts = {}
    order = sorted(available, key=lambda d: (available[d], d))
    for i, d in enumerate(order):
        counts[d] = min(available[d], (total - sum(counts.values())) // (len(order) - i))
    return {d: counts[d] for d in available}


def rows_of(stream, length):
    return np.asarray(stream[:len(stream) // length * length], np.int32).reshape(-1, length)


def build(args, raw):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    eot = tok.convert_tokens_to_ids('<|endoftext|>')
    if not isinstance(eot, int) or eot == tok.unk_token_id:
        raise ValueError('Tokenizer lacks <|endoftext|>')
    length = args.length
    bench = benchmark_rows(raw)
    dirty = Contamination([r['question'] for r in bench])
    train_quota = -(-args.train_tokens // length // len(IN_MIX)) * length * 1.15
    eval_quota = (args.eval_sequences + 2) * length  # per domain, enough for an equal share
    calib_quota = (args.calibration_sequences + 1) * length
    streams = {}  # (split, domain) -> token list
    report, idents, local_hashes = {}, {}, {}
    for domain in DOMAINS:
        cap = CAP.get(domain, DEFAULT_CAP)
        stats = dict(items=0, contaminated=0, too_long=0, too_short=0, duplicate=0)
        seen = set()
        ordered = sorted(items(domain, raw, local_hashes), key=lambda x: order_key(domain, x[0]))
        quotas = ({'validation': eval_quota, 'test': eval_quota} if domain in HELD_OUT else
                  {'train': train_quota, 'controller': eval_quota, 'test_in_domain': eval_quota, 'calibration': calib_quota})
        for ident, payload in ordered:
            split = bucket(domain, ident)
            if split not in quotas or len(streams.get((split, domain), ())) >= quotas[split]:
                if all(len(streams.get((s, domain), ())) >= q for s, q in quotas.items()):
                    break
                continue
            text = payload if isinstance(payload, str) else tok.apply_chat_template(payload, tokenize=False)
            text = unicodedata.normalize('NFC', text)
            digest = sha(' '.join(text.split()))
            if digest in seen:
                stats['duplicate'] += 1
                continue
            seen.add(digest)
            if dirty.hit(text):
                stats['contaminated'] += 1
                continue
            ids = tok.encode(text, add_special_tokens=False)
            if len(ids) > cap:
                if domain in CHAT_KINDS:
                    stats['too_long'] += 1
                    continue
                ids = ids[:cap]
            if len(ids) < MIN_TOKENS:
                stats['too_short'] += 1
                continue
            streams.setdefault((split, domain), []).extend(ids + [eot])
            idents.setdefault(split, {}).setdefault(domain, []).append(ident)
            stats['items'] += 1
        report[domain] = stats | {s: len(streams.get((s, domain), ())) for s in quotas}
        print(json.dumps({domain: report[domain]}), flush=True)
    arrays, sizes = {}, {}
    plan = {'train': (IN_MIX, args.train_tokens // length), 'calibration': (IN_MIX, args.calibration_sequences),
            'controller': (IN_MIX, args.eval_sequences), 'test_in_domain': (IN_MIX, args.eval_sequences),
            'validation': (HELD_OUT, args.eval_sequences), 'test': (HELD_OUT, args.eval_sequences)}
    for split, (domains, total) in plan.items():
        packed = {d: rows_of(streams.get((split, d), []), length) for d in domains}
        counts = allocate({d: len(v) for d, v in packed.items()}, total)
        if sum(counts.values()) != total:
            raise RuntimeError(f'{split}: only {sum(counts.values())} of {total} sequences available')
        arrays[split] = np.concatenate([packed[d][:counts[d]] for d in domains])
        arrays[f'{split}_domain'] = np.concatenate([np.full(counts[d], DOMAINS.index(d), np.uint8) for d in domains])
        sizes[split] = counts
    return arrays, sizes, report, idents, local_hashes, bench


def downstream_rows(bench):
    """Frozen downstream selection: ARC-Easy/Challenge validation in full, five MMLU test
    questions per subject, and GSM8K test in a fixed hashed order (evaluate.py takes a prefix)."""
    chosen = [r for r in bench if r['task'] in ('arc_easy', 'arc_challenge') and r['split'] == 'validation']
    mmlu = sorted((r for r in bench if r['task'] == 'mmlu'), key=lambda r: sha(f"{SALT}:mmlu:{r['id']}"))
    per = {}
    for r in mmlu:
        if per.get(r['subject'], 0) < 5:
            chosen.append(r)
            per[r['subject']] = per.get(r['subject'], 0) + 1
    chosen += sorted((r for r in bench if r['task'] == 'gsm8k'), key=lambda r: sha(f"{SALT}:gsm8k:{r['id']}"))
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tokenizer', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--raw', type=Path, help='Reuse a fetched raw folder (default: <output>/raw, fetched)')
    ap.add_argument('--length', type=int, default=2048)
    ap.add_argument('--train-tokens', type=int, default=12_000_000)
    ap.add_argument('--eval-sequences', type=int, default=32)
    ap.add_argument('--calibration-sequences', type=int, default=8)
    ap.add_argument('--fetch-only', action='store_true')
    args = ap.parse_args()
    if args.output.exists() and not args.fetch_only:
        raise FileExistsError(args.output)
    raw = args.raw or args.output / 'raw'
    fetched = json.loads((raw / 'fetch.json').read_text()) if (raw / 'fetch.json').exists() else fetch(raw)
    for name, digest in fetched['sha256'].items():
        if file_sha(raw / name) != digest:
            raise ValueError(f'Raw source changed: {name}')
    if args.fetch_only:
        return
    for name, path in LOCAL.items():
        if not path.exists():
            raise FileNotFoundError(f'{name} source missing: {path}')
    arrays, sizes, report, idents, local_hashes, bench = build(args, raw)
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / 'tokens.npz', **arrays)
    with (args.output / 'downstream.jsonl').open('x') as out:
        for row in downstream_rows(bench):
            out.write(json.dumps(row, ensure_ascii=False) + '\n')
    (args.output / 'items.json').write_text(json.dumps(dict(idents=idents, local_file_sha256=local_hashes)) + '\n')
    tok_dir = Path(args.tokenizer)
    manifest = dict(
        sequence_length=args.length, domains=list(DOMAINS), in_mix=list(IN_MIX), held_out=list(HELD_OUT),
        shapes={k: list(arrays[k].shape) for k in sizes},
        sequences_per_domain=sizes, per_domain=report,
        split_policy=('item-level salted hash; in-mix items 3% controller, 3% test_in_domain, 1% calibration, '
                      'rest train; held-out-domain items half validation (stopping rule), half test'),
        contamination=dict(benchmarks='GSM8K test, ARC-Easy/Challenge validation+test, MMLU test (questions)',
                           rule=f'drop items sharing a {NGRAM}-word n-gram with a long question, or containing a whole '
                                f'{SHORT_NGRAM}-{NGRAM - 1}-word question'),
        caps=dict(default=DEFAULT_CAP, **CAP, min_tokens=MIN_TOKENS,
                  rule='chat items over cap skipped; plain text truncated to cap'),
        train_tokens=int(arrays['train'].size), salt=SALT,
        local_sources={k: str(v) for k, v in LOCAL.items()},
        local_source_sha256={'chat': file_sha(LOCAL['chat']),
                             **{f'instruction:{p.name}': file_sha(p) for p in sorted(LOCAL['instruction'].glob('instruction-*.jsonl'))}},
        raw_fetch=fetched, raw_folder=str(raw),
        tokenizer_sha256={p.name: file_sha(p) for p in tok_dir.glob('tokenizer*') if p.is_file()},
        chat_template_sha256=file_sha(tok_dir / 'chat_template.jinja') if (tok_dir / 'chat_template.jinja').exists() else None,
        code_sha256=file_sha(__file__),
        files_sha256={p.name: file_sha(p) for p in sorted(args.output.iterdir()) if p.is_file()})
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')
    print(json.dumps(dict(shapes=manifest['shapes'], sequences_per_domain=sizes)))


if __name__ == '__main__':
    main()
