"""Download pinned free source datasets; build deduplicated, attributed teacher inputs.

  python harvest_sources.py download --out data/out/million
  python harvest_sources.py prepare --out data/out/million --n 1000000

Downloads are public data, never remote dataset code. Parquet decoding uses pyarrow.
Raw candidates are not STE pairs and never acquire a synthetic judge score.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import fnmatch
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import time
import urllib.request

from text_utils import normalized, tokenize

HERE = Path(__file__).resolve().parent
CATALOG = HERE / 'source_catalog.json'


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def request(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'STE-source-preparation/1.0'}), timeout=120)


def download_file(url: str, dest: Path, expected_size=None, expected_hash=None):
    if dest.exists() and expected_size is not None and dest.stat().st_size == expected_size:
        if expected_hash is None or sha256(dest) == expected_hash:
            return
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + '.partial')
    for attempt in range(3):
        try:
            with request(url) as response, partial.open('wb') as stream:
                for block in iter(lambda: response.read(1024 * 1024), b''):
                    stream.write(block)
            if expected_size is not None and partial.stat().st_size != expected_size:
                raise ValueError(f'incomplete download: {dest.name}')
            if expected_hash is not None and sha256(partial) != expected_hash:
                raise ValueError(f'checksum mismatch: {dest.name}')
            partial.replace(dest)
            return
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def download(out: Path, catalog: Path = CATALOG, workers: int = 3, max_gb: float = 6):
    out.mkdir(parents=True, exist_ok=True)
    entries = json.loads(catalog.read_text(encoding='utf-8'))['datasets']
    manifest = {'catalog_sha256': sha256(catalog), 'datasets': [], 'files': []}
    for entry in entries:
        root = out / 'raw' / entry['key']
        root.mkdir(parents=True, exist_ok=True)
        with request(f"https://huggingface.co/api/datasets/{entry['repo']}/revision/{entry['revision']}?blobs=true") as response:
            metadata = json.load(response)
        if metadata['sha'] != entry['revision']:
            raise ValueError('dataset revision differs from pinned catalog')
        published = metadata.get('cardData', {}).get('license')
        if entry['key'] != 'smoltalk' and published != entry['license']:
            raise ValueError(f"published license changed for {entry['key']}: {published}")
        card = f"https://huggingface.co/datasets/{entry['repo']}/resolve/{entry['revision']}/README.md"
        download_file(card, root / 'README.md')
        (root / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
        files = [f for f in metadata['siblings'] if any(fnmatch.fnmatch(f['rfilename'], pattern) for pattern in entry['patterns'])]
        if not files:
            raise ValueError(f"no files matched for {entry['key']}")
        manifest['datasets'].append(entry)
        for item in files:
            name = item['rfilename']
            # Dataset paths are untrusted; only use ordinary relative file paths.
            path = Path(name)
            if path.is_absolute() or '..' in path.parts or ':' in name or '\\' in name:
                raise ValueError('unsafe dataset filename')
            expected = item.get('lfs', {}).get('sha256')
            manifest['files'].append({'dataset': entry['key'], 'name': name,
                                     'size': item.get('size'), 'expected_sha256': expected,
                                     'path': str((root / path).relative_to(out)),
                                     'url': f"https://huggingface.co/datasets/{entry['repo']}/resolve/{entry['revision']}/{name}"})
    total = sum(f['size'] or 0 for f in manifest['files'])
    if total > max_gb * 1e9:
        raise ValueError(f'download would exceed --max-gb: {total / 1e9:.2f} GB')
    print(f"Downloading/checking {len(manifest['files'])} pinned files, {total / 1e9:.2f} GB", flush=True)
    # The manifest is written before downloads too, allowing inspection after an interruption.
    manifest_path = out / 'download_manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(download_file, f['url'], out / f['path'], f['size'], f['expected_sha256']): f for f in manifest['files']}
        for future in as_completed(futures):
            item = futures[future]
            future.result()
            item['sha256'] = sha256(out / item['path'])
            print(f"ready {item['dataset']}/{item['name']}", flush=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    return {'files': len(manifest['files']), 'bytes': total}


def records(path: Path):
    if path.suffix == '.parquet':
        import pyarrow.parquet as pq
        for batch in pq.ParquetFile(path).iter_batches(batch_size=128):
            yield from batch.to_pylist()
    else:
        with path.open(encoding='utf-8') as stream:
            for line in stream:
                if line.strip():
                    yield json.loads(line)


def texts(row: dict, fmt: str):
    if fmt == 'magicoder':
        for field, role in (('problem', 'user'), ('solution', 'assistant')):
            if isinstance(row.get(field), str):
                yield role, row[field]
        return
    messages = row.get('messages', [])
    if isinstance(messages, str):
        messages = json.loads(messages)
    for message in messages:
        if message.get('role') in {'system', 'user', 'assistant'} and isinstance(message.get('content'), str):
            yield message['role'], message['content']


def chunks(text: str, max_words: int = 60):
    """Whole-sentence chunks; fenced code is skipped rather than cut into fragments."""
    text = re.sub(r'```[\s\S]*?```', '\n\n', text)
    text = re.sub(r'<[^>\n]+>', ' ', text)
    for paragraph in re.split(r'\n\s*\n|\n(?=\s*(?:[-*+] |\d+[.)] ))', text):
        paragraph = re.sub(r'^\s*(?:#{1,6}\s+|[-*+]\s+|\d+[.)]\s+)', '', paragraph.strip())
        paragraph = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'\1 (\2)', paragraph)
        paragraph = re.sub(r'\s+', ' ', paragraph).strip()
        if not paragraph or paragraph.startswith(('|', '```')):
            continue
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9`"(])', paragraph)
        pending = []
        for sentence in sentences:
            words = len(sentence.split())
            if words > max_words:
                if pending:
                    yield ' '.join(pending)
                    pending = []
                # Preserve sentence semantics: never truncate a long sentence mid-clause.
                continue
            if pending and (len((' '.join(pending) + ' ' + sentence).split()) > max_words or len(pending) == 3):
                yield ' '.join(pending)
                pending = []
            pending.append(sentence)
        if pending:
            yield ' '.join(pending)


TECHNICAL = re.compile(r'\b(?:code|function|class|method|api|server|client|database|query|file|directory|'
                       r'command|script|python|javascript|typescript|rust|java|linux|windows|git|docker|'
                       r'kubernetes|http|json|xml|sql|array|string|integer|parameter|argument|variable|'
                       r'model|token|tensor|memory|thread|process|compile|install|configure|debug)\b', re.I)
IMPERATIVE = re.compile(r'^(?:please\s+)?(?:use|run|add|remove|open|close|set|write|read|create|delete|'
                        r'install|configure|check|select|click|copy|move|connect|return|ensure|make|do|'
                        r'implement|build|update|convert|explain|describe|list|find|give|provide|rewrite)\b', re.I)


def prepare(out: Path, n: int = 1_000_000, max_per_dataset: int = 600_000):
    manifest = json.loads((out / 'download_manifest.json').read_text(encoding='utf-8'))
    entries = {e['key']: e for e in manifest['datasets']}
    db = sqlite3.connect(out / 'sources.sqlite3')
    db.execute('PRAGMA journal_mode=WAL')
    db.executescript('''
        CREATE TABLE IF NOT EXISTS sources (id TEXT PRIMARY KEY, text TEXT NOT NULL, mode TEXT NOT NULL,
          dataset TEXT NOT NULL, technical INTEGER NOT NULL, provenance TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS documents (source_id TEXT, document_id TEXT, PRIMARY KEY(source_id,document_id));
        CREATE TABLE IF NOT EXISTS completed (file TEXT PRIMARY KEY, sha256 TEXT NOT NULL);
        CREATE INDEX IF NOT EXISTS source_dataset ON sources(dataset,id);
    ''')
    stats = Counter()
    for item in manifest['files']:
        key = item['dataset']
        entry = entries[key]
        path = out / item['path']
        actual_hash = sha256(path)
        if actual_hash != item.get('sha256'):
            raise ValueError(f'raw file has not been downloaded/verified: {path}')
        done = db.execute('SELECT sha256 FROM completed WHERE file=?', (item['path'],)).fetchone()
        if done:
            if done[0] != actual_hash:
                raise ValueError('raw file changed since preparation')
            continue
        current = db.execute('SELECT COUNT(*) FROM sources WHERE dataset=?', (key,)).fetchone()[0]
        if current >= max_per_dataset:
            continue
        print(f'preparing {key}/{item["name"]}, {current} already retained', flush=True)
        for index, row in enumerate(records(path)):
            document_hash = hashlib.sha256(json.dumps(row, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:24]
            doc = entry['repo'] + ':' + document_hash
            for role, text in texts(row, entry['format']):
                for chunk in chunks(text):
                    stats['candidate_chunks'] += 1
                    if not 5 <= len(chunk.split()) <= 60 or len(chunk) > 4000:
                        continue
                    letters = [c for c in chunk if c.isalpha()]
                    if not letters or sum(c.isascii() for c in letters) / len(letters) < .95:
                        continue
                    toks = tokenize(chunk)
                    if not toks or 2 * len(toks) > 256:
                        continue
                    # Prompts are PROC. Assistant explanations are DESC unless imperative.
                    mode = 'PROC' if role in {'user', 'system'} or IMPERATIVE.search(chunk) else 'DESC'
                    source_id = hashlib.sha256(normalized(chunk).encode()).hexdigest()[:24]
                    provenance = {'dataset': entry['repo'], 'revision': entry['revision'],
                                  'license': entry['license'], 'url': entry['card'],
                                  'file': item['name'], 'record': index, 'role': role,
                                  'domain': entry['domain'], 'document_id': doc}
                    technical = bool(TECHNICAL.search(chunk))
                    cur = db.execute('INSERT OR IGNORE INTO sources VALUES (?,?,?,?,?,?)',
                                     (source_id, chunk, mode, key, int(technical), json.dumps(provenance)))
                    current += cur.rowcount
                    db.execute('INSERT OR IGNORE INTO documents VALUES (?,?)', (source_id, doc))
                    if current >= max_per_dataset:
                        break
                if current >= max_per_dataset:
                    break
            if index % 1000 == 0:
                db.commit()
            if current >= max_per_dataset:
                break
        db.execute('INSERT OR REPLACE INTO completed VALUES (?,?)', (item['path'], actual_hash))
        db.commit()
        print(f'retained {current} unique chunks from {key}', flush=True)
    available = dict(db.execute('SELECT dataset,COUNT(*) FROM sources GROUP BY dataset'))
    # Weighted interleaving keeps software sources at the front and remains resumable by stable ID.
    cursors = {key: iter(db.execute('SELECT id,text,mode,technical,provenance FROM sources WHERE dataset=? ORDER BY id', (key,))) for key in sorted(available)}
    schedule = [key for key in sorted(cursors) for _ in range(entries[key]['export_weight'])]
    counts, modes, technical = Counter(), Counter(), 0
    target = out / 'sources.jsonl'
    temp = target.with_suffix('.jsonl.partial')
    with temp.open('w', encoding='utf-8', newline='\n') as stream:
        emitted = 0
        while schedule and emitted < n:
            for key in schedule[:]:
                try:
                    source_id, text, mode, is_technical, provenance = next(cursors[key])
                except StopIteration:
                    schedule = [value for value in schedule if value != key]
                    continue
                documents = [r[0] for r in db.execute('SELECT document_id FROM documents WHERE source_id=? ORDER BY document_id', (source_id,))]
                stream.write(json.dumps({'id': source_id, 'text': text, 'mode': mode,
                                         'document_ids': documents, 'provenance': json.loads(provenance)}, ensure_ascii=False) + '\n')
                counts[key] += 1; modes[mode] += 1; technical += is_technical; emitted += 1
                if emitted == n:
                    break
    temp.replace(target)
    summary = {'requested_sources': n, 'sources': emitted, 'available_by_dataset': available,
               'selected_by_dataset': dict(counts), 'modes': dict(modes), 'technical_keyword_matches': technical,
               'manifest_sha256': sha256(out / 'download_manifest.json'), 'sources_sha256': sha256(target),
               'status': 'ready' if emitted >= n else 'shortfall',
               'note': 'Raw source chunks only; no STE rewrite or meaning judge has run on these.'}
    (out / 'sources_summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    db.close()
    return summary


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command', choices=['download', 'prepare'])
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--catalog', type=Path, default=CATALOG)
    ap.add_argument('--n', type=int, default=1_000_000)
    ap.add_argument('--max-per-dataset', type=int, default=600_000)
    ap.add_argument('--max-gb', type=float, default=6)
    ap.add_argument('--workers', type=int, default=3)
    args = ap.parse_args()
    if min(args.n, args.max_per_dataset, args.workers, args.max_gb) <= 0:
        ap.error('counts and budgets must be positive')
    result = download(args.out, args.catalog, args.workers, args.max_gb) if args.command == 'download' else prepare(args.out, args.n, args.max_per_dataset)
    print(json.dumps(result, indent=2))
