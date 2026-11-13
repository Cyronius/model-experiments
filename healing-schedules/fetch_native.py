"""Fetch only the pinned native comparator. Run under dibs run network."""
import argparse
import json
import urllib.request
from pathlib import Path

from accounting import ROOT
from pruning import sha256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    meta = json.loads((ROOT / 'metadata/Qwen3.5-0.8B.json').read_text())
    repo, revision = meta['model_id'], meta['revision']
    with urllib.request.urlopen(f'https://huggingface.co/api/models/{repo}/revision/{revision}') as f:
        info = json.load(f)
    names = [x['rfilename'] for x in info['siblings'] if
             x['rfilename'].endswith('.safetensors') or x['rfilename'] in
             ('config.json', 'tokenizer.json', 'tokenizer_config.json', 'model.safetensors.index.json')]
    args.output.mkdir(parents=True)
    record = dict(repo=repo, revision=revision, files={})
    for name in names:
        destination = args.output / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(f'https://huggingface.co/{repo}/resolve/{revision}/{name}', timeout=120) as f:
            with destination.open('wb') as out:
                while chunk := f.read(4 * 1024 * 1024):
                    out.write(chunk)
        record['files'][name] = dict(size=destination.stat().st_size, sha256=sha256(destination))
        print(name, destination.stat().st_size, flush=True)
    (args.output / 'download_manifest.json').write_text(json.dumps(record, indent=2) + '\n')


if __name__ == '__main__':
    main()
