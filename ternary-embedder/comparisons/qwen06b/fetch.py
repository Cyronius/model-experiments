"""Fetch public model/data artifacts for the local comparison; no paid services."""
import csv,gzip,io,json
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile
from huggingface_hub import snapshot_download
ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parent.parent
REVISION='97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3'
snapshot_download('Qwen/Qwen3-Embedding-0.6B',revision=REVISION,
                  allow_patterns=['*.json','*.safetensors','merges.txt','vocab.json'],
                  cache_dir=PROJECT/'cache/hub')
for name in ('scifact','nfcorpus'):
    if (ROOT/f'data/{name}/qrels/test.tsv').exists():continue
    url=f'https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip'
    with urlopen(url) as response: archive=response.read()
    with ZipFile(io.BytesIO(archive)) as z:
        for rel in ('corpus.jsonl','queries.jsonl','qrels/test.tsv'):
            dest=ROOT/'data'/name/rel;dest.parent.mkdir(parents=True,exist_ok=True)
            dest.write_bytes(z.read(f'{name}/{rel}'))
path=ROOT/'data/stsb-test.json'
if not path.exists():
    with urlopen('https://sbert.net/datasets/stsbenchmark.tsv.gz') as response:
        text=gzip.decompress(response.read()).decode()
    rows=[r for r in csv.DictReader(io.StringIO(text),delimiter='\t',quoting=csv.QUOTE_NONE) if r['split']=='test']
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(rows))
