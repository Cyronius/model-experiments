"""Time the float all-MiniLM-L6-v2 teacher on CPU or GPU with bench_ref.py's workloads (2026-10-28)."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from bench_ref import sync, timing


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('model', type=Path)
    p.add_argument('--device', default='cpu')
    p.add_argument('--texts', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--samples', type=int, default=15)
    a = p.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModel.from_pretrained(a.model).to(a.device).eval()
    data = json.loads(a.texts.read_text(encoding='utf-8'))

    def run(texts, cap, pad):
        enc = tok(texts, truncation=True, max_length=cap, padding=pad, return_tensors='pt').to(a.device)
        h = model(**enc).last_hidden_state
        m = enc['attention_mask'][..., None].to(h.dtype)
        return F.normalize((h*m).sum(1)/m.sum(1), dim=-1).cpu()

    report = {'device': a.device, 'torch': torch.__version__, 'threads': torch.get_num_threads(), 'micro': {
        'one_query_natural': timing(lambda: run(data['query'], 128, True), a.samples, 1, a.device),
        'batch4_cap32': timing(lambda: run(data['micro'][:4], 32, 'max_length'), a.samples, 4, a.device),
        'batch1_cap128': timing(lambda: run(data['micro'][:1], 128, 'max_length'), a.samples, 1, a.device)}}
    print(json.dumps(report), flush=True)
    report['bulk'] = {}
    for name, texts in data['corpora'].items():
        run(texts[:128], 128, True)
        sync(a.device)
        start = time.perf_counter()
        tokens = tok(texts, truncation=True, max_length=128)['input_ids']
        order = sorted(range(len(tokens)), key=lambda i: len(tokens[i]))
        for s in range(0, len(order), 128):
            enc = tok.pad({'input_ids': [tokens[i] for i in order[s:s+128]]}, padding=True, return_tensors='pt').to(a.device)
            model(**enc).last_hidden_state.cpu()
        sync(a.device)
        elapsed = time.perf_counter()-start
        report['bulk'][name] = {'texts': len(texts), 'embeddings_per_second': len(texts)/elapsed}
        print(json.dumps({name: report['bulk'][name]}), flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out/'report.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
