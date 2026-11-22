"""Time and encode with the released PyTorch reference on CPU or GPU, for comparison with the NPU (2026-10-28)."""
import argparse
import json
from pathlib import Path
import statistics
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F


def load(model_dir, device):
    sys.path.insert(0, str(model_dir))
    from ternary_embedder import TernaryEmbedder, TernLinear, quant_act, tern_codes
    model = TernaryEmbedder.from_pretrained(model_dir)
    # Decode each ternary matrix once instead of on every call; the arithmetic is unchanged.
    for m in model.modules():
        if isinstance(m, TernLinear):
            codes, scale = tern_codes(m.weight)
            m.register_buffer('decoded', codes*scale)
            m.forward = (lambda self: lambda x: F.linear(quant_act(x), self.decoded, self.bias))(m)
    return model.to(device).eval()


def sync(device):
    if device.startswith('cuda'):
        torch.cuda.synchronize()


def timing(fn, samples, count, device):
    for _ in range(3):
        fn()
    sync(device)
    times = []
    for _ in range(samples):
        t = time.perf_counter()
        fn()
        sync(device)
        times.append((time.perf_counter()-t)*1000)
    return {'median_ms': statistics.median(times), 'p95_ms': float(np.percentile(times, 95)),
            'embeddings_per_second': count*1000/statistics.median(times)}


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('model', type=Path)
    p.add_argument('--device', default='cpu')
    p.add_argument('--threads', type=int, default=0, help='CPU threads; 0 keeps the PyTorch default')
    p.add_argument('--texts', type=Path, required=True, help='JSON {"query": [...], "micro": [...], "corpora": {name: [...]}}')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--samples', type=int, default=15)
    a = p.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    model = load(a.model, a.device)
    tok = model.tokenizer
    data = json.loads(a.texts.read_text(encoding='utf-8'))
    report = {'device': a.device, 'torch': torch.__version__, 'threads': torch.get_num_threads(),
              'device_name': torch.cuda.get_device_name(0) if a.device.startswith('cuda') else 'cpu', 'micro': {}}

    def run(texts, cap, pad):
        enc = tok(texts, truncation=True, max_length=cap, padding=pad, return_tensors='pt')
        return model(enc['input_ids'].to(a.device), enc['attention_mask'].bool().to(a.device)).cpu()

    query = data['query']
    report['micro']['one_query_natural'] = timing(lambda: run(query, 128, True), a.samples, 1, a.device)
    report['micro']['batch4_cap32'] = timing(lambda: run(data['micro'][:4], 32, 'max_length'), a.samples, 4, a.device)
    report['micro']['batch1_cap128'] = timing(lambda: run(data['micro'][:1], 128, 'max_length'), a.samples, 1, a.device)
    print(json.dumps(report), flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    report['bulk'] = {}
    for name, texts in data['corpora'].items():
        run(texts[:128], 128, True)
        sync(a.device)
        start = time.perf_counter()
        # encode() sorts by length and pads each batch of 128 to its longest text.
        tokens = tok(texts, truncation=True, max_length=128)['input_ids']
        order = sorted(range(len(tokens)), key=lambda i: len(tokens[i]))
        vectors = np.empty((len(texts), 384), np.float32)
        for s in range(0, len(order), 128):
            idx = order[s:s+128]
            enc = tok.pad({'input_ids': [tokens[i] for i in idx]}, padding=True, return_tensors='pt')
            vectors[idx] = model(enc['input_ids'].to(a.device), enc['attention_mask'].bool().to(a.device)).float().cpu().numpy()
        sync(a.device)
        elapsed = time.perf_counter()-start
        np.save(a.out/f'{name}.npy', vectors)
        report['bulk'][name] = {'texts': len(texts), 'seconds': elapsed, 'embeddings_per_second': len(texts)/elapsed}
        print(json.dumps({name: report['bulk'][name]}), flush=True)
    (a.out/'report.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
