"""iGPU (Radeon 8060S, ROCm) latency/throughput of the final models vs bge-micro/MiniLM. Run with a ROCm torch venv."""
import sys, time, json
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
P = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(P/'tern-embed')]
from model import Student, TernLinear, tern_codes, quant_act
from transformers import AutoModel, AutoTokenizer
dev = torch.device(sys.argv[1] if len(sys.argv) > 1 else 'cuda')
class Frozen(torch.nn.Module):
    def __init__(s, l):
        super().__init__(); c, sc = tern_codes(l.weight.detach()); s.register_buffer('w', c*sc); s.register_buffer('b', l.bias.detach().clone())
    def forward(s, x): return F.linear(quant_act(x), s.w, s.b)
def ours(name):
    m = Student.load(P/f'models/campaign-20261025/model-{name}.pt').eval()
    if name == 'ternary':
        for L in m.layers:
            for n in ('q', 'k', 'v', 'o', 'fc1', 'fc2'): setattr(L, n, Frozen(getattr(L, n)))
    return lambda ids, mk: m(ids, mk)[2], m
class Mean(torch.nn.Module):
    def __init__(s, m): super().__init__(); s.m = m
    def forward(s, ids, mk):
        h = s.m(input_ids=ids, attention_mask=mk.long()).last_hidden_state.float(); w = mk.unsqueeze(-1).float()
        return F.normalize((h*w).sum(1)/w.sum(1).clamp(min=1), dim=-1)
def hf(name, rev):
    path = P/'cache/hub'/f'models--{name.replace("/", "--")}/snapshots'/rev
    m = Mean(AutoModel.from_pretrained(path, local_files_only=True)).eval()
    return (lambda ids, mk: m(ids, mk)), m, AutoTokenizer.from_pretrained(path, local_files_only=True)
tk0 = AutoTokenizer.from_pretrained(P/'tokenizer', local_files_only=True)
cfg = []
for n in ('ternary', 'float'):
    f, m = ours(n); cfg.append((n, f, m, tk0, (torch.float32, torch.bfloat16) if n == 'float' else (torch.float32,)))
f, m, t = hf('TaylorAI/bge-micro-v2', '3edf6d7de0faa426b09780416fe61009f26ae589'); cfg.append(('bge_micro', f, m, t, (torch.float32, torch.bfloat16)))
f, m, t = hf('sentence-transformers/all-MiniLM-L6-v2', '1110a243fdf4706b3f48f1d95db1a4f5529b4d41'); cfg.append(('minilm', f, m, t, (torch.float32, torch.bfloat16)))
TXT = {32: 'The cat sits beside the window while the rain falls.', 128: 'Scientific evidence supports careful testing of retrieval models. '*20}
res = {}
def sync():
    if dev.type == 'cuda': torch.cuda.synchronize()
with torch.inference_mode():
    for name, f, m, tk, dts in cfg:
        for dt in dts:
            m.to(dev).to(dt)
            for cap, text in TXT.items():
                for batch in (1, 64):
                    def call():
                        e = tk([text]*batch, padding='max_length', truncation=True, max_length=cap, return_tensors='pt')
                        v = f(e['input_ids'].to(dev), e['attention_mask'].bool().to(dev)); return v.float().cpu()
                    t0 = time.perf_counter(); n = 0
                    while n < 5 or time.perf_counter()-t0 < 1.0: v = call(); n += 1
                    assert torch.isfinite(v).all()
                    s = []
                    for _ in range(100 if batch == 1 else 20):
                        t = time.perf_counter(); call(); s.append((time.perf_counter()-t)*1000)
                    med = float(np.median(s)); key = f'{name}-{str(dt).split(".")[1]}-cap{cap}-b{batch}'
                    res[key] = dict(median_ms=med, emb_per_s=batch*1000/med); print(key, round(med, 3), 'ms', round(batch*1000/med, 1), 'emb/s', flush=True)
json.dump(res, open(Path(__file__).parent/f'results-{dev.type}.json', 'w'), indent=1)
