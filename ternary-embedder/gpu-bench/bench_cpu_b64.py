import sys, time, json
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
P = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(P/'cpu-kernel')]
import fast
from transformers import AutoModel, AutoTokenizer
nt = int(sys.argv[1]); torch.set_num_threads(nt)
tk0 = AutoTokenizer.from_pretrained(P/'tokenizer', local_files_only=True)
path = P/'cache/hub/models--TaylorAI--bge-micro-v2/snapshots/3edf6d7de0faa426b09780416fe61009f26ae589'
bge = AutoModel.from_pretrained(path, local_files_only=True).eval(); tkb = AutoTokenizer.from_pretrained(path, local_files_only=True)
def bgef(ids, mk):
    h = bge(input_ids=ids, attention_mask=mk.long()).last_hidden_state; w = mk.unsqueeze(-1).float()
    return F.normalize((h*w).sum(1)/w.sum(1), dim=-1)
tern = fast.fast_student(P/'models/campaign-20261025/model-ternary.pt', 'p2v2')
TXT = {32: 'The cat sits beside the window while the rain falls.', 128: 'Scientific evidence supports careful testing of retrieval models. '*20}
with torch.inference_mode():
    for name, f, tk in (('ternary_p2v2', lambda i, m: tern(i, m)[2], tk0), ('bge_micro_float', bgef, tkb)):
        for cap, text in TXT.items():
            for b in (1, 64):
                def call():
                    e = tk([text]*b, padding='max_length', truncation=True, max_length=cap, return_tensors='pt'); return f(e['input_ids'], e['attention_mask'].bool())
                t0 = time.perf_counter(); n = 0
                while n < 3 or time.perf_counter()-t0 < 1: call(); n += 1
                s = []
                for _ in range(50 if b == 1 else 10): t = time.perf_counter(); call(); s.append((time.perf_counter()-t)*1000)
                med = float(np.median(s)); print(nt, 'threads', name, f'cap{cap} b{b}', round(med, 3), 'ms', round(b*1000/med, 1), 'emb/s', flush=True)
