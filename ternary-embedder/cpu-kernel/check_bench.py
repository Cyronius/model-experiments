import sys, time, json
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent; P = HERE.parent
sys.path[:0] = [str(HERE), str(P/'retrieval-diagnostics')]
from fast import fast_student
from diagnose import student
from transformers import AutoTokenizer
torch.set_num_threads(2)
ck = P/'models/campaign-20261025/model-ternary.pt'
tk = AutoTokenizer.from_pretrained(P/'tokenizer', local_files_only=True)
ref = student(ck, 'quant'); fast = {m: fast_student(ck, m) for m in ('i8', 'p2', 'i8v2', 'p2v2')}
TXT = {32: 'The cat sits beside the window while the rain falls.', 128: 'Scientific evidence supports careful testing of retrieval models. '*20}
res = {}
with torch.inference_mode():
    for cap, text in TXT.items():
        e = tk([text, 'short one', text[:90]], padding='max_length', truncation=True, max_length=cap, return_tensors='pt')
        v0 = ref(e['input_ids'], e['attention_mask'].bool())[2]
        for m, mod in fast.items():
            v = mod(e['input_ids'], e['attention_mask'].bool())[2]
            print(cap, m, 'min cos vs torch ref', float((v*v0).sum(-1).min()), flush=True)
    models = {'dense_sim': ref, **fast}
    for cap, text in TXT.items():
        for name, model in models.items():
            def call():
                e = tk([text], padding='max_length', truncation=True, max_length=cap, return_tensors='pt')
                return model(e['input_ids'], e['attention_mask'].bool())[2]
            t0 = time.perf_counter(); n = 0
            while n < 8 or time.perf_counter()-t0 < .75: call(); n += 1
            s = []
            for _ in range(200):
                t = time.perf_counter(); call(); s.append((time.perf_counter()-t)*1000)
            res[f'{name}-{cap}'] = float(np.median(s)); print(cap, name, 'median ms', round(res[f'{name}-{cap}'], 3), flush=True)
json.dump(res, open(HERE/'bench.json', 'w'), indent=1)
