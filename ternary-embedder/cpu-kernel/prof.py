import sys, time
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent; P = HERE.parent
sys.path[:0] = [str(HERE)]
import fast
from transformers import AutoTokenizer
torch.set_num_threads(2)
tk = AutoTokenizer.from_pretrained(P/'tokenizer', local_files_only=True)
m = fast.fast_student(P/'models/campaign-20261025/model-ternary.pt', 'p2v2')
acc = [0.0]
orig = fast.FastLinear.forward
def timed(self, x):
    t = time.perf_counter(); y = orig(self, x); acc[0] += time.perf_counter()-t; return y
fast.FastLinear.forward = timed
with torch.inference_mode():
    for cap, text in {32: 'The cat sits beside the window while the rain falls.', 128: 'Scientific evidence supports careful testing of retrieval models. '*20}.items():
        e = tk([text], padding='max_length', truncation=True, max_length=cap, return_tensors='pt'); ids, mk = e['input_ids'], e['attention_mask'].bool()
        for _ in range(50): m(ids, mk)
        acc[0] = 0; tt = []; 
        for _ in range(200):
            t = time.perf_counter(); m(ids, mk); tt.append(time.perf_counter()-t)
        print(cap, 'model ms', round(np.median(tt)*1e3, 3), 'linear ms/call', round(acc[0]/200*1e3, 3), 'tokenize ms', end=' ')
        t = time.perf_counter()
        for _ in range(200): tk([text], padding='max_length', truncation=True, max_length=cap, return_tensors='pt')
        print(round((time.perf_counter()-t)/200*1e3, 3))
