"""Full-corpus retrieval with the packed p2v2 CPU kernels, paired against the torch reference path and the NPU rows.

Same data, tokenizer, 128-token cap and metrics as training-v2/evaluate_round2_npu.py. Run under `dibs run cpu`.
"""
import hashlib, json, sys, time
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent; P = HERE.parent
sys.path[:0] = [str(HERE), str(P/'retrieval-diagnostics'), str(P/'comparisons/qwen06b'), str(P/'training-v2')]
from fast import fast_student
from diagnose import student, paired
from compare import data
from round2_metrics import evaluate
from transformers import AutoTokenizer

CK = P/'models/campaign-20261025/model-ternary.pt'
TASKS = ('SciFact', 'NFCorpus', 'ArguAna')
NPU = P/'training-v2/delivery/native'


@torch.inference_mode()
def encode(model, tk, texts, batch=64):
    ids = tk(texts, truncation=True, max_length=128)['input_ids']
    order = sorted(range(len(texts)), key=lambda i: len(ids[i]))
    out = np.empty((len(texts), 384), np.float32)
    for s in range(0, len(order), batch):
        idx = order[s:s+batch]
        enc = tk.pad({'input_ids': [ids[i] for i in idx]}, padding=True, return_tensors='pt')
        out[idx] = model(enc['input_ids'], enc['attention_mask'].bool())[2].numpy()
    assert np.isfinite(out).all()
    return out


def main():
    torch.set_num_threads(8)
    tk = AutoTokenizer.from_pretrained(P/'tokenizer', local_files_only=True)
    models = {'torch_ref': student(CK, 'quant'), 'p2v2': fast_student(CK, 'p2v2')}
    report = {'checkpoint_sha256': hashlib.sha256(CK.read_bytes()).hexdigest(),
              'kernel_so_sha256': hashlib.sha256((HERE/'libtern_cpu.so').read_bytes()).hexdigest(), 'tasks': {}}
    for task in TASKS:
        d = data(task); texts = d['queries'] + d['texts']; nq = len(d['queries'])
        res, rows, vecs = {}, {}, {}
        for name, m in models.items():
            t = time.perf_counter(); v = vecs[name] = encode(m, tk, texts)
            res[name], rows[name] = evaluate(v[:nq] @ v[nq:].T, d)
            res[name]['encode_seconds'] = round(time.perf_counter() - t, 1)
            print(task, name, res[name]['ndcg10'], flush=True)
        cos = (vecs['p2v2'] * vecs['torch_ref']).sum(-1)
        entry = {'metrics': res, 'vector_cosine_p2v2_vs_ref': {'min': float(cos.min()), 'mean': float(cos.mean())},
                 'paired_p2v2_minus_ref': paired(rows['torch_ref'], rows['p2v2'])}
        npu_rows = NPU/f'npu-{task}-rows.json'
        if npu_rows.exists():
            entry['paired_p2v2_minus_npu'] = paired(json.loads(npu_rows.read_text()), rows['p2v2'])
        report['tasks'][task] = entry
        print(task, json.dumps({k: v for k, v in entry.items() if k != 'metrics'}), flush=True)
    report['mean_core_retrieval'] = {n: float(np.mean([report['tasks'][t]['metrics'][n]['ndcg10'] for t in ('SciFact', 'NFCorpus')]))
                                     for n in models}
    (HERE/'retrieval.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report['mean_core_retrieval']), flush=True)


if __name__ == '__main__':
    main()
