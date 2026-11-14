"""fused2 NPU: full-corpus retrieval (paired against the CPU torch reference and the v1 NPU rows), then latency and
throughput on the same repeated texts as gpu-bench/ and cpu-kernel/ (tokenization included). Run under `dibs run npu`.

Throughput modes: 'packed' fills each 128-token dispatch with texts at their real length; 'padded' gives each text a
full cap-sized span (pack align = cap), matching the CPU/GPU rows, which pad every text to the cap."""
import hashlib, json, sys, time
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent; P = HERE.parent
sys.path[:0] = [str(HERE), str(P/'cpu-kernel'), str(P/'retrieval-diagnostics'), str(P/'comparisons/qwen06b'), str(P/'training-v2')]
from fused2_encoder import Fused2Encoder, BUILD
from eval_retrieval import encode as cpu_encode
from diagnose import student, paired
from compare import data
from round2_metrics import evaluate
from transformers import AutoTokenizer

CK = P/'models/campaign-20261025/model-ternary.pt'
TXT = {32: 'The cat sits beside the window while the rain falls.', 128: 'Scientific evidence supports careful testing of retrieval models. '*20}


def timed(fn, reps):
    t0 = time.perf_counter(); n = 0
    while n < 5 or time.perf_counter() - t0 < 1.0: fn(); n += 1
    s = []
    for _ in range(reps):
        t = time.perf_counter(); fn(); s.append((time.perf_counter() - t) * 1000)
    return float(np.median(s))


def main():
    speed_only = '--speed-only' in sys.argv
    torch.set_num_threads(2)
    tk = AutoTokenizer.from_pretrained(P/'tokenizer', local_files_only=True)
    report = {'checkpoint_sha256': hashlib.sha256(CK.read_bytes()).hexdigest(), 'build': BUILD.name,
              'host_threads': 2, 'tasks': {}, 'speed': {}}
    ref = student(CK, 'quant')
    e = Fused2Encoder(CK)
    try:
        for task in () if speed_only else ('SciFact', 'NFCorpus', 'ArguAna'):
            d = data(task); texts = d['queries'] + d['texts']; nq = len(d['queries'])
            ids = tk(texts, truncation=True, max_length=128)['input_ids']
            t = time.perf_counter()
            vq, _ = e.encode(ids[:nq], query=True); vd, slots = e.encode(ids[nq:])
            secs = time.perf_counter() - t
            v = torch.cat([vq, vd]).numpy()
            m, rows = evaluate(v[:nq] @ v[nq:].T, d)
            m.update(encode_seconds=round(secs, 1), dispatches_documents=slots)
            torch.set_num_threads(8)
            r = cpu_encode(ref, tk, texts); torch.set_num_threads(2)
            rm, rrows = evaluate(r[:nq] @ r[nq:].T, d)
            cos = (v * r).sum(-1)
            entry = {'fused2': m, 'cpu_ref_ndcg10': rm['ndcg10'],
                     'vector_cosine_vs_ref': {'min': float(cos.min()), 'mean': float(cos.mean())},
                     'paired_fused2_minus_ref': paired(rrows, rows)}
            v1 = P/f'training-v2/delivery/native/npu-{task}-rows.json'
            if v1.exists():
                entry['paired_fused2_minus_npu_v1'] = paired(json.loads(v1.read_text()), rows)
            report['tasks'][task] = entry
            print(task, json.dumps(entry), flush=True)
        if not speed_only: report['mean_core_retrieval'] = float(np.mean([report['tasks'][t]['fused2']['ndcg10'] for t in ('SciFact', 'NFCorpus')]))
        print('core mean', report.get('mean_core_retrieval'), flush=True)

        for cap, text in TXT.items():
            for batch in (1, 64):
                for mode, align in (('packed', 1), ('padded', cap)):
                    if batch == 1 and mode == 'padded': continue  # one text per dispatch either way
                    def call():
                        ids = tk([text] * batch, truncation=True, max_length=cap)['input_ids']
                        return e.encode(ids, align=align)
                    _, slots = call()
                    med = timed(call, 100 if batch == 1 else 30)
                    key = f'cap{cap}-b{batch}-{mode}'
                    report['speed'][key] = {'median_ms': med, 'emb_per_s': batch * 1000 / med, 'dispatches': slots,
                                            'tokens_per_text': len(tk(text, truncation=True, max_length=cap)['input_ids'])}
                    print(key, json.dumps(report['speed'][key]), flush=True)
        e.dispatch_ms.clear()
        for _ in range(200): e.encode([tk(TXT[128], truncation=True, max_length=128)['input_ids']])
        report['speed']['dispatch_ms_median'] = float(np.median(e.dispatch_ms))
    finally:
        e.close()
    (HERE/('fused2-speed-clean.json' if speed_only else 'fused2-eval.json')).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report['speed']), flush=True)


if __name__ == '__main__':
    main()
