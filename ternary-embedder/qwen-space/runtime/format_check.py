"""Stage 4 Python pre-checks of the runtime weight format (SS384, 128 contract, core mean):
  1. int8 embeddings with per-row scale cost <= 0.1 core nDCG (else float16 embeddings)
  2. 7-bit per-output-row linear codes in [-64, 63] cost <= 0.3 core nDCG vs model-q8.pt
Writes qwen-space/runtime/format-check.json."""
import json, sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import CKPT, dump, evaluate, features, load_student, student_vectors
import shared
from model import Student

CORE = ['SciFact', 'NFCorpus']


def score(model):
    out = {}
    for n in CORE:
        d = shared.dev_task(n)
        docs = [shared.cut_to_contract(t, 126) for t in d['texts']]
        q = student_vectors(model, features(model, d['queries'], 'single'))
        D = student_vectors(model, features(model, docs, 'single'))
        out[n] = evaluate(q @ D.T, d)[0]['ndcg10']
    out['core'] = float(np.mean([out[n] for n in CORE]))
    return out


def int8_rows(w):
    s = w.abs().amax(1, keepdim=True).clamp(min=1e-12) / 127
    return (w / s).round().clamp(-127, 127) * s


def codes7(w):
    s = w.abs().amax(1, keepdim=True).clamp(min=1e-12) / 63
    return (w / s).round().clamp(-64, 63) * s


def with_linears(model, src_state, fn):
    from export_pair_precision import ReferenceLinear
    for i, layer in enumerate(model.layers):
        for name in ('q', 'k', 'v', 'o', 'fc1', 'fc2'):
            w = src_state[f'layers.{i}.{name}.weight']
            b = src_state[f'layers.{i}.{name}.bias']
            setattr(layer, name, ReferenceLinear(fn(w.float()), b.float()))
    return model


def main():
    torch.set_num_threads(8)
    res = {}
    q8 = load_student('q8')
    res['q8_reference'] = score(q8)
    m = load_student('q8')
    with torch.no_grad():
        for emb in (m.word, m.pos, m.tok_type):
            emb.weight.copy_(int8_rows(emb.weight))
    res['q8_int8_embeddings'] = score(m)
    flt = torch.load(CKPT['float'], map_location='cpu', weights_only=False)['state']
    m7 = with_linears(load_student('q8'), flt, codes7)
    res['codes7_per_row_from_float'] = score(m7)
    with torch.no_grad():
        for emb in (m7.word, m7.pos, m7.tok_type):
            emb.weight.copy_(int8_rows(emb.weight))
    res['codes7_per_row_plus_int8_embeddings'] = score(m7)
    # also the 7-bit format re-derived from the Q8 reconstruction (what the runtime would ship)
    q8state = {k: v for k, v in load_student('q8').state_dict().items()}
    m7b = with_linears(load_student('q8'), q8state, codes7)
    res['codes7_per_row_from_q8_reconstruction'] = score(m7b)
    ref = res['q8_reference']['core']
    res['int8_embedding_cost'] = ref - res['q8_int8_embeddings']['core']
    res['codes7_cost_from_float'] = ref - res['codes7_per_row_from_float']['core']
    res['codes7_cost_from_q8'] = ref - res['codes7_per_row_from_q8_reconstruction']['core']
    res['combined_cost_from_float'] = ref - res['codes7_per_row_plus_int8_embeddings']['core']
    res['int8_embeddings_ok'] = res['int8_embedding_cost'] <= 0.1
    res['codes7_ok'] = min(res['codes7_cost_from_float'], res['codes7_cost_from_q8']) <= 0.3
    dump(Path(__file__).parent / 'format-check.json', res)
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
