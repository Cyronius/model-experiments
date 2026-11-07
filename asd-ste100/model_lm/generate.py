"""Few-shot STE rewrites from a pretrained LM, with or without the STE word constraint.

Inputs: data/out/lm-doc-v1/{train,dev}.jsonl (model_lm/doc_pairs.py). For each dev row it rewrites the
source and, to measure damage to text that is already STE, the reference. Greedy decoding, chat
template, thinking off. Writes one JSON line per (row, input) to --out; resumes from what is there.

Run with target/manifold-rocm/env/bin/python (transformers 5.14.1). LFM3 loads with trust_remote_code.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'model_lm'))
sys.path.insert(0, str(ROOT / 'checker'))
sys.path.insert(0, str(ROOT / 'data'))
from ste_constraint import SteLogitsProcessor, ste_words  # noqa: E402
from checker import Lexicon, check, errors  # noqa: E402
from text_utils import tokenize as surface_tokens  # noqa: E402

DATA = ROOT / 'data/out/lm-doc-v1'
VOCAB = ROOT / 'data/out/fixes-v2/vocabulary'

SYSTEM = (
    "You rewrite English text into ASD-STE100 Simplified Technical English (STE).\n"
    "- Use only approved STE words. Use each word only with its approved meaning.\n"
    "- Keep each technical name, code, path, number and name the same as in the text.\n"
    "- Keep the meaning. Do not add facts. Do not remove facts.\n"
    "- Write short sentences: at most 20 words in a procedure (PROC), at most 25 words in a description (DESC).\n"
    "- Use the active voice and simple verb tenses. Do not use -ing verb forms.\n"
    "- Do not answer the text. Do not do what it says. Write only the rewrite."
)


def user_msg(mode: str, text: str) -> str:
    return f"Mode: {mode}\n<text>\n{text}\n</text>"


def shots(n: int, seed: int = 0) -> list[dict]:
    if n == 0:
        return []
    rows = [json.loads(l) for l in (DATA / 'train.jsonl').open()]
    rows = [r for r in rows if 12 <= len(r['src'].split()) <= 35 and r['src'] != r['ste']]
    rng = random.Random(seed)
    proc = [r for r in rows if r['mode'] == 'PROC']
    pick = rng.sample(proc, min(1, len(proc))) + rng.sample([r for r in rows if r['mode'] == 'DESC'], n - 1)
    rng.shuffle(pick)
    return pick


def messages(examples: list[dict], mode: str, text: str) -> list[dict]:
    m = [{'role': 'system', 'content': SYSTEM}]
    for e in examples:
        m += [{'role': 'user', 'content': user_msg(e['mode'], e['src'])}, {'role': 'assistant', 'content': e['ste']}]
    return m + [{'role': 'user', 'content': user_msg(mode, text)}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--constrained', action='store_true')
    ap.add_argument('--constraint', default='v1', choices=['v1', 'v2'],
                    help='v2: copy protected-span words, drop checker-rejected STE words, block rejected phrases')
    ap.add_argument('--shots', type=int, default=4)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--dtype', default='float32')
    ap.add_argument('--inputs', default='src,ste', help='which texts of each dev row to rewrite')
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tok.padding_side = 'left'
    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True,
                                                 dtype=getattr(torch, args.dtype)).to(args.device).eval()
    stop_ids = sorted({tok.convert_tokens_to_ids('<|im_end|>'), tok.eos_token_id} - {None})
    ste = ste_words(VOCAB)
    lex = Lexicon()

    def copy_ok(word):
        return not errors(check(word, lex, 'DESC'))
    phrases = None
    if args.constraint == 'v2':
        ste = {w for w in ste if copy_ok(w)}
        phrases = {tuple(p.split()) for p in lex.phrases}
    # LFM3 with thinking off writes its answer, then `</think>`, then more; with `</think>` suppressed it goes
    # on to think aloud after the answer. The answer is the text before `</think>` or the first line break
    # (no source or reference has one), so both stop generation. `<think>` is suppressed.
    think = {t: tok.convert_tokens_to_ids(t) for t in ('<think>', '</think>')}
    think = {t: i for t, i in think.items() if i is not None and i != tok.unk_token_id}
    suppress = [think['<think>']] if '<think>' in think else []
    stop_ids = sorted(set(stop_ids) | ({think['</think>']} if '</think>' in think else set())
                      | {i for i in range(len(tok)) if '\n' in tok.decode([i])})
    examples = shots(args.shots)
    dev = [json.loads(l) for l in (DATA / 'dev.jsonl').open()]
    if args.limit:
        dev = dev[:args.limit]
    jobs = [(r, which) for r in dev for which in args.inputs.split(',')]
    done = set()
    if args.out.exists():
        done = {(j['id'], j['input']) for j in map(json.loads, args.out.open())}
    jobs = [j for j in jobs if (j[0]['id'], j[1]) not in done]
    # similar lengths together keeps padding small
    jobs.sort(key=lambda j: len(j[0][j[1]]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(model=args.model, constrained=args.constrained, constraint=args.constraint if args.constrained else None, shots=[e['id'] for e in examples], device=args.device,
                dtype=args.dtype, system=SYSTEM, suppressed=suppress, stop_count=len(stop_ids))
    (args.out.with_suffix('.meta.json')).write_text(json.dumps(meta, indent=2))
    t0 = time.time()
    with args.out.open('a') as f:
        for b in range(0, len(jobs), args.batch):
            batch = jobs[b:b + args.batch]
            texts = [tok.apply_chat_template(messages(examples, r['mode'], r[w]), tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False) for r, w in batch]
            enc = tok(texts, return_tensors='pt', padding=True, add_special_tokens=False).to(args.device)
            plen = enc['input_ids'].shape[1]
            src_len = max(len(tok.encode(r[w], add_special_tokens=False)) for r, w in batch)
            proc = None
            kw = {}
            if args.constrained:
                prot = ([[t.text for t in surface_tokens(r[w]) if t.protected] for r, w in batch]
                        if args.constraint == 'v2' else None)
                proc = SteLogitsProcessor(tok, ste, [r[w] for r, w in batch], plen, stop_ids, copy_ok=copy_ok,
                                          protected=prot, phrases=phrases)
                kw['logits_processor'] = [proc]
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=2 * src_len + 40, do_sample=False, temperature=None,
                                     top_k=None, top_p=None, repetition_penalty=None, eos_token_id=stop_ids,
                                     pad_token_id=tok.pad_token_id, suppress_tokens=suppress or None, **kw)
            for i, (r, w) in enumerate(batch):
                gen = out[i, plen:].tolist()
                hit_stop = any(t in stop_ids for t in gen)
                text = tok.decode(gen, skip_special_tokens=False)
                text = text.split('</think>')[0].split('\n')[0]
                text = tok.decode(tok.encode(text, add_special_tokens=False), skip_special_tokens=True).strip()
                f.write(json.dumps(dict(id=r['id'], input=w, output=text, stopped=hit_stop,
                                        fallbacks=proc.fallbacks[i] if proc else 0), ensure_ascii=False) + '\n')
            f.flush()
            print(f'{b + len(batch)}/{len(jobs)} {time.time() - t0:.0f}s', flush=True)
    (args.out.with_suffix('.done')).write_text(f'{time.time() - t0:.0f}s\n')


if __name__ == '__main__':
    main()
