"""Healing data shards (heal_shards_rules.json): the dense 9B's answers, graded, with its top-20.

  problems  (the bundle's HF environment, local) the problem lists of the first shards, from GSM8K
            and MATH train, OpenThoughts math with a short boxed answer, and UltraChat prompts.
  generate  (vLLM's environment, on a pod) answers one shard's problems with thinking off and on,
            grades them, and writes the shard in heal_sessions.Pool's format, chunk by chunk, until
            its time is up. Resumes a cut shard.
  summary   (either) counts per source and thinking mode, kept shares, tokens.

    python heal_shards.py problems --shards 4
    python heal_shards.py generate --model DIR --problems heal_shards/problems_000.jsonl --out DIR --until EPOCH
    python heal_shards.py summary --out DIR

generate and summary import nothing from this repo: vLLM's environment pins other torch and
transformers versions.
"""
import argparse
import json
import random
import re
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RULES = ROOT / 'heal_shards_rules.json'
STOPS = ('<|im_end|>', '<|endoftext|>')


def log(msg):
    print(time.strftime('%H:%M:%S'), msg, flush=True)


# ---------------------------------------------------------------- grading (no repo imports)

def last_boxed(text):
    """The contents of the last \\boxed{...} in text (braces matched), or None (heal_geometry's)."""
    start = text.rfind('\\boxed{')
    if start < 0:
        return None
    depth, i = 0, start + len('\\boxed')
    for j in range(i, len(text)):
        depth += {'{': 1, '}': -1}.get(text[j], 0)
        if depth == 0:
            return text[i + 1:j]
    return None


def final_number(text):
    """The last number of the answer after any reasoning, preferring a boxed one (slim_batch's)."""
    tail = text.split('</think>')[-1]
    boxed = re.findall(r'\\boxed\{([^{}]*)\}', tail)
    found = re.findall(r'-?\d[\d,]*(?:\.\d+)?', boxed[-1] if boxed else tail)
    return float(found[-1].replace(',', '')) if found else None


def grade(problem, text, thinking):
    """(correct, extracted) for an answer's text: correct is None for ungraded sources, False for a
    thinking answer that never closed its reasoning."""
    kind = problem['grader']
    if kind is None:
        return None, None
    if thinking and '</think>' not in text:
        return False, None
    tail = text.split('</think>')[-1]
    if kind == 'number':
        got = final_number(tail)
        return got is not None and abs(got - float(problem['answer'])) < 1e-6, got
    if kind == 'math':
        from math_verify import parse, verify
        try:
            pred = parse(tail)
            return bool(verify(parse('\\boxed{' + problem['answer'] + '}'), pred)), str(pred[-1]) if pred else None
        except Exception:
            return False, None
    raise ValueError(kind)


# ---------------------------------------------------------------- problems (local)

def problems(args):
    """problems_<s>.jsonl for shards 0..args.shards-1: id, source, question, answer, grader, samples."""
    import moe_carve as mc
    import pyarrow.parquet as pq
    import slim_batch as sb
    import slim_heal as sh
    rules = json.loads(RULES.read_text())
    mc.configure(sb.CARVE_RULES)
    docs = json.loads((mc.OUT / 'corpus.json').read_text())['docs']
    used = {i for d in docs for i in d['items']}
    pools = {'gsm8k': [], 'math': [], 'openthoughts': [], 'chat': []}
    for n, r in sh.paged('openai/gsm8k', 'main', [[0, 7473]]):
        if f'gsm8k-{n}' not in used:
            pools['gsm8k'].append({'id': f'gsm8k-{n}', 'question': r['question'], 'grader': 'number',
                                   'answer': r['answer'].rpartition('####')[2].strip().replace(',', '')})
    configs = json.loads(sb.CARVE_RULES.read_text())['fetch']['math_configs']
    for c in configs:
        for n, r in sh.paged('EleutherAI/hendrycks_math', c, [[0, 3000]]):
            boxed = last_boxed(r['solution'])
            if boxed is not None and f'{c}-{n}' not in used:
                pools['math'].append({'id': f'{c}-{n}', 'question': r['problem'], 'grader': 'math',
                                      'answer': boxed, 'level': r.get('level')})
    for n, r in sh.paged('open-thoughts/OpenThoughts-114k', 'metadata', [[0, 8000]]):
        boxed = last_boxed(r.get('ground_truth_solution') or '')
        proof = re.search(r'\bprove\b|\bshow that\b|\bdemonstrate\b', r['problem'], re.I) or \
            re.search(r'Q\.?E\.?D|blacksquare|square\s*$', boxed or '')
        if r['domain'] == 'math' and boxed and len(boxed) <= 80 and not proof and f'openthoughts-{n}' not in used:
            pools['openthoughts'].append({'id': f'openthoughts-{n}', 'question': r['problem'], 'grader': 'math',
                                          'answer': boxed})
    for k, row in enumerate(pq.read_table(mc.CHAT, columns=['prompt_id', 'prompt']).slice(0, 5000).to_pylist()):
        if row['prompt_id'] not in used and 20 <= len(row['prompt'].strip()) <= 1500:
            pools['chat'].append({'id': f"ultrachat-{row['prompt_id'][:16]}", 'question': row['prompt'].strip(),
                                  'grader': None, 'answer': None})
    seed = rules['teacher']['seed']
    for name, items in pools.items():
        random.Random(f'{seed}-{name}').shuffle(items)
        log(f'{name}: {len(items)} eligible')
    folder = ROOT / 'heal_shards'
    folder.mkdir(exist_ok=True)
    for s in range(args.shards):
        shard = []
        for name, cfg in rules['sources'].items():
            block = pools[name][s * cfg['per_shard']:(s + 1) * cfg['per_shard']]
            if len(block) < cfg['per_shard']:
                raise RuntimeError(f'{name}: only {len(block)} problems left for shard {s}')
            shard += [dict(p, source=name, samples=cfg['samples']) for p in block]
        random.Random(f'{seed}-shard-{s}').shuffle(shard)
        path = folder / f'problems_{s:03d}.jsonl'
        if path.exists():
            raise ValueError(f'{path} exists; refusing to overwrite')
        path.write_text(''.join(json.dumps(p, ensure_ascii=False) + '\n' for p in shard))
        log(f'{path.name}: {len(shard)} problems')


# ---------------------------------------------------------------- generation (vLLM)

def unpack(completion, stop, topk):
    """answer_ids (the stop token kept when finished), finished, the teacher's top-k rows (ids,
    log-probs) for every answer token, the summed log-prob of the written tokens, how many written
    tokens were inside the stored top-k."""
    tokens = list(completion.token_ids)
    entries = list(completion.logprobs or [])
    if len(entries) != len(tokens):
        raise RuntimeError(f'{len(tokens)} tokens but {len(entries)} log-prob entries')
    finished = completion.finish_reason == 'stop'
    ids = np.full((len(tokens), topk), -1, np.int32)
    lps = np.full((len(tokens), topk), -np.inf, np.float32)
    total, inside = 0.0, 0
    for p, (t, entry) in enumerate(zip(tokens, entries)):
        best = sorted((v.rank, k, v.logprob) for k, v in entry.items())[:topk]
        ids[p, :len(best)] = [k for _, k, _ in best]
        lps[p, :len(best)] = [lp for _, _, lp in best]
        total += entry[t].logprob
        inside += t in ids[p]
    if finished and (not tokens or tokens[-1] not in stop):
        # A stop vLLM did not return as a token: add it, with no teacher row (CE only).
        tokens.append(stop[0])
        ids = np.concatenate([ids, np.full((1, topk), -1, np.int32)])
        lps = np.concatenate([lps, np.full((1, topk), -np.inf, np.float32)])
    return tokens, finished, ids, lps, total, inside


def generate(args, llm=None):
    """One shard: every problem not yet in out/samples.jsonl, in chunks, until args.until."""
    rules = json.loads(RULES.read_text())
    teacher = rules['teacher']
    topk, cap = teacher['topk'], teacher['cap']
    if llm is None:
        from vllm import LLM
        llm = LLM(model=args.model, dtype='bfloat16', max_model_len=8192, seed=0, logprobs_mode='raw_logprobs',
                  max_logprobs=topk, generation_config='vllm', limit_mm_per_prompt={'image': 0, 'video': 0},
                  gpu_memory_utilization=0.9)
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    tok = llm.get_tokenizer()
    stop = [tok.convert_tokens_to_ids(t) for t in STOPS]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    samples = out / 'samples.jsonl'
    rows = [json.loads(l) for l in samples.open()] if samples.exists() else []
    chunks = {r['chunk'] for r in rows}
    for f in out.glob('teacher_*.npz'):
        if int(f.stem.split('_')[1]) not in chunks:
            f.unlink()          # written before a crash, its rows never recorded
    done = {r['id'] for r in rows}
    todo = [json.loads(l) for l in Path(args.problems).open()]
    todo = [p for p in todo if p['id'] not in done]
    chunk = max(chunks, default=-1) + 1
    record_path = out / 'generate.json'
    record = json.loads(record_path.read_text()) if record_path.exists() else {'chunks': [], 'skipped_long_prompts': []}
    size = rules['shards']['chunk_problems']
    estimate = 40 * 60
    while todo:
        if time.time() + 1.1 * estimate > args.until:
            log(f'stopping: {len(todo)} problems left, the next chunk would pass the deadline')
            break
        part, todo = todo[:size], todo[size:]
        requests, owners = [], []
        for p in part:
            for mode, n in (('off', p['samples']['off']), ('on', p['samples']['on'])):
                if n == 0:
                    continue
                text = tok.apply_chat_template([{'role': 'user', 'content': p['question']}], tokenize=False,
                                               add_generation_prompt=True, enable_thinking=mode == 'on')
                ids = tok.encode(text, add_special_tokens=False)
                if len(ids) > teacher['max_prompt_tokens']:
                    record['skipped_long_prompts'].append(p['id'])
                    continue
                seed = (teacher['seed'] * 1000003 + hash_id(p['id']) * 2 + (mode == 'on')) % 2 ** 31
                requests.append((TokensPrompt(prompt_token_ids=ids),
                                 SamplingParams(n=n, max_tokens=cap, logprobs=topk, seed=seed, stop_token_ids=stop,
                                                **teacher['sampling'])))
                owners.append((p, mode == 'on', ids))
        t0 = time.time()
        results = llm.generate([r[0] for r in requests], [r[1] for r in requests])
        seconds = time.time() - t0
        new, arrays, generated, inside = [], {}, 0, 0
        for (p, thinking, ids), result in zip(owners, results, strict=True):
            for s, c in enumerate(result.outputs):
                answer, finished, tids, tlps, total, ins = unpack(c, stop, topk)
                generated += len(c.token_ids)
                inside += ins
                text = tok.decode([t for t in answer if t not in stop])
                correct, extracted = grade(p, text, thinking) if finished else (False, None)
                row = len(rows) + len(new)
                if finished:
                    arrays[f'ids_{row}'], arrays[f'logprobs_{row}'] = tids, tlps.astype(np.float16)
                new.append({'id': p['id'], 'sample': s + 2 * thinking, 'source': p['source'], 'thinking': thinking,
                            'chunk': chunk, 'prompt_ids': ids, 'answer_ids': answer, 'finished': finished,
                            'finish_reason': c.finish_reason, 'correct': correct, 'gold': p['answer'],
                            'extracted': extracted, 'logprob_sum': total})
        temp = out / f'teacher_{chunk}.tmp.npz'
        np.savez_compressed(temp, **arrays)
        temp.replace(out / f'teacher_{chunk}.npz')
        with samples.open('a') as stream:
            stream.write(''.join(json.dumps(r) + '\n' for r in new))
        rows += new
        record['chunks'].append({'chunk': chunk, 'problems': len(part), 'answers': len(new), 'generated_tokens': generated,
                                 'seconds': seconds, 'tokens_per_second': generated / seconds,
                                 'written_inside_top20': inside / max(1, generated)})
        record_path.write_text(json.dumps(record, indent=1))
        log(f'chunk {chunk}: {len(part)} problems, {len(new)} answers, {generated} tokens in {seconds:.0f} s '
            f'({generated / seconds:.0f}/s), {sum(r["correct"] is True for r in new)} correct')
        estimate, chunk = seconds, chunk + 1
    record['remaining_problems'] = len(todo)
    record_path.write_text(json.dumps(record, indent=1))
    summary(args)


def hash_id(ident):
    import hashlib
    return int.from_bytes(hashlib.sha256(ident.encode()).digest()[:4], 'big')


def summary(args):
    out = Path(args.out)
    rows = [json.loads(l) for l in (out / 'samples.jsonl').open()]
    table = {}
    for r in rows:
        t = table.setdefault(f"{r['source']}_{'on' if r['thinking'] else 'off'}",
                             {'answers': 0, 'finished': 0, 'correct': 0, 'kept_candidates': 0, 'tokens': 0, 'finished_tokens': 0})
        t['answers'] += 1
        t['finished'] += r['finished']
        t['correct'] += r['correct'] is True
        t['kept_candidates'] += r['finished'] and r['correct'] is not False
        t['tokens'] += len(r['answer_ids'])
        t['finished_tokens'] += len(r['answer_ids']) * r['finished']
    for t in table.values():
        t['mean_finished_length'] = t['finished_tokens'] / max(1, t['finished'])
    record = json.loads((out / 'generate.json').read_text()) if (out / 'generate.json').exists() else {}
    chunks = record.get('chunks', [])
    result = {'problems': len({r['id'] for r in rows}), 'answers': len(rows), 'by_source': table,
              'generated_tokens': sum(c['generated_tokens'] for c in chunks),
              'seconds': sum(c['seconds'] for c in chunks)}
    result['tokens_per_second'] = result['generated_tokens'] / max(1e-9, result['seconds'])
    (out / 'summary.json').write_text(json.dumps(result, indent=1))
    for name, t in sorted(table.items()):
        log(f"{name}: {t['answers']} answers, {t['finished']} finished, {t['correct']} correct, "
            f"{t['kept_candidates']} kept candidates, mean finished length {t['mean_finished_length']:.0f}")
    log(f"{result['generated_tokens']} tokens in {result['seconds']:.0f} s ({result['tokens_per_second']:.0f}/s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['problems', 'generate', 'summary'])
    ap.add_argument('--shards', type=int, default=4)
    ap.add_argument('--model')
    ap.add_argument('--problems')
    ap.add_argument('--out')
    ap.add_argument('--until', type=float, default=float('inf'))
    args = ap.parse_args()
    {'problems': problems, 'generate': generate, 'summary': summary}[args.stage](args)


if __name__ == '__main__':
    main()
