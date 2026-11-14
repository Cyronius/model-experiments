"""vLLM proof (vllm_proof_rules.json): can vLLM's Qwen3.5-9B stand in for our HF dense teacher?

  generate  (vLLM's own environment) the check prompts with thinking on and off, and train problems
            for 40 minutes, sampled as the model card says, keeping vLLM's raw top-20 log-probs.
  compare   (the bundle's HF environment) scores every answer as the checks do, then teacher-forces
            32 thinking-on answers through HF Qwen3.5-9B in bf16 and compares log-probs.

    python vllm_proof.py generate --model DIR --out DIR
    python vllm_proof.py compare --model DIR --out DIR

generate imports nothing from this repo: vLLM's environment pins other torch and transformers versions.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RULES = ROOT / 'vllm_proof_rules.json'
SAMPLING = {'temperature': 1.0, 'top_p': 0.95, 'top_k': 20, 'presence_penalty': 1.5}
SEED = 26103103
CAP = 4096
TOPK = 20
THROUGHPUT_SECONDS = 40 * 60


def log(msg):
    print(time.strftime('%H:%M:%S'), msg, flush=True)


def generate(args):
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    llm = LLM(model=args.model, dtype='bfloat16', max_model_len=8192, seed=0, logprobs_mode='raw_logprobs',
              max_logprobs=TOPK, generation_config='vllm', limit_mm_per_prompt={'image': 0, 'video': 0},
              gpu_memory_utilization=0.9)
    tok = llm.get_tokenizer()
    stop = [tok.convert_tokens_to_ids(t) for t in ('<|im_end|>', '<|endoftext|>')]

    def prompt_ids(question, thinking):
        text = tok.apply_chat_template([{'role': 'user', 'content': question}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=thinking)
        return tok.encode(text, add_special_tokens=False)

    def params(seed, n=1):
        return SamplingParams(n=n, max_tokens=CAP, logprobs=TOPK, seed=seed, stop_token_ids=stop, **SAMPLING)

    def unpack(completion, keep_topk):
        tokens, finished = list(completion.token_ids), False
        if tokens and tokens[-1] in stop:
            tokens, finished = tokens[:-1], True
        finished = finished or completion.finish_reason == 'stop'
        sampled, top_ids, top_lp = [], [], []
        for t, entry in zip(tokens, completion.logprobs or []):
            sampled.append(entry[t].logprob)
            if keep_topk:
                best = sorted(((v.rank, k, v.logprob) for k, v in entry.items()))[:TOPK]
                top_ids.append([k for _, k, _ in best] + [-1] * (TOPK - len(best)))
                top_lp.append([lp for _, _, lp in best] + [-np.inf] * (TOPK - len(best)))
        return tokens, finished, sampled, top_ids, top_lp

    record = {'vllm': __import__('vllm').__version__, 'parts': {}}
    prompts = json.loads((ROOT / 'probe_prompts_96.json').read_text())['prompts'][:96]
    for thinking in (True, False):
        name = f"check_thinking_{'on' if thinking else 'off'}"
        ids = [prompt_ids(p['question'], thinking) for p in prompts]
        t0 = time.time()
        results = llm.generate([TokensPrompt(prompt_token_ids=i) for i in ids],
                               [params(SEED + k) for k in range(len(ids))])
        seconds = time.time() - t0
        rows, tops = [], {'ids': [], 'logprobs': []}
        for p, i, r in zip(prompts, ids, results, strict=True):
            tokens, finished, sampled, top_ids, top_lp = unpack(r.outputs[0], True)
            rows.append({'id': p['id'], 'prompt_ids': i, 'tokens': tokens, 'finished': finished,
                         'finish_reason': r.outputs[0].finish_reason, 'sampled_logprobs': sampled})
            tops['ids'].append(np.array(top_ids, dtype=np.int32).reshape(-1, TOPK))
            tops['logprobs'].append(np.array(top_lp, dtype=np.float32).reshape(-1, TOPK))
        (out / f'{name}.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
        np.savez_compressed(out / f'{name}_top{TOPK}.npz', **{f'ids_{k}': a for k, a in enumerate(tops['ids'])},
                            **{f'logprobs_{k}': a for k, a in enumerate(tops['logprobs'])})
        generated = sum(len(r['tokens']) for r in rows)
        record['parts'][name] = {'rows': len(rows), 'generated_tokens': generated, 'seconds': seconds,
                                 'tokens_per_second': generated / seconds}
        log(f'{name}: {generated} tokens in {seconds:.0f} s ({generated / seconds:.0f}/s)')
        (out / 'generate.json').write_text(json.dumps(record, indent=1))
    # Throughput: train problems, 4 samples each, in batches until the time is up.
    problems = [json.loads(l) for l in (ROOT / 'vllm_proof_problems.jsonl').open()]
    rows, t0, generated, batch = [], time.time(), 0, 128
    for start in range(0, len(problems), batch):
        if time.time() - t0 > THROUGHPUT_SECONDS:
            break
        part = problems[start:start + batch]
        ids = [prompt_ids(p['question'], True) for p in part]
        results = llm.generate([TokensPrompt(prompt_token_ids=i) for i in ids],
                               [params(SEED + 1000 + start + k, n=4) for k in range(len(part))])
        for p, r in zip(part, results, strict=True):
            for s, c in enumerate(r.outputs):
                tokens, finished, sampled, _, _ = unpack(c, False)
                generated += len(tokens)
                rows.append({'id': p['id'], 'sample': s, 'tokens': tokens, 'finished': finished,
                             'sampled_logprobs': sampled})
        log(f'throughput: {start + len(part)} problems, {generated} tokens, {generated / (time.time() - t0):.0f}/s')
    seconds = time.time() - t0
    (out / 'throughput.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    record['parts']['throughput'] = {'problems': len({r['id'] for r in rows}), 'samples': len(rows),
                                     'generated_tokens': generated, 'seconds': seconds,
                                     'tokens_per_second': generated / seconds, 'batch_prompts': batch}
    (out / 'generate.json').write_text(json.dumps(record, indent=1))
    log(f'throughput: {generated / seconds:.0f} tokens/s over {seconds:.0f} s')


def normalize(answer):
    a = answer.replace(' ', '').replace('\\left', '').replace('\\right', '').replace('$', '').replace('\\!', '')
    return a.replace('dfrac', 'frac').replace('tfrac', 'frac').rstrip('.')


def compare(args):
    import torch
    import torch.nn.functional as F
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    import heal_geometry as hg
    import heal_probe as hp
    import slim_batch as sb
    out = Path(args.out)
    rules = json.loads(RULES.read_text())
    tok = AutoTokenizer.from_pretrained(args.model)
    loop = json.loads((ROOT / 'geometry_check_rules.json').read_text())['loop']
    truth = {p['id']: p['answer'] for p in json.loads((ROOT / 'probe_prompts_96.json').read_text())['prompts']}
    report = {'rules_sha256': sb.sha(RULES), 'generate': json.loads((out / 'generate.json').read_text())}
    for name in ('check_thinking_on', 'check_thinking_off'):
        rows = [json.loads(l) for l in (out / f'{name}.jsonl').open()]
        for r in rows:
            answer = sb.final_number(tok.decode(r['tokens']))
            r.update(length=len(r['tokens']), answer=answer,
                     correct=answer is not None and abs(answer - truth[r['id']]) < 1e-6)
        s = hp.generations(rows, loop)
        report[name] = {k: v for k, v in s.items() if k != 'looping'}
        log(f"{name}: {s['correct']}/{len(rows)} right, {s['finished']} finished, {s['loops']} loops")
    answers = {json.loads(l)['id']: json.loads(l) for l in (ROOT / 'vllm_proof_problems.jsonl').open()}
    right, finished, n = 0, 0, 0
    for line in (out / 'throughput.jsonl').open():
        r = json.loads(line)
        p, text = answers[r['id']], tok.decode(r['tokens'])
        if p['source'] == 'gsm8k':
            got = sb.final_number(text)
            ok = got is not None and abs(got - float(p['answer'])) < 1e-6
        else:
            got = hg.last_boxed(text)
            ok = got is not None and normalize(got) == normalize(p['answer'])
        right, finished, n = right + ok, finished + r['finished'], n + 1
    report['throughput_answers'] = {'samples': n, 'correct': right, 'finished': finished,
                                    'note': 'GSM8K by final number; MATH by exact match of the last \\boxed{} after light normalization (undercounts).'}
    # Fidelity: HF teacher-forced log-probs for the first 32 finished thinking-on answers.
    path = Path(args.model)
    config = AutoConfig.from_pretrained(path)
    config = getattr(config, 'text_config', config)
    model = AutoModelForCausalLM.from_pretrained(path, config=config, dtype=torch.bfloat16).to('cuda').eval()
    kept = set(json.loads((ROOT / 'results/moe-carve-v2-20261029/slim/vocab64k.json').read_text())['kept_ids'])
    rows = [json.loads(l) for l in (out / 'check_thinking_on.jsonl').open()]
    top = np.load(out / f'check_thinking_on_top{TOPK}.npz')
    chosen = [k for k, r in enumerate(rows) if r['finished'] and r['tokens']][:32]
    diffs, agree, kls, outside, positions = [], [], [], [], 0
    with torch.no_grad():
        for k in chosen:
            r = rows[k]
            ids = r['prompt_ids'] + r['tokens']
            h = model.model(input_ids=torch.tensor([ids], device='cuda'), use_cache=False).last_hidden_state[0]
            start = len(r['prompt_ids']) - 1
            target = torch.tensor(r['tokens'], device='cuda')
            vids = torch.from_numpy(top[f'ids_{k}']).long().cuda()
            vlp = torch.from_numpy(top[f'logprobs_{k}']).float().cuda()
            for a in range(0, len(r['tokens']), 512):
                b = min(a + 512, len(r['tokens']))
                lp = F.log_softmax(model.lm_head(h[start + a:start + b]).float(), -1)
                hf_sampled = lp.gather(1, target[a:b, None])[:, 0]
                diffs.append((hf_sampled - torch.tensor(r['sampled_logprobs'][a:b], device='cuda')).abs().cpu())
                agree.append((lp.argmax(-1) == vids[a:b, 0]).cpu())
                valid = vids[a:b] >= 0
                pv = torch.where(valid, vlp[a:b], -torch.inf).log_softmax(-1)
                qh = torch.where(valid, lp.gather(1, vids[a:b].clamp_min(0)), -torch.inf).log_softmax(-1)
                kls.append(torch.where(valid, pv.exp() * (pv - qh), 0).sum(-1).cpu())
                mass = vlp[a:b].exp()
                out_mask = torch.tensor([[i not in kept for i in row] for row in vids[a:b].tolist()], device='cuda') & valid
                outside.append(((mass * out_mask).sum(-1) / (mass * valid).sum(-1)).cpu())
                positions += b - a
    diffs, agree, kls, outside = (torch.cat(x).double() for x in (diffs, agree, kls, outside))
    report['fidelity'] = {'answers': len(chosen), 'positions': positions,
                          'sampled_logprob_abs_diff_mean': float(diffs.mean()),
                          'sampled_logprob_abs_diff_p99': float(diffs.quantile(0.99)),
                          'top1_agreement': float(agree.mean()), 'top20_kl_mean': float(kls.mean()),
                          'top20_mass_outside_64k': float(outside.mean())}
    f, p = report['fidelity'], report['generate']['parts']['throughput']
    report['verdict'] = {
        'accuracy': 64 <= report['check_thinking_on']['correct'] <= 82,
        'logprobs': f['sampled_logprob_abs_diff_mean'] < 0.05 and f['sampled_logprob_abs_diff_p99'] < 0.5
                    and f['top1_agreement'] >= 0.97 and f['top20_kl_mean'] < 0.02,
        'throughput': p['tokens_per_second'] >= 1000,
        'vocab': f['top20_mass_outside_64k'] < 0.01}
    (out / 'compare.json').write_text(json.dumps(report, indent=1))
    log(f"fidelity: {json.dumps(f)}")
    log(f"verdict: {json.dumps(report['verdict'])}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['generate', 'compare'])
    ap.add_argument('--model', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    {'generate': generate, 'compare': compare}[args.stage](args)


if __name__ == '__main__':
    main()
