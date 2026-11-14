"""The GSM8K check (geometry_check_rules.json) on a released dense Qwen3.5 model, as the yardstick
for the MoE student: the same 96 prompts, chat template, sampler (model card, thinking mode), cap,
answer extraction and loop rule. The model runs as released: bf16 weights, its full vocabulary, the
DeltaNet state in fp32 as the student's checks keep it. Also counts its text parameters.

    dense_check.py --model Qwen/Qwen3.5-4B --decoding sampled --device cuda

Writes <carve>/dense_check/<model name>/{gen_<decoding>.jsonl, check_<decoding>.json}; never overwrites.
"""
import argparse
import json
import time
from pathlib import Path

import torch

import heal_probe as hp
import moe_carve as mc
import slim_batch as sb
import slim_opd as so

ROOT = Path(__file__).resolve().parent


def load(name):
    from huggingface_hub import snapshot_download
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    path = snapshot_download(name, local_files_only=True)
    config = AutoConfig.from_pretrained(path)
    config = getattr(config, 'text_config', config)
    model, info = AutoModelForCausalLM.from_pretrained(path, config=config, dtype=torch.bfloat16,
                                                       output_loading_info=True)
    unexpected = [k for k in info['unexpected_keys'] if not k.startswith(('model.visual', 'visual', 'mtp', 'model.mtp'))]
    if info['missing_keys'] or info.get('mismatched_keys') or unexpected:
        raise RuntimeError(f'Invalid model load: {info}')
    return model.to(mc.DEVICE).eval().requires_grad_(False), AutoTokenizer.from_pretrained(path), path


def parameters(model):
    """Text parameters, total and without the token embedding and output head (counted once if tied)."""
    seen, total, embedding = set(), 0, 0
    for name, p in model.named_parameters():
        if p.data_ptr() in seen:
            continue
        seen.add(p.data_ptr())
        total += p.numel()
        if 'embed_tokens' in name or name.startswith('lm_head'):
            embedding += p.numel()
    return {'total': total, 'non_embedding': total - embedding, 'embedding_and_head': embedding}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--rules', type=Path, default=ROOT / 'geometry_check_rules.json')
    ap.add_argument('--decoding', choices=['sampled', 'greedy'], default='sampled')
    ap.add_argument('--thinking', choices=['default', 'on', 'off'], default='default',
                    help="the chat template's enable_thinking (default: the tokenizer's own)")
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    mc.DEVICE = args.device
    mc.configure(sb.CARVE_RULES)
    rules = json.loads(args.rules.read_text())
    cfg = rules['eval']
    thinking = {'default': None, 'on': True, 'off': False}[args.thinking]
    out = mc.OUT / 'dense_check' / (args.model.split('/')[-1] + ('' if thinking is None else f'_thinking_{args.thinking}'))
    out.mkdir(parents=True, exist_ok=True)
    for f in (f'gen_{args.decoding}.jsonl', f'check_{args.decoding}.json'):
        if (out / f).exists():
            raise ValueError(f'{out / f} exists; refusing to overwrite')
    model, tok, path = load(args.model)
    counts = parameters(model)
    prompts = json.loads((ROOT / cfg['prompts']).read_text())['prompts'][:cfg['count']]
    ids = [tok.encode(so.template(tok, p['question'], thinking), add_special_tokens=False) for p in prompts]
    vocab = type('Vocab', (), {})()
    vocab.stop = [tok.convert_tokens_to_ids(t) for t in ('<|im_end|>', '<|endoftext|>')]
    vocab.removed = torch.zeros(model.config.vocab_size, device=mc.DEVICE)
    sample = args.decoding == 'sampled'
    Decode, _ = so.stopping_classes()
    x, mask, _ = so.left_pad(ids, vocab.stop[1])
    seed = cfg['seed'] + sample
    torch.manual_seed(seed)
    processor = Decode(vocab.removed, x.shape[1], torch.tensor([not sample] * len(ids), device=mc.DEVICE),
                       cfg['sampling'] if sample else None)
    t0 = time.time()
    with torch.no_grad(), sb.StoredState(model, [cfg['state']] * len(ids), seed) as stored:
        gen = so.compact(model, vocab, x, mask, processor, stored, cfg['max_new_tokens'], not sample)
    seconds = time.time() - t0
    rows = []
    for g, p in zip(gen, prompts, strict=True):
        answer = sb.final_number(tok.decode(g['tokens']))
        rows.append({'id': p['id'], 'model': args.model, 'sample': sample, 'tokens': g['tokens'],
                     'length': len(g['tokens']), 'finished': g['stop'] is not None, 'answer': answer,
                     'correct': answer is not None and abs(answer - p['answer']) < 1e-6})
    (out / f'gen_{args.decoding}.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    summary = {k: v for k, v in hp.generations(rows, rules['loop']).items()}
    at = so.truncated(rows, [dict(p) for p in prompts], tok, cfg['also_at'])
    summary_at = {k: v for k, v in hp.generations(at, rules['loop']).items() if k != 'looping'}
    record = {'model': args.model, 'snapshot': path, 'thinking': args.thinking,
              'prompt_example': so.template(tok, prompts[0]['question'], thinking), 'parameters': counts, 'decoding': args.decoding,
              'rules_sha256': sb.sha(args.rules), 'max_new_tokens': cfg['max_new_tokens'], 'seed': seed,
              'summary': summary, f'at_{cfg["also_at"]}': summary_at, 'seconds': seconds}
    (out / f'check_{args.decoding}.json').write_text(json.dumps(record, indent=1))
    print(f"{args.model} {args.decoding}: {summary['correct']}/{len(rows)} right, {summary['finished']} finished, "
          f"{summary['loops']} loops; non-embedding parameters {counts['non_embedding'] / 1e9:.2f}B "
          f"(total {counts['total'] / 1e9:.2f}B), {seconds / 60:.0f} min", flush=True)


if __name__ == '__main__':
    main()
