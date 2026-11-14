"""Short local healing runs that compare carve geometries: slim_heal's student and training loop,
unchanged, with the MLP carve (expert set and learned routers) taken from the run's rules.

Each run heals the combined model (64K vocabulary, low-rank DeltaNet, LoRA rank 64, routers
trained alongside) from the same start, on the same healing stream, with the same schedule; only
the carve differs. The low-rank DeltaNet factors come from the cached heal_factors.pt instead of
being rebuilt (an hour on the local GPU), and the healing data from the pilot's heal/ folder.

  python heal_geometry.py heal --rules heal_geometry_rules.json --run S0_K24 --device cuda
  python heal_geometry.py check --rules heal_geometry_rules.json --run S0_K24 \
      --check-rules geometry_check_rules.json --device cuda

heal writes <carve>/heal_geometry/<run>/; check runs slim_opd.py's check stage (drift reproduction
and GSM8K generations) on that run's adapter, into <run>/<check-rules folder>/. Neither overwrites.

A run with 'data' heals on one of the rules' datasets instead of the pilot's stream; build it first:

  python heal_geometry.py data --rules heal_geometry_rules.json --data template

which writes <carve>/heal_geometry/<dataset folder>/data.{npz,json}: train-split problems through
the chat template with a non-empty think block, mixed with pilot sequences the runs never saw."""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import moe_carve as mc
import slim_batch as sb
import slim_heal as sh

DATA_FILES = {'data.npz', 'data.json'}


def install(rules, name):
    """Point slim_heal at the run: its carve, its folder, the cached factors. Returns the folder."""
    run = rules['runs'][name]
    mc.configure(sb.CARVE_RULES)
    folder = mc.OUT / 'heal_geometry' / name
    pilot_out, cached = sh.out, sh.factors_path()
    if not cached.exists():
        raise FileNotFoundError(f'{cached}: run the pilot or heal_probe once to build the factors')
    if 'data' in run:
        data = mc.OUT / 'heal_geometry' / rules['datasets'][run['data']]['folder']
        sh.out = lambda n: data / n if n in DATA_FILES else folder / n
    else:
        sh.out = lambda n: pilot_out(n) if n in DATA_FILES else folder / n
    sh.state_path = lambda: folder / 'state.pt'
    sh.make_factors = lambda model, docs: torch.load(cached, weights_only=True, map_location=mc.DEVICE)
    sh.factors_path = lambda: cached

    def build(model, rules_, factors):
        s = torch.load(mc.ROOT / run['set'], weights_only=True, map_location=mc.DEVICE)
        if 'key' in run:
            s = s[run['key']]
        routers = []
        for sd in torch.load(mc.ROOT / run['routers'], weights_only=True, map_location=mc.DEVICE):
            router = nn.Linear(model.config.hidden_size, s['experts']).to(mc.DEVICE)
            router.load_state_dict(sd)
            routers.append(router)
        torch.manual_seed(rules_['student']['seed'])
        return sh.Slim(model, s, routers, factors, rules_['student']['lora_rank'], rules_['student']['lora_alpha'])

    sh.build = build
    return folder


def heal(args, rules):
    folder = install(rules, args.run)
    run = rules['runs'][args.run]
    # The heal stage saves the factors it is given; keep that copy out of the run folder.
    sh.factors_path = lambda: Path(rules['factors_copy'])
    Path(rules['factors_copy']).parent.mkdir(parents=True, exist_ok=True)
    # The reproduction check holds only for the carve the combined model was measured with.
    folder.mkdir(parents=True, exist_ok=True)
    train = dict(rules['train'], **run.get('train', {}))
    train['check'] = run.get('check', rules['train']['check'])
    if 'resume' in train:
        # The earlier run's saved state, which must be the one its heal.json recorded.
        resume = dict(train['resume'])
        earlier = json.loads((mc.ROOT / resume['heal']).read_text())
        resume['state'] = str(mc.ROOT / resume['state'])
        if sh.sha(Path(resume['state'])) != earlier['state_sha256']:
            raise ValueError(f"{resume['state']} is not the state {resume['heal']} recorded")
        last = earlier['milestones'][-1]
        # from_start: new data, so its order starts at step 0 (and tokens count this run's only).
        step = 0 if resume.pop('from_start', False) else earlier['steps']
        resume.update(step=step, kl=last['kl'], bpb_student=last['bpb_student'])
        train['resume'] = resume
    effective = dict(rules, train=train)
    (folder / 'rules.json').write_text(json.dumps(effective, indent=1))
    sh.RULES = folder / 'rules.json'
    if 'probe' in run:
        sh.AFTER_MILESTONE = probe(folder, run['probe'])
    sh.stage_heal(args)


def probe(folder, cfg):
    """After each milestone: GSM8K answers to the first cfg['prompts'] check prompts, sampled and
    greedy at their own caps (slim_opd's compact decoding), saved as probe_<tokens>_<decoding>.jsonl."""
    import heal_probe as hp
    import slim_opd as so
    check_rules = json.loads((mc.ROOT / cfg['check_rules']).read_text())
    ecfg = check_rules['eval']
    state = {}

    def run(model, slim, vocab, tokens):
        if 'prompts' not in state:
            state['tok'] = so.generation_tools(model, vocab)
            state['prompts'] = so.check_prompts(check_rules, state['tok'], vocab)[:cfg['prompts']]
        prompts, result = state['prompts'], {}
        for decoding, cap in cfg['caps'].items():
            sample = decoding == 'sampled'
            gen = so.generate(model, slim, vocab, [p['ids'] for p in prompts],
                              {'state': ecfg['state'], 'max_new_tokens': cap, 'compact': True}, True,
                              [False] * len(prompts) if sample else True, ecfg['sampling'] if sample else None,
                              ecfg['seed'] + sample)
            rows = so.check_rows(gen, prompts, state['tok'], 'healed', sample)
            (folder / f'probe_{tokens}_{decoding}.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
            summary = hp.generations(rows, check_rules['loop'])
            result[decoding] = dict({k: v for k, v in summary.items() if k != 'looping'}, cap=cap)
            sh.log(f"{tokens} tokens, {decoding} at {cap}: {summary['correct']}/{len(prompts)} right, "
                   f"{summary['finished']} finished, {summary['loops']} loops")
        return result

    return run


def last_boxed(text):
    """The contents of the last \\boxed{...} in text (braces matched), or None."""
    start = text.rfind('\\boxed{')
    if start < 0:
        return None
    depth, i = 0, start + len('\\boxed')
    for j in range(i, len(text)):
        depth += {'{': 1, '}': -1}.get(text[j], 0)
        if depth == 0:
            return text[i + 1:j]
    return None


def turn(question, reasoning, content):
    return [{'role': 'user', 'content': question.strip()},
            {'role': 'assistant', 'reasoning_content': reasoning.strip(), 'content': content.strip()}]


def templated_sources(domain, fetch):
    """(ident, chat turn) per item, from the pilot's fetch windows; idents as the pilot's domains."""
    if domain == 'math_word':
        for n, r in sh.paged('openai/gsm8k', 'main', fetch['math_word']):
            work, _, answer = r['answer'].rpartition('####')
            yield f'gsm8k-{n}', turn(r['question'], re.sub(r'<<[^>]*>>', '', work), f'The answer is {answer.strip()}.')
    elif domain == 'math':
        def config(c):
            for n, r in sh.paged('EleutherAI/hendrycks_math', c, fetch['math']):
                boxed = last_boxed(r['solution'])
                if boxed is not None:
                    yield f'{c}-{n}', turn(r['problem'], r['solution'], f'The answer is $\\boxed{{{boxed}}}$.')
        yield from sh.interleave(*[config(c) for c in fetch['math_configs']])
    elif domain == 'reasoning':
        for n, r in sh.paged('open-thoughts/OpenThoughts-114k', 'metadata', fetch['reasoning']):
            yield f'openthoughts-{n}', turn(r['problem'], r['deepseek_reasoning'], r['deepseek_solution'])
    else:
        raise ValueError(domain)


def build_data(args, rules):
    """The dataset's templated sequences (whole items only, eval overlap dropped) and the pilot
    sequences after the runs' last position in the pilot's order, in data.npz and data.json."""
    from transformers import AutoTokenizer
    import unicodedata
    spec = rules['datasets'][args.data]
    pilot_rules, docs = sh.setup()
    pilot = json.loads(sh.out('data.json').read_text())
    if pilot['rules_sha256'] != sh.sha(sh.RULES):
        raise RuntimeError(f'{sh.RULES} is not the rules the pilot data was built with')
    folder = mc.OUT / 'heal_geometry' / spec['folder']
    if (folder / 'data.npz').exists():
        raise ValueError(f'{folder}/data.npz exists; refusing to overwrite')
    hf = AutoTokenizer.from_pretrained(mc.MODEL, local_files_only=True)
    full, short, _ = sb.load_tokenizers()
    eot = full.token_to_id('<|endoftext|>')
    length = pilot['seq_len']
    exclude = {d: set() for d in mc.DOMAINS}
    for doc in docs:
        exclude[doc.get('source_domain', doc['domain'])].update(doc['items'])
    # Items an earlier dataset already templated.
    for name in spec.get('exclude_datasets', []):
        earlier = json.loads((mc.OUT / 'heal_geometry' / rules['datasets'][name]['folder'] / 'data.json').read_text())
        for d, r in earlier['templated'].items():
            exclude[d].update(r['idents'])
    evals = [[eot] + sb.retokenize(full, short, d)[2] for d in sb.docs_of(docs, 'eval')]
    total, cap = spec['templated_sequences'], spec['max_item_tokens']
    available, report = {}, {}
    for domain, cfg in spec['templated'].items():
        want = round(cfg['share'] * total)
        stream, used, skipped = [], [], 0
        for ident, item in templated_sources(domain, pilot_rules['data']['fetch']):
            if ident in exclude[domain]:
                continue
            text = unicodedata.normalize('NFC', hf.apply_chat_template(item, tokenize=False))
            ids = short.encode(text)
            if short.decode(ids) != text:
                raise ValueError(f'{ident}: the 64K tokenizer does not round-trip the item')
            if len(ids) > cap:
                skipped += 1
                continue
            stream += ids + [eot]
            used.append(ident)
            if len(stream) >= 1.15 * want * length:
                break
        seqs = np.asarray(stream[:len(stream) // length * length], np.uint32).reshape(-1, length)
        shared = sh.shared_windows(seqs, evals, pilot_rules['data']['overlap_tokens'])
        available[domain] = seqs[shared == 0]
        report[domain] = {'items': len(used), 'skipped_too_long': skipped, 'available': int(len(seqs)),
                          'dropped_for_eval_overlap': int((shared > 0).sum()), 'idents': used}
        sh.log(f'{domain}: {len(used)} items ({skipped} too long), {len(seqs)} sequences, '
               f'{int((shared > 0).sum())} dropped for eval overlap')
    # Each domain its share; a short domain's shortfall goes to the others in order.
    counts = {d: min(len(available[d]), round(c['share'] * total)) for d, c in spec['templated'].items()}
    for d in counts:
        counts[d] += min(len(available[d]) - counts[d], total - sum(counts.values()))
    if sum(counts.values()) != total:
        raise RuntimeError(f'Only {sum(counts.values())} templated sequences, not {total}')
    # The pilot sequences no run has trained on: S0_K24_4M stopped at step 246 of the pilot's order.
    seen = json.loads((mc.ROOT / spec['old_after']).read_text())
    earlier = json.loads((Path(mc.ROOT / spec['old_after']).parent / 'rules.json').read_text())['train']
    pilot_seqs = np.load(sh.out('data.npz'))
    order = np.random.default_rng(earlier['seed']).permutation(len(pilot_seqs['ids']))
    first = spec.get('old_start', seen['steps'] * earlier['batch_sequences'])
    old = order[first:first + spec['old_sequences']]
    domains = list(spec['templated'])
    ids = np.concatenate([available[d][:counts[d]] for d in domains] + [pilot_seqs['ids'][old]])
    labels = np.concatenate([np.full(counts[d], len(pilot['domains']) + i, np.uint8) for i, d in enumerate(domains)]
                            + [pilot_seqs['domain'][old]])
    folder.mkdir(parents=True, exist_ok=True)
    np.savez(folder / 'data.npz', ids=ids, domain=labels)
    record = dict(code_sha256={n: sh.sha(mc.ROOT / n) for n in ['heal_geometry.py', 'slim_heal.py']},
                  rules_sha256=sh.sha(args.rules), dataset=args.data, data_sha256=sh.sha(folder / 'data.npz'),
                  seq_len=length, sequences=int(len(ids)), tokens=int(ids.size),
                  domains=pilot['domains'] + [f'template_{d}' for d in domains],
                  templated={d: dict(report[d], sequences=counts[d]) for d in domains},
                  old={'from': spec['old_after'], 'order_seed': earlier['seed'], 'positions': [first, first + len(old)],
                       'per_domain': {pilot['domains'][k]: int((pilot_seqs['domain'][old] == k).sum())
                                      for k in range(len(pilot['domains']))}},
                  pilot_data_sha256=pilot['data_sha256'])
    (folder / 'data.json').write_text(json.dumps(record, indent=1))
    sh.log(f'{len(ids)} sequences: ' + ', '.join(f'{d} {counts[d]}' for d in domains) + f', old {len(old)}')


def check(args, rules):
    import slim_opd as so
    folder = install(rules, args.run)
    sh.RULES = folder / 'rules.json'
    so.RULES = args.check_rules.resolve()
    so.FOLDER = str(Path('heal_geometry') / args.run / json.loads(so.RULES.read_text())['folder'])
    # No distilled arm: the healed student is the run's adapter.
    so.finished_adapter = lambda spec: None
    so.stage_check(args)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['data', 'heal', 'check'])
    ap.add_argument('--rules', type=Path, required=True)
    ap.add_argument('--run')
    ap.add_argument('--data')
    ap.add_argument('--check-rules', type=Path)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    mc.DEVICE = args.device
    rules = json.loads(args.rules.read_text())
    {'data': build_data, 'heal': heal, 'check': check}[args.stage](args, rules)


if __name__ == '__main__':
    main()
