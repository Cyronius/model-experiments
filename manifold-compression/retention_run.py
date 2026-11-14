"""Fixed-endpoint retention ablation. No checkpoint selection or model promotion.

All weights and evaluation are FP32; training matmuls use BF16 autocast. The
teacher is the frozen stock model, evaluated in FP32 on verified replay only.
Vocabulary heads are checkpointed in 64-token chunks, never a whole sequence.
Program execution and blinded science grading happen locally after collection.
"""
import argparse
from contextlib import nullcontext
import gc
import json
import os
from pathlib import Path
import time

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from correctness_heal import install_lora, load_adapter, save_adapter, set_training
from local_projection_models import from_recipe
from local_projection_probe import load_source
from retention_healing import expand_lora
from retention_prepare import OUT, PARENT, ROOT, SOURCE, save, sha

RECIPE = ROOT / 'results/local-projection-20261025/shared-broad-recipe.pt'
ARMS = ('A0', 'A1', 'A2', 'S0', 'S1', 'S2')
RUN_OUT = Path(os.environ.get('RETENTION_RUN_OUT', str(OUT / 'cuda')))


def log(text):
    print(time.strftime('%H:%M:%S'), text, flush=True)


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def configure(threads=8):
    torch.set_num_threads(threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # A changed optional dependency must not silently select another algorithm.
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
    if any(getattr(qwen, key) is not None for key in
           ('causal_conv1d_fn', 'causal_conv1d_update', 'chunk_gated_delta_rule',
            'fused_recurrent_gated_delta_rule')):
        raise RuntimeError('Protocol requires torch fallback; remove optional FLA/causal-conv kernels')


def initial_model(arm, expand=True):
    if arm not in ARMS:
        raise ValueError('Unknown arm')
    torch.manual_seed(26102719)
    if arm.startswith('A'):
        parent = json.loads((PARENT / 'selected/adapter.json').read_text())
        if parent['total_step'] != 24 or parent['rank'] != 16 or parent['alpha'] != 32:
            raise ValueError('Wrong continued parent')
        if parent['recipe_sha256'] != sha(RECIPE):
            raise ValueError('Parent projection recipe changed')
        model = from_recipe(SOURCE, torch.load(RECIPE, weights_only=True, map_location='cpu'), load_source)
        info = install_lora(model, 16, 32)
        if info['modules'] != parent['modules']:
            raise ValueError('Parent module inventory changed')
        load_adapter(model, PARENT / 'selected')
        if arm != 'A0' and expand:
            expand_lora(model, rank=64, seed=26102691)
    else:
        model = load_source(SOURCE)
        if arm != 'S0':
            install_lora(model, 64, 128)
    return model.eval()


def load_endpoint(arm, out=None):
    out = RUN_OUT if out is None else Path(out)
    model = initial_model(arm)
    if arm not in ('A0', 'S0'):
        path = out / arm / 'step-0016'
        meta = json.loads((path / 'adapter.json').read_text())
        frozen = json.loads((OUT / 'frozen.json').read_text())
        if meta['arm'] != arm or meta['step'] != 16 or meta['data_sha256'] != frozen['data_sha256']:
            raise ValueError('Endpoint identity differs')
        load_adapter(model, path)
    return model.eval()


def load_A0(): return load_endpoint('A0')
def load_A1(): return load_endpoint('A1')
def load_A2(): return load_endpoint('A2')
def load_S0(): return load_endpoint('S0')
def load_S1(): return load_endpoint('S1')
def load_S2(): return load_endpoint('S2')


def exact_terms(student_logits, labels, teacher_logits=None):
    """Unnormalized answer CE and exact forward KL, both in FP32 or FP64."""
    logits = student_logits if student_logits.dtype == torch.float64 else student_logits.float()
    ce = F.cross_entropy(logits, labels, reduction='sum')
    kl = ce * 0
    if teacher_logits is not None:
        lp = teacher_logits.detach().to(logits.dtype).log_softmax(-1)
        lq = logits.log_softmax(-1)
        kl = (lp.exp() * (lp - lq)).sum()
    return torch.stack((ce, kl))


def row_terms(model, row, teacher=None, chunk=64, autocast=True):
    device = next(model.parameters()).device
    ids = torch.tensor([row['input_ids']], device=device)
    positions = row['positions']
    labels = torch.tensor(row['targets'], device=device)
    # Teacher state and head remain FP32; the teacher never sees a student prefix.
    ht = None
    if row['replay'] and teacher is not None:
        with torch.no_grad(), torch.autocast(device.type, enabled=False):
            ht = teacher.model(input_ids=ids, use_cache=False).last_hidden_state[0, positions]
    context = torch.autocast(device.type, dtype=torch.bfloat16) if autocast else nullcontext()
    with context:
        hs = model.model(input_ids=ids, use_cache=False).last_hidden_state[0, positions]

        def part(h, y, teacher_state):
            target = None
            if teacher_state is not None:
                with torch.no_grad(), torch.autocast(device.type, enabled=False):
                    target = teacher.lm_head(teacher_state).float()
            with torch.autocast(device.type, enabled=False):
                student = model.lm_head(h.to(model.lm_head.weight.dtype))
            return exact_terms(student, y, target)

        values = [checkpoint(part, hs[i:i + chunk], labels[i:i + chunk],
                             None if ht is None else ht[i:i + chunk], use_reentrant=False)
                  for i in range(0, len(labels), chunk)]
        return torch.stack(values).sum(0)


def batched_states(model, rows):
    """Right padding only; gather real answer positions independently per row."""
    device = next(model.parameters()).device
    width = max(len(r['input_ids']) for r in rows)
    ids = torch.tensor([r['input_ids'] + [0] * (width - len(r['input_ids'])) for r in rows], device=device)
    mask = torch.tensor([[1] * len(r['input_ids']) + [0] * (width - len(r['input_ids'])) for r in rows], device=device)
    states = model.model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
    return torch.cat([states[i, r['positions']] for i, r in enumerate(rows)])


def batch_terms(model, rows, teacher=None, chunk=64, autocast=True):
    device = next(model.parameters()).device
    labels = torch.tensor([t for r in rows for t in r['targets']], device=device)
    replay = torch.tensor([r['replay'] for r in rows for _ in r['targets']], dtype=torch.bool, device=device)
    teacher_states = None
    replay_rows = [r for r in rows if r['replay']]
    if teacher is not None and replay_rows:
        with torch.no_grad(), torch.autocast(device.type, enabled=False):
            target = batched_states(teacher, replay_rows)
            teacher_states = target.new_zeros((len(labels), target.shape[-1]))
            teacher_states[replay] = target
    context = torch.autocast(device.type, dtype=torch.bfloat16) if autocast else nullcontext()
    with context:
        states = batched_states(model, rows)

        def part(h, y, mask, th):
            with torch.autocast(device.type, enabled=False):
                logits = model.lm_head(h.to(model.lm_head.weight.dtype))
                ce = F.cross_entropy(logits, y, reduction='sum')
                kl = ce * 0
                if th is not None and mask.any():
                    with torch.no_grad():
                        lp = teacher.lm_head(th[mask]).to(logits.dtype).log_softmax(-1)
                    lq = logits[mask].log_softmax(-1)
                    kl = (lp.exp() * (lp - lq)).sum()
            return torch.stack((ce, kl))

        return torch.stack([checkpoint(part, states[i:i + chunk], labels[i:i + chunk],
                  replay[i:i + chunk], None if teacher_states is None else teacher_states[i:i + chunk],
                  use_reentrant=False) for i in range(0, len(labels), chunk)]).sum(0)


def update(model, optimizer, rows, teacher, lr, autocast=True, microbatch=8):
    optimizer.zero_grad(set_to_none=True)
    denominator = sum(len(r['targets']) for r in rows)
    sums = torch.zeros(2, dtype=torch.float64)
    start = time.monotonic()
    ordered = sorted(rows, key=lambda r: len(r['input_ids']))
    for start_row in range(0, len(ordered), microbatch):
        terms = batch_terms(model, ordered[start_row:start_row + microbatch], teacher, autocast=autocast)
        value = terms.sum() / denominator
        if not torch.isfinite(value):
            raise RuntimeError('Nonfinite CE/KL')
        value.backward()
        sums += terms.detach().double().cpu()
    norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                        1., error_if_nonfinite=True)
    for group in optimizer.param_groups:
        group['lr'] = lr
    optimizer.step()
    replay = sum(len(r['targets']) for r in rows if r['replay'])
    return {'ce_sum': sums[0].item(), 'kl_sum': sums[1].item(),
            'objective_per_answer_token': sums.sum().item() / denominator,
            'kl_per_replay_token': sums[1].item() / replay if replay else None,
            'answer_tokens': denominator, 'replay_tokens': replay, 'rows': len(rows), 'microbatch_rows': microbatch,
            'grad_norm': norm.item(), 'lr': lr, 'seconds': time.monotonic() - start}


@torch.no_grad()
def fixture(model, ids):
    return model(input_ids=ids, use_cache=False, logits_to_keep=4).logits.cpu()


@torch.no_grad()
def greedy_fixture(model, ids):
    return model.generate(input_ids=ids, do_sample=False, use_cache=True,
                          max_new_tokens=8, repetition_penalty=1., no_repeat_ngram_size=0).cpu()


def runtime_smoke(device='cpu'):
    """Exercise both Qwen attention types and the actual mixed-precision loss."""
    from transformers import Qwen3_5TextConfig, Qwen3_5ForCausalLM
    from copy import deepcopy
    config = Qwen3_5TextConfig(vocab_size=257, hidden_size=256, intermediate_size=512,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=64,
        linear_num_key_heads=2, linear_num_value_heads=2, linear_key_head_dim=128, linear_value_head_dim=128,
        layer_types=['linear_attention', 'full_attention'], tie_word_embeddings=False,
        rope_parameters={'rope_type': 'default', 'rope_theta': 10000., 'partial_rotary_factor': 1., 'mrope_section': [8, 8, 16]})
    config._attn_implementation = 'eager'
    torch.manual_seed(26102731)
    model = Qwen3_5ForCausalLM(config).float().to(device).eval()
    teacher = deepcopy(model).requires_grad_(False)
    install_lora(model, 4, 8)
    ids = torch.randint(0, 257, (32,)).tolist()
    row = dict(input_ids=ids, positions=list(range(15, 31)), targets=ids[16:], replay=True)
    set_training(model)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.003,
                                 betas=(.9, .95), weight_decay=0., foreach=False)
    with torch.no_grad():
        before = row_terms(model, row, teacher).detach().cpu().tolist()
    records = [update(model, optimizer, [row], teacher, .003) for _ in range(3)]
    with torch.no_grad():
        after = row_terms(model, row, teacher).detach().cpu().tolist()
    if not after[0] < before[0] or not all(r['grad_norm'] > 0 for r in records):
        raise RuntimeError('Mixed-precision hybrid Qwen training smoke failed')
    if any(p.grad is not None for p in teacher.parameters()):
        raise RuntimeError('Teacher acquired gradients')
    result = {'device': device, 'before_ce_kl': before, 'after_ce_kl': after, 'steps': records}
    del model, teacher, optimizer
    cleanup()
    return result


@torch.no_grad()
def raw_generate(model, tok, cases, path, deadline=float('inf'), batch_size=16):
    from transformers import GenerationConfig
    from healing_data import generation_metrics
    from summarize_scale_probe import repeated_line_run
    if path.exists():
        raise ValueError('Generation exists; no silent resume or overwrite')
    model.eval()
    tok.padding_side = 'left'
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    result = {'status': 'running', 'cases': [], 'grading': 'pending local isolated program/manual science review'}
    config = GenerationConfig(do_sample=False, use_cache=True, eos_token_id=tok.eos_token_id,
                              pad_token_id=tok.pad_token_id, bos_token_id=tok.bos_token_id,
                              repetition_penalty=1., no_repeat_ngram_size=0)
    start = time.monotonic()
    for limit in sorted({c['max_new_tokens'] for c in cases}):
        config.max_new_tokens = limit
        group = [c for c in cases if c['max_new_tokens'] == limit]
        for i in range(0, len(group), batch_size):
            if time.time() > deadline - 180:
                raise TimeoutError('Generation deadline; partial outputs remain partial')
            subset = group[i:i + batch_size]
            prompts = [tok.apply_chat_template(c['messages'], tokenize=False,
                       add_generation_prompt=True, enable_thinking=False) for c in subset]
            inputs = tok(prompts, add_special_tokens=False, padding=True, return_tensors='pt').to(next(model.parameters()).device)
            tokens = model.generate(**inputs, generation_config=config)[:, inputs.input_ids.shape[1]:]
            for case, ids in zip(subset, tokens.tolist(), strict=True):
                if tok.eos_token_id in ids:
                    ids = ids[:ids.index(tok.eos_token_id) + 1]
                answer = tok.decode(ids, skip_special_tokens=True)
                metrics = generation_metrics(answer, ids, tok.eos_token_id)
                line = repeated_line_run(answer) >= 8
                result['cases'].append({k: case[k] for k in ('id', 'family', 'category', 'group')} |
                                       {'answer': answer, 'token_ids': ids, **metrics,
                                        'repeated_line_flag': line,
                                        'combined_loop_flag': line or metrics['loop_flag']})
            save(result, path)
            log(f'{path.name}: {len(result["cases"])}/{len(cases)}')
    result.update(status='complete', seconds=time.monotonic() - start)
    save(result, path)
    return {'cases': len(cases), 'seconds': result['seconds']}


def check_frozen():
    frozen = json.loads((OUT / 'frozen.json').read_text())
    if frozen['status'] != 'peer_reviewed' or frozen['reviewer'] != 'claude':
        raise ValueError('Missing explicit peer review')
    for relative, expected in frozen['files'].items():
        if sha(ROOT.parent / relative) != expected:
            raise ValueError('Frozen input or code changed: ' + relative)
    return frozen


def run(args):
    from audit_rotation import cache_check
    from scale_data import load_tokenizer
    configure(args.threads)
    if args.device != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('Full experiment is authorized on the reviewed CUDA pod only')
    frozen = check_frozen()
    data = json.loads((OUT / 'data.json').read_text())
    protocol = data['protocol']
    if len(data['updates']) != 16 or protocol['steps'] != 16:
        raise ValueError('Wrong fixed endpoint')
    if args.out.exists():
        raise ValueError('Use fresh result directory; no automatic resume')
    args.out.mkdir(parents=True)
    tok = load_tokenizer(SOURCE)
    ids = torch.tensor([data['train'][0]['input_ids'][:32]], device=args.device)
    report = {'status': 'running', 'data_sha256': frozen['data_sha256'], 'arms': {},
              'torch': torch.__version__, 'cuda': torch.version.cuda,
              'gpu': torch.cuda.get_device_name(), 'precision': protocol['precision'],
              'kernels': protocol['kernels'], 'selection': 'parent default, fixed endpoints only'}

    def state(stage):
        report['stage'] = stage
        save(report, args.out / 'result.json')
        log(stage)

    try:
        state('Stock and full-width numerical controls')
        stock = load_source(SOURCE).to(args.device)
        baseline = fixture(stock, ids)
        torch.save(baseline, args.out / 'S-initial-logits.pt')
        stock_cache = cache_check(stock, ids)
        if not stock_cache['numerical_pass']:
            raise RuntimeError('Stock cache gate failed')
        del stock
        cleanup()
        recipe = torch.load(RECIPE, weights_only=True, map_location='cpu')
        recipe = dict(recipe, width=recipe['basis'].shape[-1], scales=[1.] * len(recipe['scales']), predictors=None)
        full = from_recipe(SOURCE, recipe, load_source).to(args.device)
        actual = fixture(full, ids)
        torch.testing.assert_close(actual, baseline, atol=2e-3, rtol=2e-3)
        lp, lq = baseline.double().log_softmax(-1), actual.double().log_softmax(-1)
        kl = (lp.exp() * (lp - lq)).sum(-1)
        if kl.max() > 1e-5:
            raise RuntimeError('Full-width projection KL gate failed')
        full_cache = cache_check(full, ids)
        if not full_cache['numerical_pass']:
            raise RuntimeError('Full-width cache gate failed')
        report['controls'] = {'stock_cache': stock_cache, 'fullwidth_cache': full_cache,
                              'fullwidth_max_kl': kl.max().item(),
                              'fullwidth_max_abs': (actual - baseline).abs().max().item()}
        del full, recipe
        cleanup()
        # All training finishes before any held-out outcomes are inspected.
        for arm in ('A1', 'A2', 'S1', 'S2'):
            if time.time() > args.deadline - 900:
                raise TimeoutError('Insufficient time for another complete arm')
            state(arm + ': construct')
            model = initial_model(arm, expand=False).to(args.device)
            start_logits = fixture(model, ids)
            if arm.startswith('A'):
                info = expand_lora(model, rank=64, seed=26102691)
                after = fixture(model, ids)
                torch.testing.assert_close(after, start_logits, atol=5e-5, rtol=1e-5)
                start_logits = after
            else:
                info = {'rank': 64, 'alpha': 128,
                        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)}
            baseline_path = args.out / (arm[0] + '-initial-logits.pt')
            if baseline_path.exists():
                torch.testing.assert_close(start_logits, torch.load(baseline_path, weights_only=True), atol=0, rtol=0)
            else:
                torch.save(start_logits, baseline_path)
            control = cache_check(model, ids)
            if not control['numerical_pass']:
                raise RuntimeError('Initial cache/full-sequence mismatch')
            metadata = dict(info, arm=arm, data_sha256=frozen['data_sha256'],
                            recipe_sha256=sha(RECIPE), parent_step=24 if arm.startswith('A') else 0)
            teacher = load_source(SOURCE).to(args.device).eval() if arm.endswith('2') else None
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                         lr=protocol['lr'], betas=(.9, .95), weight_decay=0., foreach=False)
            record = {'initial_cache': control, 'steps': [], 'adapter': info}
            report['arms'][arm] = record
            if teacher is not None:
                with torch.no_grad():
                    replay_rows = [data['train'][i] for i in data['updates'][0] if data['train'][i]['replay']]
                    replay_rows.sort(key=lambda r: len(r['input_ids']))
                    floor = sum(batch_terms(model, replay_rows[i:i + 8], teacher)[1].item()
                                for i in range(0, len(replay_rows), 8))
                record['initial_kl_per_replay_token'] = floor / sum(len(r['targets']) for r in replay_rows)
            set_training(model)
            for step, indices in enumerate(data['updates'], 1):
                expected = max(60, record['steps'][-1]['seconds'] * 1.5) if record['steps'] else 180
                if time.time() + expected > args.deadline - 600:
                    raise TimeoutError('Training deadline; incomplete arm will not replace endpoint16')
                metrics = update(model, optimizer, [data['train'][i] for i in indices], teacher,
                                 protocol['lr'] * min(1., step / protocol['warmup_steps']))
                record['steps'].append({'step': step, **metrics})
                state(f'{arm}: update {step}/16')
                if step in (8, 16):
                    destination = args.out / arm / f'step-{step:04d}'
                    save_adapter(model, destination, dict(metadata, step=step))
                    torch.save({'optimizer': optimizer.state_dict(), 'step': step,
                                'data_sha256': frozen['data_sha256'], 'torch_rng': torch.get_rng_state(),
                                'cuda_rng': torch.cuda.get_rng_state_all()}, destination / 'optimizer.pt')
            model.gradient_checkpointing_disable()
            model.eval()
            expected = fixture(model, ids)
            expected_tokens = greedy_fixture(model, ids)
            record['final_cache'] = cache_check(model, ids)
            if not record['final_cache']['numerical_pass']:
                raise RuntimeError('Final cache gate failed')
            del optimizer, teacher, model
            cleanup()
            # A new model built from source and recipe must reproduce the saved endpoint.
            model = load_endpoint(arm, args.out).to(args.device)
            actual = fixture(model, ids)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            torch.testing.assert_close(greedy_fixture(model, ids), expected_tokens, atol=0, rtol=0)
            record['independent_reload_exact'] = True
            record['independent_reload_greedy_8_tokens_exact'] = True
            record['status'] = 'trained_endpoint16'
            del model
            cleanup()
            state(arm + ': trained and reloaded')
        # Each split has identical batches across arms. Tests are separate sealed files.
        for arm in ARMS:
            state(arm + ': generation')
            model = load_endpoint(arm, args.out).to(args.device)
            record = report['arms'].setdefault(arm, {})
            record['dev'] = raw_generate(model, tok, data['dev'], args.out / f'{arm}-dev.json', args.deadline)
            record['test'] = raw_generate(model, tok, data['test'], args.out / f'{arm}-test-sealed.json', args.deadline)
            record['status'] = 'generation_complete'
            del model
            cleanup()
        report['status'] = 'generation_complete_pending_0b_regression_and_local_review'
        state('All preregistered endpoints generated; tests remain sealed for local review')
    except BaseException as error:
        report.update(status='incomplete', error=f'{type(error).__name__}: {error}')
        state('Stopped; parent remains default')
        raise


def regressions(args):
    from scale_data import load_tokenizer
    configure(args.threads)
    check_frozen()
    data = json.loads((OUT / 'data.json').read_text())
    report = json.loads((args.out / 'result.json').read_text())
    tok = load_tokenizer(SOURCE)
    for arm in ARMS:
        if time.time() > args.deadline - 300:
            report['regression_status'] = 'incomplete_deadline; promotion blocked'
            save(report, args.out / 'result.json')
            return
        model = load_endpoint(arm, args.out).to(args.device)
        report['arms'][arm]['regression'] = raw_generate(model, tok, data['regression'],
                    args.out / f'{arm}-regression.json', args.deadline)
        del model
        cleanup()
    report['regression_status'] = 'complete'
    save(report, args.out / 'result.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=RUN_OUT)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--deadline', type=float, required=True)
    parser.add_argument('--regression-only', action='store_true')
    args = parser.parse_args()
    regressions(args) if args.regression_only else run(args)
