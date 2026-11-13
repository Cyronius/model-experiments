"""Deterministic residual-coordinate and gated-FFN channel selection for Qwen3.5.

Two selection rules:
- `weights` (v1, data-free): embedding-column L2 for the residual, product of gate/up/down
  row/column L2 for FFN neurons. It left the 0.8B student at validation CE 8.9 (near random).
- `activation` (v2, Minitron-style): teacher statistics on calibration text. A residual
  coordinate scores by its mean squared output summed over every hidden-size RMSNorm (what
  downstream weights read); an FFN neuron by mean squared activation silu(gate)*up times the
  squared norm of its down-projection column over the kept coordinates. Optionally each
  hidden-size RMSNorm gain is multiplied by sqrt(kept-coordinate mean energy / all-coordinate
  mean energy) of its teacher input, so the pruned norm divides by about the same RMS.

No learned projection, layer removal, vocabulary pruning, or attention head pruning. The
teacher target at every depth is restricted to the selected residual coordinates.
Attention's internal dimensions remain unchanged.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from accounting import ROOT, architecture, count, target_config

# Qwen3.5-0.8B's own size: residual 1024, FFN 3584, 24 layers (exact size match since v2).
NATIVE_0_8B_PARAMETERS = 752_393_024


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def top_indices(score, n):
    # Stable ties select lower original coordinates. Keep original coordinate order.
    return torch.sort(torch.argsort(score, descending=True, stable=True)[:n]).values


def hidden_norms(model):
    """Every RMSNorm over the residual width, by state-dict prefix."""
    out = {}
    for i, block in enumerate(model.model.layers):
        out[f'model.layers.{i}.input_layernorm'] = block.input_layernorm
        out[f'model.layers.{i}.post_attention_layernorm'] = block.post_attention_layernorm
    out['model.norm'] = model.model.norm
    return out


@torch.no_grad()
def activation_stats(model, rows):
    """Teacher means over calibration tokens: norm input/output energy per coordinate, FFN activation energy."""
    from engine import stream
    width = model.config.hidden_size
    norms = hidden_norms(model)
    norm_in = {k: torch.zeros(width, dtype=torch.float64) for k in norms}
    norm_out = torch.zeros(width, dtype=torch.float64)
    ffn = [torch.zeros(model.config.intermediate_size, dtype=torch.float64) for _ in model.model.layers]
    hooks = []

    def on_norm(name):
        def hook(module, args, output):
            nonlocal norm_out
            norm_in[name] += args[0].double().reshape(-1, width).square().sum(0)
            norm_out += output.double().reshape(-1, width).square().sum(0)
        return hook

    def on_down(i):
        def hook(module, args):
            ffn[i] += args[0].double().reshape(-1, args[0].shape[-1]).square().sum(0)
        return hook

    for name, module in norms.items():
        hooks.append(module.register_forward_hook(on_norm(name)))
    for i, block in enumerate(model.model.layers):
        hooks.append(block.mlp.down_proj.register_forward_pre_hook(on_down(i)))
    tokens = 0
    try:
        for row in rows:
            ids = torch.tensor(np.asarray(row)[None], dtype=torch.long, device=next(model.parameters()).device)
            model.model.norm(stream(model, ids)[-1])
            tokens += ids.numel()
    finally:
        for h in hooks:
            h.remove()
    return dict(norm_output_energy=norm_out / tokens, ffn_activation_energy=torch.stack(ffn) / tokens,
                norm_input_energy=torch.stack([norm_in[k] for k in norms]) / tokens,
                norm_names=list(norms), tokens=tokens)


def activation_selections(get_tensor, config, hidden, width, stats, rescale_norms):
    residual = top_indices(stats['norm_output_energy'], hidden)
    ffn = []
    for layer in range(config['num_hidden_layers']):
        down = get_tensor(f'model.layers.{layer}.mlp.down_proj.weight')[residual].double().square().sum(0)
        ffn.append(top_indices(stats['ffn_activation_energy'][layer] * down, width))
    scales = {}
    if rescale_norms:
        full = stats['norm_input_energy'].shape[1]
        for name, energy in zip(stats['norm_names'], stats['norm_input_energy']):
            scales[name + '.weight'] = float(((energy[residual].sum() / hidden) / (energy.sum() / full)).sqrt())
    return residual, ffn, scales


def selections(get_tensor, config, hidden, width):
    emb = get_tensor('model.embed_tokens.weight')
    energy = torch.zeros(emb.shape[1], dtype=torch.float64)
    for part in emb.split(2048):
        energy += part.double().square().sum(0)
    residual = top_indices(energy, hidden)
    del emb
    ffn = []
    for layer in range(config['num_hidden_layers']):
        prefix = f'model.layers.{layer}.mlp.'
        gate = get_tensor(prefix + 'gate_proj.weight')[:, residual].double().square().sum(1)
        up = get_tensor(prefix + 'up_proj.weight')[:, residual].double().square().sum(1)
        down = get_tensor(prefix + 'down_proj.weight')[residual].double().square().sum(0)
        # Squared product of row/column L2 norms; no training data influences masks.
        ffn.append(top_indices(gate * up * down, width))
    return residual, ffn


def slice_parameter(name, tensor, residual, ffn):
    if name in ('model.embed_tokens.weight', 'lm_head.weight'):
        return tensor[:, residual]
    if name == 'model.norm.weight' or name.endswith(('.input_layernorm.weight', '.post_attention_layernorm.weight')):
        return tensor[residual]
    if '.mlp.' in name:
        layer = int(re.search(r'layers\.(\d+)\.', name).group(1))
        keep = ffn[layer]
        if name.endswith(('gate_proj.weight', 'up_proj.weight')):
            return tensor[keep][:, residual]
        if name.endswith('down_proj.weight'):
            return tensor[residual][:, keep]
        raise ValueError(f'Unsupported FFN parameter {name}')
    if name.endswith(('.self_attn.q_proj.weight', '.self_attn.k_proj.weight', '.self_attn.v_proj.weight',
                      '.linear_attn.in_proj_qkv.weight', '.linear_attn.in_proj_z.weight',
                      '.linear_attn.in_proj_a.weight', '.linear_attn.in_proj_b.weight')):
        return tensor[:, residual]
    if name.endswith(('.self_attn.o_proj.weight', '.linear_attn.out_proj.weight')):
        return tensor[residual]
    # Internal attention norms, convolution, and recurrence parameters are preserved.
    if '.self_attn.' in name or '.linear_attn.' in name:
        return tensor
    raise ValueError(f'Unclassified parameter {name}')


class TensorSource:
    def __init__(self, folder):
        self.files = {}
        for path in sorted(Path(folder).glob('*.safetensors')):
            with safe_open(path, framework='pt', device='cpu') as f:
                for key in f.keys():
                    canonical = key.replace('model.language_model.', 'model.')
                    if canonical in self.files:
                        raise ValueError(f'Duplicate source tensor {canonical}')
                    self.files[canonical] = (path, key)

    def get(self, name):
        if name == 'lm_head.weight' and name not in self.files:
            name = 'model.embed_tokens.weight'
        path, key = self.files[name]
        with safe_open(path, framework='pt', device='cpu') as f:
            return f.get_tensor(key)


def build_student(config, getter, hidden, target, stats=None, rescale_norms=False):
    config = dict(config.get('text_config', config))
    target_cfg = target_config(config, target, hidden)
    if stats is None:
        if rescale_norms:
            raise ValueError('Norm rescaling needs activation statistics')
        residual, ffn = selections(getter, config, hidden, target_cfg['intermediate_size'])
        scales = {}
        selection = 'weights: embedding-column L2; FFN product of gate/up/down L2; stable ties'
    else:
        residual, ffn, scales = activation_selections(getter, config, hidden, target_cfg['intermediate_size'],
                                                      stats, rescale_norms)
        selection = ('activation: residual by summed mean squared hidden-RMSNorm output; FFN by mean squared '
                     'silu(gate)*up times squared kept down-column norm; teacher calibration text; stable ties')
    cfg, cls = architecture(target_cfg)
    with torch.device('meta'):
        model = cls(cfg)
    state = {}
    for name, template in model.state_dict().items():
        value = slice_parameter(name, getter(name), residual, ffn).contiguous()
        if name in scales:
            # Qwen3.5 RMSNorm multiplies by (1 + weight).
            value = ((1 + value.double()) * scales[name] - 1).to(value.dtype)
        if value.shape != template.shape:
            raise ValueError(f'{name}: {value.shape} != {template.shape}')
        state[name] = value
    model.load_state_dict(state, assign=True, strict=True)
    model.tie_weights()
    # Rotary frequencies are nonpersistent buffers, absent from state_dict.
    # Recreate them on CPU after materializing parameters from the meta model.
    model.model.rotary_emb = type(model.model.rotary_emb)(config=cfg)
    if any(b.is_meta for b in model.buffers()):
        raise ValueError('Unmaterialized nonpersistent model buffer')
    assert sum(p.numel() for p in model.parameters()) == count(target_cfg)['total']
    return model, dict(residual_indices=residual.tolist(), ffn_indices=[x.tolist() for x in ffn],
                      target_config=target_cfg, parameter_counts=count(target_cfg),
                      selection=selection, rescaling=bool(scales), norm_scales=scales,
                      layers_removed=[], vocabulary_pruned=False,
                      attention_internal_dimensions_changed=False)


@torch.no_grad()
def plain_ce(model, rows, chunk=256):
    """Student-only next-token CE (FP32), for the start-quality probe."""
    from engine import stream
    total, n = 0.0, 0
    for row in rows:
        ids = torch.tensor(np.asarray(row)[None], dtype=torch.long)
        h = model.model.norm(stream(model, ids)[-1])[0, :-1]
        labels = ids[0, 1:]
        for a in range(0, len(labels), chunk):
            total += float(torch.nn.functional.cross_entropy(model.lm_head(h[a:a + chunk]).float(),
                                                             labels[a:a + chunk], reduction='sum'))
        n += len(labels)
    return total / n


def verify_artifact(folder):
    folder = Path(folder)
    manifest = json.loads((folder / 'pruning.json').read_text())
    for name, digest in manifest['artifact_sha256'].items():
        if sha256(folder / name) != digest:
            raise ValueError(f'Changed damaged checkpoint: {name}')
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--hidden', type=int, default=1024)
    ap.add_argument('--target', type=int, default=NATIVE_0_8B_PARAMETERS)
    ap.add_argument('--selection', choices=('weights', 'activation'), default='activation')
    ap.add_argument('--data', type=Path, help='Corpus for activation statistics and the start-quality probe')
    ap.add_argument('--train-rows', type=int, default=24, help='Train rows added to the calibration split for statistics')
    ap.add_argument('--probe-rows', type=int, default=16, help='Controller/validation rows per probe (0 skips)')
    ap.add_argument('--threads', type=int, default=8)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(args.threads)
    meta = json.loads((ROOT / 'metadata/Qwen3.5-2B.json').read_text())
    local_config = json.loads((args.source / 'config.json').read_text())
    if local_config != meta['config']:
        raise ValueError('Source configuration does not match verified revision')
    source_hashes = {p.name: sha256(p) for p in args.source.glob('*.safetensors')}
    downloaded = args.source / 'download_manifest.json'
    if downloaded.exists():
        expected = json.loads(downloaded.read_text())
        if expected['revision'] != meta['revision']:
            raise ValueError('Source revision mismatch')
        for name, digest in source_hashes.items():
            if expected['files'][name]['sha256'] != digest:
                raise ValueError(f'Source hash mismatch: {name}')
    else:
        raise ValueError('Source requires a download_manifest.json with pinned revision and file SHA256s')
    getter = TensorSource(args.source).get
    stats, extra = None, {}
    if args.selection == 'activation':
        if args.data is None:
            raise ValueError('Activation selection needs --data')
        from run import load_model
        data_meta = json.loads((args.data / 'manifest.json').read_text())
        for name, digest in data_meta['files_sha256'].items():
            if sha256(args.data / name) != digest:
                raise ValueError(f'Data hash mismatch: {name}')
        arrays = np.load(args.data / 'tokens.npz', allow_pickle=False)
        train_rows = sorted(np.random.default_rng(0).choice(len(arrays['train']), args.train_rows, replace=False).tolist())
        rows = list(arrays['calibration']) + [arrays['train'][i] for i in train_rows]
        teacher = load_model(args.source, 'cpu', torch.float32)
        stats = activation_stats(teacher, rows)
        del teacher
        extra = dict(calibration=dict(splits=['calibration (all)', 'train (seeded sample)'], train_rows=train_rows,
                                      tokens=stats['tokens'], data_manifest_sha256=sha256(args.data / 'manifest.json')))
        if args.probe_rows:
            # Choose norm rescaling on controller CE (in-mix unseen); validation is reported, not used.
            variants = {'weights': (None, False), 'activation': (stats, False), 'activation+norm_scale': (stats, True)}
            extra['probe'] = {}
            for name, (st, rescale) in variants.items():
                student, _ = build_student(meta['config'], getter, args.hidden, args.target, st, rescale)
                student.float()
                extra['probe'][name] = {split: plain_ce(student, arrays[split][:args.probe_rows])
                                        for split in ('controller', 'validation')}
                print(json.dumps({name: extra['probe'][name]}), flush=True)
                del student
            rescale = extra['probe']['activation+norm_scale']['controller'] < extra['probe']['activation']['controller']
        else:
            rescale = True
        extra['norm_rescale_rule'] = 'kept if it lowers probe controller CE' if args.probe_rows else 'on (no probe)'
        model, record = build_student(meta['config'], getter, args.hidden, args.target, stats, rescale)
    else:
        model, record = build_student(meta['config'], getter, args.hidden, args.target)
    args.output.mkdir(parents=True)
    model.save_pretrained(args.output, max_shard_size='2GB', safe_serialization=True)
    if stats is not None:
        save_file({k: v.contiguous() for k, v in stats.items() if torch.is_tensor(v)}, args.output / 'activation-stats.safetensors')
    record.update(extra)
    record.update(source_id=meta['model_id'], source_revision=meta['revision'],
                  source_sha256=source_hashes, code_sha256=sha256(__file__),
                  artifact_sha256={p.name: sha256(p) for p in args.output.iterdir() if p.is_file()})
    (args.output / 'pruning.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({'output': str(args.output), 'parameters': record['parameter_counts'], 'rescaling': record['rescaling'],
                      'manifest_sha256': sha256(args.output / 'pruning.json')}))


if __name__ == '__main__':
    main()
