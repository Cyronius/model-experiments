"""Allocation pilot: spend the shared-broad budget on less residual cut plus MLP pruning.

P0 is the existing shared-broad projection: residual width 1,520 from the broad
64k PCA basis, all 6,144 MLP neurons per layer. P1 keeps residual width 1,792 from
the same basis and the same constant RMS fit, then prunes every MLP to the widest
common size whose stored scalars do not exceed P0's. Neurons are ranked by
E[phi^2] * |down column|^2 and the down projection is refit by ridge least squares
on the pruned model's own inputs, as in build_pruned.py. See PAIR_PLAN_20261026.md.

Stages: plan (draft protocol and budget, no weights), freeze (after peer review,
with a measured runtime bound), controls (P0 reproduction and full-width
identity), build (layer-by-layer pruning; long, lease the CPU first), check.
Model-run stages refuse a draft protocol and any code or input changed since freezing.
"""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import torch
from torch import nn

from local_projection_models import build, fit_rms, recipe_bases, stored_scalars

ROOT = Path(__file__).resolve().parent
LOCAL = ROOT / 'results/local-projection-20261025'
OUT = ROOT / 'results/allocation-pilot-20261027'
P0_RECIPE = LOCAL / 'shared-broad-recipe.pt'
P1_WIDTH = 1792
RIDGE = 1e-3          # relative to the mean diagonal of the kept Gram, as in build_pruned.py
SKIP_FIRST = True     # position 0 is excluded from the Gram, as in build_pruned.py


class _Stop(Exception):
    pass


def log(msg):
    print(time.strftime('%H:%M:%S'), msg, flush=True)


def save(obj, path):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2) + '\n')
    tmp.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def meta_source():
    """The pinned source text model on the meta device: shapes only, no weights."""
    from transformers import AutoConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
    from local_projection_probe import SOURCE
    config = AutoConfig.from_pretrained(SOURCE, local_files_only=True).get_text_config()
    with torch.device('meta'):
        return Qwen3_5ForCausalLM(config).eval().requires_grad_(False)


def projected_scalars(model, width):
    """Stored scalars of the shared projection at this width, before any pruning."""
    d = model.config.hidden_size
    basis = torch.zeros(d, width, device=model.model.embed_tokens.weight.device)
    layers = len(model.model.layers)
    return stored_scalars(build(model, [basis] * layers, [1.] * (2 * layers + 1)))


def mlp_budget(factory, width, reference):
    """Widest common MLP width at this residual width with stored scalars <= reference.

    Each kept neuron stores one gate row, one up row and one down column; Qwen3.5
    MLPs have no biases, so scalars fall by exactly 3 * width per neuron per layer.
    """
    model = factory()
    neurons = model.config.intermediate_size
    layers = len(model.model.layers)
    if any(p.bias is not None for l in model.model.layers
           for p in (l.mlp.gate_proj, l.mlp.up_proj, l.mlp.down_proj)):
        raise ValueError('Budget formula assumes bias-free MLPs')
    full = projected_scalars(model, width)
    per_neuron = 3 * width * layers
    cut = max(0, math.ceil((full - reference) / per_neuron))
    if cut >= neurons:
        raise ValueError('Budget cannot be met by MLP pruning alone')
    mlp = neurons - cut
    return {'width': width, 'mlp_width': mlp, 'neurons': neurons, 'layers': layers,
            'unpruned_scalars': full, 'scalars': full - cut * per_neuron,
            'reference_scalars': reference, 'slack': reference - (full - cut * per_neuron),
            'per_neuron_scalars': per_neuron}


def shared_recipe(q, states, rows, width, layers):
    """A shared-basis recipe at any width, fitted exactly as local_projection_probe.fit does."""
    # Samples hold up to 32 states per conversation; restore token weighting.
    weights = torch.tensor([len(rows[g]['input_ids']) / min(32, len(rows[g]['input_ids']))
                            for g in states['groups'].tolist()])
    p = q[:, :width].contiguous()
    scales = [fit_rms(x, p, weights=weights)[0] for x in states['samples']]
    return {'basis': q.unsqueeze(0).expand(layers, -1, -1), 'width': width, 'local': False,
            'scales': scales, 'predictors': None, 'source': 'Qwen/Qwen3.5-2B', 'calibration': 'broad'}


@torch.no_grad()
def mlp_gram(model, layer, rows, skip_first=SKIP_FIRST):
    """Gram matrix of one MLP's hidden activations on the model's own inputs.

    The forward stops at that MLP, so earlier layers are always the current
    (already pruned) ones.
    """
    mlp = model.model.layers[layer].mlp
    n = mlp.gate_proj.out_features
    gram = torch.zeros(n, n, dtype=torch.float64)
    total = torch.zeros(n, dtype=torch.float64)
    count = 0

    def hook(module, args):
        nonlocal count
        x = args[0][:, 1:] if skip_first else args[0]
        x = x.reshape(-1, x.shape[-1]).float()
        phi = (module.act_fn(module.gate_proj(x)) * module.up_proj(x)).double()
        gram.addmm_(phi.T, phi)
        total.add_(phi.sum(0))
        count += len(phi)
        raise _Stop

    handle = mlp.register_forward_pre_hook(hook)
    try:
        for row in rows:
            try:
                model.model(input_ids=torch.tensor([row['input_ids']]), use_cache=False)
            except _Stop:
                pass
    finally:
        handle.remove()
    return gram, total, count


def select_and_refit(gram, down, keep_count, ridge=RIDGE):
    """Keep the most important neurons and refit their down projection.

    The target is the unpruned MLP output on the same inputs, so its cross term
    with the kept activations is gram[keep] @ down.T; no output states are stored.
    """
    gram, d = gram.double(), down.double()
    if not 0 < keep_count <= len(gram) or ridge < 0:
        raise ValueError(f'Need 0 < keep_count <= {len(gram)} and ridge >= 0')
    if not torch.isfinite(gram).all():
        raise ValueError('Gram matrix is not finite')
    importance = gram.diagonal() * d.square().sum(0)
    keep = importance.topk(keep_count).indices.sort().values
    a = gram[keep][:, keep]
    if not a.diagonal().mean() > 0:
        raise ValueError('Kept neurons have no activation energy')
    if ridge:
        a = a + ridge * a.diagonal().mean() * torch.eye(len(keep), dtype=a.dtype)
    w = torch.linalg.solve(a, gram[keep] @ d.T)
    return keep, w.T.float().contiguous()


def refit_r2(gram, total, count, down, keep, new_down):
    """Held-in R^2 of the pruned MLP output against the unpruned one, from Grams."""
    g, d, w = gram.double(), down.double(), new_down.double()
    target = (d @ g @ d.T).trace()
    cross = (w @ g[keep] @ d.T).trace()
    fitted = (w @ g[keep][:, keep] @ w.T).trace()
    centered = target - count * (d @ total / count).square().sum()
    return (1 - (target - 2 * cross + fitted) / centered).item()


def prune_mlp(mlp, keep, down):
    def linear(w):
        new = nn.Linear(w.shape[1], w.shape[0], bias=False, device=w.device, dtype=w.dtype)
        new.weight = nn.Parameter(w.contiguous(), requires_grad=False)
        return new
    mlp.gate_proj = linear(mlp.gate_proj.weight[keep])
    mlp.up_proj = linear(mlp.up_proj.weight[keep])
    mlp.down_proj = linear(down)
    mlp.intermediate_size = len(keep)


@torch.no_grad()
def construct(model, mlp_width, rows, ridge=RIDGE, skip_first=SKIP_FIRST, progress=None, deadline=None, done=()):
    """Prune every MLP of an already projected model, in layer order.

    Keeping every neuron leaves the MLP untouched (no ridge refit), so the
    no-pruning setting is an exact identity. Each layer replays the prefix of
    every calibration row, about half a full forward per layer on average.
    The deadline is a soft bound checked between layers, not a wall-clock cap.
    done holds (keep, down, report) for layers finished by an interrupted run;
    they are applied as saved, so a resumed build sees the same upstream model.
    progress(layer, keep, down, report) is called after each new layer.
    """
    reports, keeps, downs = [], [], []
    for i, layer in enumerate(model.model.layers):
        down = layer.mlp.down_proj.weight
        if i < len(done):
            keep, new_down, report = done[i]
        elif mlp_width == down.shape[1]:
            keep, new_down, report = torch.arange(mlp_width), down.clone(), None
        else:
            if deadline is not None and time.monotonic() > deadline:
                raise TimeoutError(f'Runtime bound exceeded before layer {i}')
            gram, total, count = mlp_gram(model, i, rows, skip_first)
            if count == 0:
                raise ValueError('No calibration tokens reached the MLP')
            keep, new_down = select_and_refit(gram, down, mlp_width, ridge)
            report = {'layer': i, 'tokens': count, 'r2_refit': refit_r2(gram, total, count, down, keep, new_down),
                      'r2_no_refit': refit_r2(gram, total, count, down, keep, down[:, keep])}
            del gram
            if progress:
                progress(i, keep, new_down, report)
        if len(keep) != down.shape[1]:
            prune_mlp(layer.mlp, keep, new_down)
        keeps.append(keep)
        downs.append(new_down)
        if report is not None:
            reports.append(report)
    model.config.intermediate_size = mlp_width
    return model, torch.stack(keeps), torch.stack(downs), reports


def apply_pruning(model, recipe):
    for layer, keep, down in zip(model.model.layers, recipe['keep'], recipe['down'], strict=True):
        prune_mlp(layer.mlp, keep, down)
    model.config.intermediate_size = recipe['mlp_width']
    return model


def load_p1(path=None, loader=None):
    """Rebuild P1 from the pinned source and its saved recipe."""
    if loader is None:
        from local_projection_probe import load_source as loader
    recipe = torch.load(path or OUT / 'p1-recipe.pt', map_location='cpu', weights_only=True)
    projection = recipe['projection']
    model = build(loader(), recipe_bases(projection), projection['scales'])
    return apply_pruning(model, recipe)


def load_p0(loader=None):
    if loader is None:
        from local_projection_probe import load_source as loader
    recipe = torch.load(P0_RECIPE, map_location='cpu', weights_only=True)
    return build(loader(), recipe_bases(recipe), recipe['scales'])


def broad_inputs():
    data = json.loads((LOCAL / 'data.json').read_text())
    q = torch.load(LOCAL / 'broad-pca.pt', weights_only=True)['q']
    states = torch.load(LOCAL / 'broad-states.pt', weights_only=True)
    return data, q, states


CODE = ['allocation_pilot.py', 'local_projection_models.py', 'local_projection_probe.py', 'audit_rotation.py']
INPUTS = [LOCAL / n for n in ('broad-pca.pt', 'broad-states.pt', 'data.json', 'shared-broad-recipe.pt', 'reference.pt')]
SOURCE_FILES = ['config.json', 'model.safetensors-00001-of-00001.safetensors', 'tokenizer.json']
MISSING = ['peer review of this protocol', 'generation-suite gates and thresholds (plan 0a)',
           'runtime bound from a timed measurement under dibs']


def require_frozen(out):
    """Model-run stages refuse a draft protocol, or code/inputs changed since freezing."""
    path = out / 'protocol.json'
    protocol = json.loads(path.read_text())
    if protocol.get('status') != 'frozen':
        raise RuntimeError(f"Protocol is a draft; missing: {protocol.get('missing')}")
    for name, digest in protocol['code_sha256'].items():
        if sha(ROOT / name) != digest:
            raise RuntimeError(f'{name} changed after the protocol was frozen')
    for name, digest in protocol['input_sha256'].items():
        if sha(ROOT.parent / name) != digest:
            raise RuntimeError(f'{name} changed after the protocol was frozen')
    return protocol, sha(path)


def stage_freeze(out, reviewed_by, max_hours, thresholds=None):
    """Freeze the build with a reviewer and a positive measured runtime bound.

    The build does not depend on the generation suite. Without its gates file,
    the protocol stays frozen for construction only and keeps that item in
    'missing', so no evaluation claim can rest on it.
    """
    from local_projection_probe import SOURCE
    path = out / 'protocol.json'
    protocol = json.loads(path.read_text())
    if protocol.get('status') == 'frozen':
        raise ValueError('Protocol is already frozen')
    if not reviewed_by or (thresholds is not None and not Path(thresholds).is_file()):
        raise ValueError('Freezing needs a reviewer, and any gates file named must exist')
    if not (isinstance(max_hours, (int, float)) and math.isfinite(max_hours) and max_hours > 0):
        raise ValueError('The runtime bound must be a positive number of hours')
    base = ROOT.parent.resolve()
    inputs = INPUTS + [Path(SOURCE) / n for n in SOURCE_FILES] + ([Path(thresholds)] if thresholds else [])
    protocol.update(status='frozen', reviewed_by=reviewed_by, max_build_hours=max_hours,
                    missing=[] if thresholds else ['0a generation gates, required before any P1 evaluation claim'],
                    generation_gates=str(Path(thresholds).resolve().relative_to(base)) if thresholds else None,
                    frozen_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                    code_sha256={n: sha(ROOT / n) for n in CODE},
                    input_sha256={str(Path(p).resolve().relative_to(base)): sha(p) for p in inputs})
    save(protocol, path)
    log('Protocol frozen')


def stage_timing(out, rows=24):
    """Bounded measurement behind the runtime bound: Gram passes at three depths on a row subset."""
    from local_projection_probe import load_source
    protocol = json.loads((out / 'protocol.json').read_text())
    mlp_width = protocol['arms']['P1']['mlp_width']
    start = time.perf_counter()
    data, q, states = broad_inputs()
    layers = (len(states['samples']) - 1) // 2
    projection = shared_recipe(q, states, data['broad'], P1_WIDTH, layers)
    del states
    model = build(load_source(), recipe_bases(projection), projection['scales'])
    setup = time.perf_counter() - start
    subset = data['broad'][::max(1, len(data['broad']) // rows)][:rows]
    tokens = sum(len(r['input_ids']) for r in subset)
    total = sum(len(r['input_ids']) for r in data['broad'])
    depth = {}
    for layer in (0, layers // 2, layers - 1):
        t = time.perf_counter()
        gram, moments, count = mlp_gram(model, layer, subset)
        depth[layer] = time.perf_counter() - t
    down = model.model.layers[layers - 1].mlp.down_proj.weight
    t = time.perf_counter()
    keep, new_down = select_and_refit(gram, down, mlp_width)
    refit_r2(gram, moments, count, down, keep, new_down)
    refit_r2(gram, moments, count, down, keep, down[:, keep])
    solve = time.perf_counter() - t
    # Prefix replay makes a layer's Gram pass grow about linearly with depth.
    slope = (depth[layers - 1] - depth[0]) / (layers - 1)
    passes = sum(depth[0] + slope * i for i in range(layers)) * total / tokens
    estimate = setup + passes + solve * layers
    result = {'rows': len(subset), 'tokens': tokens, 'calibration_tokens': total, 'threads': torch.get_num_threads(),
              'setup_seconds': setup, 'gram_seconds_by_layer': depth, 'solve_and_r2_seconds': solve,
              'estimated_build_hours': estimate / 3600}
    save(result, out / 'timing.json')
    log(f"Estimated build: {estimate / 3600:.2f} h at {torch.get_num_threads()} threads")


def stage_plan(out):
    out.mkdir(parents=True, exist_ok=True)
    path = out / 'protocol.json'
    if path.exists():
        raise ValueError('Protocol exists; edit it by hand before freezing, or remove it')
    p0_width = torch.load(P0_RECIPE, map_location='cpu', weights_only=True)['width']
    p0 = projected_scalars(meta_source(), p0_width)
    measured = json.loads((LOCAL / 'shared-broad-result.json').read_text())['stored_scalars']
    if p0 != measured:
        raise RuntimeError(f'Meta count {p0} differs from the measured shared-broad model {measured}')
    budget = mlp_budget(meta_source, P1_WIDTH, p0)
    protocol = {
        'status': 'draft',
        'missing': MISSING,
        'plan': 'manifold-compression/PAIR_PLAN_20261026.md, section B',
        'arms': {'P0': {'residual_width': 1520, 'mlp_width': budget['neurons'], 'stored_scalars': p0},
                 'P1': {'residual_width': P1_WIDTH, 'mlp_width': budget['mlp_width'],
                        'stored_scalars': budget['scalars']}},
        'budget': budget,
        'source': 'models/manifold-audit/Qwen3.5-2B (pinned; hashes in local-projection-20261025)',
        'basis': {'file': str((LOCAL / 'broad-pca.pt').relative_to(ROOT.parent)), 'sha256': sha(LOCAL / 'broad-pca.pt')},
        'rms': 'constant per-site scales from fit_rms on broad-states.pt samples, token-weighted as in local_projection_probe.fit',
        'pruning': {'importance': 'E[phi^2] * |down column|^2', 'refit': 'ridge least squares to the unpruned MLP output',
                    'ridge_relative_to_mean_diag': RIDGE, 'skip_first_position': SKIP_FIRST,
                    'inputs': 'pruned model, layer order', 'calibration': 'local-projection-20261025 data.json broad rows',
                    'precision': 'FP32 forward, FP64 Gram and solve'},
        'gates': ['P0 recipe rebuilt here matches shared-broad-recipe.pt (basis exactly, scales within 1e-6 relative)',
                  'full-width rotation without pruning reproduces stock validation logits (numerical_pass)',
                  'P1 stored scalars <= P0 stored scalars',
                  'P1 cache check passes; two independent rebuilds give identical logits',
                  'evaluation: fresh generation suite gates plus tf_metrics; no healing before both pass'],
        'cloud_spend': 0,
    }
    save(protocol, path)
    log(f"P0 {p0:,} scalars; P1 width {P1_WIDTH}, MLP {budget['mlp_width']}, {budget['scalars']:,} scalars, slack {budget['slack']:,}")


def stage_controls(out):
    from local_projection_probe import load_source, numerical
    from audit_rotation import cache_check
    _, protocol_sha = require_frozen(out)
    data, q, states = broad_inputs()
    layers = (len(states['samples']) - 1) // 2
    saved = torch.load(P0_RECIPE, map_location='cpu', weights_only=True)
    rebuilt = shared_recipe(q, states, data['broad'], saved['width'], layers)
    new, old = torch.tensor(rebuilt['scales'], dtype=torch.float64), torch.tensor(saved['scales'], dtype=torch.float64)
    # Reduction order can depend on the thread count, so exact equality is reported, not required.
    result = {'p0_basis_equal': torch.equal(recipe_bases(rebuilt)[0], recipe_bases(saved)[0]),
              'p0_scales_exact': torch.equal(new, old),
              'p0_scales_max_relative_error': ((new - old).abs() / old.abs()).max().item()}
    # Full-width rotation with no pruning: the P1 construction path must reproduce stock.
    model = build(load_source(), [q] * layers, [1.] * (2 * layers + 1))
    result['fullwidth'] = numerical(model, LOCAL, data)
    result['fullwidth']['cache'] = cache_check(model, torch.tensor([data['validation'][0]['input_ids'][:96]]))
    result['passed'] = (result['p0_basis_equal'] and result['p0_scales_max_relative_error'] <= 1e-6
                        and result['fullwidth']['numerical_pass'] and result['fullwidth']['cache']['numerical_pass'])
    result['protocol_sha256'] = protocol_sha
    save(result, out / 'controls.json')
    if not result['passed']:
        raise RuntimeError(f'Controls failed: {result}')
    log('Controls passed')


def stage_build(out):
    from local_projection_probe import load_source
    protocol, protocol_sha = require_frozen(out)
    controls = json.loads((out / 'controls.json').read_text())
    if not controls.get('passed') or controls.get('protocol_sha256') != protocol_sha:
        raise RuntimeError('Controls have not passed under the current frozen protocol')
    path = out / 'p1-recipe.pt'
    if path.exists():
        raise ValueError('P1 recipe exists; refusing to overwrite')
    data, q, states = broad_inputs()
    layers = (len(states['samples']) - 1) // 2
    projection = shared_recipe(q, states, data['broad'], P1_WIDTH, layers)
    del states
    model = build(load_source(), recipe_bases(projection), projection['scales'])
    mlp_width = protocol['arms']['P1']['mlp_width']
    # Finished layers survive an interruption; a resume uses them only under the same frozen protocol.
    parts = out / 'p1-layers'
    parts.mkdir(exist_ok=True)
    done = []
    for i in range(layers):
        part = parts / f'layer-{i:02d}.pt'
        if not part.exists():
            break
        saved = torch.load(part, weights_only=True)
        if saved['protocol_sha256'] != protocol_sha:
            raise RuntimeError(f'{part.name} was built under a different protocol')
        done.append((saved['keep'], saved['down'], saved['report']))
    if done:
        log(f'Resuming after {len(done)} finished layers')

    def checkpoint(i, keep, down, report):
        tmp = parts / f'layer-{i:02d}.pt.tmp'
        torch.save({'keep': keep, 'down': down, 'report': report, 'protocol_sha256': protocol_sha}, tmp)
        tmp.replace(parts / f'layer-{i:02d}.pt')
        log(f"L{i}: R2 refit {report['r2_refit']:.4f}, no refit {report['r2_no_refit']:.4f}")

    start = time.perf_counter()
    model, keep, down, report = construct(model, mlp_width, data['broad'], progress=checkpoint, done=done,
                                          deadline=time.monotonic() + 3600 * protocol['max_build_hours'])
    scalars = stored_scalars(model)
    if scalars != protocol['arms']['P1']['stored_scalars']:
        raise RuntimeError(f'Stored scalars {scalars} differ from the frozen plan')
    torch.save({'projection': projection, 'mlp_width': mlp_width, 'keep': keep, 'down': down,
                'ridge': RIDGE, 'skip_first': SKIP_FIRST, 'stored_scalars': scalars}, path)
    save({'layers': report, 'seconds': time.perf_counter() - start, 'threads': torch.get_num_threads(),
          'stored_scalars': scalars, 'recipe_sha256': sha(path)}, out / 'build-report.json')
    log(f'Saved {path.name}')


def stage_check(out):
    from local_projection_probe import numerical
    from audit_rotation import cache_check
    protocol, _ = require_frozen(out)
    data = json.loads((LOCAL / 'data.json').read_text())
    first, second = load_p1(out / 'p1-recipe.pt'), load_p1(out / 'p1-recipe.pt')
    ids = torch.tensor([data['validation'][0]['input_ids'][:96]])
    with torch.no_grad():
        same = torch.equal(first(ids, use_cache=False).logits, second(ids, use_cache=False).logits)
    del second
    result = {'stored_scalars': stored_scalars(first), 'p0_stored_scalars': protocol['arms']['P0']['stored_scalars'],
              'rebuild_identical': same, 'cache': cache_check(first, ids), 'validation': numerical(first, LOCAL, data)}
    result['within_budget'] = result['stored_scalars'] <= result['p0_stored_scalars']
    save(result, out / 'p1-check.json')
    if not (same and result['within_budget'] and result['cache']['numerical_pass']):
        raise RuntimeError(f'P1 check failed: {result}')
    log(f"P1 check passed; validation KL {result['validation']['mean_kl']:.4f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('stage', choices=['plan', 'timing', 'freeze', 'controls', 'build', 'check'])
    ap.add_argument('--out', type=Path, default=OUT)
    ap.add_argument('--threads', type=int, default=16)
    ap.add_argument('--reviewed-by', help='freeze: who reviewed the protocol')
    ap.add_argument('--max-hours', type=float, help='freeze: runtime bound for the build, from the timing stage')
    ap.add_argument('--gates', help='freeze: generation-suite gates file, if it already exists')
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.stage == 'freeze':
        stage_freeze(args.out, args.reviewed_by, args.max_hours, args.gates)
    else:
        globals()[f'stage_{args.stage}'](args.out)


if __name__ == '__main__':
    main()
