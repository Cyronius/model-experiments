"""Experimental per-layer residual projections and state-dependent RMS correction.

All attention heads, recurrent states and MLP neurons are retained. A bridge
on each block's MLP skip path changes its residual coordinates. The tied vocabulary
matrix stays in the first basis; the final dense gain maps back to that basis.
These research models use an explicit recipe + state_dict loader, not HF export.
"""
import copy

import torch
from torch import nn

from audit_rotation import norm_epsilon, norm_gain


def rms_features(z, n=16):
    z = z.float()
    rms = (z.square().mean(-1, keepdim=True) + 1e-6).sqrt()
    t = z[..., :min(n, z.shape[-1])] / rms
    return torch.cat([t, t.square(), rms.log()], -1)


class CorrectedNorm(nn.Module):
    def __init__(self, width, eps, scale=1., predictor=None, matrix=None):
        super().__init__()
        self.eps = eps
        # Retain a gain vector in parameter accounting, as in the existing model.
        self.weight = nn.Parameter(torch.ones(width), requires_grad=False)
        self.register_buffer("scale", torch.tensor(float(scale)))
        self.predictor = predictor is not None
        if predictor is not None:
            for name in ("mean", "std", "coef", "bounds"):
                self.register_buffer(name, predictor[name].clone())
        self.matrix = None if matrix is None else nn.Parameter(matrix, requires_grad=False)

    def factor(self, x):
        if not self.predictor:
            return self.scale
        f = (rms_features(x) - self.mean) / self.std
        return (self.scale + f @ self.coef).clamp(self.bounds[0], self.bounds[1]).unsqueeze(-1)

    def forward(self, x):
        y = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        y = y * self.weight * self.factor(x)
        if self.matrix is not None:
            y = y @ self.matrix
        return y.to(x.dtype)


class TransportLayer(nn.Module):
    def __init__(self, inner, transport):
        super().__init__()
        self.inner = inner
        self.transport = nn.Parameter(transport, requires_grad=False)

    def forward(self, hidden_states, position_embeddings, attention_mask=None,
                position_ids=None, past_key_values=None, **kwargs):
        layer = self.inner
        residual = hidden_states
        h = layer.input_layernorm(hidden_states)
        if layer.block_type == 'linear_attention':
            h = layer.linear_attn(hidden_states=h, cache_params=past_key_values,
                                  attention_mask=attention_mask, **kwargs)
        else:
            h, _ = layer.self_attn(hidden_states=h, attention_mask=attention_mask,
                position_ids=position_ids, past_key_values=past_key_values,
                position_embeddings=position_embeddings, **kwargs)
        h = residual + h
        # The MLP writer already targets the next basis. Only the skip path
        # needs a bridge; projecting the whole sum would cut the update twice.
        return h @ self.transport + layer.mlp(layer.post_attention_layernorm(h))


def norms(model):
    layers = [getattr(l, "inner", l) for l in model.model.layers]
    return [n for l in layers for n in (l.input_layernorm, l.post_attention_layernorm)] + [model.model.norm]


def stored_scalars(model):
    # Include fixed bridges and learned predictor buffers; don't count duplicate
    # tied weights or ephemeral source bases used only during construction.
    return sum(p.numel() for p in model.parameters()) + sum(b.numel() for n, b in model.named_buffers()
            if any(s in n for s in ('.mean', '.std', '.coef', '.bounds', '.scale')))


@torch.no_grad()
def build(model, bases, scales, local=False, predictors=None, decoders=None):
    """Mutate an FP32 Qwen text model into its projected counterpart."""
    if model.config.model_type != "qwen3_5_text":
        raise ValueError("This experiment supports Qwen3.5 text only")
    d, k = bases[0].shape
    layers = model.model.layers
    if len(bases) != len(layers) or len(scales) != 2 * len(layers) + 1:
        raise ValueError("Wrong number of bases or scales")
    if decoders is not None and (len(decoders)!=len(scales) or any(v.shape!=(d,k) for v in decoders)):
        raise ValueError('Need one full-state decoder per residual input site')
    if model.model.embed_tokens.weight is not model.lm_head.weight:
        raise ValueError("This experiment expects Qwen's tied embedding")
    final_gain = norm_gain(model.model.norm)
    final_eps = norm_epsilon(model.model.norm)
    model.config.head_dim = model.config.head_dim or d // model.config.num_attention_heads
    embed = model.model.embed_tokens.weight @ bases[0]
    model.model.embed_tokens = nn.Embedding.from_pretrained(embed, freeze=True)
    head = nn.Linear(k, embed.shape[0], bias=False, device="meta")
    head.weight = model.model.embed_tokens.weight
    model.lm_head = head

    def linear(old, w, b):
        new = nn.Linear(w.shape[1], w.shape[0], bias=b is not None, device="meta")
        new.weight = nn.Parameter(w, requires_grad=False)
        if b is not None:
            new.bias = nn.Parameter(b.clone(), requires_grad=False)
        return new

    for i, layer in enumerate(layers):
        p = bases[i]
        if hasattr(layer, "linear_attn"):
            mixer = layer.linear_attn
            reads, write = ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"), "out_proj"
        else:
            mixer = layer.self_attn
            reads, write = ("q_proj", "k_proj", "v_proj"), "o_proj"
        for parent, read_names, write_name, norm, site in [
                (mixer, reads, write, layer.input_layernorm, 2*i),
                (layer.mlp, ("gate_proj", "up_proj"), "down_proj", layer.post_attention_layernorm, 2*i+1)]:
            gain = norm_gain(norm)
            decoder = p if decoders is None else decoders[site]
            for name in read_names:
                old = getattr(parent, name)
                setattr(parent, name, linear(old, (old.weight * gain) @ decoder, old.bias))
            old = getattr(parent, write_name)
            destination = bases[i+1] if local and site % 2 == 1 and i < len(layers)-1 else p
            setattr(parent, write_name, linear(old, destination.T @ old.weight,
                    None if old.bias is None else destination.T @ old.bias))
            corrected = CorrectedNorm(k, norm_epsilon(norm), scales[site],
                                      None if predictors is None else predictors[site])
            if site % 2 == 0:
                layer.input_layernorm = corrected
            else:
                layer.post_attention_layernorm = corrected
            if hasattr(parent, "hidden_size"):
                parent.hidden_size = k
        layer.hidden_size = k
        if local and i < len(layers) - 1:
            # Compute in double precision; full-width maps must be exact rotations.
            t = (p.double().T @ bases[i+1].double()).float()
            layers[i] = TransportLayer(layer, t)
    final_decoder = bases[-1] if decoders is None else decoders[-1]
    matrix = (final_decoder.double().T @ (final_gain.double()[:, None] * bases[0].double())).float()
    model.model.norm = CorrectedNorm(k, final_eps, scales[-1],
        None if predictors is None else predictors[-1], matrix)
    model.config.hidden_size = k
    model.eval().requires_grad_(False)
    return model


def recipe_bases(recipe):
    q = recipe["basis"]
    return [q[i, :, :recipe["width"]].contiguous() for i in range(len(q))]


def from_recipe(source, recipe, loader):
    return build(loader(source), recipe_bases(recipe), recipe["scales"],
                 local=recipe["local"], predictors=recipe.get("predictors"),decoders=recipe.get('decoders'))


def fit_rms(x, p, eps=1e-6, weights=None):
    """Ridge-fit a bounded token-dependent correction from calibration states."""
    z = x.float() @ p
    target = ((z.square().mean(-1) + eps) / (x.float().square().mean(-1) + eps)).sqrt()
    weights = torch.ones(len(x)) if weights is None else weights.float()
    weights = weights / weights.sum()
    scale = (target * weights).sum().item()
    f = rms_features(z)
    mean = (f * weights[:, None]).sum(0)
    std = ((f-mean).square() * weights[:, None]).sum(0).sqrt().clamp_min(1e-3)
    f = (f - mean) / std
    gram = (f.T.double() * weights.double()) @ f.double()
    coef = torch.linalg.solve(gram + .01 * torch.eye(gram.shape[0]),
                             f.T.double() @ ((target.double()-scale)*weights.double())).float()
    predictor = {"mean": mean, "std": std, "coef": coef,
                 "bounds": torch.quantile(target, torch.tensor([.001, .999]))}
    return scale, predictor


def rms_error(x, p, scale, predictor):
    z = x.float() @ p
    target = ((z.square().mean(-1)+1e-6)/(x.float().square().mean(-1)+1e-6)).sqrt()
    mod = CorrectedNorm(p.shape[1], 1e-6, scale, predictor)
    estimate = mod.factor(z).squeeze(-1)
    return {"constant_mse": (target-scale).square().mean().item(),
            "predicted_mse": (target-estimate).square().mean().item(),
            "target_std": target.std().item(), "scale": scale}
