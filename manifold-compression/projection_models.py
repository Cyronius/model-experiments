"""Serializable output adapters for projected Qwen3.5 and Llama models.

The saved HF directory includes this module and an AutoModel mapping. External
callers load it with trust_remote_code=True; in-repo loading imports this module
directly. No changes to installed transformers are required.
"""
import torch
from torch import nn
from transformers import (AutoConfig, AutoModelForCausalLM, LlamaForCausalLM,
                          MistralForCausalLM, Qwen3_5ForCausalLM)
from transformers.conversion_mapping import (get_checkpoint_conversion_mapping,
                                              register_checkpoint_conversion_mapping)


class MatrixRMSNorm(nn.Module):
    """RMS normalization, adjustable coordinate gains, then a full mixing matrix."""
    def __init__(self, width, eps, offset, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.offset = offset
        self.weight = nn.Parameter(torch.full((width,), 1.0 - offset, device=device, dtype=dtype))
        self.matrix = nn.Parameter(torch.eye(width, device=device, dtype=dtype))

    def forward(self, x):
        h = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        h = h * (self.weight.float() + self.offset)
        return (h @ self.matrix.float()).to(x.dtype)


def _install_from_config(model, offset):
    mode = getattr(model.config, "manifold_output_norm", None)
    if mode is None:
        return
    if mode != "matrix_v1":
        raise ValueError(f"Unsupported manifold output norm: {mode}")
    norm = model.model.norm
    eps = norm.eps if hasattr(norm, "eps") else norm.variance_epsilon
    model.model.norm = MatrixRMSNorm(model.config.hidden_size, eps, offset,
                                    device=norm.weight.device, dtype=norm.weight.dtype)


class ManifoldQwen3_5ForCausalLM(Qwen3_5ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        _install_from_config(self, 1.0)


class ManifoldLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        _install_from_config(self, 0.0)


class ManifoldMistralForCausalLM(MistralForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        _install_from_config(self, 0.0)


ManifoldQwen3_5ForCausalLM.register_for_auto_class("AutoModelForCausalLM")
ManifoldLlamaForCausalLM.register_for_auto_class("AutoModelForCausalLM")
ManifoldMistralForCausalLM.register_for_auto_class("AutoModelForCausalLM")

CLASSES = {"qwen3_5_text": ManifoldQwen3_5ForCausalLM, "llama": ManifoldLlamaForCausalLM,
           "mistral": ManifoldMistralForCausalLM}

# Transformers skips native key conversions for custom classes unless explicitly
# registered. A loaded Qwen checkpoint saves back to model.language_model.*;
# preserve its native prefix conversion in both local and bundled-code loading.
for _model_type, _cls in CLASSES.items():
    _mapping = get_checkpoint_conversion_mapping(_model_type)
    if _mapping is not None:
        register_checkpoint_conversion_mapping(_cls.__name__, _mapping, overwrite=True)


@torch.no_grad()
def install_output_matrix(model, matrix):
    """Promote an existing model without copying its multi-billion-weight body."""
    if model.config.model_type not in CLASSES:
        raise ValueError(f"Unsupported model: {model.config.model_type}")
    norm = model.model.norm
    eps = norm.eps if hasattr(norm, "eps") else norm.variance_epsilon
    offset = 1.0 if model.config.model_type == "qwen3_5_text" else 0.0
    width = model.config.hidden_size
    if matrix.shape != (width, width):
        raise ValueError(f"Output matrix must be {width} by {width}")
    new_norm = MatrixRMSNorm(width, eps, offset, device=matrix.device, dtype=matrix.dtype)
    new_norm.matrix.copy_(matrix)
    new_norm.requires_grad_(False)
    model.model.norm = new_norm
    model.__class__ = CLASSES[model.config.model_type]
    model.config.manifold_output_norm = "matrix_v1"
    return new_norm


def load_causal_model(path, **kwargs):
    """Load trusted in-repo adapters without enabling arbitrary remote code."""
    config = AutoConfig.from_pretrained(path, local_files_only=kwargs.get("local_files_only", False))
    if getattr(config, "manifold_output_norm", None):
        cls = CLASSES[config.model_type]
        return cls.from_pretrained(path, config=config, **kwargs)
    return AutoModelForCausalLM.from_pretrained(path, **kwargs)


@torch.no_grad()
def projected_output_matrix(gain, basis, rms_scale):
    """Dense P.T D P, scaled by RMS(projected) / RMS(original).

    For rectangular P this represents the retained-space gain, not recovery of
    discarded information. The square orthogonal case is an exact rotation.
    """
    p = basis.double()
    return (float(rms_scale) * (p.T @ (gain.double()[:, None] * p))).to(basis.dtype)


@torch.no_grad()
def project_residual_width(model, basis, scales, chunk_size=None):
    """Project only the residual width; keep all heads, layers and MLP neurons.

    The caller sets the final norm afterwards. With chunk_size, project one
    weight chunk in FP32 and store it in the original dtype. This bounds scratch
    memory and avoids BF16 accumulation/early gain rounding on large models.
    """
    lm = model.model
    d, k = basis.shape
    if d != model.config.hidden_size or not 0 < k <= d:
        raise ValueError("Invalid residual basis shape")
    if len(scales) != 2 * len(lm.layers) + 1:
        raise ValueError("Need one RMS correction per residual site")
    if chunk_size is not None and chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    # Native configs sometimes derive head_dim from hidden_size. Freeze the
    # original head width before changing residual width, including on reload.
    if not getattr(model.config, "head_dim", None):
        model.config.head_dim = d // model.config.num_attention_heads

    def right(weight, gain=None, scale=1.):
        if chunk_size is None:
            return ((weight if gain is None else weight * gain[None, :]) @ basis) * scale
        out = weight.new_empty((weight.shape[0], k))
        for start in range(0, len(weight), chunk_size):
            w = weight[start:start + chunk_size].float()
            if gain is not None:
                w = w * gain.float()[None, :]
            out[start:start + chunk_size].copy_((w @ basis.float()) * scale)
        return out

    def left(weight):
        if chunk_size is None:
            return basis.T @ weight
        out = weight.new_empty((k, weight.shape[1]))
        for start in range(0, weight.shape[1], chunk_size):
            out[:, start:start + chunk_size].copy_(basis.float().T @ weight[:, start:start + chunk_size].float())
        return out

    tied = lm.embed_tokens.weight is model.lm_head.weight
    embed = right(lm.embed_tokens.weight)
    head = embed if tied else right(model.lm_head.weight)
    lm.embed_tokens = nn.Embedding.from_pretrained(embed, freeze=True)
    new_head = nn.Linear(k, head.shape[0], bias=False, device="meta", dtype=head.dtype)
    new_head.weight = lm.embed_tokens.weight if tied else nn.Parameter(head, requires_grad=False)
    model.lm_head = new_head

    def gain(norm):
        return norm.weight.float() + (1 if model.config.model_type == "qwen3_5_text" else 0)

    def make_linear(old, weights, bias):
        new = nn.Linear(weights.shape[1], weights.shape[0], bias=bias is not None,
                        device="meta", dtype=weights.dtype)
        new.weight = nn.Parameter(weights, requires_grad=False)
        if bias is not None:
            new.bias = nn.Parameter(bias.clone(), requires_grad=False)
        return new

    def read(parent, name, norm, scale):
        old = getattr(parent, name)
        weights = right(old.weight, gain(norm), scale)
        setattr(parent, name, make_linear(old, weights, old.bias))

    def write(parent, name):
        old = getattr(parent, name)
        weights = left(old.weight)
        bias = None if old.bias is None else (basis.float().T @ old.bias.float()).to(old.bias.dtype)
        setattr(parent, name, make_linear(old, weights, bias))

    def fresh_norm(old):
        eps = old.eps if hasattr(old, "eps") else old.variance_epsilon
        return type(old)(k, eps=eps).to(device=old.weight.device, dtype=old.weight.dtype)

    for i, layer in enumerate(lm.layers):
        if hasattr(layer, "linear_attn"):
            mixer = layer.linear_attn
            names, out = ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"), "out_proj"
        else:
            mixer = layer.self_attn
            names, out = ("q_proj", "k_proj", "v_proj"), "o_proj"
        for name in names:
            read(mixer, name, layer.input_layernorm, scales[2 * i])
        write(mixer, out)
        for name in ("gate_proj", "up_proj"):
            read(layer.mlp, name, layer.post_attention_layernorm, scales[2 * i + 1])
        write(layer.mlp, "down_proj")
        layer.input_layernorm = fresh_norm(layer.input_layernorm)
        layer.post_attention_layernorm = fresh_norm(layer.post_attention_layernorm)
        for module in (mixer, layer.mlp):
            if hasattr(module, "hidden_size"):
                module.hidden_size = k
    lm.norm = fresh_norm(lm.norm)
    model.config.hidden_size = k
    lm.config.hidden_size = k
    model.requires_grad_(False)
    return model


@torch.no_grad()
def materialize_stock_output(model):
    """Remove the adapter by folding it into a separate stock output head.

    This is algebraically equivalent but floating-point rounding can differ.
    It duplicates a tied vocabulary matrix. Original files are never changed.
    """
    norm = model.model.norm
    if not isinstance(norm, MatrixRMSNorm):
        raise ValueError("Expected MatrixRMSNorm")
    w = model.lm_head.weight
    folded = (w.float() @ norm.matrix.float().T) * (norm.weight.float() + norm.offset)[None, :]
    head = nn.Linear(w.shape[1], w.shape[0], bias=False, device=w.device, dtype=w.dtype)
    head.weight.copy_(folded.to(w.dtype))
    model.lm_head = head
    config = model.config
    if config.model_type == "qwen3_5_text":
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNorm
        new_norm = Qwen3_5RMSNorm(config.hidden_size, eps=norm.eps)
        cls = Qwen3_5ForCausalLM
    elif config.model_type == "llama":
        from transformers.models.llama.modeling_llama import LlamaRMSNorm
        new_norm = LlamaRMSNorm(config.hidden_size, eps=norm.eps)
        cls = LlamaForCausalLM
    elif config.model_type == "mistral":
        from transformers.models.mistral.modeling_mistral import MistralRMSNorm
        new_norm = MistralRMSNorm(config.hidden_size, eps=norm.eps)
        cls = MistralForCausalLM
    else:
        raise ValueError(f"Unsupported architecture: {config.model_type}")
    new_norm = new_norm.to(device=w.device, dtype=w.dtype)
    model.model.norm = new_norm
    config.tie_word_embeddings = False
    delattr(config, "manifold_output_norm")
    if hasattr(config, "auto_map"):
        config.auto_map = {k: v for k, v in config.auto_map.items() if k != "AutoModelForCausalLM"}
    model.__class__ = cls
    model.requires_grad_(False)
    return model
