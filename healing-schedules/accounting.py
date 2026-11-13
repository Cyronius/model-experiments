"""Count unique text-model parameters without allocating model weights."""
import argparse
import json
from pathlib import Path

import torch
from transformers import (Qwen3Config, Qwen3ForCausalLM,
                          Qwen3_5TextConfig, Qwen3_5ForCausalLM)

ROOT = Path(__file__).resolve().parent


def architecture(config):
    config = config.get('text_config', config)
    if config['model_type'] == 'qwen3':
        return Qwen3Config(**config), Qwen3ForCausalLM
    if config['model_type'] == 'qwen3_5_text':
        return Qwen3_5TextConfig(**config), Qwen3_5ForCausalLM
    raise ValueError(f"Unsupported architecture: {config['model_type']}")


def count(config):
    cfg, cls = architecture(config)
    with torch.device('meta'):
        model = cls(cfg)
    groups = {'embedding_and_tied_head': 0, 'ffn': 0, 'attention_and_norms': 0}
    for name, p in model.named_parameters():
        group = ('embedding_and_tied_head' if name in ('model.embed_tokens.weight', 'lm_head.weight')
                 else 'ffn' if '.mlp.' in name else 'attention_and_norms')
        groups[group] += p.numel()
    return dict(total=sum(groups.values()), **groups,
                ffn_only_floor=groups['embedding_and_tied_head'] + groups['attention_and_norms'],
                tied=model.lm_head.weight is model.model.embed_tokens.weight)


def target_config(config, target, hidden=None, alignment=64):
    cfg = dict(config.get('text_config', config))
    if hidden is not None:
        if not 0 < hidden <= cfg['hidden_size']:
            raise ValueError('Residual width must be positive and no larger than teacher')
        cfg['hidden_size'] = hidden
    floor = count(cfg)['ffn_only_floor']
    unit = 3 * cfg['hidden_size'] * cfg['num_hidden_layers']
    width = round((target - floor) / unit / alignment) * alignment
    if not 0 < width <= cfg['intermediate_size']:
        raise ValueError(f'Target {target:,} impossible by FFN pruning at this residual width; floor {floor:,}')
    cfg['intermediate_size'] = width
    return cfg


def audit():
    rows = {}
    for path in sorted((ROOT / 'metadata').glob('Qwen*.json')):
        meta = json.loads(path.read_text())
        rows[path.stem] = dict(model_id=meta['model_id'], revision=meta['revision'],
                               unique_text_parameters=count(meta['config']),
                               stored_tensor_elements=meta['hub_safetensors']['total'])
    for name, model, hidden, target in [('primary_0.8b', 'Qwen3.5-2B', 1024, 752_393_024),
                                       ('second_1.0b', 'Qwen3.5-2B', 1280, 1_000_000_000),
                                       ('qwen3_ffn_alternative', 'Qwen3-1.7B', 2048, 800_000_000)]:
        source = json.loads((ROOT / 'metadata' / f'{model}.json').read_text())
        cfg = target_config(source['config'], target, hidden)
        rows[name] = dict(source=model, hidden_size=hidden, intermediate_size=cfg['intermediate_size'],
                          layers=cfg['num_hidden_layers'], unique_text_parameters=count(cfg))
    return rows


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path)
    args = ap.parse_args()
    result = json.dumps(audit(), indent=2) + '\n'
    if args.output:
        args.output.write_text(result)
    print(result)
