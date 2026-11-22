"""Write a released checkpoint's ternary weights as packed 2-bit codes plus bf16 row scales (2026-10-28)."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).resolve().parent.parent/'tern-embed'))
from model import tern_codes

TERNARY = ('q', 'k', 'v', 'o', 'fc1', 'fc2')
FILENAME = 'model.ternary.safetensors'


def pack_rows(codes: torch.Tensor) -> torch.Tensor:
    c = (codes.to(torch.int8) & 3).to(torch.uint8).reshape(codes.shape[0], -1, 4)
    return c[..., 0] | c[..., 1] << 2 | c[..., 2] << 4 | c[..., 3] << 6


def unpack_rows(packed: torch.Tensor) -> torch.Tensor:
    c = torch.stack([(packed >> s) & 3 for s in (0, 2, 4, 6)], -1).reshape(packed.shape[0], -1).to(torch.int8)
    return torch.where(c == 3, torch.full_like(c, -1), c)


def is_ternary(key: str) -> bool:
    parts = key.split('.')
    return parts[0] == 'layers' and parts[2] in TERNARY and parts[3] == 'weight'


def load_ternary(path: Path) -> tuple[dict, dict]:
    """-> (float tensors, {module prefix: (codes int8, bf16 scale as float32)})."""
    raw = load_file(path/FILENAME)
    dense = {k: v for k, v in raw.items() if not k.endswith(('.codes', '.scale'))}
    ternary = {k[:-6]: (unpack_rows(raw[k]), raw[k[:-6]+'.scale'].float()[:, None])
               for k in raw if k.endswith('.codes')}
    return dense, ternary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('model', type=Path, help='Folder with config.json and float32 model.safetensors')
    a = p.parse_args()
    state = load_file(a.model/'model.safetensors')
    out = {}
    for key, value in state.items():
        if not is_ternary(key):
            out[key] = value.contiguous()
            continue
        codes, scale = tern_codes(value)
        prefix = key[:-7]
        out[prefix+'.codes'] = pack_rows(codes).contiguous()
        out[prefix+'.scale'] = scale.reshape(-1).to(torch.bfloat16).contiguous()
    save_file(out, a.model/FILENAME, metadata={'format': 'ternary-2bit-rowmajor',
              'codes': 'uint8, four codes per byte, low bits first; 0=0, 1=+1, 3=-1',
              'scale': 'bf16 absmean per output row; weight = code * scale'})
    dense, ternary = load_ternary(a.model)
    for prefix, (codes, scale) in ternary.items():
        ref_codes, ref_scale = tern_codes(state[prefix+'.weight'])
        if not (torch.equal(codes, ref_codes.to(torch.int8)) and torch.equal(scale, ref_scale)):
            raise AssertionError(f'Round trip changed {prefix}')
    size = (a.model/FILENAME).stat().st_size
    ternary_bytes = sum(out[k].numel()*out[k].element_size() for k in out if k.endswith(('.codes', '.scale')))
    print(json.dumps({'file': str(a.model/FILENAME), 'bytes': size, 'ternary_bytes': ternary_bytes,
                      'matrices': len(ternary), 'round_trip': 'exact'}))


if __name__ == '__main__':
    main()
