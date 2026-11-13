"""Numerical canary for the shared local GPU.

Since 2026-11-05 some processes on the local ROCm GPU have computed wrong values with no error.
Every GPU evaluation and healing run therefore checks the untouched teacher's per-sequence CE on
fixed calibration sequences, before and after its work, against:

  1. a CPU FP32 reference (`build`), within CPU_TOLERANCE nats per sequence (BF16 vs FP32), and
  2. the first GPU result that passed (1), within GPU_TOLERANCE (same kernels, same inputs).

A failed check raises CanaryFailure; drivers rerun the job in a fresh process.

  python canary.py --teacher <teacher> --data <data> --output baselines/canary-cpu.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from engine import logit_loss, precision, stream
from pruning import sha256

SEQUENCES = 4
CPU_TOLERANCE = 0.05
GPU_TOLERANCE = 0.01


class CanaryFailure(RuntimeError):
    pass


@torch.no_grad()
def sequence_ce(teacher, rows, device, amp):
    out = []
    for row in rows:
        ids = torch.tensor(row[None], device=device, dtype=torch.long)
        with precision(teacher, amp):
            h = stream(teacher, ids)[-1]
            ce, _ = logit_loss(teacher, teacher, h, h, ids)
        out.append(float(ce))
    return out


def identity(teacher_dir, data_dir):
    return dict(teacher_sha256={p.name: sha256(p) for p in sorted(Path(teacher_dir).glob('*.safetensors'))},
                data_manifest_sha256=sha256(Path(data_dir) / 'manifest.json'), sequences=SEQUENCES)


def check(teacher, teacher_dir, data_dir, reference, device, amp, gpu_reference=None):
    """Per-sequence comparison; writes gpu_reference on the first pass. Returns the record."""
    ref = json.loads(Path(reference).read_text())
    if {k: ref[k] for k in ('teacher_sha256', 'data_manifest_sha256', 'sequences')} != identity(teacher_dir, data_dir):
        raise ValueError('Canary reference is for a different teacher or data')
    rows = np.load(Path(data_dir) / 'tokens.npz', allow_pickle=False)['calibration'][:SEQUENCES]
    values = sequence_ce(teacher, rows, device, amp)
    cpu = [abs(a - b) for a, b in zip(values, ref['ce'])]
    record = dict(ce=values, cpu_abs_diff=cpu, cpu_tolerance=CPU_TOLERANCE, ok=max(cpu) <= CPU_TOLERANCE,
                  device_name=torch.cuda.get_device_name() if str(device).startswith('cuda') else str(device))
    if gpu_reference is not None:
        gpu_reference = Path(gpu_reference)
        if gpu_reference.exists():
            g = json.loads(gpu_reference.read_text())
            if g['device_name'] == record['device_name'] and g['cpu_reference_sha256'] == sha256(reference):
                diff = [abs(a - b) for a, b in zip(values, g['ce'])]
                record.update(gpu_abs_diff=diff, gpu_tolerance=GPU_TOLERANCE, ok=record['ok'] and max(diff) <= GPU_TOLERANCE)
        elif record['ok']:
            gpu_reference.write_text(json.dumps(dict(ce=values, device_name=record['device_name'],
                                                     cpu_reference_sha256=sha256(reference)), indent=1) + '\n')
            record['wrote_gpu_reference'] = str(gpu_reference)
    if not record['ok']:
        raise CanaryFailure(json.dumps(record))
    return record


def main():
    from run import load_model
    ap = argparse.ArgumentParser()
    ap.add_argument('--teacher', type=Path, required=True)
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--threads', type=int, default=8)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(args.threads)
    teacher = load_model(args.teacher, 'cpu', torch.float32)
    rows = np.load(args.data / 'tokens.npz', allow_pickle=False)['calibration'][:SEQUENCES]
    record = dict(identity(args.teacher, args.data), ce=sequence_ce(teacher, rows, 'cpu', False),
                  precision='CPU FP32, eager attention', code_sha256=sha256(__file__))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=1) + '\n')
    print(json.dumps(record['ce']))


if __name__ == '__main__':
    main()
