# Running the 9B checks on the local GPU

The Radeon 8060S in this machine (Strix Halo, gfx1151) runs the healed Qwen3.5-9B student with ROCm PyTorch. Set up and checked on 2026-11-01.

## The environment

`../target/manifold-rocm/env` has:
- Python 3.12 from the same base as `manifold-audit`;
- `torch==2.14.1+rocm7.14`, the official PyTorch build. The pods run 2.14.1 too. Its architecture list includes gfx1151, and it carries its own ROCm runtime, so no system ROCm install or sudo is needed;
- every other package at the version `manifold-audit` pins.

```bash
cd ../target
uv venv --python /home/cyrus/code/phlegm/target/npu-dev/env/bin/python3.12 manifold-rocm/env
uv pip install --python manifold-rocm/env/bin/python --index-url https://download.pytorch.org/whl/rocm7.14 "torch==2.14.1+rocm7.14"
manifold-audit/env/bin/python -m pip freeze | grep -v '^torch==' > manifold-rocm/requirements-from-audit.txt
uv pip install --python manifold-rocm/env/bin/python -r manifold-rocm/requirements-from-audit.txt
```

PyTorch shows the GPU as `cuda` with 61.7 GB of the shared memory. So a stage runs with `--device cuda`, as on a pod. Other sessions share the GPU, so take it through `dibs`:

```bash
dibs run gpu --label "slim_opd check" --eta 3h -- \
    ../target/manifold-rocm/env/bin/python slim_opd.py check --rules <rules>.json --device cuda
```

Point the rules at their own `folder`. A local run must not write into a folder a pod's results will be merged into.

## What was checked

- **Drift reproduces.** The healed student's benchmark KL is 0.6115, against the healing run's 0.612. The difference is −0.0006 (pods: +0.0002), well inside the 0.01 check every pod runs first. Bits per byte differ by 0.05%.
- **Speed.** About a third of an A40 or RTX A6000's speed per step:
  - 96 greedy answers of 256 tokens take 162 s, 0.63 s a step at 96 rows; the pods take 0.21 s;
  - the drift check takes 331 s;
  - a 96-answer arm with a 4,096-token cap should take roughly an hour.
  - scoring text in batches (prefill) runs about 460 tokens a second: `slim_bench.py`'s multiple-choice benchmark (section 19) takes about 16 minutes per 9B arm and 1.9 hours for all six arms, load and drift check included:

    ```bash
    dibs run gpu --label "slim_bench score" --eta 2h -- \
        ../target/manifold-rocm/env/bin/python slim_bench.py score --device cuda
    ```
- **Memory.** Peak 27 GB at 256 tokens. A 4,096-token cap adds about 14 GB of attention cache at 96 rows.
- **The first load is slow, once.** Building the low-rank DeltaNet factors took 56 minutes, against a few minutes on a pod: the SVDs are slow on this GPU. They are cached in `results/moe-carve-v2-20261029/heal_factors.pt` (2.1 GB), so later loads skip that step. As on the pods, the rebuilt factors are not bit-identical to the healing run's, and the drift check above is what vouches for them.

## Answers do not match the pods token for token

Greedy answers from this GPU are not the CUDA pods' answers:
- of 96, 2 matched the pod over the first 256 tokens;
- the median answer first differed at token 55.

The 25% student is unusually sensitive to tiny numerical changes (section 11 of `results/moe-carve-v2-20261029/README.md`), and the two GPUs round differently. So compare local runs with local runs, and with pod runs only through counts (loops, finished, right), never token by token. The reproduction check is the guard on the numbers.

## Harmless warnings

- `rocSHMEM Could not open libnuma`: the `numactl-libs` system package is not installed.
- `MIOpen(HIP): Warning [ParseAndLoadDb] File is unreadable: ... gfx1151 ... fdb.txt`: MIOpen ships no tuned-kernel database for this GPU.
