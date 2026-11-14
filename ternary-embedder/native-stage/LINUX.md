# Linux encoder — 2026-10-24

The complete encoder now runs through a local Linux XRT shared library. Both
cap-32 and cap-128 graphs were rebuilt and executed on `/dev/accel/accel0`.
The offline text CLI uses the recovered, revision-pinned bge-small tokenizer.
Windows loading remains supported; `TERNARY_NPU_LIBRARY` overrides the host
library path on either platform.

## Measured results

50 retained samples after five warmups, two PyTorch threads, four fixed English
sentences. Alternating baseline/optimized order within each iteration reduces
order bias. This is a microbenchmark, not full-corpus retrieval encoding.

| Batch / cap | Original full-output forward | Final-output forward | Text to vector | Text embeddings/s |
|---|---:|---:|---:|---:|
| 4 / 32 | 1.249 ms | 1.225 ms | 1.551 ms | 2580 |
| 1 / 128 | 1.181 ms | 1.162 ms | 1.307 ms | 765 |

Normal inference now synchronizes only the final layer's 256 KiB, rather than
both layers' 512 KiB, and avoids copying/decoding the discarded intermediate.
Paired median gains are 2.0% and 1.6%. `full_output=True` retains the original
host output path as a comparison; `trace=True` still retrieves both layers.
This does not remove the first layer's device-side trace DMA or make weights
resident. The large change from historical Windows timings is not attributed
to this small optimization: platform, input texts, and machine load differ.

All four saved traces at each cap passed a cosine threshold of 0.995 against
both the current CPU model and the saved Windows reference vectors. Actual
minimum CPU cosine was 0.999880 (32) / 0.999905 (128). Normal and traced NPU
vectors were bit-identical. CLI text vectors also passed CPU comparison.
No fresh full-corpus retrieval evaluation was run. Raw samples, fixture,
checkpoint, runtime, and build hashes are in `linux-result.json`.

## Reproduce on this machine

The existing toolchain environment is reused without modifying its packages.
The project venv inherits its mlir-aie installation. Versions used: mlir-aie
1.4.2, Peano 21.0.0.2026080301+c9c5ecb7, torch 2.14.1+cpu. XRT comes from the
local Linux SDK selected by `XILINX_XRT`.

```bash
source /home/cyrus/code/phlegm/target/npu-dev/env.sh
python -m venv --system-site-packages ternary-embedder/.venv
ternary-embedder/.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
ternary-embedder/.venv/bin/pip install transformers
ternary-embedder/.venv/bin/python ternary-embedder/native-stage/build_host.py
python ternary-embedder/native-stage/build_fused.py --cap 32
python ternary-embedder/native-stage/build_fused.py --cap 128
ternary-embedder/.venv/bin/python ternary-embedder/native-stage/bench_linux.py
ternary-embedder/.venv/bin/python ternary-embedder/native-stage/embed.py --cap 32 'A small NPU encoder.'
```

Builds use `AIE_XCLBINUTIL` or `xclbinutil` on PATH and Peano's Linux `llvm-size`.
The C ABI reports XRT exceptions as Python errors and cleans up failed opens.
An encoder instance owns mutable buffers and must be used serially.
The two restored trace fixtures and checkpoints are ignored local artifacts;
copy them with the project when reproducing elsewhere. Serving integration,
full-corpus Linux evaluation, and resident-weight/replica layouts remain open.
