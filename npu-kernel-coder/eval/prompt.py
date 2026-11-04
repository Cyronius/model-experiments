"""The shape-port prompt: format primer, build A's files, the request."""

from __future__ import annotations

import json
from pathlib import Path

PRIMER = """\
## npuasm text format (AMD XDNA2 / AIE2P NPU)

A build is a few text files that rebuild to the exact binaries.

core_cCrR.s - the program of compute core (column C, row R), AIE2P assembly,
one VLIW bundle per line; `;` separates the slots of one bundle. Labels end
with `:` and are used as `#label`. `// name @ 0x..` lines are comments.
A file that starts with `.base core_cXrY` is a patch on that core's file:
`=N` copies N lines, `-N` skips N lines, `+text` inserts a line, `=*` copies
the rest.

elfs.cdo, init.cdo, enable.cdo - array configuration, one register write per
line, using AMD aie-rt register and field names:
  @c2r3                      the tile for the lines below (row 0 shim, 1 memory tile, 2+ compute)
  write REG FIELD=v ...      only non-zero fields are listed; `0` means all zero
  maskwrite REG FIELD=v      the mask is the listed fields unless `mask=` is given
  block DMA_BD3_0 [..] [..]  one write over consecutive registers, one [...] per 32-bit word
  locks 0: 0*14,2,1          LOCK0_VALUE, LOCK1_VALUE, ... in order; `v*n` repeats
  @c1r2 =c0r2                same lines as the last block written for c0r2
  repeat N i { ... }         N copies; {a+bi} is a+b*i for copy i (also j, k when nested)
  dmawrite cCrR+0xADDR N     N data words follow on indented lines; `x*n` repeats a word
  dmawrite cCrR+0x20000 @core_cCrR   the core's program
Numbers are decimal below 1000, hex (0x..) above.

insts.txn - the host-side runtime sequence, same syntax: `block` programs a
shim DMA buffer descriptor, `ddr_patch REG arg=N plus=OFF` points it at host
buffer argument N plus a byte offset, `write ..._TASK_QUEUE` starts it,
`sync` waits for completion.
"""

REQUEST = """\
## Task

Below is the kernel `{design}` built as `{a_name}`. Port it to `{b_name}`.
The numbers in a build name are the design's shape parameters.

Reply with ONLY a unified diff that turns build `{a_name}` into build `{b_name}`:
`--- a/FILE` / `+++ b/FILE` headers, then `@@ -l,n +l,n @@` hunks with 2-3
lines of context. Line numbers may be approximate; the context lines locate
each hunk. Change only what the new shape needs. No explanation, no code
fences, no tools.
"""


def order(name: str) -> tuple[int, str]:
    return ({"init.cdo": 0, "enable.cdo": 1, "elfs.cdo": 2, "insts.txn": 3}.get(name, 4), name)


EXAMPLE = """\
## Worked example from the same design

The same design built as `{c_name}` and as `{d_name}` differs like this
(unified diff from `{c_name}` to `{d_name}`). Use it to see which lines
depend on which shape parameter.

{diff}
"""


def build(task_dir: Path, *, example: bool = False) -> str:
    meta = json.loads((task_dir / "task.json").read_text())
    parts = [PRIMER, REQUEST.format(**meta)]
    if example:
        ex = json.loads((task_dir / "example.json").read_text())
        parts.append(EXAMPLE.format(diff=(task_dir / "example.diff").read_text(), **ex))
    for f in sorted((p.name for p in (task_dir / "a").iterdir()), key=order):
        parts.append(f"=== FILE: {f} ===\n{(task_dir / 'a' / f).read_text()}=== END FILE: {f} ===\n")
    parts.append(f"Now the unified diff from `{meta['a_name']}` to `{meta['b_name']}`:\n")
    return "\n".join(parts)
