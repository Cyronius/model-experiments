#!/bin/sh
# Edit the probe's core program as npuasm text and rebuild it with no vendor
# tool: npuasm builds the CDOs, PDI and insts.bin, the rule gate checks the
# result, and npuasm packages the xclbin (fresh PDI and xclbin UUIDs, since
# the firmware skips a PDI load it has just done).
#   edit_build.sh OUT 'old line' 'new line'
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
NPUASM=$(cd "$HERE/../.." && pwd)
OUT=$1; OLD=$2; NEW=$3
PRJ=$HERE/build/final.prj
rm -rf "$OUT"; mkdir -p "$OUT"
cd "$NPUASM"
python3 -m npuasm lift "$PRJ" "$OUT/lifted" >/dev/null
python3 - "$OUT/lifted/core_c0r2.s" "$OLD" "$NEW" <<'PY'
import sys
p, old, new = sys.argv[1:]
lines = open(p).read().split("\n")
hits = [i for i, l in enumerate(lines) if l.strip() == old]
assert len(hits) == 1, f"{len(hits)} lines match {old!r}"
lines[hits[0]] = "\t" + new
open(p, "w").write("\n".join(lines))
print(f"edited line {hits[0] + 1}: {old!r} -> {new!r}")
PY
python3 -m npuasm rules "$OUT/lifted"
python3 -m npuasm build "$OUT/lifted" "$OUT/rebuilt" >/dev/null
cp "$OUT/rebuilt/insts.bin" "$OUT/insts.bin"
python3 -m npuasm package "$OUT/rebuilt/main.pdi" "$HERE/build/final.xclbin" "$OUT/final.xclbin"
