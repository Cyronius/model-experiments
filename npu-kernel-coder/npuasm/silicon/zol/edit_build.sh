#!/bin/sh
# Edit the probe's core program as npuasm text, rebuild it without the
# compiler, and package an xclbin for it.
#   edit_build.sh OUT 'old line' 'new line'
# Lifts build/final.prj, replaces one bundle line in core_c0r2.s, rebuilds
# the CDOs, PDI and insts.bin with npuasm, then packages with xclbinutil
# (as aiecc does) under a fresh PDI UUID so the firmware cannot skip the load.
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
python3 -m npuasm build "$OUT/lifted" "$OUT/rebuilt" >/dev/null
cmp -s "$OUT/rebuilt/insts.bin" "$HERE/build/insts.bin" && echo "insts.bin unchanged"
python3 - "$PRJ/partition_main.json" "$OUT/rebuilt/main.pdi" "$OUT/partition.json" <<'PY'
import json, sys, uuid
src, pdi, out = sys.argv[1:]
j = json.load(open(src))
j["aie_partition"]["PDIs"][0]["file_name"] = pdi
j["aie_partition"]["PDIs"][0]["uuid"] = str(uuid.uuid4())
json.dump(j, open(out, "w"), indent=2)
PY
xclbinutil --quiet --add-replace-section "MEM_TOPOLOGY:JSON:$PRJ/memTopology_main.json" \
  --add-kernel "$PRJ/kernels_main.json" \
  --add-replace-section "AIE_PARTITION:JSON:$OUT/partition.json" --force --output "$OUT/final.xclbin"
cp "$OUT/rebuilt/insts.bin" "$OUT/insts.bin"
echo "built $OUT/final.xclbin"
