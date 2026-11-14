#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
task_dir=training-v2/work/native-depth
mkdir -p "$task_dir"
touch "$task_dir/lock-acquired"
for ((i=0; i<300; i++)); do
  if test -f "$task_dir/ready"; then break; fi
  sleep 1
done
test -f "$task_dir/ready"
source /home/cyrus/code/phlegm/target/npu-dev/env.sh
for depth in 2 3; do
  for cap in 32 128; do
    if python - "$depth" "$cap" <<'PY'
import hashlib,json,sys
from pathlib import Path
depth,cap=map(int,sys.argv[1:]);root=Path('native-stage')
out=root/(f'build_fused_cap{cap}' if depth==2 else f'build_fused3_cap{cap}')
try:
 d=json.loads((out/'build.json').read_text())
 assert d['layers']==depth and d['cap']==cap
 for name,sha in d['source_sha256'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==sha
 for name,sha in d['artifact_sha256'].items():assert hashlib.sha256((out/name).read_bytes()).hexdigest()==sha
except (FileNotFoundError,AssertionError):sys.exit(1)
print(f'Reusing matching {depth}-layer cap{cap} build artifacts')
PY
    then :; else python native-stage/build_fused.py --cap "$cap" --layers "$depth"; fi
  done
  .venv/bin/python training-v2/verify_candidate_npu.py --freeze "training-v2/native-depth/frozen-$depth.json" --branch ternary --single --verify-only --output "training-v2/native-depth/verification-$depth.json"
done
