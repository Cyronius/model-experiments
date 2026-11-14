#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source /home/cyrus/code/phlegm/target/npu-dev/env.sh
delivery_cpus=$(python - <<'PY'
import os
from pathlib import Path
selected=[];seen=set()
for cpu in sorted(os.sched_getaffinity(0)):
 p=Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
 key=((p/'physical_package_id').read_text().strip(),(p/'core_id').read_text().strip())
 if key not in seen:seen.add(key);selected.append(cpu)
 if len(selected)==2:break
assert len(selected)==2
print(','.join(map(str,selected)))
PY
)
taskset -c "$delivery_cpus" .venv/bin/python training-v2/benchmark_campaign.py --freeze training-v2/delivery/benchmark-frozen.json --output training-v2/delivery/cpu-speed.json --qwen
taskset -c "$delivery_cpus" .venv/bin/python training-v2/verify_candidate_npu.py --freeze models/campaign-20261025/frozen.json --branch ternary --single --output training-v2/delivery/native-speed.json
