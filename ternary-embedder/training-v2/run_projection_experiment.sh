#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
DIBS_AGENT=projection_cache dibs run npu --label 'Cache audited training features with frozen ternary NPU encoder' --eta 25m -- bash -c '
source /home/cyrus/code/phlegm/target/npu-dev/env.sh
.venv/bin/python training-v2/projection_experiment.py cache
'
DIBS_AGENT=projection_train dibs run cpu --label 'Learn existing projection with ternary backbone frozen' --eta 15m -- .venv/bin/python training-v2/projection_experiment.py train
.venv/bin/python training-v2/score_campaign.py --profile projection --arms projection-0.1 projection-1 projection-10 --mteb --local-frozen
