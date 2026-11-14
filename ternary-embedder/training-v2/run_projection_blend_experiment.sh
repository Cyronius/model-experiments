#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
DIBS_AGENT=blend_projection_cache dibs run npu --label 'Cache matched training features from blended ternary backbone' --eta 20m -- bash -c '
source /home/cyrus/code/phlegm/target/npu-dev/env.sh
.venv/bin/python training-v2/projection_experiment.py cache --profile projection-blend
'
DIBS_AGENT=blend_projection_train dibs run cpu --label 'Fit matched output projections on blended ternary backbone' --eta 15m -- .venv/bin/python training-v2/projection_experiment.py train --profile projection-blend
.venv/bin/python training-v2/score_campaign.py --profile projection-blend --arms projection-0.1 projection-1 projection-10 --mteb --local-frozen
