#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m unittest discover -s training-v2 -p test_contrastive_cache.py
python -m unittest discover -s training-v2 -p test_nli_auxiliary.py
python training-v2/run_pair_plan.py --plan training-v2/science-ternary/runpod-job.json
