#!/bin/bash
# Blockwise healing job (blockwise_heal_rules.json), detached from any Claude session.
# Heal first under its own GPU claim; then the S0_K24 boundary profile and the GSM8K check under a
# second claim, so other queued GPU jobs can run in between; then the comparison (no GPU).
# Writes run.done holding the first failing exit code (0 when every stage succeeded).
set -u
cd /home/cyrus/code/model-experiments/manifold-compression
PY=../target/manifold-rocm/env/bin/python
OUT=results/blockwise-heal-20261106
RULES=blockwise_heal_rules.json
status=0

echo "$(date -Is) queued heal" >> $OUT/job.log
dibs run gpu --label "blockwise_heal HB_1M: 62-step heal of S0_K24 (separate experiment)" --eta 4h -- \
    $PY blockwise_heal.py heal --rules $RULES --run HB_1M --device cuda >> $OUT/heal.log 2>&1 || status=$?
echo "$(date -Is) heal exit $status" >> $OUT/job.log

if [ $status -eq 0 ]; then
    dibs run gpu --label "blockwise_heal: S0_K24 layer profile + HB_1M GSM8K check" --eta 2h -- bash -c "
        $PY blockwise_heal.py baseline --rules $RULES --device cuda >> $OUT/baseline.log 2>&1 &&
        $PY blockwise_heal.py check --rules $RULES --run HB_1M --check-rules geometry_check_rules.json \
            --device cuda >> $OUT/check.log 2>&1" || status=$?
    echo "$(date -Is) baseline+check exit $status" >> $OUT/job.log
fi

if [ -f $OUT/HB_1M/heal.json ]; then
    $PY blockwise_heal.py compare --rules $RULES --run HB_1M >> $OUT/compare.log 2>&1
    rc=$?
    [ $status -eq 0 ] && status=$rc
    echo "$(date -Is) compare done" >> $OUT/job.log
fi
echo $status > $OUT/run.done
