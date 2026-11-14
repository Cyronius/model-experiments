# Session 2's sampled GSM8K checks (non-finite rows end and are counted), after the repeat test.
cd /home/cyrus/code/model-experiments/manifold-compression
PY=../target/manifold-rocm/env/bin/python
R=heal_sessions_g2_rules.json
L=results/moe-carve-v2-20261029/heal_sessions
timeout 40m $PY -u $L/scripts/repeat_full.py > $L/repeat_full.log 2>&1
$PY -u heal_sessions.py check --rules $R --copy decayed_75.pt --check-rules geometry_check_nothink_rules.json --device cuda > $L/check_2_off.log 2>&1
$PY -u heal_sessions.py check --rules $R --copy decayed_75.pt --check-rules geometry_check_rules.json --device cuda > $L/check_2_on.log 2>&1
touch results/moe-carve-v2-20261029/heal_sessions/checks2.done
