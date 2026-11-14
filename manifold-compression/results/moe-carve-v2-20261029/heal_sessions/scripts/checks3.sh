# Is session 2 worse, or is the check noisy? (1) teacher-forced fit to 200 unseen shard-1 answers,
# T3M against session 2's copy; (2) the thinking-on GSM8K check at a second seed for both.
cd /home/cyrus/code/model-experiments/manifold-compression
PY=../target/manifold-rocm/env/bin/python
R=heal_sessions_g2_rules.json
L=results/moe-carve-v2-20261029/heal_sessions
timeout 60m $PY -u $L/scripts/heldout.py > $L/heldout_shard1.log 2>&1
$PY -u heal_sessions.py check --rules $R --copy decayed_75.pt --check-rules geometry_check_rules_seed2.json --device cuda > $L/check_2_on_seed2.log 2>&1
$PY -u heal_sessions.py check --rules $R --copy decayed_0.pt --check-rules geometry_check_rules_seed2.json --device cuda > $L/check_t3m_on_seed2.log 2>&1
touch $L/checks3.done
