# Session 3 (heal_sessions_g2_rules.json sessions[1]): 4 hours continuing session 2, an 8-step decayed copy,
# then the held-out fit and the GSM8K checks at two seeds.
cd /home/cyrus/code/model-experiments/manifold-compression
PY=../target/manifold-rocm/env/bin/python
R=heal_sessions_g2_rules.json
L=results/moe-carve-v2-20261029/heal_sessions
F=$L/S0_K24_G2
$PY -u heal_sessions.py session --rules $R --hours 4 --device cuda > $L/session_3.log 2>&1 || exit 1
$PY -u heal_sessions.py decay --rules $R --steps 8 --device cuda > $L/decay_3.log 2>&1 || exit 1
C=$(grep -o 'decayed_[0-9]*\.pt' $L/decay_3.log | tail -1)
[ -n "$C" ] || exit 1
echo "$C" > $L/session3_copy.txt
timeout 30m $PY -u $L/scripts/heldout.py $C > $L/heldout_shard1_3.log 2>&1
$PY -u heal_sessions.py check --rules $R --copy $C --check-rules geometry_check_rules.json --device cuda > $L/check_3_on.log 2>&1
$PY -u heal_sessions.py check --rules $R --copy $C --check-rules geometry_check_rules_seed2.json --device cuda > $L/check_3_on_seed2.log 2>&1
$PY -u heal_sessions.py check --rules $R --copy $C --check-rules geometry_check_nothink_rules.json --device cuda > $L/check_3_off.log 2>&1
$PY -u heal_sessions.py check --rules $R --copy $C --check-rules geometry_check_nothink_rules_seed2.json --device cuda > $L/check_3_off_seed2.log 2>&1
$PY -u heal_sessions.py check --rules $R --copy decayed_0.pt --check-rules geometry_check_nothink_rules_seed2.json --device cuda > $L/check_t3m_off_seed2.log 2>&1
touch $L/session3.done
