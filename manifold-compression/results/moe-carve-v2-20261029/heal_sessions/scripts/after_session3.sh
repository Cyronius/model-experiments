# After session 3's script (session3.sh) finishes: redo session 2's thinking-off checks, whose first run
# (check_75_off, 46/96) ran in a process whose dense pass computed wrong values on 19 of 116 eval docs
# (heal_sessions.dense_canary), at both seeds. Detached; waits on session3.done or session3.sh's end.
cd /home/cyrus/code/model-experiments/manifold-compression
PY=../target/manifold-rocm/env/bin/python
R=heal_sessions_g2_rules.json
L=results/moe-carve-v2-20261029/heal_sessions
while [ ! -f $L/session3.done ] && kill -0 30065 2>/dev/null; do sleep 120; done
dibs run gpu --label "heal_sessions session 2 off rechecks" --eta 2h -- bash -c "
$PY -u heal_sessions.py check --rules $R --copy decayed_75.pt --check-rules geometry_check_nothink_rules.json --device cuda > $L/check_2_off.log 2>&1
$PY -u heal_sessions.py check --rules $R --copy decayed_75.pt --check-rules geometry_check_nothink_rules_seed2.json --device cuda > $L/check_2_off_seed2.log 2>&1
"
touch $L/after_session3.done
