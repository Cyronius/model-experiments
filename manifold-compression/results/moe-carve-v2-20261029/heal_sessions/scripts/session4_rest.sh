# Session 4's remaining checks in GPU turns (gpu_turns.sh). check_4_on was already running under a dibs
# hold when this replaced session4.sh, which held the GPU for its whole run.
. results/moe-carve-v2-20261029/heal_sessions/scripts/gpu_turns.sh
while kill -0 4113727 2>/dev/null; do sleep 60; done
sleep $BREAK
C=$(cat $L/session4_copy.txt)
grep -q "dense canary after sampling.*-- clean" $L/check_4_on.log || { mv $L/check_4_on.log $L/faulty_check_4_on_q1.log; check check_4_on $C geometry_check_rules.json; }
check check_4_off $C geometry_check_nothink_rules.json
check check_4_on_seed2 $C geometry_check_rules_seed2.json
check check_4_off_seed2 $C geometry_check_nothink_rules_seed2.json
touch $L/session4.done
