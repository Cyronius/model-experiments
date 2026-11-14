# Canary-guarded checks: session 3's copy (decayed_120) at both modes and seeds, session 2's thinking-off
# (decayed_75) at both seeds, and T3M's thinking-off seed 2. Every first run of these (2026-11-06/07) ran in
# a process whose dense pass computed wrong values, while other sessions' jobs (a Vulkan llama-server, ROCm
# jobs) shared the GPU. Each check exits 3 as soon as its canary fails (heal_sessions --abort-on-fault, the
# faulty eval moved to faulty_check_*), and is retried up to 3 times in a fresh process.
cd /home/cyrus/code/model-experiments/manifold-compression
PY=../target/manifold-rocm/env/bin/python
R=heal_sessions_g2_rules.json
L=results/moe-carve-v2-20261029/heal_sessions
run() {  # log copy check_rules
    for a in 1 2 3; do
        $PY -u heal_sessions.py check --rules $R --copy $2 --check-rules $3 --device cuda --abort-on-fault > $L/$1.log 2>&1
        s=$?; echo "$(date +%H:%M) $1 attempt $a exit $s" >> $L/rechecks.log
        [ $s -eq 3 ] || return
        mv $L/$1.log $L/faulty_$1_$a.log
    done
}
run check_3_on decayed_120.pt geometry_check_rules.json
run check_3_off decayed_120.pt geometry_check_nothink_rules.json
run check_3_on_seed2 decayed_120.pt geometry_check_rules_seed2.json
run check_3_off_seed2 decayed_120.pt geometry_check_nothink_rules_seed2.json
run check_2_off decayed_75.pt geometry_check_nothink_rules.json
run check_2_off_seed2 decayed_75.pt geometry_check_nothink_rules_seed2.json
run check_t3m_off_seed2 decayed_0.pt geometry_check_nothink_rules_seed2.json
touch $L/rechecks.done
