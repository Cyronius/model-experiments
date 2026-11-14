# The remaining canary-guarded checks, each attempt started only when no other process has the GPU open.
# rechecks.sh (2026-11-07 16:45-19:24) got 1 clean canary in 20 attempts, every one beside another
# session's GPU job, though it held dibs: dibs granted the GPU to other sessions at the same time.
# This tests whether sharing causes the faults: if attempts run alone come out clean, it does.
cd /home/cyrus/code/model-experiments/manifold-compression
PY=../target/manifold-rocm/env/bin/python
CPU=../target/manifold-audit/env/bin/python
R=heal_sessions_g2_rules.json
L=results/moe-carve-v2-20261029/heal_sessions
quiet() {  # wait until no other GPU user for 2 checks a minute apart
    n=0
    while [ $n -lt 2 ]; do
        u=$($CPU -c "import heal_sessions as h; print(len(h.gpu_users()))" 2>/dev/null)
        if [ "$u" = "0" ]; then n=$((n+1)); else n=0; fi
        sleep 60
    done
}
run() {  # log copy check_rules
    for a in 1 2 3 4; do
        quiet
        echo "$(date +%H:%M) $1 attempt $a start (GPU quiet)" >> $L/rechecks_quiet.log
        $PY -u heal_sessions.py check --rules $R --copy $2 --check-rules $3 --device cuda --abort-on-fault > $L/$1.log 2>&1
        s=$?; echo "$(date +%H:%M) $1 attempt $a exit $s" >> $L/rechecks_quiet.log
        [ $s -eq 3 ] || return
        mv $L/$1.log $L/faulty_$1_q$a.log
    done
}
run check_3_on decayed_120.pt geometry_check_rules.json
run check_3_off decayed_120.pt geometry_check_nothink_rules.json
run check_3_on_seed2 decayed_120.pt geometry_check_rules_seed2.json
run check_3_off_seed2 decayed_120.pt geometry_check_nothink_rules_seed2.json
run check_2_off_seed2 decayed_75.pt geometry_check_nothink_rules_seed2.json
run check_t3m_off_seed2 decayed_0.pt geometry_check_nothink_rules_seed2.json
touch $L/rechecks_quiet.done
