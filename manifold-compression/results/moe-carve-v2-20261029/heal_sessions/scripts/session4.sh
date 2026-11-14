# Session 4 (heal_sessions_g2_rules.json sessions[2]): 4 hours continuing session 3, started on an idle GPU,
# an 8-step decayed copy, the shard-1 held-out fit, then the sampled GSM8K checks at both modes and seeds,
# each started only when no other process has the GPU open and retried if its dense canary fails.
cd /home/cyrus/code/model-experiments/manifold-compression
PY=../target/manifold-rocm/env/bin/python
CPU=../target/manifold-audit/env/bin/python
R=heal_sessions_g2_rules.json
L=results/moe-carve-v2-20261029/heal_sessions
quiet() {
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
        echo "$(date +%H:%M) $1 attempt $a start (GPU quiet)" >> $L/session4_checks.log
        $PY -u heal_sessions.py check --rules $R --copy $2 --check-rules $3 --device cuda --abort-on-fault > $L/$1.log 2>&1
        s=$?; echo "$(date +%H:%M) $1 attempt $a exit $s" >> $L/session4_checks.log
        [ $s -eq 3 ] || return
        mv $L/$1.log $L/faulty_$1_q$a.log
    done
}
quiet
$PY -u heal_sessions.py session --rules $R --hours 4 --device cuda > $L/session_4.log 2>&1 || exit 1
$PY -u heal_sessions.py decay --rules $R --steps 8 --device cuda > $L/decay_4.log 2>&1 || exit 1
C=$(grep -o 'decayed_[0-9]*\.pt' $L/decay_4.log | tail -1)
[ -n "$C" ] || exit 1
echo "$C" > $L/session4_copy.txt
quiet
timeout 30m $PY -u $L/scripts/heldout.py $C > $L/heldout_shard1_4.log 2>&1
run check_4_on $C geometry_check_rules.json
run check_4_off $C geometry_check_nothink_rules.json
run check_4_on_seed2 $C geometry_check_rules_seed2.json
run check_4_off_seed2 $C geometry_check_nothink_rules_seed2.json
touch $L/session4.done
