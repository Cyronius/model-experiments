# Sourced by the session scripts. The local GPU computes wrong values when two processes share it, and
# other sessions need it too, so every GPU job here takes short turns:
#   quiet  waits until no other process has the GPU open (two checks a minute apart);
#   turn   runs one job under dibs (preemptible: another session queueing creates $DIBS_YIELD_FILE,
#          which heal_sessions' training obeys at the next step), then leaves the GPU free for BREAK
#          seconds so other sessions can start before the next turn.
cd /home/cyrus/code/model-experiments/manifold-compression
PY=../target/manifold-rocm/env/bin/python
CPU=../target/manifold-audit/env/bin/python
R=heal_sessions_g2_rules.json
L=results/moe-carve-v2-20261029/heal_sessions
BREAK=${BREAK:-600}
quiet() {
    n=0
    while [ $n -lt 2 ]; do
        u=$($CPU -c "import heal_sessions as h; print(len(h.gpu_users()))" 2>/dev/null)
        if [ "$u" = "0" ]; then n=$((n+1)); else n=0; fi
        sleep 60
    done
}
turn() {  # label eta cmd... ; returns the job's exit code
    quiet
    dibs run gpu --preemptible --label "$1" --eta "$2" -- "${@:3}"
    s=$?
    echo "$(date +%H:%M) turn '$1' exit $s; GPU free for ${BREAK}s" >> $L/turns.log
    sleep $BREAK
    return $s
}
session() {  # log hours: training in 60-minute turns until the hours are trained
    while true; do
        turn "heal_sessions training ($1)" 70m bash -c "$PY -u heal_sessions.py session --rules $R --hours $2 --segment-minutes 60 --device cuda >> $L/$1.log 2>&1"
        s=$?
        [ $s -eq 75 ] || return $s
    done
}
check() {  # log copy check_rules: up to 4 attempts, a new one when the dense canary fails
    for a in 1 2 3 4; do
        turn "heal_sessions $1" 60m bash -c "$PY -u heal_sessions.py check --rules $R --copy $2 --check-rules $3 --device cuda --abort-on-fault > $L/$1.log 2>&1"
        s=$?
        [ $s -eq 3 ] || return $s
        mv $L/$1.log $L/faulty_$1_q$a.log
    done
}
