#!/bin/bash
# Waits for the orphaned 2B stock held-out run (pid given) to finish, then reruns the CPU re-check driver (skips finished steps).
tail --pid=$1 -f /dev/null
kill $2 2>/dev/null   # its dibs cpu wrapper, left without its steps shell
exec /home/cyrus/code/model-experiments/manifold-compression/results/trunk-gate2-20261105/run_cpu_recheck.sh
