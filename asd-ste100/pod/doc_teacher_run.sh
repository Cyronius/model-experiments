#!/bin/bash
# Documentation teacher on a pod: Qwen3.8-27B-FP8 (the model that Bonsai 2 27B is a ternary copy of)
# on vLLM, the v5 runner (v4 gates, v4 meaning check by the same model, the 3k-token "traps" prompt),
# many passages at once. Flag-gated like pod_run.sh and hard-stopped at 85 percent of BUDGET by common.sh.
#
# The doc pilot code and some modules it needs are not on GitHub main yet, so they come in a tarball
# made at home: python3 data/doc_pilot/pack_pod_v5.py -> data/out/doc-pool-v5/doc-teacher.tgz (the code,
# the lexicon files and inputs.jsonl, the 21,380-passage pool). On the pod, with it in /workspace:
#   tar xzf /workspace/doc-teacher.tgz -C /workspace pod/doc_teacher_run.sh
#   RATE=<price per hour> BUDGET=1 N=600 bash /workspace/pod/doc_teacher_run.sh   # smoke: passages per hour
#   RATE=<price per hour> BUDGET=4 N=21380 bash /workspace/pod/doc_teacher_run.sh # the run; a rerun resumes
# Output: /workspace/ste-out/doc-teacher/<RUN>/{accepted,rejected,attempts}.jsonl, summary.json.
W=${W:-/workspace}
export TEACHER=${TEACHER:-Qwen/Qwen3.8-27B-FP8}
[ -d "$W/repo" ] || git clone -q --depth 1 --branch "${BRANCH:-main}" https://github.com/Cyronius/model-experiments "$W/repo"
source "$W/repo/asd-ste100/pod/common.sh"
N=${N:-600}
RUN=${RUN:-pod-qwen38-traps}
VARIANT=${VARIANT:-traps}
MAXLEN=${MAXLEN:-16384}
VSLOTS=${VSLOTS:-128}
D=$W/repo/asd-ste100

# Not phase_env: that builds llama.cpp for the pod_run.sh judge (minutes of pod time) and the lexicon,
# which the tarball brings (the tested files, so the prompts are the tested ones).
if ! done_flag doc-env; then
  guard env; log "=== env ==="
  pip install -q --break-system-packages -U "huggingface_hub[cli,hf_transfer]" 2>&1 | tail -1
  tar xzf "$W/doc-teacher.tgz" -C "$D" || fail unpack
  nvidia-smi --query-gpu=name,memory.total --format=csv | tee -a "$LOG"
  # Into HF_HOME (common.sh) with hf_transfer, so vLLM finds the weights cached.
  hf download "$TEACHER" 2>&1 | tail -1 | tee -a "$LOG" || fail download
  mark doc-env
fi
[ -f "$D/inputs.jsonl" ] || { log "doc-teacher.tgz has no inputs.jsonl"; fail inputs; }
guard vllm; phase_vllm || fail vllm

start_doc_vllm(){   # like start_vllm in common.sh, with room for the prompt and two retries
  export PATH="$W/venv/bin:$PATH" CUDA_HOME=/usr/local/cuda
  nohup "$W/venv/bin/vllm" serve "$TEACHER" --port 8080 --served-model-name x --max-model-len "$MAXLEN" \
        --gpu-memory-utilization 0.92 --max-num-seqs "$VSLOTS" --enable-prefix-caching > "$W/vllm-doc.log" 2>&1 &
  echo $! > "$W/vllm.pid"
  for _ in $(seq 1 360); do
    curl -sf http://127.0.0.1:8080/health > /dev/null && { log "vllm up ($TEACHER, max-model-len $MAXLEN)"; return 0; }
    kill -0 "$(cat "$W/vllm.pid")" 2>/dev/null || break; sleep 5
  done
  log "vllm did not come up; tail of its log:"; tail -15 "$W/vllm-doc.log" | tee -a "$LOG"; return 1
}

if ! done_flag "doc-$RUN-$N"; then
  guard server; start_doc_vllm || fail server
  guard doc; log "=== doc teacher: $N passages, $VSLOTS at once ==="
  count(){ cat "$D/data/out/doc-teacher-v5/$RUN/accepted.jsonl" "$D/data/out/doc-teacher-v5/$RUN/rejected.jsonl" 2>/dev/null | wc -l; }
  N0=$(count); T0=$(date +%s)
  (cd "$D" && PYTHONPATH=data:checker:tokenizer:data/doc_pilot timeout "$(remaining)" python3 data/doc_pilot/synthesize_v5.py \
      --run "$RUN" --backend openai --served-model x --url http://127.0.0.1:8080 --profile qwen36 \
      --model-id "$TEACHER" --variant "$VARIANT" --parallel "$VSLOTS" --inputs "$D/inputs.jsonl" --limit "$N" \
      > "$W/doc-$RUN.log" 2>&1)
  rc=$?; T1=$(date +%s); N1=$(count); POOL=$(wc -l < "$D/inputs.jsonl")
  tail -1 "$W/doc-$RUN.log" | tee -a "$LOG"
  python3 - <<PY | tee -a "$LOG"
n, secs = $N1 - $N0, max($T1 - $T0, 1); per_hour = n / secs * 3600   # this invocation only (vLLM start excluded)
print(f"doc teacher: {n} passages in {secs}s = {per_hour:.0f} passages/hour, \${$RATE / max(per_hour, 1) * 1000:.3f} per 1000 passages")
left = $POOL - $N1
print(f"  the other {left} of the {$POOL}-passage pool: about {left / max(per_hour, 1):.1f} h, about \${left / max(per_hour, 1) * $RATE:.2f} at \${$RATE}/h")
PY
  [ $rc -eq 0 ] && mark "doc-$RUN-$N" || log "doc run cut short (budget or error); a rerun resumes it"
  stop_vllm
fi
mkdir -p "$OUT/doc-teacher" && cp -r "$D/data/out/doc-teacher-v5/$RUN" "$OUT/doc-teacher/" 2>/dev/null
log "doc teacher spent about \$$(spent)"; pack; stop_pod
