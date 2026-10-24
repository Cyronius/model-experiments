"""Restart only this control job with container-local derived caches."""
import json
from pathlib import Path
import time

from run_control import ROOT, REMOTE, STATE, helpers, save


if __name__ == "__main__":
    pod = json.loads(STATE.read_text())
    assert pod["local_status"] == "running"
    stage = helpers.remote(pod, f"cat {REMOTE}/control.status").strip()
    if stage != "teacher":
        raise RuntimeError(f"Restart is allowed only during teacher generation, got {stage}")
    helpers.copy(pod, ROOT / "float_control.sh", REMOTE + "/float_control.sh", upload=True)
    # Verify process identity and process group before terminating this own job.
    helpers.remote(pod, f"cd {REMOTE} && python - <<'PY'\n"
        "import os,pathlib,signal,time\n"
        "pid=int(pathlib.Path('control.pid').read_text())\n"
        "cmd=pathlib.Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\\0',b' ').decode()\n"
        "assert 'timeout' in cmd and 'float_control.sh' in cmd,cmd\n"
        "assert os.getpgid(pid)==pid,(pid,os.getpgid(pid))\n"
        "os.killpg(pid,signal.SIGTERM)\n"
        "time.sleep(2)\n"
        "print('Stopped own teacher job',pid)\nPY")
    seconds = int(pod["max_seconds"] - (time.time() - pod["local_started"]) - 90)
    if seconds < 600:
        raise RuntimeError("Insufficient time left for a bounded teacher restart")
    stamp = int(time.time())
    helpers.remote(pod, f"cd {REMOTE} && mv logs/teacher.log logs/teacher.partial.{stamp}.log && "
        f"(nohup timeout {seconds}s bash float_control.sh "
        "> control.log 2>&1 < /dev/null & echo $! > control.pid)")
    pod.update(teacher_batch=512, local_work_cache=True, teacher_restarted=time.time())
    save(pod)
    print(f"Teacher restarted with local caches, original batch 512; {seconds}s job limit", flush=True)
