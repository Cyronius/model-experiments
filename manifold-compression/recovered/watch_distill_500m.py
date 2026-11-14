"""Download continuation checkpoints and stop only this pod on completion/failure.

Run locally with the existing RunPod credentials. No credentials are sent to SSH.
"""
import json
from pathlib import Path
import re
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "manifold-compression/logs"
MANIFEST = LOGS / "continuation-500m-pod.json"
DEST = ROOT / "models/manifold-compression/distill-08b/continuation-500m"
SSH = "C:/Program Files/Git/usr/bin/ssh.exe"
SCP = "C:/Program Files/Git/usr/bin/scp.exe"
KEY = "C:/Users/josha/.runpod/ssh/runpodctl-ssh-key"
CONFIG = Path("C:/Users/josha/.runpod/config.toml")
KNOWN = LOGS / "continuation_known_hosts"
pod = json.loads(MANIFEST.read_text(encoding="utf-8-sig"))
pod_id = pod["id"]
started = time.time()
DEST.mkdir(parents=True, exist_ok=True)


def log(message):
    print(time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()), message, flush=True)


def api(method, suffix=""):
    match = re.search(r'^\s*(?:apikey|api_key)\s*=\s*[\"\']([^\"\']+)', CONFIG.read_text(), re.M)
    request = urllib.request.Request(f"https://rest.runpod.io/v1/pods/{pod_id}{suffix}",
                                     headers={"Authorization": "Bearer " + match[1],
                                              "User-Agent": "Mozilla/5.0 RunPodTrainingMonitor",
                                              "Accept": "application/json"}, method=method)
    with urllib.request.urlopen(request, timeout=30) as response:
        raw = response.read()
        return json.loads(raw) if raw else None


def options():
    return ["-i", KEY, "-o", "BatchMode=yes", "-o", f"UserKnownHostsFile={KNOWN}",
            "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=30"]


def remote(command):
    return subprocess.check_output([SSH, *options(), "-p", str(pod["portMappings"]["22"]),
                                    "root@" + pod["publicIp"], command], timeout=90).decode()


def copy(source, destination, recursive=False):
    subprocess.run([SCP, "-S", SSH, *options(), "-P", str(pod["portMappings"]["22"]),
                    *(["-r"] if recursive else []), f"root@{pod['publicIp']}:{source}", str(destination)],
                   check=True, timeout=1800)


log(f"watching {pod_id}; downloads to {DEST}")
while True:
    try:
        current = api("GET")
        if current["desiredStatus"] != "RUNNING":
            log("pod is stopped; monitor exiting")
            break
        if current.get("publicIp") and current.get("portMappings", {}).get("22"):
            pod.update(publicIp=current["publicIp"], portMappings=current["portMappings"])
        status = json.loads(remote("python /workspace/pod_distill_status.py"))
        log(f"stage={status['stage']} saved_progress={status['resume']}")
        for name, sizes in sorted(status["checkpoints"].items()):
            if name == "ckpt_500M" and status["stage"] != "done":
                continue
            destination = DEST / name
            marker = destination / ".download-complete"
            if marker.exists():
                continue
            copy(f"/workspace/distill-500m/{name}", DEST, recursive=True)
            if not all((destination / n).is_file() and (destination / n).stat().st_size == size
                       for n, size in sizes.items()):
                raise RuntimeError(f"incomplete download: {name}")
            marker.write_text(json.dumps(sizes, indent=1))
            log(f"downloaded and size-verified {name}")
        if status["stage"] in ("done", "failed"):
            copy("/workspace/logs/distill-500m.log", LOGS / "continuation-500m-training.log")
            if status["resume"]:
                copy("/workspace/distill-500m/history.json", DEST / "history.json")
                copy("/workspace/distill-500m/resume_latest.json", DEST / "resume_latest.json")
            if status["stage"] == "done" and not (DEST / "ckpt_500M/.download-complete").exists():
                raise RuntimeError("500M download is not complete; will retry before stopping")
            api("POST", "/stop")
            log(f"STOPPED {pod_id}: {status['stage']}; persistent volume retains full resume state")
            break
    except Exception as error:
        log(f"retry: {type(error).__name__}: {error}")
        if isinstance(error, urllib.error.HTTPError):
            log(error.read().decode(errors="replace")[:500])
    if time.time() - started > 36 * 3600:
        # The remote trainer has a 32-hour graceful timeout; this bounds idle spend
        # even if a setup or connectivity failure prevents normal completion.
        api("POST", "/stop")
        log(f"STOPPED {pod_id}: 36-hour watchdog limit; latest state stays on volume")
        break
    if "--once" in sys.argv:
        break
    time.sleep(300)
