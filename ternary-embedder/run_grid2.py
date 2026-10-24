"""Launch/watch the two remaining shallow experiments on an isolated RunPod.

Credentials stay local. The watcher downloads and SHA256-verifies results before
deleting its own pod. Failures/timeouts stop it (volume retained for recovery).
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import subprocess
import sys
import tarfile
import time
import tomllib
import urllib.error
import urllib.request
from pathlib import Path

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="backslashreplace")

ROOT = Path(__file__).resolve().parent
CODE = ROOT / "tern-embed"
LOGS = ROOT / "logs"
MANIFEST = LOGS / "grid2-pod.json"
DEST = CODE / "pulls" / "grid2"
SSH = "C:/Program Files/Git/usr/bin/ssh.exe"
SCP = "C:/Program Files/Git/usr/bin/scp.exe"
KEY = Path("C:/Users/josha/.runpod/ssh/runpodctl-ssh-key")
REMOTE = "/workspace/tern-embed-grid2"
MAX_HOURS = 4


def log(message):
    print(time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()), message, flush=True)


def api(method, route, body=None):
    config = tomllib.loads(Path("C:/Users/josha/.runpod/config.toml").read_text())
    token = config.get("apikey") or config.get("api_key")
    req = urllib.request.Request(
        "https://rest.runpod.io/v1/" + route,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Authorization": "Bearer " + token, "Content-Type": "application/json",
                 "User-Agent": "CodexTernaryEmbedder", "Accept": "application/json"},
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=45) as response:
            raw = response.read()
            return json.loads(raw) if raw else None
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")
        raise RuntimeError(f"RunPod {method} {route}: HTTP {error.code}: {detail[:1500]}") from None


def save(pod):
    MANIFEST.write_text(json.dumps(pod, indent=2), encoding="utf-8")


def options():
    return ["-i", str(KEY), "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new",
            "-o", f"UserKnownHostsFile={LOGS / 'grid2_known_hosts'}", "-o", "ConnectTimeout=15",
            "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=3"]


def remote(pod, command, timeout=90):
    return subprocess.check_output(
        [SSH, *options(), "-p", str(pod["portMappings"]["22"]), "root@" + pod["publicIp"], command],
        timeout=timeout, stderr=subprocess.STDOUT,
    ).decode(errors="replace")


def copy(pod, source, destination, upload=False):
    address = "root@" + pod["publicIp"] + ":"
    src, dst = (str(source), address + destination) if upload else (address + source, str(destination))
    subprocess.run([SCP, "-S", SSH, *options(), "-P", str(pod["portMappings"]["22"]), src, dst],
                   check=True, timeout=1200)


def pack():
    bundle = LOGS / "grid2-code.tar.gz"
    with tarfile.open(bundle, "w:gz") as archive:
        for name in ("model.py", "data.py", "train.py", "evaluate.py", "grid2.sh"):
            archive.add(CODE / name, arcname=name)
    return bundle


def stop(pod, reason):
    api("POST", f"pods/{pod['id']}/stop")
    pod.update(local_status="stopped", reason=reason)
    save(pod)
    log(f"STOPPED {pod['id']}: {reason}; volume retained")


def launch():
    if MANIFEST.exists():
        raise RuntimeError("Manifest exists; use watch/status rather than renting twice")
    # Public key only; the API token and private key are never uploaded.
    public_key = KEY.with_suffix(".pub").read_text().strip()
    body = dict(name="tern-embed-grid2", cloudType="SECURE", computeType="GPU", gpuCount=1,
                gpuTypeIds=["NVIDIA RTX A5000", "NVIDIA RTX A4500", "NVIDIA RTX 4000 Ada Generation"],
                gpuTypePriority="custom",
                templateId="runpod-torch-v280", containerDiskInGb=30, volumeInGb=30,
                volumeMountPath="/workspace", supportPublicIp=True, ports=["22/tcp"],
                env={"PUBLIC_KEY": public_key})
    pod = api("POST", "pods", body)
    pod.update(local_started=time.time(), local_status="created")
    save(pod)
    log(f"created {pod['id']} rate=${pod.get('costPerHr')}/h")
    if not pod.get("costPerHr") or pod["costPerHr"] > 0.30:
        stop(pod, "rate exceeds $0.30/h cap")
        raise RuntimeError("Rate cap failed")
    try:
        deadline = time.time() + 600
        while time.time() < deadline:
            live = api("GET", f"pods/{pod['id']}")
            pod.update(live)
            save(pod)
            if pod.get("publicIp") and (pod.get("portMappings") or {}).get("22"):
                try:
                    log(remote(pod, "python -c 'import torch; assert torch.cuda.is_available(); "
                               "x=torch.randn(512,512,device=\"cuda\"); print((x@x).norm().item())'"))
                    break
                except subprocess.CalledProcessError as error:
                    if b"AssertionError" in error.output or b"CUDA error" in error.output:
                        raise RuntimeError("Host CUDA validation failed") from error
            time.sleep(15)
        else:
            raise RuntimeError("SSH did not become ready within 10 minutes")
        bundle = pack()
        remote(pod, f"mkdir -p {REMOTE}")
        copy(pod, bundle, REMOTE + "/code.tar.gz", upload=True)
        remote(pod, f"cd {REMOTE} && tar xzf code.tar.gz --no-same-owner && "
               "(nohup timeout 3h bash grid2.sh > grid2.log 2>&1 < /dev/null & echo $! > grid2.pid)",
               timeout=30)
        pod["local_status"] = "launched"
        save(pod)
        log("launched Grid 2; 3-hour training timeout, 4-hour pod watchdog")
    except Exception:
        stop(pod, "setup failed")
        raise
    watch()


def download(pod):
    command = (
        f"cd {REMOTE} && "
        "python - <<'PY'\n"
        "import hashlib,json,pathlib,tarfile,subprocess,sys\n"
        "pathlib.Path('environment.txt').write_bytes(subprocess.check_output([sys.executable,'-m','pip','freeze']))\n"
        "files=[p for root in ('runs','logs','results') for p in pathlib.Path(root).rglob('*') if p.is_file()]\n"
        "files += [pathlib.Path(n) for n in ('grid2.log','grid2.status','grid2.summary.md','work/stats.json','environment.txt',"
        "'model.py','data.py','train.py','evaluate.py','grid2.sh') if pathlib.Path(n).exists()]\n"
        "checks={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}\n"
        "pathlib.Path('artifacts.sha256.json').write_text(json.dumps(checks,indent=1))\n"
        "with tarfile.open('artifacts.tar.gz','w:gz') as t:\n"
        " for p in files+[pathlib.Path('artifacts.sha256.json')]: t.add(p)\n"
        "print('packed',len(files),'files')\nPY"
    )
    log(remote(pod, command, timeout=180))
    DEST.mkdir(parents=True, exist_ok=True)
    archive = DEST / "artifacts.tar.gz"
    copy(pod, REMOTE + "/artifacts.tar.gz", archive)
    with tarfile.open(archive) as bundle:
        for item in bundle.getmembers():
            target = (DEST / item.name).resolve()
            if not target.is_relative_to(DEST.resolve()) or item.issym() or item.islnk():
                raise RuntimeError("Unsafe archive entry")
        bundle.extractall(DEST)
    checks = json.loads((DEST / "artifacts.sha256.json").read_text())
    for name, expected in checks.items():
        if hashlib.sha256((DEST / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"SHA256 mismatch: {name}")
    log(f"downloaded and SHA256-verified {len(checks)} files")
    return checks


def summarize(pod):
    sys.path.insert(0, str(CODE))
    import evaluate
    import numpy as np

    folders = {
        "2a_256": CODE / "pulls/grid1/results/2a_256",
        "2a_384": CODE / "pulls/grid1/results/2a_384",
        "r0_3x384": CODE / "pulls/grid1/results/r0_3x384",
        "2d_512": DEST / "results/2d_512",
        "2f_256": DEST / "results/2f_256",
        "all-MiniLM-L6-v2": ROOT / "quality-floor/results/all-MiniLM-L6-v2",
    }
    categories = list(dict.fromkeys(evaluate.TASKS.values()))
    scores, per_task = {}, {}
    for name, folder in folders.items():
        per_task[name] = {task: evaluate.score(folder, task) for task in evaluate.TASKS}
        if any(value is None for value in per_task[name].values()):
            raise RuntimeError(f"Incomplete evaluation for {name}")
        scores[name] = {kind: float(np.mean([per_task[name][task] for task, category
                                           in evaluate.TASKS.items() if category == kind]))
                        for kind in categories}
    baseline = scores["all-MiniLM-L6-v2"]
    new_names = ("2d_512", "2f_256")
    report = {
        "completed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "pod_id": pod["id"], "cost_per_hour": pod["costPerHr"],
        "elapsed_hours": (time.time() - pod["local_started"]) / 3600,
        "scores": scores, "per_task": per_task,
        "deltas_vs_MiniLM": {name: {kind: scores[name][kind] - baseline[kind] for kind in categories}
                              for name in new_names},
        "meets_STS_and_mean_retrieval_target": {
            name: all(scores[name][kind] >= baseline[kind] for kind in ("STS", "Retrieval"))
            for name in new_names},
        "throughput": "Not measured; no NPU kernel exists yet.",
    }
    (ROOT / "grid2-summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    log("Grid 2 comparison saved to grid2-summary.json")
    print("| model | " + " | ".join(categories) + " |", flush=True)
    print("|---|" + "---:|" * len(categories), flush=True)
    for name, values in scores.items():
        print("| " + name + " | " + " | ".join(f"{values[kind]:.1f}" for kind in categories) + " |", flush=True)


def watch():
    pod = json.loads(MANIFEST.read_text())
    if pod.get("local_status") in ("complete", "stopped"):
        log(f"already {pod['local_status']}; nothing to watch")
        return
    while True:
        try:
            live = api("GET", f"pods/{pod['id']}")
            if live["desiredStatus"] != "RUNNING":
                log("pod is not running; exiting watcher")
                return
            pod.update(live)
            save(pod)
            state = remote(pod, f"cd {REMOTE} && cat grid2.status; "
                           "tail -n 2 grid2.log logs/2d_512.log logs/2f_256.log 2>/dev/null || true").strip()
            log(state)
            stage = state.splitlines()[0]
            elapsed = time.time() - pod["local_started"]
            if stage in ("done", "failed") or elapsed > 3.5 * 3600:
                checks = download(pod)
                if stage == "done":
                    for name in ("2d_512", "2f_256"):
                        assert f"runs/{name}/model.pt" in checks
                        tasks = [p for p in checks if p.startswith(f"results/{name}/")
                                 and p.endswith(".json") and not p.endswith("model_meta.json")]
                        assert len(tasks) == 12, (name, len(tasks))
                    summarize(pod)
                    api("DELETE", f"pods/{pod['id']}")
                    pod.update(local_status="complete", local_finished=time.time())
                    save(pod)
                    log("COMPLETE; verified artifacts saved, own pod deleted")
                    return
                stop(pod, f"stage={stage}, elapsed={elapsed/3600:.2f}h")
                return
        except Exception as error:
            log(f"retry: {type(error).__name__}: {error}")
        if time.time() - pod["local_started"] > MAX_HOURS * 3600:
            stop(pod, "4-hour watchdog limit")
            return
        time.sleep(45)


def main():
    LOGS.mkdir(parents=True, exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("launch", "watch", "status", "summary"))
    args = parser.parse_args()
    if args.action == "launch":
        launch()
    elif args.action == "watch":
        watch()
    elif args.action == "summary":
        summarize(json.loads(MANIFEST.read_text()))
    else:
        pod = json.loads(MANIFEST.read_text())
        print(json.dumps({k: pod.get(k) for k in ("id", "local_status", "costPerHr", "reason")}, indent=2))
        if pod.get("local_status") not in ("complete", "stopped"):
            print(remote(pod, f"cd {REMOTE} && cat grid2.status && tail -n 8 grid2.log"))


if __name__ == "__main__":
    main()
