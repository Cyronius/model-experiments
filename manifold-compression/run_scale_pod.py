"""Prepare, launch, sync and terminate ONE explicitly approved scale-probe pod.

Default is a dry run. --execute is an operator switch, not user authorization:
obtain the user's approval for this exact resource, duration and cap first.
Credentials remain local. An independent watchdog survives the main process.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
JOB_ROOT = REPO / "target/manifold-scale"
RESULTS = ROOT / "results/scale-probe-20261024"
IMAGE = "pytorch/pytorch@sha256:1262abc856a428933c0477b4be87079173a0b087b7691c47fcb396b7b5775c56"
GPU = "NVIDIA H200"
GPU_RATE = 5.29
DISK_GB = 200
DISK_RATE = DISK_GB * .10 / (30 * 24)
MAX_MINUTES = 90
BUDGET = 10.
JOB_PREFIX = "mc-scale-"
BUNDLE_UPLOAD_TIMEOUT = 60
FILES = ["scale_probe.py", "scale_data.py", "projection_models.py", "audit_rotation.py",
         "healing_data.py", "scale_worker.py", "scale_requirements.txt"]


def write_json(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def credential():
    path = Path.home() / ".config/runpod/credentials.env"
    if path.stat().st_mode & 0o777 != 0o600 or path.parent.stat().st_mode & 0o777 != 0o700:
        raise PermissionError("RunPod credential permissions must be 0600 / 0700")
    for line in path.read_text().splitlines():
        if line.startswith("RUNPOD_API_KEY="):
            return line.partition("=")[2].strip().strip("\"'")
    raise ValueError("RUNPOD_API_KEY missing")


def api(method, path, data=None):
    request = urllib.request.Request("https://rest.runpod.io/v1" + path,
        data=json.dumps(data).encode() if data is not None else None, method=method,
        headers={"Authorization": "Bearer " + credential(), "Content-Type": "application/json",
                 "User-Agent": "Mozilla/5.0 ManifoldScaleProbe", "Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=20) as response:
        body = response.read()
        return json.loads(body) if body else None


def fresh_quote():
    request = urllib.request.Request("https://api.runpod.io/graphql",
        data=json.dumps({"query": "{ gpuTypes { id securePrice } }"}).encode(),
        headers={"Authorization": "Bearer " + credential(), "Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=20) as response:
        data = json.load(response)
    rate = next(x["securePrice"] for x in data["data"]["gpuTypes"] if x["id"] == GPU)
    if rate is None or float(rate) > GPU_RATE:
        raise ValueError("Current GPU price exceeds the proposed rate")
    return float(rate)


def proposal():
    return {"resource": "1 H200 SXM, 141 GB VRAM, Secure Cloud on-demand", "gpu_id": GPU,
            "gpu_usd_per_hour_ceiling": GPU_RATE, "temporary_disk_gb": DISK_GB,
            "disk_usd_per_hour_estimate": DISK_RATE, "maximum_minutes_from_create": MAX_MINUTES,
            "maximum_planned_usage_usd": (GPU_RATE + DISK_RATE) * MAX_MINUTES / 60,
            "budget_cap_usd": BUDGET, "image": IMAGE,
            "scope": "Qwen2B control, Qwen27B, Mistral24B; stock/75%/50%; no healing",
            "outputs": "metrics, generations, basis, scales, configs, pinned sources; no full weight copies",
            "termination": "Delete only the newly created pod on completion, failure or deadline; no persistent volume",
            "authorization": "pending explicit approval of this bounded paid job"}


def prepare_bundle():
    JOB_ROOT.mkdir(parents=True, exist_ok=True)
    paths = {name: ROOT / name for name in FILES}
    paths["data.json"] = RESULTS / "data.json"
    manifest = {"sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}}
    write_json(manifest, JOB_ROOT / "bundle-manifest.json")
    paths["bundle-manifest.json"] = JOB_ROOT / "bundle-manifest.json"
    bundle = JOB_ROOT / "scale-bundle.tar.gz"
    with tarfile.open(bundle, "w:gz") as archive:
        for name, path in paths.items():
            archive.add(path, arcname=name)
    write_json(proposal() | {"bundle_sha256": hashlib.sha256(bundle.read_bytes()).hexdigest(),
                            "files": manifest}, RESULTS / "job-proposal.json")
    return bundle


def matching_pod(manifest):
    if manifest.get("pod_id"):
        try:
            return api("GET", "/pods/" + manifest["pod_id"])
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return None
            raise
    # Resolve an ambiguous create response without creating a second paid pod.
    pods = api("GET", "/pods")
    if isinstance(pods, dict):
        pods = pods.get("pods", [])
    found = [p for p in pods if p["name"] == manifest["name"]]
    if len(found) > 1:
        raise RuntimeError("Unique job name unexpectedly matched multiple pods")
    return found[0] if found else None


def terminate(manifest):
    pod = matching_pod(manifest)
    if pod is None:
        return
    if pod["name"] != manifest["name"]:
        raise RuntimeError("Refusing to terminate a pod with the wrong name")
    try:
        api("DELETE", "/pods/" + pod["id"])
    except urllib.error.HTTPError as error:
        if error.code != 404:
            raise


def watchdog(path):
    """Independent process: retry termination until API confirms disappearance."""
    while True:
        m = json.loads(path.read_text())
        if m.get("terminated"):
            return
        if time.time() >= m["deadline"] or m.get("terminate_requested"):
            try:
                terminate(m)
                # A create request may still be in flight briefly after failure.
                if matching_pod(m) is None and time.time() > m["created_intent"] + 120:
                    write_json({"confirmed_at": time.time()}, path.with_name("termination-confirmed.json"))
                    return
            except Exception as error:
                print("Termination retry:", type(error).__name__, flush=True)
        time.sleep(10)


def ssh_options(job, key):
    return ["-i", str(key), "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new",
            "-o", "UserKnownHostsFile=" + str(job / "known_hosts"), "-o", "ConnectTimeout=10",
            "-o", "ServerAliveInterval=10", "-o", "ServerAliveCountMax=2"]


def execute(bundle, approved_budget):
    if approved_budget != BUDGET:
        raise ValueError(f"This proposal requires its exact ${BUDGET:g} cap")
    fresh_quote()  # Read-only price check before the sole create request.
    name = JOB_PREFIX + uuid.uuid4().hex[:12]
    job = JOB_ROOT / name
    job.mkdir()
    key = job / "ssh-key"
    subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(key), "-C", name], check=True)
    now = time.time()
    m = {"name": name, "created_intent": now, "deadline": now + MAX_MINUTES * 60,
         "budget_cap_usd": approved_budget, "proposal": proposal()}
    manifest_path = job / "pod.json"
    write_json(m, manifest_path)
    with (job / "watchdog.log").open("a") as log:
        subprocess.Popen([sys.executable, __file__, "--watchdog", str(manifest_path)],
                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    startup = "\n".join([
        "set -eu", "apt-get update -qq", "apt-get install -y -qq openssh-server rsync python3-venv",
        "mkdir -p /root/.ssh /run/sshd /workspace/scale", "chmod 700 /root/.ssh",
        "printf '%s\\n' \"$PUBLIC_KEY\" > /root/.ssh/authorized_keys", "chmod 600 /root/.ssh/authorized_keys",
        "ssh-keygen -A", "exec /usr/sbin/sshd -D -e"])
    payload = {"name": name, "imageName": IMAGE, "gpuTypeIds": [GPU], "gpuCount": 1,
        "cloudType": "SECURE", "computeType": "GPU", "interruptible": False,
        "containerDiskInGb": DISK_GB, "volumeInGb": 0, "minRAMPerGPU": 180, "minVCPUPerGPU": 16,
        "ports": ["22/tcp"], "supportPublicIp": True,
        "dockerEntrypoint": ["/bin/bash"], "dockerStartCmd": ["-c", startup],
        "env": {"PUBLIC_KEY": key.with_suffix(".pub").read_text().strip()}}
    output = RESULTS / name
    output.mkdir()
    remote = None
    def sync():
        if remote:
            command, host = remote
            try:
                copied = subprocess.run(["rsync", "-az", "--partial", "--timeout=15", "--exclude=teacher_logits/", "--exclude=*.tmp",
                    "-e", shlex.join(command), host + ":/workspace/scale/results/", str(output) + "/"],
                    timeout=40, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except subprocess.TimeoutExpired:
                return False
            return copied.returncode == 0
        return False
    try:
        pod = api("POST", "/pods", payload)  # PAID: only after explicit user approval.
        m["pod_id"] = pod["id"]
        m["quoted_pod_rate"] = pod.get("costPerHr")
        write_json(m, manifest_path)
        if float(pod.get("costPerHr") or GPU_RATE) > GPU_RATE + DISK_RATE + .01:
            raise RuntimeError("Allocated price exceeds approved proposal")
        print("Created", m["pod_id"], "deadline", m["deadline"], flush=True)
        while time.time() < min(m["deadline"] - 120, now + 900):
            pod = api("GET", "/pods/" + m["pod_id"])
            ip, port = pod.get("publicIp"), (pod.get("portMappings") or {}).get("22")
            if ip and port:
                command = ["ssh", *ssh_options(job, key), "-p", str(port)]
                host = "root@" + ip
                check = subprocess.run([*command, host, "test -d /workspace/scale"],
                    timeout=20, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                if check.returncode == 0:
                    remote = command, host
                    break
            time.sleep(10)
        if remote is None:
            raise TimeoutError("Pod did not become ready within 15 minutes")
        command, host = remote
        with bundle.open("rb") as archive:
            subprocess.run([*command, host, "tar xzf - -C /workspace/scale"], stdin=archive,
                timeout=max(1, min(BUNDLE_UPLOAD_TIMEOUT, m["deadline"] - time.time() - 120)), check=True)
        # Worker deadline includes model downloads, package setup and compilation.
        worker_deadline = m["deadline"] - 120
        setup = "\n".join(["set -eu", "cd /workspace/scale", "mkdir -p results",
            "python -m venv --system-site-packages env",
            "env/bin/python -m pip install --no-cache-dir -r scale_requirements.txt > results/setup.log 2>&1",
            "SCALE_JOB_DEADLINE=" + str(worker_deadline) + " env/bin/python -u scale_worker.py > results/worker.log 2>&1"])
        setup = setup + "\n"
        remote_start = "cd /workspace/scale || exit; nohup bash -c " + shlex.quote(setup) + \
            " > /workspace/scale/results-launch.log 2>&1 < /dev/null &"
        subprocess.run([*command, host, remote_start], timeout=20, check=True)
        while time.time() < m["deadline"] - 30:
            sync()
            state_file = output / "status.json"
            if state_file.exists():
                state = json.loads(state_file.read_text())
                print(state["stage"], state.get("model", ""), flush=True)
                if state["stage"] in ("complete", "failed"):
                    break
            elif time.time() > now + 1200:
                raise TimeoutError("Worker did not start within 20 minutes")
            time.sleep(20)
        m["final_sync_succeeded"] = sync()
        artifact_manifest = output / "artifact-manifest.json"
        if artifact_manifest.exists():
            digests = json.loads(artifact_manifest.read_text())
            m["artifact_checksums_verified"] = all((output / name).is_file() and
                hashlib.sha256((output / name).read_bytes()).hexdigest() == digest for name, digest in digests.items())
    finally:
        m["terminate_requested"] = True
        write_json(m, manifest_path)
        # Never postpone termination to finish artifact copying.
        terminate(m)
        if m.get("pod_id") and matching_pod(m) is None:
            m["terminated"] = True
            m["terminated_at"] = time.time()
            write_json(m, manifest_path)
        write_json({k: v for k, v in m.items() if k != "proposal"}, output / "job-lifecycle.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--approved-budget-usd", type=float)
    ap.add_argument("--watchdog", type=Path)
    args = ap.parse_args()
    if args.watchdog:
        watchdog(args.watchdog)
    else:
        bundle = prepare_bundle()
        if args.execute:
            execute(bundle, args.approved_budget_usd)
        else:
            print(json.dumps(proposal(), indent=2))
            print("Prepared", bundle, "— no paid action performed.")
