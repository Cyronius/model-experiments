"""Rent ONE RunPod GPU, fine-tune Qwen3.5-0.8B on the npuasm tasks with train_small.py, sync the
result back, and delete the pod.

    python3 pod/pod_small.py --quote            price check only (free)
    python3 pod/pod_small.py --execute NAME     paid run; output in runs/NAME/
      [--ledger FILE] [--data DIR] [--init MODEL_DIR] [--hours H] [--train-hours H]
      --init starts from a local checkpoint (uploaded with the bundle) instead of the base model.

Every paid run needs its own user approval, recorded in a ledger file whose total_cap_usd and
status gate the launch (the ledger named below is the first run's).

The user approved this retrain on RunPod on 2026-11-12 ("yeah use runpod. you're allowed to use
cheaper/slower if it will lower the cost. i don't mind waiting as much as 5 hours"), against the
proposal "under an hour on one H100, likely under $5". The ledger (runpod_budget_20261112.json)
caps it at $5: one 24-48 GB GPU at <= RATE USD/h, deleted on completion, failure, or the HOURS
deadline by an independent watchdog process. Not a standing authorization for other jobs.

Pod helpers (API, credential, termination, ssh options, image) come from
manifold-compression/run_scale_pod.py, which earlier paid jobs used.
"""
from __future__ import annotations

import argparse
import fcntl
import gzip
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRAIN = HERE.parent
REPO = TRAIN.parent.parent
sys.path.insert(0, str(REPO / "manifold-compression"))
from run_scale_pod import IMAGE, api, credential, matching_pod, ssh_options, terminate, write_json  # noqa: E402

LEDGER = TRAIN / "runpod_budget_20261112.json"
JOBS = REPO / "target/nkc-pod"
CAP = 5.0
# Best expected cost first (price x expected hours). The run needs ~12 GB of GPU memory.
GPUS = ("NVIDIA RTX A5000", "NVIDIA L40", "NVIDIA A40", "NVIDIA RTX A6000", "NVIDIA RTX 5000 Ada Generation",
        "NVIDIA GeForce RTX 4090", "NVIDIA GeForce RTX 3090 Ti", "NVIDIA GeForce RTX 3090")  # Ampere/Ada: the image's CUDA 12.6 torch has no Blackwell
RATE = 0.89
DISK_GB = 40
DISK_RATE = DISK_GB * .10 / (30 * 24)
HOURS = 5.0
TRAIN_HOURS = 3.9  # leaves time for setup, final eval/save and the sync inside HOURS
JOB_CAP = (RATE + DISK_RATE) * HOURS
CODE = {"train_small.py": TRAIN / "train_small.py", "memguard.py": TRAIN / "memguard.py",
        "worker.sh": HERE / "worker.sh"}
DATA = {"data/train.jsonl.gz": TRAIN / "data/tasks/train.jsonl", "data/val.jsonl.gz": TRAIN / "data/tasks/val.jsonl"}
INIT = None  # local checkpoint dir to start from (--init)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def quote() -> dict:
    q = "{ gpuTypes { id securePrice lowestPrice(input: {gpuCount: 1, secureCloud: true}) { stockStatus uninterruptablePrice } } }"
    req = urllib.request.Request("https://api.runpod.io/graphql", data=json.dumps({"query": q}).encode(),
                                 headers={"Authorization": "Bearer " + credential(), "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=20) as r:
        rows = json.load(r)["data"]["gpuTypes"]
    ok = []
    for gpu in GPUS:
        row = next((r for r in rows if r["id"] == gpu), None)
        if not row or row.get("securePrice") is None:
            continue
        low = row.get("lowestPrice") or {}
        if float(row["securePrice"]) <= RATE:  # stock flags proved unreliable; the create decides
            ok.append({"gpu": gpu, "rate": float(row["securePrice"]), "stock_status": low["stockStatus"]})
    if not ok:
        raise RuntimeError(f"none of {GPUS} in stock at <= {RATE} USD/h")
    return {"gpu": ok[0]["gpu"], "gpus": [o["gpu"] for o in ok], "options": ok, "checked_at": time.time()}


def bundle(job: Path) -> Path:
    path = job / "bundle.tar"
    with tarfile.open(path, "w") as tar:
        for name, src in CODE.items():
            tar.add(src, arcname=name)
        for name, src in DATA.items():
            gz = job / Path(name).name
            with src.open("rb") as i, gzip.open(gz, "wb") as o:
                shutil.copyfileobj(i, o)
            tar.add(gz, arcname=name)
        if INIT:
            for f in sorted(INIT.iterdir()):
                tar.add(f, arcname=f"init/{f.name}")
    write_json({"files": {n: sha(p) for n, p in {**CODE, **DATA}.items()}, "bundle_sha256": sha(path)},
               job / "bundle.json")
    return path


def watchdog(path: Path) -> None:
    write_json({"pid": os.getpid(), "ready_at": time.time()}, path.with_name("watchdog-ready.json"))
    while True:
        m = json.loads(path.read_text())
        if m.get("terminated"):
            return
        if time.time() >= m["deadline"] or m.get("terminate_requested"):
            try:
                terminate(m)
                confirm_after = m["created_intent"] + 120 if m.get("pod_id") or m.get("create_refused") else m["deadline"]
                if matching_pod(m) is None and time.time() > confirm_after:
                    write_json({"confirmed_at": time.time()}, path.with_name("termination-confirmed.json"))
                    return
            except Exception as e:  # noqa: BLE001
                print("deletion retry:", type(e).__name__, flush=True)
        time.sleep(30)


def execute(run_name: str) -> None:
    JOBS.mkdir(parents=True, exist_ok=True)
    output = TRAIN / "runs" / run_name
    if output.exists():
        raise FileExistsError(output)
    with (JOBS / "launch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ledger = json.loads(LEDGER.read_text())
        if ledger["status"] not in ("authorized_not_launched", "terminated"):
            raise ValueError("ledger does not allow a launch")
        CAP = ledger["total_cap_usd"]
        if ledger.get("spent_usd", 0) + JOB_CAP > CAP + 1e-6:
            raise ValueError("job could exceed the remaining budget")
        checked = quote()
        name = "nkc-small-" + uuid.uuid4().hex[:10]
        job = JOBS / name
        job.mkdir()
        output.mkdir(parents=True)
        archive = bundle(job)
        key = job / "ssh-key"
        subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(key), "-C", name], check=True)
        now = time.time()
        manifest = {"name": name, "run": run_name, "created_intent": now, "deadline": now + HOURS * 3600,
                    "budget_cap_usd": CAP, "job_max_estimated_usd": JOB_CAP, "quote": checked,
                    "container_disk_gb": DISK_GB, "volume_gb": 0, "bundle_sha256": sha(archive)}
        path = job / "pod.json"
        write_json(manifest, path)
        with (job / "watchdog.log").open("a") as stream:
            guard = subprocess.Popen([sys.executable, __file__, "--watchdog", str(path)], stdout=stream,
                                     stderr=subprocess.STDOUT, start_new_session=True)
        for _ in range(50):
            if guard.poll() is not None:
                raise RuntimeError("watchdog failed before launch")
            if (job / "watchdog-ready.json").exists():
                break
            time.sleep(.1)
        else:
            raise RuntimeError("no watchdog readiness before launch")
        ledger.update(status="launch_reserved", active_job=str(path.relative_to(REPO)), reserved_job_max_usd=JOB_CAP,
                      fresh_quote=checked)
        write_json(ledger, LEDGER)
        startup = "\n".join(["set -eu", "apt-get update -qq", "apt-get install -y -qq openssh-server rsync python3-venv",
                             "mkdir -p /root/.ssh /run/sshd /workspace/nkc", "chmod 700 /root/.ssh",
                             "printf '%s\\n' \"$PUBLIC_KEY\" > /root/.ssh/authorized_keys",
                             "chmod 600 /root/.ssh/authorized_keys", "ssh-keygen -A",
                             "exec /usr/sbin/sshd -D -e -o PasswordAuthentication=no"])
        payload = {"name": name, "imageName": IMAGE, "gpuTypeIds": checked["gpus"], "gpuTypePriority": "custom", "gpuCount": 1,
                   "cloudType": "SECURE", "computeType": "GPU", "interruptible": False,
                   "containerDiskInGb": DISK_GB, "volumeInGb": 0, "minRAMPerGPU": 24, "minVCPUPerGPU": 4,
                   "ports": ["22/tcp"], "supportPublicIp": True, "dockerEntrypoint": ["/bin/bash"],
                   "dockerStartCmd": ["-c", startup], "env": {"PUBLIC_KEY": key.with_suffix(".pub").read_text().strip()}}
        remote = None

        def sync() -> bool:
            if not remote or time.time() >= manifest["deadline"] - 20:
                return False
            cmd, host = remote
            try:
                r = subprocess.run(["rsync", "-az", "--partial", "--timeout=30", "-e", shlex.join(cmd),
                                    host + ":/workspace/nkc/outputs/", str(output) + "/"],
                                   timeout=min(900, max(1, manifest["deadline"] - time.time() - 20)),
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return r.returncode == 0
            except subprocess.TimeoutExpired:
                return False

        try:
            # Retry only an explicit "no instances" refusal (nothing was created), alternating Secure and
            # Community cloud, for up to 40 minutes. Any other error is ambiguous: no retry.
            pod, tries = None, 0
            while pod is None:
                payload["cloudType"] = ("SECURE", "COMMUNITY")[tries % 2]
                try:
                    pod = api("POST", "/pods", payload)
                except urllib.error.HTTPError as e:
                    body = e.read()[:500].decode(errors="replace")
                    tries += 1
                    refused = "no instances" in body.lower() or "no longer any instances" in body.lower()
                    if not refused and e.code != 429:
                        print("create failed:", e.code, body, flush=True)
                        raise
                    if time.time() > now + 2400:
                        manifest["create_refused"] = True  # every create was refused: nothing to delete
                        write_json(manifest, path)
                        raise RuntimeError(f"no GPU available after {tries} tries") from None
                    if tries % 10 == 1:
                        print("create refused (", payload["cloudType"], e.code, "), retrying", time.strftime("%H:%M"),
                              flush=True)
                    time.sleep(90 if e.code == 429 else 45)
            created = time.time()
            # The pod's 5 hours start at its creation (the cost cap is per pod-hour); time spent waiting for
            # a GPU comes out of the training budget instead, so the whole job stays near 5 hours.
            train_hours = max(1.0, TRAIN_HOURS - (created - now) / 3600)
            manifest.update(cloud=payload["cloudType"], created=created, deadline=created + HOURS * 3600,
                            train_hours=train_hours)
            manifest.update(pod_id=pod["id"], allocated_rate=pod.get("costPerHr"))
            write_json(manifest, path)
            if pod.get("costPerHr") is None or not 0 < float(pod["costPerHr"]) <= RATE + DISK_RATE + 1e-6:
                raise RuntimeError("allocated rate missing or above the approved ceiling")
            ledger.update(status="running", pod_id=pod["id"])
            write_json(ledger, LEDGER)
            print("created", pod["id"], (pod.get("machine") or {}).get("gpuTypeId", checked["gpus"]), pod.get("costPerHr"), "USD/h; deadline",
                  time.strftime("%H:%M", time.localtime(manifest["deadline"])), flush=True)
            while time.time() < min(created + 900, manifest["deadline"] - 180):
                pod = api("GET", "/pods/" + manifest["pod_id"])
                ip, port = pod.get("publicIp"), (pod.get("portMappings") or {}).get("22")
                if ip and port:
                    cmd = ["ssh", *ssh_options(job, key), "-p", str(port)]
                    host = "root@" + ip
                    ok = subprocess.run([*cmd, host, "test -d /workspace/nkc"], timeout=20,
                                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    if ok.returncode == 0:
                        remote = cmd, host
                        break
                time.sleep(10)
            if remote is None:
                raise TimeoutError("pod not reachable within 15 minutes")
            cmd, host = remote
            with archive.open("rb") as f:
                subprocess.run([*cmd, host, "tar xf - -C /workspace/nkc"], stdin=f, timeout=2400, check=True)
            # Only the backgrounded command may hold the session's fds, or ssh waits for it to exit.
            launch = ("cd /workspace/nkc || exit 1; TIME_BUDGET_H=" + str(round(train_hours, 2)) +
                      " setsid nohup bash worker.sh > launch.log 2>&1 < /dev/null &")
            subprocess.run([*cmd, "-n", host, launch], timeout=30, check=True)
            print("worker started", flush=True)
            last = None
            while time.time() < manifest["deadline"] - 600:
                sync()
                sp = output / "status.json"
                if sp.exists():
                    stage = json.loads(sp.read_text())["stage"]
                    if stage != last:
                        print("stage", stage, time.strftime("%H:%M"), flush=True)
                        last = stage
                    if stage in ("complete", "failed"):
                        break
                elif time.time() > created + 1800:
                    raise TimeoutError("worker did not start within 30 minutes")
                if guard.poll() is not None and not (job / "termination-confirmed.json").exists():
                    raise RuntimeError("watchdog exited unexpectedly")
                time.sleep(60)
            manifest["final_sync_succeeded"] = sync()
        finally:
            manifest["terminate_requested"] = True
            write_json(manifest, path)
            try:
                terminate(manifest)
                if manifest.get("pod_id") and matching_pod(manifest) is None:
                    manifest.update(terminated=True, terminated_at=time.time())
            finally:
                write_json(manifest, path)
                write_json(manifest, output / "job-lifecycle.json")
                if manifest.get("terminated"):
                    rate = float(manifest.get("allocated_rate") or RATE + DISK_RATE)
                    est = (manifest["terminated_at"] - manifest.get("created", now)) / 3600 * rate
                    ledger.update(status="terminated", spent_usd=round(ledger.get("spent_usd", 0) + est, 4),
                                  billing_basis="elapsed time at the allocated rate, not the provider invoice",
                                  reserved_job_max_usd=0, terminated_at=manifest["terminated_at"])
                else:
                    ledger.update(status="termination_pending_watchdog", reserved_job_max_usd=JOB_CAP)
                write_json(ledger, LEDGER)
            print("deletion confirmed:", bool(manifest.get("terminated")), "spent so far (est.):",
                  ledger.get("spent_usd"), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quote", action="store_true")
    ap.add_argument("--execute", metavar="NAME")
    ap.add_argument("--watchdog", type=Path)
    ap.add_argument("--ledger", type=Path, default=LEDGER)
    ap.add_argument("--data", type=Path, default=TRAIN / "data/tasks")
    ap.add_argument("--init", type=Path)
    ap.add_argument("--hours", type=float, default=HOURS, help="pod lifetime (the cost bound)")
    ap.add_argument("--train-hours", type=float, default=TRAIN_HOURS)
    a = ap.parse_args()
    LEDGER, INIT, HOURS, TRAIN_HOURS = a.ledger.resolve(), a.init, a.hours, a.train_hours
    JOB_CAP = (RATE + DISK_RATE) * HOURS
    DATA = {"data/train.jsonl.gz": a.data / "train.jsonl", "data/val.jsonl.gz": a.data / "val.jsonl"}
    assert TRAIN_HOURS < HOURS - 0.5 and all(p.exists() for p in DATA.values())
    assert INIT is None or (INIT / "model.safetensors").exists()
    if a.watchdog:
        watchdog(a.watchdog)
    elif a.execute:
        execute(a.execute)
    else:
        print(json.dumps(quote()))
