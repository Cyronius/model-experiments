"""Rent one bounded GPU for the missing 150M float-only training control.

Reuses the established project's API/SSH helpers without changing Grid 2 state.
Poll at least once per minute until collected; failures stop this pod and retain
its disk. Successful collection hashes every artifact before deleting this pod.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tarfile
import time

ROOT = Path(__file__).resolve().parent
STATE = ROOT / "pod-state.json"
REMOTE = "/workspace/tern-embed-float-control"
DEST = ROOT / "float-control"
spec = importlib.util.spec_from_file_location("grid_helpers", ROOT.parent / "run_grid2.py")
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)
helpers.LOGS = ROOT
helpers.REMOTE = REMOTE


def save(pod):
    STATE.write_text(json.dumps(pod, indent=2))


def stop(pod, reason):
    helpers.api("POST", f"pods/{pod['id']}/stop")
    pod.update(local_status="stopped", reason=reason)
    save(pod)
    print(f"Stopped own control pod {pod['id']}: {reason}; disk retained", flush=True)


def launch():
    if STATE.exists():
        raise RuntimeError("State exists; inspect it instead of renting twice")
    public_key = helpers.KEY.with_suffix(".pub").read_text().strip()
    pod = helpers.api("POST", "pods", dict(
        name="tern-embed-float150-control", cloudType="SECURE", computeType="GPU", gpuCount=1,
        gpuTypeIds=["NVIDIA RTX A5000", "NVIDIA RTX A4500", "NVIDIA RTX 4000 Ada Generation"],
        gpuTypePriority="custom", templateId="runpod-torch-v280", containerDiskInGb=30, volumeInGb=30,
        volumeMountPath="/workspace", supportPublicIp=True, ports=["22/tcp"], env={"PUBLIC_KEY": public_key}))
    pod.update(local_started=time.time(), local_status="created", max_seconds=3600)
    save(pod)
    if not pod.get("costPerHr") or pod["costPerHr"] > 0.30:
        stop(pod, "exceeds $0.30/hr cap")
        raise RuntimeError("Rate cap failed")
    print(json.dumps({k:pod.get(k) for k in ("id","costPerHr","desiredStatus")}), flush=True)


def setup(pod):
    if pod.get("local_status") != "created":
        return
    try:
        helpers.remote(pod, "python -c 'import torch; assert torch.cuda.is_available(); "
                       "x=torch.randn(512,512,device=\"cuda\"); print((x@x).norm().item())'", timeout=30)
    except subprocess.CalledProcessError:
        print("SSH/CUDA not ready", flush=True)
        return
    archive = ROOT / "control-code.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for name in ("model.py", "data.py", "train.py", "evaluate.py"):
            tar.add(ROOT.parent / "tern-embed" / name, arcname=name)
        tar.add(ROOT / "float_control.sh", arcname="float_control.sh")
    helpers.remote(pod, f"mkdir -p {REMOTE}")
    helpers.copy(pod, archive, REMOTE + "/code.tar.gz", upload=True)
    helpers.remote(pod, f"cd {REMOTE} && tar xzf code.tar.gz --no-same-owner && "
                   "(nohup timeout 55m bash float_control.sh > control.log 2>&1 < /dev/null & echo $! > control.pid)", timeout=30)
    pod["local_status"] = "running"
    save(pod)
    print("Float-only control launched, 55-minute job limit; one-hour pod guard", flush=True)


def collect(pod):
    helpers.remote(pod, f"cd {REMOTE} && python - <<'PY'\n"
        "import hashlib,json,pathlib,tarfile\n"
        "from huggingface_hub.constants import HF_HUB_CACHE\n"
        "revisions={p.name:[s.name for s in (p/'snapshots').iterdir()] for p in pathlib.Path(HF_HUB_CACHE).glob('models--*') if (p/'snapshots').exists()}\n"
        "pathlib.Path('hf-revisions.json').write_text(json.dumps(revisions,indent=2))\n"
        "corpus={}\n"
        "for name in ('ids.npy','offsets.npy','held.npy','pairs.npy','teacher.npy'):\n"
        " h=hashlib.sha256()\n"
        " with pathlib.Path('work',name).open('rb') as f:\n"
        "  for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)\n"
        " corpus[name]=h.hexdigest()\n"
        "pathlib.Path('corpus.sha256.json').write_text(json.dumps(corpus,indent=2))\n"
        "files=[p for root in ('runs','logs','results') for p in pathlib.Path(root).rglob('*') if p.is_file()]\n"
        "files += [pathlib.Path(n) for n in ('control.log','control.status','work/stats.json','environment.txt','hf-revisions.json','corpus.sha256.json',"
        "'model.py','data.py','train.py','evaluate.py','float_control.sh') if pathlib.Path(n).exists()]\n"
        "checks={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}\n"
        "pathlib.Path('artifacts.sha256.json').write_text(json.dumps(checks,indent=2))\n"
        "with tarfile.open('artifacts.tar.gz','w:gz') as t:\n"
        " for p in files+[pathlib.Path('artifacts.sha256.json')]: t.add(p)\n"
        "print('packed',len(files))\nPY", timeout=180)
    DEST.mkdir(exist_ok=True)
    archive = DEST / "artifacts.tar.gz"
    helpers.copy(pod, REMOTE + "/artifacts.tar.gz", archive)
    with tarfile.open(archive) as tar:
        for entry in tar.getmembers():
            target = (DEST / entry.name).resolve()
            if not target.is_relative_to(DEST.resolve()) or not entry.isfile():
                raise ValueError(f"Unsafe archive entry: {entry.name}")
        tar.extractall(DEST)
    checks = json.loads((DEST / "artifacts.sha256.json").read_text())
    for name, expected in checks.items():
        if hashlib.sha256((DEST / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Artifact hash mismatch: {name}")
    if "runs/float150/model.pt" not in checks:
        stop(pod, "Incomplete training; partial artifacts retained locally")
        raise RuntimeError("Missing final float model")
    tasks = {Path(name).stem for name in checks if name.startswith("results/float150/")}
    if not {"SciFact", "NFCorpus", "STSBenchmark"} <= tasks:
        stop(pod, "Incomplete evaluation; checkpoint and partial results retained")
        raise RuntimeError("Missing task results")
    helpers.api("DELETE", f"pods/{pod['id']}")
    pod.update(local_status="complete", local_finished=time.time(), verified_artifacts=len(checks))
    save(pod)
    report = {"pod_id": pod["id"], "cost_per_hour": pod["costPerHr"],
              "elapsed_hours": (pod["local_finished"] - pod["local_started"]) / 3600,
              "verified_artifacts": len(checks), "pod_deleted": True,
              "teacher_batch": pod.get("teacher_batch", 512),
              "local_work_cache": pod.get("local_work_cache", False),
              "checkpoint_sha256": checks["runs/float150/model.pt"]}
    (ROOT / "control-run.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


def poll():
    pod = json.loads(STATE.read_text())
    if pod.get("local_status") in ("complete", "stopped"):
        print(pod["local_status"], flush=True)
        return
    if time.time() - pod["local_started"] > pod["max_seconds"]:
        stop(pod, "one-hour cost watchdog")
        return
    pod.update(helpers.api("GET", f"pods/{pod['id']}"))
    save(pod)
    if pod.get("local_status") == "created":
        if pod.get("publicIp") and (pod.get("portMappings") or {}).get("22"):
            setup(pod)
        else:
            print("Waiting for public SSH endpoint", flush=True)
        return
    state = helpers.remote(pod, f"cd {REMOTE} && cat control.status; "
                           "tail -n 2 control.log logs/corpus.log logs/teacher.log logs/train.log logs/evaluate.log 2>/dev/null || true")
    print(state, flush=True)
    stage = state.splitlines()[0]
    if stage == "done":
        collect(pod)
    elif stage == "failed":
        stop(pod, "training/evaluation job failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("launch", "poll", "watch", "collect", "stop"))
    action = parser.parse_args().action
    if action == "launch":
        launch()
    elif action == "poll":
        poll()
    elif action == "watch":
        while json.loads(STATE.read_text()).get("local_status") not in ("complete", "stopped"):
            try:
                poll()
            except Exception as error:
                print(f"Retry: {type(error).__name__}: {error}", flush=True)
            if json.loads(STATE.read_text()).get("local_status") not in ("complete", "stopped"):
                time.sleep(30)
    elif action == "collect":
        collect(json.loads(STATE.read_text()))
    else:
        stop(json.loads(STATE.read_text()), "owner/agent stop")
