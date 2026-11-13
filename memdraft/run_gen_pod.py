"""Rent ONE pod on which the 9B (Q4_K_M, llama.cpp with CUDA) answers the coding prompts the local
CPU run does not reach: the prompts from place FIRST of gen_code.py's order to the end. Rows come
home as they are written, to target/memdraft/code/pod/<job>/rows/rows-<from>-<to>.jsonl.

Default only packages the bundle and prints its path; --quote asks the price. --execute rents the
pod. On 2026-11-11 the user said 'Do it' to renting RunPod time for this under a $15 cap
(runpod_budget_gen_20261111.json). One 48 GB GPU at most 0.60 USD/h, deleted on completion,
failure or the deadline by this launcher and by an independent watchdog; gen_pod_worker.py, the
worker, also deletes its own pod 15 minutes after the deadline if both have died.

The pod lifecycle is manifold-compression/run_slim_pod.py's, with this job's values. Nothing the
provider sees names this project: the pod is called gen-<random>, and the client has a plain name.

    setsid nohup dibs run runpod-memdraft network --label "MemDraft: a rented GPU writes the 9B's code answers" --eta 24h -- \\
        target/manifold-audit/env/bin/python memdraft/run_gen_pod.py --ledger runpod_budget_gen_20261111.json --execute \\
        >> target/memdraft/code/pod/launch.log 2>&1 < /dev/null &
    stop early (deletes the pod; rows copied so far stay):
        pkill -f '^[^ ]*python memdraft/run_gen_pod.py .*--execute'
"""
import argparse
import json
from pathlib import Path
import signal
import sys
import tarfile
import urllib.request

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / 'manifold-compression'))
import run_scale_pod as base
import run_slim_pod as pod

import gen_code
import gen_pod_worker as worker

FIRST = 6200          # run_gen_more.sh answers places 2000 to 6199 on this machine; the 2000 before those are held out
CAP = 15.0
LOOP = 160000         # more steps than gen_code.pick needs to hand out every prompt
CODE = ['gen_code.py', 'gpu_alone.py', 'live_bench.py', 'gen_pod_worker.py', 'gen_pod_requirements.txt']
CACHE = Path.home() / '.cache/model-experiments/memdraft'
# The files this machine generates with, by the hashes Hugging Face lists for them.
MODEL = {'repo': 'unsloth/Qwen3.5-9B-GGUF', 'revision': '3885219b6810b007914f3a7950a8d1b469d598a5',
         'file': 'Qwen3.5-9B-Q4_K_M.gguf', 'sha256': '03b74727a860a56338e042c4420bb3f04b2fec5734175f4cb9fa853daf52b7e8',
         'local': CACHE / 'models/Qwen3.5-9B-Q4_K_M.gguf'}
DATASET = {'repo': 'ise-uiuc/Magicoder-OSS-Instruct-75K', 'revision': '5f839b1f368a76b161028bb9edff055db34022b2',
           'file': 'data-oss_instruct-decontaminated.jsonl',
           'sha256': 'f988f14d3516a9f958eeea310f63903727c6d3463bc9cb435288b6ca4bb09cf9',
           'local': CACHE / 'code-prompts-src/data-oss_instruct-decontaminated.jsonl'}
# The commit the local llama.cpp is a patched copy of (the patch adds the drafter and changes nothing a plain
# server runs). tree_sha256 is gen_pod_worker.tree_sha of the unpacked archive. The options are the local CPU
# build's, with CUDA; no web UI is fetched or built.
COMMIT = '7d7f0287530bc8981690f361d8c1627dbae1c99a'
LLAMA = {'commit': COMMIT, 'url': f'https://github.com/ggml-org/llama.cpp/archive/{COMMIT}.tar.gz',
         'tree_sha256': '642c20d6bda6676bc0febdd4d4c8c17c3deeade7e5610c8d2af85e6519a4078b',
         'cmake': ['-DCMAKE_BUILD_TYPE=Release', '-DGGML_CUDA=ON', '-DGGML_CCACHE=OFF', '-DLLAMA_OPENSSL=OFF',
                   '-DLLAMA_USE_PREBUILT_UI=OFF', '-DLLAMA_BUILD_TESTS=OFF']}
# max_tokens and ctx are run_gen_more.sh's. try_parallel: the worker doubles the answers at a time while that is
# at least 8% faster, taking no new prompt after try_seconds (first try, later tries). piece: prompts per rows file.
SETTINGS = {'max_tokens': 1024, 'ctx': 3072, 'piece': 4000, 'try_parallel': [32, 64, 128, 256], 'try_seconds': [240, 180]}

pod.JOBS = REPO / 'target/memdraft/pod'
pod.OUTPUTS = REPO / 'target/memdraft/code/pod'
pod.CAP = CAP
pod.GPUS = ('NVIDIA A40', 'NVIDIA RTX A6000')
pod.RATE = 0.60
pod.DISK_GB = 60
pod.DISK_RATE = pod.DISK_GB * .10 / (30 * 24)
pod.PREFIX, pod.ARCHIVE = 'gen-', 'gen-bundle.tar'
# The launcher runs these two as manifold-compression/<name> on the pod; the bundle has that directory, empty.
pod.WORKER, pod.REQUIREMENTS = '../memdraft/gen_pod_worker.py', '../memdraft/gen_pod_requirements.txt'
# pytorch/pytorch:2.10.0-cuda12.6-cudnn9-devel (7.9 GB): Ubuntu 24.04 with the CUDA 12.6 toolkit, and a
# python on the default path as in the image the other launchers use.
pod.IMAGE = 'pytorch/pytorch@sha256:df80e10d07cd114c5f33380e3df7b6c5a3caab8481f68509ea652a7c0908316e'
# execute() starts its watchdog as `python <its own file> --watchdog <pod.json>`: make that this file, so the
# watchdog also talks to the provider through api() below.
pod.__file__ = str(Path(__file__).resolve())


def api(method, path, data=None):
    """run_scale_pod.api, with a client name that says nothing."""
    request = urllib.request.Request('https://rest.runpod.io/v1' + path,
        data=json.dumps(data).encode() if data is not None else None, method=method,
        headers={'Authorization': 'Bearer ' + base.credential(), 'Content-Type': 'application/json',
                 'User-Agent': 'Mozilla/5.0', 'Accept': 'application/json'})
    with urllib.request.urlopen(request, timeout=20) as response:
        body = response.read()
        return json.loads(body) if body else None


base.api = pod.api = api      # base: matching_pod and terminate; pod: execute


def nobody(info):
    """Tar entries without this machine's user name."""
    info.uid = info.gid = 0
    info.uname = info.gname = 'root'
    return info


def bundle():
    for pinned in (MODEL, DATASET):
        if pod.sha(pinned['local']) != pinned['sha256']:
            raise ValueError(f"{pinned['local']} is not the pinned {pinned['file']}")
    if FIRST < 2000:
        raise ValueError('The first 2000 prompts are held out')
    order = gen_code.pick(str(DATASET['local']), LOOP, gen_code.SEED, gen_code.MAX_CHARS)
    if order != gen_code.pick(str(DATASET['local']), 2 * LOOP, gen_code.SEED, gen_code.MAX_CHARS):
        raise ValueError('LOOP does not reach every prompt')
    files = {f'memdraft/{name}': pod.sha(HERE / name) for name in CODE}
    pod.JOBS.mkdir(parents=True, exist_ok=True)
    manifest = pod.JOBS / 'bundle-manifest.json'
    pod.write_json({'files': files,
                    'model': {k: v for k, v in MODEL.items() if k != 'local'},
                    'dataset': {k: v for k, v in DATASET.items() if k != 'local'},
                    'llama_cpp': LLAMA,
                    'prompts': {'first': FIRST, 'count': len(order) - FIRST, 'loop': LOOP,
                                'ids_sha256': worker.ids_sha(order[FIRST:])},
                    'budget': {'cap_usd': CAP, 'usd_per_hour': pod.RATE + pod.DISK_RATE, 'hours': pod.HOURS},
                    'settings': SETTINGS}, manifest)
    path = pod.JOBS / pod.ARCHIVE
    with tarfile.open(path, 'w') as archive:
        for relative in files:
            archive.add(REPO / relative, arcname=relative, recursive=False, filter=nobody)
        archive.add(manifest, arcname='bundle-manifest.json', filter=nobody)
        holder = tarfile.TarInfo('manifold-compression')
        holder.type, holder.mode, holder.uname, holder.gname = tarfile.DIRTYPE, 0o755, 'root', 'root'
        archive.addfile(holder)
    pod.write_json({'path': str(path), 'sha256': pod.sha(path), 'bytes': path.stat().st_size}, pod.JOBS / 'bundle.json')
    return path


def stop(signum, frame):
    raise KeyboardInterrupt      # so that execute()'s cleanup runs: it deletes the pod and closes the ledger


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--ledger', help='the budget file, in memdraft/')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--quote', action='store_true')
    parser.add_argument('--watchdog', type=Path)
    args = parser.parse_args()
    if args.watchdog:
        pod.watchdog(args.watchdog)
    elif args.quote:
        print(json.dumps(pod.quote()), flush=True)
    else:
        pod.LEDGER = HERE / args.ledger
        hourly = pod.RATE + pod.DISK_RATE
        # As long as the money left allows, at most a day.
        pod.HOURS = min(24, int((CAP - json.loads(pod.LEDGER.read_text()).get('spent_usd', 0)) / hourly * 10) / 10)
        pod.JOB_CAP = hourly * pod.HOURS
        archive = bundle()
        if args.execute:
            signal.signal(signal.SIGTERM, stop)
            signal.signal(signal.SIGINT, stop)
            pod.execute(archive)
        else:
            print(str(archive), f'{pod.HOURS} h at most, {pod.JOB_CAP:.2f} USD at most')
