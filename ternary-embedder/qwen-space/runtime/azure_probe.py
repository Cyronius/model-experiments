"""Azure App Service probe for the tde document embedder. Run on the server:

    python azure_probe.py [RELEASE_FOLDER]

Prints CPU model, core/thread counts, relevant CPU flags, glibc version and available memory.
If the tde package and model are present (RELEASE_FOLDER, default: this script's folder), it then
runs a 20-second benchmark at 64 and 504 tokens with 1, 2 and 4 worker threads, and prints one
JSON block. Uses only the standard library, numpy and tokenizers. Sends nothing anywhere.
"""
import json, os, platform, sys, time
from pathlib import Path


def cpu_info():
    info = {'machine': platform.machine(), 'logical_cpus': os.cpu_count()}
    try:
        info['affinity_cpus'] = len(os.sched_getaffinity(0))
    except Exception:
        pass
    try:
        txt = Path('/proc/cpuinfo').read_text()
        blocks = [{k.strip(): v.strip() for k, v in (l.split(':', 1) for l in b.splitlines() if ':' in l)}
                  for b in txt.split('\n\n') if b.strip()]
        first = blocks[0]
        info['model_name'] = first.get('model name')
        info['cpu_mhz'] = first.get('cpu MHz')
        flags = set(first.get('flags', '').split())
        want = ['avx', 'avx2', 'fma', 'f16c', 'avx512f', 'avx512bw', 'avx512vl', 'avx512dq', 'avx512_vnni',
                'avx_vnni', 'avx512_bf16', 'amx_int8', 'hypervisor']
        info['flags'] = {f: f in flags for f in want}
        cores = {(b.get('physical id'), b.get('core id')) for b in blocks}
        info['physical_cores_reported'] = len(cores)
        info['siblings'] = first.get('siblings'); info['cpu_cores'] = first.get('cpu cores')
    except Exception as e:
        info['cpuinfo_error'] = repr(e)
    try:
        info['glibc'] = os.confstr('CS_GNU_LIBC_VERSION')
    except Exception:
        info['glibc'] = platform.libc_ver()
    try:
        mem = dict(l.split(':', 1) for l in Path('/proc/meminfo').read_text().splitlines())
        info['mem_total_gb'] = int(mem['MemTotal'].split()[0]) / 2 ** 20
        info['mem_available_gb'] = int(mem['MemAvailable'].split()[0]) / 2 ** 20
    except Exception:
        pass
    try:
        info['loadavg'] = os.getloadavg()
    except Exception:
        pass
    info['python'] = sys.version.split()[0]
    return info


def bench(folder, seconds=20.0):
    sys.path.insert(0, str(folder))
    from tde import Embedder
    words = ('the quick brown fox jumps over a lazy dog while researchers measure protein binding in '
             'cell cultures and engineers tune database indexes for faster retrieval of documents').split()
    out = {}
    per = seconds / 6
    for ntok in (64, 504):
        text = ' '.join(words[i % len(words)] for i in range(ntok * 2))
        for workers in (1, 2, 4):
            e = Embedder(folder, workers=workers)
            ids = e.tokenize([text])[0]
            doc = e.tok.decode(ids)
            docs = [doc] * (16 * workers)
            e.encode(docs[:workers])  # warm up
            lat = []
            t0 = time.perf_counter(); n = 0
            while time.perf_counter() - t0 < per:
                t = time.perf_counter(); e.encode(docs); lat.append(time.perf_counter() - t); n += len(docs)
            el = time.perf_counter() - t0
            lat.sort()
            out[f'{ntok}tok_{workers}workers'] = {'isa': e.isa, 'content_tokens': len(e.tokenize([doc])[0]),
                                                  'docs_per_second': n / el, 'batch': len(docs),
                                                  'batch_median_ms': 1000 * lat[len(lat) // 2],
                                                  'batch_p95_ms': 1000 * lat[int(len(lat) * 0.95) - 1 if len(lat) > 1 else 0]}
            e.close()
        # single-document latency, one worker
        e = Embedder(folder, workers=1)
        lat = []
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < per / 2:
            t = time.perf_counter(); e.encode([doc]); lat.append(time.perf_counter() - t)
        lat.sort()
        out[f'{ntok}tok_single_doc'] = {'median_ms': 1000 * lat[len(lat) // 2], 'p95_ms': 1000 * lat[int(len(lat) * 0.95)]}
        e.close()
    return out


if __name__ == '__main__':
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent
    res = {'probe_version': 1, 'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'cpu': cpu_info()}
    if (folder / 'model.tde').exists():
        try:
            res['benchmark'] = bench(folder)
        except Exception as ex:
            res['benchmark_error'] = repr(ex)
    else:
        res['benchmark'] = f'skipped: no model.tde in {folder}'
    print(json.dumps(res, indent=2))
