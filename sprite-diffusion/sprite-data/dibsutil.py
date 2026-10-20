# Whether another session waits on the gpu lock this process holds; 2026-10-18.
import os, subprocess


def gpu_wanted():
    try:
        out = subprocess.run([os.environ.get('DIBS_EXE', 'dibs'), 'status'], capture_output=True, text=True, timeout=60).stdout
    except Exception:
        return False
    return any(l.startswith('gpu') and 'waiting:' in l for l in out.splitlines())
