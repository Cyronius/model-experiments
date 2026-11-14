#!/usr/bin/env python3
"""Wait until nothing else is using the GPU.

dibs can report the GPU free while a job is still on it (holds have vanished mid-run), and local
ROCm jobs compute wrong values silently when another process shares the GPU. So before any GPU
process of ours starts, look at the GPU itself: no other process may have it open, and it must be
idle, for several checks in a row.

    python gpu_alone.py --wait            # block until alone, then exit 0
    python gpu_alone.py                   # exit 0 if alone now, 1 if not (prints who is there)
    python gpu_alone.py --wait --preemptible-ok   # block until only preemptible jobs are left

A job under `dibs run --preemptible` has agreed to hand the GPU over when another session asks
dibs for it. --preemptible-ok is the moment to ask (`dibs run gpu ...`); the GPU is ours to use
only once that job has left, which is a second wait without the flag.

When that job's dibs hold has vanished, dibs no longer knows to tell it, and it keeps the GPU while we
hold it: then --preemptible-ok is wrong, and only waiting for it to end helps (run_wide_live.sh checks
`dibs status` for the hold first).
"""
import argparse
import glob
import os
import sys
import time


def users(ignore=()):
    """Other processes with the GPU open: ROCm through /dev/kfd, Vulkan through a render node.
    Desktop programs under /usr, and Flatpak apps (under /app, e.g. Discord), also hold the render
    node and are left out (as in manifold-compression's heal_sessions.gpu_users)."""
    found = []
    for pid in filter(str.isdigit, os.listdir('/proc')):
        if int(pid) == os.getpid() or int(pid) in ignore:
            continue
        try:
            fds = [os.readlink(f'/proc/{pid}/fd/{fd}') for fd in os.listdir(f'/proc/{pid}/fd')]
            cmd = open(f'/proc/{pid}/cmdline', 'rb').read().replace(b'\0', b' ').decode(errors='replace')
        except OSError:
            continue
        if '/dev/kfd' in fds or (any(f.startswith('/dev/dri/renderD') for f in fds) and not cmd.startswith(('/usr/', '/app/'))):
            found.append((int(pid), cmd[:120]))
    return found


def preemptible(pid):
    """True when `pid` runs under `dibs run --preemptible`."""
    for _ in range(16):
        try:
            cmd = open(f'/proc/{pid}/cmdline', 'rb').read().split(b'\0')
            pid = int(open(f'/proc/{pid}/stat').read().rsplit(')', 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            return False
        ours = cmd[:cmd.index(b'--')] if b'--' in cmd else cmd
        if b'dibs' in os.path.basename(ours[0]) and b'run' in ours and b'--preemptible' in ours:
            return True
        if pid <= 1:
            return False
    return False


def busy_percent():
    values = []
    for path in glob.glob('/sys/class/drm/card*/device/gpu_busy_percent'):
        try:
            values.append(int(open(path).read()))
        except (OSError, ValueError):
            pass
    return max(values, default=0)


def alone(ignore=(), busy_limit=25, preemptible_ok=False):
    """(True, '') when no other process has the GPU and it is idle, else (False, why).
    With preemptible_ok, jobs that have agreed to be preempted do not count."""
    others = users(ignore)
    if preemptible_ok and others and all(preemptible(pid) for pid, _ in others):
        return True, ''
    if others:
        return False, '; '.join(f'{pid} {cmd}' for pid, cmd in others[:3])
    busy = busy_percent()
    return (True, '') if busy <= busy_limit else (False, f'GPU {busy}% busy with no visible owner')


def wait(checks=3, gap=20, ignore=(), report=None, preemptible_ok=False, until=None):
    """Block until `checks` looks in a row, `gap` seconds apart, find the GPU alone: True.
    False if that has not happened by the time `until` (seconds since the epoch)."""
    good, told = 0, None
    while good < checks:
        ok, why = alone(ignore, preemptible_ok=preemptible_ok)
        good = good + 1 if ok else 0
        if not ok and why != told and report:
            report(f'GPU in use, waiting: {why}')
            told = why
        if good < checks:
            if until is not None and time.time() + gap > until:
                return False
            time.sleep(gap)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--wait', action='store_true')
    ap.add_argument('--checks', type=int, default=3)
    ap.add_argument('--gap', type=int, default=20)
    ap.add_argument('--preemptible-ok', action='store_true', help='jobs under `dibs run --preemptible` do not count')
    args = ap.parse_args()
    def free():
        return 'only preemptible jobs on the GPU' if users() else 'GPU alone'

    if args.wait:
        wait(args.checks, args.gap, report=lambda m: print(time.strftime('%H:%M:%S'), m, flush=True), preemptible_ok=args.preemptible_ok)
        print(time.strftime('%H:%M:%S'), free(), flush=True)
        return 0
    ok, why = alone(preemptible_ok=args.preemptible_ok)
    print(free() if ok else f'GPU in use: {why}')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
