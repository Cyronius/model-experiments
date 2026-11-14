"""Exit the process before the machine runs out of RAM.

GPU memory on this APU is system RAM shared with everyone's jobs, and the 2026-11-08 first
LoRA load drove MemAvailable to 0 (swap thrash, near freeze). start() runs a daemon thread
that checks /proc/meminfo every half second and hard-exits (status 75 by default) when
MemAvailable falls below the floor, so the kernel never has to pick a victim. Jobs run under
`dibs run --requeue` pass another status: 75 there means "hand the GPU over and rerun me".
"""
import os
import sys
import threading
import time


def available_gb() -> float:
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 2**20
    return float("inf")


def start(floor_gb: float = 12.0, interval: float = 0.5, status: int = 75) -> None:
    def watch():
        while True:
            a = available_gb()
            if a < floor_gb:
                sys.stderr.write(f"\nmemguard: MemAvailable {a:.1f} GB < {floor_gb} GB, exiting\n")
                sys.stderr.flush()
                os._exit(status)
            time.sleep(interval)

    threading.Thread(target=watch, daemon=True, name="memguard").start()
