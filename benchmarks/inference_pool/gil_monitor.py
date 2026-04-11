"""
benchmarks/tf_python/gil_monitor.py

GIL health monitor — Python equivalent of startEventLoopMonitor() in stats.ts.

Measures how often a background thread is delayed relative to its scheduled
wakeup interval. Delay = time the GIL was held by another thread (or the OS
scheduler was busy).

Key difference from the JS event loop monitor:
  - Python TF releases the GIL during TF_SessionRun (C extension).
    Concurrent inference threads do NOT block the monitor thread.
  - Python pure-Python code DOES hold the GIL and will show as stall.
  - Result: concurrent TF inference should show near-0% stall,
    confirming that Python TF threads give genuine parallelism.
"""

import threading
import time
from dataclasses import dataclass, field
from typing import List


@dataclass
class GILHealth:
    ticks:          int
    mean_stall_ms:  float
    p99_stall_ms:   float
    max_stall_ms:   float
    duration_ms:    float
    stall_fraction: float   # fraction of ticks with stall > 5ms


class GILMonitor:
    def __init__(self, tick_ms: float = 5.0):
        self._tick_ms   = tick_ms
        self._stalls:   List[float] = []
        self._stop      = threading.Event()
        self._start_t   = time.perf_counter()
        self._thread    = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        last = time.perf_counter()
        while not self._stop.is_set():
            # Sleep for tick_ms — wake up and measure how late we are.
            self._stop.wait(timeout=self._tick_ms / 1000.0)
            now   = time.perf_counter()
            delta = (now - last) * 1000.0          # ms
            stall = max(0.0, delta - self._tick_ms)
            self._stalls.append(stall)
            last = now

    def stop(self) -> GILHealth:
        self._stop.set()
        self._thread.join(timeout=1.0)
        duration_ms = (time.perf_counter() - self._start_t) * 1000.0

        stalls = self._stalls
        if not stalls:
            return GILHealth(0, 0.0, 0.0, 0.0, duration_ms, 0.0)

        sorted_s     = sorted(stalls)
        mean         = sum(sorted_s) / len(sorted_s)
        p99_idx      = max(0, int(len(sorted_s) * 0.99) - 1)
        p99          = sorted_s[p99_idx]
        max_s        = sorted_s[-1]
        stalled      = sum(1 for s in stalls if s > 5.0)
        frac         = stalled / len(stalls)

        return GILHealth(
            ticks          = len(stalls),
            mean_stall_ms  = mean,
            p99_stall_ms   = p99,
            max_stall_ms   = max_s,
            duration_ms    = duration_ms,
            stall_fraction = frac,
        )


def start_gil_monitor(tick_ms: float = 5.0) -> GILMonitor:
    return GILMonitor(tick_ms)


def format_gil_health(label: str, h: GILHealth) -> str:
    if h.ticks == 0:
        return f"  {label} GIL health: ticks=0 — monitor never fired"

    if   h.max_stall_ms < 5:   grade = "✓ excellent"
    elif h.max_stall_ms < 20:  grade = "~ acceptable"
    elif h.max_stall_ms < 100: grade = "⚠ degraded"
    else:                       grade = "✗ blocked"

    return (
        f"  {label} GIL health:"
        f"  ticks={h.ticks}"
        f"  mean={h.mean_stall_ms:.1f}ms"
        f"  p99={h.p99_stall_ms:.1f}ms"
        f"  max={h.max_stall_ms:.1f}ms"
        f"  stalled={h.stall_fraction*100:.0f}%"
        f"  {grade}"
    )