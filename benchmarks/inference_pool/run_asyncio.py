"""
benchmarks/inference_pool/run_asyncio.py

TensorFlow Python CPU inference benchmark — asyncio edition.

Mirrors run.py but dispatches inference via asyncio + ThreadPoolExecutor
rather than raw ThreadPoolExecutor. This is the Python equivalent of what
isidorus does for Node.js: the event loop stays free because blocking work
runs in a thread pool, with the event loop only handling coordination.

Key difference from run.py:
  - run.py:       raw threads, GIL monitor on a background thread
  - run_asyncio:  asyncio event loop + run_in_executor, asyncio event loop
                  health monitor (measures asyncio coroutine scheduling delay)

Since TF releases the GIL during TF_SessionRun, asyncio + run_in_executor
gives genuine parallelism AND keeps the asyncio event loop responsive — the
same two properties that isidorus targets for Node.js.

Usage:
  python benchmarks/inference_pool/run_asyncio.py bench/models/bert_model.pb
  python benchmarks/inference_pool/run_asyncio.py bench/models/bench_small.pb --profile latency
"""

import argparse
import asyncio
import concurrent.futures
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, List, Optional

import numpy as np

try:
    import tensorflow.compat.v1 as tf
    tf.compat.v1.disable_v2_behavior()
except ImportError:
    print("✗ TensorFlow not installed: pip install tensorflow-cpu")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).parent))
from config import WARMUP_REQUESTS, BENCH_REQUESTS, CONCURRENCY_LEVELS

# ── Asyncio event loop health monitor ─────────────────────────────────────────
#
# Schedules a periodic coroutine (asyncio.sleep) and measures how late each
# wake-up is relative to its scheduled time. Unlike the GIL monitor in
# gil_monitor.py (which uses a background OS thread), this runs entirely
# within the asyncio event loop — so it directly measures asyncio event loop
# responsiveness, not GIL contention.
#
# If inference is dispatched via run_in_executor, the event loop remains free
# to schedule these ticks on time → stall ≈ 0. If inference were called
# directly (blocking the loop), stall ≈ inference duration per tick.


class AsyncIOEventLoopMonitor:
    def __init__(self, tick_ms: float = 5.0):
        self._tick_ms = tick_ms
        self._stalls: List[float] = []
        self._running = True
        self._start = time.perf_counter()
        self._task: Optional[asyncio.Task] = None

    async def _run(self) -> None:
        tick_s = self._tick_ms / 1000.0
        loop = asyncio.get_event_loop()
        expected = loop.time() + tick_s
        while self._running:
            await asyncio.sleep(tick_s)
            now = loop.time()
            # How late did we fire vs when we were scheduled?
            stall = max(0.0, (now - expected) * 1000.0)
            self._stalls.append(stall)
            expected = now + tick_s

    def start(self) -> "AsyncIOEventLoopMonitor":
        self._task = asyncio.ensure_future(self._run())
        return self

    async def stop(self) -> dict:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        duration_ms = (time.perf_counter() - self._start) * 1000.0

        stalls = self._stalls
        if not stalls:
            return {
                "ticks": 0, "meanStallMs": 0.0, "p99StallMs": 0.0,
                "maxStallMs": 0.0, "durationMs": duration_ms, "stallFraction": 0.0,
            }

        sorted_s = sorted(stalls)
        mean = sum(sorted_s) / len(sorted_s)
        p99_idx = max(0, int(len(sorted_s) * 0.99) - 1)
        p99 = sorted_s[p99_idx]
        stalled = sum(1 for s in stalls if s > 5.0)

        return {
            "ticks": len(stalls),
            "meanStallMs": mean,
            "p99StallMs": p99,
            "maxStallMs": sorted_s[-1],
            "durationMs": duration_ms,
            "stallFraction": stalled / len(stalls),
        }


def format_event_loop_health(label: str, h: dict) -> str:
    if h["ticks"] == 0:
        return f"  {label} asyncio health: ticks=0 — monitor never fired"

    max_ms = h["maxStallMs"]
    if max_ms < 5:
        grade = "✓ excellent"
    elif max_ms < 20:
        grade = "~ acceptable"
    elif max_ms < 100:
        grade = "⚠ degraded"
    else:
        grade = "✗ blocked"

    return (
        f"  {label} asyncio event loop:"
        f"  ticks={h['ticks']}"
        f"  mean={h['meanStallMs']:.1f}ms"
        f"  p99={h['p99StallMs']:.1f}ms"
        f"  max={h['maxStallMs']:.1f}ms"
        f"  stalled={h['stallFraction']*100:.0f}%"
        f"  {grade}"
    )


# ── Stats ─────────────────────────────────────────────────────────────────────

def compute_stats(samples: List[float]) -> dict:
    a = sorted(samples)
    n = len(a)
    def pct(p): return a[max(0, int(n * p / 100) - 1)]
    return {
        "mean": sum(a) / n,
        "p50": pct(50),
        "p95": pct(95),
        "p99": pct(99),
        "min": a[0],
        "max": a[-1],
    }


def machine_info() -> dict:
    import multiprocessing
    return {
        "platform": platform.system(),
        "arch": platform.machine(),
        "pythonVersion": platform.python_version(),
        "cpus": multiprocessing.cpu_count(),
        "cpuModel": platform.processor(),
    }


# ── Model loading ─────────────────────────────────────────────────────────────

def load_frozen_graph(model_path: str) -> tf.Graph:
    graph = tf.Graph()
    with graph.as_default():
        graph_def = tf.GraphDef()
        with open(model_path, "rb") as f:
            graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name="")
    return graph


def make_session(graph: tf.Graph, intra: int, inter: int) -> tf.Session:
    config = tf.ConfigProto(
        intra_op_parallelism_threads=intra,
        inter_op_parallelism_threads=inter,
    )
    return tf.Session(graph=graph, config=config)


def discover_io(graph: tf.Graph):
    ops = graph.get_operations()
    inputs = [op for op in ops if op.type == "Placeholder"]
    all_outputs = {inp for op in ops for inp in op.inputs}
    outputs = [op for op in ops
               if op.type not in ("NoOp", "Assign", "AssignVariableOp")
               and not any(t in all_outputs for t in op.outputs)]
    input_tensors = sorted(
        [op.outputs[0] for op in inputs], key=lambda t: t.name
    )
    return input_tensors, [outputs[-1].outputs[0]]


# ── Autotuner (mirrors run.py) ────────────────────────────────────────────────

AUTOTUNE_ITERS = 10
AUTOTUNE_INTRA_CANDIDATES = [1, 2, 4, 8, 16, 32]
AUTOTUNE_LATENCY_MARGIN = 0.08


def autotune_sync(
    graph: tf.Graph,
    feed: dict,
    fetch,
    usable_cores: int,
) -> tuple:
    """Sync autotuner — runs before the asyncio loop starts."""
    import multiprocessing
    candidates = sorted(set(
        [c for c in AUTOTUNE_INTRA_CANDIDATES if c <= usable_cores]
        + [usable_cores]
    ))
    print(f"  Autotuning {len(candidates)} configs...")

    results = []
    for intra in candidates:
        conc = max(1, usable_cores // intra)
        inter = max(1, min(4, conc))
        sess = make_session(graph, intra, inter)

        # Sync warmup
        for _ in range(3):
            sess.run(fetch, feed_dict=feed)

        # Sync timing (don't need asyncio for autotuning)
        with concurrent.futures.ThreadPoolExecutor(max_workers=conc) as ex:
            total = AUTOTUNE_ITERS * max(conc, 1)
            t0 = time.perf_counter()
            futs = [ex.submit(lambda: sess.run(fetch, feed_dict=feed)) for _ in range(total)]
            concurrent.futures.wait(futs)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

        rps = (total * 1000.0) / elapsed_ms
        mean_ms = elapsed_ms / total
        print(f"  intra={intra:2d} maxConc={conc} rps={rps:.1f} mean={mean_ms:.2f}ms")
        results.append((rps, mean_ms, intra, conc, sess))

    top_rps = max(r[0] for r in results)
    threshold = top_rps * (1 - AUTOTUNE_LATENCY_MARGIN)
    winner = max(
        [r for r in results if r[0] >= threshold],
        key=lambda r: r[2],
    )
    for r in results:
        if r[4] is not winner[4]:
            r[4].close()

    best_rps, best_mean, best_intra, best_conc, best_sess = winner
    print(f"  Autotuned: intra={best_intra} maxConcurrent={best_conc} "
          f"[rps={best_rps:.1f} mean={best_mean:.1f}ms]")
    return best_intra, best_conc, best_sess


# ── Asyncio benchmark core ────────────────────────────────────────────────────

async def run_concurrent_async(
    loop: asyncio.AbstractEventLoop,
    executor: concurrent.futures.ThreadPoolExecutor,
    sess: tf.Session,
    feed: dict,
    fetch,
    concurrency: int,
    n_requests: int,
) -> List[float]:
    """
    Pipelined asyncio inference.

    Each request is dispatched via loop.run_in_executor() which:
    1. Runs sess.run() in a ThreadPoolExecutor thread (TF releases GIL)
    2. Returns an awaitable Future that resolves on the asyncio event loop
    3. Keeps the asyncio event loop free to tick the health monitor

    This is structurally identical to Node.js uv_queue_work / runAsync():
    blocking work runs off the event loop; the loop only sees the completion.
    """
    samples: List[float] = []
    semaphore = asyncio.Semaphore(concurrency)
    pending: List[asyncio.Task] = []

    async def one_request() -> None:
        async with semaphore:
            t0 = time.perf_counter()
            await loop.run_in_executor(
                executor,
                lambda: sess.run(fetch, feed_dict=feed),
            )
            samples.append((time.perf_counter() - t0) * 1000.0)

    tasks = [asyncio.ensure_future(one_request()) for _ in range(n_requests)]
    await asyncio.gather(*tasks)
    return samples


# ── Main ──────────────────────────────────────────────────────────────────────

async def async_main(args: argparse.Namespace) -> None:
    model_path = args.model_path
    profile = args.profile

    import multiprocessing
    hw_threads = multiprocessing.cpu_count()
    usable = hw_threads

    print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║      TensorFlow Python (asyncio)  Inference Benchmark               ║
╚══════════════════════════════════════════════════════════════════════╝

  Model:       {Path(model_path).name}
  Profile:     {profile}
  Concurrency: {', '.join(str(c) for c in CONCURRENCY_LEVELS)} concurrent callers
  Requests:    {WARMUP_REQUESTS} warmup + {BENCH_REQUESTS} timed
  Platform:    {platform.system()}-{platform.machine()}  Python: {platform.python_version()}
  HW threads:  {hw_threads}
""")

    cold_start_t0 = time.perf_counter()
    print("  Loading graph...")
    graph = load_frozen_graph(model_path)

    input_tensors, fetch = discover_io(graph)

    feed_template: dict = {}
    input_shapes: List[list] = []
    for input_tensor in input_tensors:
        raw_shape = input_tensor.shape.as_list()
        input_shape = [d if d is not None else 1 for d in raw_shape]
        input_shapes.append(input_shape)
        dummy = np.zeros(input_shape, dtype=np.float32)
        if "int" in str(input_tensor.dtype):
            dummy = dummy.astype(np.int32)
        feed_template[input_tensor] = dummy
        print(f"  Input: {input_tensor.name}  shape={raw_shape} → {input_shape}  dtype={input_tensor.dtype}")

    loop = asyncio.get_event_loop()

    # Thread/concurrency config — mirrors run.py
    if profile == "latency":
        intra, maxconcurrent = usable, 1
        inter = 1
        sess = make_session(graph, intra, inter)
        print(f"  profile=latency — intra={intra} maxConcurrent=1")
    elif profile == "throughput":
        intra = min(4, usable)
        maxconcurrent = min(max(1, usable // intra), usable)
        inter = max(1, min(4, maxconcurrent))
        sess = make_session(graph, intra, inter)
        print(f"  profile=throughput — intra={intra} maxConcurrent={maxconcurrent}")
    else:
        intra, maxconcurrent, sess = autotune_sync(graph, feed_template, fetch, usable)
        inter = max(1, min(4, maxconcurrent))

    cold_start_ms = (time.perf_counter() - cold_start_t0) * 1000.0
    print(f"  Cold start: {cold_start_ms:.0f}ms\n")

    feed = feed_template

    # The executor is shared across all concurrency levels.
    # max_workers = maxconcurrent so the asyncio semaphore and thread pool
    # are aligned — no request can get a semaphore slot without a thread.
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=maxconcurrent)

    # Warmup
    await run_concurrent_async(loop, executor, sess, feed, fetch, maxconcurrent, WARMUP_REQUESTS)

    W = 72
    print("─" * W)
    print(" TensorFlow Python (asyncio)  — latency & asyncio event loop health")
    print("─" * W)
    print(f" {'workers':8s} {'mean':>11s} {'p50':>11s} {'p95':>11s} "
          f"{'p99':>11s} {'req/s':>10s}")
    print(" " + "─" * 70)

    suite_results = []

    for concurrency in CONCURRENCY_LEVELS:
        monitor = AsyncIOEventLoopMonitor(tick_ms=5.0).start()

        wall_start = time.perf_counter()
        samples = await run_concurrent_async(
            loop, executor, sess, feed, fetch,
            concurrency=concurrency,
            n_requests=BENCH_REQUESTS,
        )
        wall_ms = (time.perf_counter() - wall_start) * 1000.0
        health = await monitor.stop()

        stats = compute_stats(samples)
        req_per_s = (BENCH_REQUESTS * 1000.0) / wall_ms

        print(f" {str(concurrency):8s}"
              f" {stats['mean']:8.2f} ms"
              f" {stats['p50']:8.2f} ms"
              f" {stats['p95']:8.2f} ms"
              f" {stats['p99']:8.2f} ms"
              f" {req_per_s:7.0f} req/s")
        print(format_event_loop_health(f"  c={concurrency}", health))

        suite_results.append({
            "concurrency": concurrency,
            "latency": stats,
            "throughput": req_per_s,
            "eventLoopHealth": health,
        })

    print("─" * W + "\n")
    executor.shutdown(wait=False)
    sess.close()

    # ── Save results ──────────────────────────────────────────────────────────
    result = {
        "runtime": "tensorflow-python (asyncio)",
        "runtimeVersion": tf.__version__,
        "model": Path(model_path).name,
        "profile": profile,
        "config": {
            "intraOpThreads": intra,
            "interOpThreads": inter,
            "maxConcurrent": maxconcurrent,
        },
        "inputShapes": input_shapes,
        "warmupIters": WARMUP_REQUESTS,
        "benchIters": BENCH_REQUESTS,
        "batches": [
            {
                "batchSize": r["concurrency"],
                "latency": r["latency"],
                "throughput": r["throughput"],
                "eventLoopHealth": r["eventLoopHealth"],
            }
            for r in suite_results
        ],
        "machineInfo": machine_info(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "coldStartMs": cold_start_ms,
    }

    model_name = Path(model_path).stem
    results_dir = Path(__file__).parent.parent.parent / "results" / "inference_pool" / model_name
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "inference_pool.json"

    existing_data = None
    if out_path.exists():
        try:
            with open(out_path, "r") as f:
                existing_data = json.load(f)
        except Exception:
            pass

    def get_result_key(res: dict) -> str:
        return f"{res.get('runtime')}::{res.get('profile', 'auto')}"

    if existing_data and isinstance(existing_data, dict) and "results" in existing_data:
        merged = list(existing_data.get("results", []))
        new_key = get_result_key(result)
        idx = next((i for i, r in enumerate(merged) if get_result_key(r) == new_key), None)
        if idx is not None:
            merged[idx] = result
        else:
            merged.append(result)
        existing_data["results"] = merged
        out_path.write_text(json.dumps(existing_data, indent=2))
    else:
        suite = {
            "name": "inference_pool",
            "description": f"Worker-pool throughput benchmark — {Path(model_path).name}",
            "results": [result],
            "comparisons": [],
        }
        out_path.write_text(json.dumps(suite, indent=2))

    print(f"Results saved → results/inference_pool/{model_name}/{out_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TF Python asyncio CPU inference benchmark"
    )
    parser.add_argument("model_path", help="Path to frozen .pb model file")
    parser.add_argument(
        "--profile",
        choices=["auto", "latency", "throughput"],
        default="auto",
    )
    args = parser.parse_args()

    if not Path(args.model_path).exists():
        print(f"✗ Model not found: {args.model_path}")
        sys.exit(1)

    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()