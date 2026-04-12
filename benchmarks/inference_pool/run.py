"""
benchmarks/tf_python/run.py

TensorFlow Python CPU inference benchmark.

Mirrors benchmarks/inference_pool/run.ts for direct comparison:
  - Same frozen .pb models
  - Same concurrency levels
  - Same metrics: mean/p50/p95/p99 latency, throughput, GIL health
  - Results saved as JSON in the same schema

Key architectural difference vs tfjs-node:
  Python TF releases the GIL during TF_SessionRun (C extension call).
  Concurrent threads therefore achieve genuine parallelism — unlike
  tfjs-node's synchronous predict() which holds the event loop.
  This makes Python TF threads the closest architectural analog to
  isidorus's runAsync() + libuv thread pool.

Usage:
  python benchmarks/tf_python/run.py bench/models/bench_small.pb
  python benchmarks/tf_python/run.py bench/models/bench_medium.pb --profile latency
  python benchmarks/tf_python/run.py bench/models/bench_large.pb --profile throughput

Profiles:
  auto       (default) — picks best intra_op_threads via brief sweep
  latency    — all cores, single concurrent request
  throughput — intra=4, concurrent requests = floor(cores/4)
"""

import argparse
import json
import os
import platform
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

import numpy as np

# ── TF import ─────────────────────────────────────────────────────────────────

try:
    import tensorflow.compat.v1 as tf
    tf.compat.v1.disable_v2_behavior()
except ImportError:
    print("✗ TensorFlow not installed. Install with:")
    print("    pip install tensorflow-cpu")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).parent))
from config import WARMUP_REQUESTS, BENCH_REQUESTS, CONCURRENCY_LEVELS
from gil_monitor import start_gil_monitor, format_gil_health, GILHealth

# ── Stats ─────────────────────────────────────────────────────────────────────

def compute_stats(samples: List[float]) -> dict:
    a = sorted(samples)
    n = len(a)
    def pct(p): return a[max(0, int(n * p / 100) - 1)]
    return {
        "mean": sum(a) / n,
        "p50":  pct(50),
        "p95":  pct(95),
        "p99":  pct(99),
        "min":  a[0],
        "max":  a[-1],
    }

def machine_info() -> dict:
    import multiprocessing
    return {
        "platform":    platform.system(),
        "arch":        platform.machine(),
        "pythonVersion": platform.python_version(),
        "cpus":        multiprocessing.cpu_count(),
        "cpuModel":    platform.processor(),
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
    """Auto-discover input Placeholder and output ops."""
    ops = graph.get_operations()
    inputs  = [op for op in ops if op.type == "Placeholder"]
    # Sink ops: no consumers within the graph
    all_outputs = {inp for op in ops for inp in op.inputs}
    outputs = [op for op in ops
               if op.type not in ("NoOp", "Assign", "AssignVariableOp")
               and not any(t in all_outputs for t in op.outputs)]
    return inputs[0].outputs[0], [outputs[-1].outputs[0]]


def warm_session(sess: tf.Session, feed: dict, fetch):
    """Run WARMUP_REQUESTS inferences to warm oneDNN and TF allocator."""
    for _ in range(3):
        sess.run(fetch, feed_dict=feed)

# ── Benchmark ─────────────────────────────────────────────────────────────────

def run_concurrent(
    sess: tf.Session,
    feed: dict,
    fetch,
    concurrency: int,
    n_requests: int,
    max_workers: int,
) -> List[float]:
    """
    Pipelined concurrent inference: keep `concurrency` requests in-flight
    at all times. Returns per-request wall-clock samples (t_dispatch → t_done).

    Python TF releases the GIL during sess.run() so threads run in genuine
    parallel on the C++ side — no artificial serialisation.
    """
    samples: List[float] = []
    completed = 0
    issued    = 0
    futures   = {}

    with ThreadPoolExecutor(max_workers=max_workers) as ex:

        def submit_one():
            nonlocal issued
            t0 = time.perf_counter()
            fut = ex.submit(sess.run, fetch, feed)
            futures[fut] = t0
            issued += 1

        # Seed
        while issued < min(concurrency, n_requests):
            submit_one()

        while completed < n_requests:
            done = next(as_completed(futures))
            t0   = futures.pop(done)
            done.result()  # propagate exceptions
            samples.append((time.perf_counter() - t0) * 1000.0)
            completed += 1
            if issued < n_requests and len(futures) < concurrency:
                submit_one()

    return samples


# ── Autotuner ─────────────────────────────────────────────────────────────────

AUTOTUNE_WARMUP = 3
AUTOTUNE_ITERS  = 10
AUTOTUNE_LATENCY_MARGIN = 0.08
AUTOTUNE_INTRA_CANDIDATES = [1, 2, 4, 8, 16, 32]


def autotune(
    graph: tf.Graph,
    feed_template: np.ndarray,
    input_tensor,
    fetch,
    usable_cores: int,
    uv_cap: int,
) -> tuple:
    """Returns (best_intra, best_maxconcurrent, best_sess)."""
    candidates = sorted(set(
        [c for c in AUTOTUNE_INTRA_CANDIDATES if c <= usable_cores]
        + [usable_cores]
    ))

    print(f"  Autotuning {len(candidates)} configs ({AUTOTUNE_ITERS} pipelined rounds each)...")

    results = []

    for intra in candidates:
        conc  = min(max(1, usable_cores // intra), uv_cap)
        inter = max(1, min(4, conc))
        sess  = make_session(graph, intra, inter)
        feed  = {input_tensor: feed_template}

        warm_session(sess, feed, fetch)

        total_reqs = AUTOTUNE_ITERS * max(conc, 1)
        t0 = time.perf_counter()
        run_concurrent(sess, feed, fetch, conc, total_reqs, max_workers=conc)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        rps     = (total_reqs * 1000.0) / elapsed_ms
        mean_ms = elapsed_ms / total_reqs

        print(f"  intra={intra:2d} maxConc={conc} inter={inter} "
              f"rps={rps:.1f} mean={mean_ms:.2f}ms")

        results.append((rps, mean_ms, intra, conc, sess))

    top_rps    = max(r[0] for r in results)
    threshold  = top_rps * (1 - AUTOTUNE_LATENCY_MARGIN)
    contenders = [r for r in results if r[0] >= threshold]
    # Tiebreaker: prefer higher intra (lower per-request latency)
    winner     = max(contenders, key=lambda r: r[2])

    for r in results:
        if r[2] != winner[2]:
            r[4].close()

    best_rps, best_mean, best_intra, best_conc, best_sess = winner
    print(f"  Autotuned: intra={best_intra} maxConcurrent={best_conc} "
          f"[rps={best_rps:.1f} mean={best_mean:.1f}ms]")
    return best_intra, best_conc, best_sess


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TF Python CPU inference benchmark")
    parser.add_argument("model_path", help="Path to frozen .pb model file")
    parser.add_argument("--profile", choices=["auto", "latency", "throughput"],
                        default="auto")
    args = parser.parse_args()

    model_path = args.model_path
    profile    = args.profile

    if not Path(model_path).exists():
        print(f"✗ Model not found: {model_path}")
        sys.exit(1)

    import multiprocessing
    hw_threads  = multiprocessing.cpu_count()
    usable      = hw_threads  # no reserveCores concept in Python benchmark
    # Mirror JS: cap at UV_THREADPOOL_SIZE equivalent = hw_threads for Python
    pool_cap    = usable

    model_name = Path(model_path).stem

    print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║         TensorFlow Python CPU  Inference Benchmark                   ║
╚══════════════════════════════════════════════════════════════════════╝

  Model:       {Path(model_path).name}
  Profile:     {profile}
  Concurrency: {', '.join(str(c) for c in CONCURRENCY_LEVELS)} concurrent callers
  Requests:    {WARMUP_REQUESTS} warmup + {BENCH_REQUESTS} timed
  Platform:    {platform.system()}-{platform.machine()}  Python: {platform.python_version()}
  HW threads:  {hw_threads}
""")

    # ── Load graph ────────────────────────────────────────────────────────────
    cold_start_t0 = time.perf_counter()
    print("  Loading graph...")
    graph = load_frozen_graph(model_path)

    # Auto-discover I/O
    input_tensor, fetch = discover_io(graph)
    raw_shape = input_tensor.shape.as_list()
    input_shape = [d if d is not None else 1 for d in raw_shape]
    print(f"  Input: {input_tensor.name}  shape={raw_shape} → {input_shape}")

    n_elems = 1
    for d in input_shape:
        n_elems *= d
    dummy_input = np.zeros(input_shape, dtype=np.float32)

    # ── Thread/concurrency config ──────────────────────────────────────────────
    if profile == "latency":
        intra, maxconcurrent = usable, 1
        inter = 1
        sess  = make_session(graph, intra, inter)
        warm_session(sess, {input_tensor: dummy_input}, fetch)
        print(f"  profile=latency — intra={intra} maxConcurrent=1")

    elif profile == "throughput":
        intra = min(4, usable)
        maxconcurrent = min(max(1, usable // intra), pool_cap)
        inter = max(1, min(4, maxconcurrent))
        sess  = make_session(graph, intra, inter)
        warm_session(sess, {input_tensor: dummy_input}, fetch)
        print(f"  profile=throughput — intra={intra} maxConcurrent={maxconcurrent}")

    else:
        intra, maxconcurrent, sess = autotune(
            graph, dummy_input, input_tensor, fetch, usable, pool_cap)
        inter = max(1, min(4, maxconcurrent))

    cold_start_ms = (time.perf_counter() - cold_start_t0) * 1000.0
    print(f"  Cold start: {cold_start_ms:.0f}ms")
    print(f"  Inference input: {input_shape}  ({n_elems} floats, {n_elems*4} bytes)")
    print()

    feed = {input_tensor: dummy_input}

    # One warmup pass at max concurrency
    run_concurrent(sess, feed, fetch, maxconcurrent,
                   WARMUP_REQUESTS, max_workers=maxconcurrent)

    # ── Benchmark loop ────────────────────────────────────────────────────────
    W = 72
    print("─" * W)
    print(" TensorFlow Python CPU  — latency & throughput by concurrency")
    print("─" * W)
    print(f" {'workers':8s} {'mean':>11s} {'p50':>11s} {'p95':>11s} "
          f"{'p99':>11s} {'req/s':>10s}")
    print(" " + "─" * 70)

    suite_results = []

    for concurrency in CONCURRENCY_LEVELS:
        monitor = start_gil_monitor(tick_ms=5.0)

        wall_start = time.perf_counter()
        samples = run_concurrent(
            sess, feed, fetch,
            concurrency=concurrency,
            n_requests=BENCH_REQUESTS,
            max_workers=maxconcurrent,
        )
        wall_ms = (time.perf_counter() - wall_start) * 1000.0
        health  = monitor.stop()

        stats   = compute_stats(samples)
        req_per_s = (BENCH_REQUESTS * 1000.0) / wall_ms

        print(f" {str(concurrency):8s}"
              f" {stats['mean']:8.2f} ms"
              f" {stats['p50']:8.2f} ms"
              f" {stats['p95']:8.2f} ms"
              f" {stats['p99']:8.2f} ms"
              f" {req_per_s:7.0f} req/s")
        print(format_gil_health(f"  c={concurrency}", health))

        suite_results.append({
            "concurrency": concurrency,
            "latency":     stats,
            "throughput":  req_per_s,
            "gilHealth": {
                "ticks":         health.ticks,
                "meanStallMs":   health.mean_stall_ms,
                "p99StallMs":    health.p99_stall_ms,
                "maxStallMs":    health.max_stall_ms,
                "durationMs":    health.duration_ms,
                "stallFraction": health.stall_fraction,
            },
        })

    print("─" * W + "\n")
    sess.close()

    # ── Save results ──────────────────────────────────────────────────────────
    import datetime
    from datetime import timezone
    plat        = f"{platform.system().lower()}-{platform.machine().lower()}"
    results_dir = Path(__file__).parent.parent.parent / "results" / "tf_python" / model_name
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Clean up old Python-generated benchmark files (keep TypeScript ones)
    try:
        for file in results_dir.glob("*.json"):
            file.unlink()
    except:
        pass  # Directory might not exist yet
    
    out_path    = results_dir / "inference_pool.json"

    result = {
        "runtime":        "tensorflow-python (concurrent threads)",
        "runtimeVersion": tf.__version__,
        "model":          Path(model_path).name,
        "profile":        profile,
        "config": {
            "intraOpThreads": intra,
            "interOpThreads": inter,
            "maxConcurrent":  maxconcurrent,
        },
        "inputShape":     input_shape,
        "warmupIters":    WARMUP_REQUESTS,
        "benchIters":     BENCH_REQUESTS,
        "batches": [
            {
                "batchSize":  r["concurrency"],
                "latency":    r["latency"],
                "throughput": r["throughput"],
                "gilHealth":  r["gilHealth"],
            }
            for r in suite_results
        ],
        "machineInfo":   machine_info(),
        "timestamp":     datetime.datetime.now(timezone.utc).isoformat(),
        "coldStartMs":   cold_start_ms,
    }

    out_path.write_text(json.dumps(result, indent=2))
    print(f"Results saved → results/tf_python/{model_name}/{out_path.name}")


if __name__ == "__main__":
    main()