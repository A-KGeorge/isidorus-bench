"""
benchmarks/training/run_python.py

TensorFlow Python training benchmark.
Mirrors benchmarks/training/run.ts for direct comparison.

Usage:
  python benchmarks/training/run_python.py
  OPTIMIZER=adam LR=0.001 BENCH_ITERS=100 python benchmarks/training/run_python.py
"""

import json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'

import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

try:
    import tensorflow.compat.v1 as tf
    tf.compat.v1.disable_v2_behavior()
except ImportError:
    print("✗ TensorFlow not installed: pip install tensorflow-cpu")
sys.path.insert(0, str(Path(__file__).parent))

# ── Config ────────────────────────────────────────────────────────────────────

INPUT_H      = 56
INPUT_W      = 56
INPUT_C      = 3
NUM_CLASSES  = 10
WARMUP_STEPS = int(os.environ.get("WARMUP_ITERS", "5"))
BENCH_STEPS  = int(os.environ.get("BENCH_ITERS",  "50"))
BATCH_SIZES  = [int(x) for x in os.environ.get("BATCH_SIZES", "1,8,32").split(",")]
OPTIMIZER    = os.environ.get("OPTIMIZER", "adam")
LR           = float(os.environ.get("LR", "0.001"))

MODEL_DESC = (
    "Conv2D(32,3,relu,SAME) → Conv2D(64,3,relu,SAME) → "
    "Conv2D(64,3,relu,VALID,s2) → Flatten → Dense(128,relu) → Dense(10,softmax)"
)

# ── Stats ─────────────────────────────────────────────────────────────────────

def compute_stats(samples):
    a = sorted(samples)
    n = len(a)
    def pct(p): return a[max(0, int(n * p / 100) - 1)]
    return {"mean": sum(a)/n, "p50": pct(50), "p95": pct(95), "p99": pct(99),
            "min": a[0], "max": a[-1]}

def machine_info():
    import multiprocessing
    return {"platform": platform.system(), "arch": platform.machine(),
            "pythonVersion": platform.python_version(),
            "cpus": multiprocessing.cpu_count(), "cpuModel": platform.processor()}

# ── Model builder ─────────────────────────────────────────────────────────────

def build_graph(batch_size: int):
    """Build a training graph for one fixed batch size. Returns (graph, placeholders, train_op, loss_op)."""
    g = tf.Graph()
    with g.as_default():
        # Inputs
        x = tf.placeholder(tf.float32, [batch_size, INPUT_H, INPUT_W, INPUT_C], name="x")
        y = tf.placeholder(tf.int32,   [batch_size], name="y")

        # Forward pass
        h = tf.nn.relu(tf.nn.conv2d(x,
            tf.Variable(tf.truncated_normal([3,3,INPUT_C,32], stddev=0.1)), [1,1,1,1], "SAME") +
            tf.Variable(tf.zeros([32])))
        h = tf.nn.relu(tf.nn.conv2d(h,
            tf.Variable(tf.truncated_normal([3,3,32,64], stddev=0.1)), [1,1,1,1], "SAME") +
            tf.Variable(tf.zeros([64])))
        h = tf.nn.relu(tf.nn.conv2d(h,
            tf.Variable(tf.truncated_normal([3,3,64,64], stddev=0.1)), [1,2,2,1], "VALID") +
            tf.Variable(tf.zeros([64])))

        # After stride-2 conv on 56×56: output is 27×27×64
        flat_dim = 27 * 27 * 64
        h = tf.reshape(h, [batch_size, flat_dim])
        h = tf.nn.relu(tf.matmul(h, tf.Variable(tf.truncated_normal([flat_dim,128], stddev=0.1))) +
                        tf.Variable(tf.zeros([128])))
        logits = tf.matmul(h, tf.Variable(tf.truncated_normal([128, NUM_CLASSES], stddev=0.1))) + \
                  tf.Variable(tf.zeros([NUM_CLASSES]))

        # Loss
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits))

        # Optimizer
        if OPTIMIZER == "adam":
            opt = tf.train.AdamOptimizer(LR)
        else:
            opt = tf.train.GradientDescentOptimizer(LR)

        train_op = opt.minimize(loss)
        init_op  = tf.global_variables_initializer()

    return g, x, y, train_op, loss, init_op


def run_for_batch(batch_size: int) -> dict:
    g, x_ph, y_ph, train_op, loss_op, init_op = build_graph(batch_size)

    config = tf.ConfigProto(
        intra_op_parallelism_threads=0,  # let TF choose
        inter_op_parallelism_threads=0,
    )
    sess = tf.Session(graph=g, config=config)
    sess.run(init_op)

    x_data = np.random.uniform(0, 0.5, [batch_size, INPUT_H, INPUT_W, INPUT_C]).astype(np.float32)
    y_data = np.random.randint(0, NUM_CLASSES, [batch_size]).astype(np.int32)
    feed   = {x_ph: x_data, y_ph: y_data}

    # Warmup
    for _ in range(WARMUP_STEPS):
        sess.run(train_op, feed_dict=feed)

    # Timed
    step_samples = []
    last_loss    = 0.0

    for _ in range(BENCH_STEPS):
        t0 = time.perf_counter()
        _, loss_val = sess.run([train_op, loss_op], feed_dict=feed)
        step_samples.append((time.perf_counter() - t0) * 1000.0)
        last_loss = float(loss_val)

    sess.close()

    stats         = compute_stats(step_samples)
    total_ms      = sum(step_samples)
    steps_per_sec = (BENCH_STEPS * 1000.0) / total_ms
    samps_per_sec = steps_per_sec * batch_size

    return {
        "batchSize":     batch_size,
        "stepStats":     stats,
        "stepsPerSec":   steps_per_sec,
        "samplesPerSec": samps_per_sec,
        "lastLoss":      last_loss,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import multiprocessing
    hw = multiprocessing.cpu_count()

    print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║         TensorFlow Python  Training Benchmark                        ║
╚══════════════════════════════════════════════════════════════════════╝

  Model:      {MODEL_DESC}
  Optimizer:  {OPTIMIZER.upper()}  lr={LR}
  Warmup:     {WARMUP_STEPS} steps     Bench: {BENCH_STEPS} steps
  Batches:    {', '.join(str(b) for b in BATCH_SIZES)}
  Platform:   {platform.system()}-{platform.machine()}  Python: {platform.python_version()}
  HW threads: {hw}
""")

    W = 72
    print("─" * W)
    print(f" tensorflow-python  v{tf.__version__}  — training")
    print("─" * W)
    print(f" {'batch':>8} {'step mean':>11} {'p99':>11} {'steps/s':>10} {'samples/s':>11} {'loss':>8}")
    print(" " + "─" * 70)

    results = []

    for b in BATCH_SIZES:
        print(f"  batch={b}  ", end="", flush=True)
        r = run_for_batch(b)

        print(
            f" {str(b):8s}"
            f" {r['stepStats']['mean']:8.2f} ms"
            f" {r['stepStats']['p99']:8.2f} ms"
            f" {r['stepsPerSec']:10.1f}"
            f" {r['samplesPerSec']:11.1f}"
            f" {r['lastLoss']:8.4f}"
        )
        results.append(r)

    print("─" * W + "\n")

    # Save
    ts       = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    plat     = f"{platform.system().lower()}-{platform.machine().lower()}"
    out_dir  = Path(__file__).parent.parent.parent / "results" / "training"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{ts}-python-{plat}.json"

    result = {
        "runtime":        "tensorflow-python",
        "runtimeVersion": tf.__version__,
        "model":          MODEL_DESC,
        "optimizer":      OPTIMIZER,
        "lr":             LR,
        "warmupSteps":    WARMUP_STEPS,
        "benchSteps":     BENCH_STEPS,
        "batches": [
            {
                "batchSize":     r["batchSize"],
                "latency":       r["stepStats"],
                "stepsPerSec":   r["stepsPerSec"],
                "samplesPerSec": r["samplesPerSec"],
                "lastLoss": r["lastLoss"],
            }
            for r in results
        ],
        "machineInfo": machine_info(),
        "timestamp":   datetime.now(timezone.utc).isoformat(),
    }

    out_path.write_text(json.dumps(result, indent=2))
    print(f"Results saved → results/training/{out_path.name}")


if __name__ == "__main__":
    main()