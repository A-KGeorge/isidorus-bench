"""
benchmarks/tf_python/config.py

Shared configuration for the TensorFlow Python CPU benchmark.
Mirrors benchmarks/inference_pool/run.ts so results are directly comparable.
"""

import os

WARMUP_REQUESTS  = int(os.environ.get("WARMUP_ITERS", "20"))
BENCH_REQUESTS   = int(os.environ.get("BENCH_ITERS",  "200"))
CONCURRENCY_LEVELS = [1, 2, 3, 4, 5, 6, 7, 8]