# isidorus-bench

Independent benchmarks for [`@isidorus/cpu`](https://github.com/A-KGeorge/isidorus) against competing Node.js ML runtimes.

Kept in a separate repository so:

- Benchmark dependencies (`@tensorflow/tfjs-node` etc.) don't touch the library's dependency tree
- Results can be updated independently of the library release cycle
- Numbers are reproducible by anyone without cloning the main repo

---

## Benchmarks

### `conv2d` — Inference latency & throughput

Compares `@isidorus/cpu` vs `@tensorflow/tfjs-node` on a Conv2D stack:

```
Conv2D(32, 3×3, relu, SAME)
Conv2D(64, 3×3, relu, SAME)
Conv2D(64, 3×3, relu, VALID, stride=2)
Flatten
Dense(128, relu)
Dense(10,  softmax)
```

Input: `[batch, 56, 56, 3]` — compute-bound on Conv2D, completes in seconds on a laptop.

**Run:**

```bash
npm run bench:conv2d
```

**Quick run (50 iters):**

```bash
npm run bench:conv2d:quick
```

**Metrics reported:**

- mean / p50 / p95 / p99 latency (ms) per batch size
- throughput (images/sec) per batch size
- speedup table: `@isidorus/cpu` vs `@tensorflow/tfjs-node`

---

### `inference_pool` — Concurrent throughput

Measures `InferencePool` worker-pool throughput across concurrency levels (1, 2, 4, N workers where N = available CPU threads).

Requires a frozen graph `.pb` file:

```bash
node --import tsx benchmarks/inference_pool/run.ts path/to/model.pb
```

---

## Setup

```bash
# Install @isidorus/cpu + tfjs-node baseline
npm install

# Run Conv2D benchmark (requires @tensorflow/tfjs-node)
npm run bench:conv2d
```

`@tensorflow/tfjs-node` is optional — if not installed, the `@isidorus/cpu` numbers are still reported and the comparison table is skipped.

---

## Results

Committed results live in `results/`:

```
results/
  conv2d/
    2025-01-15T12-30-00-linux-x64.json
    2025-01-15T12-30-00-win32-x64.json
  inference_pool/
    2025-01-15T14-00-00-linux-x64.json
```

Each file is a `BenchmarkSuite` JSON object (see `shared/types.ts`).

---

## Methodology

- **Warmup**: `WARMUP_ITERS` (default 20) runs before timing starts, allowing TF to JIT-compile kernels and populate its thread pool.
- **Synchronous completion**: Both runtimes force synchronous output materialisation before stopping the clock — `@isidorus/cpu` via `await runAsync()`, `@tensorflow/tfjs-node` via `.dataSync()`. This measures compute time, not scheduling latency.
- **Fixed input**: The same pre-allocated input buffer is reused across all iterations to isolate inference from allocation cost.
- **Same model architecture**: Identical layer sequence, filter counts, activations, and strides in both runtimes.
- **Override via env**: `BENCH_ITERS=N WARMUP_ITERS=M npm run bench:conv2d`

---

## Machine info

Results include `machineInfo` with platform, arch, Node version, CPU count, and CPU model for reproducibility.
