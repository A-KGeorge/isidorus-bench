# Benchmark Results & Visualization

This document explains the benchmark results, how they're visualized, and what the metrics mean.

## Overview

Three runtimes are compared across multiple workloads:

- **`@isidorus/cpu`** — Worker-pool based inference (blue)
- **`@tensorflow/tfjs-node`** — TFJS binding to TensorFlow C++ (orange)
- **`tensorflow-python`** — Native Python TensorFlow via threading (teal)

---

## Benchmark Categories

### 1. Conv2D — Single-threaded Inference Latency

**Model:** 6-layer CNN (Conv2D + Dense layers)  
**Input:** `[batch, 56, 56, 3]` images

**Charts:**

- `conv2d_throughput.png` — Images per second across batch sizes

**Metrics:**

- Throughput (images/sec)
- Latency (mean, p50, p95, p99 milliseconds)

**Insight:** Tests peak single-threaded compute performance. `@isidorus/cpu` and `@tensorflow/tfjs-node` are compared on the same synchronous interface.

---

### 2. Inference Pool — Concurrent Worker Throughput

**Models:** `bench_small`, `bench_medium`, `bench_large` — Pre-trained graphs

**Three profiles per model:**

#### `auto` Profile

Optimized for **maximum throughput** under concurrent load.

- Worker pool threads scale to CPU count
- Event loop responsiveness tracked as concurrency increases

#### `latency` Profile

Optimized for **low-latency single requests**.

- More conservative thread pool sizing
- Minimizes tail latency under bursty load

#### `throughput` Profile

Explicit **high-concurrency tuning**.

- Aggressive parallelism
- Measures sustainable rate under steady-state load

**Charts per profile (9 combos: 3 models × 3 profiles):**

1. **`inference_bench_{model}_{profile}_throughput.png`**
   - Requests/second vs. concurrency level
   - Shows how each runtime scales with worker count

2. **`inference_bench_{model}_{profile}_latency_detailed.png`**
   - Mean, p50, p95, p99 latency (ms)
   - Log scale — shows tail behavior
   - Dashed lines = percentiles; solid = mean

3. **`inference_bench_{model}_{profile}_blocked_percentage.png`**
   - **Blocking Ratio (β)** for each runtime
   - Cumulative stall time as % of total duration
   - **Formula:** `β = (meanStallMs × ticks) / durationMs × 100`

---

### 3. Memory Footprint

**Chart:** `memory_consolidated_comparison.png`

**Models:** `bench_small`, `bench_medium`, `bench_large`

**Metrics:**

- **Heap Used** (MB) — Active JavaScript heap
- **External / Buffers** (MB) — Tensor buffers, native allocations
- **Total RSS** (MB) — Resident set size (process footprint)

**Interpretation:**

- Stacked bars show peak memory during inference
- Higher External = more tensor data in memory
- RSS > Heap indicates native code (TensorFlow C++) using memory

---

### 4. Training — Iterative Gradient Descent

**Model:** Same CNN architecture as Conv2D, trained on synthetic data

**Chart:** `training_throughput.png`

**Metrics:**

- Training iterations per second
- Batch size variations (1–8)

**Insight:** Tests sustained compute + memory allocation patterns. Training includes forward pass, backward pass, and weight updates.

---

## Understanding the Blocking Ratio (β)

The **Blocking Ratio** is a fair metric for comparing **unresponsiveness** across different runtimes:

$$\beta = \frac{\text{meanStallMs} \times \text{ticks}}{\text{durationMs}} \times 100$$

**What it measures:**

- **Node.js (Event Loop):** Time the event loop is occupied with synchronous work, preventing other tasks from running
- **Python (GIL):** Time background threads are blocked waiting for the Global Interpreter Lock

**Why it's fair:**

- Normalizes across different timer resolutions and tick rates
- Accounts for total stall accumulation, not just individual spike magnitude
- β = 0% means fully responsive; β = 100% means completely blocked

**Interpretation:**

- `β < 10%` — Excellent responsiveness
- `β < 50%` — Acceptable for batch workloads
- `β > 50%` — Significant contention; single-threaded bottleneck evident

---

## Key Insights from Results

### Inference Pool Scaling

At **concurrency = 1** (single concurrent request):

- All runtimes have low latency and zero blocking

At **concurrency = N** (saturated):

- `@isidorus/cpu` maintains low β (true parallelism)
- `@tensorflow/tfjs-node` β → 100% (event loop saturation)
- `tensorflow-python` β varies by GIL configuration

### Model Size Impact

Larger models (`bench_large`):

- Higher absolute latency due to more computation
- Blocking ratio tends to increase with throughput demand
- Memory footprint clearly differentiates tensor buffer sizes

### Profile Differences

- **`auto`** — Balanced; good for mixed workloads
- **`latency`** — Lower β at lower concurrency, degradation at high concurrency
- **`throughput`** — Sustained high β at high concurrency, maximizes items/sec

---

## Viewing Charts

Generate or regenerate charts with:

```bash
npm run plot
```

Charts are saved to `charts/` as PNG files (1000×600px).

To open in your default image viewer:

```bash
# macOS
open charts/inference_bench_large_auto_throughput.png

# Linux
xdg-open charts/inference_bench_large_auto_throughput.png

# Windows
start charts\inference_bench_large_auto_throughput.png
```

---

## Raw Results

Raw JSON results live in `results/`:

- `results/conv2d/conv2d.json`
- `results/inference_pool/bench_{small,medium,large}/inference_pool.json`
- `results/memory/bench_{small,medium,large}/memory.json`
- `results/training/training.json`

Each file contains:

- Full result object for each runtime
- Per-batch statistics (latency percentiles, throughput)
- Event loop health (ticks, stalls, duration)
- Machine info (CPU, OS, node version)

---

## Reproducing Results

All benchmarks are deterministic given the same machine and node version:

```bash
# Full suite
npm run bench

# Individual benchmark
npm run bench:conv2d
node --import tsx benchmarks/inference_pool/run.ts ./bench/models/bench_large.pb
```

Results are timestamped but not automatically cleaned up. Manually inspect `results/` to manage old runs.

---

## Caveats

- **Single machine results** — Performance varies by CPU, cache size, thermal throttling
- **Synchronous interface** — Inference is measured from start to full completion; scheduling latency not included
- **Frozen graphs** — `@isidorus/cpu` uses `.pb` files; `@tensorflow/tfjs-node` transpiles from SavedModel
- **Python threading** — GIL contention depends on thread pool configuration; results reflect the benchmark setup, not all possible Python TF deployments

---

---

## Detailed Interpretation Guide

### Reading the Latency Chart

The `latency_detailed.png` charts show **4 metrics per runtime**:

- **Solid line (mean)** — Average latency across all requests
- **Dashed line (p50)** — Median: 50% of requests complete by this time
- **Longer dashes (p95)** — Tail latency: only 5% of requests are slower
- **Dotted line (p99)** — 99th percentile: worst-case tail behavior

**What to look for:**

- **Flat mean + high p99** → Consistent performance with occasional outliers
- **Rising mean + rising p99** → Throughput ceiling reached; queuing begins
- **Divergence between mean and p50** → Bimodal distribution (some fast, some slow batches)

### Reading the Throughput Chart

Shows **requests per second** as concurrency increases:

- **Flat line** → Runtime maxes out; adding workers doesn't help
- **Rising line** → Good scaling; more workers = more throughput
- **Peak then decline** → Over-subscription; thread pool overhead > benefits

**Example interpretation:**

- `@isidorus/cpu` maintains slope → true parallelism
- `@tensorflow/tfjs-node` flattens → single-threaded event loop limit
- `tensorflow-python` depends on GIL config → can improve with careful tuning

### Interpreting Blocking Ratio Trends

**Healthy pattern (low β across concurrency):**

```
C=1  C=2  C=3  C=4  C=5  C=6  C=7  C=8
0%   1%   2%   3%   3%   4%   4%   5%    ← @isidorus/cpu (non-blocking parallelism)
```

**Event loop saturation (β climbs to 100%):**

```
C=1  C=2  C=3  C=4  C=5  C=6  C=7  C=8
1%   27%  98%  100% 100% 97%  100% 100% ← @tensorflow/tfjs-node (single-threaded)
```

**GIL contention (β stays low due to lock release during C code):**

```
C=1  C=2  C=3  C=4  C=5  C=6  C=7  C=8
0%   0%   0%   0%   0%   0%   0%   0%    ← tensorflow-python (GIL released during TF_SessionRun)
```

---

## Real-World Use Cases

### Use `@isidorus/cpu` when:

- ✅ You need **sustained concurrent inference** (API server, batch processor)
- ✅ **Event loop responsiveness** matters (rest of app can't block)
- ✅ You control the **model format** (can convert to `.pb`)
- ✅ You want **predictable latency** under load

### Use `@tensorflow/tfjs-node` when:

- ✅ You need **maximum peak performance** on a single request
- ✅ **Concurrency is low** or not a concern (batch job, offline processing)
- ✅ You already have **JavaScript-based model conversion pipeline**
- ⚠️ Avoid if event loop responsiveness is critical

### Use `tensorflow-python` when:

- ✅ You're in a **Python environment** (Flask, FastAPI, Django)
- ✅ You need **NumPy array integration**
- ✅ You want **native TensorFlow Python API** (not a transpiled binding)
- ⚠️ GIL contention may limit scaling in pure-Python code

---

## Memory Footprint Explanation

The `memory_consolidated_comparison.png` stacked bar chart breaks down memory into three categories:

### **Heap Used (Blue)**

- Active JavaScript objects, arrays, strings
- Grows with batch size (larger input buffers)
- Should stay relatively constant across inference runs

### **External / Buffers (Gold)**

- Tensor data (weights, activations, inputs/outputs)
- Largest component for heavy models
- Proportional to model size + batch size

### **Process RSS (Gray, 30% opacity)**

- Total resident memory in use by the process
- Includes OS kernel memory, shared libraries
- RSS > (Heap + External) indicates native TensorFlow memory

**Example reading:**

```
bench_large:
  Heap:     ~50 MB   (JavaScript state)
  External: ~400 MB  (Model weights + inference buffers)
  RSS:      ~600 MB  (Total process footprint)
  → ~150 MB unaccounted for = TensorFlow C++ runtime state
```

---

## Performance Profiles & When to Use

### `auto` — Balanced

- **Best for:** General purpose, mixed workload patterns
- **Thread config:** Scales to CPU count
- **Expected β:** Moderate (0–50% depending on batch size)

**When results flatten:** You've hit the sweet spot for your hardware

### `latency` — Low-tail-latency

- **Best for:** Interactive applications, real-time systems
- **Thread config:** Conservative; few workers to reduce scheduling overhead
- **Expected β:** Very low at C=1–2, rises sharply at C > 4

**When results diverge:** Trade-off between low p99 and low throughput

### `throughput` — Maximum sustained rate

- **Best for:** Batch processing, API endpoints expecting high load
- **Thread config:** Aggressive; fully saturate CPU
- **Expected β:** Higher (50–100% is acceptable here)

**When results degrade:** Over-subscription; reduce batch size or worker count

---

## Known Limitations

### 1. **Model Freezing (isidorus only)**

- `@isidorus/cpu` requires `.pb` (frozen graph) format
- `@tensorflow/tfjs-node` can load SavedModel or `.h5`
- **Impact:** Model conversion overhead not measured; assumes pre-frozen model

### 2. **GIL Configuration Sensitivity (Python)**

- Results reflect `intraOpThreads` and `interOpThreads` settings
- Tuning these values can **dramatically** change concurrency scalability
- **Impact:** Python results may not generalize to all configurations

### 3. **Event Loop Biasing (Node.js)**

- `setInterval` monitor runs at **5ms resolution**
- Shorter intervals (1ms) show more detailed stalls; longer (10ms) smooth noise
- **Impact:** Absolute β values are resolution-dependent; comparisons are fair

### 4. **Single-Machine Variability**

- CPU frequency scaling (turbo boost on/off)
- Thermal throttling under sustained load
- Background OS processes
- **Impact:** Run multiple times; expect ±5–10% variance

### 5. **Synchronous Measurement**

- Inference time = start of request to completion of output
- Does **not** include scheduling latency or task queue wait time
- **Impact:** Results are "pure compute time," not end-to-end API latency

---

## Extending the Benchmarks

### Add a New Runtime

1. Implement a new file in `benchmarks/inference_pool/{runtime_name}.ts`
2. Export a `runBenchmark(...)` function matching [shared/types.ts](shared/types.ts#L25)
3. Add to `results[]` array in [benchmarks/inference_pool/run.ts](benchmarks/inference_pool/run.ts#L85)
4. Update `RUNTIME_MAP` in [scripts/plot_results.ts](scripts/plot_results.ts#L10) to name it

### Add a New Metric

1. Compute metric in the benchmark runner (e.g., in `eventLoop` or `gilHealth` object)
2. Update [shared/types.ts](shared/types.ts) to include new field
3. Add dataset to `plotInferenceComparison()` in [scripts/plot_results.ts](scripts/plot_results.ts#L125)

---

## FAQ

**Q: Why does β exceed 100% sometimes?**
A: Due to rounding and the 5ms tick resolution, if stalls cluster densely, `(mean × ticks) / duration` can slightly exceed 100%. This is a measurement artifact, not actual time travel. Treat β > 100% as "completely blocked."

**Q: Can I compare Python and Node.js results directly?**
A: Yes, if both are measuring with the same model. The Blocking Ratio (β) is designed to be runtime-agnostic. However, Python threading != Node.js event loop; different concurrency models mean the same β% has different implications.

**Q: My results don't match the numbers in these charts. Why?**
A: Hardware, node/python version, background processes, and thermal state all affect results. Re-run your own benchmarks on your target hardware for accurate numbers. These results are from a specific machine (see `machineInfo` in JSON files).

**Q: How do I know if my runtime is "good"?**
A: Context-dependent:

- **Interactive API (p99 matters):** p99 < 100ms and β < 50% at expected concurrency
- **Batch processing (throughput matters):** Sustained linear scaling up to CPU count
- **Mixed workload:** Low β at low concurrency; acceptable β at high concurrency

---

## See Also

- [README.md](README.md) — Benchmark methodology and setup
- [shared/stats.ts](shared/stats.ts) — Event loop monitor implementation
- [benchmarks/inference_pool/gil_monitor.py](benchmarks/inference_pool/gil_monitor.py) — GIL monitor (Python equivalent)
- [shared/types.ts](shared/types.ts) — TypeScript interfaces for all data structures
