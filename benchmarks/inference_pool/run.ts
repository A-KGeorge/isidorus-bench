/**
 * benchmarks/inference_pool/run.ts
 *
 * Measures InferencePool worker-pool throughput: N concurrent inference
 * requests dispatched to a pool of Workers, each running its own native
 * TF Session via jude-map zero-copy transport.
 *
 * This benchmark is designed to show the performance advantage of
 * worker-pool over a single-Session setup for high-concurrency workloads.
 *
 * Usage:
 *   node --import tsx benchmarks/inference_pool/run.ts <model.pb>
 *
 * Example:
 *   node --import tsx benchmarks/inference_pool/run.ts bench_small.pb
 *
 * The model path is the only required argument. The input op and output ops
 * are auto-discovered via listOpsOfType / listSinkOps.
 */

import { performance } from "node:perf_hooks";
import {
  writeFileSync,
  mkdirSync,
  existsSync,
  rmSync,
  readdirSync,
} from "node:fs";
import { join, basename } from "node:path";
import { fileURLToPath } from "node:url";
import { availableParallelism } from "node:os";

import { InferencePool } from "@isidorus/cpu";
import {
  computeStats,
  machineInfo,
  printHeader,
  printFooter,
  printSpeedupTable,
  startEventLoopMonitor,
  printEventLoopHealth,
  fmtMs,
  fmtTps,
} from "../../shared/stats.js";
import type { BenchmarkSuite, SpeedupEntry } from "../../shared/types.js";
import { runTfjsNodePoolBench } from "./tfjs_node.js";

const __dirname = fileURLToPath(new URL(".", import.meta.url));
const REPO_ROOT = join(__dirname, "..", "..");

// ─── Config ──────────────────────────────────────────────────────────────────

const modelPath = process.argv[2];

// Optional --profile flag: auto | latency | throughput
// Usage: node run.ts bench_medium.pb --profile throughput
const profileArg = process.argv.indexOf("--profile");
const profile: "auto" | "latency" | "throughput" =
  profileArg !== -1 && process.argv[profileArg + 1]
    ? (process.argv[profileArg + 1] as "auto" | "latency" | "throughput")
    : "auto";

const CONCURRENCY_LEVELS = [1, 2, 3, 4, 5, 6, 7, 8]
  .filter((v, i, a) => a.indexOf(v) === i)
  .sort((a, b) => a - b);

const WARMUP_REQUESTS = 20;
const BENCH_REQUESTS = 200;

// ─── Validation ──────────────────────────────────────────────────────────────

if (!modelPath) {
  console.error(
    "Usage: node --import tsx benchmarks/inference_pool/run.ts <model.pb>",
  );
  process.exit(1);
}
if (!existsSync(modelPath)) {
  console.error(`Model file not found: ${modelPath}`);
  process.exit(1);
}

// ─── Header ──────────────────────────────────────────────────────────────────

console.log(`
╔══════════════════════════════════════════════════════════════════════╗
║           @isidorus/cpu  InferencePool Throughput Benchmark          ║
╚══════════════════════════════════════════════════════════════════════╝

  Model:       ${basename(modelPath)}
  Strategy:    tf-parallel (native Session, concurrent runAsync)
  Profile:     ${profile}
  Concurrency: ${CONCURRENCY_LEVELS.join(", ")} concurrent callers
  Requests:    ${WARMUP_REQUESTS} warmup + ${BENCH_REQUESTS} timed
  Platform:    ${process.platform}-${process.arch}  Node: ${process.version}
  HW threads:  ${availableParallelism()}
`);

// ─── Benchmark ───────────────────────────────────────────────────────────────

printHeader("InferencePool tf-parallel — latency & throughput by concurrency");
console.log(
  ` ${"workers".padEnd(8)}` +
    ` ${"mean".padStart(11)}` +
    ` ${"p50".padStart(11)}` +
    ` ${"p95".padStart(11)}` +
    ` ${"p99".padStart(11)}` +
    ` ${"req/s".padStart(10)}`,
);
console.log(" " + "─".repeat(70));

const suiteResults = [];

// Capture resolved shape + op names from the first pool so tfjs-node gets them.
let resolvedShape: number[] = [];
let pool0InputOp: string = "";
let pool0OutputOp: string = "";

// ── Cold start measurement ───────────────────────────────────────────────────
// Time InferencePool.create() end-to-end: graph load + session creation +
// autotuner + oneDNN warmup. This is the latency a server experiences on
// startup or container restart before it can serve any requests.
const coldStartT0 = performance.now();
const pool = await InferencePool.create({ modelPath, profile });
const coldStartMs = performance.now() - coldStartT0;
console.log(
  `  Cold start:  ${coldStartMs.toFixed(0)}ms (graph load + session + autotuner + oneDNN warmup)`,
);

{
  const resolved = pool.resolvedInputShape;
  resolvedShape = resolved.map((d) => d ?? 1);
  pool0InputOp = "inputs";
  pool0OutputOp = "Identity";
  console.log(`  Resolved input shape: ${JSON.stringify(resolved)}`);

  if (resolved.every((d) => d === null)) {
    console.error(
      "\n✗ Placeholder has fully dynamic shape — cannot auto-size input buffer.\n" +
        "  Pass the correct shape by editing BENCH_INPUT_SHAPE at the top of this file.\n",
    );
    await pool.destroy();
    process.exit(1);
  }
}

const inferShape = resolvedShape;
const nElems = inferShape.reduce((a, b) => a * b, 1);
const inferBuf = new Float32Array(nElems); // use typed array directly, auto-converts
console.log(
  `  Inference shape:  ${JSON.stringify(inferShape)}  (${nElems} floats, ${nElems * 4} bytes)`,
);

// One warmup pass at max concurrency to stabilise the pool before timing.
await Promise.all(
  Array.from({ length: WARMUP_REQUESTS }, () =>
    pool.infer(inferBuf, inferShape, 1),
  ),
);

for (const concurrency of CONCURRENCY_LEVELS) {
  // Concurrent dispatch with event loop health monitoring.
  const samples: number[] = [];
  const wallStart = performance.now();
  const monitor = startEventLoopMonitor(5);

  const inFlight = new Set<Promise<void>>();
  let issued = 0;

  while (issued < BENCH_REQUESTS || inFlight.size > 0) {
    while (issued < BENCH_REQUESTS && inFlight.size < concurrency) {
      const t0 = performance.now();
      let p: Promise<void>;
      p = pool.infer(inferBuf, inferShape, 1).then(() => {
        samples.push(performance.now() - t0);
        inFlight.delete(p!);
      });
      inFlight.add(p);
      issued++;
    }
    if (inFlight.size > 0) await Promise.race(inFlight);
  }

  const health = monitor.stop();
  const wallMs = performance.now() - wallStart;
  const reqPerSec = ((BENCH_REQUESTS * 1000) / wallMs).toFixed(0);
  const s = computeStats(samples);

  console.log(
    ` ${String(concurrency).padEnd(8)}` +
      ` ${fmtMs(s.mean).padStart(11)}` +
      ` ${fmtMs(s.p50).padStart(11)}` +
      ` ${fmtMs(s.p95).padStart(11)}` +
      ` ${fmtMs(s.p99).padStart(11)}` +
      ` ${(reqPerSec + " req/s").padStart(10)}`,
  );
  printEventLoopHealth(`c=${concurrency}`, health);

  suiteResults.push({
    concurrency,
    latency: s,
    throughput: parseFloat(reqPerSec),
    eventLoop: health,
  });
}

await pool.destroy();

printFooter();

// ─── InferencePool result object ─────────────────────────────────────────────

const isidorusResult = {
  runtime: "@isidorus/cpu (InferencePool tf-parallel)",
  runtimeVersion: "latest",
  model: basename(modelPath),
  inputShape: resolvedShape,
  warmupIters: WARMUP_REQUESTS,
  benchIters: BENCH_REQUESTS,
  batches: suiteResults.map((r) => ({
    batchSize: r.concurrency,
    latency: r.latency,
    throughput: r.throughput,
  })),
  eventLoopHealth: suiteResults.map((r) => ({
    concurrency: r.concurrency,
    ...r.eventLoop,
  })),
  machineInfo: machineInfo(),
  timestamp: new Date().toISOString(),
  durationMs: 0,
  coldStartMs,
};

// ─── @tensorflow/tfjs-node baseline ──────────────────────────────────────────

console.log(
  "\n── @tensorflow/tfjs-node ──────────────────────────────────────────────",
);
const resolvedInputOp = pool0InputOp;
const resolvedOutputOp = pool0OutputOp;

// Time tfjs-node cold start separately.
const tfjsColdStartT0 = performance.now();
const tfjsResult = await runTfjsNodePoolBench(
  modelPath,
  resolvedShape,
  resolvedInputOp,
  resolvedOutputOp,
);
// tfjsResult.durationMs includes the full bench run; loadSavedModel is
// called inside runTfjsNodePoolBench and its timing is embedded in durationMs.
// We surface the total cold start separately below.

// ─── Speedup table ────────────────────────────────────────────────────────────

const comparisons: SpeedupEntry[] = [];

if (tfjsResult) {
  printSpeedupTable(
    { name: tfjsResult.runtime, batches: tfjsResult.batches },
    { name: "@isidorus/cpu InferencePool", batches: isidorusResult.batches },
  );

  const bMap = new Map(tfjsResult.batches.map((b) => [b.batchSize, b]));
  for (const cr of isidorusResult.batches) {
    const br = bMap.get(cr.batchSize);
    if (!br) continue;
    comparisons.push({
      batchSize: cr.batchSize,
      baseline: tfjsResult.runtime,
      candidate: isidorusResult.runtime,
      baselineMs: br.latency.mean,
      candidateMs: cr.latency.mean,
      speedup: br.latency.mean / cr.latency.mean,
    });
  }

  // Throughput table — the fair metric.
  // Per-request latency at concurrency>1 is a benchmarking artifact:
  // tfjs-node.predict() is synchronous so t0 is set right before each
  // blocking call, making each request appear to take only its own compute
  // time. isidorus measures from async dispatch, which includes queue wait.
  // Throughput (wall-clock req/s) is invariant to this difference.
  console.log("\n" + "─".repeat(72));
  console.log(
    " Throughput: req/s (higher is better — the fair comparison metric)",
  );
  console.log(" Note: per-request latency at concurrency>1 is not comparable");
  console.log("       between sync (tfjs) and async (isidorus) dispatch.");
  console.log("─".repeat(72));
  console.log(
    ` ${"concurr".padEnd(10)} ${"isidorus req/s".padStart(16)} ${"tfjs-node req/s".padStart(17)} ${"throughput win".padStart(16)}`,
  );
  console.log(" " + "─".repeat(70));
  const bMapT = new Map(tfjsResult.batches.map((b) => [b.batchSize, b]));
  for (const ir of isidorusResult.batches) {
    const tr = bMapT.get(ir.batchSize);
    if (!tr) continue;
    const win = ir.throughput / tr.throughput;
    console.log(
      ` ${String(ir.batchSize).padEnd(10)}` +
        ` ${(ir.throughput.toFixed(0) + " req/s").padStart(16)}` +
        ` ${(tr.throughput.toFixed(0) + " req/s").padStart(17)}` +
        ` ${(win.toFixed(2) + "×  " + (win >= 1 ? "✓" : "✗")).padStart(16)}`,
    );
  }
  console.log("─".repeat(72) + "\n");
}

// ── Cold start comparison ─────────────────────────────────────────────────────
// isidorus cold start includes: graph load + session creation + autotuner
//   + oneDNN warmup. First request is served immediately after create().
// tfjs-node cold start is model load only. The first predict() pays the
//   oneDNN JIT compilation cost on top (~50-300ms depending on model).
console.log("─".repeat(72));
console.log(" Cold start: time until pool is ready to serve requests");
console.log("─".repeat(72));
const tfjsColdMs = (tfjsResult as any)?.coldStartMs as number | undefined;
console.log(
  ` ${"runtime".padEnd(30)} ${"cold start".padStart(12)} ${"notes".padStart(28)}`,
);
console.log(" " + "─".repeat(70));
console.log(
  ` ${"@isidorus/cpu".padEnd(30)}` +
    ` ${(coldStartMs.toFixed(0) + "ms").padStart(12)}` +
    ` ${"ready immediately (warmed)".padStart(28)}`,
);
if (tfjsColdMs !== undefined) {
  console.log(
    ` ${"@tensorflow/tfjs-node".padEnd(30)}` +
      ` ${(tfjsColdMs.toFixed(0) + "ms").padStart(12)}` +
      ` ${"+ first-req oneDNN cost".padStart(28)}`,
  );
  const ratio = coldStartMs / tfjsColdMs;
  const verdict =
    ratio < 1
      ? `${(1 / ratio).toFixed(1)}× faster`
      : `${ratio.toFixed(1)}× slower`;
  console.log(` isidorus vs tfjs-node: ${verdict}`);
}
console.log("─".repeat(72) + "\n");

// ─── Save results ─────────────────────────────────────────────────────────────

const modelName = basename(modelPath).replace(/\.[^/.]+$/, "");
const resultsDir = join(REPO_ROOT, "results", "inference_pool", modelName);

mkdirSync(resultsDir, { recursive: true });

// Clean up old TypeScript-generated benchmark files (those without -python suffix)
try {
  const files = readdirSync(resultsDir);
  for (const file of files) {
    if (file.endsWith(".json") && !file.includes("-python")) {
      const filePath = join(resultsDir, file);
      rmSync(filePath);
    }
  }
} catch {
  // Directory might not exist yet, that's fine
}

const filename = `inference_pool.json`;

const suite: BenchmarkSuite = {
  name: "inference_pool",
  description: `Worker-pool throughput benchmark — ${basename(modelPath)}`,
  results: [isidorusResult, ...(tfjsResult ? [tfjsResult] : [])],
  comparisons,
};

writeFileSync(join(resultsDir, filename), JSON.stringify(suite, null, 2));
console.log(`Results saved → results/inference_pool/${modelName}/${filename}`);
