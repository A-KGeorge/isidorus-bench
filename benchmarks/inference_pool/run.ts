/**
 * benchmarks/inference_pool/run.ts
 *
 * Orchestrates InferencePool, tfjs-node, and onnxruntime-node benchmarks.
 * ORT requires a .onnx model alongside the .pb:
 *   python bench/convert_to_onnx.py <model.pb>
 *
 * Usage:
 *   node --import tsx benchmarks/inference_pool/run.ts <model.pb>
 *   node --import tsx benchmarks/inference_pool/run.ts <model.pb> --profile latency
 */

import { performance } from "node:perf_hooks";
import { writeFileSync, mkdirSync, existsSync } from "node:fs";
import { join, basename, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { availableParallelism } from "node:os";

import { InferencePool } from "@isidorus/cpu";
import { DType, dtypeItemSize } from "@isidorus/cpu";
import {
  computeStats,
  machineInfo,
  printHeader,
  printFooter,
  printSpeedupTable,
  startEventLoopMonitor,
  printEventLoopHealth,
  fmtMs,
  mergeResults,
} from "../../shared/stats.js";
import type { BenchmarkSuite, SpeedupEntry } from "../../shared/types.js";
import { runTfjsNodePoolBench } from "./tfjs_node.js";
import { runOnnxRuntimeNodeBench } from "./onnxruntime_node.js";

const __dirname = fileURLToPath(new URL(".", import.meta.url));
const REPO_ROOT = join(__dirname, "..", "..");

// ─── Config ──────────────────────────────────────────────────────────────────

const modelPath = process.argv[2];

const profileArg = process.argv.indexOf("--profile");
const profile: "auto" | "latency" | "throughput" =
  profileArg !== -1 && process.argv[profileArg + 1]
    ? (process.argv[profileArg + 1] as "auto" | "latency" | "throughput")
    : "auto";

// Optional --onnx-model flag; otherwise look for <same stem>.onnx
const onnxArg = process.argv.indexOf("--onnx-model");
const onnxModelPath: string | null =
  onnxArg !== -1 && process.argv[onnxArg + 1]
    ? process.argv[onnxArg + 1]
    : (() => {
        const candidate = modelPath?.replace(/\.pb$/, ".onnx");
        return candidate && existsSync(candidate) ? candidate : null;
      })();

const CONCURRENCY_LEVELS = [1, 2, 3, 4, 5, 6, 7, 8];
const WARMUP_REQUESTS = 20;
const BENCH_REQUESTS = 200;

// ─── Validation ──────────────────────────────────────────────────────────────

if (!modelPath) {
  console.error(
    "Usage: node --import tsx benchmarks/inference_pool/run.ts <model.pb> [--profile auto|latency|throughput] [--onnx-model model.onnx]",
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
  ORT model:   ${onnxModelPath ?? "not found — skipping (run: python bench/convert_to_onnx.py " + basename(modelPath) + ")"}
  Concurrency: ${CONCURRENCY_LEVELS.join(", ")} concurrent callers
  Requests:    ${WARMUP_REQUESTS} warmup + ${BENCH_REQUESTS} timed
  Platform:    ${process.platform}-${process.arch}  Node: ${process.version}
  HW threads:  ${availableParallelism()}
`);

// ─── InferencePool benchmark ─────────────────────────────────────────────────

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
let resolvedShape: number[] = [];

const coldStartT0 = performance.now();
const pool = await InferencePool.create({ modelPath, profile });
const inputDtype = pool.resolvedInputDtype;
const itemSize = dtypeItemSize(inputDtype);
const coldStartMs = performance.now() - coldStartT0;
console.log(
  `  Cold start:  ${coldStartMs.toFixed(0)}ms (graph load + session + autotuner + oneDNN warmup)`,
);

{
  const resolved = pool.resolvedInputShape;
  resolvedShape = resolved.map((d) => d ?? 1);
  console.log(`  Resolved input shape: ${JSON.stringify(resolved)}`);

  if (resolved.every((d) => d === null)) {
    console.error(
      "\n✗ Placeholder has fully dynamic shape — cannot auto-size input buffer.\n",
    );
    await pool.destroy();
    process.exit(1);
  }
}

const inferShape = resolvedShape;
const nElems = inferShape.reduce((a, b) => a * b, 1);
const inferBuf = Buffer.alloc(nElems * itemSize);
console.log(
  `  Inference shape:  ${JSON.stringify(inferShape)}  (${nElems} elems, ${nElems * itemSize} bytes)`,
);

await Promise.all(
  Array.from({ length: WARMUP_REQUESTS }, () =>
    pool.infer(inferBuf, inferShape, inputDtype),
  ),
);

for (const concurrency of CONCURRENCY_LEVELS) {
  const samples: number[] = [];
  const wallStart = performance.now();
  const monitor = startEventLoopMonitor(5);

  const inFlight = new Set<Promise<void>>();
  let issued = 0;

  while (issued < BENCH_REQUESTS || inFlight.size > 0) {
    while (issued < BENCH_REQUESTS && inFlight.size < concurrency) {
      const t0 = performance.now();
      let p: Promise<void>;
      p = pool.infer(inferBuf, inferShape, inputDtype).then(() => {
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

const isidorusResult = {
  runtime: "@isidorus/cpu (InferencePool tf-parallel)",
  runtimeVersion: "latest",
  model: basename(modelPath),
  profile,
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

// ─── tfjs-node baseline ───────────────────────────────────────────────────────

console.log(
  "\n── @tensorflow/tfjs-node ──────────────────────────────────────────────",
);
const tfjsResult = await runTfjsNodePoolBench(
  modelPath,
  resolvedShape,
  "inputs",
  "Identity",
  profile,
);

// ─── onnxruntime-node baseline ────────────────────────────────────────────────

let ortResult = null;
if (onnxModelPath) {
  console.log(
    "\n── onnxruntime-node ───────────────────────────────────────────────────",
  );

  // Determine input dtype from the TF model: if the pool's inputDtype is INT32
  // (e.g. BERT), pass int32 to ORT as well.
  const ortInputDtype: "float32" | "int32" =
    inputDtype === DType.INT32 ? "int32" : "float32";

  // intraOpThreads: match what InferencePool autotuned for a fair comparison.
  const intraOpThreads = Math.max(1, Math.floor(availableParallelism() / 4));

  ortResult = await runOnnxRuntimeNodeBench(
    onnxModelPath,
    inferShape,
    ortInputDtype,
    intraOpThreads,
    profile,
  );
} else {
  console.log(
    "\n── onnxruntime-node: skipped (no .onnx model found) ───────────────────",
  );
  console.log(
    `   Convert with: python bench/convert_to_onnx.py ${basename(modelPath)}\n`,
  );
}

// ─── Comparison tables ────────────────────────────────────────────────────────

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
}

// Event loop stall summary — the core comparison.
console.log("\n" + "─".repeat(72));
console.log(" Event loop stall summary (stallFraction — lower is better)");
console.log(
  " stallFraction = fraction of 5ms ticks where the loop stalled >5ms",
);
console.log("─".repeat(72));
console.log(
  ` ${"runtime".padEnd(42)} ${"c=1".padStart(6)} ${"c=4".padStart(6)} ${"c=8".padStart(6)}`,
);
console.log(" " + "─".repeat(70));

const stallRow = (name: string, batches: any[]) => {
  const get = (c: number) => {
    const b = batches.find((b: any) => b.batchSize === c);
    const h = b?.eventLoop ?? b?.eventLoop;
    if (!h) return "  n/a";
    return (h.stallFraction * 100).toFixed(0).padStart(5) + "%";
  };
  console.log(` ${name.slice(0, 42).padEnd(42)} ${get(1)} ${get(4)} ${get(8)}`);
};

stallRow(
  "@isidorus/cpu (InferencePool)",
  isidorusResult.batches.map((b: any, i: number) => ({
    ...b,
    eventLoop: isidorusResult.eventLoopHealth[i],
  })),
);
if (tfjsResult) stallRow(tfjsResult.runtime, tfjsResult.batches);
if (ortResult) stallRow(ortResult.runtime, ortResult.batches);
console.log("─".repeat(72) + "\n");

// ─── Cold start table ──────────────────────────────────────────────────────────

console.log("─".repeat(72));
console.log(" Cold start: time until pool is ready to serve requests");
console.log("─".repeat(72));
const coldRow = (name: string, ms: number, note: string) =>
  console.log(
    ` ${name.padEnd(36)} ${(ms.toFixed(0) + "ms").padStart(9)}  ${note}`,
  );
coldRow("@isidorus/cpu", coldStartMs, "ready immediately (warmed)");
if (tfjsResult?.coldStartMs !== undefined)
  coldRow(
    tfjsResult.runtime,
    tfjsResult.coldStartMs,
    "+ first-req oneDNN cost",
  );
if (ortResult?.coldStartMs !== undefined)
  coldRow(ortResult.runtime, ortResult.coldStartMs, "");
console.log("─".repeat(72) + "\n");

// ─── Save results ─────────────────────────────────────────────────────────────

const modelName = basename(modelPath).replace(/\.[^/.]+$/, "");
const resultsDir = join(REPO_ROOT, "results", "inference_pool", modelName);
mkdirSync(resultsDir, { recursive: true });

const filePath = join(resultsDir, "inference_pool.json");

const suite: BenchmarkSuite = {
  name: "inference_pool",
  description: `Worker-pool throughput benchmark — ${basename(modelPath)}`,
  results: [
    isidorusResult,
    ...(tfjsResult ? [tfjsResult] : []),
    ...(ortResult ? [ortResult] : []),
  ],
  comparisons,
};

const mergedSuite = mergeResults({ filePath, newSuite: suite, profile });
writeFileSync(filePath, JSON.stringify(mergedSuite, null, 2));
console.log(
  `Results saved → results/inference_pool/${modelName}/inference_pool.json`,
);
