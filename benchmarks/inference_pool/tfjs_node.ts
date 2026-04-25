/**
 * benchmarks/inference_pool/tfjs_node.ts
 *
 * @tensorflow/tfjs-node baseline for the InferencePool throughput benchmark.
 *
 * Strategy: single model, N concurrent callers each calling model.predict()
 * simultaneously. tfjs-node serialises inference internally (one TF session,
 * its own thread pool) — this measures how it handles request concurrency
 * compared to InferencePool's per-Worker Session approach.
 *
 * Compatible with frozen .pb files produced by bench/generate_models.py.
 */

import { performance } from "node:perf_hooks";
import { readFileSync, existsSync } from "node:fs";
import { availableParallelism } from "node:os";

import {
  batchResult,
  computeStats,
  machineInfo,
  printHeader,
  printBatchRow,
  printFooter,
  startEventLoopMonitor,
  printEventLoopHealth,
  fmtMs,
  fmtTps,
} from "../../shared/stats.js";

import type { BenchmarkResult } from "../../shared/types.js";

// ─── Config (mirrors inference_pool/run.ts) ───────────────────────────────

const WARMUP_REQUESTS = parseInt(process.env.WARMUP_ITERS ?? "20", 10);
const BENCH_REQUESTS = parseInt(process.env.BENCH_ITERS ?? "200", 10);
export const CONCURRENCY_LEVELS_TFJSNODE = [1, 2, 3, 4, 5, 6, 7, 8]
  .filter((v, i, a) => a.indexOf(v) === i)
  .sort((a, b) => a - b);

// ─── Version ─────────────────────────────────────────────────────────────────

async function getRuntimeVersion(): Promise<string> {
  try {
    const { createRequire } = await import("node:module");
    const req = createRequire(import.meta.url);
    const pkg = req("@tensorflow/tfjs-node/package.json") as {
      version: string;
    };
    return pkg.version;
  } catch {
    return "unknown";
  }
}

// ─── Load frozen .pb via tfjs-node ───────────────────────────────────────────
//
// tfjs-node can load frozen graphs via tf.node.loadSavedModel or
// tf.loadGraphModel with a file:// URL (for a SavedModel directory).
// For a raw frozen .pb we use tf.node.loadFrozenModel which is available
// as a lower-level API.

async function loadFrozenModel(
  tf: any,
  modelPath: string,
  inputOp: string,
  outputOp: string,
) {
  // If generate_models.py exported a SavedModel directory alongside the .pb, use that for tfjs-node.
  const smPath = modelPath.replace(/\.pb$/, ".savedmodel");
  if (existsSync(smPath)) {
    try {
      const model = await tf.node.loadSavedModel(smPath);
      return { model, type: "savedModel" };
    } catch {}
  }

  // tf.node.loadFrozenModel accepts (modelPath, outputNodeNames)
  // Returns a TFSavedModel-like object with .predict() / .execute()
  try {
    const model = await tf.node.loadGraphModel(
      `file://${modelPath.replace(/\\/g, "/")}`,
      {},
    );
    return { model, type: "graphModel" };
  } catch {
    // Older API
    const model = tf.node.loadFrozenModel
      ? tf.node.loadFrozenModel(modelPath, [inputOp, outputOp])
      : null;
    return model ? { model, type: "frozenModel" } : null;
  }
}

// ─── Run concurrent requests ──────────────────────────────────────────────────

// ─── Fair latency + event loop stall measurement ─────────────────────────────
//
// tfjs-node's predict() is synchronous — it blocks the JS thread for the
// duration of TF_SessionRun.
//
// Latency (fair): set t0 at burst-start so all requests include queue-wait.
//   naive:   t0 before each predict → mean ≈ 6.6ms (queue wait invisible)
//   fair:    t0 at burst-start       → mean ≈ 29ms  (matches isidorus)
//
// Event loop stall (correct): run predict() WITHOUT Promise.resolve() wrapper.
//   With Promise.resolve().then(), work runs as microtasks which drain before
//   any macrotask (setInterval) fires — timer sees ticks=0 regardless of how
//   long the burst takes. Running synchronously + yielding with setImmediate
//   after each burst lets the timer correctly record the accumulated stall.

// Replace runConcurrent's input creation and predict call

async function runConcurrent(
  tf: any,
  model: any,
  modelType: string,
  inputShape: number[],
  concurrency: number,
  nRequests: number,
): Promise<number[]> {
  const samples: number[] = [];

  // ── Build inputs from model signature ──────────────────────────────────
  // For SavedModel with multiple inputs (e.g. BERT), model.inputs is an array
  // of {name, dtype, shape}. Build a named object so predict() gets all feeds.
  // For single-input models / graphModel, fall back to the original behaviour.
  let inputArg: any;
  let inputsToDispose: any[] = [];

  if (
    modelType === "savedModel" &&
    Array.isArray(model.inputs) &&
    model.inputs.length > 1
  ) {
    const namedInputs: Record<string, any> = {};
    for (const inputInfo of model.inputs) {
      // FIX: Strip the 'serving_default_' prefix if it exists
      // The error shows the model wants 'input_ids' but you provided 'serving_default_input_ids'
      const cleanName = inputInfo.name.replace(/^serving_default_/, "");

      const shape = inputInfo.shape.map((d: any) => {
        const n = Number(d);
        return isNaN(n) || n < 1 ? 1 : Math.round(n);
      });
      const dtype = inputInfo.dtype ?? "int32";
      const t =
        dtype === "int32"
          ? tf.zeros(shape, "int32")
          : tf.randomUniform(shape, 0, 1);

      namedInputs[cleanName] = t; // Use the cleaned name here
      inputsToDispose.push(t);
    }
    inputArg = namedInputs;
  } else {
    const t = tf.randomUniform(inputShape, 0, 1);
    inputArg = t;
    inputsToDispose = [t];
  }

  for (let base = 0; base < nRequests; base += concurrency) {
    const burstSize = Math.min(concurrency, nRequests - base);
    const t0Burst = performance.now();

    for (let i = 0; i < burstSize; i++) {
      let out: any;
      if (modelType === "graphModel" || modelType === "savedModel") {
        out = model.predict(inputArg);
      } else {
        out = model.execute(inputArg);
      }
      const outputs = Array.isArray(out)
        ? out
        : typeof out === "object" && !out.dtype
          ? Object.values(out) // named output dict (BERT returns {last_hidden_state: ...})
          : [out];
      outputs.forEach((o: any) => {
        o.dataSync();
        o.dispose();
      });
      samples.push(performance.now() - t0Burst);
    }

    await new Promise<void>((resolve) => setImmediate(resolve));
  }

  inputsToDispose.forEach((t) => t.dispose());
  return samples;
}
// ─── Entry point ─────────────────────────────────────────────────────────────

export async function runTfjsNodePoolBench(
  modelPath: string,
  inputShape: number[], // e.g. [1, 224, 224, 3]
  inputOp: string,
  outputOp: string,
  profile: string = "auto",
): Promise<BenchmarkResult | null> {
  let tf: any;
  try {
    tf = await import("@tensorflow/tfjs-node");
  } catch {
    console.warn(
      "\n⚠  @tensorflow/tfjs-node not installed — skipping baseline.\n" +
        "   Install with: npm install -D @tensorflow/tfjs-node\n",
    );
    return null;
  }

  const version = await getRuntimeVersion();

  // Time model load in isolation — this is the tfjs-node cold start cost.
  // Unlike isidorus, tfjs-node pays no warmup at load time; oneDNN compiles
  // on the first predict() call instead.
  const coldStartT0 = performance.now();
  let modelObj: any;
  let modelType = "graphModel";
  try {
    const result = await loadFrozenModel(tf, modelPath, inputOp, outputOp);
    if (!result) throw new Error("No compatible loader");
    modelObj = result.model;
    modelType = result.type;
  } catch (e) {
    console.warn(`\n⚠  tfjs-node could not load ${modelPath}: ${e}\n`);
    return null;
  }
  const coldStartMs = performance.now() - coldStartT0;

  const t0 = performance.now();

  printHeader(`@tensorflow/tfjs-node  v${version}  — concurrent throughput`);
  console.log(
    ` ${"workers".padEnd(8)}` +
      ` ${"mean".padStart(11)}` +
      ` ${"p50".padStart(11)}` +
      ` ${"p95".padStart(11)}` +
      ` ${"p99".padStart(11)}` +
      ` ${"req/s".padStart(10)}`,
  );
  console.log(" " + "─".repeat(70));

  const batches = [];

  for (const concurrency of CONCURRENCY_LEVELS_TFJSNODE) {
    // Warmup
    await runConcurrent(
      tf,
      modelObj,
      modelType,
      inputShape,
      concurrency,
      WARMUP_REQUESTS,
    );

    // Timed — monitor event loop health during inference.
    // tfjs-node's predict() is synchronous: the event loop stalls for the
    // entire duration of TF_SessionRun. The monitor will show max stall ≈
    // concurrency × single_latency, confirming the blocking behaviour.
    const monitor = startEventLoopMonitor(5);
    const wallStart = performance.now();
    const samples = await runConcurrent(
      tf,
      modelObj,
      modelType,
      inputShape,
      concurrency,
      BENCH_REQUESTS,
    );
    const wallMs = performance.now() - wallStart;
    const health = monitor.stop();

    const s = computeStats(samples);
    const reqPerS = ((BENCH_REQUESTS * 1000) / wallMs).toFixed(0);

    console.log(
      ` ${String(concurrency).padEnd(8)}` +
        ` ${fmtMs(s.mean).padStart(11)}` +
        ` ${fmtMs(s.p50).padStart(11)}` +
        ` ${fmtMs(s.p95).padStart(11)}` +
        ` ${fmtMs(s.p99).padStart(11)}` +
        ` ${(reqPerS + " req/s").padStart(10)}`,
    );
    printEventLoopHealth(`c=${concurrency}`, health);

    batches.push({
      batchSize: concurrency,
      latency: s,
      throughput: parseFloat(reqPerS),
      eventLoop: health,
    });
  }

  printFooter();

  return {
    runtime: "@tensorflow/tfjs-node (single session, concurrent callers)",
    runtimeVersion: version,
    model: modelPath.split(/[\\/]/).pop() ?? modelPath,
    profile,
    inputShape,
    warmupIters: WARMUP_REQUESTS,
    benchIters: BENCH_REQUESTS,
    batches,
    machineInfo: machineInfo(),
    timestamp: new Date().toISOString(),
    durationMs: performance.now() - t0,
    coldStartMs,
  };
}
