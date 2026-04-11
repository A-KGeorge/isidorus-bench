/**
 * benchmarks/conv2d/tfjs_node.ts
 *
 * Runs the Conv2D benchmark using @tensorflow/tfjs-node.
 * Called by run.ts; can also be run standalone:
 *
 *   node --import tsx benchmarks/conv2d/tfjs_node.ts
 */

import { performance } from "node:perf_hooks";

import {
  batchResult,
  collectSamples,
  machineInfo,
  printHeader,
  printBatchRow,
  printFooter,
  startEventLoopMonitor,
  printEventLoopHealth,
} from "../../shared/stats.js";

import type { BenchmarkResult } from "../../shared/types.js";

import {
  INPUT_H,
  INPUT_W,
  INPUT_C,
  NUM_CLASSES,
  WARMUP_ITERS,
  BENCH_ITERS,
  BATCH_SIZES,
  MODEL_DESCRIPTION,
} from "./config.js";

// ─── tfjs-node version ───────────────────────────────────────────────────────

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

// ─── Single-batch benchmark ──────────────────────────────────────────────────

async function runForBatch(tf: any, model: any, batchSize: number) {
  // Build a fixed input tensor once and reuse it across iterations.
  // tfjs keeps it in its tensor pool so allocation cost is paid once.
  const input = tf.randomUniform(
    [batchSize, INPUT_H, INPUT_W, INPUT_C],
    0,
    0.5,
  ) as any;

  // Warmup
  for (let i = 0; i < WARMUP_ITERS; i++) {
    const out = model.predict(input) as any;
    // .dataSync() forces synchronous completion — without it predict()
    // is asynchronous and wall-clock timing would measure scheduling, not
    // compute. This matches how @isidorus/cpu benchmarks are timed (each
    // predict() awaits the TF_SessionRun completion before returning).
    out.dataSync();
    out.dispose();
  }

  // Timed iterations
  const monitor = startEventLoopMonitor();
  const samples = await collectSamples(BENCH_ITERS, async () => {
    const out = model.predict(input) as any;
    out.dataSync();
    out.dispose();
  });
  const eventLoopHealth = monitor.stop();

  input.dispose();
  const result = batchResult(batchSize, samples);
  result.eventLoopHealth = eventLoopHealth;
  return result;
}

// ─── Entry point ─────────────────────────────────────────────────────────────

export async function runTfjsNodeBench(): Promise<BenchmarkResult | null> {
  let tf: any;
  try {
    // Dynamic import so the file is still parseable even when tfjs-node is
    // not installed — run.ts checks for null and skips the comparison.
    tf = await import("@tensorflow/tfjs-node");
  } catch {
    console.warn(
      "\n⚠  @tensorflow/tfjs-node not installed — skipping baseline.\n" +
        "   Install with: npm install -D @tensorflow/tfjs-node\n",
    );
    return null;
  }

  const version = await getRuntimeVersion();
  const t0 = performance.now();

  // Build model once; reuse across all batch sizes.
  // tfjs-node accepts dynamic batch dims so one model covers all sizes.
  const model = tf.sequential();
  model.add(
    tf.layers.conv2d({
      filters: 32,
      kernelSize: 3,
      activation: "relu",
      padding: "same",
      inputShape: [INPUT_H, INPUT_W, INPUT_C],
    }),
  );
  model.add(
    tf.layers.conv2d({
      filters: 64,
      kernelSize: 3,
      activation: "relu",
      padding: "same",
    }),
  );
  model.add(
    tf.layers.conv2d({
      filters: 64,
      kernelSize: 3,
      activation: "relu",
      padding: "valid",
      strides: 2,
    }),
  );
  model.add(tf.layers.flatten());
  model.add(tf.layers.dense({ units: 128, activation: "relu" }));
  model.add(tf.layers.dense({ units: NUM_CLASSES, activation: "softmax" }));

  printHeader(
    `@tensorflow/tfjs-node  v${version}  (${WARMUP_ITERS} warmup + ${BENCH_ITERS} timed iters)`,
  );

  const batches = [];
  for (const b of BATCH_SIZES) {
    process.stdout.write(`  batch=${b}  `);
    const r = await runForBatch(tf, model, b);
    batches.push(r);
    printBatchRow(r);
    if (r.eventLoopHealth) {
      printEventLoopHealth(" ".repeat(14), r.eventLoopHealth);
    }
  }

  printFooter();

  return {
    runtime: "@tensorflow/tfjs-node",
    runtimeVersion: version,
    model: MODEL_DESCRIPTION,
    inputShape: [INPUT_H, INPUT_W, INPUT_C],
    warmupIters: WARMUP_ITERS,
    benchIters: BENCH_ITERS,
    batches,
    machineInfo: machineInfo(),
    timestamp: new Date().toISOString(),
    durationMs: performance.now() - t0,
  };
}

// ─── Standalone run ──────────────────────────────────────────────────────────

if (
  process.argv[1]?.endsWith("tfjs_node.ts") ||
  process.argv[1]?.endsWith("tfjs_node.js")
) {
  await runTfjsNodeBench();
}
