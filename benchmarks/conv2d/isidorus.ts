/**
 * benchmarks/conv2d/isidorus.ts
 *
 * Runs the Conv2D benchmark using @isidorus/cpu.
 * Called by run.ts; can also be run standalone:
 *
 *   node --import tsx benchmarks/conv2d/isidorus.ts
 */

import { performance } from "node:perf_hooks";
import {
  graph,
  session,
  optimizers,
  Sequential,
  Conv2D,
  Flatten,
  Dense,
} from "@isidorus/cpu";

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

// ─── @isidorus/cpu version ───────────────────────────────────────────────────

async function getRuntimeVersion(): Promise<string> {
  try {
    const { createRequire } = await import("node:module");
    const req = createRequire(import.meta.url);
    const pkg = req("@isidorus/cpu/package.json") as { version: string };
    return pkg.version;
  } catch {
    return "unknown";
  }
}

// ─── Single-batch benchmark ──────────────────────────────────────────────────
//
// Each batch size builds its own graph and session with the batch dimension
// fixed. This matches how tfjs-node handles batch shapes and gives TF's
// XLA-style graph optimizer the best chance to fuse ops for the specific size.

async function runForBatch(batchSize: number) {
  const g = graph();
  const model = new Sequential(g, [
    new Conv2D(32, {
      kernelSize: 3,
      activation: "relu",
      padding: "SAME",
      name: "c1",
    }),
    new Conv2D(64, {
      kernelSize: 3,
      activation: "relu",
      padding: "SAME",
      name: "c2",
    }),
    new Conv2D(64, {
      kernelSize: 3,
      activation: "relu",
      padding: "VALID",
      strides: 2,
      name: "c3",
    }),
    new Flatten(),
    new Dense(128, { activation: "relu", name: "fc1" }),
    new Dense(NUM_CLASSES, { activation: "softmax", name: "out" }),
  ]);

  model.compile({
    loss: "sparse_categorical_crossentropy",
    inputShape: [INPUT_H, INPUT_W, INPUT_C],
  });

  const opt = new optimizers.Adam(g, model.params, 0.001);
  const sess = session(g);
  await model.init(sess, opt);

  // Build a fixed input buffer with random floats in [0, 0.5].
  const nElem = batchSize * INPUT_H * INPUT_W * INPUT_C;
  const xBuf = Buffer.alloc(nElem * 4);
  for (let i = 0; i < nElem; i++) xBuf.writeFloatLE(Math.random() * 0.5, i * 4);
  const xShape = [batchSize, INPUT_H, INPUT_W, INPUT_C];

  // Warmup — allows TF to JIT-compile any lazy kernel builds.
  for (let i = 0; i < WARMUP_ITERS; i++)
    await model.predict(sess, xBuf, xShape);

  // Timed iterations.
  const monitor = startEventLoopMonitor();
  const samples = await collectSamples(BENCH_ITERS, () =>
    model.predict(sess, xBuf, xShape).then(() => {}),
  );
  const eventLoopHealth = monitor.stop();

  sess.destroy();
  const result = batchResult(batchSize, samples);
  result.eventLoopHealth = eventLoopHealth;
  return result;
}

// ─── Entry point ─────────────────────────────────────────────────────────────

export async function runIsidorusBench(): Promise<BenchmarkResult> {
  const version = await getRuntimeVersion();
  const t0 = performance.now();

  printHeader(
    `@isidorus/cpu  v${version}  (${WARMUP_ITERS} warmup + ${BENCH_ITERS} timed iters)`,
  );

  const batches = [];
  for (const b of BATCH_SIZES) {
    process.stdout.write(`  batch=${b}  `);
    const r = await runForBatch(b);
    batches.push(r);
    printBatchRow(r);
    if (r.eventLoopHealth) {
      printEventLoopHealth(" ".repeat(14), r.eventLoopHealth);
    }
  }

  printFooter();

  return {
    runtime: "@isidorus/cpu",
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
  process.argv[1]?.endsWith("isidorus.ts") ||
  process.argv[1]?.endsWith("isidorus.js")
) {
  await runIsidorusBench();
}
