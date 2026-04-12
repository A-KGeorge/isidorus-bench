/**
 * benchmarks/conv2d/isidorus.ts
 *
 * Runs the Conv2D benchmark using @isidorus/cpu.
 * Called by run.ts; can also be run standalone:
 *
 *   node --import tsx benchmarks/conv2d/isidorus.ts
 */

import { performance } from "node:perf_hooks";
import { Model, Conv2D, Flatten, Dense } from "@isidorus/cpu";

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
// Each batch size builds its own model with the batch dimension handled
// automatically by the high-level Model API. Simple and clean.

async function runForBatch(batchSize: number) {
  const model = new Model(
    [INPUT_H, INPUT_W, INPUT_C],
    [
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
    ],
  );

  model.compile({ loss: "sparse_categorical_crossentropy", optimizer: "adam" });

  // Build a fixed input buffer with random floats in [0, 0.5].
  // Can pass plain JS array thanks to auto-conversion!
  const nElem = batchSize * INPUT_H * INPUT_W * INPUT_C;
  const xData: number[] = [];
  for (let i = 0; i < nElem; i++) xData.push(Math.random() * 0.5);

  // Warmup — allows TF to JIT-compile any lazy kernel builds.
  // Uses auto data conversion under the hood
  for (let i = 0; i < WARMUP_ITERS; i++) await model.predict(xData, batchSize);

  // Timed iterations with event loop monitoring.
  const monitor = startEventLoopMonitor();
  const samples = await collectSamples(BENCH_ITERS, () =>
    model.predict(xData, batchSize).then(() => {}),
  );
  const eventLoopHealth = monitor.stop();

  model.dispose();
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
