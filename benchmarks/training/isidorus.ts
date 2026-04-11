/**
 * benchmarks/training/isidorus.ts
 *
 * Measures training throughput (steps/sec, samples/sec) for @isidorus/cpu.
 * Mirrors the Python training benchmark for direct comparison.
 *
 * Standalone:
 *   node --import tsx benchmarks/training/isidorus.ts
 */

import { performance } from "node:perf_hooks";
import { availableParallelism } from "node:os";
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
  computeStats,
  machineInfo,
  printHeader,
  printFooter,
} from "../../shared/stats.js";
import type { BenchmarkResult } from "../../shared/types.js";

import {
  INPUT_H,
  INPUT_W,
  INPUT_C,
  NUM_CLASSES,
  WARMUP_STEPS,
  BENCH_STEPS,
  BATCH_SIZES,
  MODEL_DESCRIPTION,
  OPTIMIZER,
  LR,
} from "./config.js";

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

  const opt =
    OPTIMIZER === "adam"
      ? new optimizers.Adam(g, model.params, LR)
      : new optimizers.SGD(g, model.params, LR);

  // Use all available cores — TF's default when called with no options is 1 thread,
  // which explains the 6× gap vs Python TF (which defaults to all cores).
  const sess = session(g, { intraOpThreads: availableParallelism() });
  await model.init(sess, opt);

  const nXElem = batchSize * INPUT_H * INPUT_W * INPUT_C;
  const xBuf = Buffer.alloc(nXElem * 4);
  const yBuf = Buffer.alloc(batchSize * 4);
  const xShape = [batchSize, INPUT_H, INPUT_W, INPUT_C];
  const yShape = [batchSize];

  // Fill with random data
  for (let i = 0; i < nXElem; i++)
    xBuf.writeFloatLE(Math.random() * 0.5, i * 4);
  for (let i = 0; i < batchSize; i++)
    yBuf.writeInt32LE(Math.floor(Math.random() * NUM_CLASSES), i * 4);

  // Warmup
  for (let i = 0; i < WARMUP_STEPS; i++)
    model.trainStepSync(sess, opt, xBuf, yBuf, xShape, yShape);

  // Timed — trainStepSync calls TF_SessionRun directly on the main thread,
  // bypassing libuv/TSFN/Promise overhead. TF's eigen pool handles parallelism.
  const stepSamples: number[] = [];
  let lastLoss = 0;

  for (let i = 0; i < BENCH_STEPS; i++) {
    const t0 = performance.now();
    const { loss } = model.trainStepSync(sess, opt, xBuf, yBuf, xShape, yShape);
    stepSamples.push(performance.now() - t0);
    lastLoss = loss;
  }

  const stepStats = computeStats(stepSamples);
  const stepsPerSec =
    (BENCH_STEPS * 1000) / stepSamples.reduce((a, b) => a + b, 0);
  const samplesPerSec = stepsPerSec * batchSize;

  sess.destroy();
  return { batchSize, stepStats, stepsPerSec, samplesPerSec, lastLoss };
}

export async function runIsidorusTrainingBench(): Promise<BenchmarkResult> {
  const version = await getRuntimeVersion();
  const t0 = performance.now();

  printHeader(
    `@isidorus/cpu  v${version}  — training (${WARMUP_STEPS} warmup + ${BENCH_STEPS} timed steps)`,
  );
  console.log(
    ` ${"batch".padEnd(8)}` +
      ` ${"step mean".padStart(11)}` +
      ` ${"p99".padStart(11)}` +
      ` ${"steps/s".padStart(10)}` +
      ` ${"samples/s".padStart(11)}` +
      ` ${"loss".padStart(8)}`,
  );
  console.log(" " + "─".repeat(70));

  const batches = [];

  for (const b of BATCH_SIZES) {
    process.stdout.write(`  batch=${b}  `);
    const r = await runForBatch(b);
    batches.push({
      batchSize: r.batchSize,
      latency: r.stepStats,
      throughput: r.samplesPerSec,
      stepsPerSec: r.stepsPerSec,
      samplesPerSec: r.samplesPerSec,
    });

    console.log(
      ` ${String(b).padEnd(8)}` +
        ` ${r.stepStats.mean.toFixed(2).padStart(9)} ms` +
        ` ${r.stepStats.p99.toFixed(2).padStart(9)} ms` +
        ` ${r.stepsPerSec.toFixed(1).padStart(10)}` +
        ` ${r.samplesPerSec.toFixed(1).padStart(11)}` +
        ` ${r.lastLoss.toFixed(4).padStart(8)}`,
    );
  }

  printFooter();

  return {
    runtime: "@isidorus/cpu",
    runtimeVersion: version,
    model: MODEL_DESCRIPTION,
    inputShape: [INPUT_H, INPUT_W, INPUT_C],
    warmupIters: WARMUP_STEPS,
    benchIters: BENCH_STEPS,
    batches,
    machineInfo: machineInfo(),
    timestamp: new Date().toISOString(),
    durationMs: performance.now() - t0,
  };
}

if (
  process.argv[1]?.endsWith("isidorus.ts") ||
  process.argv[1]?.endsWith("isidorus.js")
) {
  await runIsidorusTrainingBench();
}
