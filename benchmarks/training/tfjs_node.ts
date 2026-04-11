/**
 * benchmarks/training/tfjs_node.ts
 *
 * @tensorflow/tfjs-node training baseline.
 * Uses model.fit() for one epoch of one batch — standard tfjs-node training API.
 */

import { performance } from "node:perf_hooks";
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
    const pkg = req("@tensorflow/tfjs-node/package.json") as {
      version: string;
    };
    return pkg.version;
  } catch {
    return "unknown";
  }
}

export async function runTfjsNodeTrainingBench(): Promise<BenchmarkResult | null> {
  let tf: any;
  try {
    tf = await import("@tensorflow/tfjs-node");
  } catch {
    console.warn(
      "\n⚠  @tensorflow/tfjs-node not installed — skipping baseline.\n",
    );
    return null;
  }

  const version = await getRuntimeVersion();
  const t0 = performance.now();

  // Build model
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

  model.compile({
    optimizer: OPTIMIZER === "adam" ? tf.train.adam(LR) : tf.train.sgd(LR),
    loss: "sparseCategoricalCrossentropy",
    metrics: ["accuracy"],
  });

  printHeader(
    `@tensorflow/tfjs-node  v${version}  — training (${WARMUP_STEPS} warmup + ${BENCH_STEPS} timed steps)`,
  );
  console.log(
    ` ${"batch".padStart(8)}` +
      ` ${"step mean".padStart(11)}` +
      ` ${"p99".padStart(11)}` +
      ` ${"steps/s".padStart(10)}` +
      ` ${"samples/s".padStart(11)}` +
      ` ${"loss".padStart(8)}`,
  );
  console.log(" " + "─".repeat(70));

  const batches = [];

  for (const batchSize of BATCH_SIZES) {
    process.stdout.write(`  batch=${batchSize}  `);

    const nXElem = batchSize * INPUT_H * INPUT_W * INPUT_C;
    const xData = Float32Array.from(
      { length: nXElem },
      () => Math.random() * 0.5,
    );
    const yData = Float32Array.from({ length: batchSize }, () =>
      Math.floor(Math.random() * NUM_CLASSES),
    );
    const xs = tf.tensor4d(xData, [batchSize, INPUT_H, INPUT_W, INPUT_C]);
    const ys = tf.tensor1d(yData, "float32");

    // Warmup
    for (let i = 0; i < WARMUP_STEPS; i++) {
      const h = await model.fit(xs, ys, { epochs: 1, batchSize, verbose: 0 });
    }

    // Timed
    const stepSamples: number[] = [];
    let lastLoss = 0;

    for (let i = 0; i < BENCH_STEPS; i++) {
      const t0Step = performance.now();
      const h = await model.fit(xs, ys, { epochs: 1, batchSize, verbose: 0 });
      stepSamples.push(performance.now() - t0Step);
      lastLoss = h.history.loss[0] as number;
    }

    const stepStats = computeStats(stepSamples);
    const stepsPerSec =
      (BENCH_STEPS * 1000) / stepSamples.reduce((a, b) => a + b, 0);
    const samplesPerSec = stepsPerSec * batchSize;

    xs.dispose();
    ys.dispose();

    console.log(
      ` ${String(batchSize).padEnd(8)}` +
        ` ${stepStats.mean.toFixed(2).padStart(9)} ms` +
        ` ${stepStats.p99.toFixed(2).padStart(9)} ms` +
        ` ${stepsPerSec.toFixed(1).padStart(10)}` +
        ` ${samplesPerSec.toFixed(1).padStart(11)}` +
        ` ${lastLoss.toFixed(4).padStart(8)}`,
    );

    batches.push({
      batchSize,
      latency: stepStats,
      throughput: samplesPerSec,
      stepsPerSec,
      samplesPerSec,
    });
  }

  printFooter();

  return {
    runtime: "@tensorflow/tfjs-node",
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
  process.argv[1]?.endsWith("tfjs_node.ts") ||
  process.argv[1]?.endsWith("tfjs_node.js")
) {
  await runTfjsNodeTrainingBench();
}
