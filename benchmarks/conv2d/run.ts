/**
 * benchmarks/conv2d/run.ts
 *
 * Orchestrates both @isidorus/cpu and @tensorflow/tfjs-node benchmarks,
 * prints a side-by-side comparison table, and saves the results to:
 *
 *   results/conv2d/<timestamp>-<platform>-<arch>.json
 *
 * Usage:
 *   node --import tsx benchmarks/conv2d/run.ts
 *   BENCH_ITERS=50 WARMUP_ITERS=10 node --import tsx benchmarks/conv2d/run.ts
 */

import { writeFileSync, mkdirSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

import { printSpeedupTable, mergeResults } from "../../shared/stats.js";
import type { BenchmarkSuite, SpeedupEntry } from "../../shared/types.js";

import { runIsidorusBench } from "./isidorus.js";
import { runTfjsNodeBench } from "./tfjs_node.js";
import { BENCH_ITERS, WARMUP_ITERS, MODEL_DESCRIPTION } from "./config.js";

const __dirname = fileURLToPath(new URL(".", import.meta.url));
const REPO_ROOT = join(__dirname, "..", "..");

// ─── Header ──────────────────────────────────────────────────────────────────

console.log(`
╔══════════════════════════════════════════════════════════════════════╗
║           @isidorus/cpu  Conv2D Benchmark Suite                      ║
╚══════════════════════════════════════════════════════════════════════╝

  Model:   ${MODEL_DESCRIPTION}
  Warmup:  ${WARMUP_ITERS} iters     Bench: ${BENCH_ITERS} iters
  Platform: ${process.platform}-${process.arch}   Node: ${process.version}
`);

// ─── Run both benchmarks ─────────────────────────────────────────────────────

const isidorusResult = await runIsidorusBench();
const tfjsResult = await runTfjsNodeBench();

// ─── Speedup table ───────────────────────────────────────────────────────────

const comparisons: SpeedupEntry[] = [];

if (tfjsResult) {
  printSpeedupTable(
    { name: tfjsResult.runtime, batches: tfjsResult.batches },
    { name: isidorusResult.runtime, batches: isidorusResult.batches },
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
} else {
  console.log(
    "Skipping speedup table — @tensorflow/tfjs-node results unavailable.\n" +
      "Install with: npm install -D @tensorflow/tfjs-node\n",
  );
}

// ─── Save results ─────────────────────────────────────────────────────────────

const suite: BenchmarkSuite = {
  name: "conv2d",
  description: "Inference-only Conv2D stack, batches 1/4/16",
  results: [isidorusResult, ...(tfjsResult ? [tfjsResult] : [])],
  comparisons,
};

const resultsDir = join(REPO_ROOT, "results", "conv2d");
mkdirSync(resultsDir, { recursive: true });

const filename = "conv2d.json";
const filePath = join(resultsDir, filename);

// Merge new results with existing ones instead of overwriting
const mergedSuite = mergeResults({
  filePath,
  newSuite: suite,
});

writeFileSync(filePath, JSON.stringify(mergedSuite, null, 2));
console.log(`Results saved → results/conv2d/${filename}`);
