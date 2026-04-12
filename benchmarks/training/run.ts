/**
 * benchmarks/training/run.ts
 *
 * Orchestrates @isidorus/cpu and @tensorflow/tfjs-node training benchmarks.
 * Saves results to results/training/<timestamp>-<platform>.json
 *
 * Usage:
 *   node --import tsx benchmarks/training/run.ts
 *   BATCH_SIZES=1,8,32 BENCH_ITERS=100 node --import tsx benchmarks/training/run.ts
 */

import { writeFileSync, mkdirSync, rmSync, readdirSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { availableParallelism } from "node:os";

import {
  BATCH_SIZES,
  BENCH_STEPS,
  WARMUP_STEPS,
  OPTIMIZER,
  LR,
  MODEL_DESCRIPTION,
} from "./config.js";
import { runIsidorusTrainingBench } from "./isidorus.js";
import { runTfjsNodeTrainingBench } from "./tfjs_node.js";

const __dirname = fileURLToPath(new URL(".", import.meta.url));
const REPO_ROOT = join(__dirname, "..", "..");

console.log(`
╔══════════════════════════════════════════════════════════════════════╗
║           @isidorus/cpu  Training Benchmark Suite                    ║
╚══════════════════════════════════════════════════════════════════════╝

  Model:      ${MODEL_DESCRIPTION}
  Optimizer:  ${OPTIMIZER.toUpperCase()}  lr=${LR}
  Warmup:     ${WARMUP_STEPS} steps     Bench: ${BENCH_STEPS} steps
  Batches:    ${BATCH_SIZES.join(", ")}
  Platform:   ${process.platform}-${process.arch}  Node: ${process.version}
  HW threads: ${availableParallelism()}
`);

const isidorusResult = await runIsidorusTrainingBench();
const tfjsResult = await runTfjsNodeTrainingBench();

// ── Speedup table ─────────────────────────────────────────────────────────────

if (tfjsResult) {
  const W = 72;
  console.log("─".repeat(W));
  console.log(" Training speedup: @isidorus/cpu  vs  @tensorflow/tfjs-node");
  console.log(" (samples/sec — higher is better)");
  console.log("─".repeat(W));
  console.log(
    ` ${"batch".padStart(8)}` +
      ` ${"isidorus samp/s".padStart(17)}` +
      ` ${"tfjs-node samp/s".padStart(18)}` +
      ` ${"speedup".padStart(10)}`,
  );
  console.log(" " + "─".repeat(W - 2));

  const tfjsMap = new Map(tfjsResult.batches.map((b) => [b.batchSize, b]));
  for (const ir of isidorusResult.batches) {
    const tr = tfjsMap.get(ir.batchSize);
    if (!tr) continue;
    const iSPS = (ir as any).samplesPerSec as number;
    const tSPS = (tr as any).samplesPerSec as number;
    const ratio = iSPS / tSPS;
    const mark = ratio >= 1 ? "✓" : "✗";
    console.log(
      ` ${String(ir.batchSize).padEnd(8)}` +
        ` ${iSPS.toFixed(1).padStart(15)} samp/s` +
        ` ${tSPS.toFixed(1).padStart(16)} samp/s` +
        ` ${(ratio.toFixed(2) + "×  " + mark).padStart(10)}`,
    );
  }
  console.log("─".repeat(W) + "\n");
}

// ── Save ─────────────────────────────────────────────────────────────────────

const outDir = join(REPO_ROOT, "results", "training");
mkdirSync(outDir, { recursive: true });

// Clean up old TypeScript-generated benchmark files (those without -python suffix)
try {
  const files = readdirSync(outDir);
  for (const file of files) {
    if (file.endsWith(".json") && !file.includes("-python")) {
      const filePath = join(outDir, file);
      rmSync(filePath);
    }
  }
} catch {
  // Directory might not exist yet, that's fine
}

const suite = {
  name: "training",
  description: `Training throughput — ${MODEL_DESCRIPTION}`,
  optimizer: OPTIMIZER,
  lr: LR,
  results: [isidorusResult, ...(tfjsResult ? [tfjsResult] : [])],
};

const filename = "training.json";
writeFileSync(join(outDir, filename), JSON.stringify(suite, null, 2));
console.log(`Results saved → results/training/${filename}`);
