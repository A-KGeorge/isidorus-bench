/**
 * benchmarks/inference_pool/run_new_baselines.ts
 *
 * Runs only the new baselines (onnxruntime-node + asyncio Python) and merges
 * results into the existing inference_pool.json files — without re-running
 * Isidorus, tfjs-node, or Python TF threads.
 *
 * Prerequisites:
 *   npm install onnxruntime-node
 *   pip install tf2onnx onnx          (in your venv)
 *   python bench/convert_to_onnx.py bench/models/bert_model.pb
 *   python bench/convert_to_onnx.py bench/models/bench_small.pb
 *   # ... etc for each model you want to benchmark
 *
 * Usage:
 *   node --import tsx benchmarks/inference_pool/run_new_baselines.ts <model.pb> [--profile auto|latency|throughput] [--skip-ort]
 *
 * Example (all models, all profiles):
 *   for model in bench/models/*.pb; do
 *     for profile in auto latency throughput; do
 *       node --import tsx benchmarks/inference_pool/run_new_baselines.ts "$model" --profile "$profile"
 *     done
 *   done
 *
 * Example (skip ORT for bench_large due to unsupported operators):
 *   node --import tsx benchmarks/inference_pool/run_new_baselines.ts bench/models/bench_large.pb --skip-ort --profile auto
 */

import { spawnSync } from "node:child_process";
import { existsSync, writeFileSync, mkdirSync } from "node:fs";
import { join, basename } from "node:path";
import { fileURLToPath } from "node:url";
import { availableParallelism } from "node:os";

import { DType } from "@isidorus/cpu";
import { machineInfo, mergeResults } from "../../shared/stats.js";
import { runOnnxRuntimeNodeBench } from "./onnxruntime_node.js";

const __dirname = fileURLToPath(new URL(".", import.meta.url));
const REPO_ROOT = join(__dirname, "..", "..");

// ─── Args ─────────────────────────────────────────────────────────────────────

// Find the model path (first argument that is a file)
let modelPath = "";
for (let i = 2; i < process.argv.length; i++) {
  if (existsSync(process.argv[i])) {
    modelPath = process.argv[i];
    break;
  }
}

if (!modelPath) {
  console.error(
    "Usage: node --import tsx run_new_baselines.ts <model.pb> [--profile auto|latency|throughput] [--skip-ort]",
  );
  process.exit(1);
}

const profileArg = process.argv.indexOf("--profile");
const profile: "auto" | "latency" | "throughput" =
  profileArg !== -1 && process.argv[profileArg + 1]
    ? (process.argv[profileArg + 1] as any)
    : "auto";

const skipOrt = process.argv.includes("--skip-ort");

// Auto-detect .onnx alongside the .pb
// Note: ONNX Runtime Node only supports ONNX format, not TensorFlow SavedModel
const onnxPath = modelPath.replace(/\.pb$/, ".onnx");
const hasOnnx = existsSync(onnxPath);

// Detect Python venv
const venvPython =
  process.platform === "win32"
    ? join(REPO_ROOT, "myenv", "Scripts", "python.exe")
    : join(REPO_ROOT, "myenv", "bin", "python");
const pythonCmd = existsSync(venvPython) ? venvPython : "python3";

const modelName = basename(modelPath).replace(/\.pb$/, "");
const resultsDir = join(REPO_ROOT, "results", "inference_pool", modelName);
const filePath = join(resultsDir, "inference_pool.json");

console.log(`
╔══════════════════════════════════════════════════════════════════════╗
║      New Baselines: onnxruntime-node + Python asyncio               ║
╚══════════════════════════════════════════════════════════════════════╝

  Model:    ${basename(modelPath)}
  Profile:  ${profile}
  ORT:      ${skipOrt ? "SKIPPED (--skip-ort)" : hasOnnx ? onnxPath : "NOT FOUND — skipping (convert with bench/convert_to_onnx.py)"}
  Python:   ${pythonCmd}
  Results → ${filePath}
`);

// ─── onnxruntime-node ─────────────────────────────────────────────────────────

if (hasOnnx && !skipOrt) {
  console.log(
    "── onnxruntime-node ────────────────────────────────────────────────────\n",
  );

  // BERT uses int32 inputs; everything else is float32.
  // Heuristic: check if the model name contains "bert".
  const ortInputDtype: "float32" | "int32" = modelPath
    .toLowerCase()
    .includes("bert")
    ? "int32"
    : "float32";

  // Use intraOpThreads=4 to match what InferencePool's throughput profile
  // autotuner typically lands on for a 24-core machine. Change if needed.
  const intraOpThreads = parseInt(process.env.ORT_INTRA_THREADS ?? "4", 10);

  const ortResult = await runOnnxRuntimeNodeBench(
    onnxPath,
    // inferShape: for BERT [1, 128], otherwise infer from model metadata.
    // The function builds its own feed from session.inputNames, so this shape
    // is only used as a fallback for single-input models.
    modelPath.toLowerCase().includes("bert") ? [1, 128] : [1, 224, 224, 3],
    ortInputDtype,
    intraOpThreads,
    profile,
  );

  if (ortResult) {
    mkdirSync(resultsDir, { recursive: true });
    const suite = {
      name: "inference_pool",
      description: `Worker-pool throughput benchmark — ${basename(modelPath)}`,
      results: [ortResult],
      comparisons: [],
    };
    const merged = mergeResults({ filePath, newSuite: suite, profile });
    writeFileSync(filePath, JSON.stringify(merged, null, 2));
    console.log(`\nORT results merged → ${filePath}\n`);
  }
} else {
  if (skipOrt) {
    console.log(`Skipping onnxruntime-node — --skip-ort flag set.\n`);
  } else {
    console.log(`Skipping onnxruntime-node — ${onnxPath} not found.`);
    console.log(`Convert with: python bench/convert_to_onnx.py ${modelPath}\n`);
  }
}

// ─── Python asyncio ───────────────────────────────────────────────────────────

console.log(
  "── tensorflow-python (asyncio) ─────────────────────────────────────────\n",
);

const asyncioArgs = [
  "benchmarks/inference_pool/run_asyncio.py",
  modelPath,
  "--profile",
  profile,
];

const result = spawnSync(pythonCmd, asyncioArgs, {
  stdio: "inherit",
  cwd: REPO_ROOT,
});

if (result.status !== 0) {
  console.error(`\n⚠  asyncio benchmark failed with status ${result.status}`);
  process.exit(result.status ?? 1);
}

console.log("\nDone.");
