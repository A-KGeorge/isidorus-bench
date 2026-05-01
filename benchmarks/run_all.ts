import { spawnSync } from "node:child_process";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { rmSync, existsSync } from "node:fs";

const ROOT = join(fileURLToPath(import.meta.url), "../..");

// Clean up all previous results before starting
const resultsDir = join(ROOT, "results");
try {
  rmSync(resultsDir, { recursive: true, force: true });
  console.log("Cleaned up previous results directory");
} catch {
  // Directory might not exist yet, that's fine
}

const MODELS = [
  "bench/models/bench_small.pb",
  "bench/models/bench_medium.pb",
  "bench/models/bench_large.pb",
  "bench/models/bert_model.pb",
];

const PROFILES = ["", "latency", "throughput"];

const venvPythonPath =
  process.platform === "win32"
    ? join(ROOT, "myenv", "Scripts", "python.exe")
    : join(ROOT, "myenv", "bin", "python");

function runCommand(
  command: string,
  args: string[],
  isNodeCommand: boolean = false,
) {
  console.log(`\n\n=========================================`);
  console.log(`> ${command} ${args.join(" ")}`);
  console.log(`=========================================\n`);

  const spawnOptions: any = { stdio: "inherit", cwd: ROOT };
  if (isNodeCommand) {
    const cleanEnv = { ...process.env };
    delete cleanEnv.VIRTUAL_ENV;
    delete cleanEnv.CONDA_PREFIX;
    spawnOptions.env = cleanEnv;
  }

  const result = spawnSync(command, args, spawnOptions);
  if (result.status !== 0) {
    console.error(`Command failed with status ${result.status}`);
    process.exit(result.status ?? 1);
  }
}

// ── Convert .pb models to .onnx for onnxruntime-node ─────────────────────────
// Only runs if the .onnx doesn't already exist and tf2onnx is available.
// Failure is non-fatal — run.ts will skip ORT and print a message.
console.log(
  "\n── Converting .pb models to .onnx (requires tf2onnx) ────────────────",
);
for (const model of MODELS) {
  const onnxPath = model.replace(/\.pb$/, ".onnx");
  if (existsSync(join(ROOT, onnxPath))) {
    console.log(`  Skipping ${onnxPath} (already exists)`);
    continue;
  }
  const result = spawnSync(
    venvPythonPath,
    ["bench/convert_to_onnx.py", model],
    { stdio: "inherit", cwd: ROOT },
  );
  if (result.status !== 0) {
    console.warn(
      `  ⚠  ONNX conversion failed for ${model} — ORT benchmark will be skipped`,
    );
  }
}

// ── Inference Pool (TypeScript — includes tfjs-node + onnxruntime-node) ──────
for (const model of MODELS) {
  for (const profile of PROFILES) {
    const args = ["--import", "tsx", "benchmarks/inference_pool/run.ts", model];
    if (profile) args.push("--profile", profile);
    runCommand("node", args, true);
  }
}

console.log("\nResting for 2 seconds...");
await new Promise((resolve) => setTimeout(resolve, 2000));

// ── Memory (TypeScript) ───────────────────────────────────────────────────────
for (const model of MODELS) {
  runCommand(
    "node",
    ["--expose-gc", "--import", "tsx", "benchmarks/memory/run.ts", model],
    true,
  );
}

console.log("\nResting for 2 seconds...");
await new Promise((resolve) => setTimeout(resolve, 2000));

// ── Conv2D (TypeScript) ───────────────────────────────────────────────────────
runCommand("node", ["--import", "tsx", "benchmarks/conv2d/run.ts"], true);

console.log("\nResting for 2 seconds...");
await new Promise((resolve) => setTimeout(resolve, 2000));

// ── Training (TypeScript) ─────────────────────────────────────────────────────
runCommand("node", ["--import", "tsx", "benchmarks/training/run.ts"], true);

console.log("\nResting for 2 seconds...");
await new Promise((resolve) => setTimeout(resolve, 2000));

// ── Inference Pool — Python threaded baseline ─────────────────────────────────
for (const model of MODELS) {
  for (const profile of PROFILES) {
    const args = ["benchmarks/inference_pool/run.py", model];
    if (profile) args.push("--profile", profile);
    runCommand(venvPythonPath, args);
  }
}

console.log("\nResting for 2 seconds...");
await new Promise((resolve) => setTimeout(resolve, 2000));

// ── Inference Pool — Python asyncio ──────────────────────────────────────────
// This is the key new baseline: same TF inference, asyncio event loop instead
// of raw threads. Measures asyncio event loop responsiveness during inference —
// the Python equivalent of what isidorus measures for Node.js.
for (const model of MODELS) {
  for (const profile of PROFILES) {
    const args = ["benchmarks/inference_pool/run_asyncio.py", model];
    if (profile) args.push("--profile", profile);
    runCommand(venvPythonPath, args);
  }
}

console.log("\nResting for 2 seconds...");
await new Promise((resolve) => setTimeout(resolve, 2000));

// ── Training (Python) ─────────────────────────────────────────────────────────
runCommand(venvPythonPath, ["benchmarks/training/run.py"]);

console.log("\nAll benchmarks completed successfully!");
