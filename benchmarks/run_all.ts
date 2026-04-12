import { spawnSync } from "node:child_process";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = join(fileURLToPath(import.meta.url), "../..");

const MODELS = [
  "bench/models/bench_small.pb",
  "bench/models/bench_medium.pb",
  "bench/models/bench_large.pb",
];

const PROFILES = ["", "latency", "throughput"];

const pythonCmd = process.platform === "win32" ? "python" : "python3";

// Use Python from venv for all Python benchmarks
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

  // Ensure Node commands run without venv activation to avoid affecting output
  const spawnOptions: any = { stdio: "inherit", cwd: ROOT };
  if (isNodeCommand) {
    // Remove any venv-related environment variables for Node commands
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

// 1. Inference Pool (TypeScript)
for (const model of MODELS) {
  for (const profile of PROFILES) {
    const args = ["--import", "tsx", "benchmarks/inference_pool/run.ts", model];
    if (profile) {
      args.push("--profile", profile);
    }
    runCommand("node", args, true);
  }
}

// Wait for CPU to settle before next suite
console.log("\nResting for 2 seconds...");
await new Promise((resolve) => setTimeout(resolve, 2000));

// 3. Memory (TypeScript)
for (const model of MODELS) {
  runCommand(
    "node",
    ["--expose-gc", "--import", "tsx", "benchmarks/memory/run.ts", model],
    true,
  );
}

// Wait for CPU to settle before next suite
console.log("\nResting for 2 seconds...");
await new Promise((resolve) => setTimeout(resolve, 2000));

// 4. Conv2d (TypeScript) (No model size)
runCommand("node", ["--import", "tsx", "benchmarks/conv2d/run.ts"], true);

// Wait for CPU to settle before next suite
console.log("\nResting for 2 seconds...");
await new Promise((resolve) => setTimeout(resolve, 2000));

// 5. Training (TypeScript) (No model size)
runCommand("node", ["--import", "tsx", "benchmarks/training/run.ts"], true);

// Wait for CPU to settle before Python tests
console.log("\nResting for 2 seconds...");
await new Promise((resolve) => setTimeout(resolve, 2000));

// 2. Inference Pool (Python) - using venv
for (const model of MODELS) {
  for (const profile of PROFILES) {
    const args = ["benchmarks/inference_pool/run.py", model];
    if (profile) {
      args.push("--profile", profile);
    }
    runCommand(venvPythonPath, args);
  }
}

// Wait for CPU to settle before next Python suite
console.log("\nResting for 2 seconds...");
await new Promise((resolve) => setTimeout(resolve, 2000));

// 6. Training (Python) (No model size)
runCommand(venvPythonPath, ["benchmarks/training/run.py"]);

console.log("\nAll benchmarks completed successfully!");
