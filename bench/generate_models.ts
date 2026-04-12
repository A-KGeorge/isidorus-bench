#!/usr/bin/env node
/**
 * bench/generate_models.ts
 *
 * isidorus equivalent of bench/generate_models.py.
 *
 * Generates frozen benchmark graphs (.pb) compatible with the TF C API,
 * using isidorus's beginner-friendly Model API factory functions.
 *
 * Matches tf.keras.applications API:
 *   - mobilenetv2(): MobileNetV2 224x224x3 -> 1000 classes
 *   - resnet50():    ResNet50 224x224x3 -> 1000 classes
 *   - Dense stack:   4096 -> 1000 classes, ~44M params
 *
 * All weights are randomly initialised — suitable for throughput benchmarking.
 *
 * Usage:
 *   node --import tsx bench/generate_models.ts
 *   node --import tsx bench/generate_models.ts --root bench/models --models small,medium
 */

import { writeFileSync, mkdirSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { Model, Dense, mobilenetv2, resnet50 } from "@isidorus/cpu";

// ── CLI args ──────────────────────────────────────────────────────────────────

const __dirname = fileURLToPath(new URL(".", import.meta.url));
const REPO_ROOT = join(__dirname, "..");

let ROOT = join(REPO_ROOT, "bench", "models");
let MODELS = ["small", "medium", "large"];

for (let i = 2; i < process.argv.length; i++) {
  if (process.argv[i] === "--root" && process.argv[i + 1])
    ROOT = process.argv[++i];
  if (process.argv[i] === "--models" && process.argv[i + 1])
    MODELS = process.argv[++i].split(",");
}

mkdirSync(ROOT, { recursive: true });

// ── Model builders ───────────────────────────────────────────────────────────

/**
 * MobileNetV2 (Keras API style)
 *
 * ~3.5M parameters, 13.3 MB fp32
 * Efficient mobile architecture with inverted residual blocks.
 */
function buildSmall(): Model {
  return mobilenetv2([224, 224, 3], 1000);
}

/**
 * ResNet50 (Keras API style)
 *
 * ~25.6M parameters, 101.9 MB fp32
 * Deep residual network, strong accuracy on ImageNet.
 */
function buildMedium(): Model {
  return resnet50([224, 224, 3], 1000);
}

/**
 * Dense stack
 *
 * ~44M parameters, 167.8 MB fp32
 * Simple fully-connected layers for throughput testing.
 */
function buildLarge(): Model {
  return new Model(
    [4096],
    [
      new Dense(4096, { activation: "gelu", name: "dense_1" }),
      new Dense(4096, { activation: "gelu", name: "dense_2" }),
      new Dense(2048, { activation: "gelu", name: "dense_3" }),
      new Dense(1000, { name: "output" }),
    ],
  );
}

// ── Build + export ────────────────────────────────────────────────────────────

const BUILDERS: Record<string, () => Model> = {
  small: buildSmall,
  medium: buildMedium,
  large: buildLarge,
};

const SPECS: Record<
  string,
  { pb: string; inputShape: number[]; description: string }
> = {
  small: {
    pb: "bench_small.pb",
    inputShape: [1, 224, 224, 3],
    description: "MobileNetV2 224x224x3",
  },
  medium: {
    pb: "bench_medium.pb",
    inputShape: [1, 224, 224, 3],
    description: "ResNet50 224x224x3",
  },
  large: {
    pb: "bench_large.pb",
    inputShape: [1, 4096],
    description: "Dense stack ~44M params",
  },
};

const unknown = MODELS.filter((m) => !BUILDERS[m]);
if (unknown.length) {
  console.error(`Unknown models: ${unknown}. Valid: small, medium, large`);
  process.exit(1);
}

async function main() {
  console.log(`\nGenerating ${MODELS.join(", ")} -> ${ROOT}\n`);
  const manifest: Record<string, unknown> = {
    generated_by: "bench/generate_models.ts",
    models: {},
  };

  for (const name of MODELS) {
    const spec = SPECS[name];
    console.log(`[${name}] Building model...`);
    const t0 = Date.now();

    const model = BUILDERS[name]();
    model.compile({ loss: "sparse_categorical_crossentropy" });

    // Show summary for the first model as a demo
    if (name === MODELS[0]) {
      model.summary();
    }

    const nParams = model.countParams();

    // Export to frozen .pb file
    const outPath = join(ROOT, spec.pb);
    await model.save(outPath);
    model.dispose();

    const mb = ((nParams * 4) / 1024 / 1024).toFixed(2);
    const ms = Date.now() - t0;

    // Get file size
    const { statSync } = await import("node:fs");
    const pbSize = statSync(outPath).size;

    console.log(
      `[${name}] OK  params=${nParams.toLocaleString()}  fp32~${mb}MB  pb=${(pbSize / 1024 / 1024).toFixed(1)}MB  time=${ms}ms`,
    );
    console.log(`[${name}]     -> ${outPath}`);

    (manifest.models as any)[name] = {
      frozen_pb: spec.pb,
      savedmodel_dir: null,
      input_op: "inputs",
      input_shape: spec.inputShape,
      params: nParams,
      estimated_fp32_mb: parseFloat(mb),
      description: spec.description,
    };
  }

  const manifestPath = join(REPO_ROOT, "bench", "manifest.json");
  writeFileSync(manifestPath, JSON.stringify(manifest, null, 2));
  console.log(`\nManifest -> ${manifestPath}`);
}

main().catch(console.error);
