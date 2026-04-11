#!/usr/bin/env node
/**
 * bench/generate_models.ts
 *
 * isidorus equivalent of bench/generate_models.py.
 *
 * Generates frozen benchmark graphs (.pb) compatible with the TF C API,
 * using isidorus's arch building blocks — no Python, no Keras, no SavedModel.
 *
 * Models:
 *   small:  MobileNetV2   (224x224x3 -> 1000 classes)
 *   medium: ResNet50      (224x224x3 -> 1000 classes)
 *   large:  Dense stack   (4096      -> 1000 classes, ~44M params)
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

import type { Tensor } from "@isidorus/core";
import { DType } from "@isidorus/core";
import {
  Graph,
  getAddon,
  constGlorot,
  constZeros,
  convBnRelu,
  convBnRelu6,
  residualBlock,
  projectionBlock,
  invertedResidual,
  maxPool,
  globalAvgPool,
  softmax,
  gelu,
  matmul,
  biasAdd,
} from "@isidorus/cpu";

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

function newGraph(): Graph {
  return new Graph(new (getAddon().Graph)());
}

// ── bench_large: Dense stack ~44M params ─────────────────────────────────────

function buildLarge(): { g: Graph; nParams: number } {
  const g = newGraph();
  const [x] = g.addOp(
    "Placeholder",
    [],
    {
      dtype: { kind: "type", value: DType.FLOAT32 },
      shape: { kind: "shape", value: [1, 4096] },
    },
    "inputs",
  );

  let nParams = 0;
  function dense(
    input: Tensor,
    inF: number,
    outF: number,
    activation: "gelu" | "none",
    name: string,
  ): Tensor {
    const w = constGlorot(g, [inF, outF], `${name}/w`);
    const b = constZeros(g, outF, `${name}/b`);
    nParams += inF * outF + outF;
    const y = biasAdd(
      g,
      matmul(g, input, w, {}, `${name}/mm`),
      b,
      `${name}/ba`,
    );
    if (activation === "none") return y;
    return gelu(g, y, `${name}/gelu`);
  }

  let h = dense(x, 4096, 4096, "gelu", "dense_1");
  h = dense(h, 4096, 4096, "gelu", "dense_2");
  h = dense(h, 4096, 2048, "gelu", "dense_3");
  const out = dense(h, 2048, 1000, "none", "output");
  g.addOp("Identity", [out], {}, "output_identity");
  return { g, nParams };
}

// ── bench_medium: ResNet50 ────────────────────────────────────────────────────

function buildMedium(): { g: Graph; nParams: number } {
  const g = newGraph();
  const [x] = g.addOp(
    "Placeholder",
    [],
    {
      dtype: { kind: "type", value: DType.FLOAT32 },
      shape: { kind: "shape", value: [1, 224, 224, 3] },
    },
    "inputs",
  );

  let nParams = 0;
  const track = (n: number) => {
    nParams += n;
  };

  let [h, c] = convBnRelu(g, x, 3, 64, 7, 2, "SAME", "conv1");
  track(3 * 7 * 7 * 64 + 64 * 4);
  h = maxPool(g, h, [1, 3, 3, 1], [1, 2, 2, 1], "SAME", "pool1");

  [h, c] = projectionBlock(g, h, 64, [64, 64, 256], 1, "res2a");
  track(64 * 3 + 256 * 3);
  [h, c] = residualBlock(g, h, 256, [64, 64, 256], "res2b");
  [h, c] = residualBlock(g, h, 256, [64, 64, 256], "res2c");

  [h, c] = projectionBlock(g, h, 256, [128, 128, 512], 2, "res3a");
  track(128 * 3 + 512 * 3);
  for (const n of ["res3b", "res3c", "res3d"])
    [h, c] = residualBlock(g, h, 512, [128, 128, 512], n);

  [h, c] = projectionBlock(g, h, 512, [256, 256, 1024], 2, "res4a");
  track(256 * 3 + 1024 * 3);
  for (let i = 1; i <= 5; i++)
    [h, c] = residualBlock(g, h, 1024, [256, 256, 1024], `res4b${i}`);

  [h, c] = projectionBlock(g, h, 1024, [512, 512, 2048], 2, "res5a");
  track(512 * 3 + 2048 * 3);
  [h, c] = residualBlock(g, h, 2048, [512, 512, 2048], "res5b");
  [h, c] = residualBlock(g, h, 2048, [512, 512, 2048], "res5c");

  h = globalAvgPool(g, h, "avg_pool");
  const wOut = constGlorot(g, [2048, 1000], "fc/w");
  const bOut = constZeros(g, 1000, "fc/b");
  const out = biasAdd(g, matmul(g, h, wOut, {}, "fc/mm"), bOut, "fc/ba");
  track(2048 * 1000 + 1000);
  g.addOp("Identity", [out], {}, "output_identity");
  return { g, nParams };
}

// ── bench_small: MobileNetV2 ──────────────────────────────────────────────────

function buildSmall(): { g: Graph; nParams: number } {
  const g = newGraph();
  const [x] = g.addOp(
    "Placeholder",
    [],
    {
      dtype: { kind: "type", value: DType.FLOAT32 },
      shape: { kind: "shape", value: [1, 224, 224, 3] },
    },
    "inputs",
  );

  let nParams = 0;
  const track = (n: number) => {
    nParams += n;
  };

  // [expand_ratio, out_channels, num_blocks, stride]
  const irTable: [number, number, number, number][] = [
    [1, 16, 1, 1],
    [6, 24, 2, 2],
    [6, 32, 3, 2],
    [6, 64, 4, 2],
    [6, 96, 3, 1],
    [6, 160, 3, 2],
    [6, 320, 1, 1],
  ];

  let [h, c] = convBnRelu6(g, x, 3, 32, 3, 2, "conv_stem");
  track(3 * 3 * 3 * 32 + 32 * 4);

  let blockIdx = 0;
  for (const [t, outC, n, s] of irTable) {
    for (let i = 0; i < n; i++) {
      [h, c] = invertedResidual(
        g,
        h,
        c,
        outC,
        i === 0 ? s : 1,
        t,
        `ir_${blockIdx++}`,
      );
      track(c * t * 9 + c * t * outC + (outC + c) * 4);
    }
  }

  [h, c] = convBnRelu6(g, h, 320, 1280, 1, 1, "conv_head");
  track(320 * 1280 + 1280 * 4);
  h = globalAvgPool(g, h, "avg_pool");
  const wOut = constGlorot(g, [1280, 1000], "classifier/w");
  const bOut = constZeros(g, 1000, "classifier/b");
  track(1280 * 1000 + 1000);
  const logits = biasAdd(
    g,
    matmul(g, h, wOut, {}, "classifier/mm"),
    bOut,
    "classifier/ba",
  );
  const out = softmax(g, logits, "predictions");
  g.addOp("Identity", [out], {}, "output_identity");
  return { g, nParams };
}

// ── Build + export ────────────────────────────────────────────────────────────

const BUILDERS: Record<string, () => { g: Graph; nParams: number }> = {
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

console.log(`\nGenerating ${MODELS.join(", ")} -> ${ROOT}\n`);
const manifest: Record<string, unknown> = {
  generated_by: "bench/generate_models.ts",
  models: {},
};

for (const name of MODELS) {
  const spec = SPECS[name];
  console.log(`[${name}] Building graph...`);
  const t0 = Date.now();
  const { g, nParams } = BUILDERS[name]();
  const graphDef = g.toGraphDef();
  const outPath = join(ROOT, spec.pb);
  writeFileSync(outPath, graphDef);
  const mb = ((nParams * 4) / 1024 / 1024).toFixed(2);
  const ms = Date.now() - t0;
  console.log(
    `[${name}] OK  params=${nParams.toLocaleString()}  fp32~${mb}MB  pb=${(graphDef.byteLength / 1024 / 1024).toFixed(1)}MB  time=${ms}ms`,
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
