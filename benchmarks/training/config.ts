// benchmarks/training/config.ts
// Shared config for isidorus + tfjs-node + Python training benchmarks.

export const INPUT_H = 56;
export const INPUT_W = 56;
export const INPUT_C = 3;
export const NUM_CLASSES = 10;
export const WARMUP_STEPS = parseInt(process.env.WARMUP_ITERS ?? "5", 10);
export const BENCH_STEPS = parseInt(process.env.BENCH_ITERS ?? "50", 10);
export const BATCH_SIZES = [1, 8, 32];
export const OPTIMIZER = process.env.OPTIMIZER ?? "adam"; // "adam" | "sgd"
export const LR = parseFloat(process.env.LR ?? "0.001");

export const MODEL_DESCRIPTION =
  `Conv2D(32,3,relu,SAME) → Conv2D(64,3,relu,SAME) → ` +
  `Conv2D(64,3,relu,VALID,s2) → Flatten → Dense(128,relu) → Dense(${NUM_CLASSES},softmax)`;
