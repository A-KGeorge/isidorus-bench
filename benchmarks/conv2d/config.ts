// ─── Conv2D benchmark configuration ─────────────────────────────────────────
//
// All tunable parameters live here so both isidorus.ts and tfjs_node.ts
// build identical models and use identical timing parameters.
//
// Override via environment variables for quick runs:
//   BENCH_ITERS=50 WARMUP_ITERS=10 npm run bench:conv2d:quick

export const INPUT_H = 56;
export const INPUT_W = 56;
export const INPUT_C = 3;
export const NUM_CLASSES = 10;
export const WARMUP_ITERS = parseInt(process.env.WARMUP_ITERS ?? "20", 10);
export const BENCH_ITERS = parseInt(process.env.BENCH_ITERS ?? "200", 10);
export const BATCH_SIZES = [1, 4, 16];

// ─── Model description ────────────────────────────────────────────────────────
//
// ConvNet — same architecture in both runtimes:
//   Conv2D(32, 3×3, relu, SAME)
//   Conv2D(64, 3×3, relu, SAME)
//   Conv2D(64, 3×3, relu, VALID, stride=2)   ← spatial reduction without MaxPool
//   Flatten
//   Dense(128, relu)
//   Dense(10,  softmax)
//
// Why 56×56 instead of 224×224:
//   224×224×3 = 150K floats per image. At batch=16 that's 9.6 MB of input data
//   and a single forward pass takes 2–5 seconds on a laptop CPU — too slow for
//   a 200-iteration benchmark. 56×56 is still compute-bound (Conv2D dominates)
//   while keeping total suite time under 60 seconds on a typical dev machine.
//   You can change INPUT_H/W to 224 to get full ImageNet-sized numbers on a
//   server-class CPU.

export const MODEL_DESCRIPTION =
  `Conv2D(32,3,relu,SAME) → Conv2D(64,3,relu,SAME) → ` +
  `Conv2D(64,3,relu,VALID,s2) → Flatten → Dense(128,relu) → Dense(${NUM_CLASSES},softmax)`;
