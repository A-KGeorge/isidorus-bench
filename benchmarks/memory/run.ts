/**
 * benchmarks/memory/run.ts
 *
 * Memory leak profiler for InferencePool.
 *
 * Detects three categories of leak:
 *   1. V8 heap growth     — JS objects not collected (Napi::ObjectReference,
 *                           closures, accumulated arrays)
 *   2. External memory    — native TF_Tensor* or TF_Buffer* not freed by the
 *                           addon after each runAsync() completes
 *   3. RSS growth         — process-level footprint (includes both + mmap)
 *
 * Methodology:
 *   - Warm up the pool (oneDNN cache + V8 JIT stabilisation)
 *   - Run TOTAL_ITERS inference requests in a tight self-draining loop
 *   - Every SAMPLE_EVERY requests, call global.gc() then record memory
 *   - Fit a least-squares line through [heap, external, rss] vs iteration
 *   - Report bytes/iter slope; flag LEAK if slope > LEAK_THRESHOLD_BYTES_PER_ITER
 *
 * Requires --expose-gc so forced GC isolates leaks from collection lag.
 *
 * Usage:
 *   node --expose-gc --import tsx benchmarks/memory/run.ts <model.pb>
 *
 * The model path defaults to bench/models/bench_small.pb if omitted.
 * Always use bench_small (MobileNetV2 ~14MB) — large models dominate RSS
 * and obscure per-request residuals.
 */

import { performance } from "node:perf_hooks";
import { writeFileSync, mkdirSync } from "node:fs";
import { join, basename } from "node:path";
import { fileURLToPath } from "node:url";
import { availableParallelism } from "node:os";

import { InferencePool } from "@isidorus/cpu";
import { machineInfo } from "../../shared/stats.js";

const __dirname = fileURLToPath(new URL(".", import.meta.url));
const REPO_ROOT = join(__dirname, "..", "..");

// ── Config ────────────────────────────────────────────────────────────────────

// Always use the small model — see module doc.
const MODEL_PATH =
  process.argv[2] ?? join(REPO_ROOT, "bench", "models", "bench_small.pb");

const WARMUP_ITERS = parseInt(process.env.WARMUP_ITERS ?? "200", 10);
const TOTAL_ITERS = parseInt(process.env.TOTAL_ITERS ?? "5000", 10);
const SAMPLE_EVERY = parseInt(process.env.SAMPLE_EVERY ?? "100", 10);
const CONCURRENCY = parseInt(process.env.CONCURRENCY ?? "4", 10);

// A slope above this across the post-warmup samples is flagged as a leak.
// 512 bytes/iter ≈ one small JS object persisting per request, which is
// clearly wrong. Genuine per-request residuals are single-digit bytes.
const LEAK_THRESHOLD_BYTES_PER_ITER = 512;

// ── Validate --expose-gc ─────────────────────────────────────────────────────

if (typeof (global as any).gc !== "function") {
  console.error(`
  ERROR: global.gc() is not available.
  Run with --expose-gc:

    node --expose-gc --import tsx benchmarks/memory/run.ts [model.pb]
`);
  process.exit(1);
}

const forceGc = (): void => (global as any).gc();

// ── Snapshot type ─────────────────────────────────────────────────────────────

interface MemSample {
  iteration: number; // request count at snapshot time
  heapUsed: number; // V8 heap (bytes)
  external: number; // native memory visible to V8 (bytes)
  rss: number; // resident set size (bytes)
  arrayBuffers: number; // shared ArrayBuffer backing memory (bytes)
}

// ── Least-squares linear regression ──────────────────────────────────────────
// Returns bytes-per-iteration slope for the given series.

function lsSlope(
  samples: MemSample[],
  key: keyof Omit<MemSample, "iteration">,
): number {
  const n = samples.length;
  if (n < 2) return 0;
  const xs = samples.map((s) => s.iteration);
  const ys = samples.map((s) => s[key] as number);
  const xm = xs.reduce((a, b) => a + b, 0) / n;
  const ym = ys.reduce((a, b) => a + b, 0) / n;
  const num = xs.reduce((acc, x, i) => acc + (x - xm) * (ys[i] - ym), 0);
  const den = xs.reduce((acc, x) => acc + (x - xm) ** 2, 0);
  return den === 0 ? 0 : num / den;
}

// ── Formatting ────────────────────────────────────────────────────────────────

function fmtBytes(b: number): string {
  if (Math.abs(b) >= 1024 ** 2) return `${(b / 1024 ** 2).toFixed(2)} MB`;
  if (Math.abs(b) >= 1024) return `${(b / 1024).toFixed(1)} KB`;
  return `${b.toFixed(0)} B`;
}

function fmtSlope(slope: number): string {
  return `${slope >= 0 ? "+" : ""}${fmtBytes(slope)}/iter`;
}

function verdict(slope: number): string {
  return Math.abs(slope) < LEAK_THRESHOLD_BYTES_PER_ITER
    ? "✅ CLEAN"
    : "🚨 LEAK";
}

// ── Banner ────────────────────────────────────────────────────────────────────

console.log(`
╔══════════════════════════════════════════════════════════════════════╗
║           @isidorus/cpu  InferencePool  Memory Leak Profile          ║
╚══════════════════════════════════════════════════════════════════════╝

  Model:       ${basename(MODEL_PATH)}
  Warmup:      ${WARMUP_ITERS} requests (JIT + oneDNN stabilisation)
  Profile:     ${TOTAL_ITERS} requests total, sample every ${SAMPLE_EVERY}
  Concurrency: ${CONCURRENCY} concurrent callers
  HW threads:  ${availableParallelism()}
  Leak flag:   >${fmtBytes(LEAK_THRESHOLD_BYTES_PER_ITER)}/iter slope post-warmup
`);

// ── Create pool ───────────────────────────────────────────────────────────────

const pool = await InferencePool.create({
  modelPath: MODEL_PATH,
  profile: "latency",
});
// Use "latency" profile (maxConcurrent=1, all cores to one request) to
// isolate memory behaviour from queue mechanics. Switch to "throughput"
// or set maxConcurrent=CONCURRENCY once you confirm the baseline is clean.

const resolved = pool.resolvedInputShape;
const shape = resolved.map((d) => d ?? 1);
const nElems = shape.reduce((a, b) => a * b, 1);
const buf = Buffer.from(new Float32Array(nElems).buffer); // reuse same buffer every request
console.log(
  `  Input shape: ${JSON.stringify(resolved)}  (${nElems} floats, ${nElems * 4} bytes)\n`,
);

// ── Warmup ────────────────────────────────────────────────────────────────────

process.stdout.write(`  Warming up ${WARMUP_ITERS} requests...`);

{
  const inFlight = new Set<Promise<void>>();
  let issued = 0;
  while (issued < WARMUP_ITERS || inFlight.size > 0) {
    while (issued < WARMUP_ITERS && inFlight.size < CONCURRENCY) {
      let p: Promise<void>;
      p = pool.infer(buf, shape, 1).then(() => {
        inFlight.delete(p!);
      });
      inFlight.add(p);
      issued++;
    }
    if (inFlight.size > 0) await Promise.race(inFlight);
  }
}

// Force GC after warmup to clear any JIT-phase allocations, then record
// the post-warmup baseline. All subsequent samples are relative to this.
forceGc();
await new Promise((r) => setTimeout(r, 50)); // let GC finalizers run
forceGc();

const baseline = process.memoryUsage();
console.log(" done");
console.log(
  `  Baseline heap: ${fmtBytes(baseline.heapUsed)}  external: ${fmtBytes(baseline.external)}  RSS: ${fmtBytes(baseline.rss)}\n`,
);

// ── Profile loop ──────────────────────────────────────────────────────────────

const samples: MemSample[] = [];

console.log(
  `  ${"iter".padStart(6)}` +
    `  ${"heap".padStart(12)}` +
    `  ${"Δheap".padStart(10)}` +
    `  ${"external".padStart(12)}` +
    `  ${"Δexternal".padStart(10)}` +
    `  ${"RSS".padStart(12)}`,
);
console.log("  " + "─".repeat(72));

let totalIssued = 0;
let sampleCount = 0;

// Self-draining concurrent loop — same pattern as run.ts.
const inFlight = new Set<Promise<void>>();

async function drainOne(): Promise<void> {
  let p: Promise<void>;
  p = pool.infer(buf, shape, 1).then(() => {
    inFlight.delete(p!);
    totalIssued++;

    // After every SAMPLE_EVERY completed requests, take a snapshot.
    if (totalIssued % SAMPLE_EVERY === 0) {
      forceGc();
      const m = process.memoryUsage();
      const s: MemSample = {
        iteration: totalIssued,
        heapUsed: m.heapUsed,
        external: m.external,
        rss: m.rss,
        arrayBuffers: m.arrayBuffers,
      };
      samples.push(s);
      sampleCount++;

      const dHeap = m.heapUsed - baseline.heapUsed;
      const dExt = m.external - baseline.external;

      if (sampleCount % 5 === 0 || totalIssued === TOTAL_ITERS) {
        console.log(
          `  ${String(totalIssued).padStart(6)}` +
            `  ${fmtBytes(m.heapUsed).padStart(12)}` +
            `  ${fmtBytes(dHeap).padStart(10)}` +
            `  ${fmtBytes(m.external).padStart(12)}` +
            `  ${fmtBytes(dExt).padStart(10)}` +
            `  ${fmtBytes(m.rss).padStart(12)}`,
        );
      }
    }
  });
  inFlight.add(p);
}

const t0 = performance.now();

// Drive the loop until TOTAL_ITERS completed.
while (totalIssued < TOTAL_ITERS || inFlight.size > 0) {
  while (
    totalIssued + inFlight.size < TOTAL_ITERS &&
    inFlight.size < CONCURRENCY
  ) {
    drainOne();
  }
  if (inFlight.size > 0) await Promise.race(inFlight);
}

const wallMs = performance.now() - t0;
await pool.destroy();

// ── Analysis ──────────────────────────────────────────────────────────────────

// Split samples into early (first 20%) and late (remaining 80%) to separate
// stabilisation from steady-state. The leak verdict uses only the late window.
const splitAt = Math.floor(samples.length * 0.2);
const warmWindow = samples.slice(0, splitAt);
const lateWindow = samples.slice(splitAt);

const slopeHeap = lsSlope(lateWindow, "heapUsed");
const slopeExt = lsSlope(lateWindow, "external");
const slopeRss = lsSlope(lateWindow, "rss");

// ── Report ────────────────────────────────────────────────────────────────────

console.log(`\n  ${"─".repeat(72)}`);
console.log(
  `  Completed ${TOTAL_ITERS} requests in ${(wallMs / 1000).toFixed(1)}s  (${(TOTAL_ITERS / (wallMs / 1000)).toFixed(0)} req/s avg)\n`,
);

console.log("  Linear regression on late window (post-stabilisation):");
console.log(
  `    Heap (V8):       ${fmtSlope(slopeHeap).padEnd(20)}  ${verdict(slopeHeap)}`,
);
console.log(
  `    External (native):${fmtSlope(slopeExt).padEnd(20)} ${verdict(slopeExt)}`,
);
console.log(
  `    RSS:             ${fmtSlope(slopeRss).padEnd(20)}  ${verdict(slopeRss)}`,
);

const isClean = [slopeHeap, slopeExt, slopeRss].every(
  (s) => Math.abs(s) < LEAK_THRESHOLD_BYTES_PER_ITER,
);

console.log(
  `\n  Overall: ${isClean ? "✅  NO LEAK DETECTED" : "🚨  LEAK DETECTED"}`,
);

if (!isClean) {
  console.log(`
  Diagnosis hints:
    Heap growing?     → Check Napi::ObjectReference accumulation in addon.cc,
                        or JS closures capturing request data (e.g. samples[]).
    External growing? → TF_Tensor* or TF_Buffer* not freed after runAsync().
                        Check TF_DeleteTensor() calls in session.cc.
    RSS only growing? → Likely OS memory fragmentation from malloc / oneDNN
                        pool allocator — not a true leak if heap+ext are clean.
`);
}

// ── Per-iteration heap ceiling (practical bound) ──────────────────────────────

const lastSample = samples[samples.length - 1];
const firstSample = samples[0];
const totalHeapGrowth = lastSample.heapUsed - firstSample.heapUsed;
const totalExtGrowth = lastSample.external - firstSample.external;

console.log("  Total growth (first sample → last sample):");
console.log(`    Heap:     ${fmtBytes(totalHeapGrowth)}`);
console.log(`    External: ${fmtBytes(totalExtGrowth)}`);
console.log(
  `    Bytes per request (heap):     ${fmtBytes(totalHeapGrowth / TOTAL_ITERS)}`,
);
console.log(
  `    Bytes per request (external): ${fmtBytes(totalExtGrowth / TOTAL_ITERS)}`,
);

// ── Save results ──────────────────────────────────────────────────────────────

const ts = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
const modelName = basename(MODEL_PATH).replace(/\.[^/.]+$/, "");
const outDir = join(REPO_ROOT, "results", "memory", modelName);
mkdirSync(outDir, { recursive: true });

const filename = `${ts}-${process.platform}-${process.arch}.json`;
const result = {
  model: basename(MODEL_PATH),
  inputShape: resolved,
  warmupIters: WARMUP_ITERS,
  totalIters: TOTAL_ITERS,
  sampleEvery: SAMPLE_EVERY,
  concurrency: CONCURRENCY,
  wallMs,
  baseline: {
    heapUsed: baseline.heapUsed,
    external: baseline.external,
    rss: baseline.rss,
  },
  samples,
  regression: {
    window: `iterations ${samples[splitAt]?.iteration ?? 0}–${TOTAL_ITERS}`,
    heapSlope: slopeHeap,
    extSlope: slopeExt,
    rssSlope: slopeRss,
    threshold: LEAK_THRESHOLD_BYTES_PER_ITER,
    clean: isClean,
  },
  machineInfo: machineInfo(),
  timestamp: new Date().toISOString(),
};

writeFileSync(join(outDir, filename), JSON.stringify(result, null, 2));
console.log(`\n  Results saved → results/memory/${modelName}/${filename}`);

process.exit(isClean ? 0 : 1);
