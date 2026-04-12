import os from "node:os";
import { performance } from "node:perf_hooks";
import { existsSync, readFileSync } from "node:fs";
import type {
  IterStats,
  MachineInfo,
  BatchResult,
  BenchmarkSuite,
} from "./types.js";

// ─── Statistics ─────────────────────────────────────────────────────────────

export function computeStats(samples: number[]): IterStats {
  const s = [...samples].sort((a, b) => a - b);
  const sum = s.reduce((a, b) => a + b, 0);
  return {
    mean: sum / s.length,
    p50: pct(s, 50),
    p95: pct(s, 95),
    p99: pct(s, 99),
    min: s[0],
    max: s[s.length - 1],
  };
}

function pct(sorted: number[], p: number): number {
  const idx = Math.ceil((p / 100) * sorted.length) - 1;
  return sorted[Math.max(0, idx)];
}

export function batchResult(batchSize: number, samples: number[]): BatchResult {
  const latency = computeStats(samples);
  const throughput = (batchSize * 1000) / latency.mean;
  return { batchSize, latency, throughput };
}

// ─── Machine info ────────────────────────────────────────────────────────────

export function machineInfo(): MachineInfo {
  const cpus = os.cpus();
  return {
    platform: process.platform,
    arch: process.arch,
    nodeVersion: process.version,
    cpus: os.availableParallelism(),
    cpuModel: cpus[0]?.model ?? "unknown",
  };
}

// ─── Timing helpers ──────────────────────────────────────────────────────────

export async function runTimed(fn: () => Promise<void>): Promise<number> {
  const t0 = performance.now();
  await fn();
  return performance.now() - t0;
}

// How often to yield to the macrotask queue during sample collection.
// Every YIELD_INTERVAL iterations we await a setImmediate so:
//   1. The setInterval stall monitor gets a chance to fire and record lag.
//   2. Other pending macrotasks (I/O, timers) can run between iterations.
//
// Why setImmediate and not setTimeout(0):
//   setImmediate fires after I/O callbacks in the current event loop tick,
//   before timers — it yields the minimum amount while still crossing the
//   macrotask boundary so pending setInterval callbacks can execute.
//
// Yield interval = 1 means one macrotask yield per iteration. This is
// correct for async workloads (isidorus runAsync) where each inference
// already crosses the macrotask boundary naturally. For sync workloads
// (collectSamples wrapping synchronous code) a larger interval reduces
// overhead while still giving the timer enough resolution.
const YIELD_INTERVAL = 1;

export async function collectSamples(
  iters: number,
  fn: () => Promise<void>,
): Promise<number[]> {
  const samples: number[] = [];
  for (let i = 0; i < iters; i++) {
    samples.push(await runTimed(fn));
    // Yield to the macrotask queue so setInterval can fire and record stall.
    // Without this, all iterations run as microtask continuations and the
    // event loop health monitor sees ticks=0 regardless of actual blocking.
    if ((i + 1) % YIELD_INTERVAL === 0)
      await new Promise<void>((resolve) => setImmediate(resolve));
  }
  return samples;
}

// ─── Pretty printing ─────────────────────────────────────────────────────────

export function fmtMs(ms: number): string {
  return ms.toFixed(2) + " ms";
}

export function fmtTps(tps: number): string {
  return tps.toFixed(0) + " img/s";
}

const COL = { batch: 8, ms: 11, tps: 12 };

export function printHeader(title: string) {
  const w = 72;
  console.log("\n" + "─".repeat(w));
  console.log(` ${title}`);
  console.log("─".repeat(w));
  console.log(
    ` ${"batch".padEnd(COL.batch)}` +
      ` ${"mean".padStart(COL.ms)}` +
      ` ${"p50".padStart(COL.ms)}` +
      ` ${"p95".padStart(COL.ms)}` +
      ` ${"p99".padStart(COL.ms)}` +
      ` ${"throughput".padStart(COL.tps)}`,
  );
  console.log(" " + "─".repeat(w - 2));
}

export function printBatchRow(r: BatchResult) {
  console.log(
    ` ${String(r.batchSize).padEnd(COL.batch)}` +
      ` ${fmtMs(r.latency.mean).padStart(COL.ms)}` +
      ` ${fmtMs(r.latency.p50).padStart(COL.ms)}` +
      ` ${fmtMs(r.latency.p95).padStart(COL.ms)}` +
      ` ${fmtMs(r.latency.p99).padStart(COL.ms)}` +
      ` ${fmtTps(r.throughput).padStart(COL.tps)}`,
  );
}

export function printFooter() {
  console.log("─".repeat(72) + "\n");
}

export function printSpeedupTable(
  baseline: { name: string; batches: BatchResult[] },
  candidate: { name: string; batches: BatchResult[] },
) {
  const w = 72;
  console.log("\n" + "─".repeat(w));
  console.log(` Speedup: ${candidate.name}  vs  ${baseline.name}`);
  console.log(
    ` (mean latency — lower is better for latency, >1× = candidate faster)`,
  );
  console.log("─".repeat(w));
  console.log(
    ` ${"batch".padEnd(COL.batch)}` +
      ` ${candidate.name.slice(0, 16).padStart(18)}` +
      ` ${baseline.name.slice(0, 16).padStart(18)}` +
      ` ${"speedup".padStart(10)}`,
  );
  console.log(" " + "─".repeat(w - 2));

  const bMap = new Map(baseline.batches.map((b) => [b.batchSize, b]));
  for (const cr of candidate.batches) {
    const br = bMap.get(cr.batchSize);
    if (!br) continue;
    const speedup = br.latency.mean / cr.latency.mean;
    const marker = speedup >= 1.0 ? "✓" : "✗";
    console.log(
      ` ${String(cr.batchSize).padEnd(COL.batch)}` +
        ` ${fmtMs(cr.latency.mean).padStart(18)}` +
        ` ${fmtMs(br.latency.mean).padStart(18)}` +
        ` ${(speedup.toFixed(2) + "×  " + marker).padStart(10)}`,
    );
  }
  console.log("─".repeat(w) + "\n");
}

// ─── Event loop health ───────────────────────────────────────────────────────
//
// Measures event loop stall during a concurrent workload.
//
// A "stall" is how late a scheduled timer fires relative to its expected time.
// A healthy async runtime has stall ≈ 0ms. A blocking synchronous call (like
// tfjs-node's predict()) produces stall ≈ blocking_duration per timer tick
// that falls inside the blocked window.
//
// Usage:
//   const monitor = startEventLoopMonitor();
//   // ... run workload ...
//   const health = monitor.stop();
//   console.log(`max stall: ${health.maxStallMs.toFixed(2)}ms`);

export interface EventLoopHealth {
  /** Number of timer ticks observed during the measurement window. */
  ticks: number;
  /** Mean stall per tick (ms). Near 0 for fully async workloads. */
  meanStallMs: number;
  /** Maximum stall observed (ms). Equals blocking call duration for sync APIs. */
  maxStallMs: number;
  /** 99th percentile stall (ms). */
  p99StallMs: number;
  /** Total wall time of the measurement window (ms). */
  durationMs: number;
  /** Fraction of ticks with stall > 5ms. */
  stallFraction: number;
}

export interface EventLoopMonitor {
  stop: () => EventLoopHealth;
}

/**
 * startEventLoopMonitor — begin measuring event loop stall.
 *
 * Schedules a setInterval at tickMs (default 5ms). On each tick it records
 * how late the callback fired relative to the previous tick. Any excess over
 * tickMs is a stall — time the event loop was occupied with synchronous work.
 *
 * @param tickMs  Timer interval in ms. Lower = finer resolution, higher CPU cost.
 */
export function startEventLoopMonitor(tickMs = 5): EventLoopMonitor {
  const stalls: number[] = [];
  let lastTick = performance.now();
  const start = lastTick;

  const handle = setInterval(() => {
    const now = performance.now();
    const delta = now - lastTick;
    // Stall = how much longer than expected the tick took.
    // We subtract tickMs and floor at 0 — small natural jitter is not a stall.
    const stall = Math.max(0, delta - tickMs);
    stalls.push(stall);
    lastTick = now;
  }, tickMs);

  // setInterval keeps the event loop alive — unref so it doesn't prevent exit.
  if (handle.unref) handle.unref();

  return {
    stop(): EventLoopHealth {
      clearInterval(handle);
      const durationMs = performance.now() - start;

      if (stalls.length === 0) {
        return {
          ticks: 0,
          meanStallMs: 0,
          maxStallMs: 0,
          p99StallMs: 0,
          durationMs,
          stallFraction: 0,
        };
      }

      const sorted = [...stalls].sort((a, b) => a - b);
      const mean = sorted.reduce((a, b) => a + b, 0) / sorted.length;
      const p99 = pct(sorted, 99);
      const maxS = sorted[sorted.length - 1];
      const stalled = stalls.filter((s) => s > 5).length;

      return {
        ticks: stalls.length,
        meanStallMs: mean,
        maxStallMs: maxS,
        p99StallMs: p99,
        durationMs,
        stallFraction: stalled / stalls.length,
      };
    },
  };
}

export function printEventLoopHealth(label: string, h: EventLoopHealth): void {
  // Grade thresholds:
  //   excellent  < 5ms max   — timer fires on schedule, truly non-blocking
  //   acceptable < 20ms max  — brief overhead (e.g. runAsync callback processing)
  //   degraded   < 100ms max — noticeable blocking (long sync callbacks)
  //   blocked    >= 100ms    — event loop frozen (synchronous TF_SessionRun)
  const grade =
    h.maxStallMs < 5
      ? "✓ excellent"
      : h.maxStallMs < 20
        ? "~ acceptable"
        : h.maxStallMs < 100
          ? "⚠ degraded"
          : "✗ blocked";

  if (h.ticks === 0) {
    // Timer never fired: all work ran as microtasks before the macrotask queue
    // was processed. This means the benchmark completed a full burst without
    // ever yielding — a sign of fully synchronous execution.
    console.log(
      `  ${label} event loop health: ticks=0 — no macrotask yield during burst ⚠`,
    );
    return;
  }

  console.log(
    `  ${label} event loop health:` +
      `  ticks=${h.ticks}  mean=${h.meanStallMs.toFixed(1)}ms` +
      `  p99=${h.p99StallMs.toFixed(1)}ms  max=${h.maxStallMs.toFixed(1)}ms` +
      `  stalled=${(h.stallFraction * 100).toFixed(0)}%  ${grade}`,
  );
}

// ─── Result merging ──────────────────────────────────────────────────────────
// Instead of overwriting previous results, merge new results with existing ones.
// For profiled benchmarks, keep results from different profiles. For runtime
// results, replace results for the same runtime (identified by runtime name).

export interface MergeOptions {
  /** Path to the existing JSON file (if it exists). */
  filePath: string;
  /** New suite to merge. */
  newSuite: BenchmarkSuite;
  /** Optional profile name to track (e.g., 'latency', 'throughput'). */
  profile?: string;
}

export function mergeResults(options: MergeOptions): BenchmarkSuite {
  const { filePath, newSuite, profile } = options;

  // If the file doesn't exist, just return the new suite as-is
  if (!existsSync(filePath)) {
    return newSuite;
  }

  // Read existing suite
  let existing: BenchmarkSuite;
  try {
    const content = readFileSync(filePath, "utf-8");
    existing = JSON.parse(content);
  } catch {
    // If parsing fails, just return the new suite
    return newSuite;
  }

  // Helper to create a key from runtime and profile (for accumulating different profiles)
  const getResultKey = (result: any) => {
    const prof = result.profile ?? "auto";
    return `${result.runtime}::${prof}`;
  };

  // Build a map of existing results by runtime+profile for easy lookup
  const existingResultsMap = new Map(
    existing.results.map((r) => [getResultKey(r), r]),
  );

  // Add all new results, replacing any with the same runtime+profile combination
  const mergedResults = [...existing.results];

  for (const newResult of newSuite.results) {
    const key = getResultKey(newResult);
    const existingIndex = mergedResults.findIndex(
      (r) => getResultKey(r) === key,
    );

    if (existingIndex !== -1) {
      // Replace existing result with same runtime+profile
      mergedResults[existingIndex] = newResult;
    } else {
      // Add new result
      mergedResults.push(newResult);
    }
  }

  // Merge comparisons - just use the new ones since they're computed
  // from the results anyway
  return {
    ...existing,
    ...newSuite,
    results: mergedResults,
  };
}
