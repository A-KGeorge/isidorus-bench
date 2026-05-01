/**
 * benchmarks/inference_pool/onnxruntime_node.ts
 *
 * onnxruntime-node baseline for the InferencePool throughput benchmark.
 *
 * This directly measures whether onnxruntime-node's session.run() blocks the
 * Node.js event loop — the exact problem described in ORT issue #19611.
 *
 * Requires:
 *   npm install onnxruntime-node
 *   A .onnx model (convert with: python bench/convert_to_onnx.py model.pb)
 *
 * The ORT issue proposed wrapping RunAsync behind a TypedThreadSafeFunction so
 * session.run() becomes truly async. Depending on the installed version of
 * onnxruntime-node, the event loop health data here will show whether that
 * fix landed.
 */

import { performance } from "node:perf_hooks";

import {
  computeStats,
  machineInfo,
  startEventLoopMonitor,
  printEventLoopHealth,
  fmtMs,
} from "../../shared/stats.js";
import type { BenchmarkResult } from "../../shared/types.js";

const WARMUP_REQUESTS = parseInt(process.env.WARMUP_ITERS ?? "20", 10);
const BENCH_REQUESTS = parseInt(process.env.BENCH_ITERS ?? "200", 10);
export const CONCURRENCY_LEVELS_ORT = [1, 2, 3, 4, 5, 6, 7, 8];

async function getRuntimeVersion(): Promise<string> {
  try {
    const { createRequire } = await import("node:module");
    const req = createRequire(import.meta.url);
    const pkg = req("onnxruntime-node/package.json") as { version: string };
    return pkg.version;
  } catch {
    return "unknown";
  }
}

/**
 * Build a feed dict for the session.
 *
 * For multi-input models (e.g. BERT: input_ids, attention_mask, token_type_ids)
 * we create one tensor per named input using the shape from the session metadata.
 * Inputs named with "ids", "mask", or "type" are assumed int32; others float32.
 */
function buildFeeds(
  ort: any,
  inputNames: string[],
  inputShape: number[],
  inputDtype: "float32" | "int32",
): Record<string, any> {
  const feeds: Record<string, any> = {};
  const nElems = inputShape.reduce((a, b) => a * b, 1);

  for (const name of inputNames) {
    // Heuristic dtype detection from input name when the caller didn't specify.
    const isInt =
      inputDtype === "int32" || /id|mask|type|token|segment/i.test(name);
    const dtype = isInt ? "int32" : "float32";
    const data = isInt ? new Int32Array(nElems) : new Float32Array(nElems);
    feeds[name] = new ort.Tensor(dtype, data, inputShape);
  }
  return feeds;
}

/**
 * Run BENCH_REQUESTS concurrent inferences at the given concurrency level.
 * Measures wall-clock latency from dispatch (not just compute time) to give
 * a fair comparison with the async isidorus InferencePool numbers.
 */
async function runConcurrent(
  session: any,
  feeds: Record<string, any>,
  concurrency: number,
  nRequests: number,
): Promise<number[]> {
  const samples: number[] = [];
  const inFlight = new Set<Promise<void>>();
  let issued = 0;

  const dispatch = () => {
    const t0 = performance.now();
    let p: Promise<void>;
    p = session.run(feeds).then(() => {
      samples.push(performance.now() - t0);
      inFlight.delete(p!);
    });
    inFlight.add(p);
    issued++;
  };

  // Seed with initial concurrent requests.
  while (issued < Math.min(concurrency, nRequests)) dispatch();

  while (issued < nRequests || inFlight.size > 0) {
    await Promise.race(inFlight);
    while (issued < nRequests && inFlight.size < concurrency) dispatch();
  }

  return samples;
}

export async function runOnnxRuntimeNodeBench(
  modelPath: string,
  inputShape: number[],
  inputDtype: "float32" | "int32",
  intraOpThreads: number,
  profile: string,
): Promise<BenchmarkResult | null> {
  let ort: any;
  try {
    ort = await import("onnxruntime-node");
  } catch {
    console.warn(
      "\n⚠  onnxruntime-node not installed — skipping ORT baseline.\n" +
        "   Install with: npm install onnxruntime-node\n",
    );
    return null;
  }

  const version = await getRuntimeVersion();

  const sessionOptions = {
    executionProviders: ["cpu"],
    intraOpNumThreads: intraOpThreads,
    graphOptimizationLevel: "all",
  };

  let session: any;
  const coldStartT0 = performance.now();
  try {
    session = await ort.InferenceSession.create(modelPath, sessionOptions);
  } catch (e) {
    console.warn(
      `\n⚠  onnxruntime-node could not load ${modelPath}\n` +
        `   Error: ${e}\n` +
        `   Note: This may be due to unsupported operators (e.g., Erfc).\n` +
        `   Python asyncio benchmark will still run.\n`,
    );
    return null;
  }
  const coldStartMs = performance.now() - coldStartT0;

  const { inputNames, outputNames } = session;
  console.log(
    `  ORT session loaded in ${coldStartMs.toFixed(0)}ms` +
      `  inputs=[${inputNames.join(", ")}]  outputs=[${outputNames.join(", ")}]`,
  );

  const feeds = buildFeeds(ort, inputNames, inputShape, inputDtype);
  const t0 = performance.now();

  // Warmup
  await runConcurrent(session, feeds, 1, WARMUP_REQUESTS);

  const W = 72;
  console.log("\n" + "─".repeat(W));
  console.log(` onnxruntime-node  v${version}  — concurrent throughput`);
  console.log("─".repeat(W));
  console.log(
    ` ${"workers".padEnd(8)}` +
      ` ${"mean".padStart(11)}` +
      ` ${"p50".padStart(11)}` +
      ` ${"p95".padStart(11)}` +
      ` ${"p99".padStart(11)}` +
      ` ${"req/s".padStart(10)}`,
  );
  console.log(" " + "─".repeat(70));

  const batches = [];

  for (const concurrency of CONCURRENCY_LEVELS_ORT) {
    const monitor = startEventLoopMonitor(5);
    const wallStart = performance.now();

    const samples = await runConcurrent(
      session,
      feeds,
      concurrency,
      BENCH_REQUESTS,
    );

    const wallMs = performance.now() - wallStart;
    const health = monitor.stop();
    const s = computeStats(samples);
    const reqPerS = ((BENCH_REQUESTS * 1000) / wallMs).toFixed(0);

    console.log(
      ` ${String(concurrency).padEnd(8)}` +
        ` ${fmtMs(s.mean).padStart(11)}` +
        ` ${fmtMs(s.p50).padStart(11)}` +
        ` ${fmtMs(s.p95).padStart(11)}` +
        ` ${fmtMs(s.p99).padStart(11)}` +
        ` ${(reqPerS + " req/s").padStart(10)}`,
    );
    printEventLoopHealth(`c=${concurrency}`, health);

    batches.push({
      batchSize: concurrency,
      latency: s,
      throughput: parseFloat(reqPerS),
      eventLoop: health,
    });
  }

  console.log("─".repeat(W) + "\n");
  await session.release?.();

  return {
    runtime: "onnxruntime-node",
    runtimeVersion: version,
    model: modelPath.split(/[\\/]/).pop() ?? modelPath,
    profile,
    inputShape,
    warmupIters: WARMUP_REQUESTS,
    benchIters: BENCH_REQUESTS,
    batches,
    machineInfo: machineInfo(),
    timestamp: new Date().toISOString(),
    durationMs: performance.now() - t0,
    coldStartMs,
  };
}
