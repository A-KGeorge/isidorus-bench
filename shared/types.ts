// ─── Shared types ──────────────────────────────────────────────────────────

export interface IterStats {
  mean: number; // ms
  p50: number;
  p95: number;
  p99: number;
  min: number;
  max: number;
}

export interface BatchResult {
  batchSize: number;
  latency: IterStats; // per-call latency (ms)
  throughput: number; // images per second (batchSize * 1000 / mean)
  eventLoopHealth?: EventLoopHealth;
}

export interface EventLoopHealth {
  ticks: number;
  meanStallMs: number;
  maxStallMs: number;
  p99StallMs: number;
  durationMs: number;
  stallFraction: number;
}

export interface BenchmarkResult {
  runtime: string; // e.g. "@isidorus/cpu", "@tensorflow/tfjs-node"
  runtimeVersion: string;
  model: string; // human-readable model description
  profile?: string; // optional profile (e.g. "auto", "latency", "throughput")
  inputShape: number[]; // excluding batch dim
  warmupIters: number;
  benchIters: number;
  batches: BatchResult[];
  machineInfo: MachineInfo;
  timestamp: string; // ISO-8601
  durationMs: number; // total benchmark wall time
  coldStartMs?: number; // optional cold start latency
}

export interface MachineInfo {
  platform: string; // process.platform
  arch: string; // process.arch
  nodeVersion: string;
  cpus: number; // os.availableParallelism()
  cpuModel: string; // first cpu from os.cpus()
}

export interface BenchmarkSuite {
  name: string; // e.g. "conv2d"
  description: string;
  results: BenchmarkResult[];
  comparisons: SpeedupEntry[]; // computed by run.ts
}

export interface SpeedupEntry {
  batchSize: number;
  baseline: string; // runtime name
  candidate: string;
  baselineMs: number;
  candidateMs: number;
  speedup: number; // baselineMs / candidateMs  (>1 = candidate faster)
}
