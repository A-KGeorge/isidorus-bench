import fs from "fs";
import path from "path";
import { ChartJSNodeCanvas } from "chartjs-node-canvas";

const RESULTS_DIR = "./results";
const CHARTS_DIR = "./charts";
const WIDTH = 1000;
const HEIGHT = 600;

const chartJSNodeCanvas = new ChartJSNodeCanvas({
  width: WIDTH,
  height: HEIGHT,
  backgroundColour: "white",
});

const RUNTIME_MAP: Record<string, string> = {
  "@isidorus/cpu (InferencePool tf-parallel)": "isidorus",
  "@tensorflow/tfjs-node (single session, concurrent callers)": "tfjs",
  "tensorflow-python (concurrent threads)": "python",
};

const COLORS = {
  isidorus: "rgba(54, 162, 235, 1)",
  tfjs: "rgba(255, 159, 64, 1)",
  python: "rgba(75, 192, 192, 1)",
};

async function main() {
  if (!fs.existsSync(CHARTS_DIR)) fs.mkdirSync(CHARTS_DIR, { recursive: true });

  const allFiles = getAllFiles(RESULTS_DIR).filter((f) => f.endsWith(".json"));

  // 1. Process Inference Pool Groups
  const poolFiles = allFiles.filter((f) => f.includes("inference_pool"));
  await processInferencePool(poolFiles);

  // 2. Process ALL Memory files into one consolidated plot
  const memoryFiles = allFiles.filter((f) => f.includes("memory"));
  await processConsolidatedMemory(memoryFiles);

  // 3. Conv2D & Training
  const otherFiles = allFiles.filter(
    (f) => f.includes("conv2d") || f.includes("training"),
  );
  for (const file of otherFiles) {
    const data = JSON.parse(fs.readFileSync(file, "utf8"));
    await plotGenericThroughput(data, path.basename(file, ".json"));
  }
}

async function processInferencePool(files: string[]) {
  const registry: any = {};

  for (const file of files) {
    const data = JSON.parse(fs.readFileSync(file, "utf8"));
    // Extract model name from file path or JSON
    const model = path.basename(path.dirname(file));

    data.results.forEach((res: any) => {
      const profile = res.profile || "auto";
      const runtime = RUNTIME_MAP[res.runtime] || res.runtime;

      if (!registry[model]) registry[model] = {};
      if (!registry[model][profile]) registry[model][profile] = {};
      registry[model][profile][runtime] = res;
    });
  }

  for (const [model, profiles] of Object.entries(registry)) {
    for (const [profile, runtimes] of Object.entries(profiles as any)) {
      const chartPrefix = `inference_${model}_${profile}`;
      await plotInferenceComparison(
        model,
        profile,
        runtimes as any,
        chartPrefix,
      );
    }
  }
}

async function plotInferenceComparison(
  model: string,
  profile: string,
  runtimes: any,
  prefix: string,
) {
  const labels =
    runtimes.isidorus?.batches.map((b: any) => `C=${b.batchSize}`) || [];

  // --- Throughput Chart ---
  await saveChart(
    {
      type: "line",
      data: {
        labels,
        datasets: Object.entries(runtimes).map(
          ([name, data]: [string, any]) => ({
            label: name,
            data: data.batches.map((b: any) => b.throughput),
            borderColor: COLORS[name as keyof typeof COLORS],
            fill: false,
          }),
        ),
      },
      options: {
        plugins: {
          title: { display: true, text: `${model} (${profile}) - Throughput` },
        },
      },
    },
    `${prefix}_throughput.png`,
  );

  // --- Latency Chart (Mean, P50, P95, P99) ---
  const metrics = ["mean", "p50", "p95", "p99"];
  const styles = { mean: [], p50: [2, 2], p95: [5, 5], p99: [10, 5] };

  await saveChart(
    {
      type: "line",
      data: {
        labels,
        datasets: Object.entries(runtimes).flatMap(
          ([name, data]: [string, any]) =>
            metrics.map((m) => ({
              label: `${name} (${m})`,
              data: data.batches.map((b: any) => b.latency[m]),
              borderColor: COLORS[name as keyof typeof COLORS],
              borderDash: styles[m as keyof typeof styles],
              fill: false,
              pointRadius: m === "mean" ? 4 : 0,
            })),
        ),
      },
      options: {
        plugins: {
          title: {
            display: true,
            text: `${model} (${profile}) - Latency Comparison (ms)`,
          },
        },
        scales: { y: { type: "logarithmic" } },
      },
    },
    `${prefix}_latency_detailed.png`,
  );

  // --- Percentage Blocked Chart (Updated) ---
  await saveChart(
    {
      type: "line",
      data: {
        labels,
        datasets: Object.entries(runtimes).map(
          ([name, data]: [string, any]) => {
            let blockingRatio: number[] = [];
            if (name === "isidorus") {
              blockingRatio =
                data.eventLoopHealth?.map(
                  (h: any) => ((h.meanStallMs * h.ticks) / h.durationMs) * 100,
                ) ||
                data.batches.map((b: any) => {
                  const el = b.eventLoop;
                  return el
                    ? ((el.meanStallMs * el.ticks) / el.durationMs) * 100
                    : 0;
                }) ||
                [];
            } else if (name === "tfjs") {
              blockingRatio = data.batches.map((b: any) => {
                const el = b.eventLoop;
                return el
                  ? ((el.meanStallMs * el.ticks) / el.durationMs) * 100
                  : 0;
              });
            } else if (name === "python") {
              blockingRatio =
                data.gilHealth?.map(
                  (h: any) => ((h.meanStallMs * h.ticks) / h.durationMs) * 100,
                ) ||
                data.batches.map((b: any) => {
                  const gh = b.gilHealth;
                  return gh
                    ? ((gh.meanStallMs * gh.ticks) / gh.durationMs) * 100
                    : 0;
                }) ||
                [];
            }
            return {
              label:
                name === "python"
                  ? `${name} (Blocking Ratio β)`
                  : `${name} (Blocking Ratio β)`,
              data: blockingRatio,
              borderColor: COLORS[name as keyof typeof COLORS],
              fill: true,
              backgroundColor: COLORS[name as keyof typeof COLORS].replace(
                "1)",
                "0.1)",
              ),
            };
          },
        ),
      },
      options: {
        plugins: {
          title: {
            display: true,
            text: `${model} (${profile}) - Blocking Ratio (β) - Cumulative Stall Time %`,
          },
        },
        scales: {
          y: { min: 0, title: { display: true, text: "Blocking Ratio β (%)" } },
        },
      },
    },
    `${prefix}_blocked_percentage.png`,
  );
}

async function processConsolidatedMemory(files: string[]) {
  const labels: string[] = [];
  const heapData: number[] = [];
  const externalData: number[] = [];
  const rssData: number[] = [];

  // Sort files by size (small -> large) for the chart labels
  const sortedFiles = files.sort((a, b) => a.length - b.length);

  for (const file of sortedFiles) {
    const data = JSON.parse(fs.readFileSync(file, "utf8"));
    const modelName = path.basename(path.dirname(file)).replace("bench_", "");
    const lastSample = data.results[0].memoryProfile.samples.slice(-1)[0];

    labels.push(modelName);
    heapData.push(lastSample.heapUsed / 1024 / 1024);
    externalData.push((lastSample.external || 0) / 1024 / 1024);
    rssData.push(lastSample.rss / 1024 / 1024);
  }

  await saveChart(
    {
      type: "bar",
      data: {
        labels,
        datasets: [
          {
            label: "Heap Used (MB)",
            data: heapData,
            backgroundColor: "rgba(54, 162, 235, 0.7)",
          },
          {
            label: "External / Buffers (MB)",
            data: externalData,
            backgroundColor: "rgba(255, 206, 86, 0.7)",
          },
          {
            label: "Total RSS (MB)",
            data: rssData,
            backgroundColor: "rgba(201, 203, 207, 0.3)",
          },
        ],
      },
      options: {
        plugins: {
          title: {
            display: true,
            text: "Consolidated Memory Footprint (Stack = Process Internals)",
          },
        },
        scales: {
          x: { stacked: true, title: { display: true, text: "Model Scale" } },
          y: {
            stacked: false,
            title: { display: true, text: "Megabytes (MB)" },
          },
        },
      },
    },
    "memory_consolidated_comparison.png",
  );
}

// Utility Helpers
function getAllFiles(dir: string): string[] {
  return fs.readdirSync(dir).reduce((files: string[], file) => {
    const name = path.join(dir, file);
    return fs.statSync(name).isDirectory()
      ? [...files, ...getAllFiles(name)]
      : [...files, name];
  }, []);
}

async function saveChart(config: any, fileName: string) {
  const image = await chartJSNodeCanvas.renderToBuffer(config);
  fs.writeFileSync(path.join(CHARTS_DIR, fileName), image);
}

async function plotGenericThroughput(data: any, name: string) {
  const config = {
    type: "bar" as const,
    data: {
      labels: data.results[0].batches.map((b: any) => `Batch ${b.batchSize}`),
      datasets: data.results.map((r: any) => ({
        label: r.runtime,
        data: r.batches.map((b: any) => b.throughput || b.samplesPerSec),
      })),
    },
    options: {
      plugins: {
        title: { display: true, text: `${data.description} Throughput` },
      },
    },
  };
  await saveChart(config, `${name}_throughput.png`);
}

main().catch(console.error);
