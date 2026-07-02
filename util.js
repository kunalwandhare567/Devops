const client = require("prom-client");

// Custom histogram to track the duration of the heavy task
const heavyTaskDuration = new client.Histogram({
  name: "heavy_task_duration_seconds",
  help: "Duration of heavy task execution in seconds",
  labelNames: ["status"],
  buckets: [0.1, 0.3, 0.5, 0.8, 1.2, 1.5, 2.0]
});

// Custom counter to track total heavy tasks executed
const heavyTaskCounter = new client.Counter({
  name: "heavy_task_total",
  help: "Total number of heavy tasks executed",
  labelNames: ["status"]
});

/**
 * Simulates a heavy task (delayed execution with a random outcome).
 * Measures execution time and records it using prom-client.
 * @returns {Promise<{duration: number, message: string}>}
 */
function doSomeHeavyTask() {
  return new Promise((resolve, reject) => {
    const start = Date.now();
    // Simulate dynamic processing time between 100ms and 1500ms
    const processingTime = Math.random() * 1400 + 100;

    setTimeout(() => {
      const end = Date.now();
      const duration = (end - start) / 1000; // convert to seconds
      const isSuccess = Math.random() > 0.15; // 85% success rate
      const status = isSuccess ? "success" : "failure";

      // Record metrics
      heavyTaskDuration.observe({ status }, duration);
      heavyTaskCounter.inc({ status });

      if (isSuccess) {
        resolve({ duration, message: "Heavy task completed successfully" });
      } else {
        reject(new Error("Heavy task failed during simulation"));
      }
    }, processingTime);
  });
}

module.exports = {
  doSomeHeavyTask,
  heavyTaskDuration,
  heavyTaskCounter
};
