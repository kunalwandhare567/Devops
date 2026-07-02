const express = require("express");
const client = require("prom-client");
const { doSomeHeavyTask } = require("./util");

const app = express();
const PORT = process.env.PORT || 8000;

// Enable default system metrics collection (CPU, Memory, GC, etc.)
client.collectDefaultMetrics({ register: client.register });

// Custom histogram to track HTTP request durations
const httpRequestDurationSeconds = new client.Histogram({
  name: "http_request_duration_seconds",
  help: "Duration of HTTP requests in seconds",
  labelNames: ["method", "route", "code"],
  buckets: [0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10]
});

// Middleware to measure request duration
app.use((req, res, next) => {
  const start = Date.now();
  
  res.on("finish", () => {
    const duration = (Date.now() - start) / 1000;
    const route = req.route ? req.route.path : req.path;
    
    // Ignore metrics endpoint to avoid skewing metrics
    if (route !== "/metrics") {
      httpRequestDurationSeconds.observe(
        {
          method: req.method,
          route: route || req.path,
          code: res.statusCode
        },
        duration
      );
    }
  });

  next();
});

// Serve metrics endpoint for Prometheus scraping
app.get("/metrics", async (req, res) => {
  try {
    res.setHeader("Content-Type", client.register.contentType);
    res.send(await client.register.metrics());
  } catch (err) {
    res.status(500).send(err.message);
  }
});

// Quick endpoint
app.get("/fast", (req, res) => {
  res.json({ message: "Fast response!", latency: "negligible" });
});

// Heavy endpoint utilizing doSomeHeavyTask from util.js
app.get("/heavy", async (req, res) => {
  try {
    const result = await doSomeHeavyTask();
    res.json({ status: "success", data: result });
  } catch (error) {
    res.status(500).json({ status: "error", message: error.message });
  }
});

// Error endpoint
app.get("/error", (req, res) => {
  res.status(500).json({ error: "Internal Server Error (Simulated)" });
});

// Beautiful UI dashboard for interactive testing
app.get("/", (req, res) => {
  res.send(`
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>Express Prometheus Metrics Dashboard</title>
      <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap" rel="stylesheet">
      <style>
        :root {
          --bg-color: #0f172a;
          --panel-bg: rgba(30, 41, 59, 0.7);
          --accent-primary: #38bdf8;
          --accent-success: #34d399;
          --accent-error: #f87171;
          --text-color: #f1f5f9;
          --text-muted: #94a3b8;
        }

        * {
          box-sizing: border-box;
          margin: 0;
          padding: 0;
        }

        body {
          font-family: 'Outfit', sans-serif;
          background: radial-gradient(circle at top right, #1e1b4b, var(--bg-color));
          color: var(--text-color);
          min-height: 100vh;
          display: flex;
          flex-direction: column;
          align-items: center;
          padding: 2rem;
        }

        header {
          text-align: center;
          margin-bottom: 3rem;
        }

        h1 {
          font-size: 2.8rem;
          font-weight: 800;
          background: linear-gradient(to right, var(--accent-primary), var(--accent-success));
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
          margin-bottom: 0.5rem;
        }

        p.subtitle {
          color: var(--text-muted);
          font-size: 1.1rem;
        }

        .container {
          width: 100%;
          max-width: 900px;
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 2rem;
        }

        @media (max-width: 768px) {
          .container {
            grid-template-columns: 1fr;
          }
        }

        .card {
          background: var(--panel-bg);
          backdrop-filter: blur(12px);
          -webkit-backdrop-filter: blur(12px);
          border: 1px solid rgba(255, 255, 255, 0.1);
          border-radius: 16px;
          padding: 2rem;
          box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
        }

        .card h2 {
          font-size: 1.5rem;
          margin-bottom: 1.5rem;
          border-bottom: 1px solid rgba(255, 255, 255, 0.1);
          padding-bottom: 0.5rem;
          color: var(--accent-primary);
        }

        .btn-group {
          display: flex;
          flex-direction: column;
          gap: 1rem;
        }

        button {
          font-family: 'Outfit', sans-serif;
          font-weight: 600;
          font-size: 1rem;
          padding: 0.8rem 1.5rem;
          border: none;
          border-radius: 8px;
          cursor: pointer;
          transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
          color: #0f172a;
          display: flex;
          justify-content: space-between;
          align-items: center;
        }

        button::after {
          content: '→';
          font-size: 1.2rem;
          opacity: 0.7;
          transition: transform 0.2s;
        }

        button:hover::after {
          transform: translateX(4px);
        }

        .btn-fast {
          background: var(--accent-primary);
        }
        .btn-fast:hover {
          background: #7dd3fc;
          box-shadow: 0 0 15px rgba(56, 189, 248, 0.4);
        }

        .btn-heavy {
          background: var(--accent-success);
        }
        .btn-heavy:hover {
          background: #6ee7b7;
          box-shadow: 0 0 15px rgba(52, 211, 153, 0.4);
        }

        .btn-error {
          background: var(--accent-error);
        }
        .btn-error:hover {
          background: #fca5a5;
          box-shadow: 0 0 15px rgba(248, 113, 113, 0.4);
        }

        .btn-metrics {
          background: #ffffff;
          border: 1px solid rgba(255, 255, 255, 0.2);
          color: #0f172a;
        }
        .btn-metrics:hover {
          background: #e2e8f0;
          box-shadow: 0 0 15px rgba(255, 255, 255, 0.2);
        }

        .console {
          background: #020617;
          border-radius: 8px;
          padding: 1rem;
          font-family: 'Courier New', Courier, monospace;
          font-size: 0.9rem;
          height: 250px;
          overflow-y: auto;
          border: 1px solid rgba(255, 255, 255, 0.05);
        }

        .log-entry {
          margin-bottom: 0.5rem;
          border-bottom: 1px solid rgba(255, 255, 255, 0.03);
          padding-bottom: 0.3rem;
          animation: fadeIn 0.3s ease;
        }

        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(4px); }
          to { opacity: 1; transform: translateY(0); }
        }

        .log-time {
          color: var(--text-muted);
        }
        .log-path {
          font-weight: bold;
        }
        .log-success {
          color: var(--accent-success);
        }
        .log-error {
          color: var(--accent-error);
        }

        footer {
          margin-top: auto;
          padding: 2rem 0;
          color: var(--text-muted);
          font-size: 0.9rem;
          text-align: center;
        }
      </style>
    </head>
    <body>
      <header>
        <h1>Express + Prometheus Metrics</h1>
        <p class="subtitle">Generate load and monitor time-series metrics dynamically</p>
      </header>

      <div class="container">
        <div class="card">
          <h2>Trigger Endpoints</h2>
          <div class="btn-group">
            <button class="btn-fast" onclick="triggerEndpoint('/fast')">Trigger Fast Request</button>
            <button class="btn-heavy" onclick="triggerEndpoint('/heavy')">Trigger Heavy Task</button>
            <button class="btn-error" onclick="triggerEndpoint('/error')">Trigger Simulated Error</button>
            <button class="btn-metrics" onclick="window.open('/metrics', '_blank')">View Raw Metrics</button>
          </div>
        </div>

        <div class="card">
          <h2>Activity Log</h2>
          <div class="console" id="console">
            <div class="log-entry"><span class="log-time">[System]</span> Dashboard initialized. Click buttons on the left to fire requests.</div>
          </div>
        </div>
      </div>

      <footer>
        Express prom-client Server &copy; 2026
      </footer>

      <script>
        const consoleEl = document.getElementById('console');

        function log(path, status, duration, message) {
          const now = new Date().toLocaleTimeString();
          const entry = document.createElement('div');
          entry.className = 'log-entry';
          
          let statusSpan = '';
          if (status >= 200 && status < 300) {
            statusSpan = \`<span class="log-success">\${status} OK</span>\`;
          } else {
            statusSpan = \`<span class="log-error">\${status} Error</span>\`;
          }

          entry.innerHTML = \`
            <span class="log-time">[\${now}]</span> 
            <span class="log-path">\${path}</span> &rarr; 
            \${statusSpan} (\${duration}ms) 
            <div style="font-size: 0.8rem; margin-top: 2px; color: var(--text-muted);">\${JSON.stringify(message)}</div>
          \`;
          
          consoleEl.appendChild(entry);
          consoleEl.scrollTop = consoleEl.scrollHeight;
        }

        async function triggerEndpoint(path) {
          const start = Date.now();
          try {
            const response = await fetch(path);
            const data = await response.json();
            const duration = Date.now() - start;
            log(path, response.status, duration, data);
          } catch (err) {
            const duration = Date.now() - start;
            log(path, 500, duration, { error: err.message });
          }
        }
      </script>
    </body>
    </html>
  `);
});

// Start server
app.listen(PORT, () => {
  console.log(`Server is running at http://localhost:${PORT}`);
  console.log(`Metrics scraping available at http://localhost:${PORT}/metrics`);
});
