# Devops - Express & Prometheus Metrics Server

A lightweight Node.js Express server integrated with `prom-client` to track application and system performance metrics, monitored via Prometheus and Grafana running in Docker.

## Project Structure
```text
DEVOPS/
├── index.js               # Express application with Prometheus metrics middleware
├── util.js                # Custom heavy task execution & business metrics logic
├── package.json           # Node dependencies (express, prom-client)
├── prometheus-config.yml  # Prometheus scraper configuration
├── prometheus.yml         # Active Prometheus scraper configuration
├── docker-compose.yml     # Docker definition for Prometheus server
└── .gitignore             # Excludes node_modules and logs from Git
```

## Running the Stack

### 1. Install Node Dependencies
```bash
npm install
```

### 2. Start the Express App
```bash
node index.js
```
The server will start at `http://localhost:8000`. You can visit:
* **Interactive Dashboard**: `http://localhost:8000/` (trigger metrics manually)
* **Metrics Endpoint**: `http://localhost:8000/metrics` (raw Prometheus format)

### 3. Start Prometheus & Grafana Containers
To launch the Prometheus server:
```bash
docker-compose up -d
```
To launch the Grafana server:
```bash
docker run -d -p 3000:3000 --name=grafana grafana/grafana-oss
```

### 4. Connect Grafana to Prometheus
1. Open **http://localhost:3000** (user: `admin`, pass: `admin`).
2. Add a new **Prometheus Data Source** pointing to **`http://host.docker.internal:9090`** (or your WSL IP `172.31.80.1:9090`).
3. Import Grafana Dashboard ID **`11159`** to visualize Node.js performance metrics!
