"""
AIOps Platform -- Full Telemetry Data Generator (120k rows, 13 failure modes)
==============================================================================
Generates synthetic telemetry for all 13 HMM failure modes.

Output files:
  data/telemetry_metrics.csv   -- ~120k rows, 33 columns (one row per 2s step)
  data/telemetry_traces.csv    -- ~90k rows,  14 columns (one row per span)
  data/telemetry_logs.csv      -- ~120k rows,  8 columns (one log line per step)

Row math: 13 modes x 77 episodes x 120 steps = 120,120 metric rows.
Schema driven from the user-specified column definitions.
All values ASCII-only (Windows cp1252 safe).
"""

import argparse
import csv
import json
import math
import os
import uuid
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALL_MODES = [
    "NONE",
    "MEMORY_LEAK",
    "CPU_SATURATION",
    "LATENCY_SPIKE",
    "ERROR_STORM",
    "DB_SLOWDOWN",
    "CACHE_STAMPEDE",
    "QUEUE_BACKUP",
    "DEPENDENCY_TIMEOUT",
    "BAD_DEPLOY",
    "RETRY_STORM",
    "DISK_IO_SATURATION",
    "CASCADING_FAILURE",
]

SERVICES = ["auth-service", "payment-service", "order-service", "inventory-service"]
SERVICE_VERSIONS = ["v1.0.0", "v1.1.0", "v1.2.3", "v2.0.1"]
DOWNSTREAM_SERVICES = ["user-db", "payment-gateway", "cache-cluster", "inventory-api"]

# Span name options per the user spec
SPAN_NAMES = ["HTTP request", "db.query", "cache.get", "downstream.call"]

# Column definitions
METRIC_FIELDS = [
    "episode_id", "failure_mode", "service", "source", "elapsed_s", "timestamp",
    "active_connections", "cache_hit_rate", "cache_miss_rate", "circuit_breaker_state",
    "cpu_saturation", "cpu_utilization", "db_connection_pool", "db_connection_wait",
    "db_p99", "disk_read_latency", "disk_write_latency", "error_rate", "gc_pause_p99",
    "heap_mb", "http_4xx_rate", "http_5xx_rate", "iops_utilization", "memory_utilization",
    "network_errors", "p50_latency", "p95_latency", "p99_latency", "queue_lag",
    "retry_count_per_request", "rps", "thread_pool_queue", "upstream_timeout_rate",
]

TRACE_FIELDS = [
    "episode_id", "failure_mode", "service", "elapsed_s", "timestamp",
    "span_id", "parent_span_id", "span_name", "db_operation_type",
    "span_duration_ms", "span_status", "peer_service", "service_version", "trace_id",
]

LOG_FIELDS = [
    "episode_id", "failure_mode", "service", "elapsed_s", "timestamp",
    "log_level", "exception_type", "log_message",
]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    seed: int = 42
    episodes_per_mode: int = 77   # 13 modes x 77 eps x 120 steps = 120,120 metric rows
    steps_per_episode: int = 120
    step_interval_s: int = 2
    output_dir: str = "data"
    base_rps: float = 200.0
    db_capacity_rps: float = 250.0


# ---------------------------------------------------------------------------
# Statistical distributions
# ---------------------------------------------------------------------------

class Dist:
    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    # ---- Baseline ----
    def p50(self):       return float(np.clip(self.rng.normal(90, 8), 50, 200))
    def p95(self, p50, p99):
        lo, hi = min(p50, p99), max(p50, p99)
        if hi <= lo: return float(lo)
        return float(np.clip(self.rng.uniform(lo, hi), lo, hi))
    def p99(self):       return float(np.clip(self.rng.normal(120, 15), 70, 300))
    def heap(self):      return float(np.clip(self.rng.normal(512, 20), 400, 700))
    def gc_p99(self):    return float(np.clip(self.rng.exponential(15), 2, 60))
    def err_rate(self):  return float(self.rng.beta(2, 198))
    def cache_miss(self):return float(np.clip(self.rng.beta(1, 19), 0.01, 0.12))
    def db_p99(self):    return float(np.clip(self.rng.lognormal(3.0, 0.3), 5, 80))
    def cpu_util(self):  return float(np.clip(self.rng.normal(35, 5), 10, 65))
    def queue_lag(self): return int(self.rng.integers(0, 20))
    def retry_cnt(self): return float(np.clip(self.rng.beta(1, 49) * 2, 0, 0.1))
    def rps(self):       return float(np.clip(self.rng.normal(200, 20), 80, 400))
    def active_conn(self): return int(self.rng.integers(20, 120))
    def db_conn_pool(self): return float(np.clip(self.rng.normal(0.35, 0.08), 0.1, 0.7))
    def db_conn_wait(self): return float(np.clip(self.rng.exponential(5), 0.5, 30))
    def disk_read_lat(self): return float(np.clip(self.rng.lognormal(1.5, 0.4), 1, 20))
    def disk_write_lat(self): return float(np.clip(self.rng.lognormal(1.8, 0.4), 2, 30))
    def iops_util(self): return float(np.clip(self.rng.normal(0.25, 0.06), 0.05, 0.55))
    def mem_util(self, heap_mb): return float(np.clip(heap_mb / 2048.0 + self.rng.normal(0, 0.03), 0.1, 0.98))
    def net_errors(self): return int(self.rng.integers(0, 5))
    def http_4xx(self): return float(np.clip(self.rng.beta(1, 199), 0, 0.05))
    def http_5xx(self): return float(np.clip(self.rng.beta(1, 499), 0, 0.02))
    def thread_pool_q(self): return int(self.rng.integers(0, 15))
    def upstream_timeout(self): return float(np.clip(self.rng.beta(1, 199), 0, 0.05))
    def circuit_breaker(self, err_rate): 
        if err_rate > 0.40: return "open"
        if err_rate > 0.20: return "half-open"
        return "closed"

    # ---- MEMORY_LEAK ----
    def heap_leak(self, elapsed_s, rate=12.0):
        return float(np.clip(512 + (elapsed_s / 60.0) * rate + self.rng.normal(0, 8), 400, 1800))
    def gc_leak(self, heap_mb):
        ratio = max(0, (heap_mb - 512) / (1800 - 512))
        return float(np.clip(self.rng.lognormal(math.log(15 + ratio * 500), 0.3), 2, 900))

    # ---- CPU_SATURATION ----
    def cpu_sat_util(self): return float(np.clip(self.rng.normal(88, 4), 70, 100))
    def p99_cpu_sat(self, cpu): return float(np.clip(self.rng.normal(200 + (cpu - 70) * 8, 30), 150, 1500))

    # ---- LATENCY_SPIKE ----
    def gc_spike(self): return float(np.clip(self.rng.lognormal(5.5, 0.4), 80, 650))
    def p99_spike(self, gc_ms): return float(np.clip(gc_ms * self.rng.uniform(0.8, 1.1) + 90 + self.rng.normal(0, 30), 200, 3000))
    def gc_event(self, lam=0.08): return self.rng.poisson(lam) > 0

    # ---- ERROR_STORM ----
    def err_storm(self): return float(np.clip(self.rng.beta(8, 12), 0.25, 0.70))

    # ---- DB_SLOWDOWN ----
    def db_slow(self, elapsed_s):
        growth = (elapsed_s / 240.0) * 300
        return float(np.clip(self.rng.lognormal(math.log(max(20, 20 + growth)), 0.25), 20, 1500))

    # ---- CACHE_STAMPEDE ----
    def cache_miss_stampede(self):
        return float(np.clip(self.rng.beta(19, 1) * 0.97 + self.rng.normal(0, 0.025), 0.85, 0.99))
    def db_p99_mm1(self, miss_rate, base_rps, cap_rps):
        svc_ms = 20.0 * self.rng.lognormal(0, 0.05)
        rho = min(base_rps * miss_rate / cap_rps, 0.98)
        queue_ms = svc_ms * (rho / (1 - rho)) if rho < 0.99 else svc_ms * 50
        return float(np.clip(svc_ms + queue_ms + self.rng.normal(0, 8), 20, 1200))

    # ---- QUEUE_BACKUP ----
    def queue_lag_backup(self, elapsed_s):
        return int(min(500, int(elapsed_s * 1.8 + self.rng.integers(0, 30))))

    # ---- DEPENDENCY_TIMEOUT ----
    def timeout_rate(self): return float(np.clip(self.rng.beta(5, 5), 0.15, 0.55))

    # ---- BAD_DEPLOY ----
    def err_deploy(self): return float(np.clip(self.rng.beta(12, 8), 0.30, 0.80))

    # ---- RETRY_STORM ----
    def retry_high(self): return float(np.clip(self.rng.beta(10, 5), 0.40, 0.90))
    def rps_retry(self, retry_rate, base_rps): return float(np.clip(base_rps * (1 + retry_rate * 3), base_rps, base_rps * 4))

    # ---- DISK_IO_SATURATION ----
    def db_slow_disk(self, elapsed_s):
        growth = (elapsed_s / 240.0) * 500
        return float(np.clip(self.rng.lognormal(math.log(max(40, 40 + growth)), 0.3), 40, 2000))
    def disk_lat_sat(self): return float(np.clip(self.rng.lognormal(4.5, 0.4), 50, 2000))
    def iops_sat(self): return float(np.clip(self.rng.normal(0.92, 0.04), 0.80, 1.0))

    # ---- CASCADING_FAILURE ----
    def cascade_cpu(self): return float(np.clip(self.rng.normal(82, 6), 60, 100))
    def cascade_err(self): return float(np.clip(self.rng.beta(10, 6), 0.35, 0.85))
    def cascade_p99(self, cpu, err): return float(np.clip(self.rng.normal(800 + cpu * 8 + err * 200, 100), 500, 5000))


# ---------------------------------------------------------------------------
# Log message templates
# ---------------------------------------------------------------------------

LOG_LEVELS = {
    "NONE": "INFO", "MEMORY_LEAK": "WARNING", "CPU_SATURATION": "WARNING",
    "LATENCY_SPIKE": "WARNING", "ERROR_STORM": "ERROR", "DB_SLOWDOWN": "WARNING",
    "CACHE_STAMPEDE": "WARNING", "QUEUE_BACKUP": "WARNING",
    "DEPENDENCY_TIMEOUT": "ERROR", "BAD_DEPLOY": "ERROR",
    "RETRY_STORM": "WARNING", "DISK_IO_SATURATION": "WARNING",
    "CASCADING_FAILURE": "CRITICAL",
}

EXCEPTION_MAP = {
    "ERROR_STORM": "RuntimeException",
    "DEPENDENCY_TIMEOUT": "SocketTimeoutException",
    "BAD_DEPLOY": "NullPointerException",
    "CASCADING_FAILURE": "SystemOverloadException",
}


def build_log_msg(mode, vals):
    v = vals
    msgs = {
        "NONE":               "OK cpu={cpu:.0f}% p99={p99:.0f}ms err={err:.2f}%".format(**v),
        "MEMORY_LEAK":        "WARN heap={heap:.0f}MB gc_p99={gc:.0f}ms -- memory leak".format(**v),
        "CPU_SATURATION":     "WARN cpu={cpu:.0f}% -- saturated p99={p99:.0f}ms".format(**v),
        "LATENCY_SPIKE":      "WARN p99={p99:.0f}ms gc_p99={gc:.0f}ms -- latency spike".format(**v),
        "ERROR_STORM":        "ERROR err={err:.2f}% http5xx={h5:.2f}% -- error storm".format(**v),
        "DB_SLOWDOWN":        "WARN db_p99={db:.0f}ms -- slow query detected".format(**v),
        "CACHE_STAMPEDE":     "WARN cache_miss={miss:.0f}% db_p99={db:.0f}ms -- stampede".format(**v),
        "QUEUE_BACKUP":       "WARN queue_lag={lag}ms thread_pool={tp} -- queue backup".format(**v),
        "DEPENDENCY_TIMEOUT": "ERROR upstream_timeout={ut:.2f}% p99={p99:.0f}ms".format(**v),
        "BAD_DEPLOY":         "ERROR err={err:.2f}% http5xx={h5:.2f}% -- bad canary deploy".format(**v),
        "RETRY_STORM":        "WARN retry_cnt={rc:.2f} rps={rps:.0f} -- retry storm".format(**v),
        "DISK_IO_SATURATION": "WARN disk_read={dr:.0f}ms iops={iops:.0f}% -- disk saturation".format(**v),
        "CASCADING_FAILURE":  "CRITICAL cpu={cpu:.0f}% err={err:.2f}% p99={p99:.0f}ms -- cascade".format(**v),
    }
    return msgs.get(mode, mode + " detected")


# ---------------------------------------------------------------------------
# Trace span builder (flat CSV rows)
# ---------------------------------------------------------------------------

def build_trace_rows(
    episode_id, failure_mode, service, elapsed_s, timestamp,
    rng: np.random.Generator, svc_version: str
) -> List[dict]:
    """
    Returns a list of flat span rows (one per span).
    Span types: HTTP request (root), cache.get, db.query, downstream.call
    """
    trace_id = str(uuid.uuid4())
    root_sid = str(uuid.uuid4())
    cache_sid = str(uuid.uuid4())
    db_sid = str(uuid.uuid4())
    dep_sid = str(uuid.uuid4())

    base = dict(
        episode_id=episode_id, failure_mode=failure_mode, service=service,
        elapsed_s=elapsed_s, timestamp=timestamp, trace_id=trace_id,
        service_version=svc_version,
    )

    # --- span durations per failure mode ---
    if failure_mode == "NONE":
        cache_ms = float(np.clip(rng.normal(3, 0.5), 1, 8))
        db_ms = float(np.clip(rng.normal(15, 3), 5, 35))
        root_ms = cache_ms + db_ms + float(np.clip(rng.normal(8, 2), 2, 20))
        cache_status = "OK"; db_status = "OK"; root_status = "OK"
        db_op = "READ" if rng.random() > 0.3 else "WRITE"
        peer = ""

    elif failure_mode == "MEMORY_LEAK":
        cache_ms = float(np.clip(rng.normal(3, 0.5), 1, 8))
        db_ms = float(np.clip(rng.normal(18, 4), 8, 40))
        gc_gap = float(np.clip(rng.lognormal(5.0, 0.5), 40, 500))
        root_ms = gc_gap + cache_ms + db_ms
        cache_status = "OK"; db_status = "OK"; root_status = "OK"
        db_op = "READ"; peer = ""

    elif failure_mode == "CPU_SATURATION":
        cache_ms = float(np.clip(rng.normal(5, 1), 2, 15))
        db_ms = float(np.clip(rng.normal(25, 5), 10, 60))
        cpu_delay = float(np.clip(rng.normal(150, 40), 50, 500))
        root_ms = cpu_delay + cache_ms + db_ms
        cache_status = "OK"; db_status = "OK"; root_status = "OK"
        db_op = "READ"; peer = ""

    elif failure_mode == "LATENCY_SPIKE":
        gc_ms = float(np.clip(rng.lognormal(5.5, 0.4), 80, 650))
        affected = rng.random() < 0.08
        cache_ms = float(np.clip(rng.normal(3, 0.5), 1, 8))
        db_ms = float(np.clip(rng.normal(15, 3), 5, 35))
        root_ms = (gc_ms if affected else float(np.clip(rng.normal(8, 2), 2, 20))) + cache_ms + db_ms
        cache_status = "OK"; db_status = "OK"; root_status = "OK"
        db_op = "READ"; peer = ""

    elif failure_mode == "ERROR_STORM":
        cache_ms = float(np.clip(rng.normal(3, 0.5), 1, 8))
        db_ms = float(np.clip(rng.normal(18, 5), 8, 80))
        root_ms = cache_ms + db_ms + float(np.clip(rng.normal(10, 3), 3, 25))
        cache_status = "OK"
        db_status = "ERROR" if rng.random() < 0.45 else "OK"
        root_status = "ERROR" if db_status == "ERROR" else "OK"
        db_op = "READ" if rng.random() > 0.2 else "WRITE"
        peer = ""

    elif failure_mode == "DB_SLOWDOWN":
        cache_ms = float(np.clip(rng.normal(3, 0.5), 1, 8))
        db_ms = float(np.clip(rng.lognormal(5.5, 0.4), 100, 1500))
        root_ms = cache_ms + db_ms + float(np.clip(rng.normal(10, 3), 3, 25))
        cache_status = "OK"; db_status = "OK"; root_status = "OK"
        db_op = "READ"; peer = ""

    elif failure_mode == "CACHE_STAMPEDE":
        cache_ms = float(np.clip(rng.normal(2, 0.3), 1, 5))
        db_ms = float(np.clip(rng.lognormal(5.2, 0.35), 60, 1200))
        root_ms = cache_ms + db_ms + float(np.clip(rng.normal(10, 3), 3, 30))
        cache_status = "MISS"; db_status = "OK"; root_status = "OK"
        db_op = "READ"; peer = ""

    elif failure_mode == "QUEUE_BACKUP":
        cache_ms = float(np.clip(rng.normal(3, 0.5), 1, 8))
        db_ms = float(np.clip(rng.normal(20, 5), 8, 60))
        queue_wait = float(np.clip(rng.exponential(200), 50, 1500))
        root_ms = queue_wait + cache_ms + db_ms
        cache_status = "OK"; db_status = "OK"; root_status = "OK"
        db_op = "READ"; peer = ""

    elif failure_mode == "DEPENDENCY_TIMEOUT":
        cache_ms = float(np.clip(rng.normal(3, 0.5), 1, 8))
        dep_ms = float(np.clip(rng.normal(5000, 500), 2000, 30000))
        root_ms = dep_ms + cache_ms
        db_ms = float(np.clip(rng.normal(15, 3), 5, 35))
        cache_status = "OK"
        dep_status = "ERROR"
        root_status = "ERROR" if rng.random() < 0.6 else "OK"
        db_op = ""
        peer = str(rng.choice(DOWNSTREAM_SERVICES))

        # For dependency timeout, return 4 spans including downstream.call
        rows = [
            {**base, "span_id": root_sid, "parent_span_id": "",
             "span_name": "HTTP request", "db_operation_type": "",
             "span_duration_ms": round(root_ms, 2), "span_status": root_status, "peer_service": ""},
            {**base, "span_id": cache_sid, "parent_span_id": root_sid,
             "span_name": "cache.get", "db_operation_type": "",
             "span_duration_ms": round(cache_ms, 2), "span_status": "OK", "peer_service": ""},
            {**base, "span_id": dep_sid, "parent_span_id": root_sid,
             "span_name": "downstream.call", "db_operation_type": "",
             "span_duration_ms": round(dep_ms, 2), "span_status": dep_status, "peer_service": peer},
        ]
        return rows

    elif failure_mode == "BAD_DEPLOY":
        cache_ms = float(np.clip(rng.normal(3, 0.5), 1, 8))
        db_ms = float(np.clip(rng.normal(15, 5), 5, 50))
        root_ms = cache_ms + db_ms + float(np.clip(rng.normal(10, 3), 3, 25))
        cache_status = "OK"
        db_status = "ERROR" if rng.random() < 0.55 else "OK"
        root_status = "ERROR" if db_status == "ERROR" else "OK"
        db_op = "READ" if rng.random() > 0.3 else "WRITE"
        peer = ""

    elif failure_mode == "RETRY_STORM":
        retry_count = int(rng.integers(2, 8))
        cache_ms = float(np.clip(rng.normal(3, 0.5), 1, 8))
        db_ms = float(np.clip(rng.normal(18, 4), 8, 45)) * retry_count
        root_ms = cache_ms + db_ms + float(rng.normal(10, 5))
        cache_status = "OK"; db_status = "OK"; root_status = "OK"
        db_op = "READ"; peer = ""

    elif failure_mode == "DISK_IO_SATURATION":
        cache_ms = float(np.clip(rng.normal(3, 0.5), 1, 8))
        disk_wait = float(np.clip(rng.exponential(300), 80, 2000))
        db_ms = disk_wait + float(np.clip(rng.normal(20, 5), 8, 50))
        root_ms = cache_ms + db_ms + float(np.clip(rng.normal(10, 3), 3, 20))
        cache_status = "OK"; db_status = "OK"; root_status = "OK"
        db_op = "READ"; peer = ""

    else:  # CASCADING_FAILURE
        cache_ms = float(np.clip(rng.normal(5, 1), 2, 15))
        db_ms = float(np.clip(rng.lognormal(6.0, 0.5), 200, 5000))
        root_ms = cache_ms + db_ms + float(np.clip(rng.normal(80, 30), 20, 300))
        cache_status = "MISS"
        db_status = "ERROR" if rng.random() < 0.65 else "OK"
        root_status = "ERROR" if db_status == "ERROR" else "OK"
        db_op = "READ" if rng.random() > 0.3 else "WRITE"
        peer = ""

    rows = [
        {**base, "span_id": root_sid, "parent_span_id": "",
         "span_name": "HTTP request", "db_operation_type": "",
         "span_duration_ms": round(max(root_ms, 1), 2), "span_status": root_status, "peer_service": ""},
        {**base, "span_id": cache_sid, "parent_span_id": root_sid,
         "span_name": "cache.get", "db_operation_type": "",
         "span_duration_ms": round(cache_ms, 2), "span_status": cache_status, "peer_service": ""},
        {**base, "span_id": db_sid, "parent_span_id": root_sid,
         "span_name": "db.query", "db_operation_type": db_op,
         "span_duration_ms": round(db_ms, 2), "span_status": db_status, "peer_service": peer},
    ]
    return rows


# ---------------------------------------------------------------------------
# Episode generator -- produces metrics, logs, trace rows
# ---------------------------------------------------------------------------

def generate_episode(
    episode_id: str,
    failure_mode: str,
    config: Config,
    dist: Dist,
    rng: np.random.Generator,
) -> Tuple[List[dict], List[dict], List[dict]]:

    metrics, logs, trace_rows = [], [], []
    service = str(rng.choice(SERVICES))
    svc_version = str(rng.choice(SERVICE_VERSIONS))
    source = "python generator"   # metric source label
    base_ts = int(rng.integers(1_700_000_000, 1_700_500_000))

    for step in range(config.steps_per_episode):
        ts = base_ts + step * config.step_interval_s
        elapsed = step * config.step_interval_s

        # ----------------------------------------------------------------
        # Compute the full 33-column metric vector per failure mode
        # ----------------------------------------------------------------

        if failure_mode == "NONE":
            cpu_u = dist.cpu_util()
            heap_v = dist.heap()
            gc_v = dist.gc_p99()
            p50_v = dist.p50(); p99_v = dist.p99()
            p95_v = dist.p95(p50_v, p99_v)
            err_v = dist.err_rate()
            miss_v = dist.cache_miss(); hit_v = 1.0 - miss_v
            db_v = dist.db_p99()
            q_lag = dist.queue_lag()
            retry_v = dist.retry_cnt()
            rps_v = dist.rps()
            conn_v = dist.active_conn()
            db_pool = dist.db_conn_pool(); db_wait = dist.db_conn_wait()
            dr = dist.disk_read_lat(); dw = dist.disk_write_lat()
            iops = dist.iops_util()
            mem_u = dist.mem_util(heap_v)
            net_e = dist.net_errors()
            h4xx = dist.http_4xx(); h5xx = dist.http_5xx()
            tp_q = dist.thread_pool_q()
            ut = dist.upstream_timeout()
            cb = dist.circuit_breaker(err_v)
            cpu_sat = cpu_u / 100.0

        elif failure_mode == "MEMORY_LEAK":
            heap_v = dist.heap_leak(elapsed)
            gc_v = dist.gc_leak(heap_v)
            cpu_u = float(np.clip(dist.cpu_util() + (heap_v - 512) / 80, 15, 90))
            p99_v = float(np.clip(dist.p99() + gc_v * 0.4, 80, 2000))
            p50_v = dist.p50(); p95_v = dist.p95(p50_v, p99_v)
            err_v = dist.err_rate()
            miss_v = dist.cache_miss(); hit_v = 1.0 - miss_v
            db_v = dist.db_p99()
            q_lag = dist.queue_lag(); retry_v = dist.retry_cnt(); rps_v = dist.rps()
            conn_v = dist.active_conn()
            db_pool = float(np.clip(dist.db_conn_pool() + heap_v / 5000, 0.1, 0.9))
            db_wait = dist.db_conn_wait()
            dr = dist.disk_read_lat(); dw = dist.disk_write_lat()
            iops = dist.iops_util()
            mem_u = dist.mem_util(heap_v)
            net_e = dist.net_errors()
            h4xx = dist.http_4xx(); h5xx = float(np.clip(dist.http_5xx() + gc_v / 10000, 0, 0.05))
            tp_q = int(np.clip(dist.thread_pool_q() + gc_v / 50, 0, 200))
            ut = dist.upstream_timeout()
            cb = dist.circuit_breaker(err_v)
            cpu_sat = cpu_u / 100.0

        elif failure_mode == "CPU_SATURATION":
            cpu_u = dist.cpu_sat_util()
            cpu_sat = cpu_u / 100.0
            p99_v = dist.p99_cpu_sat(cpu_u)
            p50_v = float(np.clip(dist.p50() + (cpu_u - 35) * 2, 80, 500))
            p95_v = dist.p95(p50_v, p99_v)
            heap_v = dist.heap(); gc_v = dist.gc_p99()
            err_v = float(np.clip(dist.err_rate() + (cpu_u - 70) * 0.002, 0, 0.15))
            miss_v = dist.cache_miss(); hit_v = 1.0 - miss_v
            db_v = dist.db_p99()
            q_lag = int(np.clip(dist.queue_lag() + cpu_u * 0.5, 0, 200))
            retry_v = dist.retry_cnt(); rps_v = dist.rps()
            conn_v = int(np.clip(dist.active_conn() + cpu_u * 0.5, 20, 500))
            db_pool = dist.db_conn_pool(); db_wait = float(np.clip(dist.db_conn_wait() + cpu_u * 0.1, 0.5, 100))
            dr = dist.disk_read_lat(); dw = dist.disk_write_lat()
            iops = dist.iops_util()
            mem_u = dist.mem_util(heap_v)
            net_e = dist.net_errors()
            h4xx = dist.http_4xx()
            h5xx = float(np.clip(dist.http_5xx() + (cpu_u - 70) * 0.001, 0, 0.1))
            tp_q = int(np.clip(dist.thread_pool_q() + cpu_u * 1.5, 0, 500))
            ut = float(np.clip(dist.upstream_timeout() + (cpu_u - 70) * 0.002, 0, 0.3))
            cb = dist.circuit_breaker(err_v)

        elif failure_mode == "LATENCY_SPIKE":
            gc_fired = dist.gc_event(0.08)
            heap_v = dist.heap(); cpu_u = dist.cpu_util(); cpu_sat = cpu_u / 100.0
            if gc_fired:
                gc_v = dist.gc_spike()
                p99_v = dist.p99_spike(gc_v)
                p50_v = float(np.clip(rng.normal(92, 4), 70, 120))
                err_v = float(np.clip(dist.err_rate() + rng.uniform(0.01, 0.04), 0, 0.15))
            else:
                gc_v = dist.gc_p99(); p99_v = dist.p99(); p50_v = dist.p50()
                err_v = dist.err_rate()
            p95_v = dist.p95(p50_v, p99_v)
            miss_v = dist.cache_miss(); hit_v = 1.0 - miss_v; db_v = dist.db_p99()
            q_lag = dist.queue_lag(); retry_v = dist.retry_cnt(); rps_v = dist.rps()
            conn_v = dist.active_conn()
            db_pool = dist.db_conn_pool(); db_wait = dist.db_conn_wait()
            dr = dist.disk_read_lat(); dw = dist.disk_write_lat()
            iops = dist.iops_util(); mem_u = dist.mem_util(heap_v)
            net_e = dist.net_errors(); h4xx = dist.http_4xx()
            h5xx = float(np.clip(dist.http_5xx() + (err_v * 0.05), 0, 0.1))
            tp_q = dist.thread_pool_q(); ut = dist.upstream_timeout()
            cb = dist.circuit_breaker(err_v)

        elif failure_mode == "ERROR_STORM":
            err_v = dist.err_storm()
            h5xx = float(np.clip(err_v * 0.7, 0, 0.7))
            h4xx = float(np.clip(err_v * 0.3, 0, 0.3))
            p99_v = float(np.clip(rng.normal(130, 20), 80, 300))
            p50_v = dist.p50(); p95_v = dist.p95(p50_v, p99_v)
            heap_v = dist.heap(); gc_v = dist.gc_p99(); cpu_u = dist.cpu_util()
            miss_v = dist.cache_miss(); hit_v = 1.0 - miss_v; db_v = dist.db_p99()
            q_lag = int(np.clip(dist.queue_lag() + err_v * 50, 0, 200))
            retry_v = float(np.clip(err_v * 0.5, 0, 0.5))
            rps_v = dist.rps()
            conn_v = dist.active_conn()
            db_pool = dist.db_conn_pool(); db_wait = dist.db_conn_wait()
            dr = dist.disk_read_lat(); dw = dist.disk_write_lat()
            iops = dist.iops_util(); mem_u = dist.mem_util(heap_v)
            net_e = int(np.clip(dist.net_errors() + err_v * 20, 0, 50))
            tp_q = int(np.clip(dist.thread_pool_q() + err_v * 30, 0, 100))
            ut = float(np.clip(dist.upstream_timeout() + err_v * 0.2, 0, 0.5))
            cb = dist.circuit_breaker(err_v)
            cpu_sat = cpu_u / 100.0

        elif failure_mode == "DB_SLOWDOWN":
            db_v = dist.db_slow(elapsed)
            p99_v = float(np.clip(db_v * 1.2 + rng.normal(30, 10), db_v, db_v * 1.5))
            p50_v = dist.p50(); p95_v = dist.p95(p50_v, p99_v)
            heap_v = dist.heap(); gc_v = dist.gc_p99(); cpu_u = dist.cpu_util()
            err_v = float(np.clip(dist.err_rate() + (db_v - 80) * 0.0002, 0, 0.15))
            miss_v = dist.cache_miss(); hit_v = 1.0 - miss_v
            q_lag = dist.queue_lag(); retry_v = dist.retry_cnt(); rps_v = dist.rps()
            conn_v = int(np.clip(dist.active_conn() + db_v * 0.05, 20, 500))
            db_pool = float(np.clip(dist.db_conn_pool() + db_v / 2000, 0.1, 0.99))
            db_wait = float(np.clip(db_v * 0.3 + rng.normal(0, 10), 1, 500))
            dr = dist.disk_read_lat()
            dw = float(np.clip(dist.disk_write_lat() + db_v * 0.01, 2, 100))
            iops = float(np.clip(dist.iops_util() + db_v / 5000, 0.05, 0.95))
            mem_u = dist.mem_util(heap_v)
            net_e = dist.net_errors()
            h4xx = dist.http_4xx()
            h5xx = float(np.clip(dist.http_5xx() + err_v * 0.5, 0, 0.2))
            tp_q = int(np.clip(dist.thread_pool_q() + db_v * 0.01, 0, 100))
            ut = float(np.clip(dist.upstream_timeout() + db_v / 10000, 0, 0.3))
            cb = dist.circuit_breaker(err_v)
            cpu_sat = cpu_u / 100.0

        elif failure_mode == "CACHE_STAMPEDE":
            miss_v = dist.cache_miss_stampede(); hit_v = 1.0 - miss_v
            db_v = dist.db_p99_mm1(miss_v, config.base_rps, config.db_capacity_rps)
            p99_v = float(np.clip(db_v + rng.normal(40, 10), db_v * 0.9, db_v * 1.2))
            p50_v = dist.p50(); p95_v = dist.p95(p50_v, p99_v)
            heap_v = dist.heap(); gc_v = dist.gc_p99()
            cpu_u = float(np.clip(dist.cpu_util() + rng.uniform(10, 25), 30, 85))
            cpu_sat = cpu_u / 100.0
            err_v = float(np.clip(rng.beta(3, 12), 0.10, 0.35))
            q_lag = dist.queue_lag(); retry_v = dist.retry_cnt(); rps_v = dist.rps()
            conn_v = int(np.clip(dist.active_conn() + miss_v * 200, 20, 500))
            db_pool = float(np.clip(dist.db_conn_pool() + miss_v * 0.5, 0.1, 0.99))
            db_wait = float(np.clip(db_v * 0.4, 1, 500))
            dr = dist.disk_read_lat(); dw = dist.disk_write_lat()
            iops = float(np.clip(dist.iops_util() + miss_v * 0.3, 0.05, 0.95))
            mem_u = dist.mem_util(heap_v)
            net_e = dist.net_errors()
            h4xx = dist.http_4xx()
            h5xx = float(np.clip(dist.http_5xx() + err_v * 0.3, 0, 0.3))
            tp_q = int(np.clip(dist.thread_pool_q() + miss_v * 50, 0, 200))
            ut = dist.upstream_timeout()
            cb = dist.circuit_breaker(err_v)

        elif failure_mode == "QUEUE_BACKUP":
            q_lag = dist.queue_lag_backup(elapsed)
            p99_v = float(np.clip(rng.normal(90 + q_lag * 1.5, 20), 80, 2000))
            p50_v = float(np.clip(dist.p50() + q_lag * 0.3, 80, 500))
            p95_v = dist.p95(p50_v, p99_v)
            heap_v = dist.heap(); gc_v = dist.gc_p99(); cpu_u = dist.cpu_util()
            err_v = dist.err_rate(); miss_v = dist.cache_miss(); hit_v = 1.0 - miss_v
            db_v = dist.db_p99(); retry_v = dist.retry_cnt(); rps_v = dist.rps()
            conn_v = int(np.clip(dist.active_conn() + q_lag * 0.5, 20, 500))
            db_pool = dist.db_conn_pool()
            db_wait = float(np.clip(dist.db_conn_wait() + q_lag * 0.2, 0.5, 200))
            dr = dist.disk_read_lat(); dw = dist.disk_write_lat()
            iops = dist.iops_util(); mem_u = dist.mem_util(heap_v)
            net_e = dist.net_errors()
            h4xx = dist.http_4xx(); h5xx = dist.http_5xx()
            tp_q = int(np.clip(q_lag * 2, 0, 500))
            ut = dist.upstream_timeout()
            cb = dist.circuit_breaker(err_v)
            cpu_sat = cpu_u / 100.0

        elif failure_mode == "DEPENDENCY_TIMEOUT":
            dep_rate = dist.timeout_rate()
            ut = dep_rate
            err_v = float(np.clip(dep_rate * 0.8, 0, 0.65))
            p99_v = float(np.clip(rng.normal(300 + dep_rate * 1000, 80), 200, 3000))
            p50_v = dist.p50(); p95_v = dist.p95(p50_v, p99_v)
            heap_v = dist.heap(); gc_v = dist.gc_p99(); cpu_u = dist.cpu_util()
            miss_v = dist.cache_miss(); hit_v = 1.0 - miss_v; db_v = dist.db_p99()
            q_lag = dist.queue_lag()
            retry_v = float(np.clip(dep_rate * 0.5, 0, 0.5)); rps_v = dist.rps()
            conn_v = dist.active_conn()
            db_pool = dist.db_conn_pool(); db_wait = dist.db_conn_wait()
            dr = dist.disk_read_lat(); dw = dist.disk_write_lat()
            iops = dist.iops_util(); mem_u = dist.mem_util(heap_v)
            net_e = int(np.clip(dist.net_errors() + dep_rate * 30, 0, 60))
            h4xx = float(np.clip(dist.http_4xx() + dep_rate * 0.1, 0, 0.2))
            h5xx = float(np.clip(dist.http_5xx() + dep_rate * 0.4, 0, 0.6))
            tp_q = dist.thread_pool_q()
            cb = dist.circuit_breaker(err_v)
            cpu_sat = cpu_u / 100.0

        elif failure_mode == "BAD_DEPLOY":
            err_v = dist.err_deploy()
            h5xx = float(np.clip(err_v * 0.8, 0, 0.8))
            h4xx = float(np.clip(err_v * 0.2, 0, 0.2))
            p99_v = float(np.clip(dist.p99() + err_v * 100, 90, 500))
            p50_v = dist.p50(); p95_v = dist.p95(p50_v, p99_v)
            heap_v = dist.heap(); gc_v = dist.gc_p99(); cpu_u = dist.cpu_util()
            miss_v = dist.cache_miss(); hit_v = 1.0 - miss_v; db_v = dist.db_p99()
            q_lag = dist.queue_lag(); retry_v = dist.retry_cnt(); rps_v = dist.rps()
            conn_v = dist.active_conn()
            db_pool = dist.db_conn_pool(); db_wait = dist.db_conn_wait()
            dr = dist.disk_read_lat(); dw = dist.disk_write_lat()
            iops = dist.iops_util(); mem_u = dist.mem_util(heap_v)
            net_e = int(np.clip(dist.net_errors() + err_v * 15, 0, 40))
            tp_q = dist.thread_pool_q()
            ut = float(np.clip(dist.upstream_timeout() + err_v * 0.1, 0, 0.3))
            cb = dist.circuit_breaker(err_v)
            cpu_sat = cpu_u / 100.0

        elif failure_mode == "RETRY_STORM":
            retry_v = dist.retry_high()
            rps_v = dist.rps_retry(retry_v, config.base_rps)
            p99_v = float(np.clip(dist.p99() + retry_v * 200, 90, 1000))
            p50_v = dist.p50(); p95_v = dist.p95(p50_v, p99_v)
            err_v = dist.err_rate()
            heap_v = float(np.clip(dist.heap() + retry_v * 50, 400, 800))
            gc_v = dist.gc_p99()
            cpu_u = float(np.clip(dist.cpu_util() + retry_v * 30, 20, 90))
            miss_v = dist.cache_miss(); hit_v = 1.0 - miss_v; db_v = dist.db_p99()
            q_lag = int(np.clip(retry_v * 100, 0, 200))
            conn_v = int(np.clip(dist.active_conn() + retry_v * 200, 20, 800))
            db_pool = float(np.clip(dist.db_conn_pool() + retry_v * 0.5, 0.1, 0.99))
            db_wait = float(np.clip(dist.db_conn_wait() + retry_v * 50, 0.5, 300))
            dr = dist.disk_read_lat(); dw = dist.disk_write_lat()
            iops = float(np.clip(dist.iops_util() + retry_v * 0.2, 0.05, 0.95))
            mem_u = dist.mem_util(heap_v)
            net_e = int(np.clip(dist.net_errors() + retry_v * 10, 0, 30))
            h4xx = dist.http_4xx()
            h5xx = float(np.clip(dist.http_5xx() + retry_v * 0.05, 0, 0.15))
            tp_q = int(np.clip(retry_v * 200, 0, 1000))
            ut = float(np.clip(dist.upstream_timeout() + retry_v * 0.1, 0, 0.5))
            cb = dist.circuit_breaker(err_v)
            cpu_sat = cpu_u / 100.0

        elif failure_mode == "DISK_IO_SATURATION":
            dr = dist.disk_lat_sat(); dw = float(np.clip(dr * 1.2 + rng.normal(0, 20), 50, 3000))
            iops = dist.iops_sat()
            db_v = dist.db_slow_disk(elapsed)
            cpu_u = float(np.clip(rng.normal(22, 4), 8, 40))  # low CPU -- waiting on IO
            cpu_sat = cpu_u / 100.0
            p99_v = float(np.clip(db_v * 1.1 + rng.normal(20, 10), db_v * 0.8, db_v * 1.5))
            p50_v = float(np.clip(dist.p50() + db_v * 0.1, 80, 500))
            p95_v = dist.p95(p50_v, p99_v)
            heap_v = dist.heap(); gc_v = dist.gc_p99()
            err_v = dist.err_rate(); miss_v = dist.cache_miss(); hit_v = 1.0 - miss_v
            q_lag = dist.queue_lag(); retry_v = dist.retry_cnt(); rps_v = dist.rps()
            conn_v = dist.active_conn()
            db_pool = float(np.clip(dist.db_conn_pool() + iops * 0.5, 0.1, 0.99))
            db_wait = float(np.clip(dr * 0.5 + rng.normal(0, 10), 1, 500))
            mem_u = dist.mem_util(heap_v)
            net_e = dist.net_errors()
            h4xx = dist.http_4xx(); h5xx = dist.http_5xx()
            tp_q = int(np.clip(dist.thread_pool_q() + iops * 20, 0, 100))
            ut = dist.upstream_timeout()
            cb = dist.circuit_breaker(err_v)

        else:  # CASCADING_FAILURE
            cpu_u = dist.cascade_cpu(); cpu_sat = cpu_u / 100.0
            err_v = dist.cascade_err()
            p99_v = dist.cascade_p99(cpu_u, err_v)
            p50_v = float(np.clip(dist.p50() + cpu_u * 3, 100, 1000))
            p95_v = dist.p95(p50_v, p99_v)
            heap_v = float(np.clip(dist.heap() + cpu_u * 2, 400, 1000))
            gc_v = float(np.clip(dist.gc_p99() + cpu_u * 2, 10, 300))
            miss_v = float(np.clip(dist.cache_miss() + err_v * 0.4, 0.05, 0.80))
            hit_v = 1.0 - miss_v
            db_v = float(np.clip(dist.db_p99() + p99_v * 0.3, 50, 3000))
            q_lag = int(np.clip(cpu_u * 3, 50, 500))
            retry_v = float(np.clip(err_v * 0.6, 0, 0.6)); rps_v = dist.rps()
            conn_v = int(np.clip(dist.active_conn() + cpu_u * 3, 20, 1000))
            db_pool = float(np.clip(dist.db_conn_pool() + err_v * 0.5, 0.1, 0.99))
            db_wait = float(np.clip(db_v * 0.5, 1, 1000))
            dr = float(np.clip(dist.disk_read_lat() + cpu_u * 2, 1, 200))
            dw = float(np.clip(dist.disk_write_lat() + cpu_u * 2, 2, 300))
            iops = float(np.clip(dist.iops_util() + err_v * 0.3, 0.05, 0.99))
            mem_u = dist.mem_util(heap_v)
            net_e = int(np.clip(dist.net_errors() + err_v * 40, 0, 100))
            h4xx = float(np.clip(dist.http_4xx() + err_v * 0.3, 0, 0.5))
            h5xx = float(np.clip(dist.http_5xx() + err_v * 0.6, 0, 0.8))
            tp_q = int(np.clip(dist.thread_pool_q() + cpu_u * 5, 0, 1000))
            ut = float(np.clip(dist.upstream_timeout() + err_v * 0.3, 0, 0.6))
            cb = dist.circuit_breaker(err_v)

        # ---- Build metric row (33 columns) ----
        metric_row = {
            "episode_id": episode_id,
            "failure_mode": failure_mode,
            "service": service,
            "source": source,
            "elapsed_s": elapsed,
            "timestamp": ts,
            "active_connections": conn_v,
            "cache_hit_rate": round(hit_v, 4),
            "cache_miss_rate": round(miss_v, 4),
            "circuit_breaker_state": cb,
            "cpu_saturation": round(cpu_sat, 4),
            "cpu_utilization": round(cpu_u, 2),
            "db_connection_pool": round(db_pool, 4),
            "db_connection_wait": round(db_wait, 2),
            "db_p99": round(db_v, 2),
            "disk_read_latency": round(dr, 2),
            "disk_write_latency": round(dw, 2),
            "error_rate": round(err_v, 4),
            "gc_pause_p99": round(gc_v, 2),
            "heap_mb": round(heap_v, 1),
            "http_4xx_rate": round(h4xx, 4),
            "http_5xx_rate": round(h5xx, 4),
            "iops_utilization": round(iops, 4),
            "memory_utilization": round(mem_u, 4),
            "network_errors": net_e,
            "p50_latency": round(p50_v, 2),
            "p95_latency": round(p95_v, 2),
            "p99_latency": round(p99_v, 2),
            "queue_lag": q_lag,
            "retry_count_per_request": round(retry_v, 4),
            "rps": round(rps_v, 2),
            "thread_pool_queue": tp_q,
            "upstream_timeout_rate": round(ut, 4),
        }
        metrics.append(metric_row)

        # ---- Build log row ----
        log_vals = {
            "cpu": cpu_u, "p99": p99_v, "err": err_v * 100,
            "heap": heap_v, "gc": gc_v, "h5": h5xx * 100,
            "db": db_v, "miss": miss_v * 100,
            "lag": q_lag, "tp": tp_q,
            "ut": ut * 100, "rc": retry_v,
            "rps": rps_v, "dr": dr, "iops": iops * 100,
        }
        logs.append({
            "episode_id": episode_id,
            "failure_mode": failure_mode,
            "service": service,
            "elapsed_s": elapsed,
            "timestamp": ts,
            "log_level": LOG_LEVELS[failure_mode],
            "exception_type": EXCEPTION_MAP.get(failure_mode, ""),
            "log_message": build_log_msg(failure_mode, log_vals),
        })

        # ---- Build trace rows (every 4 steps, 3 spans each) ----
        if step % 4 == 0:
            spans = build_trace_rows(
                episode_id, failure_mode, service, elapsed, ts, rng, svc_version
            )
            trace_rows.extend(spans)

    return metrics, logs, trace_rows


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def write_csv(rows, path, fields, write_header=True, file_mode="w"):
    with open(path, file_mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate ~120k-row telemetry dataset (13 modes x 77 eps, full schema)")
    parser.add_argument("--episodes", type=int, default=77,
                        help="Episodes per failure mode (default: 77 → 120,120 total metric rows)")
    parser.add_argument("--steps",    type=int, default=120,
                        help="Steps per episode at 2s interval (default: 120 = 4 min)")
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--output",   type=str, default="data")
    args = parser.parse_args()

    config = Config(
        seed=args.seed,
        episodes_per_mode=args.episodes,
        steps_per_episode=args.steps,
        output_dir=args.output,
    )

    rng = np.random.default_rng(config.seed)
    dist = Dist(rng)
    os.makedirs(config.output_dir, exist_ok=True)

    metric_path = os.path.join(config.output_dir, "telemetry_metrics.csv")
    log_path    = os.path.join(config.output_dir, "telemetry_logs.csv")
    trace_path  = os.path.join(config.output_dir, "telemetry_traces.csv")

    expected = len(ALL_MODES) * config.episodes_per_mode * config.steps_per_episode
    print("=" * 65)
    print("AIOps Full Telemetry Generator  (~120k rows)")
    print("  Modes: {}  |  Eps/mode: {}  |  Steps: {}".format(
        len(ALL_MODES), config.episodes_per_mode, config.steps_per_episode))
    print("  Expected metric rows  : {:,}".format(expected))
    print("  Expected trace rows   : ~{:,}".format(
        len(ALL_MODES) * config.episodes_per_mode * (config.steps_per_episode // 4) * 3))
    print("  Expected log rows     : {:,}".format(expected))
    print("=" * 65)

    global_ep = 0
    mode_counts = {}
    first_mode = True

    for mode in ALL_MODES:
        idx = ALL_MODES.index(mode) + 1
        print("\n[{:02d}/13] {} ({} episodes)...".format(idx, mode, config.episodes_per_mode))

        m_all, l_all, t_all = [], [], []

        for i in range(config.episodes_per_mode):
            ep_id = "ep_{:05d}_{}".format(global_ep, mode)
            global_ep += 1
            m, l, t = generate_episode(ep_id, mode, config, dist, rng)
            m_all.extend(m); l_all.extend(l); t_all.extend(t)

        mode_counts[mode] = len(m_all)

        fm = "w" if first_mode else "a"
        write_csv(m_all, metric_path, METRIC_FIELDS, write_header=first_mode, file_mode=fm)
        write_csv(l_all, log_path,    LOG_FIELDS,    write_header=first_mode, file_mode=fm)
        write_csv(t_all, trace_path,  TRACE_FIELDS,  write_header=first_mode, file_mode=fm)

        print("     metrics={} | logs={} | trace_spans={}".format(
            len(m_all), len(l_all), len(t_all)))
        first_mode = False

    total = sum(mode_counts.values())
    print("\n" + "=" * 65)
    print("Done. Row counts by failure mode:")
    for mode, cnt in mode_counts.items():
        print("  {:<22}: {:>6,} rows".format(mode, cnt))
    print("\n  TOTAL metric rows : {:,}".format(total))
    print("  Output           : {}".format(os.path.abspath(config.output_dir)))
    print("  Files:")
    for f in [metric_path, log_path, trace_path]:
        size = os.path.getsize(f) / 1024
        print("    {:50} {:>8.1f} KB".format(os.path.basename(f), size))
    print("=" * 65)


if __name__ == "__main__":
    main()
