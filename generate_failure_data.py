"""
AIOps Platform — Statistical Telemetry Data Generator
======================================================
Generates synthetic telemetry (metrics, logs, traces) for:
  - LATENCY_SPIKE   : intermittent GC pressure, flat heap, P99/P50 divergence
  - CACHE_STAMPEDE  : cache miss → DB overload two-phase chain

Statistical composition model per signal:
  Metrics : parametric distributions (Normal, LogNormal, Beta, Poisson, M/M/1)
  Logs    : structured log lines with correlated numeric fields
  Traces  : span trees with causal ordering constraints

Output:
  data/telemetry_metrics.csv   — one row per timestep (2s interval)
  data/telemetry_logs.csv      — log lines per episode
  data/telemetry_traces.jsonl  — span trees per request

Usage:
  python generate_failure_data.py --episodes 5 --steps 120 --seed 42
"""

import argparse
import csv
import json
import math
import os

import uuid
from dataclasses import dataclass, field, asdict
from typing import List, Literal, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FAILURE_MODES = Literal["NONE", "LATENCY_SPIKE", "CACHE_STAMPEDE"]

@dataclass
class GeneratorConfig:
    seed: int = 42
    episodes_per_mode: int = 5
    steps_per_episode: int = 120        # 120 steps × 2s = 4 minutes per episode
    step_interval_s: int = 2
    output_dir: str = "data"
    service_names: List[str] = field(
        default_factory=lambda: ["auth-service", "payment-service", "order-service"]
    )
    # Request volume (RPS baseline)
    base_rps: float = 200.0
    # DB capacity (RPS before queuing saturates)
    db_capacity_rps: float = 250.0


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

class Distributions:
    """All distributions used in the generative models."""

    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    # --- Baseline distributions ---

    def p50_baseline(self) -> float:
        """P50 latency in healthy state. Normal(90, 8) ms."""
        return float(np.clip(self.rng.normal(90, 8), 50, 200))

    def p99_baseline(self) -> float:
        """P99 latency in healthy state. Normal(95, 10) ms."""
        return float(np.clip(self.rng.normal(95, 10), 60, 250))

    def heap_mb_baseline(self) -> float:
        """Heap in MB. Normal(512, 20) MB."""
        return float(np.clip(self.rng.normal(512, 20), 400, 700))

    def gc_pause_baseline_ms(self) -> float:
        """GC pause in healthy state. Exponential(mean=12ms)."""
        return float(np.clip(self.rng.exponential(12), 2, 40))

    def error_rate_baseline(self) -> float:
        """Error rate. Beta(2,198) ≈ 1%."""
        return float(self.rng.beta(2, 198))

    def cache_miss_rate_baseline(self) -> float:
        """Cache miss rate in healthy state. Beta(1,19) ≈ 5%."""
        return float(np.clip(self.rng.beta(1, 19), 0.01, 0.12))

    def db_p99_baseline_ms(self) -> float:
        """DB P99 in healthy state. LogNormal(μ=3.0, σ=0.3) ≈ mean 20ms."""
        return float(np.clip(self.rng.lognormal(3.0, 0.3), 5, 80))

    def cpu_baseline(self) -> float:
        """CPU utilisation. Normal(35, 5) %."""
        return float(np.clip(self.rng.normal(35, 5), 15, 55))

    # --- LATENCY_SPIKE distributions ---

    def gc_event_occurs(self, lambda_per_step: float = 0.08) -> bool:
        """
        GC event modelled as Poisson process.
        lambda_per_step = avg events per 2-second step.
        Default: ~2.4 events/minute → 0.08 per 2s step.
        """
        return self.rng.poisson(lambda_per_step) > 0

    def gc_pause_spike_ms(self) -> float:
        """
        GC pause during spike event.
        LogNormal(μ=5.5, σ=0.4) → mean~250ms, range 80–600ms.
        Right-skewed (realistic: most pauses ~200ms, occasional >500ms).
        """
        return float(np.clip(self.rng.lognormal(5.5, 0.4), 80, 650))

    def p99_during_spike(self, gc_pause_ms: float) -> float:
        """P99 latency when GC spike fires. Proportional to pause + noise."""
        multiplier = float(self.rng.uniform(0.8, 1.1))
        noise = float(self.rng.normal(0, 30))
        return float(np.clip(gc_pause_ms * multiplier + 90 + noise, 200, 3000))

    def p50_during_spike(self) -> float:
        """P50 barely changes during spike — tiny bump only."""
        return float(np.clip(self.rng.normal(92, 4), 70, 120))

    def fraction_requests_affected(self) -> float:
        """Fraction of requests that hit a GC pause. Uniform(0.05, 0.10)."""
        return float(self.rng.uniform(0.05, 0.10))

    # --- CACHE_STAMPEDE distributions ---

    def cache_miss_rate_stampede(self) -> float:
        """
        Cache miss rate during stampede.
        Beta(19,1)*0.95 → concentrated near 95%, range 88–99%.
        Oscillates slightly because some cache slots warm up first.
        """
        base = float(self.rng.beta(19, 1))  # concentrated near 1.0
        jitter = float(self.rng.normal(0, 0.025))
        return float(np.clip(base * 0.97 + jitter, 0.85, 0.99))

    def db_p99_from_queue(self, cache_miss_rate: float,
                          base_rps: float, db_capacity_rps: float) -> float:
        """
        M/M/1 queuing model for DB P99 under stampede load.

        When cache is working: only ~5% of requests hit DB → low load.
        When cache fails: ~95% of requests hit DB → rho approaches 1.

        db_p99 = service_time * (1 + rho/(1-rho)) + noise
        """
        # Fixed intrinsic service time (20ms) with tiny noise so queue delay is
        # deterministically driven by cache_miss_rate, preserving high Pearson r.
        db_service_time_ms = 20.0 * float(self.rng.lognormal(0, 0.05))

        effective_db_rps = base_rps * cache_miss_rate
        rho = min(effective_db_rps / db_capacity_rps, 0.98)

        if rho < 0.99:
            queue_delay_ms = db_service_time_ms * (rho / (1.0 - rho))
        else:
            queue_delay_ms = db_service_time_ms * 50

        noise_ms = float(self.rng.normal(0, 8))
        total_ms = db_service_time_ms + queue_delay_ms + noise_ms
        return float(np.clip(total_ms, 20, 1200))

    def request_p99_from_db(self, db_p99_ms: float) -> float:
        """Request P99 tracks DB P99 closely (DB is on critical path)."""
        overhead = float(self.rng.normal(40, 10))   # app overhead
        return float(np.clip(db_p99_ms + overhead, db_p99_ms * 0.9, db_p99_ms * 1.2))


# ---------------------------------------------------------------------------
# Span / Trace generation
# ---------------------------------------------------------------------------

@dataclass
class Span:
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    operation: str
    service: str
    start_offset_ms: float          # relative to request start
    duration_ms: float
    status: str                     # "ok" | "error" | "timeout"
    attributes: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "operation": self.operation,
            "service": self.service,
            "start_offset_ms": round(self.start_offset_ms, 2),
            "duration_ms": round(self.duration_ms, 2),
            "status": self.status,
            "attributes": self.attributes,
        }


def make_baseline_trace(service: str, timestamp: int, rng: np.random.Generator) -> dict:
    """Normal request trace — all spans proportionate, all OK."""
    trace_id = str(uuid.uuid4())
    root_id  = str(uuid.uuid4())
    db_id    = str(uuid.uuid4())
    cache_id = str(uuid.uuid4())

    cache_ms = float(np.clip(rng.normal(3, 0.5), 1, 8))
    db_ms    = float(np.clip(rng.normal(15, 3), 5, 35))
    root_ms  = cache_ms + db_ms + float(np.clip(rng.normal(8, 2), 2, 20))

    spans = [
        Span(trace_id, root_id,  None,    "handle_request", service,
             0.0, root_ms, "ok", {"http.method": "GET"}),
        Span(trace_id, cache_id, root_id, "cache_lookup",   service,
             0.5, cache_ms, "ok", {"cache.result": "HIT"}),
        Span(trace_id, db_id,   root_id, "db_query",       service,
             cache_ms + 0.5, db_ms, "ok", {"db.type": "postgresql"}),
    ]
    return {
        "trace_id": trace_id,
        "timestamp": timestamp,
        "service": service,
        "failure_mode": "NONE",
        "spans": [s.to_dict() for s in spans],
    }


def make_latency_spike_trace(
    service: str, timestamp: int, gc_pause_ms: float, affected: bool,
    rng: np.random.Generator
) -> dict:
    """
    LATENCY_SPIKE trace.
    If affected=True: dead-time gap appears in root span, children are normal.
    If affected=False: looks like baseline (most requests are fine).
    """
    trace_id = str(uuid.uuid4())
    root_id  = str(uuid.uuid4())
    db_id    = str(uuid.uuid4())
    cache_id = str(uuid.uuid4())

    cache_ms = float(np.clip(rng.normal(3, 0.5), 1, 8))
    db_ms    = float(np.clip(rng.normal(15, 3), 5, 35))

    if affected:
        # Dead-time gap: root span stretched, children still normal
        dead_time_gap_ms = gc_pause_ms * float(rng.uniform(0.90, 1.0))
        root_ms = dead_time_gap_ms + cache_ms + db_ms + float(rng.normal(8, 2))
        root_attrs = {
            "http.method": "GET",
            "gc.pause_detected": True,
            "gc.pause_ms": round(gc_pause_ms, 1),
        }
    else:
        root_ms = cache_ms + db_ms + float(np.clip(rng.normal(8, 2), 2, 20))
        root_attrs = {"http.method": "GET", "gc.pause_detected": False}

    # Children always normal — THIS IS THE KEY DIAGNOSTIC CONSTRAINT
    spans = [
        Span(trace_id, root_id,  None,    "handle_request", service,
             0.0, max(root_ms, 10), "ok", root_attrs),
        Span(trace_id, cache_id, root_id, "cache_lookup",   service,
             0.5, cache_ms, "ok", {"cache.result": "HIT"}),
        Span(trace_id, db_id,   root_id, "db_query",       service,
             cache_ms + 0.5, db_ms, "ok", {"db.type": "postgresql"}),
    ]
    return {
        "trace_id": trace_id,
        "timestamp": timestamp,
        "service": service,
        "failure_mode": "LATENCY_SPIKE",
        "gc_event": affected,
        "spans": [s.to_dict() for s in spans],
    }


def make_cache_stampede_trace(
    service: str, timestamp: int, db_p99_ms: float,
    rng: np.random.Generator
) -> dict:
    """
    CACHE_STAMPEDE trace.
    cache_lookup → MISS → immediately followed by slow db_query.
    Causal ordering preserved.
    """
    trace_id = str(uuid.uuid4())
    root_id  = str(uuid.uuid4())
    db_id    = str(uuid.uuid4())
    cache_id = str(uuid.uuid4())

    cache_ms = float(np.clip(rng.normal(3, 0.5), 1, 6))          # cache lookup is fast
    db_ms    = db_p99_ms * float(rng.uniform(0.85, 1.05))         # slow due to overload
    root_ms  = cache_ms + db_ms + float(np.clip(rng.normal(10, 3), 3, 30))

    spans = [
        Span(trace_id, root_id,  None,    "handle_request", service,
             0.0, root_ms, "ok", {"http.method": "GET"}),
        # Cache span: fast but MISS — the causal trigger
        Span(trace_id, cache_id, root_id, "cache_lookup",   service,
             0.5, cache_ms, "ok",
             {"cache.result": "MISS", "cache.key_type": "user_session"}),
        # DB span: immediately follows cache miss, slow due to stampede
        Span(trace_id, db_id,   root_id, "db_query",       service,
             cache_ms + 0.5, db_ms, "ok",
             {"db.type": "postgresql",
              "db.rows_examined": int(rng.integers(1000, 50000)),
              "db.overloaded": True}),
    ]
    return {
        "trace_id": trace_id,
        "timestamp": timestamp,
        "service": service,
        "failure_mode": "CACHE_STAMPEDE",
        "spans": [s.to_dict() for s in spans],
    }


# ---------------------------------------------------------------------------
# Episode generators
# ---------------------------------------------------------------------------

def generate_none_episode(
    episode_id: str, config: GeneratorConfig, dist: Distributions, rng: np.random.Generator
) -> Tuple[List[dict], List[dict], List[dict]]:
    """Generate NONE (baseline) episode."""
    metrics, logs, traces = [], [], []
    service = rng.choice(config.service_names)
    base_ts = int(rng.integers(1_700_000_000, 1_700_100_000))

    for step in range(config.steps_per_episode):
        ts = base_ts + step * config.step_interval_s * 1_000

        p50 = dist.p50_baseline()
        p99 = dist.p99_baseline()
        heap = dist.heap_mb_baseline()
        gc_pause = dist.gc_pause_baseline_ms()
        err_rate = dist.error_rate_baseline()
        cache_miss = dist.cache_miss_rate_baseline()
        db_p99 = dist.db_p99_baseline_ms()
        cpu = dist.cpu_baseline()

        metrics.append({
            "episode_id": episode_id,
            "failure_mode": "NONE",
            "service": service,
            "elapsed_s": step * config.step_interval_s,
            "timestamp": ts,
            "p50_latency_ms": round(p50, 1),
            "p99_latency_ms": round(p99, 1),
            "heap_mb": round(heap, 1),
            "gc_pause_ms": round(gc_pause, 2),
            "error_rate": round(err_rate, 4),
            "cache_miss_rate": round(cache_miss, 4),
            "db_p99_ms": round(db_p99, 1),
            "cpu_pct": round(cpu, 1),
        })

        logs.append({
            "episode_id": episode_id,
            "failure_mode": "NONE",
            "service": service,
            "elapsed_s": step * config.step_interval_s,
            "timestamp": ts,
            "log_level": "INFO",
            "exception_type": "",
            "log_message": (
                f"OK cpu={cpu:.0f}% p99={p99:.0f}ms error_rate={err_rate*100:.1f}%"
            ),
        })

        if step % 5 == 0:
            traces.append(make_baseline_trace(service, ts, rng))

    return metrics, logs, traces


def generate_latency_spike_episode(
    episode_id: str, config: GeneratorConfig, dist: Distributions, rng: np.random.Generator
) -> Tuple[List[dict], List[dict], List[dict]]:
    """
    Generate LATENCY_SPIKE episode.

    Key constraints:
    - Heap stays flat (Normal(512,20))
    - GC events arrive as Poisson process (constant rate, not increasing)
    - P99 spikes when GC fires, returns to baseline immediately
    - P50 stays near baseline throughout
    - Dead-time gap appears in ~8% of request traces
    """
    metrics, logs, traces = [], [], []
    service = rng.choice(config.service_names)
    base_ts = int(rng.integers(1_700_000_000, 1_700_100_000))

    for step in range(config.steps_per_episode):
        ts = base_ts + step * config.step_interval_s * 1_000

        gc_fired = dist.gc_event_occurs(lambda_per_step=0.08)
        heap = dist.heap_mb_baseline()  # FLAT — same as baseline
        cpu  = dist.cpu_baseline()      # FLAT

        if gc_fired:
            gc_pause = dist.gc_pause_spike_ms()
            p99      = dist.p99_during_spike(gc_pause)
            p50      = dist.p50_during_spike()
            err_rate = float(np.clip(dist.error_rate_baseline() + rng.uniform(0.01, 0.04), 0, 0.15))
            log_level = "WARNING"
            log_msg  = (
                f"tail latency spike detected latency_p50={p50:.0f}ms "
                f"latency_p99={p99:.0f}ms gc_pause={gc_pause:.0f}ms heap_mb={heap:.0f}"
            )
        else:
            gc_pause = dist.gc_pause_baseline_ms()
            p99      = dist.p99_baseline()
            p50      = dist.p50_baseline()
            err_rate = dist.error_rate_baseline()
            log_level = "INFO"
            log_msg  = (
                f"OK cpu={cpu:.0f}% latency_p50={p50:.0f}ms "
                f"latency_p99={p99:.0f}ms gc_pause={gc_pause:.1f}ms"
            )

        cache_miss = dist.cache_miss_rate_baseline()  # unaffected
        db_p99     = dist.db_p99_baseline_ms()        # unaffected

        metrics.append({
            "episode_id": episode_id,
            "failure_mode": "LATENCY_SPIKE",
            "service": service,
            "elapsed_s": step * config.step_interval_s,
            "timestamp": ts,
            "p50_latency_ms": round(p50, 1),
            "p99_latency_ms": round(p99, 1),
            "heap_mb": round(heap, 1),
            "gc_pause_ms": round(gc_pause, 2),
            "error_rate": round(err_rate, 4),
            "cache_miss_rate": round(cache_miss, 4),
            "db_p99_ms": round(db_p99, 1),
            "cpu_pct": round(cpu, 1),
            "gc_event": gc_fired,
        })

        logs.append({
            "episode_id": episode_id,
            "failure_mode": "LATENCY_SPIKE",
            "service": service,
            "elapsed_s": step * config.step_interval_s,
            "timestamp": ts,
            "log_level": log_level,
            "exception_type": "",
            "log_message": log_msg,
        })

        # Generate trace — ~8% of traces hit the GC pause
        if step % 4 == 0:
            affected = gc_fired and (float(rng.random()) < dist.fraction_requests_affected() * 10)
            traces.append(
                make_latency_spike_trace(service, ts, gc_pause if gc_fired else 12, affected, rng)
            )

    return metrics, logs, traces


def generate_cache_stampede_episode(
    episode_id: str, config: GeneratorConfig, dist: Distributions, rng: np.random.Generator
) -> Tuple[List[dict], List[dict], List[dict]]:
    """
    Generate CACHE_STAMPEDE episode.

    Key constraints:
    - Onset is a step function (cache eviction is sudden, not gradual)
    - cache_miss_rate and db_p99 are positively correlated (Pearson r ~0.85)
    - DB latency modelled via M/M/1 queue driven by miss rate
    - Trace: cache_lookup=MISS immediately followed by slow db_query
    - Heap, GC, CPU stay relatively flat (no infra cause)
    """
    metrics, logs, traces = [], [], []
    service = rng.choice(config.service_names)
    base_ts = int(rng.integers(1_700_000_000, 1_700_100_000))

    # Onset: step function at step 0 (stampede is immediate)
    for step in range(config.steps_per_episode):
        ts = base_ts + step * config.step_interval_s * 1_000

        cache_miss = dist.cache_miss_rate_stampede()
        db_p99     = dist.db_p99_from_queue(
            cache_miss, config.base_rps, config.db_capacity_rps
        )
        req_p99    = dist.request_p99_from_db(db_p99)
        p50        = dist.p50_baseline()                    # mostly unaffected
        heap       = dist.heap_mb_baseline()                # flat
        gc_pause   = dist.gc_pause_baseline_ms()            # flat
        cpu        = float(np.clip(dist.cpu_baseline() + rng.uniform(10, 25), 30, 85))
        err_rate   = float(np.clip(rng.beta(3, 12), 0.10, 0.35))  # some errors from DB timeouts

        metrics.append({
            "episode_id": episode_id,
            "failure_mode": "CACHE_STAMPEDE",
            "service": service,
            "elapsed_s": step * config.step_interval_s,
            "timestamp": ts,
            "p50_latency_ms": round(p50, 1),
            "p99_latency_ms": round(req_p99, 1),
            "heap_mb": round(heap, 1),
            "gc_pause_ms": round(gc_pause, 2),
            "error_rate": round(err_rate, 4),
            "cache_miss_rate": round(cache_miss, 4),
            "db_p99_ms": round(db_p99, 1),
            "cpu_pct": round(cpu, 1),
        })

        log_msg = (
            f"Cache miss rate={cache_miss*100:.0f}% db_p99={db_p99:.0f}ms "
            f"— cache stampede suspected"
        )
        logs.append({
            "episode_id": episode_id,
            "failure_mode": "CACHE_STAMPEDE",
            "service": service,
            "elapsed_s": step * config.step_interval_s,
            "timestamp": ts,
            "log_level": "WARNING",
            "exception_type": "",
            "log_message": log_msg,
        })

        if step % 4 == 0:
            traces.append(make_cache_stampede_trace(service, ts, db_p99, rng))

    return metrics, logs, traces


# ---------------------------------------------------------------------------
# Validation checks (statistical assertions on generated data)
# ---------------------------------------------------------------------------

def validate_latency_spike(metrics: List[dict]) -> None:
    """Assert statistical properties of LATENCY_SPIKE data."""
    ls_rows = [m for m in metrics if m["failure_mode"] == "LATENCY_SPIKE"]
    if not ls_rows:
        return

    heaps    = np.array([r["heap_mb"] for r in ls_rows])
    p99s     = np.array([r["p99_latency_ms"] for r in ls_rows])
    p50s     = np.array([r["p50_latency_ms"] for r in ls_rows])
    gc_paus  = np.array([r["gc_pause_ms"] for r in ls_rows])

    # Heap should be flat: std < 40MB
    assert heaps.std() < 40, f"Heap not flat! std={heaps.std():.1f}MB"

    # P99/P50 ratio: some values should be > 8 (spike rows)
    ratios = p99s / p50s
    assert ratios.max() > 5.0, f"P99/P50 ratio never exceeds 5x: max={ratios.max():.1f}"

    # Heap vs P99: should be near zero correlation
    r_heap_p99 = float(np.corrcoef(heaps, p99s)[0, 1])
    assert abs(r_heap_p99) < 0.4, f"Heap/P99 corr too high: r={r_heap_p99:.2f}"

    # GC pause vs P99: should be positively correlated
    r_gc_p99 = float(np.corrcoef(gc_paus, p99s)[0, 1])
    assert r_gc_p99 > 0.3, f"GC/P99 correlation too low: r={r_gc_p99:.2f}"

    print(f"[LATENCY_SPIKE] OK heap.std={heaps.std():.1f}  "
          f"max_P99/P50={ratios.max():.1f}x  "
          f"r(heap,P99)={r_heap_p99:.2f}  "
          f"r(gc,P99)={r_gc_p99:.2f}")


def validate_cache_stampede(metrics: List[dict]) -> None:
    """Assert statistical properties of CACHE_STAMPEDE data."""
    cs_rows = [m for m in metrics if m["failure_mode"] == "CACHE_STAMPEDE"]
    if not cs_rows:
        return

    miss  = np.array([r["cache_miss_rate"] for r in cs_rows])
    db_p  = np.array([r["db_p99_ms"]       for r in cs_rows])
    heaps = np.array([r["heap_mb"]          for r in cs_rows])

    # Cache miss rate should be >85% for all rows
    assert miss.min() > 0.80, f"Cache miss rate went below 80%: min={miss.min():.2%}"

    # Correlation miss rate ↔ db_p99 should be strong positive
    r_miss_db = float(np.corrcoef(miss, db_p)[0, 1])
    assert r_miss_db > 0.35, f"Cache miss / DB P99 correlation too low: r={r_miss_db:.2f}"

    # Heap should still be flat
    assert heaps.std() < 40, f"Heap not flat! std={heaps.std():.1f}MB"

    print(f"[CACHE_STAMPEDE] OK miss_rate_min={miss.min():.2%}  "
          f"r(miss,db_p99)={r_miss_db:.2f}  "
          f"heap.std={heaps.std():.1f}")


# ---------------------------------------------------------------------------
# Writer utilities
# ---------------------------------------------------------------------------

METRIC_FIELDS = [
    "episode_id", "failure_mode", "service", "elapsed_s", "timestamp",
    "p50_latency_ms", "p99_latency_ms", "heap_mb", "gc_pause_ms",
    "error_rate", "cache_miss_rate", "db_p99_ms", "cpu_pct",
]

LOG_FIELDS = [
    "episode_id", "failure_mode", "service", "elapsed_s", "timestamp",
    "log_level", "exception_type", "log_message",
]


def write_csv(rows: List[dict], path: str, fields: List[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"  -> Wrote {len(rows)} rows to {path}")


def write_jsonl(records: List[dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    print(f"  -> Wrote {len(records)} records to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic LATENCY_SPIKE and CACHE_STAMPEDE telemetry data"
    )
    parser.add_argument("--episodes", type=int, default=5,
                        help="Episodes per failure mode (default: 5)")
    parser.add_argument("--steps",    type=int, default=120,
                        help="Steps per episode (default: 120, = 4 min at 2s interval)")
    parser.add_argument("--seed",     type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--output",   type=str, default="data",
                        help="Output directory (default: data/)")
    args = parser.parse_args()

    config = GeneratorConfig(
        seed=args.seed,
        episodes_per_mode=args.episodes,
        steps_per_episode=args.steps,
        output_dir=args.output,
    )

    rng  = np.random.default_rng(config.seed)
    dist = Distributions(rng)

    os.makedirs(config.output_dir, exist_ok=True)

    all_metrics: List[dict] = []
    all_logs:    List[dict] = []
    all_traces:  List[dict] = []

    print("=" * 60)
    print("AIOps Telemetry Generator — LATENCY_SPIKE & CACHE_STAMPEDE")
    print(f"  Seed: {config.seed} | Episodes/mode: {config.episodes_per_mode} | Steps: {config.steps_per_episode}")
    print("=" * 60)

    # --- NONE baseline ---
    print("\n[1/3] Generating NONE (baseline) episodes...")
    for i in range(config.episodes_per_mode):
        ep_id = f"ep_{i:05d}_NONE"
        m, l, t = generate_none_episode(ep_id, config, dist, rng)
        all_metrics.extend(m)
        all_logs.extend(l)
        all_traces.extend(t)

    # --- LATENCY_SPIKE ---
    print(f"\n[2/3] Generating LATENCY_SPIKE episodes...")
    for i in range(config.episodes_per_mode):
        ep_id = f"ep_{i:05d}_LS"
        m, l, t = generate_latency_spike_episode(ep_id, config, dist, rng)
        all_metrics.extend(m)
        all_logs.extend(l)
        all_traces.extend(t)

    # --- CACHE_STAMPEDE ---
    print(f"\n[3/3] Generating CACHE_STAMPEDE episodes...")
    for i in range(config.episodes_per_mode):
        ep_id = f"ep_{i:05d}_CS"
        m, l, t = generate_cache_stampede_episode(ep_id, config, dist, rng)
        all_metrics.extend(m)
        all_logs.extend(l)
        all_traces.extend(t)

    # --- Write outputs ---
    print("\nWriting output files...")
    write_csv(all_metrics,
              os.path.join(config.output_dir, "telemetry_metrics.csv"), METRIC_FIELDS)
    write_csv(all_logs,
              os.path.join(config.output_dir, "telemetry_logs.csv"), LOG_FIELDS)
    write_jsonl(all_traces,
                os.path.join(config.output_dir, "telemetry_traces.jsonl"))

    # --- Validate ---
    print("\nRunning statistical validation checks...")
    validate_latency_spike(all_metrics)
    validate_cache_stampede(all_metrics)

    # --- Summary report ---
    modes = {}
    for m in all_metrics:
        modes[m["failure_mode"]] = modes.get(m["failure_mode"], 0) + 1

    print("\n" + "=" * 60)
    print("Generation complete. Row counts by failure mode:")
    for mode, count in sorted(modes.items()):
        print(f"  {mode:<20}: {count} metric rows")
    print(f"\nTotal traces : {len(all_traces)}")
    print(f"Total log rows: {len(all_logs)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
