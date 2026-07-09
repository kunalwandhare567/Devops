"""
data_pipeline.py
================
Combined Metrics + Logs Data Preparation Pipeline
--------------------------------------------------
Implements the full pipeline:

  telemetry_metrics.csv  --+
                            |--> merge on episode_id --> X (1001 x 32) + y
  telemetry_logs.csv     --+
                            |
              stratified 80/20 split (seed=42)
                            |
          Random Forest | XGBoost | LightGBM
                            |
     Accuracy + F1 + Confusion Matrix + Feature Importance

Offline prerequisites (produced by offline/train_log_templates.py):
    known_log_templates.json
    drain_state.bin

Usage:
    python data_pipeline.py
"""

from __future__ import annotations

import json
import os
import pickle
import sys
import warnings

import matplotlib
matplotlib.use("Agg")           # headless backend — no display required
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR        = os.path.join(ROOT_DIR, "data")
METRICS_CSV     = os.path.join(DATA_DIR, "telemetry_metrics.csv")
LOGS_CSV        = os.path.join(DATA_DIR, "telemetry_logs.csv")
DRAIN_INI       = os.path.join(ROOT_DIR, "drain3.ini")
TEMPLATES_JSON  = os.path.join(ROOT_DIR, "known_log_templates.json")
STATE_BIN       = os.path.join(ROOT_DIR, "drain_state.bin")

RANDOM_SEED     = 42
TEST_SIZE       = 0.20

# ---------------------------------------------------------------------------
# Metric feature definitions
# ---------------------------------------------------------------------------
# (source_column, feature_name, aggregation)
# aggregation: "mean" | "max" | "std" | "slope" | "ratio_open"
METRIC_FEATURES = [
    # CPU
    ("cpu_utilization",  "cpu_mean",             "mean"),
    ("cpu_utilization",  "cpu_max",              "max"),
    ("cpu_utilization",  "cpu_std",              "std"),
    ("cpu_utilization",  "cpu_slope",            "slope"),
    # Memory
    ("memory_utilization", "memory_mean",        "mean"),
    ("memory_utilization", "memory_max",         "max"),
    ("memory_utilization", "memory_growth_rate", "slope"),
    # Heap
    ("heap_mb",          "heap_mean",            "mean"),
    ("heap_mb",          "heap_max",             "max"),
    # Latency
    ("p50_latency",      "p50_mean",             "mean"),
    ("p95_latency",      "p95_mean",             "mean"),
    ("p99_latency",      "p99_mean",             "mean"),
    ("p99_latency",      "latency_std",          "std"),
    ("p99_latency",      "latency_slope",        "slope"),
    # Throughput
    ("rps",              "throughput_mean",      "mean"),
    ("rps",              "throughput_std",       "std"),
    # Cache
    ("cache_hit_rate",   "cache_hit_ratio",      "mean"),
    ("cache_miss_rate",  "cache_miss_ratio",     "mean"),
    # Database
    ("db_p99",           "db_latency_mean",      "mean"),
    ("active_connections", "db_connections_max", "max"),
    # Network
    ("network_errors",   "network_errors_mean",  "mean"),
    # Error
    ("error_rate",       "error_rate_mean",      "mean"),
    ("error_rate",       "error_rate_max",       "max"),
    # GC
    ("gc_pause_p99",     "gc_pause_mean",        "mean"),
    ("gc_pause_p99",     "gc_pause_max",         "max"),
    # Disk
    ("disk_read_latency",  "disk_read_mean",     "mean"),
    ("disk_write_latency", "disk_write_mean",    "mean"),
    # Circuit breaker
    ("circuit_breaker_state", "cb_open_ratio",   "ratio_open"),
]

LOG_FEATURE_COLS = [
    "log_count",
    "log_max_severity",
    "log_critical_count",
    "log_has_exception",
    "log_has_novel_template",
    "log_exception_type_encoded",
    "log_severity_ratio",
]

SEVERITY_MAP = {
    "":         0,
    "INFO":     1,
    "WARNING":  2,
    "WARN":     2,
    "ERROR":    3,
    "CRITICAL": 4,
    "FATAL":    4,
}

EXCEPTION_TYPE_MAP = {
    "":                        0,
    "RuntimeException":        1,
    "SocketTimeoutException":  2,
    "NullPointerException":    3,
    "SystemOverloadException": 4,
}


# =============================================================================
# SECTION 1 — METRICS PREPROCESSING & FEATURE ENGINEERING
# =============================================================================

def preprocess_metrics(metrics_csv: str) -> pd.DataFrame:
    """Loads and preprocesses telemetry_metrics.csv."""
    print(f"[Metrics] Loading {metrics_csv} ...")
    df = pd.read_csv(metrics_csv, dtype=str)

    # Cast numeric columns
    numeric_cols = [
        "elapsed_s", "active_connections", "cache_hit_rate", "cache_miss_rate",
        "cpu_saturation", "cpu_utilization", "db_connection_pool", "db_connection_wait",
        "db_p99", "disk_read_latency", "disk_write_latency", "error_rate",
        "gc_pause_p99", "heap_mb", "http_4xx_rate", "http_5xx_rate",
        "iops_utilization", "memory_utilization", "network_errors",
        "p50_latency", "p95_latency", "p99_latency", "queue_lag",
        "retry_count_per_request", "rps", "thread_pool_queue", "upstream_timeout_rate",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Convert timestamp to numeric
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df["elapsed_s"] = pd.to_numeric(df["elapsed_s"], errors="coerce")

    # Remove duplicates
    before = len(df)
    df = df.drop_duplicates(subset=["episode_id", "elapsed_s"])
    if len(df) < before:
        print(f"  Removed {before - len(df):,} duplicate rows")

    # Drop rows with missing key columns
    df = df.dropna(subset=["episode_id", "failure_mode", "elapsed_s"])

    # Sort
    df = df.sort_values(["episode_id", "elapsed_s"]).reset_index(drop=True)

    print(f"  Shape after preprocessing: {df.shape}")
    print(f"  Unique episodes           : {df['episode_id'].nunique()}")
    return df


def _slope(x: np.ndarray, y: np.ndarray) -> float:
    """Returns the linear regression slope of y over x."""
    if len(x) < 2:
        return 0.0
    try:
        return float(np.polyfit(x, y, 1)[0])
    except Exception:
        return 0.0


def compute_metric_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates raw metrics into one row per episode.
    Returns a DataFrame indexed by episode_id with metric feature columns.
    """
    print("[Metrics] Computing episode-level features ...")
    rows = []
    for ep_id, grp in df.groupby("episode_id", sort=False):
        t = grp["elapsed_s"].values
        row = {"episode_id": ep_id, "failure_mode": grp["failure_mode"].iloc[0]}

        for src_col, feat_name, agg in METRIC_FEATURES:
            if agg == "ratio_open":
                vals = grp[src_col].astype(str).str.lower()
                row[feat_name] = float((vals == "open").sum()) / max(len(vals), 1)
            else:
                vals = grp[src_col].values.astype(float)
                if agg == "mean":
                    row[feat_name] = float(np.nanmean(vals))
                elif agg == "max":
                    row[feat_name] = float(np.nanmax(vals))
                elif agg == "std":
                    row[feat_name] = float(np.nanstd(vals))
                elif agg == "slope":
                    row[feat_name] = _slope(t, vals)

        rows.append(row)

    result = pd.DataFrame(rows)
    metric_feat_cols = [f for _, f, _ in METRIC_FEATURES]
    print(f"  Metric feature matrix shape: {result.shape}")
    print(f"  Features: {metric_feat_cols}")
    return result


# =============================================================================
# SECTION 2 — LOGS PREPROCESSING & FEATURE ENGINEERING
# =============================================================================

def load_drain_artifacts(state_bin: str, templates_json: str, drain_ini: str):
    """Loads frozen Drain3 miner and known template IDs."""
    known_ids = set()
    if os.path.exists(templates_json):
        with open(templates_json, "r", encoding="utf-8") as f:
            known_ids = set(json.load(f))
        print(f"[Logs] Loaded {len(known_ids)} known template IDs")
    else:
        print(f"[Logs] WARNING: {templates_json} not found — novelty detection disabled")

    miner = None
    if not os.path.exists(state_bin):
        print(f"[Logs] WARNING: {state_bin} not found — run offline/train_log_templates.py first")
        return miner, known_ids

    try:
        from drain3 import TemplateMiner
        from drain3.template_miner_config import TemplateMinerConfig
        from drain3.file_persistence import FilePersistence

        config = TemplateMinerConfig()
        if os.path.exists(drain_ini):
            config.load(drain_ini)

        persistence = FilePersistence(state_bin)
        miner = TemplateMiner(persistence_handler=persistence, config=config)
        if len(list(miner.drain.clusters)) > 0:
            print(f"[Logs] Loaded Drain3 state ({len(list(miner.drain.clusters))} templates)")
            return miner, known_ids
    except Exception:
        pass

    # Fallback: pickle
    try:
        with open(state_bin, "rb") as fh:
            miner = pickle.load(fh)
        print(f"[Logs] Loaded Drain3 state (pickle, {len(list(miner.drain.clusters))} templates)")
    except Exception as exc:
        print(f"[Logs] ERROR loading Drain3 state: {exc}")

    return miner, known_ids


def engineer_log_features(
    log_lines_this_cycle: list[dict],
    template_miner,
    known_template_ids: set[int],
) -> dict:
    """Converts raw log dicts for one episode into 7 classifier features."""
    if not log_lines_this_cycle:
        return {k: 0 for k in LOG_FEATURE_COLS}

    severities:        list[int] = []
    template_ids:      list[int] = []
    has_exception:     bool      = False
    has_novel:         bool      = False
    exc_type_encoded:  int       = 0

    for line in log_lines_this_cycle:
        log_level   = str(line.get("log_level",     "")).strip().upper()
        exc_type    = str(line.get("exception_type", "")).strip()
        log_message = str(line.get("log_message",   "")).strip()

        sev = SEVERITY_MAP.get(log_level, 0)
        severities.append(sev)

        if exc_type:
            has_exception = True
            encoded = EXCEPTION_TYPE_MAP.get(exc_type, 0)
            if encoded > exc_type_encoded:
                exc_type_encoded = encoded

        if template_miner is not None and log_message:
            cluster = template_miner.match(log_message)
            if cluster is None:
                has_novel = True
                template_ids.append(-1)
            else:
                tid = cluster.cluster_id
                template_ids.append(tid)
                if tid not in known_template_ids:
                    has_novel = True

    log_count        = len(log_lines_this_cycle)
    log_max_sev      = max(severities) if severities else 0
    log_critical_cnt = sum(1 for s in severities if s == 4)
    log_sev_ratio    = round(log_critical_cnt / log_count, 6) if log_count > 0 else 0.0

    return {
        "log_count":                  log_count,
        "log_max_severity":           log_max_sev,
        "log_critical_count":         log_critical_cnt,
        "log_has_exception":          int(has_exception),
        "log_has_novel_template":     int(has_novel),
        "log_exception_type_encoded": exc_type_encoded,
        "log_severity_ratio":         log_sev_ratio,
    }


def compute_log_features(logs_csv: str, miner, known_ids: set[int]) -> pd.DataFrame:
    """Preprocesses telemetry_logs.csv and applies log feature engineering per episode."""
    print(f"[Logs] Loading {logs_csv} ...")
    df = pd.read_csv(logs_csv, dtype=str).fillna("")

    # Remove duplicates & sort
    df = df.drop_duplicates()
    df = df.sort_values(["episode_id", "timestamp"]).reset_index(drop=True)
    print(f"  Shape after preprocessing : {df.shape}")
    print(f"  Unique episodes           : {df['episode_id'].nunique()}")

    print("[Logs] Computing episode-level log features ...")
    rows = []
    for ep_id, grp in df.groupby("episode_id", sort=False):
        log_lines = grp.to_dict(orient="records")
        features  = engineer_log_features(log_lines, miner, known_ids)
        features["episode_id"]   = ep_id
        features["failure_mode"] = grp["failure_mode"].iloc[0]
        rows.append(features)

    result = pd.DataFrame(rows)
    print(f"  Log feature matrix shape  : {result.shape}")
    return result


# =============================================================================
# SECTION 3 — MERGE + SPLIT
# =============================================================================

def merge_and_split(metric_df: pd.DataFrame, log_df: pd.DataFrame):
    """
    Inner-joins metric and log feature frames on episode_id,
    removes metadata columns, performs stratified 80/20 episode-level split.
    """
    print("\n[Merge] Joining metric + log features on episode_id ...")
    combined = metric_df.merge(
        log_df.drop(columns=["failure_mode"]),
        on="episode_id",
        how="inner",
    )
    print(f"  Combined shape : {combined.shape}")

    # Verify no leakage columns in X
    drop_cols = ["episode_id", "failure_mode"]
    X = combined.drop(columns=drop_cols)
    y = combined["failure_mode"]

    print(f"  Feature matrix : {X.shape}")
    print(f"  Label series   : {y.shape}")
    print(f"  Classes        : {sorted(y.unique())}")

    # Stratified 80/20 split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_SEED,
    )
    print(f"\n[Split] Stratified 80/20 (seed={RANDOM_SEED}):")
    print(f"  X_train : {X_train.shape}   y_train : {y_train.shape}")
    print(f"  X_test  : {X_test.shape}    y_test  : {y_test.shape}")

    # Verify all classes in both splits
    assert set(y_train.unique()) == set(y_test.unique()), \
        "Not all failure modes present in both splits!"
    print(f"  [OK] All {y.nunique()} failure modes in both train and test splits")

    print("\n  Class distribution in y_train:")
    for mode, cnt in sorted(y_train.value_counts().items()):
        print(f"    {mode:25}: {cnt}")

    print("\n  Class distribution in y_test:")
    for mode, cnt in sorted(y_test.value_counts().items()):
        print(f"    {mode:25}: {cnt}")

    # Verify no novel templates in train split
    if "log_has_novel_template" in X_train.columns:
        novel_train = int(X_train["log_has_novel_template"].sum())
        novel_test  = int(X_test["log_has_novel_template"].sum())
        print(f"\n  log_has_novel_template in X_train : {novel_train} (expected 0)")
        print(f"  log_has_novel_template in X_test  : {novel_test}")

    return X_train, X_test, y_train, y_test, X.columns.tolist()


# =============================================================================
# SECTION 4 — ML TRAINING & EVALUATION
# =============================================================================

def plot_confusion_matrix(cm, classes, model_name: str, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(14, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(ax=ax, xticks_rotation=45, colorbar=True, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved confusion matrix -> {out_path}")


def plot_feature_importance(importances, feature_names, model_name: str, out_path: str) -> None:
    top_n    = 20
    indices  = np.argsort(importances)[::-1][:top_n]
    top_feat = [feature_names[i] for i in indices]
    top_imp  = importances[indices]

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(range(top_n), top_imp[::-1], color="steelblue", edgecolor="white")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_feat[::-1], fontsize=9)
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_n} Feature Importances — {model_name}", fontweight="bold")
    ax.bar_label(bars, fmt="%.4f", padding=2, fontsize=7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved feature importance  -> {out_path}")


def train_and_evaluate(
    name: str,
    clf,
    X_train: pd.DataFrame,
    X_test:  pd.DataFrame,
    y_train: pd.Series,
    y_test:  pd.Series,
    feature_names: list[str],
    out_dir: str,
) -> dict:
    """Trains one classifier, prints metrics, saves plots."""
    print(f"\n{'='*65}")
    print(f"  Model: {name}")
    print(f"{'='*65}")

    clf.fit(X_train, y_train)

    y_pred_train = clf.predict(X_train)
    y_pred_test  = clf.predict(X_test)

    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc  = accuracy_score(y_test,  y_pred_test)

    print(f"  Train accuracy : {train_acc*100:.2f}%")
    print(f"  Test  accuracy : {test_acc*100:.2f}%")

    print(f"\n  Classification Report (Test):")
    print(classification_report(y_test, y_pred_test, zero_division=0))

    # Confusion matrix
    classes = sorted(y_test.unique())
    cm      = confusion_matrix(y_test, y_pred_test, labels=classes)
    cm_path = os.path.join(out_dir, f"confusion_matrix_{name.lower().replace(' ', '_')}.png")
    plot_confusion_matrix(cm, classes, name, cm_path)

    # Feature importance
    if hasattr(clf, "feature_importances_"):
        imp     = clf.feature_importances_
        fi_path = os.path.join(out_dir, f"feature_importance_{name.lower().replace(' ', '_')}.png")
        plot_feature_importance(imp, feature_names, name, fi_path)

    return {"model": name, "train_acc": train_acc, "test_acc": test_acc}


def run_ml_pipeline(
    X_train: pd.DataFrame,
    X_test:  pd.DataFrame,
    y_train: pd.Series,
    y_test:  pd.Series,
    feature_names: list[str],
    out_dir: str,
) -> None:
    """Trains Random Forest, XGBoost, LightGBM and prints a summary."""
    results = []

    # --- 1. Random Forest ---
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=300, random_state=RANDOM_SEED, n_jobs=-1)
    results.append(train_and_evaluate(
        "Random Forest", rf, X_train, X_test, y_train, y_test, feature_names, out_dir
    ))

    # --- 2. XGBoost ---
    try:
        from xgboost import XGBClassifier
        le      = LabelEncoder()
        y_tr_enc = le.fit_transform(y_train)
        y_te_enc = le.transform(y_test)

        xgb = XGBClassifier(
            n_estimators=300,
            learning_rate=0.1,
            random_state=RANDOM_SEED,
            eval_metric="mlogloss",
            verbosity=0,
            use_label_encoder=False,
        )
        xgb.fit(X_train, y_tr_enc)

        y_pred_train_enc = xgb.predict(X_train)
        y_pred_test_enc  = xgb.predict(X_test)

        train_acc = accuracy_score(y_tr_enc, y_pred_train_enc)
        test_acc  = accuracy_score(y_te_enc, y_pred_test_enc)

        print(f"\n{'='*65}")
        print(f"  Model: XGBoost")
        print(f"{'='*65}")
        print(f"  Train accuracy : {train_acc*100:.2f}%")
        print(f"  Test  accuracy : {test_acc*100:.2f}%")

        # Decode for report
        y_pred_test_labels  = le.inverse_transform(y_pred_test_enc)
        y_pred_train_labels = le.inverse_transform(y_pred_train_enc)
        print(f"\n  Classification Report (Test):")
        print(classification_report(y_test, y_pred_test_labels, zero_division=0))

        classes = sorted(y_test.unique())
        cm      = confusion_matrix(y_test, y_pred_test_labels, labels=classes)
        plot_confusion_matrix(cm, classes, "XGBoost",
                              os.path.join(out_dir, "confusion_matrix_xgboost.png"))
        plot_feature_importance(xgb.feature_importances_, feature_names, "XGBoost",
                                os.path.join(out_dir, "feature_importance_xgboost.png"))
        results.append({"model": "XGBoost", "train_acc": train_acc, "test_acc": test_acc})

    except ImportError:
        print("\n  XGBoost not installed. Skipping. Run: pip install xgboost")

    # --- 3. LightGBM ---
    try:
        import lightgbm as lgb
        lgbm = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.1,
            random_state=RANDOM_SEED,
            verbosity=-1,
        )
        results.append(train_and_evaluate(
            "LightGBM", lgbm, X_train, X_test, y_train, y_test, feature_names, out_dir
        ))
    except ImportError:
        print("\n  LightGBM not installed. Skipping. Run: pip install lightgbm")

    # Final summary
    print(f"\n{'='*65}")
    print("  MODEL ACCURACY SUMMARY")
    print(f"  {'Model':<20}  {'Train Acc':>10}  {'Test Acc':>10}")
    print(f"  {'-'*45}")
    for r in results:
        print(f"  {r['model']:<20}  {r['train_acc']*100:>9.2f}%  {r['test_acc']*100:>9.2f}%")
    print(f"{'='*65}")


# =============================================================================
# SECTION 5 — SAVE OUTPUTS
# =============================================================================

def save_outputs(
    X_train: pd.DataFrame,
    X_test:  pd.DataFrame,
    y_train: pd.Series,
    y_test:  pd.Series,
    feature_names: list[str],
    out_dir: str,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    X_train.to_csv(os.path.join(out_dir, "X_train.csv"), index=False)
    X_test.to_csv( os.path.join(out_dir, "X_test.csv"),  index=False)
    y_train.to_csv(os.path.join(out_dir, "y_train.csv"), index=False, header=True)
    y_test.to_csv( os.path.join(out_dir, "y_test.csv"),  index=False, header=True)

    with open(os.path.join(out_dir, "feature_names.json"), "w") as f:
        json.dump(feature_names, f, indent=2)

    print(f"\n[Output] Files saved to {out_dir}/")
    for fname in ["X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv", "feature_names.json"]:
        path = os.path.join(out_dir, fname)
        if os.path.exists(path):
            size_kb = os.path.getsize(path) / 1024
            print(f"  {fname:30}  {size_kb:8.1f} KB")


# =============================================================================
# ENTRY POINT
# =============================================================================

def main() -> None:
    print("=" * 65)
    print("  DATA PREPARATION PIPELINE")
    print("=" * 65)

    # ---- Phase 1: Metrics ------------------------------------------------
    print("\n--- METRICS PREPROCESSING ---")
    metrics_raw = preprocess_metrics(METRICS_CSV)
    metric_df   = compute_metric_features(metrics_raw)

    # ---- Phase 2: Logs ---------------------------------------------------
    print("\n--- LOGS PREPROCESSING ---")
    miner, known_ids = load_drain_artifacts(STATE_BIN, TEMPLATES_JSON, DRAIN_INI)
    log_df = compute_log_features(LOGS_CSV, miner, known_ids)

    # ---- Phase 3: Merge + Split ------------------------------------------
    X_train, X_test, y_train, y_test, feature_names = merge_and_split(metric_df, log_df)

    # ---- Phase 4: Save CSVs ----------------------------------------------
    save_outputs(X_train, X_test, y_train, y_test, feature_names, DATA_DIR)

    # ---- Phase 5: ML Training --------------------------------------------
    print("\n--- MACHINE LEARNING TRAINING ---")
    run_ml_pipeline(X_train, X_test, y_train, y_test, feature_names, DATA_DIR)

    print("\n[Done] Pipeline complete.")


if __name__ == "__main__":
    main()
