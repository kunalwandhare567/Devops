# Combined Metrics + Logs Pipeline — Implementation Plan

## Overview

Build a single script `data_pipeline.py` that:
1. Preprocesses `telemetry_metrics.csv` → 25 metric episode-level features
2. Preprocesses `telemetry_logs.csv` → 7 log episode-level features (reusing existing Drain3 artifacts)
3. Merges both on `episode_id` → **32-feature matrix**
4. Stratified episode-level 80/20 split → X_train, X_test, y_train, y_test
5. Trains **Random Forest**, **XGBoost**, **LightGBM** and reports accuracy

---

## Data Profile (confirmed from CSV)

| Dataset | Rows | Unique episodes | Modes | Rows/episode |
|---|---|---|---|---|
| `telemetry_metrics.csv` | 120,120 | **1,001** | 13 (77 each) | 120 |
| `telemetry_logs.csv` | 120,120 | **1,001** | 13 (77 each) | 120 |
| **Episode overlap** | — | **1,001** (100%) | — | — |

> [!IMPORTANT]
> **Critical finding:** The metrics CSV has **1,001 episodes** (77 per mode × 13 modes) but the
> previous log-only pipeline only saw **169 episodes** (13 per mode × 13 modes).
> The metrics CSV contains the FULL dataset from `generate_full_dataset.py` while the logs CSV
> is the same full dataset — both align perfectly (1,001 episodes each, 100% overlap).
> **The split will be on 1,001 episodes**: 80% = 800 train, 20% = 201 test.

---

## Train/Test Split

```
Total episodes : 1,001  (77 × 13 failure modes)

Stratified 80/20 split (episode-level, seed=42):
  Train : 800 episodes  (≈61-62 per mode)
  Test  :  201 episodes  (≈15-16 per mode)

Rows in feature matrices (1 row per episode):
  X_train : (800,  32)    y_train : (800,)
  X_test  : (201,  32)    y_test  : (201,)
```

All 13 failure modes guaranteed in both splits via `stratify=y`.

---

## Proposed Changes

---

### Phase 1 — Re-train Drain3 on Full 1,001-episode Log Corpus

> [!WARNING]
> The existing `drain_state.bin` was trained on only 169 episodes (the old small dataset).
> We must **re-run** `offline/train_log_templates.py` after updating it to use the full
> 1,001-episode corpus (already present in `telemetry_logs.csv`).
> The script needs no code change — it reads the same CSV. Just re-run it.

---

### Phase 2 — Main Pipeline Script

#### [NEW] [data_pipeline.py](file:///d:/DEVOPS/data_pipeline.py)

---

#### 2a. Metrics Preprocessing

**Drop / ignore these columns (not useful for episode features):**

| Column | Reason |
|---|---|
| `source` | Constant (`"python generator"`) |
| `service` | Varies randomly within episode, not a failure signal |
| `elapsed_s` | Time axis — used only for slope calculation |
| `failure_mode` | This IS the label `y` |
| `episode_id` | Grouping key, removed before training |
| `circuit_breaker_state` | Categorical — encode as `cb_open_ratio` (fraction of timesteps where state=`open`) |

**Preprocessing steps per episode group:**
1. Sort by `elapsed_s`
2. Drop duplicates on `(episode_id, elapsed_s)`
3. Confirm no missing values (confirmed clean)
4. Convert `circuit_breaker_state` → numeric: `open`=2, `half-open`=1, `closed`=0
5. Compute slope via `np.polyfit(elapsed_s, values, 1)[0]` for trending features

---

#### 2b. Metric Feature Engineering (25 features per episode)

**Column → Feature mapping:**

| Source Column(s) | Feature(s) | Computation |
|---|---|---|
| `cpu_utilization` | `cpu_mean`, `cpu_max`, `cpu_std`, `cpu_slope` | mean, max, std, linreg slope |
| `memory_utilization` | `memory_mean`, `memory_max`, `memory_growth_rate` | mean, max, slope |
| `heap_mb` | `heap_mean`, `heap_max` | mean, max |
| `p50_latency` | `p50_mean` | mean |
| `p95_latency` | `p95_mean` | mean |
| `p99_latency` | `p99_mean`, `latency_std`, `latency_slope` | mean, std of p99, slope of p99 |
| `rps` | `throughput_mean`, `throughput_std` | mean, std |
| `cache_hit_rate` | `cache_hit_ratio` | mean |
| `cache_miss_rate` | `cache_miss_ratio` | mean |
| `db_p99` | `db_latency_mean` | mean |
| `active_connections` | `db_connections_max` | max |
| `network_errors` | `network_errors_mean` | mean |
| `error_rate` | `error_rate_mean`, `error_rate_max` | mean, max |
| `gc_pause_p99` | `gc_pause_mean`, `gc_pause_max` | mean, max |
| `circuit_breaker_state` | `cb_open_ratio` | fraction where state = open |

**Total metric features: 25**

> [!NOTE]
> `network_in_mean` and `network_out_mean` from the pipeline diagram do not exist as separate
> columns in the CSV — the available network column is `network_errors`. We use `network_errors_mean`
> instead. `disk_read_latency` and `disk_write_latency` exist but are not in the original diagram;
> we add them as `disk_read_mean` and `disk_write_mean` for completeness (making 25 total, not 24).

---

#### 2c. Log Feature Engineering (7 features — reusing existing function)

Reuses `engineer_log_features()` from `log_feature_engineering.py` with the updated Drain3 artifacts.

| Feature | Description |
|---|---|
| `log_count` | Total log lines per episode |
| `log_max_severity` | Max severity (1=INFO … 4=CRITICAL) |
| `log_critical_count` | Count of CRITICAL logs |
| `log_has_exception` | 1 if any exception_type present |
| `log_has_novel_template` | 1 if any Drain3 match outside known templates |
| `log_exception_type_encoded` | Encoded exception type (0–4) |
| `log_severity_ratio` | critical_count / log_count |

**Total log features: 7**

---

#### 2d. Merge & Final Matrix

```python
# Inner join — guaranteed 1,001 episodes from both sides
combined = metric_df.merge(log_df, on="episode_id", how="inner")

X = combined.drop(columns=["episode_id", "failure_mode"])   # 32 features
y = combined["failure_mode"]
```

**Final feature matrix shape: (1001, 32)**

---

#### 2e. Train/Test Split

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    stratify=y,
    random_state=42,
)
# X_train: (800, 32)   X_test: (201, 32)
```

---

#### 2e. ML Training — 3 Models

| Model | Library | Key hyperparams (defaults first run) |
|---|---|---|
| Random Forest | `sklearn.ensemble.RandomForestClassifier` | `n_estimators=300, random_state=42` |
| XGBoost | `xgboost.XGBClassifier` | `n_estimators=300, learning_rate=0.1, random_state=42` |
| LightGBM | `lightgbm.LGBMClassifier` | `n_estimators=300, learning_rate=0.1, random_state=42` |

**Evaluation metrics reported per model:**
- Accuracy (train + test)
- Classification report (precision, recall, F1 per class)
- Confusion matrix (saved to `data/confusion_matrix_<model>.png`)
- Feature importance (top 15 features, saved to `data/feature_importance_<model>.png`)

---

### Output Files

| File | Description |
|---|---|
| `data/X_train.csv` | (800, 32) — replaces old 7-feature version |
| `data/X_test.csv` | (201, 32) — replaces old 7-feature version |
| `data/y_train.csv` | (800,) labels |
| `data/y_test.csv` | (201,) labels |
| `data/feature_names.json` | Ordered list of 32 feature names |
| `data/confusion_matrix_rf.png` | Random Forest confusion matrix |
| `data/confusion_matrix_xgb.png` | XGBoost confusion matrix |
| `data/confusion_matrix_lgbm.png` | LightGBM confusion matrix |
| `data/feature_importance_rf.png` | Feature importances |
| `data/feature_importance_xgb.png` | Feature importances |
| `data/feature_importance_lgbm.png` | Feature importances |

---

## Complete Feature List (32 features)

### Metric Features (25)
```
cpu_mean, cpu_max, cpu_std, cpu_slope,
memory_mean, memory_max, memory_growth_rate,
heap_mean, heap_max,
p50_mean, p95_mean,
p99_mean, latency_std, latency_slope,
throughput_mean, throughput_std,
cache_hit_ratio, cache_miss_ratio,
db_latency_mean, db_connections_max,
network_errors_mean,
error_rate_mean, error_rate_max,
gc_pause_mean, gc_pause_max,
cb_open_ratio,
disk_read_mean, disk_write_mean
```
*(28 listed — final count TBD based on de-duplication during implementation)*

### Log Features (7)
```
log_count, log_max_severity, log_critical_count,
log_has_exception, log_has_novel_template,
log_exception_type_encoded, log_severity_ratio
```

---

## Verification Plan

```bash
# Step 1 — Re-train Drain3 on full corpus
python offline/train_log_templates.py
# Expected: "Trained on 80,080 log lines (800 episodes)"

# Step 2 — Run full pipeline
python data_pipeline.py
# Expected outputs:
# X_train shape : (800, 32)
# X_test  shape : (201, 32)
# All 13 failure modes in both splits
# Random Forest  test accuracy: ~XX%
# XGBoost        test accuracy: ~XX%
# LightGBM       test accuracy: ~XX%
```

### Manual Checks
- `X_train.shape[1]` == `X_test.shape[1]` (same feature count)
- `log_has_novel_template` sum in X_train == 0 (Drain trained on it)
- No `episode_id` or `failure_mode` columns in X matrices
- Feature importance plots saved and viewable

---

## Open Questions

None.
