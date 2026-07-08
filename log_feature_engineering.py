"""
log_feature_engineering.py
===========================
Phase 2 — Online Log Feature Extraction
----------------------------------------
Implements engineer_log_features() — the node function that converts a raw list
of log dicts (one inference cycle) into the classifier feature dict.

Also contains a __main__ driver that:
  1. Loads telemetry_logs.csv
  2. Performs the same episode-level stratified 80/20 split (seed=42)
  3. Applies engineer_log_features() per episode
  4. Saves X_train.csv, X_test.csv, y_train.csv, y_test.csv to data/

Offline prerequisites (produced by offline/train_log_templates.py):
  known_log_templates.json   — list[int] of known cluster IDs
  drain_state.bin            — Drain3 frozen state

Usage:
  python log_feature_engineering.py
"""

from __future__ import annotations

import json
import os
import pickle
import sys
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR       = os.path.dirname(os.path.abspath(__file__))
DATA_DIR       = os.path.join(ROOT_DIR, "data")
LOG_CSV        = os.path.join(DATA_DIR, "telemetry_logs.csv")
DRAIN_INI      = os.path.join(ROOT_DIR, "drain3.ini")
TEMPLATES_JSON = os.path.join(ROOT_DIR, "known_log_templates.json")
STATE_BIN      = os.path.join(ROOT_DIR, "drain_state.bin")

RANDOM_SEED    = 42
TEST_SIZE      = 0.20

# ---------------------------------------------------------------------------
# Encoding tables (must match offline/train_log_templates.py exactly)
# ---------------------------------------------------------------------------

SEVERITY_MAP: dict[str, int] = {
    "":         0,
    "INFO":     1,
    "WARNING":  2,
    "WARN":     2,
    "ERROR":    3,
    "CRITICAL": 4,
    "FATAL":    4,
}

EXCEPTION_TYPE_MAP: dict[str, int] = {
    "":                        0,
    "RuntimeException":        1,
    "SocketTimeoutException":  2,
    "NullPointerException":    3,
    "SystemOverloadException": 4,
}

# Fields that go into the classifier feature vector (no underscore)
CLASSIFIER_FIELDS = [
    "log_count",
    "log_max_severity",
    "log_critical_count",
    "log_has_exception",
    "log_has_novel_template",
    "log_exception_type_encoded",
    "log_severity_ratio",
]

# Evidence-only fields (underscore prefix) — carried through graph, never in CSV
EVIDENCE_FIELDS = [
    "_log_template_ids_seen",
    "_log_raw_lines",
]


# ---------------------------------------------------------------------------
# Core function — called once per inference cycle
# ---------------------------------------------------------------------------

def engineer_log_features(
    log_lines_this_cycle: list[dict],
    template_miner: Any,          # drain3.TemplateMiner, loaded once at startup
    known_template_ids: set[int], # loaded once at startup
) -> dict:
    """
    Converts a list of raw log dicts for one episode/cycle into the feature dict.

    Classifier features (7, no underscore prefix):
        log_count               int >= 0
        log_max_severity        int 0-4
        log_critical_count      int >= 0
        log_has_exception       0 or 1
        log_has_novel_template  0 or 1
        log_exception_type_encoded  int 0-4
        log_severity_ratio      float 0.0-1.0

    Evidence fields (2, underscore prefix — NEVER passed to classifier):
        _log_template_ids_seen  list[int]
        _log_raw_lines          list[str]

    Keyword-based flags (e.g., log_has_oom_keyword) are explicitly excluded
    from classifier input per spec §1.4. If wanted for narration, store them
    under an underscore-prefixed evidence field.
    """
    if not log_lines_this_cycle:
        return {
            "log_count":                  0,
            "log_max_severity":           0,
            "log_critical_count":         0,
            "log_has_exception":          0,
            "log_has_novel_template":     0,
            "log_exception_type_encoded": 0,
            "log_severity_ratio":         0.0,
            "_log_template_ids_seen":     [],
            "_log_raw_lines":             [],
        }

    severities:       list[int] = []
    template_ids:     list[int] = []
    raw_lines:        list[str] = []
    has_exception:    bool      = False
    has_novel:        bool      = False
    exc_type_encoded: int       = 0   # highest-priority exception seen

    for line in log_lines_this_cycle:
        log_level    = str(line.get("log_level",    "")).strip().upper()
        exc_type     = str(line.get("exception_type", "")).strip()
        log_message  = str(line.get("log_message",  "")).strip()

        # ---- Severity ----
        sev = SEVERITY_MAP.get(log_level, 0)
        severities.append(sev)

        # ---- Exception presence & encoding ----
        if exc_type:
            has_exception = True
            encoded = EXCEPTION_TYPE_MAP.get(exc_type, 0)
            # Keep the highest known encoding seen in this cycle
            if encoded > exc_type_encoded:
                exc_type_encoded = encoded

        raw_lines.append(log_message)

        # ---- Drain3 template matching ----
        if template_miner is not None and log_message:
            cluster = template_miner.match(log_message)
            if cluster is None:
                # No match within similarity threshold → genuinely novel
                has_novel  = True
                # add_log_message is called in OFFLINE training only.
                # Online we call match() read-only; novel lines get id=-1 sentinel.
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
        # ---- Classifier features ----
        "log_count":                  log_count,
        "log_max_severity":           log_max_sev,
        "log_critical_count":         log_critical_cnt,
        "log_has_exception":          int(has_exception),
        "log_has_novel_template":     int(has_novel),
        "log_exception_type_encoded": exc_type_encoded,
        "log_severity_ratio":         log_sev_ratio,
        # ---- Evidence fields (never in classifier) ----
        "_log_template_ids_seen":     template_ids,
        "_log_raw_lines":             raw_lines,
    }


# ---------------------------------------------------------------------------
# Artifact loading helpers — called once at graph startup
# ---------------------------------------------------------------------------

def load_template_miner(state_bin: str, drain_ini: str) -> Any:
    """
    Loads the frozen Drain3 TemplateMiner from disk.
    Returns None if drain3 is not installed (features will degrade gracefully).
    """
    try:
        from drain3 import TemplateMiner
        from drain3.template_miner_config import TemplateMinerConfig
        from drain3.file_persistence import FilePersistence
    except ImportError:
        print("WARNING: drain3 not installed — log_has_novel_template will be 0.")
        return None

    if not os.path.exists(state_bin):
        print(f"WARNING: {state_bin} not found — run offline/train_log_templates.py first.")
        return None

    config = TemplateMinerConfig()
    if os.path.exists(drain_ini):
        config.load(drain_ini)

    # Try native FilePersistence load first
    try:
        persistence = FilePersistence(state_bin)
        miner = TemplateMiner(persistence_handler=persistence, config=config)
        if len(list(miner.drain.clusters)) > 0:
            print(f"Loaded Drain3 state from {state_bin} "
                  f"({len(list(miner.drain.clusters))} templates)")
            return miner
    except Exception:
        pass

    # Fallback: pickle load (used when train script saved via pickle)
    try:
        with open(state_bin, "rb") as fh:
            miner = pickle.load(fh)
        print(f"Loaded Drain3 state (pickle) from {state_bin} "
              f"({len(list(miner.drain.clusters))} templates)")
        return miner
    except Exception as exc:
        print(f"ERROR: Could not load Drain3 state — {exc}")
        return None


def load_known_template_ids(templates_json: str) -> set[int]:
    """Loads the set of known cluster IDs from the frozen JSON artifact."""
    if not os.path.exists(templates_json):
        print(f"WARNING: {templates_json} not found — novelty detection disabled.")
        return set()
    with open(templates_json, "r", encoding="utf-8") as f:
        ids = json.load(f)
    print(f"Loaded {len(ids)} known template IDs from {templates_json}")
    return set(ids)


# ---------------------------------------------------------------------------
# Offline validation pipeline driver
# ---------------------------------------------------------------------------

def build_feature_matrices(
    log_csv:       str,
    template_miner: Any,
    known_ids:     set[int],
    test_size:     float = TEST_SIZE,
    seed:          int   = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Loads telemetry_logs.csv, splits at episode level (stratified), applies
    engineer_log_features() per episode, and returns (X_train, X_test, y_train, y_test).
    """
    print(f"\nLoading {log_csv} ...")
    df = pd.read_csv(log_csv, dtype=str).fillna("")
    print(f"  {len(df):,} log lines loaded.")

    # Episode-level stratified split
    episodes = (
        df[["episode_id", "failure_mode"]]
        .drop_duplicates("episode_id")
        .reset_index(drop=True)
    )
    train_eps, test_eps = train_test_split(
        episodes,
        test_size=test_size,
        stratify=episodes["failure_mode"],
        random_state=seed,
    )
    print(f"\n  Episode-level split:")
    print(f"    Train episodes : {len(train_eps)}")
    print(f"    Test  episodes : {len(test_eps)}")

    # Validate all failure modes appear in both splits
    assert set(train_eps["failure_mode"]) == set(test_eps["failure_mode"]), \
        "Not all failure modes present in both splits!"
    print(f"    [OK] All {episodes['failure_mode'].nunique()} failure modes in both splits")

    train_episode_ids = set(train_eps["episode_id"])
    test_episode_ids  = set(test_eps["episode_id"])
    ep_mode_map       = dict(zip(episodes["episode_id"], episodes["failure_mode"]))

    def process_split(ep_ids: set[str], split_name: str):
        rows, labels = [], []
        split_df = df[df["episode_id"].isin(ep_ids)]
        for ep_id, group in split_df.groupby("episode_id"):
            log_lines = group.to_dict(orient="records")
            features  = engineer_log_features(log_lines, template_miner, known_ids)
            # Extract only classifier features (no underscore)
            clf_row = {k: features[k] for k in CLASSIFIER_FIELDS}
            rows.append(clf_row)
            labels.append(ep_mode_map[ep_id])
        X = pd.DataFrame(rows, columns=CLASSIFIER_FIELDS)
        y = pd.Series(labels, name="failure_mode")
        return X, y

    print(f"\nComputing features for TRAIN split ...")
    X_train, y_train = process_split(train_episode_ids, "TRAIN")

    print(f"Computing features for TEST split ...")
    X_test, y_test   = process_split(test_episode_ids,  "TEST")

    return X_train, X_test, y_train, y_test


def validate_matrices(
    X_train: pd.DataFrame,
    X_test:  pd.DataFrame,
    y_train: pd.Series,
    y_test:  pd.Series,
) -> None:
    """Runs sanity checks and prints a verification report."""
    print("\n" + "=" * 65)
    print("VALIDATION REPORT")
    print("=" * 65)

    print(f"\nMatrix shapes:")
    print(f"  X_train : {X_train.shape}  y_train : {y_train.shape}")
    print(f"  X_test  : {X_test.shape}   y_test  : {y_test.shape}")

    assert X_train.shape[1] == len(CLASSIFIER_FIELDS), "Wrong feature count in X_train"
    assert X_test.shape[1]  == len(CLASSIFIER_FIELDS), "Wrong feature count in X_test"
    assert X_train.shape[0] == y_train.shape[0], "X_train / y_train row mismatch"
    assert X_test.shape[0]  == y_test.shape[0],  "X_test / y_test row mismatch"

    # Check log_has_novel_template in training split (should be 0 — Drain was trained on it)
    novel_train = X_train["log_has_novel_template"].sum()
    print(f"\n  log_has_novel_template sum in X_train : {novel_train} (expected 0)")
    if novel_train > 0:
        print("  WARNING: novel templates found in training split — check Drain state.")

    novel_test = X_test["log_has_novel_template"].sum()
    print(f"  log_has_novel_template sum in X_test  : {novel_test}")

    print(f"\nFailure mode distribution in y_train:")
    for mode, cnt in sorted(y_train.value_counts().items()):
        print(f"  {mode:25}: {cnt}")

    print(f"\nFailure mode distribution in y_test:")
    for mode, cnt in sorted(y_test.value_counts().items()):
        print(f"  {mode:25}: {cnt}")

    print(f"\nSample X_train row (first episode):")
    print(X_train.iloc[0].to_dict())

    print(f"\nFeature stats (X_train):")
    print(X_train.describe().to_string())

    # Confirm evidence fields NOT in X matrices
    ev_in_train = [c for c in X_train.columns if c.startswith("_")]
    ev_in_test  = [c for c in X_test.columns  if c.startswith("_")]
    assert not ev_in_train, f"Evidence fields found in X_train: {ev_in_train}"
    assert not ev_in_test,  f"Evidence fields found in X_test: {ev_in_test}"
    print("\n  [OK] No evidence/underscore fields in X matrices")

    print("\nAll validation checks passed [OK]")


def save_matrices(
    X_train: pd.DataFrame,
    X_test:  pd.DataFrame,
    y_train: pd.Series,
    y_test:  pd.Series,
    out_dir: str,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    X_train.to_csv(os.path.join(out_dir, "X_train.csv"), index=False)
    X_test.to_csv( os.path.join(out_dir, "X_test.csv"),  index=False)
    y_train.to_csv(os.path.join(out_dir, "y_train.csv"), index=False, header=True)
    y_test.to_csv( os.path.join(out_dir, "y_test.csv"),  index=False, header=True)
    print(f"\nOutput files saved to {out_dir}/:")
    for fname in ["X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv"]:
        path = os.path.join(out_dir, fname)
        size_kb = os.path.getsize(path) / 1024
        print(f"  {fname:20} {size_kb:7.1f} KB")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 65)
    print("Phase 2 — Log Feature Engineering Pipeline")
    print("=" * 65)

    # Load frozen artifacts (read-only, never retrained inline)
    template_miner = load_template_miner(STATE_BIN, DRAIN_INI)
    known_ids      = load_known_template_ids(TEMPLATES_JSON)

    # Build feature matrices
    X_train, X_test, y_train, y_test = build_feature_matrices(
        LOG_CSV, template_miner, known_ids
    )

    # Validate
    validate_matrices(X_train, X_test, y_train, y_test)

    # Save
    save_matrices(X_train, X_test, y_train, y_test, DATA_DIR)

    print("\n" + "=" * 65)
    print("Feature engineering complete.")
    print("=" * 65)


if __name__ == "__main__":
    main()
