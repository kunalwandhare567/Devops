"""
offline/train_log_templates.py
==============================
Phase 1 — Offline Drain3 Training
----------------------------------
Trains a Drain3 TemplateMiner on the TRAINING split of telemetry_logs.csv
and freezes two artifacts that the online node loads read-only:

    known_log_templates.json  — list[int] of all known cluster IDs
    drain_state.bin           — Drain3 internal state snapshot

Run once before deployment (or whenever the training corpus changes):
    python offline/train_log_templates.py

Outputs are saved to the project root (alongside weibull_params.json,
hmm_matrices.pkl, etc.).
"""

import csv
import json
import os
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Paths — resolve relative to the project root (parent of offline/)
# ---------------------------------------------------------------------------
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR     = os.path.dirname(SCRIPT_DIR)
DATA_DIR     = os.path.join(ROOT_DIR, "data")
LOG_CSV      = os.path.join(DATA_DIR, "telemetry_logs.csv")
DRAIN_INI    = os.path.join(ROOT_DIR, "drain3.ini")
OUT_TEMPLATES = os.path.join(ROOT_DIR, "known_log_templates.json")
OUT_STATE     = os.path.join(ROOT_DIR, "drain_state.bin")

RANDOM_SEED  = 42
TEST_SIZE    = 0.20


def load_training_episodes(log_csv: str) -> pd.DataFrame:
    """
    Loads telemetry_logs.csv, performs an episode-level stratified 80/20 split,
    and returns ONLY the training-split rows.

    Splitting at episode level prevents data leakage — timesteps of the same
    failure incident cannot appear in both train and test.
    """
    print(f"Loading {log_csv} ...")
    df = pd.read_csv(log_csv, dtype=str).fillna("")

    # One row per unique episode with its failure mode label
    episodes = (
        df[["episode_id", "failure_mode"]]
        .drop_duplicates("episode_id")
        .reset_index(drop=True)
    )

    print(f"  Total episodes : {len(episodes)}")
    print(f"  Failure modes  : {sorted(episodes['failure_mode'].unique())}")

    train_eps, test_eps = train_test_split(
        episodes,
        test_size=TEST_SIZE,
        stratify=episodes["failure_mode"],
        random_state=RANDOM_SEED,
    )

    print(f"  Train episodes : {len(train_eps)}  "
          f"({len(train_eps)/len(episodes)*100:.0f}%)")
    print(f"  Test  episodes : {len(test_eps)}  "
          f"({len(test_eps)/len(episodes)*100:.0f}%)")

    # Verify all classes present in both splits
    train_modes = set(train_eps["failure_mode"])
    test_modes  = set(test_eps["failure_mode"])
    assert train_modes == test_modes, (
        f"Missing failure modes in test split: {train_modes - test_modes}"
    )
    print("  [OK] All failure modes represented in both splits")

    train_df = df[df["episode_id"].isin(train_eps["episode_id"])].copy()
    print(f"  Train log lines: {len(train_df):,}")
    return train_df


def train_drain(train_df: pd.DataFrame, drain_ini: str) -> object:
    """
    Initialises and trains a Drain3 TemplateMiner on every log_message in the
    training split. Returns the fitted miner.
    """
    try:
        from drain3 import TemplateMiner
        from drain3.template_miner_config import TemplateMinerConfig
    except ImportError:
        print("\nERROR: drain3 is not installed. Run:  pip install drain3")
        sys.exit(1)

    config = TemplateMinerConfig()
    if os.path.exists(drain_ini):
        config.load(drain_ini)
        print(f"\nDrain3 config loaded from {drain_ini}")
    else:
        print(f"\nWARNING: {drain_ini} not found — using Drain3 defaults")

    miner = TemplateMiner(config=config)

    messages = train_df["log_message"].dropna().tolist()
    print(f"Training Drain3 on {len(messages):,} log messages ...")
    for msg in messages:
        miner.add_log_message(str(msg))

    n_templates = len(list(miner.drain.clusters))
    print(f"  [OK] Training complete - {n_templates} unique templates discovered")
    return miner


def save_artifacts(miner, out_templates: str, out_state: str) -> None:
    """
    Persists:
      known_log_templates.json  — sorted list of cluster IDs
      drain_state.bin           — full Drain3 internal state
    """
    known_ids = sorted([c.cluster_id for c in miner.drain.clusters])
    with open(out_templates, "w", encoding="utf-8") as f:
        json.dump(known_ids, f, indent=2)
    print(f"\nSaved {len(known_ids)} template IDs -> {out_templates}")

    # Drain3's save_state writes to the path it was initialised with;
    # we work around this by re-initialising with the desired path.
    try:
        from drain3.file_persistence import FilePersistence
        from drain3 import TemplateMiner
        from drain3.template_miner_config import TemplateMinerConfig

        config = TemplateMinerConfig()
        if os.path.exists(DRAIN_INI):
            config.load(DRAIN_INI)

        persistence = FilePersistence(out_state)
        miner_with_persistence = TemplateMiner(
            persistence_handler=persistence, config=config
        )
        # Copy clusters from the trained miner
        miner_with_persistence.drain.clusters = miner.drain.clusters
        miner_with_persistence.drain.id_to_cluster = miner.drain.id_to_cluster
        persistence.save_state(miner_with_persistence.drain)
        print(f"Saved Drain3 state           -> {out_state}")
    except Exception as exc:
        # Fallback: pickle the miner directly
        import pickle
        with open(out_state, "wb") as fh:
            pickle.dump(miner, fh)
        print(f"Saved Drain3 state (pickle)  -> {out_state}  (reason: {exc})")

    print("\nArtifacts ready for the online feature node:")
    print(f"  {out_templates}")
    print(f"  {out_state}")


def print_template_summary(miner) -> None:
    """Prints the top templates to verify Drain learned the right patterns."""
    print("\nSample learned templates (top 15 by size):")
    clusters = sorted(
        miner.drain.clusters, key=lambda c: c.size, reverse=True
    )[:15]
    for c in clusters:
        print(f"  [{c.cluster_id:3d}] size={c.size:5d}  {c.get_template()[:80]}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    print("=" * 65)
    print("Phase 1 — Offline Drain3 Template Training")
    print("=" * 65)

    train_df  = load_training_episodes(LOG_CSV)
    miner     = train_drain(train_df, DRAIN_INI)
    save_artifacts(miner, OUT_TEMPLATES, OUT_STATE)
    print_template_summary(miner)

    print("\n" + "=" * 65)
    print("Done. Load these artifacts read-only in the online node.")
    print("=" * 65)


if __name__ == "__main__":
    main()
