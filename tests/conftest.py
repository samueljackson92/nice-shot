"""Shared fixtures for the NiceShot! test suite."""

from __future__ import annotations

import importlib
import sys

import numpy as np
import pandas as pd
import pytest
import yaml


@pytest.fixture
def synthetic_shot_df() -> pd.DataFrame:
    """A small shot-statistics table with two well-separated clusters and some
    missing/infinite values to exercise imputation and variance-drop paths."""
    rng = np.random.default_rng(0)
    n_per_cluster = 8
    cluster_a = rng.normal(loc=0.0, scale=0.5, size=n_per_cluster)
    cluster_b = rng.normal(loc=10.0, scale=0.5, size=n_per_cluster)
    feature_1 = np.concatenate([cluster_a, cluster_b])
    feature_2 = np.concatenate([cluster_a * 2, cluster_b * 2])

    n = n_per_cluster * 2
    df = pd.DataFrame(
        {
            "shot_id": np.arange(1000, 1000 + n),
            "feature_1": feature_1,
            "feature_2": feature_2,
            "feature_const": np.full(n, 5.0),  # zero variance
            "feature_sparse": [np.nan] * (n - 2) + [1.0, 2.0],  # mostly NaN
            "machine": ["MAST-U"] * n,  # non-numeric column
        }
    )
    df.loc[0, "feature_1"] = np.inf
    return df


@pytest.fixture
def tmp_csv_path(tmp_path, synthetic_shot_df):
    path = tmp_path / "shots.csv"
    synthetic_shot_df.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def tmp_parquet_path(tmp_path, synthetic_shot_df):
    path = tmp_path / "shots.parquet"
    synthetic_shot_df.to_parquet(path, index=False)
    return str(path)


@pytest.fixture
def tmp_config_path(tmp_path):
    """A minimal, fully-defaulted config.yaml."""
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"projection_method": "pca"}))
    return str(path)


@pytest.fixture(scope="session")
def app_module(tmp_path_factory):
    """Import ``nice_shot.app`` once, pointed at a tiny synthetic dataset.

    ``app.py`` parses CLI args, loads a config file, and builds the initial
    dataset at *module import time* — so ``sys.argv`` must be set up before the
    first import. Subsequent tests reuse the already-imported module.
    """
    tmp_path = tmp_path_factory.mktemp("app_module")

    rng = np.random.default_rng(1)
    n = 16
    df = pd.DataFrame(
        {
            "shot_id": np.arange(2000, 2000 + n),
            "feature_1": rng.normal(size=n),
            "feature_2": rng.normal(size=n),
        }
    )
    shot_data_path = tmp_path / "shots.parquet"
    df.to_parquet(shot_data_path, index=False)

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "projection_method": "pca",
                "signals": ["ip", "ne"],
                "time_window": {"min_time": 0.0, "max_time": 1.0},
            }
        )
    )

    # One trace file per shot, so SHOW_TRACES is True and the time-trace
    # callbacks are registered. The files hold a third signal, "dalpha", that
    # the config does not list — the Configuration tab can select it, and
    # available_signals() can find it.
    data_dir = tmp_path / "traces"
    trace_dir = data_dir / "campaign"
    trace_dir.mkdir(parents=True)
    trace_time = np.linspace(0.0, 2.0, 40)
    for shot_id in df["shot_id"]:
        pd.DataFrame(
            {
                "time": trace_time,
                "ip": np.sin(trace_time),
                "ne": np.cos(trace_time),
                "dalpha": trace_time,
            }
        ).to_parquet(trace_dir / f"{shot_id}.parquet", index=False)

    umap_cache_path = tmp_path / "projection.npy"

    old_argv = sys.argv
    sys.argv = [
        "niceshot",
        str(shot_data_path),
        "--config",
        str(config_path),
        "--data-dir",
        str(data_dir),
        "--umap-cache",
        str(umap_cache_path),
    ]
    try:
        module = importlib.import_module("nice_shot.app")
    finally:
        sys.argv = old_argv

    return module
