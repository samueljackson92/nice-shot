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


@pytest.fixture
def lineage_df() -> pd.DataFrame:
    """A shot table with a reference column exercising every lineage shape.

    Graph: ``10 -> 9 -> 8 -> 7`` is a four-deep chain; ``5`` and ``6`` are
    siblings under ``4``; ``1`` is an orphan; ``2`` references itself and ``3``
    references a shot that is not in the table, so ``_build_reference_graph``
    drops both edges.

    Columns cover the comparison paths: ``ip`` varies, ``const`` has zero
    variance, ``sparse`` is mostly missing and holds an ``inf``, ``scenario`` is
    a genuine string, ``comment`` is free prose, and ``ip_str`` is a **numeric
    column stored as object strings with the literal "nan" for missing** -- the
    shape real parquet shot tables use, and the reason
    ``_coerce_reference_numeric`` exists.
    """
    n = 10
    shot_id = np.arange(1, n + 1)
    ip = shot_id * 10.0
    sparse = np.full(n, np.nan)
    sparse[0] = np.inf
    sparse[1] = 2.0
    return pd.DataFrame(
        {
            "shot_id": shot_id,
            "ref_shot": [None, 2, 999, None, 4, 4, None, 7, 8, 9],
            "ip": ip,
            "const": np.full(n, 5.0),
            "sparse": sparse,
            "scenario": ["H-mode"] * 5 + ["L-mode"] * 5,
            "comment": [f"shot {s} comment" for s in shot_id],
            "ip_str": ["nan", "nan"] + [f"{v * 2:.4f}" for v in ip[2:]],
        }
    )


# ---------------------------------------------------------------------------
# Layout tree walkers
#
# ``app.layout`` is inspected by several test modules. They go through
# :func:`layout_of` rather than reading ``mod.app.layout`` directly, so a
# future change to a callable layout needs no edit here.
# ---------------------------------------------------------------------------


def layout_of(module):
    """The rendered layout tree of *module*'s Dash app."""
    return module.app._layout_value()


def _walk(node):
    """Yield *node* and every Dash component below it."""
    yield node
    children = getattr(node, "children", None)
    candidates = children if isinstance(children, (list, tuple)) else [children]
    for child in candidates:
        if child is None or isinstance(child, str):
            continue
        yield from _walk(child)


def component_ids(node) -> list[str]:
    """Every string component id in a layout subtree.

    Pattern-matching (dict) ids are skipped -- they are matched by shape, not
    by name, so they cannot be compared against a callback's id string.
    """
    return [n.id for n in _walk(node) if isinstance(getattr(n, "id", None), str)]


def tab_values(node) -> list[str]:
    """Every ``dcc.Tab`` value in a layout tree."""
    return [n.value for n in _walk(node) if type(n).__name__ == "Tab" and isinstance(getattr(n, "value", None), str)]


def find_tab(node, value: str):
    """The ``dcc.Tab`` in a layout tree whose value is *value*, or ``None``."""
    for n in _walk(node):
        if type(n).__name__ == "Tab" and getattr(n, "value", None) == value:
            return n
    return None


# ---------------------------------------------------------------------------
# Importing app.py more than once
# ---------------------------------------------------------------------------


def import_app(name: str, argv: list[str]):
    """Import ``nice_shot/app.py`` again under the module name *name*.

    ``app.py`` reads ``sys.argv`` at import time and Python caches modules by
    name, so each alias gets its own module object and its own globals. This
    makes it possible to test settings that are fixed at import, such as the
    reference column or the trace backend.

    Do not use ``importlib.reload``: it rebinds globals in the module object
    that the session-scoped ``app_module`` fixture already holds, and the tests
    that share it depend on its accumulated state.
    """
    import importlib.util
    import pathlib

    import nice_shot

    old_argv = sys.argv
    sys.argv = ["niceshot", *argv]
    try:
        app_path = pathlib.Path(nice_shot.__file__).parent / "app.py"
        spec = importlib.util.spec_from_file_location(name, app_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    finally:
        sys.argv = old_argv
    return module


def write_shot_table(tmp_path, *, n: int = 12, start: int = 4000, reference: bool = False):
    """Write a small shot-statistics parquet and return its path.

    Set *reference* to add a ``ref_shot`` column that chains each shot to the
    one before it, which is what turns the Lineage tab on.
    """
    shot_id = np.arange(start, start + n)
    frame = pd.DataFrame(
        {
            "shot_id": shot_id,
            "feature_1": np.linspace(0.0, 1.0, n),
            "feature_2": np.linspace(1.0, 0.0, n),
            "ip_max": np.linspace(600.0, 800.0, n),
            "scenario": ["H-mode"] * (n // 2) + ["L-mode"] * (n - n // 2),
        }
    )
    if reference:
        frame["ref_shot"] = [None] + [str(s) for s in shot_id[:-1]]
    path = tmp_path / "shots.parquet"
    frame.to_parquet(path, index=False)
    return path


@pytest.fixture(scope="session")
def app_variant(tmp_path_factory):
    """Factory for extra imports of ``app.py`` with a given config.

    Usage::

        mod = app_variant("no_ref", {"projection_method": "pca"})

    Each call gets its own temp directory, its own projection cache (the cache
    key does not include the data path, so a shared path would collide on
    disk), and its own module name. Results are cached per name.
    """
    made: dict[str, object] = {}

    def make(name: str, config: dict, *, reference: bool = False, extra_argv: list[str] | None = None):
        if name in made:
            return made[name]
        tmp_path = tmp_path_factory.mktemp(f"app_{name}")
        shot_data_path = write_shot_table(tmp_path, reference=reference)
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.safe_dump(config))
        argv = [
            str(shot_data_path),
            "--config",
            str(config_path),
            "--umap-cache",
            str(tmp_path / "projection.npy"),
            *(extra_argv or []),
        ]
        module = import_app(f"nice_shot_app_{name}", argv)
        made[name] = module
        return module

    return make
