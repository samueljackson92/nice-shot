"""
NiceShot!
Run from project root: uv run python nice_shot/app.py
"""

import argparse
import atexit
import hashlib
import importlib
import json
import logging
import math
import os
import pickle
import re
import shutil
import sys
import tempfile
import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, get_args

import dash
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import yaml
from dash import ALL, Input, Output, State, dash_table, dcc, html
from plotly.subplots import make_subplots
from pydantic import ValidationError

from nice_shot.analysis import (
    ProjectionModel,
    _apply_class_color,
    _apply_cluster_color,
    _apply_filter_mask,
    _apply_outlier_color,
    _build_reference_graph,
    _classify_reference_columns,
    _extract_shot_id,
    _fit_projection,
    _is_free_text,
    _load_projection_file,
    _run_classification,
    _run_clustering,
    _run_outlier_detection,
    _shap_decision_values,
    _spawn_sklearn,
    _transform_projection,
    candidate_target_cols,
    column_stds,
    compute_active_filter_ids,
    decision_surface,
    get_reference_lineage,
    is_projection_col,
    lineage_change_matrix,
    projection_col_names,
    projection_frame,
    rank_lineage_changes,
    reference_compare_columns,
    select_changed_columns,
)
from nice_shot.backends import (
    BackendConfig,
    ShotDataBackend,
    VariableShotDataBackend,
    create_shot_data_backend,
    create_trace_backend,
    create_variable_shot_data_backend,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

# Platform-appropriate user cache directory.
if sys.platform == "darwin":
    _CACHE_DIR = Path.home() / "Library" / "Caches" / "niceshot"
elif sys.platform == "win32":
    _CACHE_DIR = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "niceshot" / "cache"
else:
    _CACHE_DIR = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "niceshot"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, _HERE)
from config_schema import (  # noqa: E402
    AppConfig,
    ProjectionMetric,
    ProjectionOptions,
    TimeWindow,
    load_app_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="niceshot",
        description="NiceShot! — interactive tokamak shot dashboard",
    )
    parser.add_argument(
        "shot_data",
        metavar="SHOT_DATA",
        help="Path to shot data file (.parquet or .csv)",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8050, help="Port to listen on (default: 8050)")
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of gunicorn worker processes (default: 4). Ignored in --debug mode.",
    )
    parser.add_argument(
        "--debug",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run the single-process Flask dev server instead of gunicorn (default: off)",
    )
    parser.add_argument(
        "--config",
        default=os.path.join(_HERE, "config.yaml"),
        metavar="PATH",
        help="Path to config YAML (default: nice_shot/config.yaml)",
    )
    parser.add_argument(
        "--data-dir",
        default=os.path.join(_ROOT, "data", "mastu"),
        metavar="PATH",
        help="Directory containing per-shot parquet files",
    )
    parser.add_argument(
        "--umap-cache",
        default=str(_CACHE_DIR / "projection.npy"),
        metavar="PATH",
        help="Path to projection cache (.npy) — ignored when --projection is set",
    )
    parser.add_argument(
        "--projection",
        default=None,
        metavar="PATH",
        help="Path to a pre-computed 2D projection file (.npy, .csv, or .parquet). "
        "CSV/parquet must have a shot ID column and two coordinate columns. "
        "Numpy: shape (n,2) is matched positionally; shape (n,3) uses column 0 as shot_id. "
        "Skips UMAP/PCA computation entirely.",
    )
    parser.add_argument(
        "--shap-data",
        default=None,
        metavar="PATH",
        help="Path to a SHAP values NetCDF file (.nc). "
        "If provided, a SHAP decision-plot tab is shown in the left pane.",
    )

    # --- Options mirroring nice_shot/config.yaml (AppConfig) ---
    # All default to None: that's the sentinel meaning "not passed on the CLI",
    # so an omitted flag falls through to the config file's value, and an
    # omitted config value falls through to AppConfig's own default. An
    # explicit value here always overrides both. See config_schema.load_app_config.
    parser.add_argument(
        "--backend",
        default=None,
        help="Backend for loading per-shot time traces, e.g. parquet, uda, sal, or a "
        "plugin-registered name (overrides config.yaml: backend)",
    )
    parser.add_argument(
        "--signals",
        nargs="+",
        default=None,
        metavar="SIGNAL",
        help="Signals to load in the time-trace pane (overrides config.yaml: signals)",
    )
    parser.add_argument(
        "--min-time",
        type=float,
        default=None,
        help="Crop time traces to start at this time, seconds (overrides config.yaml: time_window.min_time)",
    )
    parser.add_argument(
        "--max-time",
        type=float,
        default=None,
        help="Crop time traces to end at this time, seconds (overrides config.yaml: time_window.max_time)",
    )
    parser.add_argument(
        "--timebase-hz",
        type=float,
        default=None,
        help="UDA backend: interpolate signals onto a uniform time grid at this rate "
        "(overrides config.yaml: uda.timebase_hz)",
    )
    parser.add_argument(
        "--projection-method",
        default=None,
        choices=["umap", "pca"],
        help="Algorithm for the 2D projection (overrides config.yaml: projection_method)",
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=None,
        help="Number of projection dimensions; the plots use the first two "
        "(overrides config.yaml: projection_options.n_components)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=None,
        help="Random seed for the projection (overrides config.yaml: projection_options.random_state)",
    )
    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=None,
        help="UMAP only: neighbourhood size (overrides config.yaml: projection_options.n_neighbors)",
    )
    parser.add_argument(
        "--min-dist",
        type=float,
        default=None,
        help="UMAP only: minimum distance between points in the embedding "
        "(overrides config.yaml: projection_options.min_dist)",
    )
    parser.add_argument(
        "--metric",
        default=None,
        choices=list(get_args(ProjectionMetric)),
        help="UMAP only: distance metric (overrides config.yaml: projection_options.metric)",
    )
    parser.add_argument(
        "--variable-column",
        default=None,
        help="Column holding the variable name in long-format shot data (overrides config.yaml: variable_column)",
    )
    parser.add_argument(
        "--umap-features",
        nargs="+",
        default=None,
        metavar="COLUMN",
        help="Columns to use as UMAP/PCA features (overrides config.yaml: umap_features)",
    )
    parser.add_argument(
        "--umap-exclude-features",
        nargs="+",
        default=None,
        metavar="COLUMN",
        help="Columns to exclude from UMAP/PCA features (overrides config.yaml: umap_exclude_features)",
    )
    parser.add_argument(
        "--reference-shot-col",
        default=None,
        help="Column holding each shot's reference/parent shot ID (overrides config.yaml: reference_shot_col)",
    )
    parser.add_argument(
        "--plugins",
        nargs="+",
        default=None,
        metavar="MODULE",
        help="Importable plugin module paths to load at startup (overrides config.yaml: plugins)",
    )
    parser.add_argument(
        "--refresh-interval-seconds",
        type=float,
        default=None,
        help="Poll the backend for new shots this often, in seconds; omit to disable live "
        "updates (overrides config.yaml: refresh_interval_seconds)",
    )
    parser.add_argument(
        "--backend-option",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help="Extra backend option, repeatable (merged into config.yaml: backend_options, overriding matching keys)",
    )

    # parse_known_args so Dash's own reloader flags don't cause errors
    args, _ = parser.parse_known_args()
    return args


_args = parse_args()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SHOT_DATA_PATH = _args.shot_data
MASTU_DATA_DIR = _args.data_dir
UMAP_CACHE_PATH = _args.umap_cache
PROJECTION_PATH: str | None = _args.projection
SHAP_PATH: str | None = _args.shap_data

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_cfg = load_app_config(_args)

BACKEND: str = _cfg.backend
TIME_TRACE_SIGNALS: list[str] = _cfg.signals
MIN_TIME: float = _cfg.time_window.min_time
MAX_TIME: float = _cfg.time_window.max_time
UDA_TIMEBASE_HZ: float | None = _cfg.uda.timebase_hz
PROJECTION_METHOD: str = _cfg.projection_method
PROJECTION_OPTIONS: ProjectionOptions = _cfg.projection_options
VARIABLE_COLUMN: str | None = _cfg.variable_column
UMAP_FEATURES: list[str] | None = _cfg.umap_features
UMAP_EXCLUDE_FEATURES: list[str] = _cfg.umap_exclude_features
REFERENCE_SHOT_COL: str | None = _cfg.reference_shot_col
REFRESH_INTERVAL_SECONDS: float | None = _cfg.refresh_interval_seconds

# ---------------------------------------------------------------------------
# Backend initialisation
# ---------------------------------------------------------------------------

for _plugin in _cfg.plugins:
    log.info("Loading plugin: %s", _plugin)
    importlib.import_module(_plugin)

_backend_options = dict(_cfg.backend_options)
if VARIABLE_COLUMN:
    _backend_options["variable_column"] = VARIABLE_COLUMN

_backend_config = BackendConfig(
    shot_data_path=SHOT_DATA_PATH,
    data_dir=MASTU_DATA_DIR,
    signals=TIME_TRACE_SIGNALS,
    min_time=MIN_TIME,
    max_time=MAX_TIME,
    timebase_hz=UDA_TIMEBASE_HZ,
    options=_backend_options,
)

# Long-format mode: the file holds one row per (shot, variable) and the user
# picks which variable to load. Nothing is read from the file body until then.
VARIABLE_MODE: bool = VARIABLE_COLUMN is not None
if VARIABLE_MODE and PROJECTION_PATH is not None:
    raise ValueError(
        "--projection cannot be combined with variable_column: a single pre-computed "
        "embedding cannot describe more than one variable. Remove one of them."
    )

_variable_backend: VariableShotDataBackend | None = None
_flat_backend: ShotDataBackend | None = None
if VARIABLE_MODE:
    _variable_backend = create_variable_shot_data_backend(SHOT_DATA_PATH, _backend_config)
else:
    _flat_backend = create_shot_data_backend(SHOT_DATA_PATH, _backend_config)
_trace_backend = create_trace_backend(BACKEND, _backend_config)

SHOW_TRACES: bool = _trace_backend.is_available()
if not SHOW_TRACES:
    log.info(
        "Time-trace panel greyed out — backend='%s', data-dir '%s' not found or empty.",
        BACKEND,
        MASTU_DATA_DIR,
    )

# Variable names offered in the selector — read from the variable column alone,
# so startup stays instant regardless of file size.
VARIABLES: list[str] = _variable_backend.variables(SHOT_DATA_PATH) if _variable_backend else []

# ---------------------------------------------------------------------------
# DatasetKey — what a cached Dataset is built from.
#
# This used to be the selected variable alone. The Configuration tab can now
# change the projection and the reference column too, and each combination
# produces a different Dataset, so all of them belong in the cache key.
#
# The key is frozen and hashable so it can index the cache directly, and it
# round-trips through a dcc.Store as a plain dict. from_store() puts the lists
# back into tuples: JSON has no tuple, and a list would make the key
# unhashable, which would silently turn every lookup into a miss and refit the
# projection on every callback.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DatasetKey:
    """Everything that decides which :class:`Dataset` a request needs."""

    variable: str | None = None
    projection_method: str = "umap"
    umap_features: tuple[str, ...] | None = None
    umap_exclude_features: tuple[str, ...] = ()
    # The projection hyper-parameters as canonical JSON. ProjectionOptions is a
    # pydantic model and so is not hashable; the dump also goes straight into
    # the cache hash, and a field added later is covered without an edit here.
    projection_options_json: str = ""
    reference_shot_col: str | None = None

    def to_store(self) -> dict[str, Any]:
        """A JSON-safe dict for a dcc.Store."""
        return {
            "variable": self.variable,
            "projection_method": self.projection_method,
            "umap_features": list(self.umap_features) if self.umap_features is not None else None,
            "umap_exclude_features": list(self.umap_exclude_features),
            "projection_options_json": self.projection_options_json,
            "reference_shot_col": self.reference_shot_col,
        }

    @classmethod
    def from_store(cls, data: dict[str, Any]) -> "DatasetKey":
        """Rebuild a key from a dcc.Store value, restoring the tuples."""
        features = data.get("umap_features")
        return cls(
            variable=data.get("variable"),
            projection_method=data.get("projection_method") or PROJECTION_METHOD,
            umap_features=tuple(features) if features is not None else None,
            umap_exclude_features=tuple(data.get("umap_exclude_features") or ()),
            projection_options_json=data.get("projection_options_json") or _projection_options_json(),
            reference_shot_col=data.get("reference_shot_col"),
        )

    def projection_options(self) -> ProjectionOptions:
        """The hyper-parameters this key was built with."""
        return ProjectionOptions.model_validate_json(self.projection_options_json)

    def projection_fields(self) -> dict[str, Any]:
        """The subset that decides the embedding.

        ``reference_shot_col`` is deliberately absent: it changes the reference
        graph and so the Dataset, but not where any point lands, so changing it
        must not throw away a fitted projection.
        """
        return {
            "variable": self.variable,
            "projection_method": self.projection_method,
            "umap_features": sorted(self.umap_features) if self.umap_features is not None else None,
            "umap_exclude_features": sorted(self.umap_exclude_features),
            "projection_options": self.projection_options_json,
        }


def _projection_options_json() -> str:
    """The config file's projection options as canonical JSON."""
    return json.dumps(PROJECTION_OPTIONS.model_dump(), sort_keys=True)


def default_dataset_key(variable: str | None = None) -> DatasetKey:
    """The key described by the config file and the command line."""
    return DatasetKey(
        variable=variable,
        projection_method=PROJECTION_METHOD,
        umap_features=tuple(UMAP_FEATURES) if UMAP_FEATURES is not None else None,
        umap_exclude_features=tuple(UMAP_EXCLUDE_FEATURES),
        projection_options_json=_projection_options_json(),
        reference_shot_col=REFERENCE_SHOT_COL,
    )


def _dataset_key_from_stores(
    variable: str | None,
    projection: dict[str, Any] | None,
    reference: dict[str, Any] | None,
) -> DatasetKey:
    """Build a key from the Configuration tab's stores.

    Each store holds ``None`` until Apply has run in this browser, and a dict
    afterwards. A dict whose field is ``null`` means the user cleared the
    setting, which is different from never having touched it.
    """
    base = default_dataset_key(variable)
    if projection:
        options = projection.get("projection_options")
        features = projection.get("umap_features")
        base = replace(
            base,
            projection_method=projection.get("projection_method") or base.projection_method,
            umap_features=tuple(features) if features else None,
            umap_exclude_features=tuple(projection.get("umap_exclude_features") or ()),
            projection_options_json=(json.dumps(options, sort_keys=True) if options else base.projection_options_json),
        )
    if reference is not None:
        base = replace(base, reference_shot_col=reference.get("value"))
    return base


def _coerce_key(value: DatasetKey | dict[str, Any] | str | None) -> DatasetKey:
    """Accept a key, a store value, a bare variable name, or nothing.

    ``None`` means "whatever the config file says", which keeps the old
    ``get_dataset(None)`` call shape working for flat mode and for the tests.
    A bare string is read as a variable name for the same reason.
    """
    if isinstance(value, DatasetKey):
        return value
    if isinstance(value, dict):
        return DatasetKey.from_store(value)
    if isinstance(value, str):
        return default_dataset_key(value)
    return default_dataset_key(None)


# ---------------------------------------------------------------------------
# UMAP / PCA projection — fitted once, then reused: new shots are transformed
# onto the existing embedding (see _transform_projection) rather than refit.
#
# The on-disk cache key is config-only (features/method/variable) — NOT a hash
# of the shot data file's bytes, so appending rows never invalidates the cache
# or forces a refit. This is a one-time cache-format break from older versions:
# the first run after upgrading always misses (no fitted model exists yet in the
# old cache format) and pays exactly one full refit, which then persists a model
# for every subsequent run/refresh to reuse.
# ---------------------------------------------------------------------------


def _umap_cache_hash(key: DatasetKey | dict[str, Any] | str | None = None) -> str:
    """The cache key for one fitted projection.

    It covers the shot data path, the feature selection, the method and every
    hyper-parameter -- everything that changes the embedding. It does **not**
    cover the rows, so appending shots never forces a refit; new rows are
    transformed onto the fitted model instead (see :func:`refresh_dataset`).

    The path is part of the key because two different shot statistics files
    with the same ``umap_features`` would otherwise share one embedding and one
    fitted model, and the second file's shots would be transformed onto the
    first file's projection.

    The options are hashed as a whole dump rather than field by field, so a
    field added to :class:`ProjectionOptions` later is covered without an edit
    here.
    """
    key = _coerce_key(key)
    h = hashlib.md5()
    h.update(json.dumps(key.projection_fields(), sort_keys=True).encode())
    h.update(os.path.abspath(SHOT_DATA_PATH).encode())
    # modelversion:3 adds the shot data path and the projection options. Every
    # cache written before it misses once and refits once -- the same one-time
    # cost described above.
    h.update(b"modelversion:3")
    return h.hexdigest()


def _umap_cache_path(key: DatasetKey | dict[str, Any] | str | None = None) -> str:
    """Cache path for *key*.

    Each variable is projected and cached separately, and a short hash of the
    projection settings is appended so two settings cannot fight over one
    path. The ``.hash`` sidecar written next to it stays the real guard -- this
    only keeps the files apart.
    """
    key = _coerce_key(key)
    stem, ext = os.path.splitext(UMAP_CACHE_PATH)
    suffix = _umap_cache_hash(key)[:8]
    if key.variable is None:
        return f"{stem}.{suffix}{ext}"
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in key.variable)
    return f"{stem}.{safe}.{suffix}{ext}"


def _umap_model_path(key: DatasetKey | dict[str, Any] | str | None = None) -> str:
    return _umap_cache_path(key) + ".model.joblib"


def _atomic_save(path: str, save_fn) -> None:
    """Write via a same-directory temp file + os.replace so a concurrent reader
    (e.g. another gunicorn worker) never observes a partially-written file.
    *save_fn* receives the temp path and must write the final bytes to it."""
    directory = os.path.dirname(path) or "."
    fd, tmp_path = tempfile.mkstemp(dir=directory, prefix=os.path.basename(path) + ".tmp")
    os.close(fd)
    try:
        save_fn(tmp_path)
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def _np_save_exact(path: str, arr: np.ndarray) -> None:
    """np.save() but writing to exactly *path* (no auto-appended .npy suffix)."""
    with open(path, "wb") as f:
        np.save(f, arr)


def _model_n_components(model: ProjectionModel | None) -> int:
    """How many components *model* was fitted with.

    Read from the fitted reducer, never from the current configuration: a cache
    written under an earlier n_components must be read at its own width.
    """
    if model is None:
        return 2
    return int(getattr(model.reducer, "n_components", 2))


def _project_new_rows(model: ProjectionModel, new_rows: pd.DataFrame) -> pd.DataFrame:
    """Transform *new_rows* onto *model* (never refits).

    The frame is as wide as the fitted model, so a refreshed shot gets every
    coordinate the existing rows have. Taking the width from *model* rather
    than from the current config matters: a cache fitted under an earlier
    n_components must keep its own width, or the saved array and its hash
    disagree and nothing reports it.
    """
    coords, shot_ids = _transform_projection(model, new_rows)
    return projection_frame(shot_ids, coords)


def get_projection_model(
    data: pd.DataFrame, key: DatasetKey | None = None
) -> tuple[ProjectionModel, np.ndarray, np.ndarray]:
    """Return (model, projection, shot_ids) covering every shot_id in *data*.

    Loads the fitted model + cached embedding when the config hash matches;
    shots present in *data* but not yet in the cached embedding are transformed
    (never refit) and merged in, and the extended embedding is re-saved so a
    future process restart doesn't need to re-transform them either.
    """
    key = _coerce_key(key)
    cache_path = _umap_cache_path(key)
    hash_path = cache_path + ".hash"
    shots_path = cache_path + ".shots.npy"
    model_path = _umap_model_path(key)
    current_hash = _umap_cache_hash(key)

    cache_valid = all(os.path.exists(p) for p in [cache_path, hash_path, shots_path, model_path])
    if cache_valid:
        with open(hash_path) as f:
            cache_valid = f.read().strip() == current_hash

    if cache_valid:
        log.info("Loading projection model from cache: %s", model_path)
        model: ProjectionModel = joblib.load(model_path)
        cached_projection = np.load(cache_path)
        cached_shot_ids = np.load(shots_path)
    else:
        log.info(
            "Fitting %s projection (this may take a moment)...",
            key.projection_method.upper(),
        )
        model, cached_projection, cached_shot_ids = _fit_projection(
            data,
            method=key.projection_method,
            umap_features=list(key.umap_features) if key.umap_features is not None else None,
            umap_exclude_features=list(key.umap_exclude_features),
            options=key.projection_options(),
        )
        _atomic_save(model_path, lambda tmp: joblib.dump(model, tmp))
        _atomic_save(cache_path, lambda tmp: _np_save_exact(tmp, cached_projection))
        _atomic_save(shots_path, lambda tmp: _np_save_exact(tmp, cached_shot_ids.astype(np.int64)))
        with open(hash_path, "w") as f:
            f.write(current_hash)
        log.info("Projection model saved to cache: %s", model_path)

    known_ids = set(int(s) for s in cached_shot_ids)
    new_mask = ~data["shot_id"].astype(int).isin(known_ids)
    new_rows = data[new_mask]
    if new_rows.empty:
        return model, cached_projection, cached_shot_ids

    log.info("Transforming %d new shot(s) onto the existing projection...", len(new_rows))
    new_emb = _project_new_rows(model, new_rows)
    coord_cols = projection_col_names(cached_projection.shape[1])
    all_projection = np.concatenate([cached_projection, new_emb[coord_cols].values], axis=0)
    all_shot_ids = np.concatenate([cached_shot_ids, new_emb["shot_id"].values.astype(np.int64)], axis=0)

    _atomic_save(cache_path, lambda tmp: _np_save_exact(tmp, all_projection))
    _atomic_save(shots_path, lambda tmp: _np_save_exact(tmp, all_shot_ids.astype(np.int64)))

    return model, all_projection, all_shot_ids


from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

# ---------------------------------------------------------------------------
# Dataset — everything derived from one variable's rows.
#
# In long-format mode a dataset is built lazily the first time a variable is
# selected and then cached per process. The selected variable is held in browser
# state and passed into every data callback, so each gunicorn worker builds its
# own cache on demand and no worker can serve another variable's data.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Dataset:
    """A loaded, projected shot table plus the indices built from it."""

    df: pd.DataFrame
    search_cols: list[str]
    search_ids: np.ndarray
    search_X: np.ndarray
    search_nn: NearestNeighbors
    x_label: str = "Dim 1"
    y_label: str = "Dim 2"
    ref_adjacency: dict[int, list[int]] = field(default_factory=dict)
    ref_parent: dict[int, int] = field(default_factory=dict)
    # Lineage-tab precomputation: the candidate comparison columns, which of
    # them compare as numbers, and each one's spread over the whole table. All
    # three are per-dataset, so they are built once in _finalize_dataset rather
    # than on every click -- see the Lineage callbacks.
    ref_compare_cols: tuple[str, ...] = ()
    ref_numeric_cols: tuple[str, ...] = ()
    ref_stds: pd.Series | None = None
    shap_idx: dict[int, int] = field(default_factory=dict)
    # None exactly when --projection (a precomputed embedding file) is in use —
    # there's no fitted transformer to reuse, so refresh_dataset() is a no-op then.
    model: ProjectionModel | None = None


def _numeric_cols_of(data: pd.DataFrame) -> list[str]:
    return sorted(c for c in data.select_dtypes(include=[np.number]).columns if c != "shot_id")


def _project_dataset(data: pd.DataFrame, key: DatasetKey) -> tuple[pd.DataFrame, str, str, ProjectionModel | None]:
    """Merge 2D projection coordinates onto *data*. Returns (data, x_label, y_label, model)."""
    if PROJECTION_PATH is not None:
        emb, x_label, y_label = _load_projection_file(PROJECTION_PATH, data)
        data = data.merge(emb, on="shot_id", how="inner")
        return data, x_label, y_label, None

    model, projection, proj_shot_ids = get_projection_model(data, key)
    emb = projection_frame(proj_shot_ids, projection)
    data = data.merge(emb, on="shot_id", how="inner")
    return data, "Dim 1", "Dim 2", model


def _finalize_dataset(
    data: pd.DataFrame,
    model: ProjectionModel | None,
    x_label: str,
    y_label: str,
    shap_idx: dict[int, int],
    key: DatasetKey | None = None,
) -> Dataset:
    """Build the similarity index and reference graph from an already-projected *data*.

    Cheap enough to rerun on every refresh (unlike the projection fit itself) —
    see :func:`refresh_dataset`.
    """
    key = _coerce_key(key)
    umap_features = list(key.umap_features) if key.umap_features is not None else None
    reference_shot_col = key.reference_shot_col
    # Exclude the projection coordinates. They are numeric columns sitting in
    # the same frame, so the default "all numeric columns" would otherwise feed
    # the embedding back in as similarity features -- and a projection with
    # more components would weight it more heavily still.
    feature_cols = [c for c in _numeric_cols_of(data) if not is_projection_col(c)]
    search_cols = [f for f in (umap_features or feature_cols) if f in data.columns]
    search_raw = data[["shot_id"] + search_cols].copy()
    # astype("float64") converts any numpy-nullable columns (Int64/Float64,
    # which use pd.NA) to plain numpy floats with NaN -- pd.NA can't be cast
    # with float(), which .values.astype(float) below would otherwise hit.
    search_raw[search_cols] = search_raw[search_cols].astype("float64").replace([np.inf, -np.inf], np.nan)
    # Impute with column means so every shot is searchable, even those with missing features.
    search_X = StandardScaler().fit_transform(
        SimpleImputer(strategy="mean").fit_transform(search_raw[search_cols].values.astype(float))
    )
    search_nn = NearestNeighbors(metric="euclidean", algorithm="auto").fit(search_X)
    log.info("Similarity index built: %d shots × %d features", len(search_raw), len(search_cols))

    ref_adjacency: dict[int, list[int]] = {}
    ref_parent: dict[int, int] = {}
    if reference_shot_col and reference_shot_col in data.columns:
        ref_adjacency, ref_parent = _build_reference_graph(data, reference_shot_col)
        if ref_adjacency:
            log.info(
                "Reference graph: '%s' — %d edges, %d unique nodes",
                reference_shot_col,
                len(ref_parent),
                len(ref_adjacency),
            )
        else:
            log.warning("reference_shot_col='%s' produced no valid edges.", reference_shot_col)

    # Lineage-tab precomputation. Only worth doing when there is a reference
    # graph to walk, and deliberately independent of `numeric_cols`: the tab
    # needs the numeric/categorical split (text compares by equality, not
    # subtraction), and long-format sources keep numeric columns as dtype=object
    # (backends.py passes coerce_objects=False there). See
    # _classify_reference_columns.
    ref_compare_cols: tuple[str, ...] = ()
    ref_numeric_cols: tuple[str, ...] = ()
    ref_stds: pd.Series | None = None
    if ref_adjacency:
        # ref_adjacency is only non-empty when the column is set, but be explicit for the type.
        exclude = [reference_shot_col] if reference_shot_col else []
        compare = reference_compare_columns(data, search_cols=search_cols, exclude=exclude)
        numeric, _categorical = _classify_reference_columns(data, compare)
        ref_compare_cols = tuple(compare)
        ref_numeric_cols = tuple(numeric)
        ref_stds = column_stds(data, numeric)
        log.info(
            "Lineage comparison: %d candidate column(s), %d compare as numbers",
            len(ref_compare_cols),
            len(ref_numeric_cols),
        )

    return Dataset(
        df=data,
        search_cols=search_cols,
        search_ids=search_raw["shot_id"].values,
        search_X=search_X,
        search_nn=search_nn,
        x_label=x_label,
        y_label=y_label,
        ref_adjacency=ref_adjacency,
        ref_parent=ref_parent,
        ref_compare_cols=ref_compare_cols,
        ref_numeric_cols=ref_numeric_cols,
        ref_stds=ref_stds,
        shap_idx=shap_idx,
        model=model,
    )


def _build_dataset(data: pd.DataFrame, key: DatasetKey) -> Dataset:
    """Project *data*, build the similarity index and the reference graph."""
    # Positional index for SHAP lookup, taken before the projection merge drops rows.
    # The .nc file uses 0-based indices matching the original sorted shot order.
    # Fixed at first build — refresh_dataset() carries it over unchanged, since it
    # indexes into a static SHAP file that never grows with new shots.
    shap_idx = {int(s): i for i, s in enumerate(data["shot_id"].values) if pd.notna(s)}
    data, x_label, y_label, model = _project_dataset(data, key)
    return _finalize_dataset(data, model, x_label, y_label, shap_idx, key)


# Bounded, because the Configuration tab can now ask for a new key at will and
# each Dataset holds a full DataFrame, a scaled feature matrix and a fitted
# NearestNeighbors. Least-recently-used entries are dropped.
_DATASET_CACHE_MAX = 4
_dataset_cache: OrderedDict[DatasetKey, Dataset] = OrderedDict()
_dataset_cache_lock = threading.Lock()


def _cache_get(key: DatasetKey) -> Dataset | None:
    with _dataset_cache_lock:
        ds = _dataset_cache.get(key)
        if ds is not None:
            _dataset_cache.move_to_end(key)
        return ds


def _cache_put(key: DatasetKey, ds: Dataset) -> None:
    with _dataset_cache_lock:
        _dataset_cache[key] = ds
        _dataset_cache.move_to_end(key)
        while len(_dataset_cache) > _DATASET_CACHE_MAX:
            _dataset_cache.popitem(last=False)


# ---------------------------------------------------------------------------
# Cross-worker cache directory
#
# gunicorn hands each request to whichever worker is free, so the worker that
# trained a model is almost never the one asked to explain a shot with it. A
# plain in-process dict therefore misses about (workers - 1) / workers of the
# time: the SHAP pane reads "train again" at random, and a click on a plot
# recomputes the decision surface because that worker has never drawn it.
#
# The directory below is a second level behind both caches. It is named once,
# at import, which under gunicorn happens in the master before the workers are
# forked (preload_app), so every worker inherits the same path.
# ---------------------------------------------------------------------------
_SHARED_CACHE_DIR = os.path.join(tempfile.gettempdir(), f"nice-shot-cache-{os.getpid()}-{uuid.uuid4().hex[:8]}")
_SHARED_CACHE_OWNER = os.getpid()


def _shared_cache_cleanup() -> None:
    """Remove the directory, but only in the process that named it.

    Forked workers inherit this handler, and a worker exiting must not delete
    the cache its siblings are still reading from.
    """
    if os.getpid() == _SHARED_CACHE_OWNER:
        shutil.rmtree(_SHARED_CACHE_DIR, ignore_errors=True)


atexit.register(_shared_cache_cleanup)


def _shared_path(name: str) -> str:
    return os.path.join(_SHARED_CACHE_DIR, name + ".pkl")


def _shared_load(name: str):
    """Read one shared entry, or None when it is absent or unreadable.

    A miss is never an error: the caller recomputes, exactly as it did when
    this level did not exist.
    """
    try:
        with open(_shared_path(name), "rb") as handle:
            return pickle.load(handle)
    except FileNotFoundError:
        return None
    except Exception as exc:
        log.debug("[shared cache] could not read %s: %s", name, exc)
        return None


def _write_pickle(path: str, value) -> None:
    with open(path, "wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)


def _shared_store(name: str, value) -> None:
    """Write one shared entry atomically, so a sibling worker never reads a
    half-written pickle. Failure is logged and ignored -- the in-process dict
    still holds the value for this worker."""
    try:
        os.makedirs(_SHARED_CACHE_DIR, exist_ok=True)
        _atomic_save(_shared_path(name), lambda tmp: _write_pickle(tmp, value))
    except Exception as exc:
        log.debug("[shared cache] could not write %s: %s", name, exc)


def _shared_names(prefix: str) -> list[str]:
    try:
        return [n[:-4] for n in os.listdir(_SHARED_CACHE_DIR) if n.startswith(prefix) and n.endswith(".pkl")]
    except OSError:
        return []


def _shared_drop(name: str) -> None:
    try:
        os.remove(_shared_path(name))
    except OSError:
        pass


def _shared_prune(prefix: str, limit: int) -> list[str]:
    """Keep the *limit* most recently written entries under *prefix*.

    Returns the names dropped, so a caller can cascade the eviction.
    """
    names = _shared_names(prefix)
    if len(names) <= limit:
        return []

    def written_at(name: str) -> float:
        try:
            return os.path.getmtime(_shared_path(name))
        except OSError:
            return 0.0

    doomed = sorted(names, key=written_at)[: len(names) - limit]
    for name in doomed:
        _shared_drop(name)
    return doomed


# ---------------------------------------------------------------------------
# Trained classifier cache
#
# A fitted estimator is not JSON and must not be sent to the browser, so the
# store carries an opaque token and the model itself stays here. Two are kept
# per worker: enough that flipping back to a previous model is free, few
# enough that a forgotten Gaussian Process cannot pin memory. The shared
# directory holds the same two, so any worker can answer for either.
#
# A token that no longer resolves -- a restarted server or an eviction -- is
# not an error. Everything that reads this cache degrades to "train again".
# ---------------------------------------------------------------------------
_MODEL_CACHE_MAX = 2
_MODEL_PREFIX = "model-"
_model_cache: OrderedDict[str, dict] = OrderedDict()
_model_cache_lock = threading.Lock()

# Surface grids, keyed by (token, x_col, y_col, class_index, shots). The last
# part is a fingerprint of the filtered shot ids: the surface is cropped to
# the shots on screen, so moving a filter asks a different question and gets
# its own grid. Room for a few, because stepping a filter back and forth is
# the usual way of reading one.
_SURFACE_CACHE_MAX = 16
_SURFACE_PREFIX = "surface-"
_surface_cache: OrderedDict[tuple, dict] = OrderedDict()
_surface_cache_lock = threading.Lock()


def _model_remember(token: str, entry: dict) -> None:
    """Hold *entry* in this worker's dict, evicting the least recent."""
    evicted: list[str] = []
    with _model_cache_lock:
        _model_cache[token] = entry
        _model_cache.move_to_end(token)
        while len(_model_cache) > _MODEL_CACHE_MAX:
            evicted.append(_model_cache.popitem(last=False)[0])
    for gone in evicted:
        _surface_cache_drop(gone)


def _model_get(token: str | None) -> dict | None:
    if not token:
        return None
    with _model_cache_lock:
        entry = _model_cache.get(token)
        if entry is not None:
            _model_cache.move_to_end(token)
            return entry
    # Not this worker's model, but it may still be another's.
    entry = _shared_load(_MODEL_PREFIX + token)
    if entry is None:
        return None
    _model_remember(token, entry)
    return entry


def _model_put(token: str, entry: dict) -> None:
    _shared_store(_MODEL_PREFIX + token, entry)
    for gone in _shared_prune(_MODEL_PREFIX, _MODEL_CACHE_MAX):
        _surface_cache_drop(gone[len(_MODEL_PREFIX) :])
    _model_remember(token, entry)


def _surface_cache_drop(token: str) -> None:
    """Forget the grids of an evicted model, which can never be asked for again."""
    with _surface_cache_lock:
        for key in [k for k in _surface_cache if k[0] == token]:
            del _surface_cache[key]
    for name in _shared_names(_SURFACE_PREFIX + token + "-"):
        _shared_drop(name)


def _surface_name(key: tuple) -> str:
    """A filename for one grid key: the token stays legible so an evicted
    model can drop its grids by prefix, the rest is hashed."""
    token = str(key[0])
    digest = hashlib.blake2b(repr(key[1:]).encode(), digest_size=12).hexdigest()
    return f"{_SURFACE_PREFIX}{token}-{digest}"


def _surface_remember(key: tuple, grid: dict) -> None:
    with _surface_cache_lock:
        _surface_cache[key] = grid
        _surface_cache.move_to_end(key)
        while len(_surface_cache) > _SURFACE_CACHE_MAX:
            _surface_cache.popitem(last=False)


def _surface_get(key: tuple) -> dict | None:
    with _surface_cache_lock:
        grid = _surface_cache.get(key)
        if grid is not None:
            _surface_cache.move_to_end(key)
            return grid
    grid = _shared_load(_surface_name(key))
    if grid is None:
        return None
    _surface_remember(key, grid)
    return grid


def _surface_put(key: tuple, grid: dict) -> None:
    _shared_store(_surface_name(key), grid)
    _shared_prune(_SURFACE_PREFIX, _SURFACE_CACHE_MAX)
    _surface_remember(key, grid)


def get_dataset(key: DatasetKey | dict[str, Any] | str | None) -> Dataset | None:
    """Return the dataset for *key*, building and caching it on first use.

    *key* is a :class:`DatasetKey`, a value read from the ``dataset-key`` store,
    or ``None`` for whatever the config file says.

    Returns ``None`` in long-format mode until the user picks a variable — that
    is the signal for callbacks to render their "select a variable" empty state.
    Once built, a dataset stays cached until :func:`refresh_dataset` replaces it
    (e.g. via the periodic poll callback) — it is never silently reloaded.
    """
    key = _coerce_key(key)
    ds = _cache_get(key)
    if ds is not None:
        return ds

    if _variable_backend is not None:
        if key.variable is None:
            return None
        data = _variable_backend.load_variable(SHOT_DATA_PATH, key.variable)
    else:
        assert _flat_backend is not None  # exactly one backend is created at startup
        data = _flat_backend.load(SHOT_DATA_PATH)
    ds = _build_dataset(data, key)

    _cache_put(key, ds)
    return ds


def refresh_dataset(key: DatasetKey | dict[str, Any] | str | None) -> int | None:
    """Poll the backend for shots newer than the current dataset and merge them in.

    Never refits the projection — new rows are transformed onto the existing
    model (see :func:`_project_new_rows`). Returns the new max shot_id on
    success, or ``None`` if there's nothing cached yet to refresh, no new rows
    were found, or refresh isn't supported (``--projection`` mode has no fitted
    model to reuse).
    """
    key = _coerce_key(key)
    ds = _cache_get(key)
    if ds is None or ds.df.empty:
        return None
    if ds.model is None:
        log.warning("refresh_dataset: --projection mode has no fitted model; restart to pick up new data.")
        return None

    since_id = int(ds.df["shot_id"].max())
    try:
        if _variable_backend is not None:
            # A cached Dataset only ever exists for a real variable in
            # long-format mode (get_dataset(None) returns None without caching).
            assert key.variable is not None
            new_rows = _variable_backend.poll_new_variable(SHOT_DATA_PATH, key.variable, since_id)
        else:
            assert _flat_backend is not None
            new_rows = _flat_backend.poll_new(SHOT_DATA_PATH, since_id)
    except Exception:
        log.exception("refresh_dataset: poll_new failed for variable=%r", key.variable)
        return None

    if new_rows is None or new_rows.empty:
        return None

    emb = _project_new_rows(ds.model, new_rows)
    new_rows = new_rows.merge(emb, on="shot_id", how="inner")
    combined = pd.concat([ds.df, new_rows], ignore_index=True)

    new_ds = _finalize_dataset(combined, ds.model, ds.x_label, ds.y_label, ds.shap_idx, key)

    cache_path = _umap_cache_path(key)
    # The width follows the fitted model, so the saved array keeps matching the
    # hash that describes it even if the configured n_components has changed.
    all_projection = combined[projection_col_names(_model_n_components(ds.model))].values
    all_shot_ids = combined["shot_id"].values.astype(np.int64)
    _atomic_save(cache_path, lambda tmp: _np_save_exact(tmp, all_projection))
    _atomic_save(cache_path + ".shots.npy", lambda tmp: _np_save_exact(tmp, all_shot_ids))

    _cache_put(key, new_ds)
    log.info("refresh_dataset: merged %d new shot(s), latest shot_id=%d", len(new_rows), int(all_shot_ids.max()))
    return int(all_shot_ids.max())


def _require_dataset(key: DatasetKey | dict[str, Any] | str | None) -> Dataset:
    """Like :func:`get_dataset` but never ``None`` — for flat mode, where the
    single dataset is always available."""
    ds = get_dataset(key)
    if ds is None:
        raise RuntimeError(f"No dataset available for key {_coerce_key(key)!r}")
    return ds


# ---------------------------------------------------------------------------
# Column schema — drives every widget in the layout.
#
# Long-format mode reads it from file metadata (no rows), which is valid because
# every variable in the file shares the same columns; flat mode takes it from
# the one dataset, which is loaded eagerly here exactly as it always was.
# ---------------------------------------------------------------------------
if _variable_backend is not None:
    _schema_df = _variable_backend.schema(SHOT_DATA_PATH)
else:
    _schema_df = _require_dataset(None).df

numeric_cols = _numeric_cols_of(_schema_df)
all_cols = sorted(c for c in _schema_df.columns if c != "shot_id")
_pair_axis_cols = ["shot_id"] + numeric_cols
_search_cols = [f for f in (UMAP_FEATURES or numeric_cols) if f in _schema_df.columns]
# Anything with a handful of distinct values can be learned; a continuous
# column cannot. Computed once here so the dropdown is populated at import,
# like every other column picker in the layout.
_classify_target_cols = candidate_target_cols(_schema_df)

_table_cols = [c for c in _schema_df.columns if not is_projection_col(c)]
_CLUSTER_COLOR_VALUE = "__cluster__"
_OUTLIER_COLOR_VALUE = "__outliers__"
_CLASS_COLOR_VALUE = "__label__"
_color_col_options = (
    [{"label": "shot_id", "value": "shot_id"}]
    + [{"label": c, "value": c} for c in all_cols]
    + [
        {"label": "Cluster", "value": _CLUSTER_COLOR_VALUE},
        {"label": "Outliers", "value": _OUTLIER_COLOR_VALUE},
        {"label": "Label (model)", "value": _CLASS_COLOR_VALUE},
    ]
)

_table_column_defs = [
    {"name": c, "id": c, "type": "numeric", "format": {"specifier": ".4g"}}
    if pd.api.types.is_float_dtype(_schema_df[c])
    else {"name": c, "id": c}
    for c in _table_cols
]

# The toggle is shown whenever the reference column exists. In flat mode we can
# also confirm it yields edges; in long-format mode no rows are loaded yet.
if VARIABLE_MODE:
    SHOW_REF_TOGGLE = bool(REFERENCE_SHOT_COL and REFERENCE_SHOT_COL in _schema_df.columns)
else:
    SHOW_REF_TOGGLE = bool(_require_dataset(None).ref_adjacency)

# ---------------------------------------------------------------------------
# Lineage tab — sizing and caps
#
# Candidate columns come from the Dataset (ds.ref_compare_cols), which is built
# per dataset and therefore correct in long-format mode too. This list is only
# the startup fallback for the dropdown, before any lineage has been resolved.
# ---------------------------------------------------------------------------
_lin_fallback_cols = [c for c in (UMAP_FEATURES or numeric_cols) if c != REFERENCE_SHOT_COL]
_LIN_DEFAULT_N = 20  # columns seeded by "Top changed"
_LIN_OPTION_LIMIT = 200  # options returned per dropdown search
_LIN_MAX_STYLE_CELLS = 4000  # cap on style_data_conditional entries
_LIN_CARD_MAX = 500  # ranked rows the summary card will render
_LIN_CARD_H = "300px"  # scroll height of the summary card's ranked table
_LIN_SPARK_PANELS = 12  # variables seeded into the sparkline selector
# A safety limit, not a setting: the panels are whatever is selected, but a few
# hundred Plotly subplots take seconds to draw and scroll. The view says so
# whenever it bites, which on a hand-made selection it never does.
_LIN_SPARK_HARD_MAX = 48
_LIN_NOTE_CHARS = 200  # free-text truncation in the notes table

_LIN_SCOPE_OPTIONS = [
    {"label": "Ancestor chain", "value": "chain"},
    {"label": "Connected", "value": "component"},
    {"label": "Siblings", "value": "siblings"},
]
_LIN_METRIC_OPTIONS = [
    {"label": "z-scored change", "value": "zscore"},
    {"label": "percent change", "value": "percent"},
    {"label": "absolute change", "value": "absolute"},
]
# The summary card ranks by its own measure: "what moved most" is a different
# question from "what should the history table colour by". Both options are
# comparable between variables, which is what a ranking needs -- one against
# the spread of the column across every shot, the other against the value the
# reference shot held. The percentage says nothing useful about a column whose
# reference value is near 0, so the card reports it as undefined rather than
# ranking on an invented number.
_LIN_CARD_METRIC_OPTIONS = [
    {"label": " z-scored", "value": "zscore"},
    {"label": " percentage", "value": "percent"},
]
# Shown under the colour ramp so nobody reads an "absolute" heatmap as if the
# intensities were comparable between columns.
_LIN_METRIC_HINTS = {
    "zscore": "Change divided by the column spread across all shots. Comparable between columns.",
    "percent": "Change as a percentage of the previous value. Undefined when that value is 0.",
    "absolute": "Change in the column's own units. Not comparable between columns.",
}

# ---------------------------------------------------------------------------
# SHAP data loading
# ---------------------------------------------------------------------------
SHOW_SHAP = False
_shap_da = None
_shap_feature_names: list[str] = []


def _load_shap(path: str) -> tuple:
    import xarray as xr

    _shap_ds = xr.open_dataset(path)
    _da = _shap_ds["__xarray_dataarray_variable__"]
    _feature_names = list(_da.coords["feature"].values)
    return _da, _feature_names


if SHAP_PATH is not None:
    try:
        _shap_da, _shap_feature_names = _load_shap(SHAP_PATH)
        SHOW_SHAP = True
        _n_shap = len(_shap_da.coords["shot_id"])
        _n_feat = len(_shap_feature_names)
        log.info(
            "SHAP data loaded: %s (%d shots x %d features)",
            SHAP_PATH,
            _n_shap,
            _n_feat,
        )
    except Exception as _shap_exc:
        log.warning("Could not load SHAP data from '%s': %s", SHAP_PATH, _shap_exc)

# ---------------------------------------------------------------------------
# Shot time-trace helpers
# ---------------------------------------------------------------------------


def _effective_signals(signals: list[str] | None) -> list[str]:
    """Return *signals*, or the signal list from the config file if it is empty.

    The Configuration tab sends the live selection. A page that has not applied
    one yet sends nothing, and then the config file applies as before.
    """
    return list(signals) if signals else list(TIME_TRACE_SIGNALS)


def _effective_window(time_window: dict | None) -> tuple[float, float]:
    """Return the live time window, or the window from the config file."""
    if not time_window:
        return MIN_TIME, MAX_TIME
    return (
        float(time_window.get("min_time", MIN_TIME)),
        float(time_window.get("max_time", MAX_TIME)),
    )


def _trace_overrides(
    signals: list[str] | None = None,
    time_window: dict | None = None,
    timebase_hz: dict | None = None,
    backend_options: dict | None = None,
) -> dict[str, Any]:
    """Build the per-request BackendConfig overrides from the tab's stores.

    Each store holds ``None`` until Apply has run in this browser, so an
    untouched setting is simply absent and the config file's value applies.
    ``timebase_hz`` and ``backend_options`` arrive wrapped in a dict, because a
    cleared value is a real ``None`` that has to be told apart from "not set".
    """
    overrides: dict[str, Any] = {}
    if signals:
        overrides["signals"] = list(signals)
    if time_window:
        min_time, max_time = _effective_window(time_window)
        overrides["min_time"] = min_time
        overrides["max_time"] = max_time
    if timebase_hz is not None:
        overrides["timebase_hz"] = timebase_hz.get("value")
    if backend_options is not None:
        # Merged over the config file's options, so the tab only has to carry
        # the keys it changes -- the same rule as --backend-option on the CLI.
        merged = dict(_backend_options)
        merged.update(backend_options.get("value") or {})
        overrides["options"] = merged
    return overrides


def _trace_backend_for(**stores: Any):
    """The trace backend for one request, with this browser's settings applied.

    Always a copy: the UI lets each browser choose its own settings, and the
    app serves requests on more than one thread, so the shared backend is never
    changed in place.
    """
    return _trace_backend.with_overrides(**_trace_overrides(**stores))


def load_shot_traces(
    shot_id: int,
    signals: list[str] | None = None,
    time_window: dict | None = None,
    timebase_hz: dict | None = None,
    backend_options: dict | None = None,
) -> pd.DataFrame | None:
    """Load the traces of one shot.

    The settings override the config file for this call only. They come from the
    Configuration tab. The backend is copied, not changed, so one browser's
    selection cannot affect another's.
    """
    return _trace_backend_for(
        signals=signals,
        time_window=time_window,
        timebase_hz=timebase_hz,
        backend_options=backend_options,
    ).load(shot_id)


def empty_traces_fig(message: str = "Click a point to load shot traces") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=14, color="#aaa"),
    )
    fig.update_layout(**_trace_layout())
    return fig


def make_traces_fig(shot_df: pd.DataFrame, signals: list[str] | None = None) -> go.Figure:
    requested = _effective_signals(signals)
    available = [s for s in requested if s in shot_df.columns]
    if not available:
        return empty_traces_fig("No recognisable signals in this shot file")

    n = len(available)
    fig = make_subplots(
        rows=n,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=available,
    )
    colors = px.colors.qualitative.Plotly

    for i, signal in enumerate(available):
        row = i + 1
        mask = shot_df[signal].notna()
        fig.add_trace(
            go.Scatter(
                x=shot_df.loc[mask, "time"],
                y=shot_df.loc[mask, signal],
                name=signal,
                mode="lines",
                line=dict(color=colors[i % len(colors)], width=1.5),
                showlegend=False,
            ),
            row=row,
            col=1,
        )
        fig.update_yaxes(
            title_text=signal,
            title_font=dict(size=11),
            row=row,
            col=1,
            gridcolor="#333",
            zerolinecolor="#555",
        )

    fig.update_xaxes(
        title_text="Time (s)",
        row=n,
        col=1,
        gridcolor="#333",
        zerolinecolor="#555",
    )
    fig.update_layout(**_trace_layout())
    return fig


def _trace_layout(**extra) -> dict:
    return dict(
        margin=dict(l=70, r=20, t=40, b=50),
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#16213e",
        font=dict(color="#e0e0e0", size=11),
        autosize=True,
        **extra,
    )


# ---------------------------------------------------------------------------
# SHAP plot rendering
# ---------------------------------------------------------------------------


def _render_decision_plot(
    base_value: float,
    shap_values,
    feature_names: list[str],
    title: str | None = None,
) -> str:
    """Draw one SHAP decision plot in the dashboard's colours as a base64 PNG.

    Matplotlib is used because shap draws with it; the styling below is what
    makes the result sit inside a dark Dash panel without looking pasted in.
    """
    import base64
    import io

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    with plt.style.context("dark_background"):
        plt.rcParams.update({"font.size": 7})
        shap.decision_plot(
            base_value,
            shap_values,
            feature_names=feature_names,
            show=False,
        )
        fig = plt.gcf()
        fig.set_size_inches(5, 7)
        fig.patch.set_facecolor("#1a1a2e")
        ax = fig.axes[0]
        ax.set_facecolor("#16213e")
        if title:
            ax.set_title(title)
        # Ensure all text is white and consistently small
        for artist in (
            [ax.title, ax.xaxis.label, ax.yaxis.label]
            + ax.get_xticklabels()
            + ax.get_yticklabels()
            + [t for t in ax.texts]
        ):
            artist.set_color("white")
            artist.set_fontsize(7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#555")
        ax.tick_params(colors="white", labelsize=7)
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", facecolor="#1a1a2e", dpi=110)
        plt.close("all")

    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def make_shap_fig(ds: Dataset, shot_id: int) -> str | None:
    """Return a base64-encoded PNG of the SHAP decision plot for one shot, or None.

    This is the pre-computed path: the values come from the ``--shap-data``
    file, not from a model trained in the app.
    """
    if _shap_da is None:
        return None
    idx = ds.shap_idx.get(int(shot_id))
    if idx is None:
        return None
    shap_values = _shap_da.isel(shot_id=idx).sel(**{"class": True}).values
    return _render_decision_plot(0.0, shap_values, _shap_feature_names)


def make_model_shap_fig(token: str | None, shot_id: int, class_index: int) -> str | None:
    """SHAP decision plot for one shot under the model trained in this session.

    Returns None when there is no such model, when the shot is not in the
    table it was trained on, or when ``shap`` is not installed -- all of which
    the caller reports as a message rather than an error.
    """
    entry = _model_get(token)
    if entry is None:
        return None
    row = entry["row_of"].get(int(shot_id))
    if row is None:
        return None
    classes = entry["classes"]
    safe_class = min(max(int(class_index), 0), len(classes) - 1)
    # The explainer runs in a spawned subprocess for the same reason every
    # other sklearn call does: BLAS in a forked gunicorn worker can SIGSEGV.
    out = _spawn_sklearn(
        _shap_decision_values,
        entry["model"],
        entry["X_background"],
        entry["X_all"][row],
        safe_class,
    )
    return _render_decision_plot(
        out["base_value"],
        np.asarray(out["values"]),
        entry["features"],
        title=f"Shot {shot_id} — P({classes[safe_class]})",
    )


# ---------------------------------------------------------------------------
# Clustering helpers
# ---------------------------------------------------------------------------

_CLUSTER_ALGORITHMS = [
    {"label": "K-Means", "value": "kmeans"},
    {"label": "DBSCAN", "value": "dbscan"},
    {"label": "Agglomerative", "value": "agglomerative"},
]

_SHAP_EMPTY_MSG = "Click a point to see SHAP values"
_SHAP_NO_MODEL_MSG = "Train a model in the Classification tab, or pass --shap-data"

_CLASSIFY_ALGORITHMS = [
    {"label": "Gradient Boosting", "value": "gradient_boosting"},
    {"label": "Random Forest", "value": "random_forest"},
    {"label": "Gaussian Process", "value": "gaussian_process"},
]
_CLASSIFY_GP_KERNELS = [
    {"label": "RBF", "value": "rbf"},
    {"label": "Matern", "value": "matern"},
]
# Blue-through-red, the same palette the Correlation heatmap uses, so a low
# and a high probability read the same way in both places.
_SURFACE_COLORSCALE = "RdBu_r"
_SURFACE_RESOLUTION = 120
_SURFACE_OPACITY = 0.35


def _load_cluster_representative_traces(
    representatives: dict,
    signals: list[str] | None = None,
    time_window: dict | None = None,
    timebase_hz: dict | None = None,
    backend_options: dict | None = None,
) -> dict | None:
    """Load time traces for the real representative shot of each cluster.
    Returns {str(cluster_id): {col: [values]}} suitable for dcc.Store, or None on failure.
    """
    if not representatives or not SHOW_TRACES:
        return None

    wanted = _effective_signals(signals)
    result: dict[str, dict] = {}
    for cid_str, shot_id in sorted(representatives.items(), key=lambda kv: int(kv[0])):
        try:
            sdf = load_shot_traces(int(shot_id), signals, time_window, timebase_hz, backend_options)
        except Exception:
            continue
        if sdf is None or sdf.empty:
            continue
        entry: dict[str, list] = {"time": sdf["time"].tolist()}
        for sig in wanted:
            if sig in sdf.columns and sdf[sig].notna().any():
                entry[sig] = sdf[sig].fillna(0).tolist()
        result[cid_str] = entry
    return result or None


def _render_centroid_fig(centroid_data: dict, cluster_names: dict, signals: list[str] | None = None) -> go.Figure:
    """Build a subplot figure from pre-computed centroid data (no I/O)."""
    requested = _effective_signals(signals)
    available = [s for s in requested if any(s in cdf for cdf in centroid_data.values())]
    if not available:
        return empty_traces_fig("No matching signals in centroid data")

    colors = px.colors.qualitative.Plotly
    n = len(available)
    fig = make_subplots(rows=n, cols=1, shared_xaxes=True, vertical_spacing=0.04, subplot_titles=available)
    for cid_str, cdf in sorted(centroid_data.items(), key=lambda x: int(x[0])):
        cid = int(cid_str)
        name = (cluster_names or {}).get(cid_str) or f"Cluster {cid}"
        color = colors[cid % len(colors)]
        time_arr = cdf.get("time", [])
        for i, sig in enumerate(available):
            if sig not in cdf:
                continue
            fig.add_trace(
                go.Scatter(
                    x=time_arr,
                    y=cdf[sig],
                    name=name,
                    mode="lines",
                    line=dict(color=color, width=2),
                    legendgroup=f"c{cid}",
                    showlegend=(i == 0),
                ),
                row=i + 1,
                col=1,
            )
        fig.update_yaxes(
            title_text=sig,
            title_font=dict(size=11),
            row=i + 1,
            col=1,
            gridcolor="#333",
            zerolinecolor="#555",
        )
    fig.update_xaxes(title_text="Time (s)", row=n, col=1, gridcolor="#333", zerolinecolor="#555")
    fig.update_layout(**_trace_layout(), showlegend=True, legend=dict(bgcolor="rgba(0,0,0,0)"))
    return fig


# ---------------------------------------------------------------------------
# Outlier detection helpers
# ---------------------------------------------------------------------------

_OUTLIER_ALGORITHMS = [
    {"label": "Isolation Forest", "value": "isoforest"},
    {"label": "Local Outlier Factor", "value": "lof"},
]
_OUTLIER_RED = "#ff4444"
_INLIER_BLUE = "#4488cc"


def _compute_outlier_traces_data(
    outlier_labels: dict,
    n_samples: int = 5,
    signals: list[str] | None = None,
    time_window: dict | None = None,
    timebase_hz: dict | None = None,
    backend_options: dict | None = None,
) -> dict | None:
    """Load time traces for up to n_samples outlier shots.
    Returns {str(shot_id): {col: [values]}} or None.
    """
    if not outlier_labels or not SHOW_TRACES:
        return None
    outlier_ids = [int(k) for k, v in outlier_labels.items() if int(v) == 1]
    if not outlier_ids:
        return None
    wanted = _effective_signals(signals)
    result: dict[str, dict] = {}
    for sid in outlier_ids[:n_samples]:
        try:
            sdf = load_shot_traces(sid, signals, time_window, timebase_hz, backend_options)
            if sdf is None or sdf.empty:
                continue
            entry: dict[str, list] = {"time": sdf["time"].tolist()}
            for sig in wanted:
                if sig in sdf.columns:
                    entry[sig] = sdf[sig].tolist()
            result[str(sid)] = entry
        except Exception:
            pass
    return result or None


def _load_shots_traces(
    shot_ids: list[int],
    n_samples: int = 10,
    signals: list[str] | None = None,
    time_window: dict | None = None,
    timebase_hz: dict | None = None,
    backend_options: dict | None = None,
) -> dict | None:
    """Load time traces for up to n_samples shots from a plain list of shot IDs.
    Returns {str(shot_id): {col: [values]}} or None.
    """
    if not shot_ids or not SHOW_TRACES:
        return None
    wanted = _effective_signals(signals)
    result: dict[str, dict] = {}
    for sid in shot_ids[:n_samples]:
        try:
            sdf = load_shot_traces(sid, signals, time_window, timebase_hz, backend_options)
            if sdf is None or sdf.empty:
                continue
            entry: dict[str, list] = {"time": sdf["time"].tolist()}
            for sig in wanted:
                if sig in sdf.columns:
                    entry[sig] = sdf[sig].tolist()
            result[str(sid)] = entry
        except Exception:
            pass
    return result or None


def _render_outlier_traces_fig(outlier_traces_data: dict, signals: list[str] | None = None) -> go.Figure:
    """Overlay individual outlier shot traces in a subplot figure (no I/O)."""
    requested = _effective_signals(signals)
    available = [s for s in requested if any(s in td for td in outlier_traces_data.values())]
    if not available:
        return empty_traces_fig("No matching signals in outlier trace data")

    colors = px.colors.qualitative.Plotly
    shot_ids = sorted(outlier_traces_data.keys(), key=int)
    n = len(available)
    fig = make_subplots(rows=n, cols=1, shared_xaxes=True, vertical_spacing=0.04, subplot_titles=available)
    for idx, sid_str in enumerate(shot_ids):
        td = outlier_traces_data[sid_str]
        color = colors[idx % len(colors)]
        time_arr = td.get("time", [])
        for i, sig in enumerate(available):
            if sig not in td:
                continue
            fig.add_trace(
                go.Scatter(
                    x=time_arr,
                    y=td[sig],
                    name=f"Shot {sid_str}",
                    mode="lines",
                    line=dict(color=color, width=1.5),
                    legendgroup=sid_str,
                    showlegend=(i == 0),
                ),
                row=i + 1,
                col=1,
            )
        fig.update_yaxes(
            title_text=sig,
            title_font=dict(size=11),
            row=i + 1,
            col=1,
            gridcolor="#333",
            zerolinecolor="#555",
        )
    fig.update_xaxes(title_text="Time (s)", row=n, col=1, gridcolor="#333", zerolinecolor="#555")
    fig.update_layout(**_trace_layout(), showlegend=True, legend=dict(bgcolor="rgba(0,0,0,0)"))
    return fig


# ---------------------------------------------------------------------------
# Reference-graph helpers
# ---------------------------------------------------------------------------


def _ref_shot_color(shot_id: int, min_id: int, max_id: int) -> str:
    """Map a shot_id to a Turbo colorscale colour (old=dark blue, new=dark red)."""
    import plotly.colors as pc

    t = (shot_id - min_id) / (max_id - min_id) if max_id > min_id else 0.5
    return pc.sample_colorscale("Turbo", [t])[0]


# ---------------------------------------------------------------------------
# App layout
# ---------------------------------------------------------------------------
DARK_BG = "#0f0f23"
PANEL_BG = "#1a1a2e"
BORDER = "1px solid #2a2a4a"
TEXT = "#e0e0e0"
ACCENT = "#4a9eff"

DROPDOWN_STYLE = dict(
    backgroundColor="#16213e",
    color="#000000",
    width="260px",
    fontSize="12px",
)

_BTN_STYLE = dict(
    backgroundColor=ACCENT,
    color="#000",
    border="none",
    padding="4px 12px",
    cursor="pointer",
    borderRadius="4px",
    fontSize="11px",
    fontWeight="600",
)
# Delay before a spinner appears. Long enough that a fast render never flashes
# one, short enough that a slow one is never mistaken for a broken tab.
_SPINNER_DELAY = 250

# Style fragments for showing and hiding a block. A feature that can be turned
# on while the app runs keeps its widgets in the tree and toggles these, so the
# callbacks that address those widgets always have something to write to.
_SHOW: dict[str, str] = {}
_HIDE = {"display": "none"}

_BTN_STYLE_SECONDARY = dict(_BTN_STYLE, backgroundColor="#2a2a4a", color=TEXT)

_CLUSTER_LABEL_STYLE = dict(fontSize="10px", color="#888", display="block", marginBottom="2px")
_CLUSTER_INPUT_STYLE = dict(
    backgroundColor="#16213e",
    color=TEXT,
    border=BORDER,
    padding="4px 6px",
    fontSize="11px",
    width="64px",
    borderRadius="4px",
    outline="none",
)


def _cluster_param_block(label: str, control, block_id: str | None = None) -> html.Div:
    children = [html.Label(label, style=_CLUSTER_LABEL_STYLE), control]
    if block_id:
        return html.Div(children, id=block_id)
    return html.Div(children)


def _config_summary_table() -> html.Table:
    """Build the read-only table of settings that only a restart can change.

    Only settings the tab cannot change belong here. The rows that became
    editable were removed rather than left in: a second rendering of a value
    the user has just changed is worse than no rendering at all.

    ``plugins`` is read-only on purpose, not because it is hard. It is a list
    of Python modules that the app imports, and the app listens on every
    interface with no authentication, so a module path that a browser could set
    would be a way to run arbitrary code in the server process.
    """
    rows = [
        ("config file", _args.config),
        ("shot data", SHOT_DATA_PATH),
        ("backend", BACKEND),
        ("data dir", MASTU_DATA_DIR),
        ("variable column", VARIABLE_COLUMN),
        ("projection cache", UMAP_CACHE_PATH),
        ("pre-computed projection", PROJECTION_PATH),
        ("SHAP data", SHAP_PATH),
        ("plugins", ", ".join(_cfg.plugins) if _cfg.plugins else None),
        ("host / port", f"{_args.host}:{_args.port}"),
        ("workers", 1 if _args.debug else _args.workers),
        ("debug", "on" if _args.debug else "off"),
        # Which worker answered this page. With more than one worker, the most
        # confusing thing that can happen is two tabs disagreeing, and this row
        # makes that visible instead of mysterious.
        ("process id", os.getpid()),
    ]
    return html.Table(
        style=dict(width="100%", maxWidth="720px", borderCollapse="collapse", fontSize="11px"),
        children=[
            html.Tr(
                style=dict(
                    borderBottom="1px solid #2a2a4a",
                    backgroundColor="#16213e" if i % 2 == 0 else PANEL_BG,
                ),
                children=[
                    html.Td(
                        key,
                        style=dict(
                            color=ACCENT,
                            padding="3px 8px",
                            whiteSpace="nowrap",
                            fontWeight="600",
                            width="35%",
                        ),
                    ),
                    html.Td(
                        "—" if value is None else str(value),
                        style=dict(
                            color=TEXT if value is not None else "#555",
                            padding="3px 8px",
                            wordBreak="break-all",
                        ),
                    ),
                ],
            )
            for i, (key, value) in enumerate(rows)
        ],
    )


# ---------------------------------------------------------------------------
# Configuration tab — one builder per section.
#
# Built as separate functions rather than inlined, because the layout literal
# below is already deeply nested and ten more sections in it would be unreadable.
#
# Each section carries a scope badge. Two scopes exist, and the difference
# matters: "applies now" settings are read per request, so a change redraws the
# affected pane immediately, while "rebuilds the projection" settings change the
# dataset cache key, so a change recomputes the embedding. Both are per-browser.
# ---------------------------------------------------------------------------

_CONFIG_TEXT_STYLE = dict(_CLUSTER_INPUT_STYLE, width="100%")
_CONFIG_WIDE_INPUT = dict(_CLUSTER_INPUT_STYLE, width="120px")

# Free-form backend option rows. A fixed count, like the filter grid, so the
# ids exist in the layout from the start and no callback has to create them.
MAX_BACKEND_OPTIONS = 8

# Built-in trace backends that read nothing from backend_options, so the
# key/value grid would be dead UI for them. Checked against this list rather
# than against the documented option names, because a plugin backend can read
# anything and must keep the grid.
_BACKENDS_WITHOUT_OPTIONS = frozenset({"parquet", "uda", "sal"})

# Backends that resample onto a uniform time grid, i.e. the ones that read
# BackendConfig.timebase_hz. An unknown (plugin) backend keeps the control for
# the same reason it keeps the grid.
_BACKENDS_WITH_TIMEBASE = frozenset({"uda", "sal", "fairmast"})


# Every backend the app ships. A name that is not here came from a plugin, and
# a plugin can read anything, so it keeps every control.
_BUILT_IN_BACKENDS = frozenset({"parquet", "uda", "sal", "postgres", "fairmast"})


def _backend_reads_options(backend: str) -> bool:
    return backend not in _BACKENDS_WITHOUT_OPTIONS


def _backend_reads_timebase(backend: str) -> bool:
    return backend in _BACKENDS_WITH_TIMEBASE or backend not in _BUILT_IN_BACKENDS


# How many option rows to render. Zero removes the grid: these are
# pattern-matching ids, so a callback that reads them with ALL simply gets an
# empty list, which is why they can be left out when a plain string id could
# not be.
_BACKEND_OPTION_ROWS = MAX_BACKEND_OPTIONS if _backend_reads_options(BACKEND) else 0

# Cap on the names rendered into a column dropdown. A latent-space table can
# hold hundreds of columns, and a select box that long is slow to open and no
# use to read. The same cap the Lineage variable control uses; the paste box in
# the features section is how you give a long list.
_CFG_OPTION_LIMIT = 200

# Columns offered as projection inputs. The coordinate columns are numeric and
# sit in the same table, but choosing one as a feature would feed the embedding
# back into itself, so they are never offered.
_cfg_feature_pool = [c for c in numeric_cols if not is_projection_col(c)]

_SCOPE_NOW = "applies now"
_SCOPE_PROJECTION = "rebuilds the projection"


def _scope_badge(text: str) -> html.Span:
    """A small label saying when a section's settings take effect."""
    return html.Span(
        text,
        style=dict(
            fontSize="9px",
            color="#7a7a9a",
            border="1px solid #2a2a4a",
            borderRadius="3px",
            padding="1px 5px",
            marginLeft="8px",
            textTransform="uppercase",
            letterSpacing="0.5px",
            whiteSpace="nowrap",
        ),
    )


def _config_section(
    title: str,
    description: str,
    children: list,
    scope: str | None = None,
    hidden: bool = False,
) -> html.Div:
    """Wrap one block of the Configuration tab in a titled panel.

    Set *hidden* for a panel that holds nothing the user can act on. Its
    children stay in the tree, because callbacks address them by name.
    """
    heading: list = [title]
    if scope:
        heading.append(_scope_badge(scope))
    return html.Div(
        style=dict(
            border=BORDER,
            borderRadius="6px",
            padding="12px 14px",
            marginBottom="14px",
            backgroundColor=PANEL_BG,
            **(_HIDE if hidden else _SHOW),
        ),
        children=[
            html.Div(
                heading,
                style=dict(
                    fontSize="13px",
                    fontWeight="600",
                    color=ACCENT,
                    marginBottom="2px",
                    display="flex",
                    alignItems="center",
                ),
            ),
            html.Div(description, style=dict(fontSize="11px", color="#888", marginBottom="10px")),
            *children,
        ],
    )


def _cfg_row(children: list) -> html.Div:
    """A row of labelled controls that wraps on a narrow window."""
    return html.Div(
        style=dict(display="flex", alignItems="flex-end", gap="16px", flexWrap="wrap"),
        children=children,
    )


def _cfg_signals_section() -> html.Div:
    return _config_section(
        "Signals",
        "Select the signals to show in the time-trace pane. Type a name to add one that is not in the list.",
        [
            dcc.Dropdown(
                id="cfg-signal-select",
                options=[{"label": s, "value": s} for s in TIME_TRACE_SIGNALS],
                value=list(TIME_TRACE_SIGNALS),
                multi=True,
                placeholder="Select or type signal names\u2026",
                style=dict(DROPDOWN_STYLE, width="100%"),
            ),
            html.Div(
                style=dict(display="flex", alignItems="center", gap="10px", marginTop="10px"),
                children=[
                    html.Button(
                        "Discover signals",
                        id="cfg-discover-btn",
                        n_clicks=0,
                        style=_BTN_STYLE_SECONDARY,
                    ),
                    html.Span(id="cfg-discover-status", style=dict(fontSize="11px", color="#888")),
                ],
            ),
        ],
        scope=_SCOPE_NOW,
    )


def _cfg_time_window_section() -> html.Div:
    # The two inputs use no debounce: Apply reads them as State, and a
    # blur-on-click could otherwise lose the last edit.
    return _config_section(
        "Time window",
        "Crop the time traces to this range, in seconds.",
        [
            _cfg_row(
                [
                    _cluster_param_block(
                        "min_time (s)",
                        dcc.Input(id="cfg-min-time", type="number", value=MIN_TIME, style=_CLUSTER_INPUT_STYLE),
                    ),
                    _cluster_param_block(
                        "max_time (s)",
                        dcc.Input(id="cfg-max-time", type="number", value=MAX_TIME, style=_CLUSTER_INPUT_STYLE),
                    ),
                ]
            )
        ],
        scope=_SCOPE_NOW,
    )


def _cfg_backend_section() -> html.Div:
    """The settings the active trace backend actually reads.

    A backend that reads no options gets no key/value grid, and a backend that
    does no resampling gets no timebase control. The timebase input stays in
    the tree either way, hidden, because callbacks address it by name and Dash
    fails on an output or state that is not rendered. The option rows use
    pattern-matching ids, which may legitimately match nothing, so those are
    left out rather than hidden.
    """
    show_timebase = _backend_reads_timebase(BACKEND)
    return _config_section(
        f"Trace backend options ({BACKEND})",
        "Extra settings passed to the backend that loads the time traces. "
        "The backend itself, and the data directory, need a restart to change.",
        [
            html.Div(
                id="cfg-uda-block",
                style=dict(_SHOW if show_timebase else _HIDE),
                children=[
                    _cfg_row(
                        [
                            _cluster_param_block(
                                "timebase_hz",
                                dcc.Input(
                                    id="cfg-timebase-hz-input",
                                    type="number",
                                    value=UDA_TIMEBASE_HZ,
                                    placeholder="native",
                                    style=_CONFIG_WIDE_INPUT,
                                ),
                            )
                        ]
                    ),
                    html.Div(
                        "Resample every signal onto one time grid at this rate. Clear it to use the native time axis.",
                        style=dict(fontSize="10px", color="#666", marginTop="6px"),
                    ),
                ],
            ),
            html.Div(
                id="cfg-backend-option-hint",
                style=dict(fontSize="10px", color="#666", margin="10px 0 6px"),
            ),
            *[
                html.Div(
                    style=dict(display="flex", alignItems="center", gap="6px", marginBottom="6px"),
                    children=[
                        dcc.Input(
                            id={"type": "cfg-opt-key", "index": i},
                            type="text",
                            placeholder="option\u2026",
                            value="",
                            debounce=False,
                            style=dict(_CLUSTER_INPUT_STYLE, width="150px"),
                        ),
                        dcc.Input(
                            id={"type": "cfg-opt-val", "index": i},
                            type="text",
                            placeholder="value\u2026",
                            value="",
                            debounce=False,
                            style=dict(_CLUSTER_INPUT_STYLE, width="220px"),
                        ),
                    ],
                )
                for i in range(_BACKEND_OPTION_ROWS)
            ],
        ],
        scope=_SCOPE_NOW,
        hidden=not (show_timebase or _BACKEND_OPTION_ROWS),
    )


def _cfg_live_update_section() -> html.Div:
    return _config_section(
        "Live updates",
        "Check the backend for new shots at this interval, in seconds. Clear it to stop checking.",
        [
            _cfg_row(
                [
                    _cluster_param_block(
                        "refresh_interval_seconds",
                        dcc.Input(
                            id="cfg-refresh-interval",
                            type="number",
                            min=1,
                            value=REFRESH_INTERVAL_SECONDS,
                            placeholder="off",
                            style=_CONFIG_WIDE_INPUT,
                        ),
                    )
                ]
            )
        ],
        scope=_SCOPE_NOW,
    )


def _cfg_projection_section() -> html.Div:
    return _config_section(
        "Projection",
        "How the shot statistics are reduced to a 2-D map. A change here recomputes the embedding.",
        [
            _cfg_row(
                [
                    _cluster_param_block(
                        "method",
                        dcc.RadioItems(
                            id="cfg-projection-method",
                            options=[
                                {"label": " UMAP", "value": "umap"},
                                {"label": " PCA", "value": "pca"},
                            ],
                            value=PROJECTION_METHOD,
                            inline=True,
                            style=dict(fontSize="11px"),
                        ),
                    ),
                    _cluster_param_block(
                        "n_components",
                        dcc.Input(
                            id="cfg-n-components",
                            type="number",
                            min=2,
                            max=50,
                            step=1,
                            value=PROJECTION_OPTIONS.n_components,
                            style=_CLUSTER_INPUT_STYLE,
                        ),
                    ),
                    _cluster_param_block(
                        "random_state",
                        dcc.Input(
                            id="cfg-random-state",
                            type="number",
                            step=1,
                            value=PROJECTION_OPTIONS.random_state,
                            placeholder="none",
                            style=_CLUSTER_INPUT_STYLE,
                        ),
                    ),
                ]
            ),
            html.Div(
                "The plots draw the first two components. Components 3 and up become extra columns "
                "(umap_3, umap_4 \u2026) that you can choose as plot axes. Clear random_state for an "
                "unseeded, faster UMAP that does not repeat exactly.",
                style=dict(fontSize="10px", color="#666", marginTop="8px"),
            ),
            html.Div(
                id="cfg-umap-block",
                style=dict(marginTop="10px"),
                children=[
                    _cfg_row(
                        [
                            _cluster_param_block(
                                "n_neighbors",
                                dcc.Input(
                                    id="cfg-n-neighbors",
                                    type="number",
                                    min=2,
                                    step=1,
                                    value=PROJECTION_OPTIONS.n_neighbors,
                                    style=_CLUSTER_INPUT_STYLE,
                                ),
                            ),
                            _cluster_param_block(
                                "min_dist",
                                dcc.Input(
                                    id="cfg-min-dist",
                                    type="number",
                                    min=0,
                                    max=0.99,
                                    step=0.05,
                                    value=PROJECTION_OPTIONS.min_dist,
                                    style=_CLUSTER_INPUT_STYLE,
                                ),
                            ),
                            _cluster_param_block(
                                "metric",
                                dcc.Dropdown(
                                    id="cfg-metric",
                                    options=[{"label": m, "value": m} for m in get_args(ProjectionMetric)],
                                    value=PROJECTION_OPTIONS.metric,
                                    clearable=False,
                                    style=dict(DROPDOWN_STYLE, width="160px"),
                                ),
                            ),
                        ]
                    ),
                    html.Div(
                        "These three apply to UMAP only. PCA ignores them, and they are kept so that "
                        "a change back to UMAP uses them again.",
                        style=dict(fontSize="10px", color="#666", marginTop="6px"),
                    ),
                ],
            ),
        ],
        scope=_SCOPE_PROJECTION,
    )


def _cfg_features_section() -> html.Div:
    return _config_section(
        "Projection features",
        "The columns used as input to the projection.",
        [
            html.Label("umap_features", style=_CLUSTER_LABEL_STYLE),
            dcc.Dropdown(
                id="cfg-umap-features",
                options=[{"label": c, "value": c} for c in _cfg_feature_pool[:_CFG_OPTION_LIMIT]],
                value=list(UMAP_FEATURES) if UMAP_FEATURES else [],
                multi=True,
                placeholder="Empty = every numeric column",
                style=dict(DROPDOWN_STYLE, width="100%"),
            ),
            html.Label("umap_exclude_features", style=dict(_CLUSTER_LABEL_STYLE, marginTop="10px")),
            dcc.Dropdown(
                id="cfg-umap-exclude-features",
                options=[{"label": c, "value": c} for c in _cfg_feature_pool[:_CFG_OPTION_LIMIT]],
                value=list(UMAP_EXCLUDE_FEATURES),
                multi=True,
                placeholder="Nothing excluded",
                style=dict(DROPDOWN_STYLE, width="100%"),
            ),
            html.Div(
                "Exclusions apply after the feature list, so they are the short way to keep one bad "
                "column out of a long list. A shot with no value in a listed column is still projected: "
                "the column mean is used.",
                style=dict(fontSize="10px", color="#666", marginTop="8px"),
            ),
            html.Label(
                "Paste a feature list",
                style=dict(_CLUSTER_LABEL_STYLE, marginTop="10px"),
            ),
            dcc.Textarea(
                id="cfg-features-paste",
                placeholder="One name per line, or separated by commas. Use this for a long list.",
                style=dict(
                    width="100%",
                    height="54px",
                    backgroundColor="#16213e",
                    color=TEXT,
                    border=BORDER,
                    borderRadius="4px",
                    fontSize="11px",
                    padding="4px 6px",
                ),
            ),
            html.Div(
                style=dict(display="flex", alignItems="center", gap="10px", marginTop="6px"),
                children=[
                    html.Button(
                        "Use pasted list",
                        id="cfg-features-paste-btn",
                        n_clicks=0,
                        style=_BTN_STYLE_SECONDARY,
                    ),
                    html.Span(id="cfg-features-paste-status", style=dict(fontSize="10px", color="#888")),
                ],
            ),
        ],
        scope=_SCOPE_PROJECTION,
    )


def _cfg_columns_section() -> html.Div:
    return _config_section(
        "Table columns",
        "Columns that the app reads for a particular purpose.",
        [
            html.Label("reference_shot_col", style=_CLUSTER_LABEL_STYLE),
            dcc.Dropdown(
                id="cfg-reference-shot-col-select",
                options=[{"label": c, "value": c} for c in all_cols[:_CFG_OPTION_LIMIT]],
                value=REFERENCE_SHOT_COL,
                clearable=True,
                placeholder="None \u2014 the Lineage tab stays off",
                style=dict(DROPDOWN_STYLE, width="100%"),
            ),
            html.Div(
                "The column that holds the reference (parent) shot of each shot. Set it to use the "
                "Lineage tab and the reference-graph toggle. Clear it to turn both off.",
                style=dict(fontSize="10px", color="#666", marginTop="8px"),
            ),
        ],
        scope=_SCOPE_PROJECTION,
    )


def _cfg_actions_row() -> html.Div:
    return html.Div(
        style=dict(display="flex", alignItems="center", gap="10px", marginBottom="16px", flexWrap="wrap"),
        children=[
            html.Button("Apply", id="cfg-apply-btn", n_clicks=0, style=_BTN_STYLE),
            html.Button(
                "Reset to config file",
                id="cfg-reset-btn",
                n_clicks=0,
                style=_BTN_STYLE_SECONDARY,
            ),
            html.Button(
                "Copy as YAML",
                id="cfg-yaml-btn",
                n_clicks=0,
                style=_BTN_STYLE_SECONDARY,
            ),
            html.Span(id="cfg-status", style=dict(fontSize="11px", color="#888")),
        ],
    )


def _cfg_yaml_block() -> html.Div:
    return html.Div(
        id="cfg-yaml-block",
        style=dict(_HIDE),
        children=[
            _config_section(
                "Your settings as a config file",
                "The app never writes your config file. Copy this into it, or pass it with --config, "
                "to make these settings the default at every start.",
                [
                    dcc.Textarea(
                        id="cfg-yaml-output",
                        readOnly=True,
                        style=dict(
                            width="100%",
                            height="240px",
                            backgroundColor="#16213e",
                            color=TEXT,
                            border=BORDER,
                            borderRadius="4px",
                            fontSize="11px",
                            fontFamily="monospace",
                            padding="6px",
                        ),
                    )
                ],
            )
        ],
    )


# The interval a parked dcc.Interval keeps. It never fires while disabled, so
# the value only has to be large enough not to matter.
_REFRESH_INTERVAL_OFF_MS = 3_600_000


def _cfg_column_options(pool: list[str], selected, search_value: str | None) -> list[dict[str, str]]:
    """Options for a column dropdown, always including what is already selected.

    Dash clears a value that has no matching option, and the rendered list is
    capped, so a name that came from a paste or from the config file has to be
    added back or it would disappear on the next render.
    """
    chosen = [c for c in (selected or []) if c]
    names = list(dict.fromkeys([*chosen, *pool]))
    if search_value:
        matches = [c for c in names if search_value.lower() in c.lower()]
        names = list(dict.fromkeys([*chosen, *matches]))
    return [{"label": c, "value": c} for c in names[:_CFG_OPTION_LIMIT]]


def _settings_as_yaml(signals, time_window, timebase, options, projection, reference) -> str:
    """The applied settings as a config file, plus the equivalent CLI flags.

    Anything the user has not applied is left out, so the result describes the
    settings rather than freezing today's defaults into a file.
    """
    body: dict[str, Any] = {}
    if signals:
        body["signals"] = list(signals)
    if time_window:
        body["time_window"] = {
            "min_time": time_window.get("min_time"),
            "max_time": time_window.get("max_time"),
        }
    if timebase is not None and timebase.get("value") is not None:
        body["uda"] = {"timebase_hz": timebase["value"]}
    if options is not None and options.get("value"):
        body["backend_options"] = dict(options["value"])
    if projection:
        body["projection_method"] = projection.get("projection_method")
        if projection.get("umap_features"):
            body["umap_features"] = list(projection["umap_features"])
        if projection.get("umap_exclude_features"):
            body["umap_exclude_features"] = list(projection["umap_exclude_features"])
        if projection.get("projection_options"):
            body["projection_options"] = dict(projection["projection_options"])
    if reference is not None:
        body["reference_shot_col"] = reference.get("value")

    if not body:
        return "# Nothing applied yet in this browser. Change a setting, then select Apply."

    header = (
        "# NiceShot settings from the Configuration tab.\n"
        "# Copy these into your config file, or save them and pass --config <path>,\n"
        "# to get them at every start. The app never writes your config file.\n"
        f"# Shot data: {SHOT_DATA_PATH}\n"
        f"# Backend:   {BACKEND}\n"
    )
    return header + "\n" + yaml.safe_dump(body, sort_keys=False, default_flow_style=False)


def _cfg_tab_children() -> list:
    """The whole Configuration tab body."""
    return [
        html.Div(
            style=dict(padding="12px 4px", overflow="auto"),
            children=[
                html.Div(
                    "These settings apply to this browser only. The server keeps no per-user "
                    "configuration, so another browser is unaffected, and a restart returns to the "
                    "config file.",
                    style=dict(fontSize="11px", color="#888", marginBottom="14px"),
                ),
                _cfg_signals_section(),
                _cfg_time_window_section(),
                _cfg_backend_section(),
                _cfg_live_update_section(),
                _cfg_projection_section(),
                _cfg_features_section(),
                _cfg_columns_section(),
                _cfg_actions_row(),
                _cfg_yaml_block(),
                _config_section(
                    "Active configuration",
                    "These settings come from the config file and the command line. Restart the app to change them.",
                    [_config_summary_table()],
                ),
            ],
        )
    ]


# ---------------------------------------------------------------------------
# Lineage tab — pure render helpers
#
# These live outside the `if SHOW_REF_TOGGLE:` guard on purpose: the callbacks
# below are only registered when a reference column exists, but these functions
# must stay importable (and unit-testable) either way.
# ---------------------------------------------------------------------------

# Diverging ramp for a signed, normalised change. Darker and less saturated
# than Plotly's RdBu_r, which is built for white backgrounds and washes out
# 11px text on the #16213e cell fill. Same hue family as the data table's
# selection (#2a3a6e) and latest-shot (#1e4a33) highlights.
_LIN_UP = ["#4a2b38", "#6b2f3e", "#8f3548", "#b63e54"]  # weak -> strong increase
_LIN_DOWN = ["#1b2f52", "#1c3f74", "#1d5199", "#2266bd"]  # weak -> strong decrease
_LIN_BINS = (0.15, 0.35, 0.60, 0.85)  # |normalised| break points
_LIN_UP_TEXT = "#ff8fa3"
_LIN_DOWN_TEXT = "#7fb8ff"

# Full-saturation point for each metric. The first two are fixed so a colour
# means the same thing in every lineage: a 2-sigma move, or a doubling. The
# absolute metric has no natural scale, so it is normalised per lineage and
# the UI says so.
_LIN_METRIC_FULL_SCALE = {"zscore": 2.0, "percent": 100.0}


def _lin_message(text: str) -> html.Div:
    """The tab's empty-state text, styled like the Shot Info panel's."""
    return html.Div(text, style=dict(fontSize="11px", color="#555", padding="8px 4px"))


def _lin_colour_scale(matrix) -> float:
    """Divisor that maps a metric value onto the [-1, 1] colour ramp.

    Fixed for the z-scored and percent metrics, so intensity is comparable
    between shots and between lineages. The absolute metric is normalised
    against the largest change in this lineage, because its units are whatever
    the column happens to use.
    """
    fixed = _LIN_METRIC_FULL_SCALE.get(matrix.metric)
    if fixed:
        return float(fixed)
    if matrix.magnitude.empty:
        return 1.0
    biggest = matrix.magnitude.to_numpy(dtype=float)
    biggest = np.nanmax(biggest) if np.isfinite(biggest).any() else 0.0
    return float(biggest) or 1.0


def _lin_cell_color(normalised) -> str | None:
    """Ramp colour for a signed, normalised change, or ``None`` to leave it be.

    Returning ``None`` below the first bin matters twice: an unchanged cell then
    inherits the ordinary zebra styling, and the conditional-style list stays
    short enough to send.
    """
    if normalised is None:
        return None
    try:
        value = float(normalised)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value) or abs(value) < _LIN_BINS[0]:
        return None
    ramp = _LIN_UP if value > 0 else _LIN_DOWN
    index = min(int(np.searchsorted(_LIN_BINS, abs(value), side="right")) - 1, len(ramp) - 1)
    return ramp[index]


def _lin_relation_labels(shot_ids: list[int], subject: int) -> list[str]:
    """Position of each lineage shot relative to the subject."""
    try:
        origin = shot_ids.index(subject)
    except ValueError:
        origin = 0
    labels = []
    for i in range(len(shot_ids)):
        step = i - origin
        labels.append("subject" if step == 0 else f"{-step:+d}")
    return labels


def _lin_format(value) -> str:
    """Render one cell for display, shortening free prose."""
    if value is None:
        return ""
    if isinstance(value, float):
        if np.isnan(value):
            return ""
        return f"{value:.4g}"
    text = str(value).strip()
    if text.lower() in ("nan", "none", "nat"):
        return ""
    return text if len(text) <= 40 else text[:39] + "…"


def _lin_table_columns(columns: list[str], kinds: dict[str, str]) -> list[dict]:
    """Column definitions for the history table.

    Floats format as ``.4g``, matching ``_table_column_defs`` so a value reads
    the same here as in the Data Table.
    """
    defs: list[dict] = [
        {"name": "shot", "id": "shot_id"},
        {"name": "rel", "id": "_rel"},
    ]
    for col in columns:
        if kinds.get(col) == "numeric":
            defs.append({"name": col, "id": col, "type": "numeric", "format": {"specifier": ".4g"}})
        else:
            defs.append({"name": col, "id": col})
    return defs


def _lin_table_data(matrix, columns: list[str], subject: int) -> list[dict]:
    """One record per lineage shot, newest first."""
    labels = _lin_relation_labels(matrix.shot_ids, subject)
    records = []
    for label, shot_id in zip(labels, matrix.shot_ids):
        row: dict = {"shot_id": shot_id, "_rel": label}
        for col in columns:
            raw = matrix.values.at[shot_id, col]
            if matrix.kinds.get(col) == "numeric":
                numeric = pd.to_numeric(pd.Series([raw]), errors="coerce").iloc[0]
                row[col] = None if pd.isna(numeric) else float(numeric)
            else:
                row[col] = _lin_format(raw)
        records.append(row)
    return records


def _lin_tooltip_data(matrix, columns: list[str]) -> list[dict]:
    """Per-cell hover text carrying the change the colour stands for.

    The cell itself always shows the raw value; the delta, percentage and
    metric live here so no cell has to hold three numbers.
    """
    tooltips = []
    for shot_id in matrix.shot_ids:
        row: dict = {}
        for col in columns:
            note = str(matrix.note.at[shot_id, col] or "")
            if matrix.kinds.get(col) != "numeric":
                row[col] = note or "unchanged"
                continue
            signed = matrix.delta.at[shot_id, col]
            metric_value = matrix.metric_value.at[shot_id, col]
            parts = []
            if pd.notna(signed):
                parts.append(f"Δ = {signed:+.4g}")
            if pd.notna(metric_value):
                unit = "%" if matrix.metric == "percent" else ""
                label = {"zscore": "z", "percent": "pct", "absolute": "abs"}[matrix.metric]
                parts.append(f"{label} = {metric_value:+.3g}{unit}")
            if note:
                parts.append(note)
            row[col] = "  ".join(parts) or "unchanged"
        tooltips.append(row)
    return tooltips


def _lin_style_data_conditional(matrix, columns: list[str], subject: int) -> list[dict]:
    """Per-cell colouring for the history table.

    ``filter_query`` targets whole rows, which is what ``highlight_table_row``
    needs but not what this table needs — a single cell requires
    ``{"if": {"column_id": ..., "row_index": ...}}``. Dash applies entries in
    order and later ones win, so the broad rules are emitted first.
    """
    styles: list[dict] = [
        {"if": {"row_index": "odd"}, "backgroundColor": PANEL_BG},
        {
            "if": {"column_id": "shot_id"},
            "backgroundColor": PANEL_BG,
            "color": ACCENT,
            "fontWeight": "600",
        },
        {"if": {"column_id": "_rel"}, "backgroundColor": PANEL_BG, "color": "#888"},
    ]
    scale = _lin_colour_scale(matrix)
    budget = _LIN_MAX_STYLE_CELLS
    for row_index, shot_id in enumerate(matrix.shot_ids):
        if shot_id == subject:
            styles.append({"if": {"row_index": row_index}, "borderTop": f"2px solid {ACCENT}"})
        if shot_id in matrix.excluded:
            # Hidden by the active filters. Greyed rather than dropped, so the
            # deltas either side of it stay real.
            styles.append({"if": {"row_index": row_index}, "color": "#555", "fontStyle": "italic"})
            continue
        for col in columns:
            if budget <= 0:
                return styles
            if matrix.kinds.get(col) != "numeric":
                if bool(matrix.changed.at[shot_id, col]):
                    styles.append(
                        {
                            "if": {"column_id": col, "row_index": row_index},
                            "backgroundColor": "#3a3358",
                            "color": "#ffffff",
                        }
                    )
                    budget -= 1
                continue
            metric_value = matrix.metric_value.at[shot_id, col]
            color = _lin_cell_color(metric_value / scale if pd.notna(metric_value) else None)
            if color is None:
                continue
            styles.append(
                {
                    "if": {"column_id": col, "row_index": row_index},
                    "backgroundColor": color,
                    "color": "#ffffff",
                }
            )
            budget -= 1
    return styles


# Header label for the card's change column. The unit belongs in the header,
# not on every row: the card lists every variable, so a repeated suffix is 500
# lines of noise.
# The z-scored ranking has nowhere else to put its measure, so the change
# column carries it. Every other ranking shows the change in the column's own
# units there and puts its measure in the cell alongside the bar.
_LIN_CARD_CHANGE_LABEL = {"zscore": "change (σ)", "percent": "change", "absolute": "change"}


def _lin_signed_metric(item) -> float:
    """The direction of the item's change: its measure, or its raw delta.

    Used for the colour and the bar only. A column whose measure is undefined
    (zero variance under z-scores, a zero baseline under percent) still has a
    direction in its own units, and that is enough to tint a row.
    """
    return item.metric_value if np.isfinite(item.metric_value) else item.delta


def _lin_change_text(item, metric: str) -> tuple[str, str]:
    """Text and colour for one ranked row's change column.

    Reports the measure the rows are *sorted* by, so the number on screen is
    the number that decided the order. ``item.changed`` is what separates an
    unchanged categorical column from a changed one — text carries no delta to
    read a zero out of — and a column with no usable value says why rather than
    showing a blank.
    """
    if not item.changed:
        return "unchanged", "#666"
    if item.kind != "numeric":
        return "changed", "#b9a7ff"
    # Under the z-scored ranking this column *is* the measure, so it shows that
    # and nothing else: the header names the unit, and a number in any other
    # unit would be a lie rather than a fallback. Under every other ranking the
    # measure has its own cell, so this one shows the change in the column's
    # own units. Either way a value that does not exist reports why.
    signed = item.metric_value if metric == "zscore" else item.delta
    if not np.isfinite(signed):
        return (item.note or "—"), "#888"
    color = _LIN_UP_TEXT if signed > 0 else _LIN_DOWN_TEXT
    return f"{signed:+.2f}" if metric == "zscore" else f"{signed:+.4g}", color


# Full-bar point for the percentage shown beside an absolute ranking: a
# doubling, the same reading the percent colour metric uses.
_LIN_PCT_FULL_SCALE = 100.0


def _lin_percent_text(item) -> str:
    """The item's change as a percentage of the reference value, if it has one."""
    if not item.changed or item.pct is None or not np.isfinite(item.pct):
        return ""
    return f"{item.pct:+.1f}%"


def _lin_magnitude_cell(item, scale: float, metric: str) -> html.Td:
    """The card's right-hand cell: a bar, labelled with a percentage where one fits.

    The change column beside it holds a number in the column's own units, and
    those units say nothing about whether a change is large — 40 kA and 40 kW
    share a scale and nothing else. The percentage does say it, and needs no
    scale to be read against, so it both labels this cell and sets the bar's
    length.

    Under the z-scored ranking the change column is already comparable between
    columns, so the bar keeps its fixed 2-sigma scale and stays unlabelled.
    """
    if metric != "zscore":
        percent = _lin_percent_text(item)
        share = 0.0 if not percent else item.pct / _LIN_PCT_FULL_SCALE
    else:
        percent = ""
        share = 0.0 if not np.isfinite(item.magnitude) else item.magnitude / scale
        if _lin_signed_metric(item) < 0:
            share = -share
    fraction = min(1.0, abs(share))
    bar = html.Div(
        style=dict(backgroundColor="#2a2a4a", borderRadius="2px", width="100%", height="6px"),
        children=html.Div(
            style=dict(
                backgroundColor=_lin_cell_color(share) or "#2a2a4a",
                width=f"{fraction * 100:.0f}%",
                height="6px",
                borderRadius="2px",
            )
        ),
    )
    if metric == "zscore":
        return html.Td(bar, style=dict(padding="3px 8px", width="110px"))
    return html.Td(
        html.Div(
            style=dict(display="flex", alignItems="center", gap="8px"),
            children=[
                html.Span(
                    percent or "—",
                    style=dict(
                        color=(_LIN_UP_TEXT if (percent and item.pct > 0) else _LIN_DOWN_TEXT) if percent else "#888",
                        fontSize="10px",
                        minWidth="54px",
                        textAlign="right",
                        whiteSpace="nowrap",
                    ),
                ),
                html.Div(bar, style=dict(flex="1")),
            ],
        ),
        style=dict(padding="3px 8px", width="170px"),
    )


def _lin_card_header(metric: str) -> html.Thead:
    """Sticky column labels for the summary card.

    Sticky because the card scrolls: 500 rows down, "which number is this"
    needs an answer that is still on screen.
    """
    labels = (
        "variable",
        "reference → subject",
        _LIN_CARD_CHANGE_LABEL.get(metric, "change"),
        "" if metric == "zscore" else "% change",
    )
    return html.Thead(
        html.Tr(
            [
                html.Th(
                    label,
                    style=dict(
                        position="sticky",
                        top="0",
                        zIndex=2,
                        backgroundColor=PANEL_BG,
                        color="#888",
                        fontSize="10px",
                        fontWeight="600",
                        textAlign="right" if index == 2 else "left",
                        padding="3px 8px",
                        borderBottom=f"1px solid {ACCENT}",
                    ),
                )
                for index, label in enumerate(labels)
            ]
        )
    )


def _lin_change_row(item, index: int, scale: float, metric: str = "zscore") -> html.Tr:
    """One ranked-change row: name, old -> new, the change, and the bar."""
    change_text, change_color = _lin_change_text(item, metric)
    return html.Tr(
        style=dict(
            borderBottom="1px solid #2a2a4a",
            backgroundColor="#16213e" if index % 2 == 0 else PANEL_BG,
        ),
        children=[
            html.Td(
                item.column,
                title=item.column,
                style=dict(
                    color=ACCENT,
                    fontWeight="600",
                    padding="3px 8px",
                    whiteSpace="nowrap",
                    maxWidth="220px",
                    overflow="hidden",
                    textOverflow="ellipsis",
                ),
            ),
            html.Td(
                f"{_lin_format(item.old)} → {_lin_format(item.new)}",
                style=dict(color=TEXT, padding="3px 8px", whiteSpace="nowrap"),
            ),
            html.Td(
                change_text,
                style=dict(color=change_color, padding="3px 8px", whiteSpace="nowrap", textAlign="right"),
            ),
            _lin_magnitude_cell(item, scale, metric),
        ],
    )


def _lin_render_change_cards(matrix, items, subject: int, n_compared: int) -> html.Div:
    """The summary card that sits above the history table.

    Lists every variable it is given, biggest change first, rather than a top
    ten: the whole ranking is the summary, and the table scrolls inside the
    card so the heading and the tally stay put while it does.
    """
    reference = matrix.shot_ids[1] if len(matrix.shot_ids) > 1 else None
    n_changed = int(matrix.changed.loc[subject].sum()) if subject in matrix.changed.index else 0
    scale = _lin_colour_scale(matrix)

    heading = f"Shot {subject}"
    if reference is not None:
        heading += f"  ·  reference {reference}"
    heading += f"  ·  {len(matrix.shot_ids)} shots in lineage"

    if not items:
        body = _lin_message(f"No differences found across {n_compared} variable(s).")
    else:
        body = html.Div(
            style=dict(maxHeight=_LIN_CARD_H, overflowY="auto"),
            children=html.Table(
                style=dict(width="100%", borderCollapse="collapse", fontSize="11px"),
                children=[
                    _lin_card_header(matrix.metric),
                    html.Tbody([_lin_change_row(item, i, scale, matrix.metric) for i, item in enumerate(items)]),
                ],
            ),
        )

    footer_bits = [f"{n_changed} of {n_compared} variable(s) changed"]
    ordering = "largest change first"
    if len(items) < n_compared:
        ordering = f"{len(items)} shown, {ordering}"
    footer_bits.append(ordering)
    footer_bits.append(_LIN_METRIC_HINTS.get(matrix.metric, ""))

    return html.Div(
        children=[
            html.Div(
                heading,
                style=dict(fontSize="12px", fontWeight="600", color=TEXT, marginBottom="6px"),
            ),
            body,
            html.Div(
                "  ·  ".join(b for b in footer_bits if b),
                style=dict(fontSize="10px", color="#888", marginTop="6px"),
            ),
        ]
    )


def _lin_note_columns(compare_cols, numeric_cols) -> list[str]:
    """Every text column in the table, the prose and scenario fields first.

    A column the tab cannot compare as a number holds text, and text is what
    answers "what were they trying" — so all of it is shown. A fixed list of
    acceptable names cannot do that: the names differ between machines, and the
    one field an operator actually fills in would be hidden with no way to ask
    for it. The recognised prose and scenario names still come first, because
    they carry the intent while a status flag does not.
    """
    numeric = set(numeric_cols or ())
    named = ("scenario", "shot_type", "programme", "campaign", "session", "gas", "pellet")
    text = [c for c in compare_cols if c not in numeric]
    preferred = [c for c in text if _is_free_text(c) or any(hint in c.lower() for hint in named)]
    return preferred + [c for c in text if c not in set(preferred)]


def _lin_notes_table(df: pd.DataFrame, shot_ids: list[int], subject: int, columns: list[str]):
    """One block per lineage shot showing its notes, subject first.

    A block per shot rather than a wide table: these fields hold sentences, and
    sentences do not fit in a 80px table cell.
    """
    if not columns:
        return _lin_message("This table has no text columns to show.")
    indexed = df.drop_duplicates("shot_id").set_index(df.drop_duplicates("shot_id")["shot_id"].astype(int).values)
    blocks = []
    for label, shot_id in zip(_lin_relation_labels(shot_ids, subject), shot_ids):
        if shot_id not in indexed.index:
            continue
        row = indexed.loc[shot_id]
        entries = []
        for col in columns:
            text = _lin_format_note(row.get(col))
            if text:
                entries.append(
                    html.Div(
                        style=dict(display="flex", gap="8px", marginBottom="2px"),
                        children=[
                            html.Span(
                                col,
                                style=dict(
                                    color=ACCENT,
                                    fontSize="10px",
                                    fontWeight="600",
                                    minWidth="150px",
                                    flexShrink="0",
                                ),
                            ),
                            html.Span(text, title=str(row.get(col)), style=dict(color=TEXT, fontSize="11px")),
                        ],
                    )
                )
        blocks.append(
            html.Div(
                style=dict(
                    border=f"1px solid {ACCENT}" if shot_id == subject else BORDER,
                    borderRadius="6px",
                    padding="8px 10px",
                    marginBottom="8px",
                    backgroundColor=PANEL_BG,
                ),
                children=[
                    html.Div(
                        f"shot {shot_id}  ({label})",
                        style=dict(fontSize="11px", fontWeight="600", color=TEXT, marginBottom="6px"),
                    ),
                    *(entries or [_lin_message("No notes recorded for this shot.")]),
                ],
            )
        )
    return html.Div(blocks) if blocks else _lin_message("No notes recorded for this lineage.")


# Some shot-log fields hold HTML (MAST-U's "programme" column is a list of
# links). Dash escapes it, so it is shown verbatim rather than rendered —
# stripping the tags is what makes the text readable.
_LIN_TAG_RE = re.compile(r"<[^>]+>")
_LIN_SPACE_RE = re.compile(r"\s+")


def _lin_format_note(value) -> str:
    """Trim one free-text field for display, keeping the full text in a tooltip."""
    if value is None:
        return ""
    if isinstance(value, float) and not np.isfinite(value):
        return ""
    text = _LIN_SPACE_RE.sub(" ", _LIN_TAG_RE.sub(" ", str(value))).strip()
    if text.lower() in ("nan", "none", "nat", ""):
        return ""
    return text if len(text) <= _LIN_NOTE_CHARS else text[: _LIN_NOTE_CHARS - 1] + "…"


_LIN_SUBTAB_STYLE = dict(color=TEXT, backgroundColor=PANEL_BG, fontSize="12px", padding="4px 10px")
_LIN_SUBTAB_SELECTED = dict(
    color=ACCENT,
    backgroundColor=DARK_BG,
    borderTop=f"2px solid {ACCENT}",
    fontSize="12px",
    padding="4px 10px",
)
_LIN_VIEW_H = "calc(100vh - 430px)"


def _lin_subtab(label: str, value: str, children: list, controls: list | None = None) -> dcc.Tab:
    """One sub-view, with a spinner over its output while the callback runs.

    Every sub-view reads the whole lineage, and the History table builds one
    styled cell per shot per variable — slow enough on a wide table that
    without this the tab looks broken rather than busy. The overlay keeps the
    previous content on screen, dimmed, so it stays clear which view is
    loading.

    *controls* stay outside the overlay: a dimmed selector that cannot be
    clicked while the figure it drives redraws is worse than no feedback.
    """
    body = dcc.Loading(
        children=children,
        type="circle",
        color=ACCENT,
        delay_show=_SPINNER_DELAY,
        overlay_style=dict(visibility="visible", opacity=0.35),
    )
    return dcc.Tab(
        label=label,
        value=value,
        style=_LIN_SUBTAB_STYLE,
        selected_style=_LIN_SUBTAB_SELECTED,
        children=(controls or []) + [body],
    )


def _lin_control_block(label: str, control) -> html.Div:
    return html.Div([html.Label(label, style=_CLUSTER_LABEL_STYLE), control])


def _lineage_tab_children() -> list:
    """Body of the Lineage tab.

    A factory rather than an inline literal: the layout expression in this
    module is already thousands of lines, and this keeps the tab reviewable.
    Callable whether or not a reference column is configured, so tests can
    build it from the ordinary app fixture.
    """
    return [
        html.Div(
            style=dict(display="flex", flexDirection="column", padding="8px 4px 12px"),
            children=[
                # -- Control bar --
                html.Div(
                    style=dict(display="flex", alignItems="flex-end", gap="16px", flexWrap="wrap"),
                    children=[
                        _lin_control_block(
                            "Lineage",
                            dcc.RadioItems(
                                id="lin-scope",
                                options=_LIN_SCOPE_OPTIONS,
                                value="chain",
                                inline=True,
                                labelStyle=dict(marginRight="10px", fontSize="11px", color=TEXT),
                                inputStyle=dict(marginRight="4px"),
                            ),
                        ),
                        _lin_control_block(
                            "Colour by",
                            dcc.Dropdown(
                                id="lin-metric",
                                options=_LIN_METRIC_OPTIONS,
                                value="zscore",
                                clearable=False,
                                style=dict(DROPDOWN_STYLE, width="190px"),
                            ),
                        ),
                        html.Div(
                            style=dict(flex="1", minWidth="260px"),
                            children=[
                                html.Label("Variables", style=_CLUSTER_LABEL_STYLE),
                                dcc.Dropdown(
                                    id="lin-columns-dd",
                                    options=[],
                                    value=[],
                                    multi=True,
                                    placeholder="Type to search columns…",
                                    style=dict(DROPDOWN_STYLE, width="100%"),
                                ),
                            ],
                        ),
                        html.Div(
                            style=dict(display="flex", alignItems="center", gap="6px", flexWrap="wrap"),
                            children=[
                                html.Button(
                                    "Top changed",
                                    id="lin-cols-changed-btn",
                                    n_clicks=0,
                                    style=_BTN_STYLE_SECONDARY,
                                ),
                                html.Button(
                                    "Projection features",
                                    id="lin-cols-features-btn",
                                    n_clicks=0,
                                    style=_BTN_STYLE_SECONDARY,
                                ),
                                html.Button(
                                    "Clear",
                                    id="lin-cols-clear-btn",
                                    n_clicks=0,
                                    style=_BTN_STYLE_SECONDARY,
                                ),
                                html.Span(id="lin-columns-count", style=dict(fontSize="10px", color="#888")),
                            ],
                        ),
                        dcc.Checklist(
                            id="lin-respect-filters",
                            options=[{"label": " Mark filtered shots", "value": "respect"}],
                            value=[],
                            labelStyle=dict(fontSize="11px", color=TEXT),
                        ),
                    ],
                ),
                html.Span(
                    id="lin-subject-display",
                    style=dict(fontSize="11px", color="#888", margin="8px 0 4px"),
                ),
                # -- Summary card: always visible, so it stays the key for
                #    whichever sub-view is open. It lists every variable, so
                #    the scroll lives on its table rather than on this box —
                #    the heading and the tally must not scroll away. --
                html.Div(
                    style=dict(display="flex", alignItems="center", gap="8px", margin="0 0 4px"),
                    children=[
                        html.Label("Rank by", style=dict(_CLUSTER_LABEL_STYLE, display="inline", marginBottom="0")),
                        dcc.RadioItems(
                            id="lin-card-metric",
                            options=_LIN_CARD_METRIC_OPTIONS,
                            value="zscore",
                            inline=True,
                            labelStyle=dict(marginRight="10px", fontSize="11px", color=TEXT),
                            inputStyle=dict(marginRight="4px"),
                        ),
                    ],
                ),
                html.Div(
                    id="lin-change-cards",
                    style=dict(
                        border=BORDER,
                        borderRadius="6px",
                        padding="10px 12px",
                        marginBottom="10px",
                        backgroundColor=PANEL_BG,
                    ),
                ),
                # -- Sub-views. dcc.Tabs renders only the selected child, so
                #    a sub-view's figure is never built until it is asked for. --
                dcc.Tabs(
                    id="lin-subtabs",
                    value="lin-history",
                    colors=dict(border=BORDER, primary=ACCENT, background=PANEL_BG),
                    children=[
                        _lin_subtab(
                            "History",
                            "lin-history",
                            [
                                html.Div(id="lin-history-msg", style=dict(fontSize="10px", color="#888")),
                                dash_table.DataTable(
                                    id="lin-history-table",
                                    columns=[],
                                    data=[],
                                    # Not virtualized: the two clientside repaint
                                    # workarounds the Data Table needs are
                                    # virtualization bugs, and they would bite
                                    # harder here because lineage data arrives
                                    # while the table is already on screen. A
                                    # lineage is at most ~100 rows anyway.
                                    virtualization=False,
                                    # Not sortable: row_index in
                                    # style_data_conditional indexes `data` as
                                    # supplied and is not re-derived after a
                                    # native sort, so sorting would detach every
                                    # colour from its value. The newest-first
                                    # order is also itself the information.
                                    sort_action="none",
                                    page_action="none",
                                    fixed_rows={"headers": True},
                                    fixed_columns={"headers": True, "data": 2},
                                    tooltip_data=[],
                                    tooltip_duration=None,
                                    style_table={
                                        "maxHeight": _LIN_VIEW_H,
                                        "overflowY": "auto",
                                        "overflowX": "auto",
                                        "minWidth": "100%",
                                    },
                                    style_cell=dict(
                                        backgroundColor="#16213e",
                                        color=TEXT,
                                        fontSize="11px",
                                        padding="3px 10px",
                                        border="1px solid #2a2a4a",
                                        minWidth="90px",
                                        maxWidth="200px",
                                        whiteSpace="nowrap",
                                        overflow="hidden",
                                        textOverflow="ellipsis",
                                    ),
                                    style_header=dict(
                                        backgroundColor=PANEL_BG,
                                        color=ACCENT,
                                        fontWeight="600",
                                        fontSize="11px",
                                        border="1px solid #2a2a4a",
                                    ),
                                    style_data_conditional=[],
                                ),
                            ],
                        ),
                        _lin_subtab(
                            "Notes",
                            "lin-notes",
                            [
                                html.Div(
                                    id="lin-notes-panel",
                                    style=dict(maxHeight=_LIN_VIEW_H, overflowY="auto", padding="8px 2px"),
                                )
                            ],
                        ),
                        _lin_subtab(
                            "Tree",
                            "lin-tree",
                            [
                                dcc.Graph(
                                    id="lin-tree-plot",
                                    config=dict(displayModeBar=True, displaylogo=False),
                                    style=dict(height=_LIN_VIEW_H, minHeight="240px"),
                                )
                            ],
                        ),
                        _lin_subtab(
                            "Sparklines",
                            "lin-spark",
                            [
                                html.Div(
                                    style=dict(maxHeight=_LIN_VIEW_H, overflowY="auto"),
                                    children=dcc.Graph(
                                        id="lin-spark-plot",
                                        config=dict(displayModeBar=False),
                                    ),
                                ),
                            ],
                            controls=[
                                html.Div(
                                    style=dict(
                                        display="flex",
                                        alignItems="flex-end",
                                        gap="12px",
                                        padding="8px 2px",
                                        flexWrap="wrap",
                                    ),
                                    children=[
                                        html.Div(
                                            style=dict(flex="1", minWidth="260px"),
                                            children=[
                                                html.Label("Visualise", style=_CLUSTER_LABEL_STYLE),
                                                dcc.Dropdown(
                                                    id="lin-spark-columns",
                                                    options=[],
                                                    value=[],
                                                    multi=True,
                                                    placeholder="Type to search variables…",
                                                    style=dict(DROPDOWN_STYLE, width="100%"),
                                                ),
                                            ],
                                        ),
                                        html.Button(
                                            f"Top {_LIN_SPARK_PANELS}",
                                            id="lin-spark-top-btn",
                                            n_clicks=0,
                                            style=_BTN_STYLE_SECONDARY,
                                        ),
                                        html.Div(
                                            id="lin-spark-msg",
                                            style=dict(fontSize="10px", color="#888", paddingBottom="4px"),
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        )
    ]


# ---------------------------------------------------------------------------
# Classification tab
#
# Built by a function rather than written inline, because it carries far more
# controls than its sibling tabs and the layout literal is deep enough
# already. It sits in the lower left-hand tab strip beside Clustering and
# Outlier Detection: the same shape of question, asked with a target column.
# ---------------------------------------------------------------------------

_CLASSIFY_TAB_STYLE = dict(color=TEXT, backgroundColor=PANEL_BG, fontSize="12px", padding="4px 10px")
_CLASSIFY_TAB_SELECTED_STYLE = dict(
    color=ACCENT,
    backgroundColor=DARK_BG,
    borderTop=f"2px solid {ACCENT}",
    fontSize="12px",
    padding="4px 10px",
)
_CLASSIFY_ROW_STYLE = dict(display="flex", gap="8px", marginBottom="6px", flexWrap="wrap", alignItems="flex-end")


def _classify_dropdown(component_id: str, options, value, width: str, clearable: bool = False) -> dcc.Dropdown:
    return dcc.Dropdown(
        id=component_id,
        options=options,
        value=value,
        clearable=clearable,
        style=dict(backgroundColor="#16213e", color="#000", width=width, fontSize="11px"),
    )


def _classify_tab() -> dcc.Tab:
    """The Classification control panel."""
    default_features = [c for c in (UMAP_FEATURES or numeric_cols) if c in numeric_cols][:8]
    return dcc.Tab(
        label="Classification",
        value="classification",
        style=_CLASSIFY_TAB_STYLE,
        selected_style=_CLASSIFY_TAB_SELECTED_STYLE,
        children=[
            # The spinner targets classify-status, which the training callback
            # writes directly. dcc.Loading only follows Outputs that land on
            # one of its own children, so targeting the status span -- rather
            # than a store further down the chain -- is what keeps it visible
            # for the whole fit.
            dcc.Loading(
                type="circle",
                color=ACCENT,
                delay_show=_SPINNER_DELAY,
                target_components={"classify-status": "children"},  # type: ignore
                overlay_style=dict(visibility="visible", opacity=0.35),
                children=html.Div(
                    style=dict(padding="8px 4px", overflowY="auto", maxHeight="150px"),
                    children=[
                        html.Div(
                            style=_CLASSIFY_ROW_STYLE,
                            children=[
                                _cluster_param_block(
                                    "Model",
                                    _classify_dropdown(
                                        "classify-algorithm",
                                        _CLASSIFY_ALGORITHMS,
                                        "gradient_boosting",
                                        "150px",
                                    ),
                                ),
                                _cluster_param_block(
                                    "Target",
                                    _classify_dropdown(
                                        "classify-target",
                                        [{"label": c, "value": c} for c in _classify_target_cols],
                                        _classify_target_cols[0] if _classify_target_cols else None,
                                        "160px",
                                        clearable=True,
                                    ),
                                ),
                                _cluster_param_block(
                                    "test_fraction",
                                    dcc.Input(
                                        id="classify-test-fraction",
                                        type="number",
                                        value=0.25,
                                        min=0.0,
                                        max=0.9,
                                        step=0.05,
                                        style=_CLUSTER_INPUT_STYLE,
                                    ),
                                ),
                                _cluster_param_block(
                                    "seed",
                                    dcc.Input(
                                        id="classify-seed",
                                        type="number",
                                        value=42,
                                        step=1,
                                        style=_CLUSTER_INPUT_STYLE,
                                    ),
                                ),
                            ],
                        ),
                        # Per-model hyperparameters. Every block stays in the
                        # tree and is shown or hidden by toggle_classify_params,
                        # because a callback cannot reference an id the layout
                        # does not contain.
                        html.Div(
                            style=_CLASSIFY_ROW_STYLE,
                            children=[
                                _cluster_param_block(
                                    "n_estimators",
                                    dcc.Input(
                                        id="classify-n-estimators",
                                        type="number",
                                        value=100,
                                        min=1,
                                        max=2000,
                                        step=10,
                                        style=_CLUSTER_INPUT_STYLE,
                                    ),
                                    block_id="classify-n-estimators-block",
                                ),
                                _cluster_param_block(
                                    "max_depth",
                                    dcc.Input(
                                        id="classify-max-depth",
                                        type="number",
                                        value=3,
                                        min=1,
                                        max=50,
                                        step=1,
                                        style=_CLUSTER_INPUT_STYLE,
                                    ),
                                    block_id="classify-max-depth-block",
                                ),
                                _cluster_param_block(
                                    "learning_rate",
                                    dcc.Input(
                                        id="classify-learning-rate",
                                        type="number",
                                        value=0.1,
                                        min=0.001,
                                        max=1.0,
                                        step=0.01,
                                        style=_CLUSTER_INPUT_STYLE,
                                    ),
                                    block_id="classify-learning-rate-block",
                                ),
                                _cluster_param_block(
                                    "subsample",
                                    dcc.Input(
                                        id="classify-subsample",
                                        type="number",
                                        value=1.0,
                                        min=0.1,
                                        max=1.0,
                                        step=0.05,
                                        style=_CLUSTER_INPUT_STYLE,
                                    ),
                                    block_id="classify-subsample-block",
                                ),
                                _cluster_param_block(
                                    "min_samples_leaf",
                                    dcc.Input(
                                        id="classify-min-samples-leaf",
                                        type="number",
                                        value=1,
                                        min=1,
                                        step=1,
                                        style=_CLUSTER_INPUT_STYLE,
                                    ),
                                    block_id="classify-min-samples-leaf-block",
                                ),
                                html.Div(
                                    id="classify-balanced-block",
                                    children=[
                                        dcc.Checklist(
                                            id="classify-balanced",
                                            options=[{"label": " Balance classes", "value": "balanced"}],
                                            value=[],
                                            inputStyle=dict(marginRight="4px"),
                                            labelStyle=dict(fontSize="11px", color=TEXT, cursor="pointer"),
                                        ),
                                    ],
                                ),
                                _cluster_param_block(
                                    "kernel",
                                    _classify_dropdown(
                                        "classify-gp-kernel",
                                        _CLASSIFY_GP_KERNELS,
                                        "rbf",
                                        "100px",
                                    ),
                                    block_id="classify-gp-kernel-block",
                                ),
                                _cluster_param_block(
                                    "length_scale",
                                    dcc.Input(
                                        id="classify-gp-length-scale",
                                        type="number",
                                        value=1.0,
                                        min=0.01,
                                        step=0.1,
                                        style=_CLUSTER_INPUT_STYLE,
                                    ),
                                    block_id="classify-gp-length-scale-block",
                                ),
                                _cluster_param_block(
                                    "restarts",
                                    dcc.Input(
                                        id="classify-gp-restarts",
                                        type="number",
                                        value=0,
                                        min=0,
                                        max=10,
                                        step=1,
                                        style=_CLUSTER_INPUT_STYLE,
                                    ),
                                    block_id="classify-gp-restarts-block",
                                ),
                                _cluster_param_block(
                                    "max_train_rows",
                                    dcc.Input(
                                        id="classify-max-train-rows",
                                        type="number",
                                        value=2000,
                                        min=20,
                                        step=100,
                                        style=_CLUSTER_INPUT_STYLE,
                                    ),
                                    block_id="classify-max-train-rows-block",
                                ),
                            ],
                        ),
                        html.Div(
                            style=dict(marginBottom="6px"),
                            children=[
                                html.Label("Features", style=_CLUSTER_LABEL_STYLE),
                                dcc.Dropdown(
                                    id="classify-features",
                                    options=[{"label": c, "value": c} for c in numeric_cols],
                                    value=default_features,
                                    multi=True,
                                    placeholder="Select feature columns...",
                                    style=dict(backgroundColor="#16213e", color="#000", fontSize="11px"),
                                ),
                            ],
                        ),
                        html.Div(
                            style=dict(display="flex", alignItems="center", gap="8px", marginBottom="6px"),
                            children=[
                                html.Button("Train model", id="run-classify-btn", n_clicks=0, style=_BTN_STYLE),
                                html.Span(id="classify-status", style=dict(fontSize="11px", color="#888")),
                            ],
                        ),
                        html.Div(
                            style=dict(display="flex", alignItems="center", gap="8px", flexWrap="wrap"),
                            children=[
                                dcc.Checklist(
                                    id="classify-show-surface",
                                    options=[{"label": " Decision surface", "value": "surface"}],
                                    value=["surface"],
                                    inputStyle=dict(marginRight="4px"),
                                    labelStyle=dict(fontSize="11px", color=TEXT, cursor="pointer"),
                                ),
                                _cluster_param_block(
                                    "Surface class",
                                    _classify_dropdown("classify-surface-class", [], None, "140px"),
                                ),
                            ],
                        ),
                    ],
                ),
            ),
        ],
    )


# Scatter Graph height — fills viewport minus header + tab bar + controls + padding
_SCATTER_H = "calc(100vh - 183px)"

MAX_FILTERS = 6
OPERATORS = [">=", "<=", ">", "<", "==", "!=", "contains"]

app = dash.Dash(__name__, title="NiceShot!")
server = app.server  # WSGI callable for gunicorn
app.index_string = """<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        <link rel="icon" type="image/svg+xml" href="/assets/favicon.svg">
        {%css%}
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>"""

app.layout = html.Div(
    style=dict(
        backgroundColor=DARK_BG,
        color=TEXT,
        fontFamily="'Segoe UI', Arial, sans-serif",
        height="100vh",
        overflow="hidden",
        display="flex",
        flexDirection="column",
    ),
    children=[
        dcc.Store(id="active-filters"),
        dcc.Store(id="selected-shot"),
        dcc.Store(id="_table_scroll_sink"),
        dcc.Store(id="_table_repaint_sink"),
        # Subject shot for the Lineage tab: the selected shot, or the latest
        # shot when nothing is selected. Resolved centrally so every Lineage
        # view agrees, and declared unconditionally so the callback that
        # writes it can fire before the tab has ever been opened — a callback
        # whose Output is not in the rendered tree is never dispatched.
        dcc.Store(id="lin-subject-shot", data=None),
        # What the Sparklines view last seeded into its own variable selector.
        # Comparing against it is what separates "the app filled this in" from
        # "someone chose this", so a new subject can refresh the default
        # without discarding a hand-picked set. Declared here rather than in
        # the tab because dcc.Tabs unmounts the tab it is not showing, and an
        # unmounted Store forgets.
        dcc.Store(id="lin-spark-seeded", data=None),
        dcc.Store(id="cluster-labels", data=None),
        dcc.Store(id="cluster-representatives", data=None),
        dcc.Store(id="cluster-names", data={}),
        dcc.Store(id="centroid-data", data=None),
        dcc.Store(id="outlier-labels", data=None),
        dcc.Store(id="outlier-traces-data", data=None),
        # Classification results. "class-labels" is the predicted class of
        # every shot, "class-proba" the full probability matrix the decision
        # surface is drawn from, and "class-model" only an opaque token into
        # the server-side model cache -- a fitted estimator is not JSON.
        dcc.Store(id="class-labels", data=None),
        dcc.Store(id="class-proba", data=None),
        dcc.Store(id="class-model", data=None),
        dcc.Store(id="search-results", data=None),
        dcc.Store(id="search-traces-data", data=None),
        dcc.Store(id="search-highlight-enabled", data=True),
        # Selected variable in long-format mode; always None in flat mode, where
        # get_dataset(None) returns the single dataset.
        dcc.Store(id="selected-variable", data=None),
        # Everything a cached Dataset is built from, as one value. Data
        # callbacks depend on this rather than on the individual settings, so a
        # change fans out once, atomically, instead of firing each callback
        # several times with a half-updated combination of settings.
        dcc.Store(id="dataset-key", data=None),
        dcc.Download(id="table-download"),
        # Live-update: poll the backend for new shots at REFRESH_INTERVAL_SECONDS.
        # Disabled (no-op) when unset — see nice_shot/config_schema.py.
        dcc.Interval(
            id="refresh-interval",
            interval=int(REFRESH_INTERVAL_SECONDS * 1000) if REFRESH_INTERVAL_SECONDS else _REFRESH_INTERVAL_OFF_MS,
            disabled=REFRESH_INTERVAL_SECONDS is None,
        ),
        # Bumped by poll_for_updates() after a successful refresh_dataset() call;
        # render callbacks depend on it so they pick up new data without the user
        # touching a filter.
        dcc.Store(id="dataset-version", data=0),
        # Max shot_id currently loaded — recomputed whenever the dataset changes.
        dcc.Store(id="latest-shot", data=None),
        dcc.Store(id="latest-shot-highlight-enabled", data=True),
        # Live overrides from the Configuration tab. These are seeded from the
        # config file, so a page that never opens that tab behaves as before.
        #
        # storage_type="session" (every other store here uses the default,
        # "memory") keeps a selection across a tab refresh. The value stays in
        # the browser: the server holds no per-user config, so two browsers can
        # show different signals at the same time. See TraceBackend.with_overrides.
        dcc.Store(id="cfg-signals", data=list(TIME_TRACE_SIGNALS), storage_type="session"),
        dcc.Store(
            id="cfg-time-window",
            data={"min_time": MIN_TIME, "max_time": MAX_TIME},
            storage_type="session",
        ),
        # Projection settings from the Configuration tab. A dict once Apply has
        # run, and None before that.
        #
        # None means "nothing applied in this browser", not "empty": a cleared
        # setting is still a dict, with the field set to null. Without that
        # distinction, clearing the reference column would be read as "fall
        # back to the config file" and the column would come straight back.
        dcc.Store(id="cfg-projection", data=None, storage_type="session"),
        dcc.Store(id="cfg-reference-shot-col", data=None, storage_type="session"),
        # UDA timebase and the free-form backend options, same convention.
        dcc.Store(id="cfg-timebase-hz", data=None, storage_type="session"),
        dcc.Store(id="cfg-backend-options", data=None, storage_type="session"),
        # Signals found on the backend by the "Discover signals" button.
        dcc.Store(id="cfg-discovered-signals", data=None),
        # Header
        html.Div(
            style=dict(
                padding="12px 24px",
                display="flex",
                justifyContent="space-between",
                alignItems="center",
                borderBottom=BORDER,
                backgroundColor=PANEL_BG,
            ),
            children=[
                html.Span(
                    "NiceShot!",
                    style=dict(fontSize="20px", fontWeight="600", color=ACCENT),
                ),
                *(
                    [
                        html.Div(
                            style=dict(
                                display="flex",
                                alignItems="center",
                                gap="8px",
                                flex="1",
                                justifyContent="center",
                            ),
                            children=[
                                html.Label(
                                    "Variable:",
                                    style=dict(fontSize="13px", color=TEXT),
                                ),
                                dcc.Dropdown(
                                    id="variable-select",
                                    options=[{"label": v, "value": v} for v in VARIABLES],
                                    value=None,
                                    clearable=False,
                                    placeholder="Select a variable…",
                                    style=dict(DROPDOWN_STYLE, width="260px"),
                                ),
                            ],
                        )
                    ]
                    if VARIABLE_MODE
                    else []
                ),
                html.Span(id="filter-count-display", style=dict(fontSize="13px", color="#888")),
            ],
        ),
        # Main content
        html.Div(
            style=dict(display="flex", flex="1", overflow="hidden"),
            children=[
                # -- Left pane --
                html.Div(
                    style=dict(
                        flex="1",
                        minWidth="0",
                        padding="16px",
                        borderRight=BORDER,
                        backgroundColor=PANEL_BG,
                        display="flex",
                        flexDirection="column",
                        gap="8px",
                        overflow="hidden",
                    ),
                    children=[
                        html.H3(
                            id="traces-title",
                            children="Time Traces",
                            style=dict(
                                margin="0 0 4px 0",
                                fontSize="14px",
                                color=ACCENT,
                            ),
                        ),
                        html.Div(
                            style=dict(fontSize="11px", color="#666", lineHeight="1.6"),
                            children=[
                                html.Span(
                                    f"backend: {BACKEND}",
                                    style=dict(marginRight="16px"),
                                ),
                                html.Span(
                                    id="shot-count-display",
                                    style=dict(marginRight="16px"),
                                ),
                                html.Span(
                                    id="active-time-display",
                                    children=f"time: {MIN_TIME}–{MAX_TIME} s",
                                    style=dict(marginRight="16px"),
                                ),
                                html.Span(
                                    id="active-signals-display",
                                    children=f"signals: {', '.join(TIME_TRACE_SIGNALS)}",
                                ),
                            ],
                        ),
                        html.Div(
                            style=dict(display="flex", gap="8px", flexWrap="wrap"),
                            children=[
                                html.Button(
                                    "Similar shots: ON",
                                    id="search-highlight-btn",
                                    n_clicks=0,
                                    style=dict(
                                        backgroundColor="#1a3a6a",
                                        color=ACCENT,
                                        border=f"1px solid {ACCENT}",
                                        padding="4px 12px",
                                        cursor="pointer",
                                        borderRadius="4px",
                                        fontSize="11px",
                                        fontWeight="600",
                                    ),
                                ),
                                html.Button(
                                    "Latest shot: ON",
                                    id="latest-shot-highlight-btn",
                                    n_clicks=0,
                                    style=dict(
                                        backgroundColor="#1a3a6a",
                                        color=ACCENT,
                                        border=f"1px solid {ACCENT}",
                                        padding="4px 12px",
                                        cursor="pointer",
                                        borderRadius="4px",
                                        fontSize="11px",
                                        fontWeight="600",
                                    ),
                                ),
                            ],
                        ),
                        dcc.Tabs(
                            id="left-upper-tabs",
                            value="traces",
                            style=dict(flex="1", minHeight="0"),
                            colors=dict(
                                border=BORDER,
                                primary=ACCENT,
                                background=PANEL_BG,
                            ),
                            children=[
                                dcc.Tab(
                                    label="Time Traces",
                                    value="traces",
                                    disabled=not SHOW_TRACES,
                                    style=dict(
                                        color=TEXT,
                                        backgroundColor=PANEL_BG,
                                        fontSize="12px",
                                        padding="4px 10px",
                                    ),
                                    selected_style=dict(
                                        color=ACCENT,
                                        backgroundColor=DARK_BG,
                                        borderTop=f"2px solid {ACCENT}",
                                        fontSize="12px",
                                        padding="4px 10px",
                                    ),
                                    disabled_style=dict(
                                        color="#444",
                                        backgroundColor=PANEL_BG,
                                        fontSize="12px",
                                        padding="4px 10px",
                                        cursor="not-allowed",
                                    ),
                                    children=[
                                        dcc.Loading(
                                            type="circle",
                                            color=ACCENT,
                                            target_components={"traces-plot": "figure"},  # type: ignore
                                            children=dcc.Graph(
                                                id="traces-plot",
                                                figure=empty_traces_fig(),
                                                responsive=True,
                                                config=dict(
                                                    displayModeBar=True,
                                                    displaylogo=False,
                                                    modeBarButtonsToRemove=[
                                                        "select2d",
                                                        "lasso2d",
                                                    ],
                                                ),
                                                style=dict(
                                                    height="calc(100vh - 430px)",
                                                    minHeight="220px",
                                                ),
                                            ),
                                        )
                                        if SHOW_TRACES
                                        else html.Div(
                                            style=dict(
                                                height="calc(100vh - 430px)",
                                                minHeight="220px",
                                                display="flex",
                                                alignItems="center",
                                                justifyContent="center",
                                            ),
                                            children=html.Span(
                                                "No data directory — pass --data-dir to enable time traces",
                                                style=dict(fontSize="12px", color="#444"),
                                            ),
                                        ),
                                    ],
                                ),
                                # Always present, never omitted: the tab now
                                # serves a model trained in the Classification
                                # tab as well as a --shap-data file, and either
                                # can appear after the layout is built.
                                dcc.Tab(
                                    label="SHAP",
                                    value="shap",
                                    style=dict(
                                        color=TEXT,
                                        backgroundColor=PANEL_BG,
                                        fontSize="12px",
                                        padding="4px 10px",
                                    ),
                                    selected_style=dict(
                                        color=ACCENT,
                                        backgroundColor=DARK_BG,
                                        borderTop=f"2px solid {ACCENT}",
                                        fontSize="12px",
                                        padding="4px 10px",
                                    ),
                                    children=[
                                        dcc.Loading(
                                            type="circle",
                                            color=ACCENT,
                                            delay_show=_SPINNER_DELAY,
                                            target_components={"shap-container": "children"},  # type: ignore
                                            overlay_style=dict(visibility="visible", opacity=0.35),
                                            children=html.Div(
                                                id="shap-container",
                                                style=dict(
                                                    height="calc(100vh - 430px)",
                                                    minHeight="220px",
                                                    overflowY="auto",
                                                    padding="4px",
                                                ),
                                                children=[
                                                    html.Span(
                                                        _SHAP_EMPTY_MSG,
                                                        style=dict(
                                                            fontSize="11px",
                                                            color="#555",
                                                        ),
                                                    )
                                                ],
                                            ),
                                        ),
                                    ],
                                ),
                                dcc.Tab(
                                    label="Cluster Traces",
                                    value="cluster-traces",
                                    style=dict(
                                        color=TEXT,
                                        backgroundColor=PANEL_BG,
                                        fontSize="12px",
                                        padding="4px 10px",
                                    ),
                                    selected_style=dict(
                                        color=ACCENT,
                                        backgroundColor=DARK_BG,
                                        borderTop=f"2px solid {ACCENT}",
                                        fontSize="12px",
                                        padding="4px 10px",
                                    ),
                                    children=[
                                        html.Div(
                                            style=dict(
                                                display="flex",
                                                alignItems="center",
                                                gap="8px",
                                                padding="6px 4px 6px",
                                            ),
                                            children=[
                                                html.Button(
                                                    "Compute centroid traces",
                                                    id="compute-centroid-btn",
                                                    n_clicks=0,
                                                    style=dict(
                                                        backgroundColor="#2a2a4a",
                                                        color=TEXT,
                                                        border=BORDER,
                                                        padding="4px 12px",
                                                        cursor="pointer",
                                                        borderRadius="4px",
                                                        fontSize="11px",
                                                    ),
                                                ),
                                                html.Span(
                                                    id="centroid-status",
                                                    style=dict(fontSize="11px", color="#888"),
                                                ),
                                            ],
                                        ),
                                        dcc.Loading(
                                            type="circle",
                                            color=ACCENT,
                                            children=dcc.Graph(
                                                id="cluster-traces-plot",
                                                figure=empty_traces_fig(
                                                    "Run clustering, then click 'Compute centroid traces'"
                                                ),
                                                responsive=True,
                                                config=dict(displayModeBar=True, displaylogo=False),
                                                style=dict(
                                                    height="calc(100vh - 465px)",
                                                    minHeight="200px",
                                                ),
                                            ),
                                        ),
                                    ],
                                ),
                                dcc.Tab(
                                    label="Outlier Traces",
                                    value="outlier-traces",
                                    style=dict(
                                        color=TEXT,
                                        backgroundColor=PANEL_BG,
                                        fontSize="12px",
                                        padding="4px 10px",
                                    ),
                                    selected_style=dict(
                                        color=ACCENT,
                                        backgroundColor=DARK_BG,
                                        borderTop=f"2px solid {ACCENT}",
                                        fontSize="12px",
                                        padding="4px 10px",
                                    ),
                                    children=[
                                        html.Div(
                                            style=dict(
                                                display="flex",
                                                alignItems="center",
                                                padding="6px 4px 6px",
                                            ),
                                            children=[
                                                html.Span(
                                                    id="outlier-traces-status",
                                                    style=dict(fontSize="11px", color="#888"),
                                                ),
                                            ],
                                        ),
                                        dcc.Loading(
                                            type="circle",
                                            color=ACCENT,
                                            children=dcc.Graph(
                                                id="outlier-traces-plot",
                                                figure=empty_traces_fig("Run outlier detection to load sample traces"),
                                                responsive=True,
                                                config=dict(displayModeBar=True, displaylogo=False),
                                                style=dict(
                                                    height="calc(100vh - 465px)",
                                                    minHeight="200px",
                                                ),
                                            ),
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        html.Div(
                            style=dict(flexShrink="0", overflow="hidden"),
                            children=[
                                dcc.Tabs(
                                    value="shot-info",
                                    style=dict(
                                        marginTop="8px",
                                        borderTop=BORDER,
                                        paddingTop="4px",
                                    ),
                                    colors=dict(
                                        border=BORDER,
                                        primary=ACCENT,
                                        background=PANEL_BG,
                                    ),
                                    children=[
                                        dcc.Tab(
                                            label="Shot Info",
                                            value="shot-info",
                                            style=dict(
                                                color=TEXT,
                                                backgroundColor=PANEL_BG,
                                                fontSize="12px",
                                                padding="4px 10px",
                                            ),
                                            selected_style=dict(
                                                color=ACCENT,
                                                backgroundColor=DARK_BG,
                                                borderTop=f"2px solid {ACCENT}",
                                                fontSize="12px",
                                                padding="4px 10px",
                                            ),
                                            children=[
                                                html.Div(
                                                    id="shot-info-panel",
                                                    style=dict(
                                                        overflowY="auto",
                                                        maxHeight="150px",
                                                    ),
                                                ),
                                            ],
                                        ),
                                        dcc.Tab(
                                            label="Clustering",
                                            value="clustering",
                                            style=dict(
                                                color=TEXT,
                                                backgroundColor=PANEL_BG,
                                                fontSize="12px",
                                                padding="4px 10px",
                                            ),
                                            selected_style=dict(
                                                color=ACCENT,
                                                backgroundColor=DARK_BG,
                                                borderTop=f"2px solid {ACCENT}",
                                                fontSize="12px",
                                                padding="4px 10px",
                                            ),
                                            children=[
                                                html.Div(
                                                    style=dict(
                                                        padding="8px 4px",
                                                        overflowY="auto",
                                                        maxHeight="150px",
                                                    ),
                                                    children=[
                                                        # Row 1: algorithm + params
                                                        html.Div(
                                                            style=dict(
                                                                display="flex",
                                                                gap="8px",
                                                                marginBottom="6px",
                                                                flexWrap="wrap",
                                                                alignItems="flex-end",
                                                            ),
                                                            children=[
                                                                _cluster_param_block(
                                                                    "Algorithm",
                                                                    dcc.Dropdown(
                                                                        id="cluster-algorithm",
                                                                        options=_CLUSTER_ALGORITHMS,
                                                                        value="kmeans",
                                                                        clearable=False,
                                                                        style=dict(
                                                                            backgroundColor="#16213e",
                                                                            color="#000",
                                                                            width="120px",
                                                                            fontSize="11px",
                                                                        ),
                                                                    ),
                                                                ),
                                                                _cluster_param_block(
                                                                    "n_clusters",
                                                                    dcc.Input(
                                                                        id="cluster-n",
                                                                        type="number",
                                                                        value=5,
                                                                        min=2,
                                                                        max=50,
                                                                        step=1,
                                                                        style=_CLUSTER_INPUT_STYLE,
                                                                    ),
                                                                    block_id="cluster-n-block",
                                                                ),
                                                                _cluster_param_block(
                                                                    "eps",
                                                                    dcc.Input(
                                                                        id="cluster-eps",
                                                                        type="number",
                                                                        value=0.5,
                                                                        min=0,
                                                                        style=_CLUSTER_INPUT_STYLE,
                                                                    ),
                                                                    block_id="cluster-eps-block",
                                                                ),
                                                                _cluster_param_block(
                                                                    "min_samples",
                                                                    dcc.Input(
                                                                        id="cluster-min-samples",
                                                                        type="number",
                                                                        value=5,
                                                                        min=1,
                                                                        step=1,
                                                                        style=_CLUSTER_INPUT_STYLE,
                                                                    ),
                                                                    block_id="cluster-min-samples-block",
                                                                ),
                                                            ],
                                                        ),
                                                        # Row 2: projection toggle + feature selection
                                                        html.Div(
                                                            style=dict(marginBottom="4px"),
                                                            children=[
                                                                dcc.Checklist(
                                                                    id="cluster-use-projection",
                                                                    options=[
                                                                        {
                                                                            "label": " Use projection coordinates",
                                                                            "value": "projection",
                                                                        }
                                                                    ],
                                                                    value=[],
                                                                    inputStyle=dict(marginRight="4px"),
                                                                    labelStyle=dict(
                                                                        fontSize="11px", color=TEXT, cursor="pointer"
                                                                    ),
                                                                ),
                                                            ],
                                                        ),
                                                        html.Div(
                                                            id="cluster-features-row",
                                                            style=dict(marginBottom="6px"),
                                                            children=[
                                                                html.Label(
                                                                    "Features",
                                                                    style=_CLUSTER_LABEL_STYLE,
                                                                ),
                                                                dcc.Dropdown(
                                                                    id="cluster-features",
                                                                    options=[
                                                                        {"label": c, "value": c} for c in numeric_cols
                                                                    ],
                                                                    value=(UMAP_FEATURES or numeric_cols)[:8],
                                                                    multi=True,
                                                                    placeholder="Select feature columns...",
                                                                    style=dict(
                                                                        backgroundColor="#16213e",
                                                                        color="#000",
                                                                        fontSize="11px",
                                                                    ),
                                                                ),
                                                            ],
                                                        ),
                                                        # Row 3: run button + status
                                                        html.Div(
                                                            style=dict(
                                                                display="flex",
                                                                alignItems="center",
                                                                gap="8px",
                                                                marginBottom="6px",
                                                            ),
                                                            children=[
                                                                html.Button(
                                                                    "Run clustering",
                                                                    id="run-cluster-btn",
                                                                    n_clicks=0,
                                                                    style=_BTN_STYLE,
                                                                ),
                                                                html.Span(
                                                                    id="cluster-status",
                                                                    style=dict(fontSize="11px", color="#888"),
                                                                ),
                                                            ],
                                                        ),
                                                        # Cluster name inputs (rendered dynamically)
                                                        html.Div(id="cluster-name-inputs"),
                                                    ],
                                                ),
                                            ],
                                        ),
                                        dcc.Tab(
                                            label="Outlier Detection",
                                            value="outliers",
                                            style=dict(
                                                color=TEXT,
                                                backgroundColor=PANEL_BG,
                                                fontSize="12px",
                                                padding="4px 10px",
                                            ),
                                            selected_style=dict(
                                                color=ACCENT,
                                                backgroundColor=DARK_BG,
                                                borderTop=f"2px solid {ACCENT}",
                                                fontSize="12px",
                                                padding="4px 10px",
                                            ),
                                            children=[
                                                html.Div(
                                                    style=dict(
                                                        padding="8px 4px",
                                                        overflowY="auto",
                                                        maxHeight="150px",
                                                    ),
                                                    children=[
                                                        html.Div(
                                                            style=dict(
                                                                display="flex",
                                                                gap="8px",
                                                                marginBottom="6px",
                                                                flexWrap="wrap",
                                                                alignItems="flex-end",
                                                            ),
                                                            children=[
                                                                _cluster_param_block(
                                                                    "Algorithm",
                                                                    dcc.Dropdown(
                                                                        id="outlier-algorithm",
                                                                        options=_OUTLIER_ALGORITHMS,
                                                                        value="isoforest",
                                                                        clearable=False,
                                                                        style=dict(
                                                                            backgroundColor="#16213e",
                                                                            color="#000",
                                                                            width="140px",
                                                                            fontSize="11px",
                                                                        ),
                                                                    ),
                                                                ),
                                                                _cluster_param_block(
                                                                    "contamination",
                                                                    dcc.Input(
                                                                        id="outlier-contamination",
                                                                        type="number",
                                                                        value=0.1,
                                                                        min=0.01,
                                                                        max=0.5,
                                                                        step=0.01,
                                                                        style=_CLUSTER_INPUT_STYLE,
                                                                    ),
                                                                ),
                                                                _cluster_param_block(
                                                                    "n_neighbors",
                                                                    dcc.Input(
                                                                        id="outlier-n-neighbors",
                                                                        type="number",
                                                                        value=20,
                                                                        min=2,
                                                                        step=1,
                                                                        style=_CLUSTER_INPUT_STYLE,
                                                                    ),
                                                                    block_id="outlier-n-neighbors-block",
                                                                ),
                                                            ],
                                                        ),
                                                        html.Div(
                                                            style=dict(marginBottom="4px"),
                                                            children=[
                                                                dcc.Checklist(
                                                                    id="outlier-use-projection",
                                                                    options=[
                                                                        {
                                                                            "label": " Use projection coordinates",
                                                                            "value": "projection",
                                                                        }
                                                                    ],
                                                                    value=[],
                                                                    inputStyle=dict(marginRight="4px"),
                                                                    labelStyle=dict(
                                                                        fontSize="11px", color=TEXT, cursor="pointer"
                                                                    ),
                                                                ),
                                                            ],
                                                        ),
                                                        html.Div(
                                                            id="outlier-features-row",
                                                            style=dict(marginBottom="6px"),
                                                            children=[
                                                                html.Label(
                                                                    "Features",
                                                                    style=_CLUSTER_LABEL_STYLE,
                                                                ),
                                                                dcc.Dropdown(
                                                                    id="outlier-features",
                                                                    options=[
                                                                        {"label": c, "value": c} for c in numeric_cols
                                                                    ],
                                                                    value=(UMAP_FEATURES or numeric_cols)[:8],
                                                                    multi=True,
                                                                    placeholder="Select feature columns...",
                                                                    style=dict(
                                                                        backgroundColor="#16213e",
                                                                        color="#000",
                                                                        fontSize="11px",
                                                                    ),
                                                                ),
                                                            ],
                                                        ),
                                                        html.Div(
                                                            style=dict(
                                                                display="flex",
                                                                alignItems="center",
                                                                gap="8px",
                                                            ),
                                                            children=[
                                                                html.Button(
                                                                    "Run outlier detection",
                                                                    id="run-outlier-btn",
                                                                    n_clicks=0,
                                                                    style=dict(
                                                                        backgroundColor=_OUTLIER_RED,
                                                                        color="#fff",
                                                                        border="none",
                                                                        padding="4px 12px",
                                                                        cursor="pointer",
                                                                        borderRadius="4px",
                                                                        fontSize="11px",
                                                                        fontWeight="600",
                                                                    ),
                                                                ),
                                                                html.Span(
                                                                    id="outlier-status",
                                                                    style=dict(
                                                                        fontSize="11px",
                                                                        color="#888",
                                                                    ),
                                                                ),
                                                            ],
                                                        ),
                                                    ],
                                                ),
                                            ],
                                        ),
                                        _classify_tab(),
                                        dcc.Tab(
                                            label="Filters",
                                            value="filters",
                                            style=dict(
                                                color=TEXT,
                                                backgroundColor=PANEL_BG,
                                                fontSize="12px",
                                                padding="4px 10px",
                                            ),
                                            selected_style=dict(
                                                color=ACCENT,
                                                backgroundColor=DARK_BG,
                                                borderTop=f"2px solid {ACCENT}",
                                                fontSize="12px",
                                                padding="4px 10px",
                                            ),
                                            children=[
                                                html.Div(
                                                    style=dict(
                                                        padding="8px 4px",
                                                        overflowY="auto",
                                                        maxHeight="150px",
                                                    ),
                                                    children=[
                                                        # Controls row
                                                        html.Div(
                                                            style=dict(
                                                                display="flex",
                                                                alignItems="center",
                                                                gap="16px",
                                                                marginBottom="10px",
                                                            ),
                                                            children=[
                                                                html.Div(
                                                                    [
                                                                        html.Label(
                                                                            "Combine with:",
                                                                            style=dict(
                                                                                fontSize="11px",
                                                                                marginRight="6px",
                                                                            ),
                                                                        ),
                                                                        dcc.RadioItems(
                                                                            id="filter-logic",
                                                                            options=[
                                                                                {
                                                                                    "label": "AND",
                                                                                    "value": "AND",
                                                                                },
                                                                                {
                                                                                    "label": "OR",
                                                                                    "value": "OR",
                                                                                },
                                                                            ],
                                                                            value="AND",
                                                                            inline=True,
                                                                            labelStyle=dict(
                                                                                marginRight="10px",
                                                                                fontSize="11px",
                                                                                cursor="pointer",
                                                                                color=TEXT,
                                                                            ),
                                                                        ),
                                                                    ],
                                                                    style=dict(
                                                                        display="flex",
                                                                        alignItems="center",
                                                                    ),
                                                                ),
                                                                html.Button(
                                                                    "Clear all",
                                                                    id="filter-clear-all",
                                                                    style=dict(
                                                                        backgroundColor="#2a2a4a",
                                                                        color=TEXT,
                                                                        border=BORDER,
                                                                        padding="3px 8px",
                                                                        cursor="pointer",
                                                                        borderRadius="4px",
                                                                        fontSize="11px",
                                                                    ),
                                                                ),
                                                            ],
                                                        ),
                                                        # Filter rows
                                                        *[
                                                            html.Div(
                                                                style=dict(
                                                                    display="flex",
                                                                    alignItems="center",
                                                                    gap="6px",
                                                                    marginBottom="6px",
                                                                ),
                                                                children=[
                                                                    dcc.Dropdown(
                                                                        id={
                                                                            "type": "filter-col",
                                                                            "index": i,
                                                                        },
                                                                        options=[
                                                                            {
                                                                                "label": c,
                                                                                "value": c,
                                                                            }
                                                                            for c in all_cols
                                                                        ],
                                                                        value=None,
                                                                        clearable=True,
                                                                        placeholder="Column...",
                                                                        style=dict(
                                                                            backgroundColor="#16213e",
                                                                            color="#000000",
                                                                            width="160px",
                                                                            fontSize="11px",
                                                                        ),
                                                                    ),
                                                                    dcc.Dropdown(
                                                                        id={
                                                                            "type": "filter-op",
                                                                            "index": i,
                                                                        },
                                                                        options=[
                                                                            {
                                                                                "label": op,
                                                                                "value": op,
                                                                            }
                                                                            for op in OPERATORS
                                                                        ],
                                                                        value=">=",
                                                                        clearable=False,
                                                                        style=dict(
                                                                            backgroundColor="#16213e",
                                                                            color="#000000",
                                                                            width="70px",
                                                                            fontSize="11px",
                                                                        ),
                                                                    ),
                                                                    dcc.Input(
                                                                        id={
                                                                            "type": "filter-val",
                                                                            "index": i,
                                                                        },
                                                                        type="text",
                                                                        placeholder="Value...",
                                                                        value="",
                                                                        debounce=False,
                                                                        style=dict(
                                                                            backgroundColor="#16213e",
                                                                            color=TEXT,
                                                                            border=BORDER,
                                                                            padding="4px 6px",
                                                                            fontSize="11px",
                                                                            width="90px",
                                                                            borderRadius="4px",
                                                                            outline="none",
                                                                        ),
                                                                    ),
                                                                    html.Button(
                                                                        "x",
                                                                        id={
                                                                            "type": "filter-clear",
                                                                            "index": i,
                                                                        },
                                                                        style=dict(
                                                                            background="none",
                                                                            border="none",
                                                                            color="#555",
                                                                            cursor="pointer",
                                                                            fontSize="16px",
                                                                            lineHeight="1",
                                                                            padding="0 2px",
                                                                        ),
                                                                    ),
                                                                ],
                                                            )
                                                            for i in range(MAX_FILTERS)
                                                        ],
                                                    ],
                                                )
                                            ],
                                        ),
                                    ],
                                )
                            ],
                        ),
                    ],
                ),
                # -- Right pane: tabs --
                html.Div(
                    style=dict(
                        flex="2",
                        minWidth="0",
                        padding="12px",
                        overflow="hidden",
                        display="flex",
                        flexDirection="column",
                    ),
                    children=[
                        dcc.Tabs(
                            id="tabs",
                            value="umap",
                            parent_style=dict(
                                display="flex",
                                flexDirection="column",
                                flex="1",
                                minHeight="0",
                            ),
                            content_style=dict(
                                flex="1",
                                minHeight="0",
                                overflow="auto",
                                display="flex",
                                flexDirection="column",
                            ),
                            colors=dict(
                                border=BORDER,
                                primary=ACCENT,
                                background=PANEL_BG,
                            ),
                            children=[
                                # -- UMAP tab --
                                dcc.Tab(
                                    label="Projection",
                                    value="umap",
                                    style=dict(color=TEXT, backgroundColor=PANEL_BG),
                                    selected_style=dict(
                                        color=ACCENT,
                                        backgroundColor=DARK_BG,
                                        borderTop=f"2px solid {ACCENT}",
                                    ),
                                    children=[
                                        html.Div(
                                            style=dict(
                                                display="flex",
                                                alignItems="center",
                                                gap="16px",
                                                padding="8px 4px 12px",
                                            ),
                                            children=[
                                                html.Label(
                                                    "Color by:",
                                                    style=dict(fontSize="13px"),
                                                ),
                                                dcc.Dropdown(
                                                    id="umap-color-col",
                                                    options=_color_col_options,
                                                    value="breakdown_type" if "breakdown_type" in all_cols else None,
                                                    clearable=True,
                                                    style=DROPDOWN_STYLE,
                                                ),
                                            ],
                                        ),
                                        # Changing a projection setting refits the
                                        # embedding, which takes long enough to
                                        # look like the app has stopped. The
                                        # spinner covers the plot until the new
                                        # figure arrives.
                                        dcc.Loading(
                                            type="circle",
                                            color=ACCENT,
                                            delay_show=_SPINNER_DELAY,
                                            target_components={"umap-plot": "figure"},  # type: ignore
                                            overlay_style=dict(visibility="visible", opacity=0.35),
                                            children=dcc.Graph(
                                                id="umap-plot",
                                                config=dict(displayModeBar=True, displaylogo=False),
                                                style=dict(height=_SCATTER_H),
                                            ),
                                        ),
                                    ],
                                ),
                                # -- Pairplot tab --
                                dcc.Tab(
                                    label="Pairwise Scatter",
                                    value="pair",
                                    style=dict(color=TEXT, backgroundColor=PANEL_BG),
                                    selected_style=dict(
                                        color=ACCENT,
                                        backgroundColor=DARK_BG,
                                        borderTop=f"2px solid {ACCENT}",
                                    ),
                                    children=[
                                        html.Div(
                                            style=dict(
                                                display="flex",
                                                alignItems="flex-end",
                                                gap="16px",
                                                padding="8px 4px 12px",
                                                flexWrap="wrap",
                                            ),
                                            children=[
                                                # X axis
                                                html.Div(
                                                    [
                                                        html.Label(
                                                            "X axis",
                                                            style=dict(
                                                                fontSize="12px",
                                                                display="block",
                                                                marginBottom="4px",
                                                            ),
                                                        ),
                                                        html.Div(
                                                            [
                                                                dcc.Dropdown(
                                                                    id="pair-x-col",
                                                                    options=[
                                                                        {
                                                                            "label": c,
                                                                            "value": c,
                                                                        }
                                                                        for c in _pair_axis_cols
                                                                    ],
                                                                    value=_pair_axis_cols[0]
                                                                    if _pair_axis_cols
                                                                    else None,
                                                                    clearable=False,
                                                                    style=DROPDOWN_STYLE,
                                                                ),
                                                                dcc.RadioItems(
                                                                    id="pair-x-scale",
                                                                    options=[
                                                                        {
                                                                            "label": "Lin",
                                                                            "value": "linear",
                                                                        },
                                                                        {
                                                                            "label": "Log",
                                                                            "value": "log",
                                                                        },
                                                                    ],
                                                                    value="linear",
                                                                    inline=True,
                                                                    labelStyle=dict(
                                                                        marginRight="10px",
                                                                        fontSize="12px",
                                                                        cursor="pointer",
                                                                        color=TEXT,
                                                                    ),
                                                                    style=dict(
                                                                        whiteSpace="nowrap",
                                                                        paddingLeft="8px",
                                                                    ),
                                                                ),
                                                            ],
                                                            style=dict(
                                                                display="flex",
                                                                alignItems="center",
                                                            ),
                                                        ),
                                                    ]
                                                ),
                                                # Y axis
                                                html.Div(
                                                    [
                                                        html.Label(
                                                            "Y axis",
                                                            style=dict(
                                                                fontSize="12px",
                                                                display="block",
                                                                marginBottom="4px",
                                                            ),
                                                        ),
                                                        html.Div(
                                                            [
                                                                dcc.Dropdown(
                                                                    id="pair-y-col",
                                                                    options=[
                                                                        {
                                                                            "label": c,
                                                                            "value": c,
                                                                        }
                                                                        for c in _pair_axis_cols
                                                                    ],
                                                                    value=_pair_axis_cols[1]
                                                                    if len(_pair_axis_cols) > 1
                                                                    else None,
                                                                    clearable=False,
                                                                    style=DROPDOWN_STYLE,
                                                                ),
                                                                dcc.RadioItems(
                                                                    id="pair-y-scale",
                                                                    options=[
                                                                        {
                                                                            "label": "Lin",
                                                                            "value": "linear",
                                                                        },
                                                                        {
                                                                            "label": "Log",
                                                                            "value": "log",
                                                                        },
                                                                    ],
                                                                    value="linear",
                                                                    inline=True,
                                                                    labelStyle=dict(
                                                                        marginRight="10px",
                                                                        fontSize="12px",
                                                                        cursor="pointer",
                                                                        color=TEXT,
                                                                    ),
                                                                    style=dict(
                                                                        whiteSpace="nowrap",
                                                                        paddingLeft="8px",
                                                                    ),
                                                                ),
                                                            ],
                                                            style=dict(
                                                                display="flex",
                                                                alignItems="center",
                                                            ),
                                                        ),
                                                    ]
                                                ),
                                                # Color by
                                                html.Div(
                                                    [
                                                        html.Label(
                                                            "Color by (optional)",
                                                            style=dict(
                                                                fontSize="12px",
                                                                display="block",
                                                                marginBottom="4px",
                                                            ),
                                                        ),
                                                        dcc.Dropdown(
                                                            id="pair-color-col",
                                                            options=_color_col_options,
                                                            value=None,
                                                            clearable=True,
                                                            placeholder="None",
                                                            style=DROPDOWN_STYLE,
                                                        ),
                                                    ]
                                                ),
                                            ],
                                        ),
                                        # Switching an axis, its scale or a
                                        # filter rebuilds the decision
                                        # surface, which takes long enough to
                                        # look like nothing happened. The
                                        # spinner covers the plot, and the
                                        # previous figure stays visible
                                        # underneath, until the new one
                                        # arrives.
                                        dcc.Loading(
                                            type="circle",
                                            color=ACCENT,
                                            delay_show=_SPINNER_DELAY,
                                            target_components={"pair-plot": "figure"},  # type: ignore
                                            overlay_style=dict(visibility="visible", opacity=0.35),
                                            children=dcc.Graph(
                                                id="pair-plot",
                                                config=dict(displayModeBar=True, displaylogo=False),
                                                style=dict(height=_SCATTER_H),
                                            ),
                                        ),
                                    ],
                                ),
                                # -- Data Table tab --
                                dcc.Tab(
                                    label="Data Table",
                                    value="datatable",
                                    style=dict(color=TEXT, backgroundColor=PANEL_BG),
                                    selected_style=dict(
                                        color=ACCENT,
                                        backgroundColor=DARK_BG,
                                        borderTop=f"2px solid {ACCENT}",
                                    ),
                                    children=[
                                        html.Div(
                                            style=dict(
                                                padding="8px 4px 6px",
                                                display="flex",
                                                alignItems="center",
                                                gap="8px",
                                            ),
                                            children=[
                                                html.Label(
                                                    "Search shot ID:",
                                                    style=dict(fontSize="12px", color="#888", whiteSpace="nowrap"),
                                                ),
                                                dcc.Input(
                                                    id="shot-id-search",
                                                    type="text",
                                                    placeholder="e.g. 5304",
                                                    debounce=True,
                                                    style=dict(
                                                        backgroundColor="#16213e",
                                                        color=TEXT,
                                                        border=BORDER,
                                                        borderRadius="4px",
                                                        padding="4px 8px",
                                                        fontSize="12px",
                                                        width="160px",
                                                        outline="none",
                                                    ),
                                                ),
                                                html.Button(
                                                    "Download CSV",
                                                    id="download-table-btn",
                                                    n_clicks=0,
                                                    style=dict(
                                                        marginLeft="auto",
                                                        backgroundColor="#2a2a4a",
                                                        color=TEXT,
                                                        border=BORDER,
                                                        padding="4px 12px",
                                                        cursor="pointer",
                                                        borderRadius="4px",
                                                        fontSize="11px",
                                                    ),
                                                ),
                                            ],
                                        ),
                                        dash_table.DataTable(
                                            id="shot-table",
                                            columns=_table_column_defs,  # type: ignore
                                            data=[],
                                            virtualization=True,
                                            page_action="none",
                                            sort_action="native",
                                            sort_mode="multi",
                                            fixed_rows={"headers": True},
                                            style_table={
                                                "height": "600px",
                                                "overflowY": "auto",
                                                "overflowX": "auto",
                                                "minWidth": "100%",
                                            },
                                            style_cell=dict(
                                                backgroundColor="#16213e",
                                                color=TEXT,
                                                fontSize="11px",
                                                padding="3px 10px",
                                                border="1px solid #2a2a4a",
                                                minWidth="80px",
                                                whiteSpace="nowrap",
                                                overflow="hidden",
                                                textOverflow="ellipsis",
                                            ),
                                            style_header=dict(
                                                backgroundColor=PANEL_BG,
                                                color=ACCENT,
                                                fontWeight="600",
                                                fontSize="11px",
                                                border="1px solid #2a2a4a",
                                            ),
                                            style_data_conditional=[],
                                        ),
                                    ],
                                ),
                                # -- Search tab --
                                dcc.Tab(
                                    label="Search",
                                    value="search",
                                    style=dict(color=TEXT, backgroundColor=PANEL_BG),
                                    selected_style=dict(
                                        color=ACCENT,
                                        backgroundColor=DARK_BG,
                                        borderTop=f"2px solid {ACCENT}",
                                    ),
                                    children=[
                                        html.Div(
                                            style=dict(
                                                padding="8px 4px 24px",
                                                overflowY="auto",
                                                height=_SCATTER_H,
                                            ),
                                            children=[
                                                # ── Similarity search ──────────────────────
                                                html.Div(
                                                    style=dict(
                                                        borderBottom=BORDER,
                                                        paddingBottom="12px",
                                                        marginBottom="12px",
                                                    ),
                                                    children=[
                                                        html.Label(
                                                            "Find similar shots",
                                                            style=dict(
                                                                fontSize="12px",
                                                                color=ACCENT,
                                                                fontWeight="600",
                                                                display="block",
                                                                marginBottom="8px",
                                                            ),
                                                        ),
                                                        html.Div(
                                                            style=dict(
                                                                display="flex",
                                                                gap="8px",
                                                                alignItems="flex-end",
                                                                flexWrap="wrap",
                                                                marginBottom="8px",
                                                            ),
                                                            children=[
                                                                html.Div(
                                                                    [
                                                                        html.Label(
                                                                            "Shot ID",
                                                                            style=_CLUSTER_LABEL_STYLE,
                                                                        ),
                                                                        dcc.Input(
                                                                            id="search-query-shot",
                                                                            type="number",
                                                                            placeholder="e.g. 45000",
                                                                            debounce=False,
                                                                            style=dict(
                                                                                backgroundColor="#16213e",
                                                                                color=TEXT,
                                                                                border=BORDER,
                                                                                padding="4px 6px",
                                                                                fontSize="11px",
                                                                                width="90px",
                                                                                borderRadius="4px",
                                                                                outline="none",
                                                                            ),
                                                                        ),
                                                                    ]
                                                                ),
                                                                html.Div(
                                                                    [
                                                                        html.Label(
                                                                            "K results",
                                                                            style=_CLUSTER_LABEL_STYLE,
                                                                        ),
                                                                        dcc.Input(
                                                                            id="search-k",
                                                                            type="number",
                                                                            value=10,
                                                                            min=1,
                                                                            max=50,
                                                                            step=1,
                                                                            style=_CLUSTER_INPUT_STYLE,
                                                                        ),
                                                                    ]
                                                                ),
                                                            ],
                                                        ),
                                                        html.Div(
                                                            style=dict(marginBottom="8px"),
                                                            children=[
                                                                html.Label(
                                                                    "Features",
                                                                    style=_CLUSTER_LABEL_STYLE,
                                                                ),
                                                                dcc.Dropdown(
                                                                    id="search-features",
                                                                    options=[
                                                                        {"label": c, "value": c} for c in _search_cols
                                                                    ],
                                                                    value=_search_cols,
                                                                    multi=True,
                                                                    style=dict(
                                                                        backgroundColor="#16213e",
                                                                        color="#000",
                                                                        fontSize="11px",
                                                                    ),
                                                                ),
                                                            ],
                                                        ),
                                                        html.Button(
                                                            "Find similar shots",
                                                            id="find-similar-btn",
                                                            n_clicks=0,
                                                            style=dict(
                                                                backgroundColor=ACCENT,
                                                                color="#000",
                                                                border="none",
                                                                padding="4px 12px",
                                                                cursor="pointer",
                                                                borderRadius="4px",
                                                                fontSize="11px",
                                                                fontWeight="600",
                                                            ),
                                                        ),
                                                    ],
                                                ),
                                                # ── Traces (above table) ──────────────────
                                                html.Span(
                                                    id="search-traces-status",
                                                    style=dict(
                                                        fontSize="11px",
                                                        color="#888",
                                                        display="block",
                                                        marginBottom="4px",
                                                    ),
                                                ),
                                                dcc.Loading(
                                                    type="circle",
                                                    color=ACCENT,
                                                    children=dcc.Graph(
                                                        id="search-traces-plot",
                                                        figure=empty_traces_fig("Select a shot to load similar traces"),
                                                        responsive=True,
                                                        config=dict(
                                                            displayModeBar=True,
                                                            displaylogo=False,
                                                        ),
                                                        style=dict(height="600px"),
                                                    ),
                                                ),
                                                # ── Results table ─────────────────────────
                                                html.Hr(
                                                    style=dict(
                                                        borderColor="#2a2a4a",
                                                        margin="10px 0",
                                                    )
                                                ),
                                                html.Span(
                                                    id="search-status",
                                                    style=dict(
                                                        fontSize="11px",
                                                        color="#888",
                                                        display="block",
                                                        marginBottom="6px",
                                                    ),
                                                ),
                                                dash_table.DataTable(
                                                    id="search-results-table",
                                                    columns=[
                                                        {"name": "shot_id", "id": "shot_id"},
                                                        {"name": "rank", "id": "rank"},
                                                        {
                                                            "name": "score",
                                                            "id": "score",
                                                            "type": "numeric",
                                                            "format": {"specifier": ".3f"},
                                                        },
                                                    ],
                                                    data=[],
                                                    page_size=20,
                                                    style_table={"overflowX": "auto"},
                                                    style_cell=dict(
                                                        backgroundColor="#16213e",
                                                        color=TEXT,
                                                        fontSize="11px",
                                                        padding="3px 10px",
                                                        border="1px solid #2a2a4a",
                                                    ),
                                                    style_header=dict(
                                                        backgroundColor=PANEL_BG,
                                                        color=ACCENT,
                                                        fontWeight="600",
                                                        fontSize="11px",
                                                        border="1px solid #2a2a4a",
                                                    ),
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                                # -- Correlation tab --
                                dcc.Tab(
                                    label="Correlation",
                                    value="correlation",
                                    style=dict(color=TEXT, backgroundColor=PANEL_BG),
                                    selected_style=dict(
                                        color=ACCENT,
                                        backgroundColor=DARK_BG,
                                        borderTop=f"2px solid {ACCENT}",
                                    ),
                                    children=[
                                        html.Div(
                                            style=dict(
                                                padding="8px 4px 12px",
                                                display="flex",
                                                alignItems="flex-end",
                                                gap="16px",
                                            ),
                                            children=[
                                                html.Div(
                                                    style=dict(flex="1"),
                                                    children=[
                                                        html.Label(
                                                            "Features",
                                                            style=dict(
                                                                fontSize="12px",
                                                                display="block",
                                                                marginBottom="4px",
                                                            ),
                                                        ),
                                                        dcc.Dropdown(
                                                            id="corr-features",
                                                            options=[{"label": c, "value": c} for c in numeric_cols],
                                                            value=(UMAP_FEATURES or numeric_cols),
                                                            multi=True,
                                                            placeholder="Select feature columns...",
                                                            style=dict(
                                                                backgroundColor="#16213e",
                                                                color="#000",
                                                                fontSize="12px",
                                                            ),
                                                        ),
                                                    ],
                                                ),
                                            ],
                                        ),
                                        dcc.Graph(
                                            id="corr-plot",
                                            config=dict(displayModeBar=True, displaylogo=False),
                                            style=dict(height=_SCATTER_H),
                                        ),
                                    ],
                                ),
                                # -- Lineage tab (needs a reference column) --
                                #
                                # Always in the tree and disabled when there is
                                # no reference column. Leaving it out would
                                # break its callbacks, which are registered at
                                # import and cannot be added afterwards, and the
                                # Configuration tab can set the column while the
                                # app runs.
                                #
                                # Disabled always means "no reference column"
                                # here, and then the tab is hidden instead of
                                # greyed out: a tab that can never open is only
                                # clutter. dcc.Tab uses disabled_style in place
                                # of style whenever disabled is True, so this
                                # needs no callback of its own -- the existing
                                # one writing `disabled` hides and shows it.
                                dcc.Tab(
                                    label="Lineage",
                                    value="lineage",
                                    id="lineage-tab",
                                    disabled=not SHOW_REF_TOGGLE,
                                    disabled_style=dict(_HIDE),
                                    style=dict(color=TEXT, backgroundColor=PANEL_BG),
                                    selected_style=dict(
                                        color=ACCENT,
                                        backgroundColor=DARK_BG,
                                        borderTop=f"2px solid {ACCENT}",
                                    ),
                                    children=_lineage_tab_children(),
                                ),
                                # -- Configuration tab --
                                dcc.Tab(
                                    label="Configuration",
                                    value="config",
                                    style=dict(color=TEXT, backgroundColor=PANEL_BG),
                                    selected_style=dict(
                                        color=ACCENT,
                                        backgroundColor=DARK_BG,
                                        borderTop=f"2px solid {ACCENT}",
                                    ),
                                    children=_cfg_tab_children(),
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
    ],
)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


def _add_selection_highlight(fig: go.Figure, plot_df: pd.DataFrame, x_col: str, y_col: str, selected_shot) -> go.Figure:
    """Overlay a highlighted marker on the selected shot so it persists across re-renders."""
    if selected_shot is None:
        return fig
    sel = plot_df[plot_df["shot_id"] == selected_shot]
    if sel.empty:
        return fig
    fig.add_trace(
        go.Scatter(
            x=sel[x_col],
            y=sel[y_col],
            mode="markers",
            marker=dict(
                size=14,
                color="white",
                line=dict(color=ACCENT, width=2.5),
                symbol="circle",
            ),
            showlegend=False,
            hoverinfo="skip",
            name="_selection",
        )
    )
    return fig


def _add_search_highlight(
    fig: go.Figure,
    plot_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    search_results: list[int] | None,
) -> go.Figure:
    """Overlay gold ring markers for search results, fading by rank."""
    if not search_results:
        return fig
    n = len(search_results)
    for rank, shot_id in enumerate(search_results):
        row = plot_df[plot_df["shot_id"] == shot_id]
        if row.empty:
            continue
        opacity = max(0.35, 1.0 - rank / max(n - 1, 1) * 0.65)
        fig.add_trace(
            go.Scatter(
                x=row[x_col],
                y=row[y_col],
                mode="markers",
                marker=dict(
                    size=12,
                    color="rgba(0,0,0,0)",
                    line=dict(color=f"rgba(255,215,0,{opacity:.2f})", width=2),
                    symbol="circle",
                ),
                customdata=row[["shot_id"]].values,
                hovertemplate=f"rank {rank + 1}: %{{customdata[0]}}<extra></extra>",
                showlegend=False,
                name="_search",
            )
        )
    return fig


_LATEST_SHOT_COLOR = "#2ecc71"


def _add_latest_shot_highlight(
    fig: go.Figure,
    plot_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    latest_shot,
    enabled: bool,
) -> go.Figure:
    """Overlay a marker on the most-recently-added shot, updated as new shots arrive."""
    if not enabled or latest_shot is None:
        return fig
    row = plot_df[plot_df["shot_id"] == latest_shot]
    if row.empty:
        return fig
    fig.add_trace(
        go.Scatter(
            x=row[x_col],
            y=row[y_col],
            mode="markers",
            marker=dict(
                size=13,
                color="rgba(0,0,0,0)",
                line=dict(color=_LATEST_SHOT_COLOR, width=2.5),
                symbol="diamond",
            ),
            customdata=row[["shot_id"]].values,
            hovertemplate="latest shot: %{customdata[0]}<extra></extra>",
            showlegend=False,
            name="_latest",
        )
    )
    return fig


_SCATTER_LAYOUT = dict(
    paper_bgcolor=DARK_BG,
    plot_bgcolor="#16213e",
    font=dict(color=TEXT, size=11),
    margin=dict(l=50, r=30, t=40, b=50),
    autosize=True,
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#333"),
    xaxis=dict(gridcolor="#2a2a4a", zerolinecolor="#444"),
    yaxis=dict(gridcolor="#2a2a4a", zerolinecolor="#444"),
    clickmode="event+select",
)


# ---------------------------------------------------------------------------
# Scatter colouring and the decision surface
#
# The Projection and the Pairwise Scatter differ only in their axes, so the
# colour rules and the surface live here once rather than twice.
# ---------------------------------------------------------------------------


def _class_color_map(classes: list[str]) -> dict[str, str]:
    """A fixed colour per class, by position rather than by name.

    Positional so the surface's colourbar and the point colours agree, and so
    a class keeps its colour when the model is retrained with the same target.
    """
    palette = px.colors.qualitative.Plotly
    return {name: palette[i % len(palette)] for i, name in enumerate(classes)}


def _resolve_color(
    plot_df: pd.DataFrame,
    color_col,
    cluster_labels,
    cluster_names,
    outlier_labels,
    class_labels,
    class_model,
) -> tuple[pd.DataFrame, dict]:
    """Return the frame to plot and the px.scatter kwargs that colour it.

    The three computed colourings -- cluster, outlier, predicted label -- all
    live in stores rather than in the dataset, so each merges its own column
    into a copy of *plot_df* and names it.
    """
    if color_col == _CLUSTER_COLOR_VALUE and cluster_labels:
        enriched, col = _apply_cluster_color(plot_df, cluster_labels, cluster_names or {})
        return enriched, {"color": col}
    if color_col == _OUTLIER_COLOR_VALUE and outlier_labels:
        enriched, col = _apply_outlier_color(plot_df, outlier_labels)
        return enriched, {
            "color": col,
            "color_discrete_map": {"Outlier": _OUTLIER_RED, "Inlier": _INLIER_BLUE},
        }
    if color_col == _CLASS_COLOR_VALUE and class_labels:
        enriched, col = _apply_class_color(plot_df, class_labels)
        classes = (class_model or {}).get("classes") or sorted(set(class_labels.values()))
        return enriched, {
            "color": col,
            "category_orders": {col: classes},
            "color_discrete_map": _class_color_map(classes),
        }
    if color_col and color_col in plot_df.columns:
        valid = plot_df[color_col].notna()
        if valid.any():
            return plot_df[valid], {"color": color_col}
    return plot_df, {}


def _surface_class_index(class_model, surface_class) -> int:
    """Position of the class whose probability the surface shows.

    Defaults to the last class, so a 0/1 target shows P(1) -- the reading the
    word "probability" has for a binary problem.
    """
    classes = (class_model or {}).get("classes") or []
    if surface_class in classes:
        return classes.index(surface_class)
    return max(len(classes) - 1, 0)


def _shot_fingerprint(df: pd.DataFrame) -> str:
    """A cheap, exact identifier for which shots a frame holds.

    Hashing the ids themselves rather than the filter definitions: two
    different sets of filters that keep the same shots produce the same
    surface, and should share the cached one.

    blake2b rather than the built-in ``hash``, which is salted per process:
    the grids behind this key are shared between gunicorn workers, so two
    workers looking at the same shots have to name them the same way.
    """
    ids = np.ascontiguousarray(df["shot_id"].astype("int64").values).tobytes()
    return hashlib.blake2b(ids, digest_size=16).hexdigest()


def _surface_grid(
    df: pd.DataFrame,
    proba,
    class_model,
    x_col: str,
    y_col: str,
    class_index: int,
    log_axes: tuple[bool, bool] = (False, False),
) -> dict | None:
    """The probability grid for one plane, computed once and cached.

    *df* is the filtered frame, so the surface covers the shots on screen and
    no others. A filter is a statement about which shots are of interest, and
    a surface drawn over the discarded ones would assert a boundary through
    data the user has just said to leave out.

    *log_axes* is part of the key as well as the calculation: a log plot needs
    a grid spaced in log10, so it is a different grid, not the same one drawn
    differently.
    """
    token = (class_model or {}).get("token")
    if not token or not proba or x_col not in df.columns or y_col not in df.columns:
        return None
    key = (token, x_col, y_col, class_index, _shot_fingerprint(df), log_axes)
    cached = _surface_get(key)
    if cached is not None:
        return cached

    values = np.asarray(proba["values"], dtype=float)
    if class_index >= values.shape[1]:
        return None
    by_shot = pd.Series(values[:, class_index], index=[int(s) for s in proba["shot_ids"]])
    p = df["shot_id"].astype(int).map(by_shot)
    keep = p.notna()
    if not keep.any():
        return None
    try:
        grid = decision_surface(
            df.loc[keep, [x_col, y_col]].values.astype(float),
            p[keep].values.astype(float),
            resolution=_SURFACE_RESOLUTION,
            log_axes=log_axes,
        )
    except Exception as exc:
        log.error("[decision surface] %s", exc)
        return None
    if grid is not None:
        _surface_put(key, grid)
    return grid


def _decision_surface_trace(grid: dict, class_name: str) -> go.Heatmap:
    """The surface as a heatmap trace.

    ``hoverinfo="skip"`` matters as much as the colours: without it the
    surface sits over every point and swallows the clicks that select a shot.
    """
    return go.Heatmap(
        x=grid["x"],
        y=grid["y"],
        z=grid["z"],
        zmin=0.0,
        zmax=1.0,
        colorscale=_SURFACE_COLORSCALE,
        opacity=_SURFACE_OPACITY,
        hoverinfo="skip",
        showlegend=False,
        colorbar=dict(
            title=dict(text=f"P({class_name})", font=dict(color=TEXT, size=10)),
            tickfont=dict(color=TEXT, size=9),
            bgcolor=PANEL_BG,
            bordercolor="#2a2a4a",
            thickness=12,
            len=0.55,
            x=1.02,
        ),
    )


def _surface_for(
    df: pd.DataFrame,
    proba,
    class_model,
    show_surface,
    surface_class,
    x_col: str,
    y_col: str,
    log_axes: tuple[bool, bool] = (False, False),
) -> go.Heatmap | None:
    """The surface trace for one plane, or None when there is nothing to draw.

    *df* is the filtered frame the points are drawn from, so the surface is
    cropped to the same shots.
    """
    if not show_surface or not class_model or not proba:
        return None
    index = _surface_class_index(class_model, surface_class)
    grid = _surface_grid(df, proba, class_model, x_col, y_col, index, log_axes)
    if grid is None:
        return None
    classes = class_model.get("classes") or []
    return _decision_surface_trace(grid, classes[index] if index < len(classes) else "class")


def _scatter_with_surface(kwargs: dict, surface: go.Heatmap | None) -> go.Figure:
    """px.scatter, optionally over a decision surface.

    Plotly draws traces in the order they are added, so the heatmap has to go
    in before the points or it would cover them.
    """
    scatter = px.scatter(**kwargs)
    if surface is None:
        return scatter
    fig = go.Figure(layout=scatter.layout)
    fig.add_trace(surface)
    for trace in scatter.data:
        fig.add_trace(trace)
    return fig


SELECT_VARIABLE_MSG = "Select a variable to load data"
_LIN_NO_REFERENCE_MSG = (
    "No reference shot column is set. Choose one under Table columns in the Configuration tab to use this tab."
)


def _empty_fig(message: str) -> go.Figure:
    """A blank, dark-themed figure carrying a centred message."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=14, color="#aaa"),
    )
    fig.update_layout(
        paper_bgcolor=DARK_BG,
        plot_bgcolor="#16213e",
        margin=dict(l=50, r=30, t=40, b=50),
    )
    return fig


# ---------------------------------------------------------------------------
# Lineage tab — figure builders
# ---------------------------------------------------------------------------


def _lin_tree_fig(matrix, ref_parent: dict[int, int], subject: int) -> go.Figure:
    """Node-link timeline of the lineage: shot ID across, generation down.

    Shares ``_ref_shot_color`` with the scatter-plot overlay, so the same shot
    is the same colour in both reference views.
    """
    ids = list(matrix.shot_ids)
    if len(ids) <= 1:
        return _empty_fig("Only one shot in this lineage")

    # Generation = distance from the oldest ancestor reachable inside the
    # lineage, so a branch and its siblings sit on different rows.
    in_lineage = set(ids)
    depth: dict[int, int] = {}

    def _depth(shot: int, seen: frozenset[int] = frozenset()) -> int:
        if shot in depth:
            return depth[shot]
        parent = ref_parent.get(shot)
        if parent is None or parent not in in_lineage or shot in seen:
            depth[shot] = 0
        else:
            depth[shot] = _depth(parent, seen | {shot}) + 1
        return depth[shot]

    for shot in ids:
        _depth(shot)

    lo, hi = min(ids), max(ids)
    fig = go.Figure()
    for shot in ids:
        parent = ref_parent.get(shot)
        if parent is None or parent not in in_lineage:
            continue
        fig.add_trace(
            go.Scatter(
                x=[parent, shot],
                y=[depth[parent], depth[shot]],
                mode="lines",
                line=dict(color=_ref_shot_color(min(shot, parent), lo, hi), width=2, dash="dot"),
                showlegend=False,
                hoverinfo="skip",
                name="_lin_edge",
            )
        )

    others = [s for s in ids if s != subject]
    if others:
        fig.add_trace(
            go.Scatter(
                x=others,
                y=[depth[s] for s in others],
                mode="markers+text",
                marker=dict(
                    size=13,
                    color=[_ref_shot_color(s, lo, hi) for s in others],
                    line=dict(color="rgba(0,0,0,0.4)", width=1),
                ),
                text=[str(s) for s in others],
                textposition="top center",
                textfont=dict(size=9, color="#888"),
                # _extract_shot_id reads hovertext first, so this is what makes
                # clicking a node select that shot everywhere else.
                hovertext=[str(s) for s in others],
                hovertemplate="shot %{hovertext}<extra></extra>",
                showlegend=False,
                name="_lin_nodes",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=[subject],
            y=[depth.get(subject, 0)],
            mode="markers+text",
            marker=dict(size=18, color=ACCENT, symbol="star", line=dict(color="#fff", width=1)),
            text=[str(subject)],
            textposition="top center",
            textfont=dict(size=10, color=TEXT),
            hovertext=[str(subject)],
            hovertemplate="shot %{hovertext} (subject)<extra></extra>",
            showlegend=False,
            name="_lin_subject",
        )
    )

    # _SCATTER_LAYOUT already carries xaxis/yaxis, so the axis titles go on
    # afterwards rather than as duplicate update_layout keywords.
    fig.update_layout(**_SCATTER_LAYOUT)
    fig.update_xaxes(title="shot ID")
    fig.update_yaxes(title="generation", autorange="reversed", showticklabels=False)
    return fig


def _lin_spark_fig(matrix, columns: list[str], subject: int) -> go.Figure:
    """Small multiples: each variable's value across the lineage, oldest first.

    Ordered most-changed first, so the panels that answer the question come
    before the ones that do not.
    """
    if not columns:
        return _empty_fig("Select at least one variable")
    numeric = [c for c in columns if matrix.kinds.get(c) == "numeric"]
    if not numeric:
        return _empty_fig("Sparklines need at least one numeric variable")

    # Oldest -> newest reads left to right, which is how a trend is read.
    order = list(reversed(matrix.shot_ids))
    cols = 2
    rows = math.ceil(len(numeric) / cols)
    fig = make_subplots(
        rows=rows,
        cols=cols,
        shared_xaxes=False,
        vertical_spacing=min(0.08, 0.6 / max(rows, 1)),
        horizontal_spacing=0.10,
        subplot_titles=numeric,
    )
    for i, col in enumerate(numeric):
        row, column = divmod(i, cols)
        series = pd.to_numeric(matrix.values.loc[order, col], errors="coerce")
        fig.add_trace(
            go.Scatter(
                x=[str(s) for s in order],
                y=series.to_numpy(dtype=float),
                mode="lines+markers",
                line=dict(color=ACCENT, width=1.5),
                marker=dict(
                    size=[11 if s == subject else 6 for s in order],
                    color=[ACCENT if s == subject else "#4488cc" for s in order],
                    symbol=["star" if s == subject else "circle" for s in order],
                ),
                showlegend=False,
                hovertemplate="shot %{x}<br>%{y:.4g}<extra></extra>",
                name=col,
            ),
            row=row + 1,
            col=column + 1,
        )
    fig.update_layout(
        paper_bgcolor=DARK_BG,
        plot_bgcolor="#16213e",
        font=dict(color=TEXT, size=10),
        margin=dict(l=50, r=20, t=30, b=30),
        height=max(300, 130 * rows),
    )
    fig.update_xaxes(gridcolor="#2a2a4a", tickfont=dict(size=8), showticklabels=True)
    fig.update_yaxes(gridcolor="#2a2a4a", tickfont=dict(size=8))
    for annotation in fig.layout.annotations:
        annotation.font.size = 10
        annotation.font.color = ACCENT
    return fig


@app.callback(
    Output("active-filters", "data"),
    Input({"type": "filter-col", "index": ALL}, "value"),
    Input({"type": "filter-op", "index": ALL}, "value"),
    Input({"type": "filter-val", "index": ALL}, "value"),
    Input("filter-logic", "value"),
    Input("dataset-key", "data"),
    Input("dataset-version", "data"),
)
def apply_filters(cols, ops, vals, logic, variable, _dataset_version):
    ds = get_dataset(variable)
    if ds is None:
        return None
    return compute_active_filter_ids(ds.df, cols, ops, vals, logic)


@app.callback(
    Output("filter-count-display", "children"),
    Input("active-filters", "data"),
    Input("dataset-key", "data"),
    Input("dataset-version", "data"),
)
def update_filter_count(active_filters, variable, _dataset_version):
    ds = get_dataset(variable)
    if ds is None or active_filters is None:
        return ""
    return f"{len(active_filters):,} / {len(ds.df):,} shots shown"


@app.callback(
    Output("shot-count-display", "children"),
    Input("dataset-key", "data"),
    Input("dataset-version", "data"),
)
def update_shot_count(variable, _dataset_version):
    ds = get_dataset(variable)
    return "shots: —" if ds is None else f"shots: {len(ds.df):,}"


@app.callback(
    Output({"type": "filter-col", "index": ALL}, "value"),
    Output({"type": "filter-val", "index": ALL}, "value"),
    Input("filter-clear-all", "n_clicks"),
    Input({"type": "filter-clear", "index": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def clear_filters(_, _row_clicks):
    triggered = dash.ctx.triggered_id
    if triggered == "filter-clear-all":
        return [None] * MAX_FILTERS, [""] * MAX_FILTERS
    if isinstance(triggered, dict) and triggered.get("type") == "filter-clear":
        idx = triggered["index"]
        return (
            [None if i == idx else dash.no_update for i in range(MAX_FILTERS)],
            ["" if i == idx else dash.no_update for i in range(MAX_FILTERS)],
        )
    return dash.no_update, dash.no_update


@app.callback(
    Output("search-highlight-enabled", "data"),
    Output("search-highlight-btn", "children"),
    Output("search-highlight-btn", "style"),
    Input("search-highlight-btn", "n_clicks"),
    State("search-highlight-enabled", "data"),
    prevent_initial_call=True,
)
def toggle_search_highlight(n_clicks, currently_enabled):
    enabled = not currently_enabled
    if enabled:
        label = "Similar shots: ON"
        style = dict(
            backgroundColor="#1a3a6a",
            color=ACCENT,
            border=f"1px solid {ACCENT}",
            padding="4px 12px",
            cursor="pointer",
            borderRadius="4px",
            fontSize="11px",
            fontWeight="600",
        )
    else:
        label = "Similar shots: OFF"
        style = dict(
            backgroundColor="#2a2a4a",
            color="#888",
            border="1px solid #3a3a6a",
            padding="4px 12px",
            cursor="pointer",
            borderRadius="4px",
            fontSize="11px",
        )
    return enabled, label, style


@app.callback(
    Output("latest-shot-highlight-enabled", "data"),
    Output("latest-shot-highlight-btn", "children"),
    Output("latest-shot-highlight-btn", "style"),
    Input("latest-shot-highlight-btn", "n_clicks"),
    State("latest-shot-highlight-enabled", "data"),
    prevent_initial_call=True,
)
def toggle_latest_shot_highlight(n_clicks, currently_enabled):
    enabled = not currently_enabled
    if enabled:
        label = "Latest shot: ON"
        style = dict(
            backgroundColor="#1a3a6a",
            color=ACCENT,
            border=f"1px solid {ACCENT}",
            padding="4px 12px",
            cursor="pointer",
            borderRadius="4px",
            fontSize="11px",
            fontWeight="600",
        )
    else:
        label = "Latest shot: OFF"
        style = dict(
            backgroundColor="#2a2a4a",
            color="#888",
            border="1px solid #3a3a6a",
            padding="4px 12px",
            cursor="pointer",
            borderRadius="4px",
            fontSize="11px",
        )
    return enabled, label, style


@app.callback(
    Output("dataset-version", "data"),
    Input("refresh-interval", "n_intervals"),
    State("dataset-key", "data"),
    State("dataset-version", "data"),
    prevent_initial_call=True,
)
def poll_for_updates(n_intervals, variable, version):
    """Periodic tick from the dcc.Interval — check the backend for new shots."""
    latest = refresh_dataset(variable)
    if latest is None:
        return dash.no_update
    return (version or 0) + 1


@app.callback(
    Output("latest-shot", "data"),
    Input("dataset-version", "data"),
    Input("dataset-key", "data"),
)
def update_latest_shot(_dataset_version, variable):
    """Recomputed from whatever's currently loaded — fires on initial load too,
    so the highlight is correct immediately, not just after the first poll."""
    ds = get_dataset(variable)
    if ds is None or ds.df.empty:
        return None
    return int(ds.df["shot_id"].max())


@app.callback(
    Output("umap-plot", "figure"),
    Input("umap-color-col", "value"),
    Input("active-filters", "data"),
    Input("selected-shot", "data"),
    Input("cluster-labels", "data"),
    Input("cluster-names", "data"),
    Input("outlier-labels", "data"),
    Input("class-labels", "data"),
    Input("class-proba", "data"),
    Input("class-model", "data"),
    Input("classify-show-surface", "value"),
    Input("classify-surface-class", "value"),
    Input("search-results", "data"),
    Input("search-highlight-enabled", "data"),
    Input("latest-shot", "data"),
    Input("latest-shot-highlight-enabled", "data"),
    Input("dataset-key", "data"),
    Input("dataset-version", "data"),
)
def update_umap(
    color_col,
    active_filters,
    selected_shot,
    cluster_labels,
    cluster_names,
    outlier_labels,
    class_labels,
    class_proba,
    class_model,
    show_surface,
    surface_class,
    search_results,
    search_highlight_enabled,
    latest_shot,
    latest_shot_highlight_enabled,
    variable,
    _dataset_version,
) -> go.Figure:
    ds = get_dataset(variable)
    if ds is None:
        return _empty_fig(SELECT_VARIABLE_MSG)
    plot_df = _apply_filter_mask(ds.df, active_filters)
    kwargs: dict = dict(
        data_frame=plot_df,
        x="umap_x",
        y="umap_y",
        custom_data=["shot_id"],
        hover_name="shot_id",
        labels={"umap_x": ds.x_label, "umap_y": ds.y_label},
    )
    frame, color_kwargs = _resolve_color(
        plot_df, color_col, cluster_labels, cluster_names, outlier_labels, class_labels, class_model
    )
    kwargs["data_frame"] = frame
    kwargs.update(color_kwargs)

    fig = _scatter_with_surface(
        kwargs, _surface_for(plot_df, class_proba, class_model, show_surface, surface_class, "umap_x", "umap_y")
    )
    fig.update_traces(
        marker=dict(size=5, opacity=0.75),
        unselected=dict(marker=dict(opacity=0.75)),
        selector=dict(type="scatter"),
    )
    fig.update_layout(**_SCATTER_LAYOUT, uirevision="umap")
    if search_highlight_enabled:
        _add_search_highlight(fig, plot_df, "umap_x", "umap_y", search_results)
    _add_latest_shot_highlight(fig, plot_df, "umap_x", "umap_y", latest_shot, latest_shot_highlight_enabled)
    _add_selection_highlight(fig, plot_df, "umap_x", "umap_y", selected_shot)
    return fig


@app.callback(
    Output("pair-plot", "figure"),
    Input("pair-x-col", "value"),
    Input("pair-y-col", "value"),
    Input("pair-color-col", "value"),
    Input("pair-x-scale", "value"),
    Input("pair-y-scale", "value"),
    Input("active-filters", "data"),
    Input("selected-shot", "data"),
    Input("cluster-labels", "data"),
    Input("cluster-names", "data"),
    Input("outlier-labels", "data"),
    Input("class-labels", "data"),
    Input("class-proba", "data"),
    Input("class-model", "data"),
    Input("classify-show-surface", "value"),
    Input("classify-surface-class", "value"),
    Input("search-results", "data"),
    Input("search-highlight-enabled", "data"),
    Input("latest-shot", "data"),
    Input("latest-shot-highlight-enabled", "data"),
    Input("dataset-key", "data"),
    Input("dataset-version", "data"),
)
def update_pair_plot(
    x_col,
    y_col,
    color_col,
    x_scale,
    y_scale,
    active_filters,
    selected_shot,
    cluster_labels,
    cluster_names,
    outlier_labels,
    class_labels,
    class_proba,
    class_model,
    show_surface,
    surface_class,
    search_results,
    search_highlight_enabled,
    latest_shot,
    latest_shot_highlight_enabled,
    variable,
    _dataset_version,
) -> go.Figure:
    ds = get_dataset(variable)
    if ds is None:
        return _empty_fig(SELECT_VARIABLE_MSG)
    if not x_col or not y_col:
        return go.Figure()

    plot_df = _apply_filter_mask(ds.df, active_filters)
    kwargs: dict = dict(
        data_frame=plot_df,
        x=x_col,
        y=y_col,
        custom_data=["shot_id"],
        hover_name="shot_id",
    )
    frame, color_kwargs = _resolve_color(
        plot_df, color_col, cluster_labels, cluster_names, outlier_labels, class_labels, class_model
    )
    kwargs["data_frame"] = frame
    kwargs.update(color_kwargs)

    # The grid is built in whichever space each axis is drawn in, so a log
    # scale gets log-spaced cells rather than a linear grid stretched across
    # it. Plotly still wants the coordinates in data units and takes the
    # logarithm itself, which is what decision_surface returns.
    log_axes = (x_scale == "log", y_scale == "log")
    fig = _scatter_with_surface(
        kwargs,
        _surface_for(plot_df, class_proba, class_model, show_surface, surface_class, x_col, y_col, log_axes),
    )
    fig.update_traces(
        marker=dict(size=5, opacity=0.75),
        unselected=dict(marker=dict(opacity=0.75)),
        selector=dict(type="scatter"),
    )
    fig.update_layout(
        **_SCATTER_LAYOUT,
        uirevision=f"{x_col}-{y_col}",
        xaxis_type=x_scale,
        yaxis_type=y_scale,
    )
    if search_highlight_enabled:
        _add_search_highlight(fig, plot_df, x_col, y_col, search_results)
    _add_latest_shot_highlight(fig, plot_df, x_col, y_col, latest_shot, latest_shot_highlight_enabled)
    _add_selection_highlight(fig, plot_df, x_col, y_col, selected_shot)
    return fig


@app.callback(
    Output("selected-shot", "data"),
    Input("umap-plot", "clickData"),
    Input("pair-plot", "clickData"),
    Input("shot-table", "active_cell"),
    Input("dataset-key", "data"),
    State("shot-table", "derived_virtual_data"),
    prevent_initial_call=True,
)
def update_selected_shot(umap_click, pair_click, active_cell, variable, virtual_data):
    triggered_id = dash.ctx.triggered_id
    # Switching variable clears the selection — the shot may not exist in the new table.
    if triggered_id == "dataset-key":
        return None
    ds = get_dataset(variable)
    if ds is None:
        return None
    if triggered_id == "umap-plot":
        return _extract_shot_id(ds.df, umap_click)
    if triggered_id == "pair-plot":
        return _extract_shot_id(ds.df, pair_click)
    if triggered_id == "shot-table" and active_cell and virtual_data:
        return int(virtual_data[active_cell["row"]]["shot_id"])
    return dash.no_update


@app.callback(
    Output("shot-table", "data"),
    Input("shot-id-search", "value"),
    Input("dataset-key", "data"),
    Input("dataset-version", "data"),
)
def filter_table_by_shot_id(search, variable, _dataset_version):
    ds = get_dataset(variable)
    if ds is None:
        return []
    if not search or not str(search).strip():
        return ds.df[_table_cols].to_dict("records")
    query = str(search).strip()
    mask = ds.df["shot_id"].astype(str).str.contains(query, na=False)
    return ds.df.loc[mask, _table_cols].to_dict("records")


@app.callback(
    Output("shot-table", "style_data_conditional"),
    Input("selected-shot", "data"),
    Input("latest-shot", "data"),
    Input("latest-shot-highlight-enabled", "data"),
)
def highlight_table_row(selected_shot, latest_shot, latest_shot_highlight_enabled):
    styles = []
    if latest_shot_highlight_enabled and latest_shot is not None:
        styles.append(
            {
                "if": {"filter_query": f"{{shot_id}} = {latest_shot}"},
                "backgroundColor": "#1e4a33",
                "color": "white",
            }
        )
    if selected_shot is not None:
        styles.append(
            {
                "if": {"filter_query": f"{{shot_id}} = {selected_shot}"},
                "backgroundColor": "#2a3a6e",
                "color": "white",
                "fontWeight": "600",
            }
        )
    return styles


app.clientside_callback(
    """
    function(selected_shot, virtual_data) {
        if (selected_shot == null || !virtual_data) return null;
        var rowIndex = -1;
        for (var i = 0; i < virtual_data.length; i++) {
            if (virtual_data[i]['shot_id'] === selected_shot) { rowIndex = i; break; }
        }
        if (rowIndex < 0) return null;
        var tableEl = document.getElementById('shot-table');
        if (!tableEl) return null;
        var grids = tableEl.querySelectorAll('.ReactVirtualized__Grid');
        var grid = grids[grids.length - 1];
        if (grid) {
            grid.scrollTop = Math.max(0, rowIndex * 30 - grid.clientHeight / 2);
        }
        return null;
    }
    """,
    Output("_table_scroll_sink", "data"),
    Input("selected-shot", "data"),
    State("shot-table", "derived_virtual_data"),
    prevent_initial_call=True,
)


# The virtualized DataTable does not recompute its scroll viewport when its data
# is first populated while the table is already visible — it renders zero rows
# until a resize event forces a re-measure (switching tabs happens to do this).
# Nudge it with a resize whenever the row data changes so the rows always paint.
app.clientside_callback(
    """
    function(data) {
        var n = (data && data.length) || 0;
        if (n === 0) { return null; }
        var tries = 0;
        function nudge() {
            tries += 1;
            var table = document.getElementById('shot-table');
            if (table) {
                if (table.querySelectorAll('tbody tr td').length > 0) { return; }
                // The virtualizer measures the element via an element-resize
                // detector, not window.resize. Hiding then re-showing the table
                // forces a re-measure — the same thing a tab-switch does — so the
                // rows paint even when the data arrives while the table is visible.
                table.style.display = 'none';
                void table.offsetHeight;
                table.style.display = '';
            }
            if (tries < 40) { setTimeout(nudge, 100); }
        }
        window.requestAnimationFrame(nudge);
        return null;
    }
    """,
    Output("_table_repaint_sink", "data"),
    Input("shot-table", "data"),
    prevent_initial_call=True,
)


@app.callback(
    Output("shot-info-panel", "children"),
    Input("selected-shot", "data"),
    Input("dataset-key", "data"),
)
def update_shot_info(selected_shot, variable):
    ds = get_dataset(variable)
    if ds is None:
        return html.Span(
            SELECT_VARIABLE_MSG,
            style=dict(fontSize="11px", color="#555"),
        )
    if selected_shot is None:
        return html.Span(
            "Click a point to see shot details",
            style=dict(fontSize="11px", color="#555"),
        )
    row = ds.df[ds.df["shot_id"] == selected_shot]
    if row.empty:
        return html.Span(
            f"No data for shot {selected_shot}",
            style=dict(fontSize="11px", color="#555"),
        )
    items = row.iloc[0][_table_cols].items()
    return html.Table(
        style=dict(width="100%", borderCollapse="collapse", fontSize="11px"),
        children=[
            html.Tr(
                style=dict(
                    borderBottom="1px solid #2a2a4a",
                    backgroundColor="#16213e" if i % 2 == 0 else PANEL_BG,
                ),
                children=[
                    html.Td(
                        k,
                        style=dict(
                            color=ACCENT,
                            padding="3px 8px",
                            whiteSpace="nowrap",
                            fontWeight="600",
                            width="45%",
                        ),
                    ),
                    html.Td(
                        f"{v:.4g}" if isinstance(v, float) else str(v),
                        style=dict(color=TEXT, padding="3px 8px"),
                    ),
                ],
            )
            for i, (k, v) in enumerate(items)
        ],
    )


if SHOW_TRACES:

    @app.callback(
        Output("traces-plot", "figure"),
        Output("traces-title", "children"),
        Input("selected-shot", "data"),
        Input("cfg-signals", "data"),
        Input("cfg-time-window", "data"),
        Input("cfg-timebase-hz", "data"),
        Input("cfg-backend-options", "data"),
        prevent_initial_call=True,
    )
    def update_traces(shot_id, cfg_signals, cfg_time_window, cfg_timebase, cfg_options):
        if shot_id is None:
            return dash.no_update, dash.no_update
        try:
            shot_df = load_shot_traces(shot_id, cfg_signals, cfg_time_window, cfg_timebase, cfg_options)
        except Exception as exc:
            log.error("[update_traces] error loading shot %d: %s", shot_id, exc)
            return empty_traces_fig(f"Error loading shot {shot_id}"), f"Shot {shot_id} — error"
        if shot_df is None:
            return empty_traces_fig(f"No data found for shot {shot_id}"), f"Shot {shot_id} — not found"

        # Name the selected signals this shot does not have, so a typo is visible.
        missing = [s for s in _effective_signals(cfg_signals) if s not in shot_df.columns]
        title = f"Shot {shot_id}"
        if missing:
            title += f" — no data for: {', '.join(missing)}"
        return make_traces_fig(shot_df, cfg_signals), title


# ---------------------------------------------------------------------------
# SHAP callback
#
# Registered unconditionally: a model trained in the Classification tab is a
# second source of SHAP values, and it does not exist when the layout is
# built. Callbacks cannot be added after import, so the decision of what to
# show has to be made here, per request, rather than at startup.
# ---------------------------------------------------------------------------


def _shap_message(text: str) -> html.Span:
    return html.Span(text, style=dict(fontSize="11px", color="#555"))


@app.callback(
    Output("shap-container", "children"),
    Input("selected-shot", "data"),
    Input("class-model", "data"),
    Input("classify-surface-class", "value"),
    Input("dataset-key", "data"),
)
def update_shap(shot_id, class_model, surface_class, variable):
    """Explain the selected shot.

    A model trained in this session wins over the pre-computed file: it is the
    one the user is currently reasoning about, and its features are the ones
    they chose.
    """
    if shot_id is None:
        return _shap_message(_SHAP_EMPTY_MSG if (SHOW_SHAP or class_model) else _SHAP_NO_MODEL_MSG)

    if class_model:
        classes = class_model.get("classes") or []
        index = classes.index(surface_class) if surface_class in classes else len(classes) - 1
        try:
            img_b64 = make_model_shap_fig(class_model.get("token"), shot_id, index)
        except ImportError:
            return _shap_message("SHAP is not installed — pip install 'nice-shot[shap]'")
        except Exception as exc:
            log.error("[shap] %s", exc)
            return _shap_message(f"Could not explain shot {shot_id}: {exc}")
        if img_b64 is not None:
            return html.Img(src=f"data:image/png;base64,{img_b64}", style=dict(width="100%", height="auto"))
        if not SHOW_SHAP:
            return _shap_message(f"No trained-model SHAP for shot {shot_id} — train again")

    ds = get_dataset(variable)
    if ds is None or not SHOW_SHAP:
        return _shap_message(_SHAP_NO_MODEL_MSG)
    img_b64 = make_shap_fig(ds, shot_id)
    if img_b64 is None:
        return _shap_message(f"No SHAP data for shot {shot_id}")
    return html.Img(src=f"data:image/png;base64,{img_b64}", style=dict(width="100%", height="auto"))


# ---------------------------------------------------------------------------
# Clustering callbacks
# ---------------------------------------------------------------------------


@app.callback(
    Output("cluster-labels", "data"),
    Output("cluster-representatives", "data"),
    Output("cluster-status", "children"),
    Output("umap-color-col", "value"),
    Output("pair-color-col", "value"),
    Output("cluster-traces-plot", "className", allow_duplicate=True),
    Input("run-cluster-btn", "n_clicks"),
    State("cluster-algorithm", "value"),
    State("cluster-features", "value"),
    State("cluster-n", "value"),
    State("cluster-eps", "value"),
    State("cluster-min-samples", "value"),
    State("cluster-use-projection", "value"),
    State("dataset-key", "data"),
    prevent_initial_call=True,
)
def run_clustering(n_clicks, algorithm, features, n_clusters, eps, min_samples, use_projection, variable):
    # The trailing "" output is a dummy write to a prop of the dcc.Loading-wrapped
    # cluster-traces-plot: dcc.Loading only detects callbacks whose Output lands
    # directly on one of its children, not further up a chained-callback graph, so
    # this keeps the spinner showing for this (slow) first hop of that chain too.
    ds = get_dataset(variable)
    if ds is None:
        return dash.no_update, dash.no_update, SELECT_VARIABLE_MSG, dash.no_update, dash.no_update, ""
    active_features = ["umap_x", "umap_y"] if use_projection else list(features or [])
    if not active_features:
        return dash.no_update, dash.no_update, "Select at least one feature", dash.no_update, dash.no_update, ""
    eps_val = float(eps or 0.5)
    if eps_val <= 0:
        return dash.no_update, dash.no_update, "eps must be greater than 0", dash.no_update, dash.no_update, ""
    try:
        labels, representatives = _run_clustering(
            ds.df,
            algorithm=algorithm or "kmeans",
            features=active_features,
            n_clusters=int(n_clusters or 5),
            eps=eps_val,
            min_samples=int(min_samples or 5),
        )
    except Exception as exc:
        log.error("[clustering] %s", exc)
        return dash.no_update, dash.no_update, f"Error: {exc}", dash.no_update, dash.no_update, ""
    if not labels:
        return None, None, "No shots clustered — check features", dash.no_update, dash.no_update, ""
    unique = sorted(set(labels.values()))
    n_valid = sum(1 for v in unique if v >= 0)
    noise = sum(1 for v in labels.values() if v < 0)
    msg = f"{n_valid} cluster(s) across {len(labels):,} shots"
    if noise:
        msg += f" · {noise:,} noise"
    return labels, representatives, msg, _CLUSTER_COLOR_VALUE, _CLUSTER_COLOR_VALUE, ""


@app.callback(
    Output("cluster-name-inputs", "children"),
    Input("cluster-labels", "data"),
)
def render_cluster_name_inputs(cluster_labels):
    if not cluster_labels:
        return []
    counts: dict[int, int] = {}
    for v in cluster_labels.values():
        counts[v] = counts.get(v, 0) + 1
    valid_ids = sorted(cid for cid in counts if cid >= 0)
    noise = counts.get(-1, 0)
    rows = []
    if noise:
        rows.append(html.Div(f"Noise: {noise:,} shots", style=dict(fontSize="10px", color="#666", marginBottom="4px")))
    rows.append(html.Div("Label clusters:", style=dict(fontSize="10px", color="#888", marginBottom="4px")))
    for cid in valid_ids:
        rows.append(
            html.Div(
                style=dict(display="flex", alignItems="center", gap="6px", marginBottom="4px"),
                children=[
                    html.Span(
                        f"C{cid} ({counts[cid]:,})",
                        style=dict(fontSize="10px", color=ACCENT, minWidth="65px", fontVariantNumeric="tabular-nums"),
                    ),
                    dcc.Input(
                        id={"type": "cluster-name", "index": cid},
                        type="text",
                        placeholder=f"Cluster {cid}",
                        debounce=True,
                        style=dict(
                            backgroundColor="#16213e",
                            color=TEXT,
                            border=BORDER,
                            padding="3px 6px",
                            fontSize="11px",
                            width="130px",
                            borderRadius="4px",
                            outline="none",
                        ),
                    ),
                ],
            )
        )
    return rows


@app.callback(
    Output("cluster-names", "data"),
    Input({"type": "cluster-name", "index": ALL}, "value"),
    State("cluster-labels", "data"),
    prevent_initial_call=True,
)
def update_cluster_names(name_values, cluster_labels):
    if not cluster_labels:
        return {}
    valid_ids = sorted(cid for cid in set(cluster_labels.values()) if cid >= 0)
    return {str(cid): (name_values[i] or f"Cluster {cid}") for i, cid in enumerate(valid_ids) if i < len(name_values)}


@app.callback(
    Output("centroid-data", "data"),
    Output("cluster-traces-plot", "className", allow_duplicate=True),
    Input("cluster-representatives", "data"),
    Input("compute-centroid-btn", "n_clicks"),
    Input("cfg-signals", "data"),
    Input("cfg-time-window", "data"),
    Input("cfg-timebase-hz", "data"),
    Input("cfg-backend-options", "data"),
    prevent_initial_call=True,
)
def compute_centroid_data(cluster_representatives, _btn, cfg_signals, cfg_time_window, cfg_timebase, cfg_options):
    if not cluster_representatives:
        return None, ""
    return (
        _load_cluster_representative_traces(
            cluster_representatives, cfg_signals, cfg_time_window, cfg_timebase, cfg_options
        ),
        "",
    )


@app.callback(
    Output("cluster-traces-plot", "figure"),
    Output("centroid-status", "children"),
    Input("centroid-data", "data"),
    Input("cluster-names", "data"),
    Input("cfg-signals", "data"),
)
def render_centroid_fig(centroid_data, cluster_names, cfg_signals):
    if not centroid_data:
        if not SHOW_TRACES:
            return empty_traces_fig("No data directory — pass --data-dir to enable time traces"), ""
        return empty_traces_fig("Run clustering to compute centroid traces"), ""
    fig = _render_centroid_fig(centroid_data, cluster_names or {}, cfg_signals)
    n = len(centroid_data)
    return fig, f"Centroid traces · {n} cluster(s)"


@app.callback(
    Output("table-download", "data"),
    Input("download-table-btn", "n_clicks"),
    State("cluster-labels", "data"),
    State("cluster-names", "data"),
    State("class-labels", "data"),
    State("class-proba", "data"),
    State("class-model", "data"),
    State("dataset-key", "data"),
    prevent_initial_call=True,
)
def download_table(n_clicks, cluster_labels, cluster_names, class_labels, class_proba, class_model, variable):
    ds = get_dataset(variable)
    if ds is None:
        return dash.no_update
    export = ds.df[_table_cols].copy()
    if cluster_labels:
        label_map = {int(k): v for k, v in cluster_labels.items()}
        export["cluster_id"] = export["shot_id"].map(label_map)
        names = cluster_names or {}

        def _cname(cid):
            if pd.isna(cid):
                return ""
            cid = int(cid)
            return names.get(str(cid)) or (f"Cluster {cid}" if cid >= 0 else "Noise")

        export["cluster_name"] = export["cluster_id"].apply(_cname)
    if class_labels:
        export["label"] = export["shot_id"].map({int(k): v for k, v in class_labels.items()})
    # The probabilities go out too: a decision surface read off the screen is
    # not a number anyone can work with afterwards.
    if class_proba and class_model:
        classes = class_model.get("classes") or []
        values = np.asarray(class_proba["values"], dtype=float)
        index = [int(sid) for sid in class_proba["shot_ids"]]
        for i, name in enumerate(classes):
            export[f"p_{name}"] = export["shot_id"].map(pd.Series(values[:, i], index=index))
    return dcc.send_data_frame(export.to_csv, "niceshot_export.csv", index=False)


# ---------------------------------------------------------------------------
# Parameter visibility callbacks
# ---------------------------------------------------------------------------


@app.callback(
    Output("cluster-n-block", "style"),
    Output("cluster-eps-block", "style"),
    Output("cluster-min-samples-block", "style"),
    Input("cluster-algorithm", "value"),
)
def toggle_cluster_params(algorithm):
    if algorithm == "dbscan":
        return _HIDE, _SHOW, _SHOW
    return _SHOW, _HIDE, _HIDE  # kmeans / agglomerative


@app.callback(
    Output("outlier-n-neighbors-block", "style"),
    Input("outlier-algorithm", "value"),
)
def toggle_outlier_params(algorithm):
    return _SHOW if algorithm == "lof" else _HIDE


@app.callback(
    Output("cluster-features-row", "style"),
    Input("cluster-use-projection", "value"),
)
def toggle_cluster_features_row(use_proj):
    return _HIDE if use_proj else dict(marginBottom="6px")


@app.callback(
    Output("outlier-features-row", "style"),
    Input("outlier-use-projection", "value"),
)
def toggle_outlier_features_row(use_proj):
    return _HIDE if use_proj else dict(marginBottom="6px")


# ---------------------------------------------------------------------------
# Outlier detection callbacks
# ---------------------------------------------------------------------------


@app.callback(
    Output("outlier-labels", "data"),
    Output("outlier-status", "children"),
    Output("umap-color-col", "value", allow_duplicate=True),
    Output("pair-color-col", "value", allow_duplicate=True),
    Output("outlier-traces-plot", "className", allow_duplicate=True),
    Input("run-outlier-btn", "n_clicks"),
    State("outlier-algorithm", "value"),
    State("outlier-features", "value"),
    State("outlier-contamination", "value"),
    State("outlier-n-neighbors", "value"),
    State("outlier-use-projection", "value"),
    State("dataset-key", "data"),
    prevent_initial_call=True,
)
def run_outlier_detection(n_clicks, algorithm, features, contamination, n_neighbors, use_projection, variable):
    # The trailing "" output is a dummy write to a prop of the dcc.Loading-wrapped
    # outlier-traces-plot: dcc.Loading only detects callbacks whose Output lands
    # directly on one of its children, not further up a chained-callback graph, so
    # this keeps the spinner showing for this (slow) first hop of that chain too.
    ds = get_dataset(variable)
    if ds is None:
        return dash.no_update, SELECT_VARIABLE_MSG, dash.no_update, dash.no_update, ""
    active_features = ["umap_x", "umap_y"] if use_projection else list(features or [])
    if not active_features:
        return dash.no_update, "Select at least one feature", dash.no_update, dash.no_update, ""
    try:
        labels = _run_outlier_detection(
            ds.df,
            algorithm=algorithm or "isoforest",
            features=active_features,
            contamination=float(contamination or 0.1),
            n_neighbors=int(n_neighbors or 20),
        )
    except Exception as exc:
        log.error("[outliers] %s", exc)
        return dash.no_update, f"Error: {exc}", dash.no_update, dash.no_update, ""
    if not labels:
        return None, "No shots processed — check features", dash.no_update, dash.no_update, ""
    n_out = sum(v for v in labels.values())
    pct = 100 * n_out / len(labels)
    msg = f"{n_out:,} outliers ({pct:.1f}%) across {len(labels):,} shots"
    return labels, msg, _OUTLIER_COLOR_VALUE, _OUTLIER_COLOR_VALUE, ""


@app.callback(
    Output("outlier-traces-data", "data"),
    Output("outlier-traces-plot", "className", allow_duplicate=True),
    Input("outlier-labels", "data"),
    Input("cfg-signals", "data"),
    Input("cfg-time-window", "data"),
    Input("cfg-timebase-hz", "data"),
    Input("cfg-backend-options", "data"),
    prevent_initial_call=True,
)
def compute_outlier_traces(outlier_labels, cfg_signals, cfg_time_window, cfg_timebase, cfg_options):
    return (
        _compute_outlier_traces_data(
            outlier_labels,
            signals=cfg_signals,
            time_window=cfg_time_window,
            timebase_hz=cfg_timebase,
            backend_options=cfg_options,
        ),
        "",
    )


@app.callback(
    Output("outlier-traces-plot", "figure"),
    Output("outlier-traces-status", "children"),
    Input("outlier-traces-data", "data"),
    Input("cfg-signals", "data"),
)
def render_outlier_traces(outlier_traces_data, cfg_signals):
    if not outlier_traces_data:
        if not SHOW_TRACES:
            return (
                empty_traces_fig("No data directory — pass --data-dir to enable time traces"),
                "",
            )
        return empty_traces_fig("Run outlier detection to load sample traces"), ""
    fig = _render_outlier_traces_fig(outlier_traces_data, cfg_signals)
    n = len(outlier_traces_data)
    return fig, f"Showing {n} outlier sample(s)"


# ---------------------------------------------------------------------------
# Classification callbacks
# ---------------------------------------------------------------------------

# Which hyperparameter blocks each model uses. A block not named here is
# hidden rather than removed: a callback cannot reference an id the layout
# does not contain, so every control stays in the tree.
_CLASSIFY_PARAM_BLOCKS = (
    "classify-n-estimators-block",
    "classify-max-depth-block",
    "classify-learning-rate-block",
    "classify-subsample-block",
    "classify-min-samples-leaf-block",
    "classify-balanced-block",
    "classify-gp-kernel-block",
    "classify-gp-length-scale-block",
    "classify-gp-restarts-block",
    "classify-max-train-rows-block",
)
_CLASSIFY_PARAMS_FOR = {
    "gradient_boosting": {
        "classify-n-estimators-block",
        "classify-max-depth-block",
        "classify-learning-rate-block",
        "classify-subsample-block",
    },
    "random_forest": {
        "classify-n-estimators-block",
        "classify-max-depth-block",
        "classify-min-samples-leaf-block",
        "classify-balanced-block",
    },
    "gaussian_process": {
        "classify-gp-kernel-block",
        "classify-gp-length-scale-block",
        "classify-gp-restarts-block",
        "classify-max-train-rows-block",
    },
}


@app.callback(
    [Output(block_id, "style") for block_id in _CLASSIFY_PARAM_BLOCKS],
    Input("classify-algorithm", "value"),
)
def toggle_classify_params(algorithm):
    """Show only the hyperparameters the chosen model actually reads."""
    shown = _CLASSIFY_PARAMS_FOR.get(algorithm or "gradient_boosting", set())
    return [_SHOW if block_id in shown else _HIDE for block_id in _CLASSIFY_PARAM_BLOCKS]


def _classify_params(
    algorithm: str,
    n_estimators,
    max_depth,
    learning_rate,
    subsample,
    min_samples_leaf,
    balanced,
    gp_kernel,
    gp_length_scale,
    gp_restarts,
    max_train_rows,
) -> dict:
    """Collect the controls the chosen model reads into its keyword arguments."""
    if algorithm == "gradient_boosting":
        return {
            "n_estimators": int(n_estimators or 100),
            "max_depth": int(max_depth or 3),
            "learning_rate": float(learning_rate or 0.1),
            "subsample": float(subsample or 1.0),
        }
    if algorithm == "random_forest":
        return {
            "n_estimators": int(n_estimators or 100),
            "max_depth": int(max_depth) if max_depth else None,
            "min_samples_leaf": int(min_samples_leaf or 1),
            "balanced": bool(balanced),
        }
    return {
        "kernel": gp_kernel or "rbf",
        "length_scale": float(gp_length_scale or 1.0),
        "n_restarts_optimizer": int(gp_restarts or 0),
        "max_train_rows": int(max_train_rows or 2000),
    }


def _classify_status(result: dict) -> str:
    """One line describing what the fit produced."""
    metrics = result["metrics"]
    parts = [
        f"{len(result['classes'])} class(es)",
        f"{len(result['labels']):,} shots",
        f"train acc {metrics['train_accuracy']:.2f}",
    ]
    if metrics.get("n_test"):
        parts.append(f"test acc {metrics['test_accuracy']:.2f}")
        parts.append(f"macro F1 {metrics['test_f1']:.2f}")
    else:
        parts.append("no test split")
    if metrics.get("subsampled"):
        parts.append(f"trained on {metrics['subsampled']:,} sampled rows")
    return " · ".join(parts)


@app.callback(
    Output("class-labels", "data"),
    Output("class-proba", "data"),
    Output("class-model", "data"),
    Output("classify-status", "children"),
    Output("umap-color-col", "value", allow_duplicate=True),
    Output("pair-color-col", "value", allow_duplicate=True),
    Input("run-classify-btn", "n_clicks"),
    State("classify-algorithm", "value"),
    State("classify-target", "value"),
    State("classify-features", "value"),
    State("classify-test-fraction", "value"),
    State("classify-seed", "value"),
    State("classify-n-estimators", "value"),
    State("classify-max-depth", "value"),
    State("classify-learning-rate", "value"),
    State("classify-subsample", "value"),
    State("classify-min-samples-leaf", "value"),
    State("classify-balanced", "value"),
    State("classify-gp-kernel", "value"),
    State("classify-gp-length-scale", "value"),
    State("classify-gp-restarts", "value"),
    State("classify-max-train-rows", "value"),
    State("dataset-key", "data"),
    prevent_initial_call=True,
)
def run_classification(
    n_clicks,
    algorithm,
    target,
    features,
    test_fraction,
    seed,
    n_estimators,
    max_depth,
    learning_rate,
    subsample,
    min_samples_leaf,
    balanced,
    gp_kernel,
    gp_length_scale,
    gp_restarts,
    max_train_rows,
    variable,
):
    """Train the chosen model and publish its predictions.

    The features are robustly scaled before the fit, so a column measured in
    megamps and a column measured in per-cent count for the same amount.
    """
    nothing = (dash.no_update,) * 5
    ds = get_dataset(variable)
    if ds is None:
        return (*nothing[:3], SELECT_VARIABLE_MSG, *nothing[3:])
    if not target:
        return (*nothing[:3], "Select a target column", *nothing[3:])
    if not features:
        return (*nothing[:3], "Select at least one feature", *nothing[3:])

    algorithm = algorithm or "gradient_boosting"
    params = _classify_params(
        algorithm,
        n_estimators,
        max_depth,
        learning_rate,
        subsample,
        min_samples_leaf,
        balanced,
        gp_kernel,
        gp_length_scale,
        gp_restarts,
        max_train_rows,
    )
    try:
        fit = _run_classification(
            ds.df,
            algorithm=algorithm,
            features=list(features),
            target=target,
            params=params,
            test_fraction=float(test_fraction if test_fraction is not None else 0.25),
            seed=int(seed if seed is not None else 42),
        )
    except Exception as exc:
        log.error("[classification] %s", exc)
        return (*nothing[:3], f"Error: {exc}", *nothing[3:])

    result = fit["result"]
    # The estimator stays here; the browser only ever sees the token.
    token = uuid.uuid4().hex
    _model_put(
        token,
        {
            "model": fit["model"],
            "X_all": fit["X_all"],
            "X_background": fit["X_background"],
            "row_of": {sid: i for i, sid in enumerate(fit["shot_ids"])},
            "features": result["features"],
            "classes": result["classes"],
            "algorithm": algorithm,
        },
    )
    model_store = {
        "token": token,
        "classes": result["classes"],
        "features": result["features"],
        "target": target,
        "algorithm": algorithm,
    }
    return (
        result["labels"],
        result["proba"],
        model_store,
        _classify_status(result),
        _CLASS_COLOR_VALUE,
        _CLASS_COLOR_VALUE,
    )


@app.callback(
    Output("classify-surface-class", "options"),
    Output("classify-surface-class", "value"),
    Input("class-model", "data"),
)
def update_surface_class_options(class_model):
    """Offer the classes the model learned.

    The last class is the default, so a 0/1 target shows P(1) without the user
    choosing anything -- the reading "probability" has for a binary problem.
    """
    classes = (class_model or {}).get("classes") or []
    if not classes:
        return [], None
    return [{"label": c, "value": c} for c in classes], classes[-1]


# ---------------------------------------------------------------------------
# Correlation callback
# ---------------------------------------------------------------------------


@app.callback(
    Output("corr-plot", "figure"),
    Input("corr-features", "value"),
    Input("active-filters", "data"),
    Input("dataset-key", "data"),
)
def update_correlation(features, active_filters, variable):
    ds = get_dataset(variable)
    if ds is None:
        return _empty_fig(SELECT_VARIABLE_MSG)
    if not features or len(features) < 2:
        return _empty_fig("Select at least 2 features")

    plot_df = _apply_filter_mask(ds.df, active_filters)
    valid = [f for f in features if f in plot_df.columns and pd.api.types.is_numeric_dtype(plot_df[f])]
    if len(valid) < 2:
        return _empty_fig("Need at least 2 numeric features")

    corr = plot_df[valid].corr().fillna(0)
    labels = corr.columns.tolist()
    z = corr.values.tolist()
    text = [[f"{corr.iloc[i, j]:.2f}" for j in range(len(labels))] for i in range(len(labels))]

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=labels,
            y=labels,
            text=text,
            texttemplate="%{text}",
            textfont=dict(size=10),
            colorscale="RdBu_r",
            zmin=-1,
            zmax=1,
            colorbar=dict(
                title=dict(text="r", font=dict(color=TEXT)),
                tickvals=[-1, -0.5, 0, 0.5, 1],
                ticktext=["-1", "-0.5", "0", "0.5", "1"],
                tickfont=dict(color=TEXT),
                bgcolor=PANEL_BG,
                bordercolor="#2a2a4a",
            ),
        )
    )
    fig.update_layout(
        paper_bgcolor=DARK_BG,
        plot_bgcolor="#16213e",
        font=dict(color=TEXT, size=11),
        margin=dict(l=120, r=20, t=40, b=120),
        autosize=True,
        xaxis=dict(tickangle=-45, tickfont=dict(size=10), side="bottom"),
        yaxis=dict(tickfont=dict(size=10), autorange="reversed"),
    )
    return fig


# ---------------------------------------------------------------------------
# Lineage tab callbacks — registered only when a reference column exists.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _LineageView:
    """Everything the Lineage views need, resolved once per callback."""

    ds: Dataset
    subject: int
    lineage: list[int]
    matrix: Any
    columns: list[str]
    compare_cols: list[str]


def _lin_candidates(ds: Dataset | None) -> list[str]:
    """Comparison columns for this dataset, or the startup fallback.

    Taken from the Dataset rather than a module global so long-format mode gets
    the columns of the variable actually loaded.
    """
    if ds is None or not ds.ref_compare_cols:
        return list(_lin_fallback_cols)
    return list(ds.ref_compare_cols)


def _lin_option_pool(candidates: list[str], selected, search_value) -> list[dict]:
    """A searchable slice of *candidates*, never the whole list.

    The current selection is always included. Dash clears a value that is
    absent from ``options``, so leaving it out would wipe the user's picks the
    moment they typed in the box.
    """
    keep = [c for c in (selected or []) if c in set(candidates)]
    if not search_value:
        rest = [c for c in candidates if c not in set(keep)]
        pool = keep + rest[:_LIN_OPTION_LIMIT]
    else:
        query = str(search_value).lower()
        hits = [c for c in candidates if query in c.lower()][:_LIN_OPTION_LIMIT]
        pool = sorted(set(keep) | set(hits))
    return [{"label": c, "value": c} for c in pool]


def _lin_projection_features(ds: Dataset | None) -> list[str]:
    """The projection's own feature columns, in projection order.

    These are the variables the embedding is built from, so they are the ones
    a lineage is worth tracking: they are what makes two shots near or far
    apart on the scatter plot. Filtered against the comparison candidates,
    because a feature the tab cannot compare must not be offered.
    """
    candidates = set(_lin_candidates(ds))
    features = list(ds.search_cols) if ds is not None and ds.search_cols else list(UMAP_FEATURES or numeric_cols)
    return [c for c in features if c in candidates]


def _lin_resolve(
    variable,
    subject,
    scope,
    metric,
    respect,
    active_filters,
    columns=None,
) -> tuple[_LineageView | None, str | None]:
    """Resolve the lineage and change matrix, or explain why there is none.

    Pass ``columns=None`` to compare every candidate column — the ranked card
    does that, so its "N of M changed" tally describes the whole table rather
    than whichever subset the user happens to have selected.
    """
    ds = get_dataset(variable)
    if ds is None:
        return None, SELECT_VARIABLE_MSG
    # The whole tab needs a reference column. Its callbacks are registered
    # whether one is configured or not, because Dash cannot add a callback after
    # the app starts and the Configuration tab can set the column while the app
    # runs. Stopping here keeps them from doing the work for a lineage that
    # cannot exist, and tells the user what is missing.
    if not ds.ref_adjacency:
        return None, _LIN_NO_REFERENCE_MSG
    if subject is None:
        return None, "Select a shot, or wait for the first shot to load"
    subject = int(subject)
    if not (ds.df["shot_id"].astype(int) == subject).any():
        return None, f"Shot {subject} is not in the loaded data"
    if subject not in ds.ref_adjacency:
        return None, f"Shot {subject} has no reference shot — nothing to compare"

    # Filters never drop a lineage shot: removing one would silently turn
    # "change vs the previous shot" into a comparison between two shots that
    # were never linked. Opting in marks them instead.
    restrict = set(active_filters) if ("respect" in (respect or []) and active_filters is not None) else None
    lineage, excluded = get_reference_lineage(
        subject,
        ds.ref_parent,
        ds.ref_adjacency,
        scope=scope or "chain",
        restrict_to=restrict,
    )
    if len(lineage) <= 1:
        return None, f"Shot {subject} is the only shot in this lineage"

    compare = _lin_candidates(ds)
    if columns is None:
        chosen = compare
    else:
        available = set(compare)
        chosen = [c for c in (columns or []) if c in available]
        if not chosen:
            return None, "Select at least one variable"

    matrix = lineage_change_matrix(
        ds.df,
        lineage,
        chosen,
        metric=metric or "zscore",
        stds=ds.ref_stds,
        numeric_cols=list(ds.ref_numeric_cols) or None,
        excluded=excluded,
    )
    return _LineageView(ds, subject, lineage, matrix, chosen, compare), None


def _lin_card_ranked_columns(variable, subject, scope, metric, limit: int | None = None) -> list[str]:
    """The numeric variables in the order the summary card lists them.

    Reads the card's own ranking rather than a second one, so a view seeded
    from it opens on the variables the card puts at the top. Text columns are
    left out: they rank in the card but cannot be drawn as a line.
    """
    view, _message = _lin_resolve(variable, subject, scope, metric, [], None)
    if view is None:
        return []
    ranked = rank_lineage_changes(view.matrix, top_n=None, changed_only=False)
    numeric = [item.column for item in ranked if item.kind == "numeric"]
    return numeric[:limit] if limit else numeric


def _lin_default_columns(variable, subject, scope, metric) -> list[str]:
    """The columns that actually changed, most-changed first.

    A static default cannot work here: with hundreds of columns it would mostly
    show variables that did not move, so the first render would fail to answer
    the tab's question.
    """
    view, _ = _lin_resolve(variable, subject, scope, metric, [], None, columns=None)
    if view is None:
        return _lin_candidates(get_dataset(variable))[:_LIN_DEFAULT_N]
    picked = select_changed_columns(view.matrix, max_columns=_LIN_DEFAULT_N)
    return picked or view.compare_cols[:_LIN_DEFAULT_N]


@app.callback(
    Output("lin-subject-shot", "data"),
    Input("selected-shot", "data"),
    Input("latest-shot", "data"),
    Input("dataset-key", "data"),
    Input("dataset-version", "data"),
)
def resolve_lineage_subject(selected_shot, latest_shot, variable, _dataset_version):
    """Pick the shot the Lineage tab describes.

    Its only Output is a store in the root layout, so this fires on initial
    load — a callback whose Output sits inside an unopened tab never would, and
    the tab would then open with no subject.
    """
    ds = get_dataset(variable)
    if ds is None or ds.df.empty:
        return None
    present = set(ds.df["shot_id"].astype(int))
    for candidate in (selected_shot, latest_shot):
        if candidate is not None and int(candidate) in present:
            return int(candidate)
    return None


@app.callback(
    Output("lin-subject-display", "children"),
    Input("lin-subject-shot", "data"),
    Input("lin-scope", "value"),
    Input("dataset-key", "data"),
    Input("dataset-version", "data"),
    State("selected-shot", "data"),
)
def update_lineage_subject_display(subject, scope, variable, _dataset_version, selected_shot):
    ds = get_dataset(variable)
    if ds is None:
        return SELECT_VARIABLE_MSG
    if subject is None:
        return "No shot selected, and no shots loaded yet"
    subject = int(subject)
    origin = "selected" if selected_shot is not None and int(selected_shot) == subject else "latest"
    if subject not in ds.ref_adjacency:
        return f"Subject: shot {subject} ({origin}) — no reference shot"
    lineage, excluded = get_reference_lineage(subject, ds.ref_parent, ds.ref_adjacency, scope=scope or "chain")
    text = f"Subject: shot {subject} ({origin}) — {len(lineage)} shots in lineage"
    reference = ds.ref_parent.get(subject)
    if reference is not None:
        text += f", reference {reference}"
    return text


@app.callback(
    Output("lin-columns-dd", "options"),
    Input("lin-columns-dd", "search_value"),
    Input("dataset-key", "data"),
    State("lin-columns-dd", "value"),
)
def update_lineage_column_options(search_value, variable, selected):
    """Offer a searchable slice of the comparison columns."""
    return _lin_option_pool(_lin_candidates(get_dataset(variable)), selected, search_value)


@app.callback(
    Output("lin-columns-dd", "value"),
    Input("lin-cols-changed-btn", "n_clicks"),
    Input("lin-cols-features-btn", "n_clicks"),
    Input("lin-cols-clear-btn", "n_clicks"),
    Input("lin-subject-shot", "data"),
    Input("lin-scope", "value"),
    Input("dataset-key", "data"),
    State("lin-metric", "value"),
    State("lin-columns-dd", "value"),
)
def seed_lineage_columns(_changed, _features, _clear, subject, scope, variable, metric, current):
    """Seed the selector with every projection feature, and leave a hand-picked selection alone.

    The projection features are the default because they are the variables
    the embedding — and therefore the whole dashboard's notion of "similar
    shot" — is built from: a lineage is worth tracking in exactly those.
    All of them, not a truncated head, because a silently shortened default
    reads as "these are the features".

    Re-seeding on every click would discard the columns the user chose,
    which makes the tab unusable for comparing one variable across shots.
    """
    triggered = dash.ctx.triggered_id
    if triggered == "lin-cols-clear-btn":
        return []
    if triggered == "lin-cols-features-btn":
        return _lin_projection_features(get_dataset(variable))
    if triggered == "lin-cols-changed-btn":
        return _lin_default_columns(variable, subject, scope, metric)
    if current:
        return dash.no_update
    return _lin_projection_features(get_dataset(variable)) or _lin_default_columns(variable, subject, scope, metric)


@app.callback(
    Output("lin-columns-count", "children"),
    Input("lin-columns-dd", "value"),
    Input("dataset-key", "data"),
)
def update_lineage_column_count(columns, variable):
    return f"{len(columns or [])} of {len(_lin_candidates(get_dataset(variable)))} variables"


@app.callback(
    Output("lin-change-cards", "children"),
    Input("lin-subject-shot", "data"),
    Input("lin-scope", "value"),
    Input("lin-card-metric", "value"),
    Input("lin-respect-filters", "value"),
    Input("active-filters", "data"),
    Input("dataset-key", "data"),
    Input("dataset-version", "data"),
)
def update_lineage_cards(subject, scope, card_metric, respect, active_filters, variable, _dataset_version):
    """The summary card: every variable, biggest change first.

    Built over every candidate column rather than the user's selection, so
    it summarises the whole table and the tally underneath means what it
    says. Unchanged variables are kept: they land at the end of the
    ranking, where they answer "did anything else move" without the reader
    having to widen the selection to find out.

    Ranked by its own measure, not the history table's colour metric —
    "what moved most" is a different question, and only a measure that is
    comparable between columns can answer it.
    """
    view, message = _lin_resolve(variable, subject, scope, card_metric, respect, active_filters)
    if view is None:
        return _lin_message(message or "")
    items = rank_lineage_changes(view.matrix, top_n=_LIN_CARD_MAX, changed_only=False)
    return _lin_render_change_cards(view.matrix, items, view.subject, len(view.compare_cols))


@app.callback(
    Output("lin-history-table", "data"),
    Output("lin-history-table", "columns"),
    Output("lin-history-table", "style_data_conditional"),
    Output("lin-history-table", "tooltip_data"),
    Output("lin-history-msg", "children"),
    Input("lin-subject-shot", "data"),
    Input("lin-scope", "value"),
    Input("lin-metric", "value"),
    Input("lin-columns-dd", "value"),
    Input("lin-respect-filters", "value"),
    Input("active-filters", "data"),
    Input("dataset-key", "data"),
    Input("dataset-version", "data"),
)
def update_lineage_history(subject, scope, metric, columns, respect, active_filters, variable, _dataset_version):
    view, message = _lin_resolve(variable, subject, scope, metric, respect, active_filters, columns=columns or [])
    if view is None:
        return [], [], [], [], _lin_message(message or "")

    matrix = view.matrix
    notes = [f"{len(matrix.columns)} variable(s) shown; scroll sideways for more"]
    if matrix.excluded:
        notes.append(f"{len(matrix.excluded)} lineage shot(s) hidden by the active filters, shown in grey")
    if matrix.metric == "absolute":
        notes.append("colour is scaled to this lineage only")
    return (
        _lin_table_data(matrix, matrix.columns, view.subject),
        _lin_table_columns(matrix.columns, matrix.kinds),
        _lin_style_data_conditional(matrix, matrix.columns, view.subject),
        _lin_tooltip_data(matrix, matrix.columns),
        html.Div("  ·  ".join(notes), style=dict(fontSize="10px", color="#888", padding="4px 2px")),
    )


@app.callback(
    Output("lin-notes-panel", "children"),
    Input("lin-subject-shot", "data"),
    Input("lin-scope", "value"),
    Input("dataset-key", "data"),
    Input("dataset-version", "data"),
)
def update_lineage_notes(subject, scope, variable, _dataset_version):
    """The operator's own notes for each shot in the lineage."""
    view, message = _lin_resolve(variable, subject, scope, "zscore", [], None)
    if view is None:
        return _lin_message(message or "")
    note_cols = _lin_note_columns(view.compare_cols, view.ds.ref_numeric_cols)
    return _lin_notes_table(view.ds.df, view.lineage, view.subject, note_cols)


@app.callback(
    Output("lin-tree-plot", "figure"),
    Input("lin-subject-shot", "data"),
    Input("lin-scope", "value"),
    Input("lin-respect-filters", "value"),
    Input("active-filters", "data"),
    Input("dataset-key", "data"),
    Input("dataset-version", "data"),
)
def update_lineage_tree(subject, scope, respect, active_filters, variable, _dataset_version):
    view, message = _lin_resolve(variable, subject, scope, "zscore", respect, active_filters)
    if view is None:
        return _empty_fig(message or "")
    return _lin_tree_fig(view.matrix, view.ds.ref_parent, view.subject)


@app.callback(
    Output("lin-spark-columns", "options"),
    Input("lin-spark-columns", "search_value"),
    Input("dataset-key", "data"),
    State("lin-spark-columns", "value"),
)
def update_lineage_spark_options(search_value, variable, selected):
    """Offer the same searchable column pool as the tab's own selector."""
    return _lin_option_pool(_lin_candidates(get_dataset(variable)), selected, search_value)


@app.callback(
    Output("lin-spark-columns", "value"),
    Output("lin-spark-seeded", "data"),
    Input("lin-spark-top-btn", "n_clicks"),
    Input("lin-subject-shot", "data"),
    Input("lin-scope", "value"),
    Input("lin-card-metric", "value"),
    Input("dataset-key", "data"),
    State("lin-spark-columns", "value"),
    State("lin-spark-seeded", "data"),
)
def seed_lineage_spark_columns(_top, subject, scope, card_metric, variable, current, seeded):
    """Open on the summary card's top variables, and keep a hand-picked set.

    The card is already the answer to "what moved", so the panels start on
    the head of that same ranking and the two views agree on the first
    screen. That default depends on the subject, so it has to be refreshed
    when the subject changes — but only while it is still the default:
    comparing the selection against the last seeded value is what tells a
    selection the app filled in from one somebody chose, and a chosen
    variable must survive a click on another shot. The button re-seeds
    either way.
    """
    if dash.ctx.triggered_id != "lin-spark-top-btn" and current and list(current) != list(seeded or []):
        return dash.no_update, dash.no_update
    picked = _lin_card_ranked_columns(variable, subject, scope, card_metric, _LIN_SPARK_PANELS)
    return picked, picked


@app.callback(
    Output("lin-spark-plot", "figure"),
    Output("lin-spark-msg", "children"),
    Input("lin-subtabs", "value"),
    Input("lin-subject-shot", "data"),
    Input("lin-scope", "value"),
    Input("lin-card-metric", "value"),
    Input("lin-spark-columns", "value"),
    Input("dataset-key", "data"),
    Input("dataset-version", "data"),
)
def update_lineage_sparklines(subtab, subject, scope, card_metric, columns, variable, _dataset_version):
    """One panel per variable chosen for this view, in the card's order.

    The panels are the selection — every variable in it, ordered the way
    the summary card ranks them. Ranking columns out of a view the user
    selected by hand would make it disagree with its own selector, and
    "this one held still" is itself an answer.
    """
    if subtab != "lin-spark":
        return dash.no_update, dash.no_update
    view, message = _lin_resolve(variable, subject, scope, card_metric, [], None, columns=columns or [])
    if view is None:
        return _empty_fig(message or ""), ""
    selected = list(view.matrix.columns)
    ranked = [item.column for item in rank_lineage_changes(view.matrix, top_n=None, changed_only=False)]
    ranked = ranked or selected  # a lineage of one has nothing to rank
    drawn = [c for c in ranked if view.matrix.kinds.get(c) == "numeric"]
    capped = drawn[:_LIN_SPARK_HARD_MAX]
    notes = [f"{len(capped)} of {len(selected)} selected variable(s) drawn"]
    if len(capped) < len(drawn):
        notes.append(f"at most {_LIN_SPARK_HARD_MAX} panels are drawn — narrow the selection")
    n_text = len(selected) - len(drawn)
    if n_text:
        notes.append(f"{n_text} text variable(s) have no sparkline")
    return _lin_spark_fig(view.matrix, capped, view.subject), "  ·  ".join(notes)


@app.callback(
    Output("selected-shot", "data", allow_duplicate=True),
    Input("lin-tree-plot", "clickData"),
    State("dataset-key", "data"),
    prevent_initial_call=True,
)
def select_shot_from_lineage_tree(click_data, variable):
    """Clicking a lineage node selects that shot everywhere else.

    Needs allow_duplicate because update_selected_shot already owns this
    Output, and it cannot take the tree as an Input: that callback is
    registered unconditionally, and Dash rejects an Input naming a
    component that is absent when no reference column is configured.
    """
    ds = get_dataset(variable)
    if ds is None:
        return dash.no_update
    return _extract_shot_id(ds.df, click_data) or dash.no_update


# ---------------------------------------------------------------------------
# Semantic search callbacks
# ---------------------------------------------------------------------------


@app.callback(
    Output("search-query-shot", "value"),
    Input("selected-shot", "data"),
)
def populate_search_from_selection(selected_shot):
    return selected_shot


@app.callback(
    Output("search-results", "data"),
    Output("search-status", "children"),
    Output("search-results-table", "data"),
    Output("search-traces-plot", "className", allow_duplicate=True),
    Input("find-similar-btn", "n_clicks"),
    State("search-query-shot", "value"),
    State("search-k", "value"),
    State("search-features", "value"),
    State("dataset-key", "data"),
    prevent_initial_call=True,
)
def find_similar_shots(_n, query_shot_id, k, features, variable):
    # The trailing "" output is a dummy write to a prop of the dcc.Loading-wrapped
    # search-traces-plot: dcc.Loading only detects callbacks whose Output lands
    # directly on one of its children, not further up a chained-callback graph, so
    # this keeps the spinner showing for this (slow) first hop of that chain too.
    ds = get_dataset(variable)
    if ds is None:
        return None, SELECT_VARIABLE_MSG, [], ""
    if query_shot_id is None:
        return None, "", [], ""

    query_id = int(query_shot_id)
    k = int(k or 10)

    # Find row in the search index
    idx = np.where(ds.search_ids == query_id)[0]
    if len(idx) == 0:
        return None, f"Shot {query_id} not found in search index", [], ""

    # If the user selected different features, rebuild a local index with imputation
    valid_features = [f for f in (features or ds.search_cols) if f in ds.df.columns]
    if valid_features and set(valid_features) != set(ds.search_cols):
        sub = ds.df[["shot_id"] + valid_features].copy()
        sub[valid_features] = sub[valid_features].replace([np.inf, -np.inf], np.nan)
        local_ids = sub["shot_id"].values
        local_X = StandardScaler().fit_transform(
            SimpleImputer(strategy="mean").fit_transform(sub[valid_features].values.astype(float))
        )
        local_nn = NearestNeighbors(metric="euclidean", algorithm="auto").fit(local_X)
        local_idx = np.where(local_ids == query_id)[0]
        if len(local_idx) == 0:
            return None, f"Shot {query_id} not found in index", [], ""
        distances, indices = local_nn.kneighbors(local_X[local_idx], n_neighbors=min(k + 1, len(local_ids)))
        result_ids = [int(local_ids[i]) for i in indices[0] if int(local_ids[i]) != query_id][:k]
        result_scores = [float(d) for i, d in zip(indices[0], distances[0]) if int(local_ids[i]) != query_id][:k]
    else:
        distances, indices = ds.search_nn.kneighbors(ds.search_X[idx], n_neighbors=min(k + 1, len(ds.search_ids)))
        result_ids = [int(ds.search_ids[i]) for i in indices[0] if int(ds.search_ids[i]) != query_id][:k]
        result_scores = [float(d) for i, d in zip(indices[0], distances[0]) if int(ds.search_ids[i]) != query_id][:k]

    table_data = [
        {"shot_id": sid, "rank": rank + 1, "score": score}
        for rank, (sid, score) in enumerate(zip(result_ids, result_scores))
    ]
    status = f"{len(result_ids)} shots similar to shot {query_id}"
    return result_ids, status, table_data, ""


# ---------------------------------------------------------------------------
# Search traces callbacks
# ---------------------------------------------------------------------------


@app.callback(
    Output("search-traces-data", "data"),
    Output("search-traces-plot", "className", allow_duplicate=True),
    Input("search-results", "data"),
    Input("cfg-signals", "data"),
    Input("cfg-time-window", "data"),
    Input("cfg-timebase-hz", "data"),
    Input("cfg-backend-options", "data"),
    prevent_initial_call=True,
)
def compute_search_traces(search_results, cfg_signals, cfg_time_window, cfg_timebase, cfg_options):
    return (
        _load_shots_traces(
            search_results or [],
            signals=cfg_signals,
            time_window=cfg_time_window,
            timebase_hz=cfg_timebase,
            backend_options=cfg_options,
        ),
        "",
    )


@app.callback(
    Output("search-traces-plot", "figure"),
    Output("search-traces-status", "children"),
    Input("search-traces-data", "data"),
    Input("cfg-signals", "data"),
)
def render_search_traces(search_traces_data, cfg_signals):
    if not search_traces_data:
        if not SHOW_TRACES:
            return (
                empty_traces_fig("No data directory — pass --data-dir to enable time traces"),
                "",
            )
        return empty_traces_fig("Select a shot to load similar traces"), ""
    fig = _render_outlier_traces_fig(search_traces_data, cfg_signals)
    return fig, f"Traces for {len(search_traces_data)} similar shot(s)"


# ---------------------------------------------------------------------------
# Configuration tab callbacks
#
# These are registered whether or not the time-trace pane is available, so the
# tab always shows the active configuration.
# ---------------------------------------------------------------------------


@app.callback(
    Output("lineage-tab", "disabled"),
    Input("dataset-key", "data"),
)
def update_reference_feature_visibility(key):
    """Show or hide the Lineage tab.

    Always in the layout, so this only changes whether it is visible: the tab
    carries a ``disabled_style`` that hides it, so setting ``disabled`` takes
    it out of the tab bar. That is what lets the Configuration tab switch the
    feature on without a restart: a callback cannot be registered after the
    app starts, but a ``disabled`` flag can be written at any time.
    """
    ds = get_dataset(key)
    available = bool(ds is not None and ds.ref_adjacency)
    return not available


@app.callback(
    Output("dataset-key", "data"),
    Input("selected-variable", "data"),
    Input("cfg-projection", "data"),
    Input("cfg-reference-shot-col", "data"),
)
def compute_dataset_key(variable, projection, reference):
    """Combine the selected variable and the applied settings into one key.

    Every data callback depends on this single value, so a settings change fans
    out once. Anything not applied in this browser falls back to the config
    file, which is what makes a page that never opens the tab behave as before.
    """
    return _dataset_key_from_stores(variable, projection, reference).to_store()


@app.callback(
    Output("cfg-signals", "data"),
    Output("cfg-time-window", "data"),
    Output("cfg-timebase-hz", "data"),
    Output("cfg-backend-options", "data"),
    Output("cfg-projection", "data"),
    Output("cfg-reference-shot-col", "data"),
    Output("cfg-status", "children"),
    Input("cfg-apply-btn", "n_clicks"),
    State("cfg-signal-select", "value"),
    State("cfg-min-time", "value"),
    State("cfg-max-time", "value"),
    State("cfg-timebase-hz-input", "value"),
    State({"type": "cfg-opt-key", "index": ALL}, "value"),
    State({"type": "cfg-opt-val", "index": ALL}, "value"),
    State("cfg-projection-method", "value"),
    State("cfg-n-components", "value"),
    State("cfg-random-state", "value"),
    State("cfg-n-neighbors", "value"),
    State("cfg-min-dist", "value"),
    State("cfg-metric", "value"),
    State("cfg-umap-features", "value"),
    State("cfg-umap-exclude-features", "value"),
    State("cfg-reference-shot-col-select", "value"),
    prevent_initial_call=True,
)
def apply_config(
    _n_clicks,
    signals,
    min_time,
    max_time,
    timebase_hz,
    option_keys,
    option_values,
    projection_method,
    n_components,
    random_state,
    n_neighbors,
    min_dist,
    metric,
    umap_features,
    umap_exclude_features,
    reference_shot_col,
):
    """Publish every setting in the tab. A bad value changes nothing.

    Validation is left to the pydantic models in config_schema.py, which hold
    the only definition of each rule, and their message is shown as written. So
    a rule is never stated twice, and the tab cannot drift from the config file.
    """
    nothing = (dash.no_update,) * 6

    selected = [s.strip() for s in (signals or []) if s and s.strip()]
    if not selected:
        return (*nothing, "Select at least one signal.")
    if min_time is None or max_time is None:
        return (*nothing, "Give a number for min_time and for max_time.")

    try:
        window = TimeWindow(min_time=min_time, max_time=max_time)
        options = ProjectionOptions(
            n_components=n_components,
            random_state=random_state,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
        )
    except ValidationError as exc:
        return (*nothing, exc.errors()[0]["msg"].removeprefix("Value error, "))

    backend_options, bad_option = _collect_backend_options(option_keys, option_values)
    if bad_option:
        return (*nothing, bad_option)

    return (
        selected,
        {"min_time": window.min_time, "max_time": window.max_time},
        {"value": timebase_hz},
        {"value": backend_options},
        {
            "projection_method": projection_method or PROJECTION_METHOD,
            "umap_features": [c for c in (umap_features or []) if c],
            "umap_exclude_features": [c for c in (umap_exclude_features or []) if c],
            "projection_options": options.model_dump(),
        },
        {"value": reference_shot_col or None},
        _apply_summary(selected, window, options, projection_method or PROJECTION_METHOD),
    )


def _collect_backend_options(keys, values) -> tuple[dict[str, str], str | None]:
    """Read the key/value grid. Returns the options and an error, if any.

    Values stay strings, exactly as ``--backend-option KEY=VALUE`` leaves them,
    so a backend reads them the same way whichever route they came in by.
    """
    options: dict[str, str] = {}
    for key, value in zip(keys or [], values or []):
        name = (key or "").strip()
        if not name:
            continue
        if name in options:
            return {}, f"The backend option '{name}' is given more than once."
        options[name] = (value or "").strip()
    return options, None


def _apply_summary(signals, window, options, method) -> str:
    """One line describing what was applied."""
    parts = [
        f"{len(signals)} signal(s)",
        f"{window.min_time}\u2013{window.max_time} s",
        method.upper(),
    ]
    if options.n_components != 2:
        parts.append(f"{options.n_components} components")
    return "Applied: " + ", ".join(parts) + "."


@app.callback(
    Output("cfg-signals", "data", allow_duplicate=True),
    Output("cfg-time-window", "data", allow_duplicate=True),
    Output("cfg-timebase-hz", "data", allow_duplicate=True),
    Output("cfg-backend-options", "data", allow_duplicate=True),
    Output("cfg-projection", "data", allow_duplicate=True),
    Output("cfg-reference-shot-col", "data", allow_duplicate=True),
    Output("cfg-status", "children", allow_duplicate=True),
    Output("cfg-signal-select", "value"),
    Output("cfg-min-time", "value"),
    Output("cfg-max-time", "value"),
    Output("cfg-timebase-hz-input", "value"),
    Output("cfg-projection-method", "value"),
    Output("cfg-n-components", "value"),
    Output("cfg-random-state", "value"),
    Output("cfg-n-neighbors", "value"),
    Output("cfg-min-dist", "value"),
    Output("cfg-metric", "value"),
    Output("cfg-umap-features", "value"),
    Output("cfg-umap-exclude-features", "value"),
    Output("cfg-reference-shot-col-select", "value"),
    Output({"type": "cfg-opt-key", "index": ALL}, "value"),
    Output({"type": "cfg-opt-val", "index": ALL}, "value"),
    Input("cfg-reset-btn", "n_clicks"),
    prevent_initial_call=True,
)
def reset_config(_n_clicks):
    """Put every setting, and every widget, back to the config file.

    The stores go back to None rather than to the config values: None means
    "nothing applied in this browser", which is what a reset is, and it keeps
    a cleared setting from being remembered as an override.
    """
    # One value per rendered row. A backend that reads no options renders none,
    # and Dash requires the list to match what the ALL pattern matched.
    blank = [""] * _BACKEND_OPTION_ROWS
    return (
        None,
        None,
        None,
        None,
        None,
        None,
        "Reset to the config file.",
        list(TIME_TRACE_SIGNALS),
        MIN_TIME,
        MAX_TIME,
        UDA_TIMEBASE_HZ,
        PROJECTION_METHOD,
        PROJECTION_OPTIONS.n_components,
        PROJECTION_OPTIONS.random_state,
        PROJECTION_OPTIONS.n_neighbors,
        PROJECTION_OPTIONS.min_dist,
        PROJECTION_OPTIONS.metric,
        list(UMAP_FEATURES) if UMAP_FEATURES else [],
        list(UMAP_EXCLUDE_FEATURES),
        REFERENCE_SHOT_COL,
        blank,
        blank,
    )


@app.callback(
    Output("cfg-discovered-signals", "data"),
    Output("cfg-discover-status", "children"),
    Input("cfg-discover-btn", "n_clicks"),
    State("selected-shot", "data"),
    prevent_initial_call=True,
)
def discover_signals(_n_clicks, shot_id):
    """Ask the backend which signals it has for the selected shot."""
    if shot_id is None:
        return dash.no_update, "Select a shot first."
    try:
        found = _trace_backend.available_signals(int(shot_id))
    except Exception as exc:
        log.error("[discover_signals] shot %s: %s", shot_id, exc)
        return dash.no_update, f"Could not list signals: {exc}"
    if not found:
        return dash.no_update, f"The '{BACKEND}' backend cannot list its signals — type the names."
    return found, f"Found {len(found)} signal(s) in shot {shot_id}."


@app.callback(
    Output("cfg-signal-select", "options"),
    Input("cfg-signal-select", "search_value"),
    Input("cfg-discovered-signals", "data"),
    State("cfg-signal-select", "value"),
)
def update_signal_options(search_value, discovered, selected):
    """Build the dropdown options.

    dcc.Dropdown has no free-entry mode, so whatever the user types is added as
    an option. The selected values are always kept: Dash clears a value that
    has no matching option.
    """
    options = set(discovered or []) | set(selected or []) | set(TIME_TRACE_SIGNALS)
    if search_value:
        options.add(search_value.strip())
    return [{"label": s, "value": s} for s in sorted(options) if s]


@app.callback(
    Output("active-signals-display", "children"),
    Output("active-time-display", "children"),
    Input("cfg-signals", "data"),
    Input("cfg-time-window", "data"),
    Input("cfg-timebase-hz", "data"),
    Input("cfg-backend-options", "data"),
)
def update_config_summary(signals, time_window, timebase, _options):
    """Keep the line above the time-trace pane the same as the applied values."""
    names = _effective_signals(signals)
    min_time, max_time = _effective_window(time_window)
    time_text = f"time: {min_time}–{max_time} s"
    if timebase is not None and timebase.get("value"):
        time_text += f" @ {timebase['value']} Hz"
    return f"signals: {', '.join(names)}", time_text


@app.callback(
    Output("cfg-umap-block", "style"),
    Input("cfg-projection-method", "value"),
)
def toggle_projection_umap_block(method):
    """Hide the UMAP-only parameters while the method is PCA.

    The widgets stay in the tree and keep their values, so a change back to
    UMAP uses what was set before.
    """
    return dict(marginTop="10px", **(_HIDE if method == "pca" else _SHOW))


@app.callback(
    Output("cfg-backend-option-hint", "children"),
    Input("cfg-apply-btn", "n_clicks"),
)
def update_backend_option_hint(_n_clicks):
    """Name the options the active backend reads, so the grid is not a guess.

    A backend with no documented options gets no line at all. Saying that it
    has none tells the user nothing they can act on.
    """
    known = _BACKEND_OPTION_NAMES.get(BACKEND)
    if not known:
        return ""
    return f"The '{BACKEND}' backend reads: {', '.join(known)}."


# The documented option names per backend, from docs/configuration.md. Used only
# as a hint in the UI; an unlisted key is still passed through, because a plugin
# backend can read anything.
_BACKEND_OPTION_NAMES: dict[str, tuple[str, ...]] = {
    "postgres": ("dsn", "trace_table", "shot_table", "schema", "shot_col", "time_col"),
    "fairmast": ("format", "storage_options"),
}


@app.callback(
    # Shared with reset_config, which also refills this selector.
    Output("cfg-umap-features", "value", allow_duplicate=True),
    Output("cfg-features-paste-status", "children"),
    Input("cfg-features-paste-btn", "n_clicks"),
    State("cfg-features-paste", "value"),
    State("cfg-umap-features", "options"),
    prevent_initial_call=True,
)
def use_pasted_features(_n_clicks, pasted, options):
    """Put a pasted list of names into the feature selector.

    Choosing 192 latent dimensions one at a time is not usable, and the
    dropdown only renders the first few hundred names, so a long list has to
    arrive some other way. Names not in the table are reported rather than
    dropped in silence, because a typo here quietly changes the projection.
    """
    names = [n.strip() for n in re.split(r"[,\n\r\t;]+", pasted or "") if n.strip()]
    if not names:
        return dash.no_update, "Paste one or more column names first."

    known = set(_cfg_feature_pool)
    chosen = [n for n in dict.fromkeys(names) if n in known]
    missing = [n for n in dict.fromkeys(names) if n not in known]
    if not chosen:
        return dash.no_update, f"None of those {len(names)} name(s) is a usable feature column."

    status = f"Selected {len(chosen)} column(s)."
    if missing:
        shown = ", ".join(missing[:5])
        more = f" and {len(missing) - 5} more" if len(missing) > 5 else ""
        status += f" Not usable as features: {shown}{more}."
    # The selected values must also be options, or Dash drops them.
    return chosen, status


@app.callback(
    Output("cfg-umap-features", "options"),
    Input("cfg-umap-features", "value"),
    Input("cfg-umap-features", "search_value"),
)
def update_feature_options(selected, search_value):
    """Keep the selected names as options.

    Dash clears a value that has no matching option, and the list is capped, so
    a name chosen from a paste or from the config file has to be added back.
    """
    return _cfg_column_options(_cfg_feature_pool, selected, search_value)


@app.callback(
    Output("cfg-umap-exclude-features", "options"),
    Input("cfg-umap-exclude-features", "value"),
    Input("cfg-umap-exclude-features", "search_value"),
)
def update_exclude_feature_options(selected, search_value):
    return _cfg_column_options(_cfg_feature_pool, selected, search_value)


@app.callback(
    Output("cfg-reference-shot-col-select", "options"),
    Input("cfg-reference-shot-col-select", "value"),
    Input("cfg-reference-shot-col-select", "search_value"),
)
def update_reference_column_options(selected, search_value):
    return _cfg_column_options(all_cols, [selected] if selected else [], search_value)


@app.callback(
    Output("refresh-interval", "interval"),
    Output("refresh-interval", "disabled"),
    Output("cfg-refresh-interval", "value"),
    Input("cfg-apply-btn", "n_clicks"),
    State("cfg-refresh-interval", "value"),
)
def update_refresh_interval(_n_clicks, seconds):
    """Retime the live-update poll.

    The dcc.Interval is already in the layout, so this only rewrites its props:
    an empty value stops the poll, and any positive number restarts it. A value
    that AppConfig would reject leaves the interval alone.
    """
    if seconds is None:
        return _REFRESH_INTERVAL_OFF_MS, True, None
    try:
        AppConfig(refresh_interval_seconds=seconds)
    except ValidationError:
        return dash.no_update, dash.no_update, dash.no_update
    return int(float(seconds) * 1000), False, seconds


@app.callback(
    Output("cfg-yaml-block", "style"),
    Output("cfg-yaml-output", "value"),
    Input("cfg-yaml-btn", "n_clicks"),
    State("cfg-signals", "data"),
    State("cfg-time-window", "data"),
    State("cfg-timebase-hz", "data"),
    State("cfg-backend-options", "data"),
    State("cfg-projection", "data"),
    State("cfg-reference-shot-col", "data"),
    prevent_initial_call=True,
)
def render_config_yaml(_n_clicks, signals, time_window, timebase, options, projection, reference):
    """Show the applied settings as a config file.

    The app never writes a config file: the settings live in this browser, and
    a restart returns to the file. This is how a user keeps what they tuned.
    """
    return _SHOW, _settings_as_yaml(signals, time_window, timebase, options, projection, reference)


@app.callback(
    Output("cluster-labels", "data", allow_duplicate=True),
    Output("cluster-representatives", "data", allow_duplicate=True),
    Output("cluster-names", "data", allow_duplicate=True),
    Output("centroid-data", "data", allow_duplicate=True),
    Output("outlier-labels", "data", allow_duplicate=True),
    Output("outlier-traces-data", "data", allow_duplicate=True),
    Output("class-labels", "data", allow_duplicate=True),
    Output("class-proba", "data", allow_duplicate=True),
    Output("class-model", "data", allow_duplicate=True),
    Output("search-results", "data", allow_duplicate=True),
    Output("search-traces-data", "data", allow_duplicate=True),
    Output("lin-subject-shot", "data", allow_duplicate=True),
    Output("lin-spark-seeded", "data", allow_duplicate=True),
    Input("dataset-key", "data"),
    prevent_initial_call=True,
)
def clear_results_on_new_dataset(_key):
    """Drop the results that the previous projection produced.

    A new key means every point has moved, so a cluster assignment, an outlier
    label, a similar-shot list and a centroid are all answers to a question
    about coordinates that no longer exist. Drawing them against the new
    embedding would be wrong rather than merely stale.

    A trained classifier goes with them: its feature matrix was built from the
    columns of the dataset that has just been replaced, so its predictions
    describe shots as they were measured before, not as they are now.

    ``selected-shot`` and ``active-filters`` are deliberately kept: both hold
    shot ids, and the set of shots does not change with the projection.
    """
    return None, None, {}, None, None, None, None, None, None, None, None, None, None


# ---------------------------------------------------------------------------
# Variable selection
# ---------------------------------------------------------------------------

if VARIABLE_MODE:

    @app.callback(
        Output("selected-variable", "data"),
        Input("variable-select", "value"),
    )
    def select_variable(variable):
        """Publish the chosen variable so every data callback reloads for it."""
        return variable


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if _args.debug:
        log.info("Debug mode: starting Flask development server (single process)")
        app.run(debug=True, host=_args.host, port=_args.port, use_reloader=False)
        return

    from gunicorn.app.base import BaseApplication

    class _StandaloneApp(BaseApplication):
        def load_config(self):
            assert self.cfg is not None
            self.cfg.set("bind", f"{_args.host}:{_args.port}")
            self.cfg.set("workers", _args.workers)
            self.cfg.set("preload_app", True)
            self.cfg.set("timeout", 120)
            self.cfg.set("loglevel", "info")

        def load(self):
            return server

    _StandaloneApp().run()


if __name__ == "__main__":
    main()
