"""
NiceShot!
Run from project root: uv run python nice_shot/app.py
"""

import argparse
import hashlib
import logging
import os
import sys
import yaml
from pathlib import Path
import numpy as np
import pandas as pd
import duckdb
import dash
from dash import dcc, html, dash_table, Input, Output, State, ALL, MATCH
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
    _CACHE_DIR = (
        Path(os.environ.get("LOCALAPPDATA", Path.home())) / "niceshot" / "cache"
    )
else:
    _CACHE_DIR = (
        Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "niceshot"
    )
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, _HERE)
from config_schema import AppConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="niceshot",
        description="NiceShot! — interactive tokamak shot dashboard",
    )
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=8050, help="Port to listen on (default: 8050)"
    )
    parser.add_argument(
        "--debug",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable Dash debug mode (default: on)",
    )
    parser.add_argument(
        "--config",
        default=os.path.join(_HERE, "config.yaml"),
        metavar="PATH",
        help="Path to config YAML (default: nice_shot/config.yaml)",
    )
    parser.add_argument(
        "--shot-data",
        default=os.path.join(_ROOT, "outputs", "shot_stats.parquet"),
        metavar="PATH",
        help="Path to shot data file (.parquet or .csv)",
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
with open(_args.config) as f:
    _cfg = AppConfig.model_validate(yaml.safe_load(f) or {})

BACKEND: str = _cfg.backend
TIME_TRACE_SIGNALS: list[str] = _cfg.signals
MIN_TIME: float = _cfg.time_window.min_time
MAX_TIME: float = _cfg.time_window.max_time
UDA_TIMEBASE_HZ: float | None = _cfg.uda.timebase_hz
PROJECTION_METHOD: str = _cfg.projection_method
UMAP_FEATURES: list[str] | None = _cfg.umap_features
REFERENCE_SHOT_COL: str | None = _cfg.reference_shot_col

# Time-trace panel is shown only when a data source is actually configured.
# UDA/SAL: always enabled (live connection defined in config).
# parquet: enabled only when --data-dir exists and is non-empty.
if BACKEND in ("uda", "sal"):
    SHOW_TRACES = True
else:
    SHOW_TRACES = os.path.isdir(MASTU_DATA_DIR) and bool(os.listdir(MASTU_DATA_DIR))

if not SHOW_TRACES:
    log.info(
        "Time-trace panel greyed out — backend='%s', "
        "data-dir '%s' not found or empty.",
        BACKEND,
        MASTU_DATA_DIR,
    )

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

_SHOT_ID_CANDIDATES = [
    "shot_id",
    "shot",
    "pulse",
    "number",
    "exp_number",
    "pulse_id",
    "shot_number",
]


def _detect_shot_col(frame: pd.DataFrame) -> str:
    for candidate in _SHOT_ID_CANDIDATES:
        if candidate in frame.columns:
            return candidate
    raise ValueError(
        f"Could not detect shot ID column. "
        f"Expected one of {_SHOT_ID_CANDIDATES}. Found: {list(frame.columns)}"
    )


def _load_shot_data(path: str) -> pd.DataFrame:
    log.info("Loading %s...", path)
    _ext = os.path.splitext(path)[1].lower()
    if _ext == ".csv":
        _df = pd.read_csv(path)
    elif _ext in (".parquet", ".pq"):
        _df = pd.read_parquet(path)
    else:
        raise ValueError(
            f"Unsupported shot stats format '{_ext}' — expected .csv or .parquet"
        )

    # Coerce object columns to numeric where possible; leave non-numeric columns as-is.
    _obj_cols = _df.select_dtypes(include="object").columns
    if len(_obj_cols):
        _coerced = _df[_obj_cols].apply(pd.to_numeric, errors="coerce")
        _converted = [c for c in _obj_cols if _coerced[c].notna().any()]
        if _converted:
            log.info(
                "Coerced %d object column(s) to numeric: %s",
                len(_converted),
                _converted,
            )
        _df[_obj_cols] = _coerced

    _shot_col = _detect_shot_col(_df)
    if _shot_col != "shot_id":
        log.info("Renaming shot ID column '%s' -> 'shot_id'", _shot_col)
        _df = _df.rename(columns={_shot_col: "shot_id"})

    return _df


df = _load_shot_data(SHOT_DATA_PATH)

# Build positional index for SHAP lookup before the UMAP merge drops rows.
# The .nc file uses 0-based indices matching the original sorted shot order.
_shot_to_shap_idx: dict[int, int] = {
    int(s): i for i, s in enumerate(df["shot_id"].values) if pd.notna(s)
}

numeric_cols = sorted(
    c for c in df.select_dtypes(include=[np.number]).columns if c != "shot_id"
)
all_cols = sorted(c for c in df.columns if c != "shot_id")

# ---------------------------------------------------------------------------
# UMAP
# ---------------------------------------------------------------------------


def _projection_feature_cols(data: pd.DataFrame) -> list[str]:
    if UMAP_FEATURES:
        missing = [c for c in UMAP_FEATURES if c not in data.columns]
        if missing:
            log.warning("[projection] umap_features not found in data: %s", missing)
        return [c for c in UMAP_FEATURES if c in data.columns]
    return [
        c for c in data.select_dtypes(include=[np.number]).columns if c != "shot_id"
    ]


def _compute_projection(data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return (projection, shot_ids) for rows without NaN in the feature columns."""
    from sklearn.preprocessing import StandardScaler

    tag = PROJECTION_METHOD.upper()
    feature_cols = _projection_feature_cols(data)
    log.info(
        "[%s] %d feature columns: %s%s",
        tag,
        len(feature_cols),
        feature_cols[:15],
        "..." if len(feature_cols) > 15 else "",
    )

    if not feature_cols:
        raise ValueError(
            f"No usable feature columns found for {tag}. "
            "Check umap_features in config or that the file has numeric columns."
        )

    X = data[feature_cols].copy()

    # Drop columns that are entirely NaN — they carry no information.
    all_nan_cols = X.columns[X.isna().all()].tolist()
    if all_nan_cols:
        log.info(
            "[%s] Dropping %d all-NaN columns: %s", tag, len(all_nan_cols), all_nan_cols
        )
        X = X.drop(columns=all_nan_cols)

    if X.shape[1] == 0:
        raise ValueError(
            "All feature columns are entirely NaN. "
            "Use 'umap_features' in config to specify columns with data."
        )

    # Coerce to float so np.isfinite can handle all column dtypes.
    X = X.apply(pd.to_numeric, errors="coerce")

    # Drop rows with NaN or inf in any remaining column.
    finite = np.isfinite(X.values)
    valid = finite.all(axis=1)
    n_dropped = int((~valid).sum())
    if n_dropped:
        bad_cols = X.columns[~finite.all(axis=0)].tolist()
        log.warning(
            "[%s] dropping %d / %d shots (%.1f%%) with NaN/inf in: %s",
            tag,
            n_dropped,
            len(data),
            n_dropped / len(data) * 100,
            bad_cols,
        )

    X = X[valid]
    shot_ids = data.loc[valid, "shot_id"].values

    if X.empty:
        raise ValueError(
            f"No shots remain after dropping NaN rows across {X.shape[1]} columns. "
            f"Use 'umap_features' in config to select a smaller set of well-populated columns."
        )

    log.info("[%s] fitting on %d rows x %d columns", tag, X.shape[0], X.shape[1])
    X_scaled = StandardScaler().fit_transform(X)

    if PROJECTION_METHOD == "pca":
        from sklearn.decomposition import PCA

        projection = PCA(n_components=2, random_state=42).fit_transform(X_scaled)
    else:
        from umap import UMAP

        projection = UMAP(n_components=2, random_state=42).fit_transform(X_scaled)

    return projection, shot_ids


def _umap_cache_hash() -> str:
    h = hashlib.md5()
    with open(SHOT_DATA_PATH, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    features_key = ",".join(sorted(UMAP_FEATURES)) if UMAP_FEATURES else "__all__"
    h.update(features_key.encode())
    h.update(PROJECTION_METHOD.encode())
    return h.hexdigest()


def get_projection(data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return (projection, shot_ids), loading from cache when valid."""
    _hash_path = UMAP_CACHE_PATH + ".hash"
    _shots_path = UMAP_CACHE_PATH + ".shots.npy"
    current_hash = _umap_cache_hash()

    if all(os.path.exists(p) for p in [UMAP_CACHE_PATH, _hash_path, _shots_path]):
        with open(_hash_path) as f:
            if f.read().strip() == current_hash:
                log.info("Loading projection from cache: %s", UMAP_CACHE_PATH)
                return np.load(UMAP_CACHE_PATH), np.load(_shots_path)
        log.info("Shot data or config changed — recomputing projection...")
    else:
        log.info(
            "Computing %s projection (this may take a moment)...",
            PROJECTION_METHOD.upper(),
        )

    projection, shot_ids = _compute_projection(data)
    np.save(UMAP_CACHE_PATH, projection)
    np.save(_shots_path, shot_ids.astype(np.int64))
    with open(_hash_path, "w") as f:
        f.write(current_hash)
    log.info("Projection saved to cache: %s", UMAP_CACHE_PATH)
    return projection, shot_ids


def _load_projection_file(path: str) -> tuple[pd.DataFrame, str, str]:
    """Load a pre-computed projection. Returns (df with shot_id/umap_x/umap_y, x_label, y_label)."""
    ext = os.path.splitext(path)[1].lower()

    if ext == ".npy":
        arr = np.load(path)
        if arr.ndim != 2 or arr.shape[1] < 2:
            raise ValueError(
                f"Numpy projection must be 2-D with shape (n, 2) or (n, 3); got {arr.shape}"
            )
        if arr.shape[1] >= 3:
            # First column is shot_id, next two are coordinates.
            result = pd.DataFrame(
                {
                    "shot_id": arr[:, 0].astype(np.int64),
                    "umap_x": arr[:, 1],
                    "umap_y": arr[:, 2],
                }
            )
        else:
            # (n, 2) — row order must match the shot data file.
            log.info(
                "[projection] numpy file has shape %s with no shot_id column; "
                "rows are matched positionally to the shot data file.",
                arr.shape,
            )
            if len(arr) != len(df):
                raise ValueError(
                    f"Numpy projection has {len(arr)} rows but shot data has {len(df)} rows. "
                    f"Provide a (n, 3) array with shot_id as the first column, or use a "
                    f".csv / .parquet file."
                )
            result = pd.DataFrame(
                {
                    "shot_id": df["shot_id"].values,
                    "umap_x": arr[:, 0],
                    "umap_y": arr[:, 1],
                }
            )
        log.info("Loaded numpy projection from %s: %d rows", path, len(result))
        return result, "Dim 1", "Dim 2"

    if ext == ".csv":
        emb = pd.read_csv(path)
    elif ext in (".parquet", ".pq"):
        emb = pd.read_parquet(path)
    else:
        raise ValueError(
            f"Unsupported projection format '{ext}' — expected .npy, .csv, or .parquet"
        )

    shot_col = _detect_shot_col(emb)
    if shot_col != "shot_id":
        emb = emb.rename(columns={shot_col: "shot_id"})

    coord_cols = [c for c in emb.columns if c != "shot_id"]
    if len(coord_cols) < 2:
        raise ValueError(
            f"Projection file must have at least 2 coordinate columns; found: {coord_cols}"
        )
    x_col, y_col = coord_cols[0], coord_cols[1]
    log.info(
        "Loaded projection from %s: %d rows, axes '%s' / '%s'",
        path,
        len(emb),
        x_col,
        y_col,
    )
    result = emb[["shot_id", x_col, y_col]].rename(
        columns={x_col: "umap_x", y_col: "umap_y"}
    )
    return result, x_col, y_col


if PROJECTION_PATH is not None:
    _emb_df, UMAP_X_LABEL, UMAP_Y_LABEL = _load_projection_file(PROJECTION_PATH)
    df = df.merge(_emb_df, on="shot_id", how="inner")
else:
    _projection, _proj_shot_ids = get_projection(df)
    _umap_df = pd.DataFrame(
        {
            "shot_id": _proj_shot_ids,
            "umap_x": _projection[:, 0],
            "umap_y": _projection[:, 1],
        }
    )
    df = df.merge(_umap_df, on="shot_id", how="inner")
    UMAP_X_LABEL, UMAP_Y_LABEL = "Dim 1", "Dim 2"

_table_cols = [c for c in df.columns if c not in ("umap_x", "umap_y")]
_table_column_defs = [
    {"name": c, "id": c, "type": "numeric", "format": {"specifier": ".4g"}}
    if pd.api.types.is_float_dtype(df[c])
    else {"name": c, "id": c}
    for c in _table_cols
]

# ---------------------------------------------------------------------------
# Reference-shot graph
# ---------------------------------------------------------------------------
SHOW_REF_TOGGLE = False
_ref_adjacency: dict[int, list[int]] = {}  # undirected: shot_id → [connected shot_ids]
_ref_parent: dict[int, int] = {}  # directed: shot_id → its reference shot


def _build_reference_graph(
    data: pd.DataFrame, col: str
) -> tuple[dict[int, list[int]], dict[int, int]]:
    adjacency: dict[int, list[int]] = {}
    parent: dict[int, int] = {}
    _pairs = data[["shot_id", col]].copy()
    _pairs[col] = pd.to_numeric(_pairs[col], errors="coerce")
    _pairs = _pairs.dropna(subset=[col]).astype({col: int})
    _valid_shots = set(data["shot_id"].astype(int))
    for shot, ref in zip(_pairs["shot_id"].astype(int), _pairs[col]):
        if shot != ref and ref in _valid_shots:
            parent[shot] = ref
            adjacency.setdefault(shot, []).append(ref)
            adjacency.setdefault(ref, []).append(shot)
    return adjacency, parent


if REFERENCE_SHOT_COL and REFERENCE_SHOT_COL in df.columns:
    _ref_adjacency, _ref_parent = _build_reference_graph(df, REFERENCE_SHOT_COL)
    if _ref_adjacency:
        SHOW_REF_TOGGLE = True
        log.info(
            "Reference graph: '%s' — %d edges, %d unique nodes",
            REFERENCE_SHOT_COL,
            len(_ref_parent),
            len(_ref_adjacency),
        )
    else:
        log.warning(
            "reference_shot_col='%s' produced no valid edges.", REFERENCE_SHOT_COL
        )

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


def find_shot_file(shot_id: int) -> str | None:
    for subdir in sorted(os.listdir(MASTU_DATA_DIR)):
        for ext in (".parquet", ".csv"):
            path = os.path.join(MASTU_DATA_DIR, subdir, f"{int(shot_id)}{ext}")
            if os.path.exists(path):
                return path
    return None


def _load_local(shot_id: int) -> pd.DataFrame | None:
    path = find_shot_file(shot_id)
    if path is None:
        return None
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        result = pd.read_csv(path)
        return result[
            (result["time"] >= MIN_TIME) & (result["time"] <= MAX_TIME)
        ].reset_index(drop=True)
    con = duckdb.connect()
    result = con.execute(
        f"SELECT * FROM '{path}' WHERE time >= {MIN_TIME} AND time <= {MAX_TIME}"
    ).df()
    con.close()
    return result


def _load_remote(shot_id: int, url_fn) -> pd.DataFrame | None:
    """Shared loader for UDA and SAL backends. url_fn(signal, shot_id) -> URL string."""
    import xarray as xr

    engine = "uda" if BACKEND == "uda" else "sal"

    if UDA_TIMEBASE_HZ is not None:
        n = int(round((MAX_TIME - MIN_TIME) * UDA_TIMEBASE_HZ))
        time_ref: np.ndarray | None = np.linspace(MIN_TIME, MAX_TIME, n)
    else:
        time_ref = None

    signal_data: dict[str, np.ndarray] = {}

    for signal in TIME_TRACE_SIGNALS:
        try:
            ds = xr.open_dataset(url_fn(signal, shot_id), engine=engine)
            if time_ref is None:
                time_ref = ds.coords["time"].values.astype(float)
            values = ds["data"].interp(time=time_ref).values
            signal_data[signal] = values
        except Exception as exc:
            log.error(
                "[%s] Could not load '%s' for shot %d: %s",
                engine.upper(),
                signal,
                shot_id,
                exc,
            )

    if time_ref is None:
        return None

    result = pd.DataFrame({"time": time_ref, **signal_data})
    return result[
        (result["time"] >= MIN_TIME) & (result["time"] <= MAX_TIME)
    ].reset_index(drop=True)


def _load_uda(shot_id: int) -> pd.DataFrame | None:
    return _load_remote(shot_id, lambda signal, shot: f"uda://{signal}:{shot}")


def _load_sal(shot_id: int) -> pd.DataFrame | None:
    return _load_remote(shot_id, lambda signal, shot: f"sal://pulse/{shot}/{signal}")


def load_shot_traces(shot_id: int) -> pd.DataFrame | None:
    if BACKEND == "uda":
        return _load_uda(shot_id)
    if BACKEND == "sal":
        return _load_sal(shot_id)
    return _load_local(shot_id)


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


def make_traces_fig(shot_df: pd.DataFrame) -> go.Figure:
    available = [s for s in TIME_TRACE_SIGNALS if s in shot_df.columns]
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


def make_shap_fig(shot_id: int) -> str | None:
    """Return a base64-encoded PNG of the SHAP decision plot for one shot, or None."""
    if _shap_da is None:
        return None
    idx = _shot_to_shap_idx.get(int(shot_id))
    if idx is None:
        return None

    import io
    import base64
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    shap_values = _shap_da.isel(shot_id=idx).sel(**{"class": True}).values

    with plt.style.context("dark_background"):
        plt.rcParams.update({"font.size": 7})
        shap.decision_plot(
            0.0,
            shap_values,
            feature_names=_shap_feature_names,
            show=False,
        )
        fig = plt.gcf()
        fig.set_size_inches(5, 7)
        fig.patch.set_facecolor("#1a1a2e")
        ax = fig.axes[0]
        ax.set_facecolor("#16213e")
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
        plt.savefig(
            buf, format="png", bbox_inches="tight", facecolor="#1a1a2e", dpi=110
        )
        plt.close("all")

    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


# ---------------------------------------------------------------------------
# Reference-graph helpers
# ---------------------------------------------------------------------------


def get_reference_graph(shot_id: int) -> set[int]:
    """BFS over the undirected reference graph — returns the full connected component."""
    if not _ref_adjacency:
        return set()
    visited: set[int] = set()
    queue = [shot_id]
    while queue:
        cur = queue.pop()
        if cur in visited:
            continue
        visited.add(cur)
        queue.extend(n for n in _ref_adjacency.get(cur, []) if n not in visited)
    return visited


def _ref_shot_color(shot_id: int, min_id: int, max_id: int) -> str:
    """Map a shot_id to a Turbo colorscale colour (old=dark blue, new=dark red)."""
    import plotly.colors as pc
    t = (shot_id - min_id) / (max_id - min_id) if max_id > min_id else 0.5
    return pc.sample_colorscale("Turbo", [t])[0]


def _add_reference_graph_overlay(
    fig: go.Figure,
    plot_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    selected_shot: int,
) -> go.Figure:
    """Add edge lines and node markers for the reference graph of selected_shot.

    Nodes and edges are coloured by shot_id along the Turbo scale so the
    temporal ordering is immediately visible (old = dark purple, new = yellow).
    """
    graph = get_reference_graph(selected_shot)
    if len(graph) <= 1:
        return fig

    # Position lookup — only shots visible in plot_df
    pos = {
        int(r["shot_id"]): (r[x_col], r[y_col])
        for _, r in plot_df[plot_df["shot_id"].isin(graph)].iterrows()
    }

    visible = set(pos.keys())
    if not visible:
        return fig

    min_id = min(visible)
    max_id = max(visible)

    # -- Nodes (all connected shots except the primary selection) --
    related = graph - {selected_shot}
    rel_df = plot_df[plot_df["shot_id"].isin(related)]
    if not rel_df.empty:
        node_colors = [
            _ref_shot_color(int(s), min_id, max_id) for s in rel_df["shot_id"]
        ]
        fig.add_trace(
            go.Scatter(
                x=rel_df[x_col],
                y=rel_df[y_col],
                mode="markers",
                marker=dict(
                    size=11,
                    color=node_colors,
                    line=dict(color="rgba(0,0,0,0.4)", width=1),
                    symbol="circle",
                ),
                customdata=rel_df[["shot_id"]].values,
                hovertemplate="ref: %{customdata[0]}<extra></extra>",
                showlegend=False,
                name="_ref_nodes",
            )
        )

    # -- Edges — one trace per edge so each can carry its own colour --
    seen_edges: set[frozenset] = set()
    for shot, ref in _ref_parent.items():
        if shot not in graph or ref not in graph:
            continue
        edge = frozenset((shot, ref))
        if edge in seen_edges:
            continue
        seen_edges.add(edge)
        if shot not in pos or ref not in pos:
            continue
        # Colour by the older (smaller) shot id in the pair
        color = _ref_shot_color(min(shot, ref), min_id, max_id)
        fig.add_trace(
            go.Scatter(
                x=[pos[shot][0], pos[ref][0]],
                y=[pos[shot][1], pos[ref][1]],
                mode="lines",
                line=dict(color=color, width=2, dash="dot"),
                showlegend=False,
                hoverinfo="skip",
                name="_ref_edge",
            )
        )

    return fig


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

# Scatter Graph height — fills viewport minus header + tab bar + controls + padding
_SCATTER_H = "calc(100vh - 183px)"

MAX_FILTERS = 6
OPERATORS = [">=", "<=", ">", "<", "==", "!=", "contains"]

app = dash.Dash(__name__, title="NiceShot!")
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
        dcc.Store(id="ref-graph-enabled", data=False),
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
                html.Span(
                    id="filter-count-display", style=dict(fontSize="13px", color="#888")
                ),
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
                                    style=dict(
                                        fontSize="11px", color="#666", lineHeight="1.6"
                                    ),
                                    children=[
                                        html.Span(
                                            f"backend: {BACKEND}",
                                            style=dict(marginRight="16px"),
                                        ),
                                        html.Span(
                                            f"shots: {len(df):,}",
                                            style=dict(marginRight="16px"),
                                        ),
                                        html.Span(
                                            f"time: {MIN_TIME}–{MAX_TIME} s",
                                            style=dict(marginRight="16px"),
                                        ),
                                        html.Span(
                                            f"signals: {', '.join(TIME_TRACE_SIGNALS)}"
                                        ),
                                    ],
                                ),
                                *(
                                    [
                                        html.Button(
                                            "Reference graph: OFF",
                                            id="ref-toggle-btn",
                                            n_clicks=0,
                                            style=dict(
                                                alignSelf="flex-start",
                                                backgroundColor="#2a2a4a",
                                                color="#888",
                                                border="1px solid #3a3a6a",
                                                padding="4px 12px",
                                                cursor="pointer",
                                                borderRadius="4px",
                                                fontSize="11px",
                                            ),
                                        )
                                    ]
                                    if SHOW_REF_TOGGLE
                                    else []
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
                                                dcc.Graph(
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
                                                ) if SHOW_TRACES else html.Div(
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
                                        *(
                                            [
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
                                                        html.Div(
                                                            id="shap-container",
                                                            style=dict(
                                                                height="calc(100vh - 430px)",
                                                                minHeight="220px",
                                                                overflowY="auto",
                                                                padding="4px",
                                                            ),
                                                            children=[
                                                                html.Span(
                                                                    "Click a point to see SHAP values",
                                                                    style=dict(
                                                                        fontSize="11px",
                                                                        color="#555",
                                                                    ),
                                                                )
                                                            ],
                                                        ),
                                                    ],
                                                )
                                            ]
                                            if SHOW_SHAP
                                            else []
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
                                                                                debounce=True,
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
                                                                    for i in range(
                                                                        MAX_FILTERS
                                                                    )
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
                            style=dict(flex="1", minHeight="0"),
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
                                                    options=[
                                                        {"label": c, "value": c}
                                                        for c in all_cols
                                                    ],
                                                    value="breakdown_type"
                                                    if "breakdown_type" in all_cols
                                                    else None,
                                                    clearable=True,
                                                    style=DROPDOWN_STYLE,
                                                ),
                                            ],
                                        ),
                                        dcc.Graph(
                                            id="umap-plot",
                                            config=dict(
                                                displayModeBar=True, displaylogo=False
                                            ),
                                            style=dict(height=_SCATTER_H),
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
                                                                        for c in numeric_cols
                                                                    ],
                                                                    value=numeric_cols[
                                                                        0
                                                                    ]
                                                                    if numeric_cols
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
                                                                        for c in numeric_cols
                                                                    ],
                                                                    value=numeric_cols[
                                                                        1
                                                                    ]
                                                                    if len(numeric_cols)
                                                                    > 1
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
                                                            options=[
                                                                {"label": c, "value": c}
                                                                for c in all_cols
                                                            ],
                                                            value=None,
                                                            clearable=True,
                                                            placeholder="None",
                                                            style=DROPDOWN_STYLE,
                                                        ),
                                                    ]
                                                ),
                                            ],
                                        ),
                                        dcc.Graph(
                                            id="pair-plot",
                                            config=dict(
                                                displayModeBar=True, displaylogo=False
                                            ),
                                            style=dict(height=_SCATTER_H),
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
                                            ],
                                        ),
                                        dash_table.DataTable(
                                            id="shot-table",
                                            columns=_table_column_defs,
                                            data=df[_table_cols].to_dict("records"),
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
                                                border=f"1px solid #2a2a4a",
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
                                                border=f"1px solid #2a2a4a",
                                            ),
                                            style_data_conditional=[],
                                        ),
                                    ],
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


def _add_selection_highlight(
    fig: go.Figure, plot_df: pd.DataFrame, x_col: str, y_col: str, selected_shot
) -> go.Figure:
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


def _apply_filter_mask(active_filters: list | None) -> pd.DataFrame:
    """Return the filtered dataframe (or full df when no filters are active)."""
    if not active_filters:
        return df
    return df[df["shot_id"].isin(active_filters)]


@app.callback(
    Output("active-filters", "data"),
    Output("filter-count-display", "children"),
    Input({"type": "filter-col", "index": ALL}, "value"),
    Input({"type": "filter-op", "index": ALL}, "value"),
    Input({"type": "filter-val", "index": ALL}, "value"),
    Input("filter-logic", "value"),
)
def apply_filters(cols, ops, vals, logic):
    active = [
        (c, o, v)
        for c, o, v in zip(cols, ops, vals)
        if c and o and v is not None and str(v).strip() != ""
    ]
    if not active:
        return None, ""

    masks = []
    for col, op, val in active:
        try:
            v: float | str = float(val)
        except (ValueError, TypeError):
            v = str(val)
        try:
            s = df[col]
            if op == ">=":
                masks.append(s >= v)
            elif op == "<=":
                masks.append(s <= v)
            elif op == ">":
                masks.append(s > v)
            elif op == "<":
                masks.append(s < v)
            elif op == "==":
                masks.append(s == v)
            elif op == "!=":
                masks.append(s != v)
            elif op == "contains":
                masks.append(s.astype(str).str.contains(str(val), case=False, na=False))
        except Exception:
            pass

    if not masks:
        return None, ""

    mask = masks[0]
    for m in masks[1:]:
        mask = (mask | m) if logic == "OR" else (mask & m)

    shot_ids = df.loc[mask, "shot_id"].tolist()
    n = int(mask.sum())
    label = "OR".join if logic == "OR" else "AND".join
    return shot_ids, f"{n:,} / {len(df):,} shots shown"


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


if SHOW_REF_TOGGLE:

    @app.callback(
        Output("ref-graph-enabled", "data"),
        Output("ref-toggle-btn", "children"),
        Output("ref-toggle-btn", "style"),
        Input("ref-toggle-btn", "n_clicks"),
        State("ref-graph-enabled", "data"),
        prevent_initial_call=True,
    )
    def toggle_ref_graph(n_clicks, currently_enabled):
        enabled = not currently_enabled
        if enabled:
            label = "Reference graph: ON"
            style = dict(
                alignSelf="flex-start",
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
            label = "Reference graph: OFF"
            style = dict(
                alignSelf="flex-start",
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
    Output("umap-plot", "figure"),
    Input("umap-color-col", "value"),
    Input("active-filters", "data"),
    Input("selected-shot", "data"),
    Input("ref-graph-enabled", "data"),
)
def update_umap(
    color_col: str | None, active_filters, selected_shot, ref_graph_enabled
) -> go.Figure:
    plot_df = _apply_filter_mask(active_filters)
    kwargs: dict = dict(
        data_frame=plot_df,
        x="umap_x",
        y="umap_y",
        custom_data=["shot_id"],
        hover_name="shot_id",
        labels={"umap_x": UMAP_X_LABEL, "umap_y": UMAP_Y_LABEL},
    )
    if color_col and color_col in plot_df.columns:
        valid = plot_df[color_col].notna()
        if valid.any():
            kwargs["data_frame"] = plot_df[valid]
            kwargs["color"] = color_col

    fig = px.scatter(**kwargs)
    fig.update_traces(
        marker=dict(size=5, opacity=0.75),
        unselected=dict(marker=dict(opacity=0.75)),
    )
    fig.update_layout(**_SCATTER_LAYOUT, uirevision="umap")
    if ref_graph_enabled and selected_shot is not None:
        _add_reference_graph_overlay(fig, plot_df, "umap_x", "umap_y", selected_shot)
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
    Input("ref-graph-enabled", "data"),
)
def update_pair_plot(
    x_col: str | None,
    y_col: str | None,
    color_col: str | None,
    x_scale: str,
    y_scale: str,
    active_filters,
    selected_shot,
    ref_graph_enabled,
) -> go.Figure:
    if not x_col or not y_col:
        return go.Figure()

    plot_df = _apply_filter_mask(active_filters)
    kwargs: dict = dict(
        data_frame=plot_df,
        x=x_col,
        y=y_col,
        custom_data=["shot_id"],
        hover_name="shot_id",
    )
    if color_col and color_col in plot_df.columns:
        valid = plot_df[color_col].notna()
        if valid.any():
            kwargs["data_frame"] = plot_df[valid]
            kwargs["color"] = color_col

    fig = px.scatter(**kwargs)
    fig.update_traces(
        marker=dict(size=5, opacity=0.75),
        unselected=dict(marker=dict(opacity=0.75)),
    )
    fig.update_layout(
        **_SCATTER_LAYOUT,
        uirevision=f"{x_col}-{y_col}",
        xaxis_type=x_scale,
        yaxis_type=y_scale,
    )
    if ref_graph_enabled and selected_shot is not None:
        _add_reference_graph_overlay(fig, plot_df, x_col, y_col, selected_shot)
    _add_selection_highlight(fig, plot_df, x_col, y_col, selected_shot)
    return fig


def _extract_shot_id(click_data: dict) -> int | None:
    """Pull shot id out of Plotly 6 clickData.

    Plotly 6 serialises customdata as binary (dtype/bdata/shape), so the
    decoded value in clickData may vary by Plotly.js version.  We store the
    shot id in three places and try them in order of reliability:
      1. hovertext  – set via hover_name, always a plain string
      2. customdata – decoded by Plotly.js, shape depends on version
      3. pointIndex – index into df (works only when no color split)
    """
    if not click_data or not click_data.get("points"):
        return None
    point = click_data["points"][0]

    # 1. hovertext (most reliable in Plotly 6)
    ht = point.get("hovertext")
    if ht is not None:
        try:
            return int(ht)
        except (TypeError, ValueError):
            pass

    # 2. customdata
    custom = point.get("customdata")
    if custom is not None:
        val = custom[0] if isinstance(custom, (list, tuple)) else custom
        try:
            return int(val)
        except (TypeError, ValueError):
            pass

    # 3. pointIndex fallback (only safe when figure has a single trace)
    pi = point.get("pointIndex")
    if pi is not None and "color" not in click_data:
        try:
            return int(df.iloc[int(pi)]["shot_id"])
        except Exception:
            pass

    return None


@app.callback(
    Output("selected-shot", "data"),
    Input("umap-plot", "clickData"),
    Input("pair-plot", "clickData"),
    Input("shot-table", "active_cell"),
    State("shot-table", "derived_virtual_data"),
    prevent_initial_call=True,
)
def update_selected_shot(umap_click, pair_click, active_cell, virtual_data):
    triggered_id = dash.ctx.triggered_id
    if triggered_id == "umap-plot":
        return _extract_shot_id(umap_click)
    if triggered_id == "pair-plot":
        return _extract_shot_id(pair_click)
    if triggered_id == "shot-table" and active_cell and virtual_data:
        return int(virtual_data[active_cell["row"]]["shot_id"])
    return dash.no_update


@app.callback(
    Output("shot-table", "data"),
    Input("shot-id-search", "value"),
)
def filter_table_by_shot_id(search):
    if not search or not str(search).strip():
        return df[_table_cols].to_dict("records")
    query = str(search).strip()
    mask = df["shot_id"].astype(str).str.contains(query, na=False)
    return df.loc[mask, _table_cols].to_dict("records")


@app.callback(
    Output("shot-table", "style_data_conditional"),
    Input("selected-shot", "data"),
)
def highlight_table_row(selected_shot):
    if selected_shot is None:
        return []
    return [
        {
            "if": {"filter_query": f"{{shot_id}} = {selected_shot}"},
            "backgroundColor": "#2a3a6e",
            "color": "white",
            "fontWeight": "600",
        }
    ]


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

@app.callback(
    Output("shot-info-panel", "children"),
    Input("selected-shot", "data"),
)
def update_shot_info(selected_shot):
        if selected_shot is None:
            return html.Span(
                "Click a point to see shot details",
                style=dict(fontSize="11px", color="#555"),
            )
        row = df[df["shot_id"] == selected_shot]
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
        prevent_initial_call=True,
    )
    def update_traces(shot_id):
        if shot_id is None:
            return dash.no_update, dash.no_update
        try:
            shot_df = load_shot_traces(shot_id)
        except Exception as exc:
            log.error("[update_traces] error loading shot %d: %s", shot_id, exc)
            return empty_traces_fig(
                f"Error loading shot {shot_id}"
            ), f"Shot {shot_id} — error"
        if shot_df is None:
            return empty_traces_fig(
                f"No data found for shot {shot_id}"
            ), f"Shot {shot_id} — not found"
        return make_traces_fig(shot_df), f"Shot {shot_id}"

    if SHOW_SHAP:

        @app.callback(
            Output("shap-container", "children"),
            Input("selected-shot", "data"),
        )
        def update_shap(shot_id):
            if shot_id is None:
                return html.Span(
                    "Click a point to see SHAP values",
                    style=dict(fontSize="11px", color="#555"),
                )
            img_b64 = make_shap_fig(shot_id)
            if img_b64 is None:
                return html.Span(
                    f"No SHAP data for shot {shot_id}",
                    style=dict(fontSize="11px", color="#555"),
                )
            return html.Img(
                src=f"data:image/png;base64,{img_b64}",
                style=dict(width="100%", height="auto"),
            )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    app.run(debug=_args.debug, host=_args.host, port=_args.port, use_reloader=False)


if __name__ == "__main__":
    main()
