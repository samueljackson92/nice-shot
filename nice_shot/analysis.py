"""
Pure, side-effect-free analysis logic used by the NiceShot! dashboard.

Extracted from ``nice_shot.app`` so these functions can be imported and unit
tested without triggering that module's import-time CLI parsing, config
loading, and data/backend initialisation. Nothing here reads ``sys.argv``,
opens a config file, or touches a Dash app.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from nice_shot.backends import detect_shot_col

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Projection (UMAP / PCA)
# ---------------------------------------------------------------------------


def _projection_feature_cols(
    data: pd.DataFrame,
    umap_features: list[str] | None = None,
    umap_exclude_features: list[str] | None = None,
) -> list[str]:
    if umap_features:
        missing = [c for c in umap_features if c not in data.columns]
        if missing:
            log.warning("[projection] umap_features not found in data: %s", missing)
        cols = [c for c in umap_features if c in data.columns]
    else:
        cols = [c for c in data.select_dtypes(include=[np.number]).columns if c != "shot_id"]
    if umap_exclude_features:
        excluded = [c for c in umap_exclude_features if c in cols]
        if excluded:
            log.info("[projection] excluding %d columns via umap_exclude_features: %s", len(excluded), excluded)
        cols = [c for c in cols if c not in umap_exclude_features]
    return cols


def _compute_projection(
    data: pd.DataFrame,
    method: str = "umap",
    umap_features: list[str] | None = None,
    umap_exclude_features: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (projection, shot_ids) using mean imputation for NaN/Inf values."""
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    tag = method.upper()
    feature_cols = _projection_feature_cols(data, umap_features, umap_exclude_features)
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
        log.info("[%s] Dropping %d all-NaN columns: %s", tag, len(all_nan_cols), all_nan_cols)
        X = X.drop(columns=all_nan_cols)

    if X.shape[1] == 0:
        raise ValueError(
            "All feature columns are entirely NaN. Use 'umap_features' in config to specify columns with data."
        )

    # Coerce to float and replace ±inf with NaN so the imputer can handle them.
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)

    # Report columns that have any missing values (informational only — they are imputed, not dropped).
    nan_cols = X.columns[X.isna().any()].tolist()
    if nan_cols:
        log.info(
            "[%s] imputing NaN/inf values in %d column(s) with column means: %s",
            tag,
            len(nan_cols),
            nan_cols,
        )

    # Impute remaining NaN with column means so all shots are included in the projection.
    X_imputed = SimpleImputer(strategy="mean").fit_transform(X.values.astype(float))
    X = pd.DataFrame(X_imputed, columns=X.columns, index=X.index)
    shot_ids = data["shot_id"].values

    # Drop zero- or non-finite-variance columns — StandardScaler divides by std,
    # so std=0 or std=NaN (from overflow on very large values) produces NaN output.
    col_stds = X.std()
    bad_var_cols = col_stds[~np.isfinite(col_stds) | (col_stds == 0)].index.tolist()
    if bad_var_cols:
        log.warning(
            "[%s] dropping %d zero/non-finite-variance columns before scaling: %s",
            tag,
            len(bad_var_cols),
            bad_var_cols,
        )
        X = X.drop(columns=bad_var_cols)

    if X.shape[1] == 0:
        raise ValueError("No columns with finite variance remain after filtering. Check your feature data.")

    log.info("[%s] fitting on %d rows x %d columns", tag, X.shape[0], X.shape[1])
    X_scaled = StandardScaler().fit_transform(X)

    if method == "pca":
        from sklearn.decomposition import PCA

        projection = PCA(n_components=2, random_state=42).fit_transform(X_scaled)
    else:
        from umap import UMAP

        projection = UMAP(n_components=2, random_state=42).fit_transform(X_scaled)

    return projection, shot_ids


def _load_projection_file(path: str, data: pd.DataFrame) -> tuple[pd.DataFrame, str, str]:
    """Load a pre-computed projection. Returns (df with shot_id/umap_x/umap_y, x_label, y_label)."""
    import os

    ext = os.path.splitext(path)[1].lower()

    if ext == ".npy":
        arr = np.load(path)
        if arr.ndim != 2 or arr.shape[1] < 2:
            raise ValueError(f"Numpy projection must be 2-D with shape (n, 2) or (n, 3); got {arr.shape}")
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
            if len(arr) != len(data):
                raise ValueError(
                    f"Numpy projection has {len(arr)} rows but shot data has {len(data)} rows. "
                    f"Provide a (n, 3) array with shot_id as the first column, or use a "
                    f".csv / .parquet file."
                )
            result = pd.DataFrame(
                {
                    "shot_id": data["shot_id"].values,
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
        raise ValueError(f"Unsupported projection format '{ext}' — expected .npy, .csv, or .parquet")

    shot_col = detect_shot_col(emb)
    if shot_col != "shot_id":
        emb = emb.rename(columns={shot_col: "shot_id"})

    coord_cols = [c for c in emb.columns if c != "shot_id"]
    if len(coord_cols) < 2:
        raise ValueError(f"Projection file must have at least 2 coordinate columns; found: {coord_cols}")
    x_col, y_col = coord_cols[0], coord_cols[1]
    log.info(
        "Loaded projection from %s: %d rows, axes '%s' / '%s'",
        path,
        len(emb),
        x_col,
        y_col,
    )
    result = emb[["shot_id", x_col, y_col]].rename(columns={x_col: "umap_x", y_col: "umap_y"})
    return result, x_col, y_col


# ---------------------------------------------------------------------------
# Reference graph
# ---------------------------------------------------------------------------


def _build_reference_graph(data: pd.DataFrame, col: str) -> tuple[dict[int, list[int]], dict[int, int]]:
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


def get_reference_graph(adjacency: dict[int, list[int]], shot_id: int) -> set[int]:
    """BFS over the undirected reference graph — returns the full connected component."""
    if not adjacency:
        return set()
    visited: set[int] = set()
    queue = [shot_id]
    while queue:
        cur = queue.pop()
        if cur in visited:
            continue
        visited.add(cur)
        queue.extend(n for n in adjacency.get(cur, []) if n not in visited)
    return visited


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------


def _apply_filter_mask(df: pd.DataFrame, active_filters: list | None) -> pd.DataFrame:
    """Return the filtered dataframe (or the full table when no filters are active)."""
    if active_filters is None:
        return df
    return df[df["shot_id"].isin(active_filters)]


def compute_active_filter_ids(
    df: pd.DataFrame,
    cols: list,
    ops: list,
    vals: list,
    logic: str,
) -> list[int] | None:
    """Build the list of shot_ids that pass the active column/operator/value filters.

    ``cols``/``ops``/``vals`` are the parallel per-row filter widget values; a row is
    "active" only when all three are set. ``logic`` combines multiple active filters
    with "AND" or "OR". Returns ``None`` when there are no active filters.
    """
    active = [(c, o, v) for c, o, v in zip(cols, ops, vals) if c and o and v is not None and str(v).strip() != ""]
    if not active:
        return None

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
        return None

    mask = masks[0]
    for m in masks[1:]:
        mask = (mask | m) if logic == "OR" else (mask & m)

    return df.loc[mask, "shot_id"].tolist()


# ---------------------------------------------------------------------------
# Plotly clickData parsing
# ---------------------------------------------------------------------------


def _extract_shot_id(df: pd.DataFrame, click_data: dict | None) -> int | None:
    """Pull shot id out of Plotly 6 clickData.

    Plotly 6 serialises customdata as binary (dtype/bdata/shape), so the
    decoded value in clickData may vary by Plotly.js version.  We store the
    shot id in three places and try them in order of reliability:
      1. hovertext  – set via hover_name, always a plain string
      2. customdata – decoded by Plotly.js, shape depends on version
      3. pointIndex – index into the shot table (only when no color split)
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


# ---------------------------------------------------------------------------
# Subprocess workers for sklearn fits.
#
# Gunicorn forks workers after numpy/BLAS is initialised; calling BLAS in a
# forked process can cause SIGSEGV. Running fits in a fresh spawned subprocess
# avoids this. Functions must be module-level so they can be pickled by
# ProcessPoolExecutor.
# ---------------------------------------------------------------------------


def _sklearn_kmeans(X: list, n_clusters: int) -> list:
    import numpy as np
    from sklearn.cluster import KMeans

    return KMeans(n_clusters=n_clusters, random_state=42, n_init="auto").fit_predict(np.array(X)).tolist()


def _sklearn_dbscan(X: list, eps: float, min_samples: int) -> list:
    import numpy as np
    from sklearn.cluster import DBSCAN

    return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(np.array(X)).tolist()


def _sklearn_agglomerative(X: list, n_clusters: int) -> list:
    import numpy as np
    from sklearn.cluster import AgglomerativeClustering

    return AgglomerativeClustering(n_clusters=n_clusters).fit_predict(np.array(X)).tolist()


def _sklearn_isoforest(X: list, contamination: float) -> list:
    import numpy as np
    from sklearn.ensemble import IsolationForest

    return IsolationForest(contamination=contamination, random_state=42).fit_predict(np.array(X)).tolist()


def _sklearn_lof(X: list, n_neighbors: int, contamination: float) -> list:
    import numpy as np
    from sklearn.neighbors import LocalOutlierFactor

    return LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination).fit_predict(np.array(X)).tolist()


def _spawn_sklearn(fn, *args):
    """Run fn(*args) in a fresh spawned process to avoid fork+BLAS SIGSEGV."""
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=1, mp_context=ctx) as exe:
        return exe.submit(fn, *args).result(timeout=120)


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------


def _run_clustering(
    df: pd.DataFrame, algorithm: str, features: list[str], n_clusters: int, eps: float, min_samples: int
) -> dict:
    """Fit clustering on selected feature columns. Returns {str(shot_id): cluster_id}."""
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    valid = [f for f in features if f in df.columns]
    if not valid:
        return {}
    sub = df[["shot_id"] + valid].copy()
    if sub.empty:
        return {}
    raw = sub[valid].replace([np.inf, -np.inf], np.nan).values.astype(float)
    X = StandardScaler().fit_transform(SimpleImputer(strategy="mean").fit_transform(raw)).tolist()
    if algorithm == "kmeans":
        labels = _spawn_sklearn(_sklearn_kmeans, X, int(n_clusters))
    elif algorithm == "dbscan":
        labels = _spawn_sklearn(_sklearn_dbscan, X, float(eps), int(min_samples))
    elif algorithm == "agglomerative":
        labels = _spawn_sklearn(_sklearn_agglomerative, X, int(n_clusters))
    else:
        return {}
    return {str(int(sid)): int(lbl) for sid, lbl in zip(sub["shot_id"].values, labels)}


def _apply_cluster_color(plot_df: pd.DataFrame, cluster_labels: dict, cluster_names: dict) -> tuple[pd.DataFrame, str]:
    """Merge cluster labels into plot_df for scatter colouring. Returns (enriched_df, color_col)."""
    label_map = {int(k): v for k, v in cluster_labels.items()}
    enriched = plot_df.copy()
    enriched["_cluster_id"] = enriched["shot_id"].map(label_map)
    enriched = enriched[enriched["_cluster_id"].notna()].copy()
    enriched["_cluster_id"] = enriched["_cluster_id"].astype(int)
    enriched["cluster"] = enriched["_cluster_id"].apply(
        lambda cid: (cluster_names or {}).get(str(cid)) or (f"Cluster {cid}" if cid >= 0 else "Noise")
    )
    return enriched.drop(columns=["_cluster_id"]), "cluster"


# ---------------------------------------------------------------------------
# Outlier detection
# ---------------------------------------------------------------------------


def _run_outlier_detection(
    df: pd.DataFrame, algorithm: str, features: list[str], contamination: float, n_neighbors: int
) -> dict:
    """Return {str(shot_id): 1 (outlier) | 0 (inlier)}."""
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    valid = [f for f in features if f in df.columns]
    if not valid:
        return {}
    sub = df[["shot_id"] + valid].copy()
    if sub.empty:
        return {}
    raw = sub[valid].replace([np.inf, -np.inf], np.nan).values.astype(float)
    X = StandardScaler().fit_transform(SimpleImputer(strategy="mean").fit_transform(raw)).tolist()
    if algorithm == "isoforest":
        preds = _spawn_sklearn(_sklearn_isoforest, X, contamination)
    elif algorithm == "lof":
        preds = _spawn_sklearn(_sklearn_lof, X, int(n_neighbors), contamination)
    else:
        return {}
    # sklearn: -1 = outlier, 1 = inlier → convert to 1/0
    return {str(int(sid)): int(p == -1) for sid, p in zip(sub["shot_id"].values, preds)}


def _apply_outlier_color(plot_df: pd.DataFrame, outlier_labels: dict) -> tuple[pd.DataFrame, str]:
    """Merge outlier flags into plot_df. Returns (enriched_df, color_col)."""
    label_map = {int(k): v for k, v in outlier_labels.items()}
    enriched = plot_df.copy()
    enriched["_is_outlier"] = enriched["shot_id"].map(label_map)
    enriched = enriched[enriched["_is_outlier"].notna()].copy()
    enriched["Outlier"] = enriched["_is_outlier"].apply(lambda v: "Outlier" if int(v) == 1 else "Inlier")
    return enriched.drop(columns=["_is_outlier"]), "Outlier"
