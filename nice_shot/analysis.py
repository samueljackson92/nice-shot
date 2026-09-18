"""
Pure, side-effect-free analysis logic used by the NiceShot! dashboard.

Extracted from ``nice_shot.app`` so these functions can be imported and unit
tested without triggering that module's import-time CLI parsing, config
loading, and data/backend initialisation. Nothing here reads ``sys.argv``,
opens a config file, or touches a Dash app.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from nice_shot.backends import detect_shot_col

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Projection (UMAP / PCA)
# ---------------------------------------------------------------------------


@dataclass
class ProjectionModel:
    """A fitted projection pipeline, kept around so new shots can be projected
    with :func:`_transform_projection` instead of refitting on the whole dataset.

    ``imputer_cols`` / ``scaler_cols`` record the *exact* column set and order
    each fitted step expects — feature selection happens in two stages during
    fitting (all-NaN columns dropped before imputation, zero/non-finite-variance
    columns dropped after imputation but before scaling), so both must be
    captured to replay the same column alignment on new data.
    """

    method: str
    imputer_cols: list[str]
    scaler_cols: list[str]
    # Typed loosely (fitted sklearn/umap estimators, imported lazily in
    # _fit_projection to keep this module importable without those heavy deps).
    imputer: Any
    scaler: Any
    reducer: Any


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


def _fit_projection(
    data: pd.DataFrame,
    method: str = "umap",
    umap_features: list[str] | None = None,
    umap_exclude_features: list[str] | None = None,
) -> tuple[ProjectionModel, np.ndarray, np.ndarray]:
    """Fit imputer/scaler/reducer on *data* and return (model, projection, shot_ids).

    Uses mean imputation for NaN/Inf values. The returned :class:`ProjectionModel`
    can later be reused via :func:`_transform_projection` to project new shots
    without refitting.
    """
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
    imputer_cols = X.columns.tolist()
    imputer = SimpleImputer(strategy="mean")
    X_imputed = imputer.fit_transform(X.values.astype(float))
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
    scaler_cols = X.columns.tolist()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if method == "pca":
        from sklearn.decomposition import PCA

        reducer = PCA(n_components=2, random_state=42)
    else:
        from umap import UMAP

        reducer = UMAP(n_components=2, random_state=42)
    projection = reducer.fit_transform(X_scaled)

    model = ProjectionModel(
        method=method,
        imputer_cols=imputer_cols,
        scaler_cols=scaler_cols,
        imputer=imputer,
        scaler=scaler,
        reducer=reducer,
    )
    return model, projection, shot_ids


def _transform_projection(model: ProjectionModel, data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Project *data* onto an already-fitted :class:`ProjectionModel`.

    Never calls ``.fit``/``.fit_transform`` — only ``.transform`` — so existing
    points' coordinates are unaffected. Columns the model was fit on but that are
    absent from *data* are treated as missing (imputed with the fit-time column
    mean), rather than raising, so a shot missing one late-computed diagnostic can
    still be projected.
    """
    X = data.reindex(columns=model.imputer_cols)
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    X_imputed = model.imputer.transform(X.values.astype(float))
    X = pd.DataFrame(X_imputed, columns=pd.Index(model.imputer_cols), index=X.index)
    X_scaled = model.scaler.transform(X[model.scaler_cols])
    projection = model.reducer.transform(X_scaled)
    shot_ids = data["shot_id"].values
    return projection, shot_ids


def _compute_projection(
    data: pd.DataFrame,
    method: str = "umap",
    umap_features: list[str] | None = None,
    umap_exclude_features: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (projection, shot_ids) using mean imputation for NaN/Inf values.

    Thin wrapper around :func:`_fit_projection` for callers that don't need the
    fitted model (e.g. existing tests, one-shot scripts).
    """
    _model, projection, shot_ids = _fit_projection(data, method, umap_features, umap_exclude_features)
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
# Reference lineage / change analysis
# ---------------------------------------------------------------------------

# Strings that real shot tables use to spell "missing". Parquet and SQL sources
# round-trip absent values inconsistently -- mastu_metadata.parquet stores them
# as the literal string "nan" in otherwise-numeric columns -- and ``dropna``
# does not remove those. Blanking them before any coercion or comparison is
# what lets a physics column such as ``ip_av`` be recognised as numeric at all,
# and it stops a comment column reporting a change between "" and None.
_NULL_STRINGS = frozenset({"", "nan", "none", "null", "nat", "n/a", "na", "-"})

# Columns whose names suggest free prose. These differ on almost every shot, so
# they are demoted to the end of the ranked change list rather than excluded --
# the comment diff is informative, it just must not crowd out the settings.
_FREE_TEXT_HINTS = ("comment", "summary", "objective", "postshot", "preshot", "title", "note")

# Relative tolerance for "unchanged". Relative rather than absolute because with
# hundreds of columns there is no single scale: 1e-6 is enormous for a normalised
# latent feature and negligible for a current in amps. 1e-9 sits ~4 orders above
# float64 eps, so it absorbs the string-parse round-trip these files force, and
# ~6 orders below the smallest change an operator can dial in. atol is 0 so that
# 0.0 -> 1e-12 still counts as "switched on".
_UNCHANGED_RTOL = 1e-9

# Depth/size cap for a lineage. Real ancestor chains reach 32, sibling groups 97
# and connected components 298, so this only bites on the wider scopes.
_MAX_LINEAGE = 100

# Columns never offered for comparison: the shot ID and the projection coords.
# Mirrors _table_cols in nice_shot.app.
_ALWAYS_EXCLUDED = ("shot_id", "umap_x", "umap_y")

REFERENCE_METRICS: tuple[str, ...] = ("zscore", "percent", "absolute")
REFERENCE_SCOPES: tuple[str, ...] = ("chain", "component", "siblings")


def _coerce_reference_numeric(series: pd.Series) -> pd.Series:
    """Blank out null-ish strings, then coerce to float.

    Object columns in real shot tables store missing values as the literal
    string ``"nan"`` (see ``mastu_metadata.parquet``), which ``dropna`` does not
    remove. Blanking those first is what lets a physics column such as ``ip_av``
    be recognised as numeric at all.
    """
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    text = series.astype(str).str.strip()
    return pd.to_numeric(text.mask(text.str.lower().isin(_NULL_STRINGS)), errors="coerce")


def _norm_categorical(value: Any) -> str | None:
    """Normalise one categorical cell for equality comparison.

    ``None``, non-finite floats and the usual null-ish strings all collapse to
    ``None``; everything else becomes a stripped string. Comparison stays
    case-sensitive because ``scenario__name`` and ``shot_type`` are canonical
    identifiers where case carries meaning.
    """
    if value is None:
        return None
    if isinstance(value, float) and not np.isfinite(value):
        return None
    text = str(value).strip()
    return None if text.lower() in _NULL_STRINGS else text


def _classify_reference_columns(
    df: pd.DataFrame,
    columns: list[str],
    sample: int = 200,
    threshold: float = 0.9,
) -> tuple[list[str], list[str]]:
    """Split *columns* into ``(numeric, categorical)`` for change comparison.

    Dtype alone is not enough. Long-format sources are prepared with
    ``coerce_objects=False`` (``nice_shot.backends``), so a column of numeric
    strings keeps dtype ``object`` there -- on ``mastu_metadata.parquet`` that
    would hide 121 physics columns from ``is_numeric_dtype``. An object column
    therefore counts as numeric when at least *threshold* of its
    **non-null-ish** values parse as numbers.

    The split is needed in its own right as well: comparing text means testing
    equality, not subtracting, so the caller has to know which columns are
    which whatever their dtype happens to be.

    Datetimes are treated as categorical on purpose: a nanosecond delta would
    dominate every z-score, and the ``datetime`` column in real files is text.
    """
    numeric: list[str] = []
    categorical: list[str] = []
    for col in columns:
        series = df[col]
        if pd.api.types.is_bool_dtype(series):
            numeric.append(col)
        elif pd.api.types.is_datetime64_any_dtype(series) or pd.api.types.is_timedelta64_dtype(series):
            categorical.append(col)
        elif pd.api.types.is_numeric_dtype(series):
            numeric.append(col)
        else:
            text = series.astype(str).str.strip()
            present = series[~text.str.lower().isin(_NULL_STRINGS)]
            if present.empty:
                categorical.append(col)
                continue
            probe = present.head(sample)
            rate = pd.to_numeric(probe.astype(str).str.strip(), errors="coerce").notna().mean()
            (numeric if rate >= threshold else categorical).append(col)
    return numeric, categorical


def column_stds(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    """Per-column standard deviation over the **whole** table, for the z-scored metric.

    Non-finite values are dropped first, mirroring the cleaning in
    :func:`_fit_projection`. Zero and non-finite results come back as ``NaN`` so
    that callers divide by ``NaN`` rather than by zero -- the same
    "unusable column" predicate the projection uses.

    The denominator must come from the whole dataset, not from one lineage: with
    a two-shot lineage a local std makes every z-delta exactly +/-1.41, which
    renders as a uniformly-coloured heatmap that looks plausible and means
    nothing.
    """
    present = [c for c in columns if c in df.columns]
    if not present:
        return pd.Series(dtype=float)
    numeric = df[present].apply(_coerce_reference_numeric).replace([np.inf, -np.inf], np.nan)
    stds = numeric.std(skipna=True)
    return stds.mask(~np.isfinite(stds) | (stds == 0))


def _children_map(ref_parent: dict[int, int]) -> dict[int, list[int]]:
    """Invert the child->parent map into parent->[children], newest child first."""
    children: dict[int, list[int]] = {}
    for child, par in ref_parent.items():
        children.setdefault(int(par), []).append(int(child))
    for kids in children.values():
        kids.sort(reverse=True)
    return children


def _cap_lineage(ids: list[int], subject: int, max_shots: int) -> list[int]:
    """Keep at most *max_shots* ids, never dropping *subject*.

    Clicking a shot must never produce a view that omits it, and real connected
    components reach 298 members.
    """
    if len(ids) <= max_shots:
        return ids
    kept = ids[:max_shots]
    if subject not in kept:
        kept[-1] = subject
        kept.sort(reverse=True)
        log.warning("[lineage] truncated to %d shots; kept the subject %s", max_shots, subject)
    else:
        log.warning("[lineage] truncated to %d of %d shots", max_shots, len(ids))
    return kept


def get_reference_lineage(
    shot_id: int,
    ref_parent: dict[int, int],
    ref_adjacency: dict[int, list[int]] | None = None,
    scope: str = "chain",
    max_shots: int = _MAX_LINEAGE,
    include_parent: bool = True,
    restrict_to: set[int] | None = None,
) -> tuple[list[int], frozenset[int]]:
    """Ordered, newest-first lineage of *shot_id*, plus the ids hidden by filters.

    ``scope``:
      * ``"chain"``     -- subject -> reference -> reference's reference -> ...
      * ``"component"`` -- the full connected component, via :func:`get_reference_graph`
      * ``"siblings"``  -- shots sharing the subject's reference, with that
                           reference appended last when *include_parent*

    The order is always newest-first, and the subject is always present. For
    ``"chain"`` the order is *graph* order rather than a sort, so the pairing
    stays semantically correct for the rare shot whose reference ID is higher
    than its own; newest-first is that order's (near-always true) physical
    reading. Callers rely on this: the change matrix compares row *i* against
    row *i+1*, so there is exactly one definition of "the previous shot".

    *restrict_to* does not drop shots. It reports them in the second return
    value instead, so a filtered-out ancestor greys out in the UI without
    silently turning "change vs the previous shot" into a comparison between two
    non-adjacent shots.
    """
    subject = int(shot_id)
    if scope not in REFERENCE_SCOPES:
        log.warning("[lineage] unknown scope %r; using 'chain'", scope)
        scope = "chain"

    if scope == "component":
        component = get_reference_graph(ref_adjacency or {}, subject)
        ids = sorted(component, reverse=True) or [subject]
        ids = _cap_lineage(ids, subject, max_shots)
    elif scope == "siblings":
        parent = ref_parent.get(subject)
        if parent is None:
            ids = [subject]
        else:
            parent = int(parent)
            group = set(_children_map(ref_parent).get(parent, ()))
            group.add(subject)
            group.discard(parent)
            ids = sorted(group, reverse=True)
            ids = _cap_lineage(ids, subject, max_shots - (1 if include_parent else 0))
            if include_parent:
                ids = ids + [parent]
    else:
        ids = [subject]
        seen = {subject}
        cur = subject
        while len(ids) < max_shots:
            nxt = ref_parent.get(cur)
            if nxt is None:
                break
            nxt = int(nxt)
            if nxt in seen:
                log.warning("[lineage] reference cycle at shot %s; stopping the walk", nxt)
                break
            ids.append(nxt)
            seen.add(nxt)
            cur = nxt
        else:
            log.warning("[lineage] chain of shot %s truncated at max_shots=%d", subject, max_shots)

    excluded = frozenset(i for i in ids if i != subject and i not in restrict_to) if restrict_to else frozenset()
    return ids, excluded


def reference_compare_columns(
    df: pd.DataFrame,
    search_cols: list[str] | None = None,
    exclude: list[str] | None = None,
    include_categorical: bool = True,
) -> list[str]:
    """The candidate column set for lineage comparison, free-text columns last.

    *search_cols* (the config's ``umap_features``) only reorders the result --
    it never restricts it. On the datasets that configure a reference column
    that list is either absent or a set of learned latent dimensions, neither of
    which answers "what was changed", while the categorical metadata
    (``scenario__name``, ``shot_type``, ``gas_valves``) often does.
    """
    dropped = set(_ALWAYS_EXCLUDED) | {c for c in (exclude or []) if c}
    candidates = [c for c in df.columns if c not in dropped]
    numeric, categorical = _classify_reference_columns(df, candidates)
    ordered = list(numeric)
    if include_categorical:
        ordered += categorical

    preferred = [c for c in (search_cols or []) if c in set(ordered)]
    rest = [c for c in ordered if c not in set(preferred)]
    ordered = preferred + rest

    plain = [c for c in ordered if not _is_free_text(c)]
    prose = [c for c in ordered if _is_free_text(c)]
    return plain + prose


def _is_free_text(column: str) -> bool:
    lowered = column.lower()
    return any(hint in lowered for hint in _FREE_TEXT_HINTS)


@dataclass(frozen=True)
class ChangeMatrix:
    """Aligned wide frames describing how a lineage changed, shot by shot.

    Every frame is indexed by ``shot_ids`` (newest first) with ``columns`` as
    its columns, so a consumer can index one cell with
    ``magnitude.at[shot_id, column]`` and hand ``values`` straight to a
    DataTable. ``delta`` compares each row against the **next** row, which is
    the older shot -- see :func:`get_reference_lineage` for why the order is
    canonical.
    """

    shot_ids: list[int]
    columns: list[str]
    kinds: dict[str, str]
    values: pd.DataFrame
    delta: pd.DataFrame
    metric_value: pd.DataFrame
    magnitude: pd.DataFrame
    changed: pd.DataFrame
    note: pd.DataFrame
    metric: str
    excluded: frozenset[int] = frozenset()


@dataclass(frozen=True)
class ChangeItem:
    """One variable's change between the subject shot and the shot before it.

    ``delta`` is the change in the column's own units and ``metric_value`` the
    same change under the matrix's metric, so ``magnitude`` is its absolute
    value. A renderer that sorts by ``magnitude`` can therefore show the signed
    number it sorted by, rather than a second, differently-scaled one.

    ``changed`` is the only way to tell an unchanged categorical column from a
    changed one: text has no delta to read a zero out of.
    """

    column: str
    kind: str
    old: Any
    new: Any
    delta: float
    pct: float | None
    magnitude: float
    note: str
    metric_value: float = float("nan")
    changed: bool = True


def _empty_change_matrix(lineage: list[int], columns: list[str], metric: str, excluded: frozenset[int]) -> ChangeMatrix:
    index = pd.Index(lineage, name="shot_id")
    return ChangeMatrix(
        shot_ids=list(lineage),
        columns=list(columns),
        kinds={},
        values=pd.DataFrame(index=index, columns=pd.Index(columns), dtype=object),
        delta=pd.DataFrame(index=index, columns=pd.Index(columns), dtype=float),
        metric_value=pd.DataFrame(index=index, columns=pd.Index(columns), dtype=float),
        magnitude=pd.DataFrame(index=index, columns=pd.Index(columns), dtype=float),
        changed=pd.DataFrame(False, index=index, columns=pd.Index(columns), dtype=bool),
        note=pd.DataFrame("", index=index, columns=pd.Index(columns), dtype=object),
        metric=metric,
        excluded=excluded,
    )


def lineage_change_matrix(
    df: pd.DataFrame,
    lineage: list[int],
    columns: list[str],
    metric: str = "zscore",
    stds: pd.Series | None = None,
    numeric_cols: list[str] | None = None,
    excluded: frozenset[int] = frozenset(),
) -> ChangeMatrix:
    """Per-shot, per-column value, signed change, and a normalised magnitude.

    *lineage* must be newest-first, as :func:`get_reference_lineage` returns it:
    the previous shot for row *i* is row *i+1*, so the oldest shot carries no
    delta and acts as the baseline.

    *metric* selects what drives the colour scale:

    ``"zscore"``
        change divided by that column's standard deviation across the whole
        dataset, so intensity means the same thing in every column.
    ``"percent"``
        change as a percentage of the previous value. Divided by the
        *magnitude* of the baseline, so a rise from -2 to -1 reads as a rise.
    ``"absolute"``
        the change itself. Not comparable between columns.
    """
    cols = [c for c in columns if c in df.columns]
    absent = [c for c in columns if c not in df.columns]
    if absent:
        log.warning("[lineage] columns not found in data: %s", absent)
    if metric not in REFERENCE_METRICS:
        log.warning("[lineage] unknown metric %r; using 'zscore'", metric)
        metric = "zscore"
    ids = [int(s) for s in (lineage or [])]
    if not ids or not cols:
        return _empty_change_matrix(ids, cols, metric, excluded)

    if numeric_cols is None:
        num_cols, cat_cols = _classify_reference_columns(df, cols)
    else:
        wanted = set(numeric_cols)
        num_cols = [c for c in cols if c in wanted]
        cat_cols = [c for c in cols if c not in wanted]
    kinds = {c: ("numeric" if c in set(num_cols) else "categorical") for c in cols}

    index = pd.Index(ids, name="shot_id")
    sub = df.loc[df["shot_id"].astype(int).isin(ids), ["shot_id"] + cols].copy()
    sub["shot_id"] = sub["shot_id"].astype(int)
    sub = sub.set_index("shot_id")
    if sub.index.has_duplicates:
        # reindex() raises on duplicate labels, and duplicate shot IDs are a live
        # risk in long-format mode -- one duplicated row would break the tab.
        log.warning("[lineage] %d duplicate shot_id row(s); keeping the first", int(sub.index.duplicated().sum()))
        sub = sub[~sub.index.duplicated(keep="first")]
    # reindex also inserts all-NaN rows for lineage shots missing from the table.
    sub = sub.reindex(index=index)

    values = pd.DataFrame(index=index, columns=pd.Index(cols), dtype=object)
    delta = pd.DataFrame(np.nan, index=index, columns=pd.Index(cols), dtype=float)
    metric_value = pd.DataFrame(np.nan, index=index, columns=pd.Index(cols), dtype=float)
    changed = pd.DataFrame(False, index=index, columns=pd.Index(cols), dtype=bool)
    note = pd.DataFrame("", index=index, columns=pd.Index(cols), dtype=object)

    if num_cols:
        raw = sub[num_cols].apply(_coerce_reference_numeric)
        exact = raw.replace([np.inf, -np.inf], np.nan)
        prev = exact.shift(-1)
        signed = exact - prev
        values[num_cols] = raw
        delta[num_cols] = signed

        if metric == "zscore":
            spread = (column_stds(df, num_cols) if stds is None else stds).reindex(num_cols)
            spread = spread.mask(~np.isfinite(spread) | (spread == 0))
            scaled = signed.div(spread, axis=1)
        elif metric == "percent":
            base = prev.abs()
            scaled = 100.0 * signed / base.mask(base == 0)
            # 0 -> 0 is not an undefined percentage, it is no change.
            scaled = scaled.mask((base == 0) & signed.eq(0), 0.0)
            spread = None
        else:
            scaled = signed
            spread = None
        metric_value[num_cols] = scaled.replace([np.inf, -np.inf], np.nan)

        close = pd.DataFrame(
            np.isclose(
                exact.to_numpy(dtype=float),
                prev.to_numpy(dtype=float),
                rtol=_UNCHANGED_RTOL,
                atol=0.0,
                equal_nan=True,
            ),
            index=index,
            columns=pd.Index(num_cols),
        )
        both_missing = exact.isna() & prev.isna()
        one_missing = exact.isna() ^ prev.isna()
        changed[num_cols] = (~close & ~both_missing) | one_missing

        marks = pd.DataFrame("", index=index, columns=pd.Index(num_cols), dtype=object)
        marks = marks.mask(raw.isin([np.inf, -np.inf]), "non_finite")
        marks = marks.mask(prev.isna() & exact.notna() & (marks == ""), "new")
        marks = marks.mask(exact.isna() & prev.notna() & (marks == ""), "missing")
        if metric == "zscore" and spread is not None:
            for col in num_cols:
                if pd.isna(spread.get(col, np.nan)):
                    marks[col] = marks[col].mask(marks[col] == "", "zero_variance")
        elif metric == "percent":
            marks = marks.mask((prev.abs() == 0) & signed.ne(0) & signed.notna() & (marks == ""), "zero_baseline")
        note[num_cols] = marks

    if cat_cols:
        normalised = sub[cat_cols].apply(lambda col: col.map(_norm_categorical))
        prev_cat = normalised.shift(-1)
        values[cat_cols] = sub[cat_cols]
        changed[cat_cols] = (normalised != prev_cat) & ~(normalised.isna() & prev_cat.isna())
        marks = pd.DataFrame("", index=index, columns=pd.Index(cat_cols), dtype=object)
        marks = marks.mask(prev_cat.isna() & normalised.notna(), "new")
        marks = marks.mask(normalised.isna() & prev_cat.notna(), "missing")
        note[cat_cols] = marks

    # The oldest shot is the baseline: it has nothing before it to differ from.
    changed.iloc[-1] = False
    note.iloc[-1] = "baseline"

    return ChangeMatrix(
        shot_ids=ids,
        columns=cols,
        kinds=kinds,
        values=values,
        delta=delta,
        metric_value=metric_value,
        magnitude=metric_value.abs(),
        changed=changed,
        note=note,
        metric=metric,
        excluded=excluded,
    )


def _change_sort_key(column: str, kind: str, magnitude: float | None, changed: bool = True) -> tuple:
    """Rank real, comparable numeric changes first and free prose last.

    A column with no usable magnitude -- zero variance, a zero baseline, or a
    value recorded for only one of the two shots -- still appears, but below
    every ranked change. Otherwise "this diagnostic was not recorded" would
    outrank the setting the operator actually altered.

    An unchanged column sorts below all of those, whatever its magnitude says.
    Its magnitude is a perfectly usable 0.0, so without this it would outrank
    every change that cannot be measured -- "nothing happened here" above "this
    diagnostic appeared for the first time".
    """
    usable = magnitude is not None and bool(np.isfinite(magnitude))
    return (
        not changed,
        _is_free_text(column),
        kind != "numeric",
        not usable,
        -(float(magnitude) if usable else 0.0),
        column,
    )


def rank_lineage_changes(
    matrix: ChangeMatrix,
    top_n: int | None = 10,
    changed_only: bool = True,
) -> list[ChangeItem]:
    """What changed between the subject shot and the shot before it, ranked.

    Reads the top row of *matrix* rather than recomputing, so the ranked list
    can never disagree with the history table it sits above.

    *changed_only* drops the variables that hold the value they already had.
    Pass ``False`` for a complete ranking of every compared variable: the
    unchanged ones then sit at the end, where their zero magnitude puts them.
    """
    if len(matrix.shot_ids) < 2 or not matrix.columns:
        return []
    subject, reference = matrix.shot_ids[0], matrix.shot_ids[1]
    items: list[ChangeItem] = []
    for col in matrix.columns:
        if changed_only and not bool(matrix.changed.at[subject, col]):
            continue
        old = matrix.values.at[reference, col]
        new = matrix.values.at[subject, col]
        signed = matrix.delta.at[subject, col]
        signed = float(signed) if pd.notna(signed) else float("nan")
        old_numeric = pd.to_numeric(pd.Series([old]), errors="coerce").iloc[0]
        pct = (
            float(100.0 * signed / abs(old_numeric))
            if pd.notna(old_numeric) and old_numeric != 0 and np.isfinite(signed)
            else None
        )
        magnitude = matrix.magnitude.at[subject, col]
        scaled = matrix.metric_value.at[subject, col]
        items.append(
            ChangeItem(
                column=col,
                kind=matrix.kinds.get(col, "numeric"),
                old=old,
                new=new,
                delta=signed,
                pct=pct,
                magnitude=float(magnitude) if pd.notna(magnitude) else float("nan"),
                note=str(matrix.note.at[subject, col] or ""),
                metric_value=float(scaled) if pd.notna(scaled) else float("nan"),
                changed=bool(matrix.changed.at[subject, col]),
            )
        )
    items.sort(key=lambda it: _change_sort_key(it.column, it.kind, it.magnitude, it.changed))
    return items[:top_n] if top_n else items


def select_changed_columns(matrix: ChangeMatrix, max_columns: int | None = 20, changed_only: bool = True) -> list[str]:
    """Columns worth rendering, most-changed first.

    Ranked by the largest magnitude anywhere in the lineage, so truncation keeps
    the columns that moved rather than the alphabetically first ones. Callers
    must state the truncation in the UI: a silently shortened diff is worse than
    none, because the reader concludes the missing variable did not change.
    """
    if not matrix.columns:
        return []
    biggest = {c: matrix.magnitude[c].max(skipna=True) for c in matrix.columns}
    ever_changed = {c: bool(matrix.changed[c].any()) for c in matrix.columns}
    pool = [c for c in matrix.columns if ever_changed[c]] if changed_only else list(matrix.columns)
    if not pool:
        return []
    pool.sort(
        key=lambda c: _change_sort_key(c, matrix.kinds.get(c, "numeric"), biggest.get(c, np.nan), ever_changed[c])
    )
    if max_columns and len(pool) > max_columns:
        log.info("[lineage] showing %d of %d changed columns", max_columns, len(pool))
        pool = pool[:max_columns]
    return pool


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


def _sklearn_kmeans(X: list, n_clusters: int) -> dict:
    import numpy as np
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto").fit(np.array(X))
    return {"labels": km.labels_.tolist(), "centers": km.cluster_centers_.tolist()}


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


def _cluster_representatives(
    X: np.ndarray, shot_ids: np.ndarray, labels: list[int], centers: list[list[float]] | None
) -> dict[str, int]:
    """Pick one real shot per cluster: nearest to `centers` (kmeans) or to the
    within-cluster barycenter (other algorithms). Never averages/synthesizes a shot.
    """
    labels_arr = np.asarray(labels)
    reps: dict[str, int] = {}
    for cid in sorted(set(labels)):
        if cid < 0:
            continue
        mask = labels_arr == cid
        cluster_X = X[mask]
        cluster_shot_ids = shot_ids[mask]
        center = np.asarray(centers[cid]) if centers is not None else cluster_X.mean(axis=0)
        dists = np.linalg.norm(cluster_X - center, axis=1)
        reps[str(int(cid))] = int(cluster_shot_ids[int(np.argmin(dists))])
    return reps


def _run_clustering(
    df: pd.DataFrame, algorithm: str, features: list[str], n_clusters: int, eps: float, min_samples: int
) -> tuple[dict, dict]:
    """Fit clustering on selected feature columns.

    Returns (labels, representatives): `labels` maps {str(shot_id): cluster_id};
    `representatives` maps {str(cluster_id): representative shot_id} — the real
    shot closest to that cluster's center/barycenter in feature space.
    """
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    valid = [f for f in features if f in df.columns]
    if not valid:
        return {}, {}
    sub = df[["shot_id"] + valid].copy()
    if sub.empty:
        return {}, {}
    raw = sub[valid].replace([np.inf, -np.inf], np.nan).values.astype(float)
    X = StandardScaler().fit_transform(SimpleImputer(strategy="mean").fit_transform(raw))
    centers = None
    if algorithm == "kmeans":
        result = _spawn_sklearn(_sklearn_kmeans, X.tolist(), int(n_clusters))
        labels, centers = result["labels"], result["centers"]
    elif algorithm == "dbscan":
        labels = _spawn_sklearn(_sklearn_dbscan, X.tolist(), float(eps), int(min_samples))
    elif algorithm == "agglomerative":
        labels = _spawn_sklearn(_sklearn_agglomerative, X.tolist(), int(n_clusters))
    else:
        return {}, {}
    shot_ids = sub["shot_id"].values
    label_map = {str(int(sid)): int(lbl) for sid, lbl in zip(shot_ids, labels)}
    representatives = _cluster_representatives(X, shot_ids, labels, centers)
    return label_map, representatives


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
