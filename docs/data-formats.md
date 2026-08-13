# Data Formats

---

## Shot statistics file (`SHOT_DATA`)

A flat table of per-shot summary statistics. Accepted formats: `.parquet`, `.csv`, `.pg` (PostgreSQL).

**Required:** one column that identifies the shot. The following names are detected automatically (in order of preference):

```
shot_id  shot  pulse  number  exp_number  pulse_id  shot_number
```

The column is renamed to `shot_id` internally regardless of its original name.

All other columns can be anything. Object columns that can be coerced to numeric are converted automatically. Non-numeric columns are kept and available for coloring and filtering.

**Example schema:**

| shot_id | ip_max | ne_max | breakdown_type | reference__number |
|---------|--------|--------|----------------|------------------|
| 45000   | 1.2e6  | 3.5e19 | ohmic          | 44990            |
| 45001   | 1.4e6  | 4.1e19 | NBI            | 45000            |

---

## Long-format shot statistics (`variable_column`)

A file holding **one row per `(shot, variable)` pair** — the same set of statistic columns repeated for every variable, distinguished by a variable column. Set [`variable_column`](configuration.md#variable_column) in config to the name of that column; the file is then read one variable at a time and a variable selector appears in the header.

**Required:** a shot-ID column (detected as above) and the column named by `variable_column`. Both may be stored as index levels — pandas writes MultiIndex levels as columns, and they are flattened on load. All remaining columns are the per-variable statistics.

**Example schema** (`variable_column: variable_name`):

| shot_id | variable_name | mean | std | iqr | nan_percent |
|---------|---------------|------|-----|-----|-------------|
| 11766   | beta_pol      | 0.084 | 0.034 | 0.012 | 0.42 |
| 11766   | q95           | 7.455 | 1.204 | 0.331 | 0.42 |
| 11767   | beta_pol      | 0.081 | 0.029 | 0.011 | 0.39 |
| 11767   | q95           | 7.201 | 1.118 | 0.298 | 0.39 |

Every variable must share the same columns — the UI builds its widgets once from the file schema. The variable column itself is dropped after loading, since it is constant within one variable's slice.

Only `.parquet` / `.pq` are supported, because the format is read with predicate pushdown so a single variable can be fetched without scanning the whole file. Column dtypes are taken from the Parquet schema and used as-is; unlike flat CSV sources, string columns are **not** sniffed and coerced to numeric, so store numeric statistics with numeric types.

Nothing is read from the file body until a variable is chosen. Each variable's projection is cached to its own file next to the `--umap-cache` path.

---

## Shot statistics from PostgreSQL (`.pg`, postgres backend)

Use a `.pg` file extension for `SHOT_DATA` to read shot statistics directly from a PostgreSQL table via DuckDB's postgres extension. The file path stem is used as the default table name (e.g. `nice-shot shots.pg` reads from the `shots` table). Configure the connection and table via `backend_options` in config:

```yaml
backend_options:
  dsn: "postgresql://user:pass@host/db"
  shot_table: shots     # optional — defaults to the SHOT_DATA path stem
  schema: public        # optional — defaults to public
```

The same shot-ID column detection and renaming rules apply as for CSV/Parquet sources.

---

## Per-shot time trace files (`--data-dir`, parquet backend)

Each shot lives in its own file under `--data-dir`:

```
<data-dir>/
  <any-subdir>/
    <shot_id>.parquet   # or .csv
```

**Required columns:**

- `time` — time in seconds (filtered to `time_window`)
- one column per signal listed in `signals` config

The `<any-subdir>` layer is traversed but its name is not significant — all subdirectories are searched for a matching shot file.

---

## Projection files (`--projection`)

A pre-computed 2-D embedding that skips UMAP/PCA entirely. Three formats are accepted:

### NumPy `.npy`

- Shape `(n, 2)` — rows matched positionally to the shot data file (must have the same row count).
- Shape `(n, 3)` — first column is `shot_id`, next two are coordinates. Joined on `shot_id`.

### CSV or Parquet

Must contain a shot ID column (same auto-detection as the shot stats file) and at least two coordinate columns. The first two non-shot-ID columns are used as X and Y axes; their names appear as axis labels in the UI.

**Example parquet schema:**

| shot_id | umap_x | umap_y |
|---------|--------|--------|
| 45000   | 2.31   | -1.04  |
| 45001   | 2.44   | -0.87  |

---

## SHAP values file (`--shap-data`)

A NetCDF file (`.nc`) containing a single `xarray.DataArray` with two named dimensions:

- `shot_id` — integer shot identifiers
- `feature` — feature names (strings)
- one additional dimension for class (the index `True` / class=1 is used)

The array is opened with `xr.open_dataset` and accessed via the default variable key `__xarray_dataarray_variable__`.

The shot-to-index mapping is built from the shot statistics file at load time, so `shot_id` values in the SHAP file must be a subset of those in `SHOT_DATA`.
