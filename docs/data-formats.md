# Data Formats

---

## Shot statistics file (`--shot-data`)

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

## Shot statistics from PostgreSQL (`.pg`, postgres backend)

Use a `.pg` file extension for `--shot-data` to read shot statistics directly from a PostgreSQL table via DuckDB's postgres extension. The file path stem is used as the default table name (e.g. `--shot-data shots.pg` reads from the `shots` table). Configure the connection and table via `backend_options` in config:

```yaml
backend_options:
  dsn: "postgresql://user:pass@host/db"
  shot_table: shots     # optional — defaults to the --shot-data path stem
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

The shot-to-index mapping is built from the shot statistics file at load time, so `shot_id` values in the SHAP file must be a subset of those in `--shot-data`.
