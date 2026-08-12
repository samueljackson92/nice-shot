# Configuration

NiceShot reads a YAML config file at startup (`nice_shot/config.yaml` by default, overridable with `--config`). CLI flags take no precedence over config values — they control paths and server settings only.

---

## `backend`

```yaml
backend: parquet   # parquet | uda | sal | postgres | fairmast
```

Controls how per-shot time traces are loaded.

| Value | Behaviour |
|-------|-----------|
| `parquet` | Reads `.parquet` or `.csv` files from `--data-dir`. The time-trace panel is hidden if the directory is absent or empty. |
| `uda` | Fetches live data from UDA via `uda-xarray`. URL form: `uda://<signal>:<shot>`. Requires `uda-xarray` installed separately. |
| `sal` | Fetches live data from SAL via `sal-xarray`. URL form: `sal://pulse/<shot>/<signal>`. Requires `sal-xarray` installed separately. |
| `postgres` | Queries a PostgreSQL table via DuckDB's postgres extension. Requires `dsn` in `backend_options`. The time-trace panel is hidden if the database is unreachable at startup. |
| `fairmast` | Reads per-shot Zarr or netCDF stores (local or remote, e.g. FAIR MAST's S3-hosted level2 data) via `xarray`. The time-trace panel is hidden if `--data-dir` is unreachable at startup. |

---

## `signals`

```yaml
signals:
  - ip
  - ne
  - dalpha
  - loopv
  - plasma_energy
```

Signals shown in the time-trace panel. For the `parquet` backend these must match column names in the per-shot files. For `uda`/`sal` they are passed directly as signal names. For `fairmast` they are `"<group>/<variable>"` strings identifying a diagnostic group and variable within the store (e.g. `thomson_scattering/t_e`); a name with no `/` is read from the store's root group.

---

## `time_window`

```yaml
time_window:
  min_time: 0.0
  max_time: 1.0
```

Crop time traces to this window (seconds). Applied to all backends. `min_time` must be less than `max_time`.

---

## `projection_method`

```yaml
projection_method: umap   # umap | pca
```

Algorithm used to reduce shot statistics to 2-D for the Projection tab.

| Value | Notes |
|-------|-------|
| `umap` | Non-linear; often preserves cluster structure better. Slower on first run; result is cached. |
| `pca` | Linear; fast and deterministic. No caching needed but cache is still written. |

Changing this setting invalidates the projection cache and forces a recompute.

---

## `umap_features`

```yaml
umap_features:
  - ip_max
  - ne_max
  - ff_slope
```

Columns from the shot statistics file to use as features when computing the projection. Shots with `NaN` in any listed column are excluded. Defaults to all numeric columns (excluding `shot_id`) when omitted.

Changing this list invalidates the cache.

---

## `variable_column`

```yaml
variable_column: variable_name
```

Enables **long-format mode** for the shot statistics file: one row per `(shot, variable)` pair, with the named column identifying which variable each row describes. See [Data Formats](data-formats.md#long-format-shot-statistics-variable_column) for the required layout.

When set, a variable selector appears in the header. No row data is read at startup — only the list of variable names. Picking a variable reads just that variable's rows and computes the projection, similarity index and reference graph for them alone. Each variable's projection is cached separately on disk, so revisiting one is instant.

Omit (or set to `null`) for a normal flat, one-row-per-shot file.

!!! note
    `variable_column` cannot be combined with `--projection`, since a single pre-computed embedding cannot describe more than one variable. Startup fails with an error if both are given.

---

## `reference_shot_col`

```yaml
reference_shot_col: reference__number
```

Column in the shot statistics file that holds the reference (parent) shot ID. When set, a toggle button appears in the left panel; enabling it draws the full connected reference graph on the scatter plots when a shot is clicked. Omit (or set to `null`) to hide the feature.

---

## `uda` options

```yaml
uda:
  timebase_hz: 1000
```

Only relevant when `backend: uda`. Interpolates all signals onto a uniform time grid at the given sample rate. If omitted, the native time axis of the first successfully loaded signal is used.

---

## `postgres` options

```yaml
backend: postgres

backend_options:
  dsn: "postgresql://user:pass@host/db"   # required
  trace_table: traces                      # optional — default: traces
  schema: public                           # optional — default: public
  shot_col: shot_id                        # optional — default: shot_id
  time_col: time                           # optional — default: time
```

Only relevant when `backend: postgres` or when using a `.pg` shot statistics file. Uses DuckDB's postgres extension to query the database directly — no separate driver installation is needed beyond DuckDB itself.

| Option | Default | Description |
|--------|---------|-------------|
| `dsn` | _(required)_ | libpq connection string passed to DuckDB's `ATTACH`. |
| `trace_table` | `traces` | Table that holds per-shot time-series data. |
| `schema` | `public` | PostgreSQL schema containing the table. |
| `shot_col` | `shot_id` | Column used to filter rows by shot ID. |
| `time_col` | `time` | Column used for the time axis; renamed to `time` in the returned data if different. |
| `shot_table` | path stem of `--shot-data` | Table to read for shot statistics (only when using a `.pg` shot data file). |

The trace table must contain at least `shot_col`, `time_col`, and one column per signal listed under `signals`. Rows are filtered to the configured `time_window` and the matching `shot_col` value in the database query, so only relevant data is transferred.

---

## `fairmast` options

```yaml
backend: fairmast

data_dir: s3://mast/level2/shots   # or a local directory of <shot_id>.zarr / .nc files

signals:
  - magnetics/ip
  - summary/ip
  - pf_active/coil_current

backend_options:
  format: zarr                          # optional — zarr (default) | netcdf
  storage_options:                      # optional — passed to fsspec / the xarray engine
    anon: true
    client_kwargs:
      endpoint_url: https://s3.echo.stfc.ac.uk
```

Loads per-shot Zarr or netCDF-4 stores via `xarray`, one file per shot at `<data_dir>/<shot_id>.zarr` (or `.nc`). `data_dir` may be a local directory or any URL understood by `fsspec` (e.g. `s3://...`).

Signals are `"<group>/<variable>"` strings identifying a diagnostic group and variable within the store (e.g. `thomson_scattering/t_e`); a name with no `/` is read from the store's root group. **Only scalar (time-only) variables are supported** — multi-dimensional profile variables (e.g. Thomson scattering channel profiles, equilibrium 2-D fields) are skipped with a logged error rather than crashing the trace load.

| Option | Default | Description |
|--------|---------|--------------|
| `format` | `zarr` | Storage format of the per-shot files: `zarr` or `netcdf`. |
| `storage_options` | `{}` | Passed through to `fsspec`/the xarray engine for remote stores (credentials, custom S3 endpoint, etc). Ignored for local paths. |

For FAIR MAST's public level2 data specifically, no credentials are required — only the custom endpoint shown above, since it is served from a non-AWS S3-compatible host.

---

## Example — MAST-U config

```yaml
backend: parquet

signals:
  - ip
  - ne
  - tf_current
  - plasma_energy
  - loopv

time_window:
  min_time: 0.0
  max_time: 1.0

projection_method: umap

umap_features:
  - ip_max
  - ne_max
  - bt_max
  - betmhd_max
  - wmhd_ipmax

reference_shot_col: reference__number
```
