# Configuration

NiceShot reads a YAML config file at startup (`nice_shot/config.yaml` by default, overridable with `--config`).

A setting can come from three places. A value from a later place replaces a value from an earlier place:

1. The built-in default.
2. The config file.
3. The command line.

Almost every setting below also has a command-line flag of the same name. Run `nice-shot --help` for the list.

The [Configuration tab](#configuration-tab) changes many of these settings while the app runs. Those changes apply to your browser only, and a restart returns to the config file.

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

This value is the startup default. The **Configuration** tab changes the list while the app runs — see [Configuration tab](#configuration-tab).

---

## `time_window`

```yaml
time_window:
  min_time: 0.0
  max_time: 1.0
```

Crop time traces to this window (seconds). Applied to all backends. `min_time` must be less than `max_time`.

This value is the startup default. The **Configuration** tab changes the window while the app runs — see [Configuration tab](#configuration-tab).

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

Set the hyper-parameters of the algorithm with [`projection_options`](#projection_options).

---

## `umap_features`

```yaml
umap_features:
  - ip_max
  - ne_max
  - ff_slope
```

Columns from the shot statistics file to use as features when computing the projection. Omit it, or give an empty list, to use every numeric column except `shot_id`.

A shot that has no value in a listed column is still projected: the app uses the mean of the column in its place.

The projection coordinate columns are never used as features, even when this setting is omitted. A coordinate column as an input would put the projection into itself.

Changing this list invalidates the cache.

---

## `umap_exclude_features`

```yaml
umap_exclude_features:
  - bad_column
  - another_bad_column
```

Columns to remove from the feature set. The app applies this list after it resolves [`umap_features`](#umap_features), or after the default of all numeric columns. Use it to keep one or two columns out of a long feature set, instead of listing all the other columns.

Changing this list invalidates the cache.

---

## `projection_options`

```yaml
projection_options:
  n_components: 2         # both methods
  random_state: 42        # both methods
  n_neighbors: 15         # umap only
  min_dist: 0.1           # umap only
  metric: euclidean       # umap only
```

The hyper-parameters of the projection.

| Option | Default | Applies to | Description |
|--------|---------|------------|-------------|
| `n_components` | `2` | both | Number of dimensions to calculate. A value from 2 to 50. |
| `random_state` | `42` | both | Seed for the random number generator. Set it to `null` for an unseeded projection, which lets UMAP use more than one processor and is faster, but does not give the same result twice. |
| `n_neighbors` | `15` | `umap` | Size of the neighbourhood UMAP examines around each shot. A small value shows local structure, a large value shows global structure. A value of 2 or more. The app makes the value smaller if the table has fewer shots than this. |
| `min_dist` | `0.1` | `umap` | Smallest distance UMAP puts between two points. A small value makes tight clusters, a large value spreads the points out. A value from 0.0 to less than 1.0. |
| `metric` | `euclidean` | `umap` | How UMAP measures the distance between two shots. One of `euclidean`, `manhattan`, `chebyshev`, `minkowski`, `canberra`, `braycurtis`, `cosine`, `correlation`, `hamming` or `jaccard`. |

PCA ignores the three UMAP options. The app keeps their values, so a change back to `umap` uses them again.

The defaults give the same projection as the versions of NiceShot that had no `projection_options`.

### More than two components

The plots draw the first two components, which are always the columns `umap_x` and `umap_y`. A projection with more components adds the other ones as the columns `umap_3`, `umap_4` and so on, numbered from 1.

These extra columns are ordinary numeric columns. You can select them as the axes of the Pairwise Scatter tab and as the colour of a plot. The app does not put them in the Data Table, in the Lineage comparison or in the similarity index.

`n_components` must not be more than the number of usable feature columns. The app removes a column that holds no values, and a column whose values are all the same, before it calculates the projection, so the number of usable columns can be much smaller than the number of columns in the file. The app gives you the real number in the error message.

Changing any of these values invalidates the cache.

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

## `refresh_interval_seconds`

```yaml
refresh_interval_seconds: 30
```

Poll the shot data backend for new shots this often, in seconds, and merge any new ones into the running dashboard. Omit (or set to `null`, the default) to disable — the dashboard then only ever loads data at startup, as before. Must be positive if set.

Polling is driven by a browser timer (Dash's `dcc.Interval`), so it only runs while a browser tab is open, and only refreshes whichever gunicorn worker happens to serve that tick's request. With `--workers 1` every request is served by the same process, so the dashboard converges immediately; with more workers, different open tabs (or different requests from the same tab) may briefly be served by workers that haven't polled yet, so the "latest shot" and point count can be momentarily inconsistent across requests until every worker has processed at least one tick. This is a transient, self-healing inconsistency, not persistent staleness — set `--workers 1` if strict consistency matters more than throughput.

New shots are **transformed** onto the existing UMAP/PCA projection, never refit — existing points never move. See [`postgres options`](#postgres-options) / [`sql options`](#sql-options) for how backends fetch only the new rows efficiently; other backends fall back to reloading and diffing, which is fine for local files but wasteful for large remote backends polled frequently.

Only additions are picked up this way — edits to an existing shot's feature values, or shots removed from the source, are not detected; restart the process (or clear the projection cache under `--umap-cache`) to pick those up.

---

## `reference_shot_col`

```yaml
reference_shot_col: reference__number
```

Column in the shot statistics file that holds the reference (parent) shot ID.

When you set this column, the dashboard adds two features:

- A toggle button in the left panel. Enable the button, then click a shot to draw the full connected reference graph on the scatter plots.
- A **Lineage** tab. The tab shows how one shot differs from the shots before it in the same lineage.

Omit the column, or set it to `null`, to hide both features.

### The Lineage tab

The tab uses the selected shot. If no shot is selected, the tab uses the shot with the highest shot ID.

Use the lineage control to select which shots the tab includes:

| Lineage | Shots included |
|---------|----------------|
| Ancestor chain | The shot, its reference shot, the reference shot of that shot, and so on. This is the default. |
| Connected | All shots that connect to the shot through reference links. |
| Siblings | All shots that have the same reference shot. |

A lineage holds a maximum of 100 shots. The tab always keeps the selected shot.

#### The summary card

At the top, the tab shows a card with one row for each variable in the table. Each row gives the old value, the new value, the size of the change, and a bar. The largest change comes first, in either direction. Scroll the card to see the smaller changes and then the variables that did not change. The card covers all the variables in the table, not only the variables you select below it.

Use the **Rank by** control to select the measure that sorts the card and gives the value in the change column:

| Rank by | Description |
|---------|-------------|
| z-scored | The change divided by the spread of that column across all shots. This is the default. Use it to compare variables that have different units. |
| percentage | The change as a percentage of the value of the reference shot. The card gives the percentage in its own column, and the percentage sets the length of the bar. The change column then gives the change in the units of the variable. The tab cannot calculate a percentage if the value of the reference shot is 0. |

#### The history table

The **History** view shows one row for each shot in the lineage. The newest shot is the first row. The `rel` column gives the position of each shot relative to the selected shot.

The table gives one column for each variable you select. The colour of a cell shows the change from the previous shot in the lineage. Red is an increase. Blue is a decrease. The text in a cell is always the raw value. Point at a cell to see the change.

Use the colour control to select the measure:

| Measure | Description |
|---------|-------------|
| z-scored change | The change divided by the spread of that column across all shots. This is the default. It makes the colours comparable between variables that use different units. |
| percent change | The change as a percentage of the previous value. The tab cannot calculate this value if the previous value is 0. |
| absolute change | The change in the units of the variable. The colours are not comparable between variables. |

#### The other views

| View | Description |
|------|-------------|
| Notes | The text fields for each shot in the lineage. The tab shows all the columns that it cannot compare as numbers. The comment, objective and scenario fields come first. |
| Tree | The reference links between the shots in the lineage. Click a node to select that shot. |
| Sparklines | One small graph for each variable, in shot order. Use the **Visualise** control to select the variables. The view starts with the first 12 variables from the summary card, in the same order, and follows the **Rank by** control. Select **Top 12** to set this selection again. |

The tab builds the **Sparklines** view only while you look at it. It shows a spinner over a view while the view loads. A wide table, or a lineage that holds many shots, can need some seconds.

#### Variables

A shot table can hold many variables. The variable control therefore searches the column names as you type, and shows a maximum of 200 names at a time.

The tab selects all the projection features: the `umap_features` columns, or all the numeric columns if you do not set `umap_features`. These are the variables that give each shot its position on the scatter plot.

Select **Top changed** to select the 20 variables that changed most instead. Select **Projection features** to select the projection features again. The tab keeps your selection when you click a different shot.

#### Column filters

The Lineage tab does not use the column filters. A hidden shot would make the change values incorrect, because the tab would then compare two shots that are not linked.

Select **Mark filtered shots** to show which shots the filters hide. The tab keeps these shots in the table, but shows them in grey. The change values stay correct.

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
| `shot_table` | path stem of `SHOT_DATA` | Table to read for shot statistics (only when using a `.pg` shot data file). |

The trace table must contain at least `shot_col`, `time_col`, and one column per signal listed under `signals`. Rows are filtered to the configured `time_window` and the matching `shot_col` value in the database query, so only relevant data is transferred.

---

## `sql` options

```yaml
backend_options:
  url: "postgresql+psycopg://user:pass@host/db"   # optional for .sqlite/.db, required for .sql
  shot_table: shots                                 # optional — defaults to the SHOT_DATA path stem
  query: null                                       # optional — raw SELECT, overrides shot_table
  shot_col: shot_id                                 # optional — defaults to shot_id
```

Only relevant when `SHOT_DATA` has a `.sqlite`, `.db`, or `.sql` extension — see [Data Formats](data-formats.md#shot-statistics-from-sql-sqlite-db-sql) for the full option reference. Unlike `postgres` above (DuckDB-based, PostgreSQL only), this backend works with any [SQLAlchemy](https://www.sqlalchemy.org/)-supported engine; sqlite needs no extra driver, other engines need their driver package installed separately.

`shot_col` matters most when [`refresh_interval_seconds`](#refresh_interval_seconds) is set: it's the column the live-update poll filters on (`WHERE shot_col > last_known_shot_id`), pushed down into the database so each poll only transfers new rows rather than reloading the whole table.

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

## Configuration tab

The **Configuration** tab in the right-hand pane changes most of the config file while the app runs. You do not have to edit the file and restart.

The settings apply to your browser only. The server keeps no per-user configuration, so two browsers can use different settings at the same time, and a restart returns to the config file. Use the config file, or a command-line flag, for a value you want at every start. The **Copy as YAML** button helps you do that.

### What each section changes

Each section has a label that tells you when a change takes effect.

| Section | Setting | Label |
|---------|---------|-------|
| Signals | [`signals`](#signals) | applies now |
| Time window | [`time_window`](#time_window) | applies now |
| Trace backend options | [`uda.timebase_hz`](#uda-options), `backend_options` | applies now |
| Live updates | [`refresh_interval_seconds`](#refresh_interval_seconds) | applies now |
| Projection | [`projection_method`](#projection_method), [`projection_options`](#projection_options) | rebuilds the projection |
| Projection features | [`umap_features`](#umap_features), [`umap_exclude_features`](#umap_exclude_features) | rebuilds the projection |
| Table columns | [`reference_shot_col`](#reference_shot_col) | rebuilds the projection |

**Applies now** means the app reads the setting for each request. Select **Apply** and the affected pane redraws.

**Rebuilds the projection** means the setting decides which dataset the app uses. Select **Apply** and the app calculates the projection again, then redraws every plot. The Projection tab shows a spinner over the plot while it calculates, and keeps the old plot visible below it. The first calculation for a new set of values can need some seconds. The app keeps the result, so a change back to an earlier set of values is immediate.

### The buttons

| Button | Effect |
|--------|--------|
| **Apply** | Use the new values. A value that the app refuses changes nothing, and the message beside the buttons tells you what is wrong. |
| **Reset to config file** | Put every setting, and every control, back to the config file. |
| **Copy as YAML** | Show your settings as a config file. Copy the text into your config file, or save it and give it with `--config`, to get these settings at every start. |
| **Discover signals** | Ask the backend which signals it holds for the selected shot, and put them in the list. Select a shot first. |
| **Use pasted list** | Read a list of column names from the box above it, and select those columns as the projection features. Use this for a long feature set. |

The **Trace backend options** section shows only the controls that the backend in use can read:

| Backend | Shows |
|---------|-------|
| `uda`, `sal` | The timebase only. These backends read no options. |
| `postgres` | The grid of names and values only. |
| `fairmast` | Both. |
| `parquet` | Neither, so the section does not appear. |
| A plugin backend | Both, because a plugin can read anything. |

The app passes a name and value from the grid to the backend, exactly as the `--backend-option` flag does. The names each backend reads are given under [`postgres` options](#postgres-options), [`sql` options](#sql-options) and [`fairmast` options](#fairmast-options), and the tab names them for you above the grid.

**Discover signals** works with the `parquet`, `fairmast` and `postgres` backends. The `uda` and `sal` backends address a signal by name only and cannot list what they hold, so the tab tells you to type the names.

A signal that the selected shot does not have is not an error. The pane plots the signals that are present, and the title above it names the others.

### Results that a new projection replaces

A change that rebuilds the projection moves every point. The app therefore clears the results that describe the old positions: the clusters, the outliers, the classification labels and model, the list of similar shots, and the cluster centre traces. Calculate them again on the new projection.

The app keeps your selected shot and your filters. Both name shots, and the set of shots does not change.

### The reference shot column

Set **reference_shot_col** under **Table columns** to use the [Lineage tab](#the-lineage-tab) and the **Reference graph** button. Clear it to turn both off. You do not have to restart.

The Lineage tab is always in the tab bar. It is switched off, and tells you what is missing, until you set the column.

### What the tab does not change

The app never writes your config file.

These settings need a restart, and the **Active configuration** table at the foot of the tab shows them:

- The shot statistics file, the config file, and the `--data-dir`, `--umap-cache`, `--projection` and `--shap-data` paths.
- [`backend`](#backend) and [`variable_column`](#variable_column).
- The host, the port, the number of workers, and debug mode.
- `plugins`. The app imports the Python modules in this list. It listens on every network interface and asks for no password, so a module path that a web page could set would let any visitor run code in the server process. Use the config file or `--plugins` for this setting.

That table also gives the process ID of the worker that built the page. The app runs four workers by default, and each one keeps its own copy of the data, so this tells you which worker answered you.

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

projection_options:
  n_components: 2
  n_neighbors: 15
  min_dist: 0.1

umap_features:
  - ip_max
  - ne_max
  - bt_max
  - betmhd_max
  - wmhd_ipmax

reference_shot_col: reference__number
```
