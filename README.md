# NiceShot!

An interactive dashboard for exploring tokamak plasma shot data. Point it at a shot-statistics file and get an instant browser UI for slicing, visualising, and comparing shots.


![NiceShot! dashboard](docs/assets/ui.png)

---

## Features

- **Projection** — UMAP or PCA scatter of every shot, coloured by any column. Backed by a content-hash cache so reloads are instant.
- **Pairwise scatter** — any two numeric columns plotted against each other, with linear/log axis toggles.
- **Correlation** — interactive Pearson correlation heatmap for any selection of numeric columns.
- **Data table** — sortable, virtualized table with shot-ID search, cross-highlight with scatter plots, and CSV export.
- **Time traces** — per-shot signal plots loaded on click. Supports local parquet/CSV files, live UDA, and live SAL backends.
- **Filters** — up to 6 simultaneous column filters combinable with AND / OR logic. All plots update live.
- **Clustering** — run K-Means, DBSCAN, or Agglomerative clustering on any set of numeric columns. Results colour the scatter plots immediately; clusters can be given human-readable class names.
- **Cluster centroid traces** — mean time-series per cluster, computed automatically after clustering and relabelled live as class names change.
- **Outlier detection** — flag anomalous shots with Isolation Forest or Local Outlier Factor. Outliers are highlighted in red on the scatter plots and sample traces are loaded automatically.
- **CSV export** — download the full data table with `cluster_id`, `cluster_name` columns appended when clustering has been run.
- **SHAP decision plots** — per-shot feature attribution rendered inline (optional, requires `--shap-data`).
- **Reference graph** — overlay the full reference-shot lineage on any scatter plot (optional, requires `reference_shot_col` in config).
- **Semantic search** — find shots similar to a selected one via nearest-neighbour search in feature space. Results are highlighted on the scatter plots with gold ring markers.
- **Extensible backends** — add support for new data sources (MDSplus, HDF5, custom APIs, …) by subclassing `TraceBackend` or `ShotDataBackend` and registering via `plugins:` in config.


---

## Requirements

Python ≥ 3.12

---

## Install

```sh
pip install nice-shot
pip install "nice-shot[shap]"   # + SHAP plots, xarray, matplotlib
```

---

## Run

```sh
nice-shot --shot-data path/to/shot_stats.parquet
```

Open **http://localhost:8050** in a browser.

By default `nice-shot` runs under **gunicorn** with 4 worker processes, which supports multiple concurrent users. On first run, UMAP/PCA is computed in the master process and cached; subsequent starts are instant.

For local development with hot-reload use `--debug`:

```sh
nice-shot --shot-data path/to/shot_stats.parquet --debug
```

### Common flags

| Flag | Default | Description |
|------|---------|-------------|
| `--shot-data PATH` | `outputs/shot_stats.parquet` | Shot statistics file (`.csv` or `.parquet`) |
| `--config PATH` | `nice_shot/config.yaml` | YAML config file |
| `--data-dir PATH` | `data/mastu/` | Directory of per-shot files (parquet backend) |
| `--projection PATH` | — | Pre-computed 2-D embedding; skips UMAP/PCA entirely |
| `--shap-data PATH` | — | SHAP values NetCDF (`.nc`); enables the SHAP tab |
| `--workers N` | `4` | Gunicorn worker processes (ignored in `--debug` mode) |
| `--port PORT` | `8050` | Port to listen on |
| `--debug` | off | Use the single-process Flask dev server instead of gunicorn |

---

## Configuration

Edit `nice_shot/config.yaml` (or pass `--config` to point elsewhere):

```yaml
backend: parquet        # parquet | uda | sal

signals:                # columns shown in the time-trace panel
  - ip
  - ne
  - dalpha

time_window:
  min_time: 0.0
  max_time: 1.0

projection_method: umap # umap | pca

umap_features:          # omit to use all numeric columns
  - ip_max
  - ne_max
  - bt_max

reference_shot_col: reference__number   # omit to hide the feature
```

---

## Data

**Shot statistics file** (`--shot-data`) — a flat `.parquet` or `.csv` with one row per shot. The shot ID column is detected automatically (`shot_id`, `shot`, `pulse`, `number`, …).

**Per-shot traces** (`--data-dir`) — one `.parquet` or `.csv` per shot, laid out as:
```
<data-dir>/<any-subdir>/<shot_id>.parquet
```
Each file needs a `time` column and one column per configured signal.

**Pre-computed projection** (`--projection`) — a `.npy` (shape `(n,2)` or `(n,3)`), `.csv`, or `.parquet` with shot ID and two coordinate columns.

**SHAP values** (`--shap-data`) — an `xarray` NetCDF file with `shot_id` and `feature` dimensions.

See [`docs/data-formats.md`](docs/data-formats.md) for full schema details.

---

## Docs

```sh
uv run --dev zensical serve
```

Opens the full documentation at **http://localhost:8000**.
