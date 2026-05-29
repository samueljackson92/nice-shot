# Quickstart

## Install

```sh
pip install nice-shot
```

With optional extras (SHAP plots, xarray, matplotlib):

```sh
pip install "nice-shot[shap]"
```

---

## Run

```sh
nice-shot --shot-data path/to/shot_stats.parquet
```

Open `http://localhost:8050` in a browser.

By default `nice-shot` starts a **gunicorn** server with 4 worker processes, suitable for multiple concurrent users. The first run computes and caches the UMAP/PCA projection in the master process before workers are forked; subsequent starts are instant.

---

## Development mode

For local development with Dash hot-reload use `--debug`. This starts the single-process Flask dev server instead of gunicorn:

```sh
nice-shot --shot-data path/to/shot_stats.parquet --debug
```

---

## Minimal example

If you have a single parquet file of shot statistics and nothing else:

```sh
nice-shot --shot-data outputs/shot_stats.parquet
```

The time-trace panel is hidden automatically when `--data-dir` does not exist or is empty.

---

## With per-shot time traces (parquet backend)

Lay out per-shot files under a directory:

```
data/mastu/
  <subdir>/
    45000.parquet
    45001.parquet
    ...
```

Each file must have a `time` column and one column per signal. Then:

```sh
nice-shot \
  --shot-data outputs/shot_stats.parquet \
  --data-dir data/mastu \
  --config configs/config_mastu.yml
```

---

## With a pre-computed projection

Skip UMAP computation entirely by supplying your own 2-D embedding:

```sh
nice-shot \
  --shot-data outputs/shot_stats.parquet \
  --projection outputs/my_embedding.parquet
```

See [Data Formats](data-formats.md) for accepted shapes.

---

## Clustering shots

The **Clustering** tab (lower-left panel) lets you group shots by their statistics without any pre-processing.

### Workflow

1. **Choose an algorithm** — K-Means, DBSCAN, or Agglomerative.
2. **Select feature columns** — defaults to the same columns used for the projection. Pick any subset of numeric columns from the shot statistics file.
3. **Set parameters**:
   - `n_clusters` — number of clusters (K-Means / Agglomerative).
   - `eps` / `min_samples` — neighbourhood radius and density threshold (DBSCAN).
4. **Click "Run clustering"** — the Projection scatter switches to cluster colours automatically.
5. **Label your clusters** — type a class name next to each cluster; the scatter legend and centroid traces update immediately.

### Cluster centroid traces

After clustering the **Cluster Traces** tab (upper-left) shows the mean time-series for every shot in each cluster, with one coloured line per cluster. The traces are computed automatically on each run and re-labelled live as you edit class names. The button "Compute centroid traces" can be used to re-trigger computation manually if needed.

> Centroid traces require `--data-dir` (parquet backend) or a UDA/SAL backend. The tab is still visible without traces configured but will show a placeholder.

### Exporting with cluster labels

The **Data Table** tab has a **Download CSV** button. When clustering has been run the exported file includes two extra columns:

| Column | Description |
|--------|-------------|
| `cluster_id` | Integer cluster index (`-1` = DBSCAN noise) |
| `cluster_name` | Human-readable label entered in the Clustering tab, or `Cluster N` if not set |

---

## Detecting outliers

The **Outlier Detection** tab (lower-left panel, next to Clustering) flags anomalous shots using scikit-learn anomaly detectors.

### Workflow

1. **Choose an algorithm** — Isolation Forest or Local Outlier Factor.
2. **Select feature columns** — defaults to the projection features.
3. **Set parameters**:
   - `contamination` — expected proportion of outliers (0.01–0.5, both algorithms).
   - `n_neighbors` — neighbourhood size for Local Outlier Factor only.
4. **Click "Run outlier detection"** — both scatter plots switch to an Outlier/Inlier colour scheme (red = outlier, blue = inlier).

### Outlier sample traces

The **Outlier Traces** tab (upper-left, next to Cluster Traces) automatically loads time traces for up to 5 outlier shots after each run, overlaid as individual coloured lines labelled by shot ID.

> Requires `--data-dir` or a UDA/SAL backend. Shows a placeholder if no time-trace backend is configured.

---

## Correlation heatmap

The **Correlation** tab (right panel, next to Data Table) shows a Pearson correlation matrix for any set of numeric columns.

- **Feature selector** — defaults to the same columns used for the projection (`umap_features`). Change the selection to explore different subsets.
- The heatmap updates whenever the feature selection or active filters change, so it always reflects the currently visible shots.
- Colour scale runs from −1 (dark red) through 0 (white) to +1 (dark blue). Each cell is annotated with the `r` value.

---

## With SHAP values

```sh
nice-shot \
  --shot-data outputs/shot_stats.parquet \
  --shap-data outputs/shap_values.nc
```

A **SHAP** tab appears in the left panel. Click any point to see its decision plot.

---

## Semantic shot search

The **Search** tab (right pane) lets you find shots similar to a selected one using nearest-neighbour search in feature space.

1. Click a shot on the scatter plot — the search runs automatically. Or type a shot ID directly and click **"Find similar shots"**.
2. Set **K** — how many similar shots to return (default: 10).
3. Optionally adjust the **features** used for similarity (defaults to the same columns as the projection).

Results appear as gold ring markers on the Projection and Pairwise Scatter plots, fading from bright (most similar) to dim (least similar). A ranked table with distance scores and time traces for the similar shots are shown in the tab. Use the **"Similar shots: ON/OFF"** toggle in the left panel to hide the markers without clearing the results.

---

## Custom backends

NiceShot! uses a registry/factory system so you can add support for new data sources without modifying the app code. Two extension points are available:

- **`TraceBackend`** — how per-shot time-series are loaded (new instrument backends, custom file formats, remote APIs, etc.)
- **`ShotDataBackend`** — how the shot-statistics table is loaded (new file formats such as HDF5, Zarr, databases, etc.)

### Writing a custom trace backend

```python
# my_package/my_backends.py
import pandas as pd
from nice_shot.backends import BackendConfig, TraceBackend, register_trace_backend

class MdsTraceBackend(TraceBackend):
    def __init__(self, config: BackendConfig) -> None:
        super().__init__(config)
        self._server = config.options.get("server", "localhost")

    def load(self, shot_id: int) -> pd.DataFrame | None:
        # fetch data for shot_id and return a DataFrame with a
        # 'time' column and one column per signal, or None
        ...

    def is_available(self) -> bool:
        return True  # or check connectivity

register_trace_backend("mds", MdsTraceBackend)
```

Then in `config.yaml`:

```yaml
backend: mds

plugins:
  - my_package.my_backends

backend_options:
  server: "mds.mylab.ac.uk"
```

### Writing a custom shot data backend

```python
# my_package/my_backends.py
import pandas as pd
from nice_shot.backends import BackendConfig, ShotDataBackend, register_shot_data_backend

class HDF5ShotDataBackend(ShotDataBackend):
    def load(self, path: str) -> pd.DataFrame:
        import h5py
        # load from HDF5 and return a normalised DataFrame
        # (must have a 'shot_id' column — call self._prepare(df) to auto-detect)
        ...

register_shot_data_backend(".h5", HDF5ShotDataBackend)
register_shot_data_backend(".hdf5", HDF5ShotDataBackend)
```

The backend is selected automatically from the file extension of `--shot-data`.

### Plugin loading order

Plugins listed under `plugins:` are imported in order before the backends are instantiated, so all `register_*` calls take effect in time.

---

## CLI reference

| Flag | Default | Description |
|------|---------|-------------|
| `--config PATH` | `nice_shot/config.yaml` | Path to YAML config file |
| `--shot-data PATH` | `outputs/shot_stats.parquet` | Shot statistics file (`.csv` or `.parquet`) |
| `--data-dir PATH` | `data/mastu/` | Directory of per-shot files for the parquet backend |
| `--projection PATH` | _(none)_ | Pre-computed 2-D embedding (skips UMAP/PCA) |
| `--shap-data PATH` | _(none)_ | SHAP values NetCDF file |
| `--umap-cache PATH` | platform cache dir | Where to read/write the projection cache |
| `--host HOST` | `0.0.0.0` | Bind address |
| `--port PORT` | `8050` | Port |
| `--workers N` | `4` | Gunicorn worker processes (production mode only) |
| `--debug / --no-debug` | off | Use Flask dev server instead of gunicorn |
