# Quickstart

## Install

```sh
git clone <repo>
cd nice_shot
uv sync
```

Optional extras (SHAP plots, UDA/SAL backends):

```sh
uv sync --extra shap
```

---

## Run

```sh
uv run python nice_shot/app.py --shot-data path/to/shot_stats.parquet
```

Open `http://localhost:8050` in a browser.

The first run computes the UMAP projection and caches it. Subsequent starts load from cache unless the data file or `umap_features` list changes.

---

## Minimal example

If you have a single parquet file of shot statistics and nothing else:

```sh
uv run python nice_shot/app.py \
  --shot-data outputs/shot_stats.parquet \
  --no-debug
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
uv run python nice_shot/app.py \
  --shot-data outputs/shot_stats.parquet \
  --data-dir data/mastu \
  --config configs/config_mastu.yml
```

---

## With a pre-computed projection

Skip UMAP computation entirely by supplying your own 2-D embedding:

```sh
uv run python nice_shot/app.py \
  --shot-data outputs/shot_stats.parquet \
  --projection outputs/my_embedding.parquet
```

See [Data Formats](data-formats.md) for accepted shapes.

---

## With SHAP values

```sh
uv run python nice_shot/app.py \
  --shot-data outputs/shot_stats.parquet \
  --shap-data outputs/shap_values.nc
```

A **SHAP** tab appears in the left panel. Click any point to see its decision plot.

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
| `--debug / --no-debug` | debug on | Dash debug / hot-reload mode |
