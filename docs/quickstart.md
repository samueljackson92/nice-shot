# Quickstart

## Install

```sh
pip install nice-shot
```

To also get SHAP decision plots, install the `shap` extra:

```sh
pip install "nice-shot[shap]"
```

The `sal` trace backend also needs `sal-xarray`. Install it separately.

### Install on Friea

We do not recommend that you run NiceShot! on Friea. If you must run it on the Friea cluster, load the Python 3.12 mini bundle first:

```sh
module unload python
module load python/3.12-minibundle
pip install nice-shot
```

Then run it with:

```sh
~/.local/bin/nice-shot path/to/shot_stats.parquet
```

---

## Run

```sh
nice-shot path/to/shot_stats.parquet
```

`SHOT_DATA` is the only required argument. Open `http://localhost:8050` in a browser.

By default, `nice-shot` starts a **gunicorn** server with 4 worker processes. This lets many users use the app at the same time. At the first start, the main process calculates the UMAP or PCA projection and keeps it in a cache before it starts the workers. The next starts are immediate.

If the time-trace directory (`--data-dir`) does not exist or is empty, the app hides the time-trace panel. All the other tabs operate.

### Development mode

For local development with Dash hot-reload, use `--debug`. This starts the single-process Flask development server instead of gunicorn:

```sh
nice-shot path/to/shot_stats.parquet --debug
```

---

## Add per-shot time traces (parquet backend)

Put one file for each shot in a directory:

```
data/mastu/
  <subdir>/
    45000.parquet
    45001.parquet
    ...
```

Each file must have a `time` column and one column for each signal. Then:

```sh
nice-shot outputs/shot_stats.parquet \
  --data-dir data/mastu \
  --config configs/config_mastu.yml
```

For the other trace backends (UDA, SAL, PostgreSQL and FAIR MAST), refer to [`backend`](configuration.md#backend).

---

## Use a pre-computed projection

To use your own 2-D embedding instead of UMAP or PCA:

```sh
nice-shot outputs/shot_stats.parquet \
  --projection outputs/my_embedding.parquet
```

Refer to [Projection files](data-formats.md#projection-files-projection) for the accepted formats.

---

## Add SHAP values

```sh
nice-shot outputs/shot_stats.parquet \
  --shap-data outputs/shap_values.nc
```

Click a point to see its decision plot in the **SHAP** tab. Refer to [SHAP](dashboard.md#shap).

---

## Next steps

- [Using the Dashboard](dashboard.md) — what each tab does.
- [Configuration](configuration.md) — all the settings and command-line flags.
- [Data Formats](data-formats.md) — the layout of each input file.
