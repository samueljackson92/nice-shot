# Configuration

NiceShot reads a YAML config file at startup (`nice_shot/config.yaml` by default, overridable with `--config`). CLI flags take no precedence over config values — they control paths and server settings only.

---

## `backend`

```yaml
backend: parquet   # parquet | uda | sal
```

Controls how per-shot time traces are loaded.

| Value | Behaviour |
|-------|-----------|
| `parquet` | Reads `.parquet` or `.csv` files from `--data-dir`. The time-trace panel is hidden if the directory is absent or empty. |
| `uda` | Fetches live data from UDA via `uda-xarray`. URL form: `uda://<signal>:<shot>`. Requires `uda-xarray` installed separately. |
| `sal` | Fetches live data from SAL via `sal-xarray`. URL form: `sal://pulse/<shot>/<signal>`. Requires `sal-xarray` installed separately. |

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

Signals shown in the time-trace panel. For the `parquet` backend these must match column names in the per-shot files. For `uda`/`sal` they are passed directly as signal names.

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
