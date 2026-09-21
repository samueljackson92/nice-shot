"""Tests for nice_shot/config_schema.py."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import get_args

import pytest
import yaml
from pydantic import ValidationError

from nice_shot.config_schema import (
    AppConfig,
    ProjectionMetric,
    ProjectionOptions,
    TimeWindow,
    load_app_config,
    merge_cli_overrides,
)

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"

# Every CLI attribute that merge_cli_overrides looks at, all defaulting to
# None ("not passed on the CLI") -- mirrors app.py's parse_args() defaults.
_NULL_CLI_ARGS = dict(
    backend=None,
    signals=None,
    min_time=None,
    max_time=None,
    timebase_hz=None,
    projection_method=None,
    variable_column=None,
    umap_features=None,
    umap_exclude_features=None,
    reference_shot_col=None,
    plugins=None,
    backend_option=None,
    refresh_interval_seconds=None,
    n_components=None,
    random_state=None,
    n_neighbors=None,
    min_dist=None,
    metric=None,
)


def _args(**overrides) -> argparse.Namespace:
    return argparse.Namespace(**{**_NULL_CLI_ARGS, **overrides})


def test_defaults():
    cfg = AppConfig.model_validate({})
    assert cfg.backend == "parquet"
    assert cfg.signals == ["ip", "ne", "dalpha", "loopv", "plasma_energy"]
    assert cfg.time_window == TimeWindow(min_time=0.0, max_time=1.0)
    assert cfg.projection_method == "umap"
    assert cfg.variable_column is None
    assert cfg.plugins == []
    assert cfg.refresh_interval_seconds is None


def test_refresh_interval_seconds_accepts_positive_value():
    cfg = AppConfig.model_validate({"refresh_interval_seconds": 30.0})
    assert cfg.refresh_interval_seconds == 30.0


def test_refresh_interval_seconds_rejects_zero():
    with pytest.raises(ValidationError, match="must be positive"):
        AppConfig.model_validate({"refresh_interval_seconds": 0})


def test_refresh_interval_seconds_rejects_negative():
    with pytest.raises(ValidationError, match="must be positive"):
        AppConfig.model_validate({"refresh_interval_seconds": -5})


def test_time_window_valid_order():
    tw = TimeWindow(min_time=0.0, max_time=1.0)
    assert tw.min_time < tw.max_time


def test_time_window_raises_when_min_not_less_than_max():
    with pytest.raises(ValidationError, match="must be less than"):
        TimeWindow(min_time=1.0, max_time=1.0)


def test_time_window_raises_when_min_greater_than_max():
    with pytest.raises(ValidationError, match="must be less than"):
        TimeWindow(min_time=2.0, max_time=1.0)


@pytest.mark.parametrize("config_file", sorted(CONFIGS_DIR.glob("*.y*ml")))
def test_example_configs_validate(config_file):
    raw = yaml.safe_load(config_file.read_text()) or {}
    AppConfig.model_validate(raw)


# ---------------------------------------------------------------------------
# CLI / config merge precedence: CLI (explicit) > config file > AppConfig default
# ---------------------------------------------------------------------------


def test_merge_cli_overrides_uses_default_when_absent_from_cli_and_config():
    raw = merge_cli_overrides({}, _args())
    cfg = AppConfig.model_validate(raw)
    assert cfg.backend == "parquet"
    assert cfg.projection_method == "umap"


def test_merge_cli_overrides_config_wins_over_default():
    raw = merge_cli_overrides({"backend": "uda"}, _args())
    cfg = AppConfig.model_validate(raw)
    assert cfg.backend == "uda"


def test_merge_cli_overrides_cli_wins_over_config_and_default():
    raw = merge_cli_overrides({"backend": "uda"}, _args(backend="sal"))
    cfg = AppConfig.model_validate(raw)
    assert cfg.backend == "sal"


def test_merge_cli_overrides_cli_wins_over_default_when_config_absent():
    raw = merge_cli_overrides({}, _args(projection_method="pca"))
    cfg = AppConfig.model_validate(raw)
    assert cfg.projection_method == "pca"


def test_merge_cli_overrides_nested_time_window():
    raw = merge_cli_overrides({"time_window": {"min_time": 0.2, "max_time": 0.8}}, _args(min_time=0.5))
    cfg = AppConfig.model_validate(raw)
    assert cfg.time_window.min_time == 0.5
    assert cfg.time_window.max_time == 0.8  # untouched by CLI, config value kept


def test_merge_cli_overrides_list_fields_replace_wholesale():
    raw = merge_cli_overrides({"signals": ["ip", "ne"]}, _args(signals=["ip", "loopv"]))
    cfg = AppConfig.model_validate(raw)
    assert cfg.signals == ["ip", "loopv"]


def test_merge_cli_overrides_backend_option_merges_over_config_keys():
    raw = merge_cli_overrides(
        {"backend_options": {"server": "old-host", "tree": "mast"}},
        _args(backend_option=["server=new-host"]),
    )
    cfg = AppConfig.model_validate(raw)
    assert cfg.backend_options == {"server": "new-host", "tree": "mast"}


def test_merge_cli_overrides_backend_option_rejects_malformed_entry():
    with pytest.raises(ValueError, match="KEY=VALUE"):
        merge_cli_overrides({}, _args(backend_option=["not-a-kv-pair"]))


def test_merge_cli_overrides_refresh_interval_seconds():
    raw = merge_cli_overrides({}, _args(refresh_interval_seconds=15.0))
    cfg = AppConfig.model_validate(raw)
    assert cfg.refresh_interval_seconds == 15.0


def test_load_app_config_reads_file_and_merges(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({"backend": "uda", "projection_method": "pca"}))
    args = _args(config=str(config_path), backend="sal")
    cfg = load_app_config(args)
    assert cfg.backend == "sal"  # CLI override
    assert cfg.projection_method == "pca"  # config value, no CLI override
    assert cfg.signals == ["ip", "ne", "dalpha", "loopv", "plasma_energy"]  # AppConfig default


# ---------------------------------------------------------------------------
# ProjectionOptions
# ---------------------------------------------------------------------------


def test_projection_options_defaults_repeat_the_old_hardcoded_values():
    """Before these options existed the code used n_components=2, random_state=42
    and UMAP's own defaults. An existing config must give the same embedding."""
    opts = AppConfig.model_validate({}).projection_options
    assert opts.n_components == 2
    assert opts.random_state == 42
    assert opts.n_neighbors == 15
    assert opts.min_dist == 0.1
    assert opts.metric == "euclidean"


@pytest.mark.parametrize("n_components", [1, 0, -1, 51])
def test_projection_options_rejects_n_components_outside_the_range(n_components):
    with pytest.raises(ValidationError, match="must be from 2 to 50"):
        ProjectionOptions(n_components=n_components)


@pytest.mark.parametrize("n_components", [2, 3, 50])
def test_projection_options_accepts_n_components_inside_the_range(n_components):
    assert ProjectionOptions(n_components=n_components).n_components == n_components


@pytest.mark.parametrize("n_neighbors", [1, 0, -5])
def test_projection_options_rejects_n_neighbors_below_two(n_neighbors):
    with pytest.raises(ValidationError, match="must be 2 or more"):
        ProjectionOptions(n_neighbors=n_neighbors)


@pytest.mark.parametrize("min_dist", [-0.1, 1.0, 1.5])
def test_projection_options_rejects_min_dist_outside_the_range(min_dist):
    with pytest.raises(ValidationError, match="must be from 0.0 to less than 1.0"):
        ProjectionOptions(min_dist=min_dist)


@pytest.mark.parametrize("min_dist", [0.0, 0.5, 0.99])
def test_projection_options_accepts_min_dist_inside_the_range(min_dist):
    assert ProjectionOptions(min_dist=min_dist).min_dist == min_dist


def test_projection_options_rejects_an_unknown_metric():
    # The bad value is deliberate, so the type checker is told to allow it: the
    # Literal stops this at edit time, and pydantic must also stop it at run
    # time, because the value can come from a YAML file or the UI.
    with pytest.raises(ValidationError):
        ProjectionOptions(metric="not-a-metric")  # ty: ignore[invalid-argument-type]


def test_projection_options_accepts_every_offered_metric():
    """The UI builds its dropdown from get_args of the same alias, so each
    option it can offer must validate."""
    for metric in get_args(ProjectionMetric):
        assert ProjectionOptions(metric=metric).metric == metric


def test_projection_options_random_state_accepts_none():
    """None means unseeded, which lets UMAP use its faster parallel path."""
    assert ProjectionOptions(random_state=None).random_state is None


def test_projection_options_keeps_umap_fields_when_the_method_is_pca():
    """PCA ignores them, but the value must survive a change back to umap --
    the same rule as uda.timebase_hz under a non-uda backend."""
    cfg = AppConfig.model_validate(
        {"projection_method": "pca", "projection_options": {"n_neighbors": 7, "min_dist": 0.3}}
    )
    assert cfg.projection_options.n_neighbors == 7
    assert cfg.projection_options.min_dist == 0.3


def test_projection_options_nested_cli_override():
    raw = merge_cli_overrides({"projection_options": {"n_neighbors": 7}}, _args(min_dist=0.4))
    cfg = AppConfig.model_validate(raw)
    assert cfg.projection_options.min_dist == 0.4  # from the CLI
    assert cfg.projection_options.n_neighbors == 7  # untouched, config value kept


def test_projection_options_cli_wins_over_config():
    raw = merge_cli_overrides({"projection_options": {"n_components": 3}}, _args(n_components=5))
    assert AppConfig.model_validate(raw).projection_options.n_components == 5
