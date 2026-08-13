"""Tests for nice_shot/config_schema.py."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from nice_shot.config_schema import AppConfig, TimeWindow, load_app_config, merge_cli_overrides

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


def test_load_app_config_reads_file_and_merges(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({"backend": "uda", "projection_method": "pca"}))
    args = _args(config=str(config_path), backend="sal")
    cfg = load_app_config(args)
    assert cfg.backend == "sal"  # CLI override
    assert cfg.projection_method == "pca"  # config value, no CLI override
    assert cfg.signals == ["ip", "ne", "dalpha", "loopv", "plasma_energy"]  # AppConfig default
