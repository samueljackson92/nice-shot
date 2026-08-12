"""Tests for nice_shot/config_schema.py."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from nice_shot.config_schema import AppConfig, TimeWindow

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"


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
