from __future__ import annotations

import argparse
from typing import Any, Literal

import yaml
from pydantic import BaseModel, model_validator


class TimeWindow(BaseModel):
    min_time: float = 0.0
    max_time: float = 1.0

    @model_validator(mode="after")
    def check_order(self) -> TimeWindow:
        if self.min_time >= self.max_time:
            raise ValueError(
                f"time_window.min_time ({self.min_time}) must be less than time_window.max_time ({self.max_time})"
            )
        return self


class UDAOptions(BaseModel):
    timebase_hz: float | None = None


class AppConfig(BaseModel):
    backend: str = "parquet"
    signals: list[str] = ["ip", "ne", "dalpha", "loopv", "plasma_energy"]
    time_window: TimeWindow = TimeWindow()
    uda: UDAOptions = UDAOptions()
    projection_method: Literal["umap", "pca"] = "umap"
    variable_column: str | None = None
    umap_features: list[str] | None = None
    umap_exclude_features: list[str] = []
    reference_shot_col: str | None = None
    plugins: list[str] = []
    backend_options: dict[str, Any] = {}


# Dotted config path -> CLI namespace attribute, for every AppConfig field that
# also has a CLI flag. CLI flags for these default to None, which means
# "not explicitly passed" -- so a None value here never overrides the config file.
_CLI_CONFIG_FIELDS: list[tuple[tuple[str, ...], str]] = [
    (("backend",), "backend"),
    (("signals",), "signals"),
    (("time_window", "min_time"), "min_time"),
    (("time_window", "max_time"), "max_time"),
    (("uda", "timebase_hz"), "timebase_hz"),
    (("projection_method",), "projection_method"),
    (("variable_column",), "variable_column"),
    (("umap_features",), "umap_features"),
    (("umap_exclude_features",), "umap_exclude_features"),
    (("reference_shot_col",), "reference_shot_col"),
    (("plugins",), "plugins"),
]


def merge_cli_overrides(raw: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Overlay explicitly-set CLI values from ``args`` onto a raw config dict.

    Precedence: an explicit CLI value (non-``None``) always wins. Otherwise the
    config file's value (if any) is left untouched, and ``AppConfig``'s own
    field default applies once ``raw`` is validated.
    """
    for path, attr in _CLI_CONFIG_FIELDS:
        value = getattr(args, attr, None)
        if value is None:
            continue
        node = raw
        for key in path[:-1]:
            node = node.setdefault(key, {})
        node[path[-1]] = value

    backend_option = getattr(args, "backend_option", None)
    if backend_option:
        opts = dict(raw.get("backend_options") or {})
        for item in backend_option:
            key, sep, value = item.partition("=")
            if not sep:
                raise ValueError(f"--backend-option must be KEY=VALUE, got: {item!r}")
            opts[key] = value
        raw["backend_options"] = opts

    return raw


def load_app_config(args: argparse.Namespace) -> AppConfig:
    """Load the config file at ``args.config`` and merge in CLI overrides."""
    with open(args.config) as f:
        raw = yaml.safe_load(f) or {}
    raw = merge_cli_overrides(raw, args)
    return AppConfig.model_validate(raw)
