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


# Distance metrics offered for the UMAP projection. The UI builds its dropdown
# from get_args() of this alias, so the widget and the validator cannot drift.
ProjectionMetric = Literal[
    "euclidean",
    "manhattan",
    "chebyshev",
    "minkowski",
    "canberra",
    "braycurtis",
    "cosine",
    "correlation",
    "hamming",
    "jaccard",
]


class ProjectionOptions(BaseModel):
    """Hyper-parameters for the 2-D projection.

    The defaults repeat what the code did before these options existed, so an
    existing config file gives exactly the same embedding as before.

    ``n_components`` and ``random_state`` apply to both methods. The other
    three apply to UMAP only, and PCA ignores them. A value for them is kept
    (not rejected) while the method is ``pca``, so it survives a change back to
    ``umap`` -- the same reason ``uda.timebase_hz`` is kept when the backend is
    not ``uda``.
    """

    n_components: int = 2
    random_state: int | None = 42
    n_neighbors: int = 15
    min_dist: float = 0.1
    metric: ProjectionMetric = "euclidean"

    @model_validator(mode="after")
    def check_ranges(self) -> ProjectionOptions:
        # The scatter plots read the first two components, so one is not usable.
        # The upper bound catches a typo that would otherwise build thousands of
        # columns and use all the memory.
        if not 2 <= self.n_components <= 50:
            raise ValueError(f"projection_options.n_components ({self.n_components}) must be from 2 to 50")
        if self.n_neighbors < 2:
            raise ValueError(f"projection_options.n_neighbors ({self.n_neighbors}) must be 2 or more")
        # UMAP needs min_dist <= spread, and spread keeps its default of 1.0.
        if not 0.0 <= self.min_dist < 1.0:
            raise ValueError(f"projection_options.min_dist ({self.min_dist}) must be from 0.0 to less than 1.0")
        return self


class AppConfig(BaseModel):
    backend: str = "parquet"
    signals: list[str] = ["ip", "ne", "dalpha", "loopv", "plasma_energy"]
    time_window: TimeWindow = TimeWindow()
    uda: UDAOptions = UDAOptions()
    projection_method: Literal["umap", "pca"] = "umap"
    projection_options: ProjectionOptions = ProjectionOptions()
    variable_column: str | None = None
    umap_features: list[str] | None = None
    umap_exclude_features: list[str] = []
    reference_shot_col: str | None = None
    plugins: list[str] = []
    backend_options: dict[str, Any] = {}
    refresh_interval_seconds: float | None = None

    @model_validator(mode="after")
    def check_refresh_interval(self) -> AppConfig:
        if self.refresh_interval_seconds is not None and self.refresh_interval_seconds <= 0:
            raise ValueError(
                f"refresh_interval_seconds ({self.refresh_interval_seconds}) must be positive, or null to disable"
            )
        return self


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
    (("projection_options", "n_components"), "n_components"),
    (("projection_options", "random_state"), "random_state"),
    (("projection_options", "n_neighbors"), "n_neighbors"),
    (("projection_options", "min_dist"), "min_dist"),
    (("projection_options", "metric"), "metric"),
    (("variable_column",), "variable_column"),
    (("umap_features",), "umap_features"),
    (("umap_exclude_features",), "umap_exclude_features"),
    (("reference_shot_col",), "reference_shot_col"),
    (("plugins",), "plugins"),
    (("refresh_interval_seconds",), "refresh_interval_seconds"),
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
