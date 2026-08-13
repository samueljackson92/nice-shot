"""Tests for nice_shot/app.py's parse_args().

Uses the session-scoped ``app_module`` fixture: the module is already
imported (with all its import-time side effects done), so calling
``app_module.parse_args()`` again here is just a plain, side-effect-free
argparse call driven by whatever ``sys.argv`` the test sets up.
"""

from __future__ import annotations

import sys

import pytest


def test_shot_data_is_required(app_module, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["niceshot"])
    with pytest.raises(SystemExit):
        app_module.parse_args()


def test_shot_data_only_leaves_config_flags_unset(app_module, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["niceshot", "shots.parquet"])
    args = app_module.parse_args()
    assert args.shot_data == "shots.parquet"
    for attr in (
        "backend",
        "signals",
        "min_time",
        "max_time",
        "timebase_hz",
        "projection_method",
        "variable_column",
        "umap_features",
        "umap_exclude_features",
        "reference_shot_col",
        "plugins",
        "backend_option",
        "refresh_interval_seconds",
    ):
        assert getattr(args, attr) is None


def test_config_flags_round_trip(app_module, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "niceshot",
            "shots.parquet",
            "--backend",
            "uda",
            "--signals",
            "ip",
            "ne",
            "--min-time",
            "0.1",
            "--max-time",
            "0.9",
            "--timebase-hz",
            "1000",
            "--projection-method",
            "pca",
            "--variable-column",
            "variable_name",
            "--umap-features",
            "ip_max",
            "ne_max",
            "--umap-exclude-features",
            "shot_id",
            "--reference-shot-col",
            "reference__number",
            "--plugins",
            "my_package.my_backends",
            "--backend-option",
            "server=localhost:8080",
            "--backend-option",
            "tree=mast",
            "--refresh-interval-seconds",
            "15",
        ],
    )
    args = app_module.parse_args()
    assert args.shot_data == "shots.parquet"
    assert args.backend == "uda"
    assert args.signals == ["ip", "ne"]
    assert args.min_time == 0.1
    assert args.max_time == 0.9
    assert args.timebase_hz == 1000.0
    assert args.projection_method == "pca"
    assert args.variable_column == "variable_name"
    assert args.umap_features == ["ip_max", "ne_max"]
    assert args.umap_exclude_features == ["shot_id"]
    assert args.reference_shot_col == "reference__number"
    assert args.plugins == ["my_package.my_backends"]
    assert args.backend_option == ["server=localhost:8080", "tree=mast"]
    assert args.refresh_interval_seconds == 15.0


def test_projection_method_rejects_invalid_choice(app_module, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["niceshot", "shots.parquet", "--projection-method", "bogus"])
    with pytest.raises(SystemExit):
        app_module.parse_args()
