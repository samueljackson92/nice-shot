"""Callback-level UI tests for nice_shot/app.py.

Dash callbacks are plain Python functions under ``@app.callback`` — calling
them directly with fake arguments exercises the real wiring between the UI
and the extracted analysis logic, without needing a browser or a running
Dash server. These tests intentionally cover only a representative handful
of callbacks: the clustering/outlier/projection logic itself is already
covered by tests/test_analysis_*.py.

All tests use the session-scoped ``app_module`` fixture (see conftest.py),
which imports ``nice_shot.app`` once against a small synthetic dataset with
shot_id 2000..2015 and two numeric feature columns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def test_apply_filters_no_active_filters_returns_none(app_module):
    result = app_module.apply_filters([], [], [], "AND", None, 0)
    assert result is None

    result = app_module.apply_filters([None], [None], [None], "AND", None, 0)
    assert result is None


def test_apply_filters_filters_by_shot_id(app_module):
    result = app_module.apply_filters(["shot_id"], [">="], ["2005"], "AND", None, 0)
    assert result == list(range(2005, 2016))


def test_filter_table_by_shot_id_no_search_returns_all_rows(app_module):
    records = app_module.filter_table_by_shot_id(None, None, 0)
    assert len(records) == 16
    assert set(records[0].keys()) == {"shot_id", "feature_1", "feature_2"}


def test_filter_table_by_shot_id_with_search(app_module):
    records = app_module.filter_table_by_shot_id("2005", None, 0)
    assert [r["shot_id"] for r in records] == [2005]


def test_update_umap_returns_figure_with_expected_points(app_module):
    import plotly.graph_objects as go

    fig = app_module.update_umap(
        None,  # color_col
        None,  # active_filters
        None,  # selected_shot
        False,  # ref_graph_enabled
        None,  # cluster_labels
        None,  # cluster_names
        None,  # outlier_labels
        None,  # search_results
        False,  # search_highlight_enabled
        None,  # latest_shot
        False,  # latest_shot_highlight_enabled
        None,  # variable
        0,  # _dataset_version
    )
    assert isinstance(fig, go.Figure)
    assert len(fig.data[0].x) == 16


def test_download_table_produces_csv_payload(app_module):
    result = app_module.download_table(1, None, None, None)
    assert result["filename"] == "niceshot_export.csv"


def test_run_clustering_callback_returns_labels_and_message(app_module):
    labels, representatives, msg, umap_color, pair_color, _classname = app_module.run_clustering(
        1,  # n_clicks
        "kmeans",  # algorithm
        ["feature_1", "feature_2"],  # features
        2,  # n_clusters
        0.5,  # eps
        5,  # min_samples
        False,  # use_projection
        None,  # variable
    )
    assert set(labels.keys()) == {str(sid) for sid in range(2000, 2016)}
    assert umap_color == app_module._CLUSTER_COLOR_VALUE
    assert pair_color == app_module._CLUSTER_COLOR_VALUE
    assert "cluster" in msg.lower()
    # Every representative is a real shot belonging to the cluster it represents.
    assert set(representatives.keys()) == {str(cid) for cid in set(labels.values()) if cid >= 0}
    for cid_str, shot_id in representatives.items():
        assert labels[str(shot_id)] == int(cid_str)


def test_run_clustering_callback_no_features_selected(app_module):
    labels, representatives, msg, umap_color, pair_color, _classname = app_module.run_clustering(
        1, "kmeans", [], 2, 0.5, 5, False, None
    )
    assert msg == "Select at least one feature"


def test_run_outlier_detection_callback_returns_labels_and_message(app_module):
    labels, msg, umap_color, pair_color, _classname = app_module.run_outlier_detection(
        1,  # n_clicks
        "isoforest",  # algorithm
        ["feature_1", "feature_2"],  # features
        0.1,  # contamination
        20,  # n_neighbors
        False,  # use_projection
        None,  # variable
    )
    assert set(labels.keys()) == {str(sid) for sid in range(2000, 2016)}
    assert umap_color == app_module._OUTLIER_COLOR_VALUE
    assert "outlier" in msg.lower()


def test_add_latest_shot_highlight_pure_function(app_module):
    import plotly.graph_objects as go

    plot_df = pd.DataFrame({"shot_id": [1, 2, 3], "umap_x": [0.0, 1.0, 2.0], "umap_y": [0.0, 1.0, 2.0]})

    fig = app_module._add_latest_shot_highlight(go.Figure(), plot_df, "umap_x", "umap_y", 2, True)
    assert any(trace.name == "_latest" for trace in fig.data)

    fig = app_module._add_latest_shot_highlight(go.Figure(), plot_df, "umap_x", "umap_y", 2, False)
    assert not any(trace.name == "_latest" for trace in fig.data)

    fig = app_module._add_latest_shot_highlight(go.Figure(), plot_df, "umap_x", "umap_y", None, True)
    assert not any(trace.name == "_latest" for trace in fig.data)

    # A shot_id not present in plot_df (e.g. filtered out) -- no-op, not an error.
    fig = app_module._add_latest_shot_highlight(go.Figure(), plot_df, "umap_x", "umap_y", 999, True)
    assert not any(trace.name == "_latest" for trace in fig.data)


# ---------------------------------------------------------------------------
# Live-update: incremental projection + dataset refresh.
#
# These tests mutate the backing parquet file and/or the projection cache, so
# they're placed last and use their own tmp_path-scoped cache path where the
# assertion depends on a controlled cache miss -- earlier tests in this file
# (and test_cli.py, the only other app_module consumer) don't depend on the
# dataset staying at its original 16 rows once these run.
# ---------------------------------------------------------------------------


def test_get_projection_model_transforms_new_rows_without_refitting(app_module, monkeypatch, tmp_path):
    # Fresh cache path so this test controls exactly when a fit happens,
    # independent of whatever earlier tests already built/cached.
    monkeypatch.setattr(app_module, "UMAP_CACHE_PATH", str(tmp_path / "fresh_projection.npy"))

    calls = {"n": 0}
    original_fit = app_module._fit_projection

    def _spy(*args, **kwargs):
        calls["n"] += 1
        return original_fit(*args, **kwargs)

    monkeypatch.setattr(app_module, "_fit_projection", _spy)

    data = app_module._flat_backend.load(app_module.SHOT_DATA_PATH)
    model1, projection1, ids1 = app_module.get_projection_model(data, None)
    assert calls["n"] == 1  # cache miss -> exactly one fit

    extra = data.iloc[[0]].copy()
    extra["shot_id"] = int(data["shot_id"].max()) + 1000
    grown = pd.concat([data, extra], ignore_index=True)

    model2, projection2, ids2 = app_module.get_projection_model(grown, None)
    assert calls["n"] == 1  # still just the one fit -- new row was transformed, not refit
    assert len(ids2) == len(ids1) + 1
    assert model2.imputer_cols == model1.imputer_cols
    assert projection2.shape == (len(ids1) + 1, 2)


def test_refresh_dataset_merges_new_shot_without_moving_existing_points(app_module):
    ds_before = app_module.get_dataset(None)
    assert ds_before is not None
    before_ids = {int(s) for s in ds_before.df["shot_id"]}
    before_coords = ds_before.df.set_index("shot_id")[["umap_x", "umap_y"]].copy()
    before_shap_idx = dict(ds_before.shap_idx)

    existing = pd.read_parquet(app_module.SHOT_DATA_PATH)
    new_id = int(existing["shot_id"].max()) + 1
    new_row = pd.DataFrame({"shot_id": [new_id], "feature_1": [0.05], "feature_2": [-0.05]})
    pd.concat([existing, new_row], ignore_index=True).to_parquet(app_module.SHOT_DATA_PATH, index=False)

    latest = app_module.refresh_dataset(None)
    assert latest == new_id

    ds_after = app_module.get_dataset(None)
    assert ds_after is not ds_before
    assert {int(s) for s in ds_after.df["shot_id"]} == before_ids | {new_id}
    # SHAP index is fixed at first build -- never recomputed on refresh (it
    # indexes into a static SHAP file that doesn't grow with new shots).
    assert ds_after.shap_idx == before_shap_idx

    after_coords = ds_after.df.set_index("shot_id")[["umap_x", "umap_y"]]
    for shot_id in before_ids:
        np.testing.assert_allclose(before_coords.loc[shot_id].values, after_coords.loc[shot_id].values, atol=1e-10)

    # get_dataset now serves the refreshed dataset without rebuilding.
    assert app_module.get_dataset(None) is ds_after


def test_refresh_dataset_returns_none_when_nothing_new(app_module):
    assert app_module.refresh_dataset(None) is None


# ---------------------------------------------------------------------------
# Configuration tab
#
# The tab lets the user change the displayed signals and the time window while
# the app runs. The config file supplies the startup values: signals
# ["ip", "ne"] over 0.0-1.0 s, with a third signal "dalpha" present in the
# trace files but not listed in the config.
# ---------------------------------------------------------------------------


def test_apply_config_publishes_selection(app_module):
    signals, window, message = app_module.apply_config(1, ["ip", "dalpha"], 0.0, 0.5)
    assert signals == ["ip", "dalpha"]
    assert window == {"min_time": 0.0, "max_time": 0.5}
    assert "2 signal(s)" in message


def test_apply_config_strips_and_drops_blank_names(app_module):
    signals, _window, _message = app_module.apply_config(1, ["  ", "ip ", None], 0.0, 1.0)
    assert signals == ["ip"]


def test_apply_config_rejects_empty_signal_list(app_module):
    import dash

    signals, window, message = app_module.apply_config(1, [], 0.0, 1.0)
    assert isinstance(signals, dash._callback.NoUpdate)
    assert isinstance(window, dash._callback.NoUpdate)
    assert message == "Select at least one signal."


def test_apply_config_rejects_inverted_time_window(app_module):
    import dash

    signals, window, message = app_module.apply_config(1, ["ip"], 2.0, 1.5)
    assert isinstance(signals, dash._callback.NoUpdate)
    assert isinstance(window, dash._callback.NoUpdate)
    assert "must be less than" in message


def test_apply_config_rejects_missing_time_values(app_module):
    _signals, _window, message = app_module.apply_config(1, ["ip"], None, 1.5)
    assert message == "Give a number for min_time and for max_time."


def test_reset_config_restores_config_file_values(app_module):
    signals, window, message, value, min_time, max_time = app_module.reset_config(1)
    assert signals == app_module.TIME_TRACE_SIGNALS == ["ip", "ne"]
    assert window == {"min_time": app_module.MIN_TIME, "max_time": app_module.MAX_TIME}
    assert value == app_module.TIME_TRACE_SIGNALS
    assert (min_time, max_time) == (app_module.MIN_TIME, app_module.MAX_TIME)
    assert "Reset" in message


def test_discover_signals_needs_a_shot(app_module):
    import dash

    found, status = app_module.discover_signals(1, None)
    assert isinstance(found, dash._callback.NoUpdate)
    assert status == "Select a shot first."


def test_discover_signals_lists_trace_file_columns(app_module):
    found, status = app_module.discover_signals(1, 2000)
    assert found == ["ip", "ne", "dalpha"]
    assert "time" not in found
    assert "3 signal(s)" in status


def test_update_signal_options_keeps_selected_and_typed_names(app_module):
    options = app_module.update_signal_options("my/typed/sig", ["ip", "ne"], ["dalpha"])
    values = [o["value"] for o in options]
    # Discovered, selected, typed and config names are all offered; a selected
    # value missing from options would be cleared by Dash.
    assert values == ["dalpha", "ip", "my/typed/sig", "ne"]


def test_update_signal_options_tolerates_no_search_value(app_module):
    options = app_module.update_signal_options(None, None, None)
    assert [o["value"] for o in options] == ["ip", "ne"]


def test_update_config_summary_follows_the_applied_values(app_module):
    signals_text, time_text = app_module.update_config_summary(["ip", "dalpha"], {"min_time": 0.0, "max_time": 0.5})
    assert signals_text == "signals: ip, dalpha"
    assert time_text == "time: 0.0–0.5 s"


def test_update_config_summary_falls_back_to_the_config_file(app_module):
    signals_text, time_text = app_module.update_config_summary(None, None)
    assert signals_text == "signals: ip, ne"
    assert time_text == f"time: {app_module.MIN_TIME}–{app_module.MAX_TIME} s"


def test_update_traces_uses_the_selected_signals_and_window(app_module):
    fig, title = app_module.update_traces(2000, ["ip", "dalpha"], {"min_time": 0.0, "max_time": 0.5})
    assert [trace.name for trace in fig.data] == ["ip", "dalpha"]
    assert max(fig.data[0].x) <= 0.5
    assert title == "Shot 2000"


def test_update_traces_without_overrides_matches_the_config_file(app_module):
    fig, _title = app_module.update_traces(2000, None, None)
    assert [trace.name for trace in fig.data] == ["ip", "ne"]


def test_update_traces_names_signals_the_shot_does_not_have(app_module):
    _fig, title = app_module.update_traces(2000, ["ip", "not_a_signal"], None)
    assert title == "Shot 2000 — no data for: not_a_signal"


def test_selecting_signals_does_not_change_the_shared_backend(app_module):
    """Each request copies the backend, so one browser cannot affect another."""
    before = list(app_module._trace_backend.config.signals)
    window_before = (
        app_module._trace_backend.config.min_time,
        app_module._trace_backend.config.max_time,
    )
    app_module.update_traces(2000, ["dalpha"], {"min_time": 0.0, "max_time": 0.25})
    assert app_module._trace_backend.config.signals == before
    assert (
        app_module._trace_backend.config.min_time,
        app_module._trace_backend.config.max_time,
    ) == window_before


def test_multi_shot_panels_use_the_selected_signals(app_module):
    data = app_module._load_shots_traces(
        [2000, 2001], signals=["dalpha"], time_window={"min_time": 0.0, "max_time": 0.5}
    )
    assert sorted(data) == ["2000", "2001"]
    assert sorted(data["2000"]) == ["dalpha", "time"]
    fig = app_module._render_outlier_traces_fig(data, ["dalpha"])
    assert [a.text for a in fig.layout.annotations] == ["dalpha"]
