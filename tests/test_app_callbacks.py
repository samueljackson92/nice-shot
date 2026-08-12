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


def test_apply_filters_no_active_filters_returns_none(app_module):
    result = app_module.apply_filters([], [], [], "AND", None)
    assert result is None

    result = app_module.apply_filters([None], [None], [None], "AND", None)
    assert result is None


def test_apply_filters_filters_by_shot_id(app_module):
    result = app_module.apply_filters(["shot_id"], [">="], ["2005"], "AND", None)
    assert result == list(range(2005, 2016))


def test_filter_table_by_shot_id_no_search_returns_all_rows(app_module):
    records = app_module.filter_table_by_shot_id(None, None)
    assert len(records) == 16
    assert set(records[0].keys()) == {"shot_id", "feature_1", "feature_2"}


def test_filter_table_by_shot_id_with_search(app_module):
    records = app_module.filter_table_by_shot_id("2005", None)
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
        None,  # variable
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
