"""Tests for clustering logic in nice_shot/analysis.py."""

from __future__ import annotations

import pandas as pd

from nice_shot.analysis import (
    _apply_cluster_color,
    _run_clustering,
    _sklearn_agglomerative,
    _sklearn_dbscan,
    _sklearn_kmeans,
)

# Two well-separated 2-D blobs.
_X = [[0.0, 0.0], [0.1, -0.1], [-0.1, 0.1], [0.0, 0.2], [10.0, 10.0], [10.1, 9.9], [9.9, 10.1], [10.0, 9.8]]


class TestSklearnClusteringHelpers:
    def test_kmeans_finds_two_clusters(self):
        labels = _sklearn_kmeans(_X, n_clusters=2)
        assert len(set(labels[:4])) == 1
        assert len(set(labels[4:])) == 1
        assert labels[0] != labels[4]

    def test_dbscan_finds_two_clusters(self):
        labels = _sklearn_dbscan(_X, eps=1.0, min_samples=2)
        assert len(set(labels[:4])) == 1
        assert len(set(labels[4:])) == 1
        assert labels[0] != labels[4]

    def test_agglomerative_finds_two_clusters(self):
        labels = _sklearn_agglomerative(_X, n_clusters=2)
        assert len(set(labels[:4])) == 1
        assert len(set(labels[4:])) == 1
        assert labels[0] != labels[4]


class TestRunClustering:
    def test_kmeans_end_to_end(self, synthetic_shot_df):
        labels = _run_clustering(
            synthetic_shot_df,
            algorithm="kmeans",
            features=["feature_1", "feature_2"],
            n_clusters=2,
            eps=0.5,
            min_samples=5,
        )
        assert set(labels.keys()) == {str(sid) for sid in synthetic_shot_df["shot_id"]}
        assert len(set(labels.values())) == 2

    def test_unknown_algorithm_returns_empty(self, synthetic_shot_df):
        labels = _run_clustering(
            synthetic_shot_df, algorithm="nonexistent", features=["feature_1"], n_clusters=2, eps=0.5, min_samples=5
        )
        assert labels == {}

    def test_no_matching_features_returns_empty(self, synthetic_shot_df):
        labels = _run_clustering(
            synthetic_shot_df, algorithm="kmeans", features=["does_not_exist"], n_clusters=2, eps=0.5, min_samples=5
        )
        assert labels == {}

    def test_empty_dataframe_returns_empty(self):
        empty = pd.DataFrame({"shot_id": [], "feature_1": []})
        labels = _run_clustering(
            empty, algorithm="kmeans", features=["feature_1"], n_clusters=2, eps=0.5, min_samples=5
        )
        assert labels == {}


class TestApplyClusterColor:
    def test_merges_labels_and_names_clusters(self):
        plot_df = pd.DataFrame({"shot_id": [1, 2, 3], "x": [0.1, 0.2, 0.3]})
        cluster_labels = {"1": 0, "2": 1, "3": -1}
        enriched, color_col = _apply_cluster_color(plot_df, cluster_labels, {})
        assert color_col == "cluster"
        result = dict(zip(enriched["shot_id"], enriched["cluster"]))
        assert result == {1: "Cluster 0", 2: "Cluster 1", 3: "Noise"}

    def test_custom_cluster_names_applied(self):
        plot_df = pd.DataFrame({"shot_id": [1, 2], "x": [0.1, 0.2]})
        cluster_labels = {"1": 0, "2": 1}
        enriched, _ = _apply_cluster_color(plot_df, cluster_labels, {"0": "Quiescent"})
        result = dict(zip(enriched["shot_id"], enriched["cluster"]))
        assert result == {1: "Quiescent", 2: "Cluster 1"}

    def test_shots_without_labels_are_dropped(self):
        plot_df = pd.DataFrame({"shot_id": [1, 2, 3], "x": [0.1, 0.2, 0.3]})
        enriched, _ = _apply_cluster_color(plot_df, {"1": 0}, {})
        assert list(enriched["shot_id"]) == [1]
