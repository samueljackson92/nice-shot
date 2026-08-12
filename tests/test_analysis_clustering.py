"""Tests for clustering logic in nice_shot/analysis.py."""

from __future__ import annotations

import numpy as np
import pandas as pd

from nice_shot.analysis import (
    _apply_cluster_color,
    _cluster_representatives,
    _run_clustering,
    _sklearn_agglomerative,
    _sklearn_dbscan,
    _sklearn_kmeans,
)

# Two well-separated 2-D blobs.
_X = [[0.0, 0.0], [0.1, -0.1], [-0.1, 0.1], [0.0, 0.2], [10.0, 10.0], [10.1, 9.9], [9.9, 10.1], [10.0, 9.8]]


class TestSklearnClusteringHelpers:
    def test_kmeans_finds_two_clusters(self):
        result = _sklearn_kmeans(_X, n_clusters=2)
        labels = result["labels"]
        assert len(set(labels[:4])) == 1
        assert len(set(labels[4:])) == 1
        assert labels[0] != labels[4]
        assert len(result["centers"]) == 2

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
        labels, representatives = _run_clustering(
            synthetic_shot_df,
            algorithm="kmeans",
            features=["feature_1", "feature_2"],
            n_clusters=2,
            eps=0.5,
            min_samples=5,
        )
        assert set(labels.keys()) == {str(sid) for sid in synthetic_shot_df["shot_id"]}
        assert len(set(labels.values())) == 2
        # Every representative is a real shot that actually belongs to the cluster it represents.
        assert set(representatives.keys()) == {str(cid) for cid in set(labels.values()) if cid >= 0}
        for cid_str, shot_id in representatives.items():
            assert labels[str(shot_id)] == int(cid_str)

    def test_unknown_algorithm_returns_empty(self, synthetic_shot_df):
        labels, representatives = _run_clustering(
            synthetic_shot_df, algorithm="nonexistent", features=["feature_1"], n_clusters=2, eps=0.5, min_samples=5
        )
        assert labels == {}
        assert representatives == {}

    def test_no_matching_features_returns_empty(self, synthetic_shot_df):
        labels, representatives = _run_clustering(
            synthetic_shot_df, algorithm="kmeans", features=["does_not_exist"], n_clusters=2, eps=0.5, min_samples=5
        )
        assert labels == {}
        assert representatives == {}

    def test_empty_dataframe_returns_empty(self):
        empty = pd.DataFrame({"shot_id": [], "feature_1": []})
        labels, representatives = _run_clustering(
            empty, algorithm="kmeans", features=["feature_1"], n_clusters=2, eps=0.5, min_samples=5
        )
        assert labels == {}
        assert representatives == {}


class TestClusterRepresentatives:
    def test_kmeans_picks_shot_nearest_to_center(self):
        X = np.array(_X)
        shot_ids = np.array([100, 101, 102, 103, 200, 201, 202, 203])
        labels = [0, 0, 0, 0, 1, 1, 1, 1]
        centers = [[0.0, 0.0], [10.0, 10.0]]
        reps = _cluster_representatives(X, shot_ids, labels, centers)
        assert reps == {"0": 100, "1": 200}

    def test_non_kmeans_picks_shot_nearest_to_barycenter(self):
        X = np.array(_X)
        shot_ids = np.array([100, 101, 102, 103, 200, 201, 202, 203])
        labels = [0, 0, 0, 0, 1, 1, 1, 1]
        reps = _cluster_representatives(X, shot_ids, labels, centers=None)
        cluster_0_mean = X[:4].mean(axis=0)
        expected_0 = shot_ids[np.argmin(np.linalg.norm(X[:4] - cluster_0_mean, axis=1))]
        assert reps["0"] == int(expected_0)

    def test_noise_points_excluded(self):
        X = np.array(_X)
        shot_ids = np.array([100, 101, 102, 103, 200, 201, 202, 203])
        labels = [0, 0, 0, -1, 1, 1, 1, -1]
        reps = _cluster_representatives(X, shot_ids, labels, centers=None)
        assert set(reps.keys()) == {"0", "1"}


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
