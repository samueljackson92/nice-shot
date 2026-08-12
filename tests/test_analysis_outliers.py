"""Tests for outlier-detection logic in nice_shot/analysis.py."""

from __future__ import annotations

import pandas as pd

from nice_shot.analysis import _apply_outlier_color, _run_outlier_detection, _sklearn_isoforest, _sklearn_lof

# A tight cluster plus one obvious outlier.
_X = [[0.0, 0.0], [0.1, -0.1], [-0.1, 0.1], [0.05, 0.05], [0.0, -0.05], [50.0, 50.0]]


class TestSklearnOutlierHelpers:
    def test_isoforest_flags_obvious_outlier(self):
        preds = _sklearn_isoforest(_X, contamination=0.2)
        assert preds[-1] == -1
        assert preds[0] == 1

    def test_lof_flags_obvious_outlier(self):
        preds = _sklearn_lof(_X, n_neighbors=3, contamination=0.2)
        assert preds[-1] == -1
        assert preds[0] == 1


class TestRunOutlierDetection:
    def test_contamination_respected(self):
        df = pd.DataFrame(
            {
                "shot_id": range(6),
                "feature_1": [x for x, _ in _X],
                "feature_2": [y for _, y in _X],
            }
        )
        labels = _run_outlier_detection(
            df, algorithm="isoforest", features=["feature_1", "feature_2"], contamination=0.2, n_neighbors=3
        )
        assert labels[str(5)] == 1  # outlier
        assert labels[str(0)] == 0  # inlier

    def test_no_matching_features_returns_empty(self, synthetic_shot_df):
        labels = _run_outlier_detection(
            synthetic_shot_df, algorithm="isoforest", features=["does_not_exist"], contamination=0.1, n_neighbors=5
        )
        assert labels == {}

    def test_unknown_algorithm_returns_empty(self, synthetic_shot_df):
        labels = _run_outlier_detection(
            synthetic_shot_df, algorithm="nonexistent", features=["feature_1"], contamination=0.1, n_neighbors=5
        )
        assert labels == {}


class TestApplyOutlierColor:
    def test_maps_labels_to_outlier_inlier(self):
        plot_df = pd.DataFrame({"shot_id": [1, 2, 3], "x": [0.1, 0.2, 0.3]})
        outlier_labels = {"1": 1, "2": 0, "3": 1}
        enriched, color_col = _apply_outlier_color(plot_df, outlier_labels)
        assert color_col == "Outlier"
        result = dict(zip(enriched["shot_id"], enriched["Outlier"]))
        assert result == {1: "Outlier", 2: "Inlier", 3: "Outlier"}

    def test_shots_without_labels_are_dropped(self):
        plot_df = pd.DataFrame({"shot_id": [1, 2], "x": [0.1, 0.2]})
        enriched, _ = _apply_outlier_color(plot_df, {"1": 0})
        assert list(enriched["shot_id"]) == [1]
