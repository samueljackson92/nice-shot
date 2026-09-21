"""Tests for the supervised classification helpers in nice_shot.analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nice_shot.analysis import (
    _apply_class_color,
    _class_name,
    _encode_target,
    _robust_feature_matrix,
    _run_classification,
    candidate_target_cols,
    decision_surface,
)

FEATURES = ["feature_1", "feature_2", "feature_const", "feature_sparse"]


@pytest.fixture
def labelled_df(synthetic_shot_df):
    """The shared fixture with a binary target that follows its two blobs."""
    df = synthetic_shot_df.copy()
    df["target"] = (df["feature_2"] > 5).astype(int)
    return df


@pytest.fixture
def three_class_df():
    """Enough rows for a stratified split over three classes."""
    rng = np.random.default_rng(3)
    n = 120
    df = pd.DataFrame(
        {
            "shot_id": np.arange(5000, 5000 + n),
            "a": rng.normal(size=n),
            "b": rng.normal(size=n),
        }
    )
    df["target"] = pd.cut(df["a"] + df["b"], 3, labels=["low", "mid", "high"]).astype(str)
    return df


class TestRobustFeatureMatrix:
    def test_tolerates_inf_nan_and_zero_variance(self, synthetic_shot_df):
        X, used = _robust_feature_matrix(synthetic_shot_df, FEATURES)
        assert used == FEATURES
        assert X.shape == (len(synthetic_shot_df), 4)
        assert np.isfinite(X).all()

    def test_drops_all_nan_columns(self, synthetic_shot_df):
        df = synthetic_shot_df.copy()
        df["feature_empty"] = np.nan
        _, used = _robust_feature_matrix(df, [*FEATURES, "feature_empty"])
        assert "feature_empty" not in used

    def test_unknown_columns_are_ignored(self, synthetic_shot_df):
        _, used = _robust_feature_matrix(synthetic_shot_df, ["feature_1", "not_a_column"])
        assert used == ["feature_1"]

    def test_no_usable_columns_gives_an_empty_matrix(self, synthetic_shot_df):
        X, used = _robust_feature_matrix(synthetic_shot_df, ["not_a_column"])
        assert used == []
        assert X.shape[1] == 0

    def test_outliers_do_not_set_the_scale(self):
        """The point of a robust scaler: one extreme row must not flatten the rest."""
        df = pd.DataFrame({"a": [*range(20), 10_000.0]})
        X, _ = _robust_feature_matrix(df, ["a"])
        assert np.ptp(X[:20]) > 1.0


class TestEncodeTarget:
    def test_codes_names_and_mask(self):
        codes, classes, labelled = _encode_target(pd.Series(["b", "a", None, "b"]))
        assert classes == ["a", "b"]
        assert list(codes) == [1, 0, 1]
        assert list(labelled) == [True, True, False, True]

    def test_whole_floats_are_named_as_integers(self):
        _, classes, _ = _encode_target(pd.Series([0.0, 1.0, 1.0]))
        assert classes == ["0", "1"]

    def test_class_name_handles_bool_and_text(self):
        assert _class_name(True) == "True"
        assert _class_name(2.5) == "2.5"
        assert _class_name("disrupted") == "disrupted"


class TestCandidateTargetCols:
    def test_picks_low_cardinality_columns_only(self, labelled_df):
        # feature_1 has one distinct value per row, so the cap has to be below
        # the row count for "continuous" to mean anything on a 16-row table.
        cols = candidate_target_cols(labelled_df, max_classes=4)
        assert "target" in cols
        assert "feature_1" not in cols  # continuous
        assert "feature_const" not in cols  # a single value is not a choice
        assert "shot_id" not in cols

    def test_a_text_column_with_few_values_qualifies(self, labelled_df):
        df = labelled_df.copy()
        df["outcome"] = ["disrupted", "clean"] * (len(df) // 2)
        assert "outcome" in candidate_target_cols(df, max_classes=4)

    def test_excludes_projection_coordinates(self, labelled_df):
        df = labelled_df.copy()
        df["umap_x"] = [0, 1] * (len(df) // 2)
        assert "umap_x" not in candidate_target_cols(df)


class TestRunClassification:
    @pytest.mark.parametrize(
        ("algorithm", "params"),
        [
            ("gradient_boosting", {"n_estimators": 20}),
            ("random_forest", {"n_estimators": 20}),
            ("gaussian_process", {"max_train_rows": 50}),
        ],
    )
    def test_binary_end_to_end(self, labelled_df, algorithm, params):
        out = _run_classification(
            labelled_df, algorithm, ["feature_1", "feature_2"], "target", params, test_fraction=0.25
        )
        result = out["result"]
        assert result["classes"] == ["0", "1"]
        assert set(result["labels"]) == {str(sid) for sid in labelled_df["shot_id"]}
        assert set(result["labels"].values()) <= {"0", "1"}
        assert result["metrics"]["train_accuracy"] == pytest.approx(1.0)
        assert out["model"]  # the pickled estimator comes back for SHAP

    def test_probabilities_are_one_column_per_class_and_sum_to_one(self, three_class_df):
        result = _run_classification(three_class_df, "random_forest", ["a", "b"], "target", {"n_estimators": 20})[
            "result"
        ]
        values = np.asarray(result["proba"]["values"])
        assert values.shape == (len(three_class_df), 3)
        assert np.allclose(values.sum(axis=1), 1.0)
        assert result["proba"]["shot_ids"] == list(three_class_df["shot_id"])

    def test_multiclass_labels_are_the_argmax_class(self, three_class_df):
        result = _run_classification(three_class_df, "gradient_boosting", ["a", "b"], "target", {"n_estimators": 20})[
            "result"
        ]
        classes = result["classes"]
        assert classes == ["high", "low", "mid"]
        values = np.asarray(result["proba"]["values"])
        for sid, row in zip(result["proba"]["shot_ids"], values):
            assert result["labels"][str(sid)] == classes[int(row.argmax())]

    def test_unlabelled_shots_are_still_predicted(self, three_class_df):
        df = three_class_df.copy()
        df.loc[:9, "target"] = np.nan
        result = _run_classification(df, "random_forest", ["a", "b"], "target", {"n_estimators": 20})["result"]
        assert result["metrics"]["n_train"] + result["metrics"]["n_test"] == len(df) - 10
        assert len(result["labels"]) == len(df)

    def test_metrics_report_the_held_out_split(self, three_class_df):
        metrics = _run_classification(
            three_class_df, "random_forest", ["a", "b"], "target", {"n_estimators": 20}, test_fraction=0.25
        )["result"]["metrics"]
        assert metrics["n_test"] == pytest.approx(len(three_class_df) * 0.25, abs=1)
        assert 0.0 <= metrics["test_accuracy"] <= 1.0
        assert 0.0 <= metrics["test_f1"] <= 1.0

    def test_zero_test_fraction_trains_on_everything(self, three_class_df):
        metrics = _run_classification(
            three_class_df, "random_forest", ["a", "b"], "target", {"n_estimators": 20}, test_fraction=0.0
        )["result"]["metrics"]
        assert metrics["n_test"] == 0
        assert "test_accuracy" not in metrics

    def test_gaussian_process_reports_its_subsample(self, three_class_df):
        metrics = _run_classification(three_class_df, "gaussian_process", ["a", "b"], "target", {"max_train_rows": 30})[
            "result"
        ]["metrics"]
        assert 0 < metrics["subsampled"] <= 30 + 3  # one row per class may be rounded up

    def test_seed_makes_the_fit_reproducible(self, three_class_df):
        args = (three_class_df, "random_forest", ["a", "b"], "target", {"n_estimators": 20})
        first = _run_classification(*args, seed=7)["result"]
        second = _run_classification(*args, seed=7)["result"]
        assert first["labels"] == second["labels"]

    def test_unknown_algorithm_raises(self, labelled_df):
        with pytest.raises(ValueError, match="Unknown algorithm"):
            _run_classification(labelled_df, "svm", ["feature_1"], "target")

    def test_missing_target_raises(self, labelled_df):
        with pytest.raises(ValueError, match="target column"):
            _run_classification(labelled_df, "random_forest", ["feature_1"], "not_a_column")

    def test_no_usable_features_raises(self, labelled_df):
        with pytest.raises(ValueError, match="feature column"):
            _run_classification(labelled_df, "random_forest", ["not_a_column"], "target")

    def test_single_class_target_raises(self, labelled_df):
        df = labelled_df.copy()
        df["target"] = 1
        with pytest.raises(ValueError, match="fewer than two classes"):
            _run_classification(df, "random_forest", ["feature_1"], "target")

    def test_too_few_labelled_shots_raises(self, labelled_df):
        """Three rows over two classes is not a training set, whatever sklearn says."""
        df = labelled_df.copy()
        df["target"] = np.nan
        df.loc[df.index[:3], "target"] = [0, 0, 1]
        with pytest.raises(ValueError, match="Not enough labelled shots"):
            _run_classification(df, "random_forest", ["feature_1"], "target")


class TestApplyClassColor:
    def test_adds_a_column_named_label(self):
        plot_df = pd.DataFrame({"shot_id": [1, 2], "umap_x": [0.0, 1.0]})
        enriched, col = _apply_class_color(plot_df, {"1": "disrupted", "2": "clean"})
        assert col == "label"
        assert list(enriched["label"]) == ["disrupted", "clean"]

    def test_shots_without_a_prediction_are_dropped(self):
        plot_df = pd.DataFrame({"shot_id": [1, 2, 3], "umap_x": [0.0, 1.0, 2.0]})
        enriched, _ = _apply_class_color(plot_df, {"1": "a", "3": "b"})
        assert list(enriched["shot_id"]) == [1, 3]

    def test_the_input_frame_is_not_modified(self):
        plot_df = pd.DataFrame({"shot_id": [1], "umap_x": [0.0]})
        _apply_class_color(plot_df, {"1": "a"})
        assert "label" not in plot_df.columns


class TestDecisionSurface:
    @pytest.fixture
    def points(self):
        rng = np.random.default_rng(0)
        xy = rng.normal(size=(80, 2))
        return xy, (xy[:, 0] > 0).astype(float)

    def test_grid_shape_and_range(self, points):
        xy, p = points
        grid = decision_surface(xy, p, resolution=16)
        assert len(grid["x"]) == 16
        assert len(grid["y"]) == 16
        assert len(grid["z"]) == 16 and all(len(row) == 16 for row in grid["z"])

    def test_probabilities_stay_in_range(self, points):
        xy, p = points
        z = np.array(decision_surface(xy, p, resolution=16)["z"], dtype=float)
        finite = z[np.isfinite(z)]
        assert finite.size
        assert finite.min() >= 0.0 and finite.max() <= 1.0

    def test_the_boundary_falls_where_the_classes_meet(self, points):
        xy, p = points
        grid = decision_surface(xy, p, resolution=21, mask_dist=1.0)
        z = np.array(grid["z"], dtype=float)
        gx = np.array(grid["x"])
        # Left of x=0 is class 0, right of it is class 1.
        assert np.nanmean(z[:, gx < -0.5]) < 0.25
        assert np.nanmean(z[:, gx > 0.5]) > 0.75

    def test_cells_far_from_any_shot_are_masked(self):
        # Two tight blobs in opposite corners leave the middle unevidenced.
        xy = np.concatenate([np.zeros((20, 2)) + 0.01 * np.arange(20)[:, None], np.ones((20, 2))])
        p = np.concatenate([np.zeros(20), np.ones(20)])
        z = np.array(decision_surface(xy, p, resolution=21, mask_dist=0.05)["z"], dtype=float)
        assert np.isnan(z).any()

    def test_too_few_points_returns_none(self):
        assert decision_surface(np.zeros((2, 2)), np.zeros(2)) is None

    def test_a_degenerate_axis_returns_none(self):
        xy = np.column_stack([np.zeros(10), np.arange(10.0)])
        assert decision_surface(xy, np.zeros(10)) is None

    def test_non_finite_rows_are_dropped(self, points):
        xy, p = points
        xy = xy.copy()
        xy[0] = np.inf
        assert decision_surface(xy, p, resolution=8) is not None


class TestDecisionSurfaceOnLogAxes:
    @pytest.fixture
    def decades(self):
        """Points spread over four decades, where a linear grid would bunch up."""
        rng = np.random.default_rng(1)
        x = 10.0 ** rng.uniform(0, 4, size=80)
        y = rng.normal(size=80)
        return np.column_stack([x, y]), (x > 100).astype(float)

    def test_a_log_axis_is_sampled_evenly_in_log10(self, decades):
        xy, p = decades
        grid = decision_surface(xy, p, resolution=16, log_axes=(True, False))
        x = np.asarray(grid["x"])
        assert (x > 0).all()
        steps = np.diff(np.log10(x))
        assert np.allclose(steps, steps[0])

    def test_the_other_axis_stays_linear(self, decades):
        xy, p = decades
        grid = decision_surface(xy, p, resolution=16, log_axes=(True, False))
        steps = np.diff(np.asarray(grid["y"]))
        assert np.allclose(steps, steps[0])

    def test_both_axes_can_be_logarithmic(self):
        rng = np.random.default_rng(2)
        xy = 10.0 ** rng.uniform(0, 3, size=(60, 2))
        grid = decision_surface(xy, (xy[:, 0] > 30).astype(float), resolution=12, log_axes=(True, True))
        for axis in ("x", "y"):
            steps = np.diff(np.log10(np.asarray(grid[axis])))
            assert np.allclose(steps, steps[0])

    def test_the_boundary_lands_in_the_right_decade(self, decades):
        """A linear grid over four decades would put almost every cell above 1000."""
        xy, p = decades
        grid = decision_surface(xy, p, resolution=41, log_axes=(True, False), mask_dist=1.0)
        z = np.array(grid["z"], dtype=float)
        x = np.asarray(grid["x"])
        assert np.nanmean(z[:, x < 10]) < 0.25
        assert np.nanmean(z[:, x > 1000]) > 0.75

    def test_non_positive_values_are_dropped_on_a_log_axis(self, decades):
        xy, p = decades
        xy = xy.copy()
        xy[:3, 0] = [0.0, -1.0, -50.0]
        grid = decision_surface(xy, p, resolution=8, log_axes=(True, False))
        assert grid is not None
        assert min(grid["x"]) > 0

    def test_a_point_is_dropped_when_either_log_axis_cannot_place_it(self):
        """Plotly omits such a point too, so the surface must not be fitted to it."""
        rng = np.random.default_rng(4)
        xy = 10.0 ** rng.uniform(0, 3, size=(60, 2))
        xy[0] = [500.0, -1.0]  # fine on x, impossible on y
        grid = decision_surface(xy, (xy[:, 0] > 30).astype(float), resolution=12, log_axes=(True, True))
        assert grid is not None
        # The outlier's x would have widened the grid had it been kept.
        assert min(grid["y"]) > 0

    def test_too_few_positive_values_returns_none(self):
        xy = np.column_stack([[-1.0, -2.0, -3.0, 4.0], np.arange(4.0)])
        assert decision_surface(xy, np.zeros(4), log_axes=(True, False)) is None

    def test_the_log_grid_differs_from_the_linear_one(self, decades):
        xy, p = decades
        linear = decision_surface(xy, p, resolution=16)
        logged = decision_surface(xy, p, resolution=16, log_axes=(True, False))
        assert not np.allclose(np.asarray(linear["x"]), np.asarray(logged["x"]))
