"""Tests for projection logic in nice_shot/analysis.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nice_shot.analysis import (
    _compute_projection,
    _fit_projection,
    _load_projection_file,
    _projection_feature_cols,
    _transform_projection,
)


class TestProjectionFeatureCols:
    def test_defaults_to_numeric_columns_excluding_shot_id(self, synthetic_shot_df):
        cols = _projection_feature_cols(synthetic_shot_df)
        assert "shot_id" not in cols
        assert "feature_1" in cols
        assert "machine" not in cols  # non-numeric

    def test_umap_features_allow_list(self, synthetic_shot_df):
        cols = _projection_feature_cols(synthetic_shot_df, umap_features=["feature_1"])
        assert cols == ["feature_1"]

    def test_umap_features_drops_missing_columns(self, synthetic_shot_df):
        cols = _projection_feature_cols(synthetic_shot_df, umap_features=["feature_1", "does_not_exist"])
        assert cols == ["feature_1"]

    def test_umap_exclude_features(self, synthetic_shot_df):
        cols = _projection_feature_cols(synthetic_shot_df, umap_exclude_features=["feature_1"])
        assert "feature_1" not in cols
        assert "feature_2" in cols


class TestComputeProjection:
    def test_pca_output_shape(self, synthetic_shot_df):
        projection, shot_ids = _compute_projection(synthetic_shot_df, method="pca")
        assert projection.shape == (len(synthetic_shot_df), 2)
        assert len(shot_ids) == len(synthetic_shot_df)

    def test_all_nan_column_is_dropped_not_fatal(self, synthetic_shot_df):
        # feature_sparse is mostly-but-not-all NaN; a fully NaN column should be
        # dropped silently rather than raising.
        df = synthetic_shot_df.copy()
        df["feature_all_nan"] = np.nan
        projection, _ = _compute_projection(df, method="pca")
        assert projection.shape == (len(df), 2)

    def test_zero_variance_column_dropped(self, synthetic_shot_df):
        # feature_const has zero variance; the projection must still succeed
        # using the two remaining columns (PCA needs >= 2 features for n_components=2).
        projection, _ = _compute_projection(
            synthetic_shot_df, method="pca", umap_features=["feature_1", "feature_2", "feature_const"]
        )
        assert projection.shape == (len(synthetic_shot_df), 2)

    def test_raises_when_no_usable_feature_columns(self):
        df = pd.DataFrame({"shot_id": [1, 2, 3], "machine": ["a", "b", "c"]})
        with pytest.raises(ValueError, match="No usable feature columns"):
            _compute_projection(df, method="pca")

    def test_raises_when_all_features_are_nan(self):
        df = pd.DataFrame({"shot_id": [1, 2, 3], "feature_1": [np.nan, np.nan, np.nan]})
        with pytest.raises(ValueError, match="entirely NaN"):
            _compute_projection(df, method="pca")

    def test_raises_when_no_finite_variance_columns_remain(self):
        df = pd.DataFrame({"shot_id": [1, 2, 3], "feature_1": [5.0, 5.0, 5.0]})
        with pytest.raises(ValueError, match="finite variance"):
            _compute_projection(df, method="pca")


class TestFitAndTransformProjection:
    """_fit_projection / _transform_projection: the split that lets new shots be
    projected without refitting. Uses method="pca" throughout for determinism
    (umap-learn's fit is nondeterministic/slow and not worth testing bit-for-bit).
    """

    def test_fit_projection_returns_populated_model(self, synthetic_shot_df):
        model, projection, shot_ids = _fit_projection(synthetic_shot_df, method="pca")
        assert projection.shape == (len(synthetic_shot_df), 2)
        assert len(shot_ids) == len(synthetic_shot_df)
        assert model.method == "pca"
        # feature_const has zero variance -- dropped after imputation, before scaling.
        assert set(model.imputer_cols) == {"feature_1", "feature_2", "feature_const", "feature_sparse"}
        assert set(model.scaler_cols) == {"feature_1", "feature_2", "feature_sparse"}

    def test_fit_projection_drops_all_nan_column_from_imputer_cols(self, synthetic_shot_df):
        df = synthetic_shot_df.copy()
        df["feature_all_nan"] = np.nan
        model, _, _ = _fit_projection(df, method="pca")
        assert "feature_all_nan" not in model.imputer_cols

    def test_transform_never_calls_fit(self, synthetic_shot_df, monkeypatch):
        from sklearn.decomposition import PCA

        model, _, _ = _fit_projection(synthetic_shot_df, method="pca")

        def _boom(*args, **kwargs):
            raise AssertionError("fit should not be called by _transform_projection")

        monkeypatch.setattr(PCA, "fit", _boom)
        monkeypatch.setattr(PCA, "fit_transform", _boom)

        new_row = synthetic_shot_df.iloc[[0]].copy()
        new_row["shot_id"] = 9999
        coords, shot_ids = _transform_projection(model, new_row)
        assert coords.shape == (1, 2)
        assert list(shot_ids) == [9999]

    def test_transform_reproduces_fit_coordinates_for_known_rows(self, synthetic_shot_df):
        # Transforming rows that were part of the original fit, through that same
        # fitted model, must reproduce fit_transform's own output for those rows.
        model, full_projection, full_ids = _fit_projection(synthetic_shot_df, method="pca")
        full_ids_list = list(full_ids)
        subset = synthetic_shot_df.iloc[[2, 3, 10, 11]]

        transformed, transformed_ids = _transform_projection(model, subset)

        for shot_id, coord in zip(transformed_ids, transformed):
            idx = full_ids_list.index(shot_id)
            np.testing.assert_allclose(coord, full_projection[idx], atol=1e-8)

    def test_transform_projects_new_shot_near_similar_existing_shots(self, synthetic_shot_df):
        # A genuinely new shot (not part of the fit) with cluster_a-like feature
        # values should land near cluster_a's existing points, not cluster_b's --
        # comparisons stay within the one model's coordinate space, so this is
        # robust to PCA's arbitrary component sign.
        model, full_projection, full_ids = _fit_projection(synthetic_shot_df, method="pca")
        full_ids_list = list(full_ids)

        new_row = pd.DataFrame(
            {
                "shot_id": [99999],
                "feature_1": [0.1],
                "feature_2": [0.2],
                "feature_const": [5.0],
                "feature_sparse": [np.nan],
                "machine": ["MAST-U"],
            }
        )
        coords, _ = _transform_projection(model, new_row)
        new_coord = coords[0]

        cluster_a_coord = full_projection[full_ids_list.index(1000)]
        cluster_b_coord = full_projection[full_ids_list.index(1008)]
        assert np.linalg.norm(new_coord - cluster_a_coord) < np.linalg.norm(new_coord - cluster_b_coord)

    def test_transform_imputes_missing_feature_column(self, synthetic_shot_df):
        model, _, _ = _fit_projection(synthetic_shot_df, method="pca")
        new_row = synthetic_shot_df.iloc[[0]].drop(columns=["feature_sparse"]).copy()
        new_row["shot_id"] = 9999
        coords, _ = _transform_projection(model, new_row)
        assert coords.shape == (1, 2)
        assert not np.isnan(coords).any()

    def test_compute_projection_wrapper_delegates_to_fit_projection(self, synthetic_shot_df):
        projection, shot_ids = _compute_projection(synthetic_shot_df, method="pca")
        assert projection.shape == (len(synthetic_shot_df), 2)
        assert len(shot_ids) == len(synthetic_shot_df)


class TestLoadProjectionFile:
    def test_npy_with_shot_id_column(self, tmp_path, synthetic_shot_df):
        arr = np.column_stack(
            [synthetic_shot_df["shot_id"].values, np.arange(len(synthetic_shot_df)), np.arange(len(synthetic_shot_df))]
        )
        path = tmp_path / "proj.npy"
        np.save(path, arr)
        result, x_label, y_label = _load_projection_file(str(path), synthetic_shot_df)
        assert list(result.columns) == ["shot_id", "umap_x", "umap_y"]
        assert (x_label, y_label) == ("Dim 1", "Dim 2")

    def test_npy_positional_matching_two_columns(self, tmp_path, synthetic_shot_df):
        arr = np.column_stack([np.arange(len(synthetic_shot_df)), np.arange(len(synthetic_shot_df))])
        path = tmp_path / "proj.npy"
        np.save(path, arr.astype(float))
        result, _, _ = _load_projection_file(str(path), synthetic_shot_df)
        assert list(result["shot_id"]) == list(synthetic_shot_df["shot_id"])

    def test_npy_row_count_mismatch_raises(self, tmp_path, synthetic_shot_df):
        arr = np.zeros((len(synthetic_shot_df) - 1, 2))
        path = tmp_path / "proj.npy"
        np.save(path, arr)
        with pytest.raises(ValueError, match="rows but shot data has"):
            _load_projection_file(str(path), synthetic_shot_df)

    def test_npy_bad_shape_raises(self, tmp_path, synthetic_shot_df):
        arr = np.zeros((5,))
        path = tmp_path / "proj.npy"
        np.save(path, arr)
        with pytest.raises(ValueError, match="must be 2-D"):
            _load_projection_file(str(path), synthetic_shot_df)

    def test_csv_projection(self, tmp_path, synthetic_shot_df):
        n = len(synthetic_shot_df)
        emb = pd.DataFrame({"shot_id": synthetic_shot_df["shot_id"], "x": range(n), "y": range(n)})
        path = tmp_path / "proj.csv"
        emb.to_csv(path, index=False)
        result, x_label, y_label = _load_projection_file(str(path), synthetic_shot_df)
        assert (x_label, y_label) == ("x", "y")
        assert list(result.columns) == ["shot_id", "umap_x", "umap_y"]

    def test_csv_missing_coordinate_columns_raises(self, tmp_path, synthetic_shot_df):
        emb = pd.DataFrame({"shot_id": synthetic_shot_df["shot_id"], "x": range(len(synthetic_shot_df))})
        path = tmp_path / "proj.csv"
        emb.to_csv(path, index=False)
        with pytest.raises(ValueError, match="at least 2 coordinate columns"):
            _load_projection_file(str(path), synthetic_shot_df)

    def test_unsupported_extension_raises(self, tmp_path, synthetic_shot_df):
        path = tmp_path / "proj.txt"
        path.write_text("nope")
        with pytest.raises(ValueError, match="Unsupported projection format"):
            _load_projection_file(str(path), synthetic_shot_df)
