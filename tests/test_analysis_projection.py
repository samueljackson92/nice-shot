"""Tests for projection logic in nice_shot/analysis.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nice_shot.analysis import _compute_projection, _load_projection_file, _projection_feature_cols


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
