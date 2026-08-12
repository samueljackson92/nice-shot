"""Tests for the root-level ETL scripts extract_shot_equilibrium.py and extract_mean_latents.py.

These scripts are not part of the installed ``nice_shot`` package; they are
importable because ``pyproject.toml`` adds the repo root to ``pythonpath`` for
pytest.
"""

from __future__ import annotations

import pandas as pd
import pytest

from extract_mean_latents import extract_mean, merge_cpf, merge_equilibrium, merge_fair, merge_shots
from extract_shot_equilibrium import extract_shot_id


class TestExtractShotId:
    def test_parses_shot_id_from_zarr_path(self):
        assert extract_shot_id("/data/fairmast/upload-tmp/level2/11766.zarr") == 11766

    def test_parses_shot_id_from_bare_filename(self):
        assert extract_shot_id("30420.zarr") == 30420

    def test_raises_on_non_numeric_stem(self):
        with pytest.raises(ValueError):
            extract_shot_id("/data/level2/not_a_number.zarr")


class TestExtractMean:
    def test_filters_to_mean_aggregation(self, tmp_path):
        df = pd.DataFrame(
            {
                "shot_id": [1, 1, 2, 2],
                "aggregation": ["mean", "std", "mean", "std"],
                "value": [1.0, 2.0, 3.0, 4.0],
            }
        )
        path = tmp_path / "latents.parquet"
        df.to_parquet(path, index=False)
        result = extract_mean(path)
        assert list(result["aggregation"].unique()) == ["mean"]
        assert len(result) == 2


class TestMergeCpf:
    def test_left_join_coerces_cpf_shot_id(self, tmp_path):
        mean_df = pd.DataFrame({"shot_id": [1, 2], "value": [10.0, 20.0]})
        cpf = pd.DataFrame({"shot_id": ["1", "3"], "cpf_col": ["a", "b"]})
        path = tmp_path / "cpf.parquet"
        cpf.to_parquet(path, index=False)
        result = merge_cpf(mean_df, path)
        assert len(result) == 2  # left join keeps mean_df's row count
        row1 = result[result["shot_id"] == 1].iloc[0]
        assert row1["cpf_col"] == "a"
        row2 = result[result["shot_id"] == 2].iloc[0]
        assert pd.isna(row2["cpf_col"])


class TestMergeShots:
    def test_renames_number_to_shot_id_and_suffixes_collisions(self, tmp_path):
        df = pd.DataFrame({"shot_id": [1, 2], "value": [10.0, 20.0]})
        shots = pd.DataFrame({"number": [1, 2], "value": ["x", "y"], "datetime": ["2020-01-01", "2020-01-02"]})
        path = tmp_path / "shots.csv"
        shots.to_csv(path, index=False)
        result = merge_shots(df, path)
        assert "value_shots" in result.columns
        assert list(result["datetime"]) == ["2020-01-01", "2020-01-02"]


class TestMergeFair:
    def test_fills_missing_categoricals_with_nan_string(self, tmp_path):
        df = pd.DataFrame({"shot_id": [1, 2]})
        fair = pd.DataFrame(
            {
                "shot_id": [1, 2],
                "plasma_shape": ["circular", None],
                "current_range": [None, "high"],
            }
        )
        path = tmp_path / "fair.parquet"
        fair.to_parquet(path, index=False)
        result = merge_fair(df, path)
        row2 = result[result["shot_id"] == 2].iloc[0]
        assert row2["plasma_shape"] == "nan"
        row1 = result[result["shot_id"] == 1].iloc[0]
        assert row1["current_range"] == "nan"


class TestMergeEquilibrium:
    def test_pivots_variable_name_into_columns(self, tmp_path):
        df = pd.DataFrame({"shot_id": [11766, 30420]})
        index = pd.MultiIndex.from_tuples(
            [
                ("/data/level2/11766.zarr", "ip"),
                ("/data/level2/11766.zarr", "ne"),
                ("/data/level2/30420.zarr", "ip"),
            ],
            names=["file_path", "variable_name"],
        )
        equil = pd.DataFrame({"mean": [1.5, 2.5, 3.5]}, index=index)
        path = tmp_path / "equil.parquet"
        equil.to_parquet(path)
        result = merge_equilibrium(df, path)
        assert "ip" in result.columns and "ne" in result.columns
        row = result[result["shot_id"] == 11766].iloc[0]
        assert row["ip"] == 1.5
        assert row["ne"] == 2.5
        row2 = result[result["shot_id"] == 30420].iloc[0]
        assert pd.isna(row2["ne"])
