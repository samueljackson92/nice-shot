"""Tests for nice_shot/backends.py — data loading backends."""

from __future__ import annotations

import pandas as pd
import pytest

from nice_shot.backends import (
    BackendConfig,
    CsvShotDataBackend,
    FairMastTraceBackend,
    LocalParquetTraceBackend,
    LongParquetShotDataBackend,
    ParquetShotDataBackend,
    create_shot_data_backend,
    create_trace_backend,
    create_variable_shot_data_backend,
    detect_shot_col,
)


class TestDetectShotCol:
    @pytest.mark.parametrize("name", ["shot_id", "shot", "pulse", "number", "exp_number", "pulse_id", "shot_number"])
    def test_resolves_each_candidate(self, name):
        df = pd.DataFrame({name: [1, 2, 3], "other": [4, 5, 6]})
        assert detect_shot_col(df) == name

    def test_raises_when_no_candidate_present(self):
        df = pd.DataFrame({"foo": [1, 2, 3]})
        with pytest.raises(ValueError, match="Could not detect shot ID column"):
            detect_shot_col(df)


class TestShotDataBackendPrepare:
    def test_coerces_numeric_object_columns(self):
        df = pd.DataFrame({"shot_id": [1, 2], "value": ["1.5", "2.5"], "machine": ["MAST-U", "MAST-U"]})
        backend = CsvShotDataBackend(BackendConfig())
        result = backend._prepare(df)
        assert pd.api.types.is_float_dtype(result["value"])
        assert pd.api.types.is_object_dtype(result["machine"])

    def test_renames_shot_column(self):
        df = pd.DataFrame({"pulse": [1, 2], "value": [1.0, 2.0]})
        backend = CsvShotDataBackend(BackendConfig())
        result = backend._prepare(df)
        assert "shot_id" in result.columns
        assert "pulse" not in result.columns

    def test_coerce_objects_false_skips_coercion(self):
        df = pd.DataFrame({"shot_id": [1, 2], "value": ["1.5", "2.5"]})
        backend = CsvShotDataBackend(BackendConfig())
        result = backend._prepare(df, coerce_objects=False)
        assert pd.api.types.is_object_dtype(result["value"])


class TestCsvAndParquetBackends:
    def test_csv_load_roundtrip(self, tmp_csv_path):
        backend = CsvShotDataBackend(BackendConfig())
        result = backend.load(tmp_csv_path)
        assert "shot_id" in result.columns
        assert len(result) == 16

    def test_parquet_load_roundtrip(self, tmp_parquet_path):
        backend = ParquetShotDataBackend(BackendConfig())
        result = backend.load(tmp_parquet_path)
        assert "shot_id" in result.columns
        assert len(result) == 16


class TestLongParquetShotDataBackend:
    @pytest.fixture
    def long_format_path(self, tmp_path):
        df = pd.DataFrame(
            {
                "shot_id": [1, 2, 1, 2],
                "variable": ["ip", "ip", "ne", "ne"],
                "value": [1.0, 2.0, 3.0, 4.0],
            }
        )
        path = tmp_path / "long.parquet"
        df.to_parquet(path, index=False)
        return str(path)

    @pytest.fixture
    def backend(self):
        return LongParquetShotDataBackend(BackendConfig(options={"variable_column": "variable"}))

    def test_variables(self, backend, long_format_path):
        assert backend.variables(long_format_path) == ["ip", "ne"]

    def test_schema_has_no_rows(self, backend, long_format_path):
        schema = backend.schema(long_format_path)
        assert len(schema) == 0
        assert "variable" not in schema.columns

    def test_load_variable_filters_rows(self, backend, long_format_path):
        result = backend.load_variable(long_format_path, "ip")
        assert len(result) == 2
        assert "variable" not in result.columns
        assert set(result["shot_id"]) == {1, 2}


class TestLocalParquetTraceBackend:
    def test_is_available_false_for_missing_dir(self, tmp_path):
        backend = LocalParquetTraceBackend(BackendConfig(data_dir=str(tmp_path / "missing")))
        assert backend.is_available() is False

    def test_is_available_false_for_empty_dir(self, tmp_path):
        backend = LocalParquetTraceBackend(BackendConfig(data_dir=str(tmp_path)))
        assert backend.is_available() is False

    def test_is_available_true_when_files_present(self, tmp_path):
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "123.parquet").write_bytes(b"")
        backend = LocalParquetTraceBackend(BackendConfig(data_dir=str(tmp_path)))
        assert backend.is_available() is True

    def test_find_shot_file(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        trace_df = pd.DataFrame({"time": [0.0, 0.5, 1.0], "ip": [1.0, 2.0, 3.0]})
        trace_df.to_parquet(sub / "123.parquet", index=False)
        backend = LocalParquetTraceBackend(BackendConfig(data_dir=str(tmp_path)))
        assert backend.find_shot_file(123) == str(sub / "123.parquet")
        assert backend.find_shot_file(999) is None

    def test_load_filters_time_window(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        trace_df = pd.DataFrame({"time": [0.0, 0.5, 1.0, 1.5], "ip": [1.0, 2.0, 3.0, 4.0]})
        trace_df.to_parquet(sub / "123.parquet", index=False)
        backend = LocalParquetTraceBackend(BackendConfig(data_dir=str(tmp_path), min_time=0.4, max_time=1.1))
        result = backend.load(123)
        assert result is not None
        assert list(result["time"]) == [0.5, 1.0]

    def test_load_returns_none_when_shot_not_found(self, tmp_path):
        backend = LocalParquetTraceBackend(BackendConfig(data_dir=str(tmp_path)))
        assert backend.load(999) is None


class TestRegistry:
    def test_create_shot_data_backend_csv(self):
        backend = create_shot_data_backend("data.csv", BackendConfig())
        assert isinstance(backend, CsvShotDataBackend)

    def test_create_shot_data_backend_parquet(self):
        backend = create_shot_data_backend("data.parquet", BackendConfig())
        assert isinstance(backend, ParquetShotDataBackend)

    def test_create_shot_data_backend_unknown_extension_raises(self):
        with pytest.raises(ValueError, match="No shot data backend registered"):
            create_shot_data_backend("data.xyz", BackendConfig())

    def test_create_variable_shot_data_backend_unknown_extension_raises(self):
        with pytest.raises(ValueError, match="no long-format shot data backend"):
            create_variable_shot_data_backend("data.xyz", BackendConfig())

    def test_create_trace_backend_parquet(self):
        backend = create_trace_backend("parquet", BackendConfig())
        assert isinstance(backend, LocalParquetTraceBackend)

    def test_create_trace_backend_unknown_name_raises(self):
        with pytest.raises(ValueError, match="No trace backend registered"):
            create_trace_backend("nonexistent", BackendConfig())

    def test_create_trace_backend_fairmast(self):
        backend = create_trace_backend("fairmast", BackendConfig())
        assert isinstance(backend, FairMastTraceBackend)
