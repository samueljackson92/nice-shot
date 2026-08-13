"""Tests for nice_shot/backends.py's SqlShotDataBackend — a generic, driver-agnostic
SQL backend built on SQLAlchemy. Exercised against a real temporary sqlite database
(stdlib ``sqlite3``), matching this test suite's "test backends against the real
thing" style rather than mocking the database layer.
"""

from __future__ import annotations

import sqlite3

import pytest

from nice_shot.backends import (
    BackendConfig,
    SqlShotDataBackend,
    create_shot_data_backend,
)


def _make_sqlite_db(path, table="shots", shot_col="shot_id"):
    con = sqlite3.connect(path)
    con.execute(f"CREATE TABLE {table} ({shot_col} INTEGER, feature_1 REAL, feature_2 REAL)")
    con.executemany(
        f"INSERT INTO {table} VALUES (?, ?, ?)",
        [(1000, 0.1, 1.1), (1001, 0.2, 1.2), (1002, 0.3, 1.3)],
    )
    con.commit()
    con.close()


@pytest.fixture
def sqlite_path(tmp_path):
    path = tmp_path / "shots.sqlite"
    _make_sqlite_db(str(path))
    return str(path)


class TestLoad:
    def test_zero_config_load_roundtrip(self, sqlite_path):
        backend = SqlShotDataBackend(BackendConfig())
        result = backend.load(sqlite_path)
        assert "shot_id" in result.columns
        assert len(result) == 3
        assert set(result["shot_id"]) == {1000, 1001, 1002}

    def test_shot_table_option(self, tmp_path):
        path = tmp_path / "data.sqlite"
        _make_sqlite_db(str(path), table="pulses")
        backend = SqlShotDataBackend(BackendConfig(options={"shot_table": "pulses"}))
        result = backend.load(str(path))
        assert len(result) == 3

    def test_query_option_overrides_shot_table(self, sqlite_path):
        backend = SqlShotDataBackend(BackendConfig(options={"query": "SELECT * FROM shots WHERE shot_id > 1000"}))
        result = backend.load(sqlite_path)
        assert len(result) == 2
        assert set(result["shot_id"]) == {1001, 1002}

    def test_shot_col_option_renames_before_prepare(self, tmp_path):
        # "pulse_number" isn't in SHOT_ID_CANDIDATES, so detect_shot_col alone
        # couldn't find it -- shot_col must rename it before _prepare() runs.
        path = tmp_path / "custom.sqlite"
        _make_sqlite_db(str(path), shot_col="pulse_number")
        backend = SqlShotDataBackend(BackendConfig(options={"shot_col": "pulse_number", "shot_table": "shots"}))
        result = backend.load(str(path))
        assert "shot_id" in result.columns
        assert "pulse_number" not in result.columns
        assert set(result["shot_id"]) == {1000, 1001, 1002}

    def test_db_extension_also_defaults_to_sqlite(self, tmp_path):
        path = tmp_path / "shots.db"
        _make_sqlite_db(str(path))
        backend = SqlShotDataBackend(BackendConfig())
        result = backend.load(str(path))
        assert len(result) == 3

    def test_url_option_takes_precedence_over_extension_default(self, tmp_path):
        real_path = tmp_path / "real.sqlite"
        _make_sqlite_db(str(real_path))
        # A .sqlite path that doesn't exist -- only reachable if the explicit
        # url is actually used instead of the path-derived default.
        missing_path = str(tmp_path / "does_not_exist.sqlite")
        backend = SqlShotDataBackend(BackendConfig(options={"url": f"sqlite:///{real_path}", "shot_table": "shots"}))
        result = backend.load(missing_path)
        assert len(result) == 3

    def test_sql_extension_without_url_raises(self, tmp_path):
        backend = SqlShotDataBackend(BackendConfig())
        with pytest.raises(ValueError, match="requires 'url'"):
            backend.load(str(tmp_path / "shots.sql"))


class TestPollNew:
    def test_returns_only_newer_rows(self, sqlite_path):
        backend = SqlShotDataBackend(BackendConfig())
        result = backend.poll_new(sqlite_path, since_shot_id=1000)
        assert set(result["shot_id"]) == {1001, 1002}

    def test_returns_empty_when_nothing_newer(self, sqlite_path):
        backend = SqlShotDataBackend(BackendConfig())
        result = backend.poll_new(sqlite_path, since_shot_id=1002)
        assert result.empty

    def test_respects_shot_col_option(self, tmp_path):
        path = tmp_path / "custom.sqlite"
        _make_sqlite_db(str(path), shot_col="pulse_number")
        backend = SqlShotDataBackend(BackendConfig(options={"shot_col": "pulse_number", "shot_table": "shots"}))
        result = backend.poll_new(str(path), since_shot_id=1000)
        assert set(result["shot_id"]) == {1001, 1002}

    def test_respects_query_option(self, sqlite_path):
        backend = SqlShotDataBackend(BackendConfig(options={"query": "SELECT * FROM shots WHERE feature_1 > 0.1"}))
        result = backend.poll_new(sqlite_path, since_shot_id=1000)
        assert set(result["shot_id"]) == {1001, 1002}


class TestRegistry:
    @pytest.mark.parametrize("ext", [".sqlite", ".db", ".sql"])
    def test_dispatches_to_sql_backend(self, ext):
        backend = create_shot_data_backend(f"data{ext}", BackendConfig())
        assert isinstance(backend, SqlShotDataBackend)
