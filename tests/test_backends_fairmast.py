"""Tests for nice_shot/backends.py — FairMastTraceBackend.

Kept in a separate module (rather than tests/test_backends.py) so that
importorskip for the optional 'fairmast' extra only skips these tests, not
the whole backends test suite, when zarr/h5netcdf aren't installed.
"""

from __future__ import annotations

import numpy as np
import pytest

from nice_shot.backends import BackendConfig, FairMastTraceBackend

zarr = pytest.importorskip("zarr")
h5netcdf = pytest.importorskip("h5netcdf")
xr = pytest.importorskip("xarray")


class TestFairMastTraceBackend:
    @pytest.fixture(params=["zarr", "netcdf"])
    def fmt(self, request):
        return request.param

    @pytest.fixture
    def shot_store(self, tmp_path, fmt):
        """Write a per-shot store (shot 123) with two groups: 'magnetics'
        (scalar ip) and 'thomson_scattering' (2-D profile t_e), plus a
        root-group scalar 'ip'."""
        ext = "zarr" if fmt == "zarr" else "nc"
        path = tmp_path / f"123.{ext}"

        time = np.array([0.0, 0.5, 1.0, 1.5])
        root = xr.Dataset({"ip": ("time", [10.0, 20.0, 30.0, 40.0])}, coords={"time": time})
        magnetics = xr.Dataset({"ip": ("time", [1.0, 2.0, 3.0, 4.0])}, coords={"time": time})
        thomson = xr.Dataset(
            {"t_e": (("time", "channel"), np.zeros((4, 3)))},
            coords={"time": time, "channel": [0, 1, 2]},
        )

        if fmt == "zarr":
            root.to_zarr(path, mode="w")
            magnetics.to_zarr(path, group="magnetics", mode="a")
            thomson.to_zarr(path, group="thomson_scattering", mode="a")
        else:
            root.to_netcdf(path, mode="w", engine="h5netcdf")
            magnetics.to_netcdf(path, group="magnetics", mode="a", engine="h5netcdf")
            thomson.to_netcdf(path, group="thomson_scattering", mode="a", engine="h5netcdf")

        return tmp_path, fmt

    def _backend(self, data_dir, fmt, signals, **config_kwargs):
        return FairMastTraceBackend(
            BackendConfig(data_dir=str(data_dir), signals=signals, options={"format": fmt}, **config_kwargs)
        )

    def test_is_available_true_for_present_dir(self, shot_store):
        tmp_path, fmt = shot_store
        backend = self._backend(tmp_path, fmt, [])
        assert backend.is_available() is True

    def test_is_available_false_for_missing_dir(self, tmp_path):
        backend = self._backend(tmp_path / "missing", "zarr", [])
        assert backend.is_available() is False

    def test_load_scalar_signal(self, shot_store):
        tmp_path, fmt = shot_store
        backend = self._backend(tmp_path, fmt, ["magnetics/ip"], min_time=0.0, max_time=1.5)
        result = backend.load(123)
        assert result is not None
        assert list(result["magnetics/ip"]) == [1.0, 2.0, 3.0, 4.0]

    def test_load_filters_time_window(self, shot_store):
        tmp_path, fmt = shot_store
        backend = self._backend(tmp_path, fmt, ["magnetics/ip"], min_time=0.4, max_time=1.1)
        result = backend.load(123)
        assert result is not None
        assert list(result["time"]) == [0.5, 1.0]

    def test_load_root_group_signal(self, shot_store):
        tmp_path, fmt = shot_store
        backend = self._backend(tmp_path, fmt, ["ip"], min_time=0.0, max_time=1.5)
        result = backend.load(123)
        assert result is not None
        assert list(result["ip"]) == [10.0, 20.0, 30.0, 40.0]

    def test_load_multiple_signals_share_group(self, shot_store):
        tmp_path, fmt = shot_store
        backend = self._backend(tmp_path, fmt, ["magnetics/ip", "ip"], min_time=0.0, max_time=1.5)
        result = backend.load(123)
        assert result is not None
        assert list(result["magnetics/ip"]) == [1.0, 2.0, 3.0, 4.0]
        assert list(result["ip"]) == [10.0, 20.0, 30.0, 40.0]

    def test_load_skips_profile_signal_without_crashing_others(self, shot_store):
        tmp_path, fmt = shot_store
        backend = self._backend(tmp_path, fmt, ["magnetics/ip", "thomson_scattering/t_e"], min_time=0.0, max_time=1.5)
        result = backend.load(123)
        assert result is not None
        assert "magnetics/ip" in result.columns
        assert "thomson_scattering/t_e" not in result.columns

    def test_load_returns_none_when_shot_not_found(self, shot_store):
        tmp_path, fmt = shot_store
        backend = self._backend(tmp_path, fmt, ["magnetics/ip"])
        assert backend.load(999) is None

    def test_load_returns_none_when_all_signals_missing(self, shot_store):
        tmp_path, fmt = shot_store
        backend = self._backend(tmp_path, fmt, ["nope/nope"])
        assert backend.load(123) is None

    def test_load_via_file_url(self, shot_store):
        tmp_path, fmt = shot_store
        backend = self._backend(f"file://{tmp_path}", fmt, ["magnetics/ip"], min_time=0.0, max_time=1.5)
        result = backend.load(123)
        assert result is not None
        assert list(result["magnetics/ip"]) == [1.0, 2.0, 3.0, 4.0]
