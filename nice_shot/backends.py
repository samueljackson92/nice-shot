"""
Extensible backend system for NiceShot!

Two backend hierarchies are defined here:

  ShotDataBackend  — loads shot-statistics tabular data from a file.
  TraceBackend     — loads per-shot time-series traces on demand.

Built-in implementations are registered at the bottom of this module.
Custom backends can be added at runtime by:

  1. Subclassing ShotDataBackend or TraceBackend.
  2. Calling register_shot_data_backend() or register_trace_backend().
  3. Declaring the module path under `plugins:` in config.yaml so it is
     imported before the backends are created.

Example config.yaml entry::

    plugins:
      - my_package.my_backends

Example custom trace backend::

    from nice_shot.backends import BackendConfig, TraceBackend, register_trace_backend
    import pandas as pd

    class MyTraceBackend(TraceBackend):
        def load(self, shot_id: int) -> pd.DataFrame | None:
            # fetch data for shot_id and return a DataFrame with
            # a 'time' column and one column per signal, or None
            ...

        def is_available(self) -> bool:
            return True   # or check connectivity / file existence

    register_trace_backend("my_backend", MyTraceBackend)
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Candidate column names that may hold the shot ID in user data files.
# ---------------------------------------------------------------------------
SHOT_ID_CANDIDATES = [
    "shot_id",
    "shot",
    "pulse",
    "number",
    "exp_number",
    "pulse_id",
    "shot_number",
]


# ---------------------------------------------------------------------------
# Shot-ID column detection — used by shot data backends and projection loader.
# ---------------------------------------------------------------------------


def detect_shot_col(df: pd.DataFrame) -> str:
    """Return the name of the shot-ID column in *df*.

    Tries each name in :data:`SHOT_ID_CANDIDATES` in order and returns the
    first match. Raises :exc:`ValueError` if none are found.
    """
    for candidate in SHOT_ID_CANDIDATES:
        if candidate in df.columns:
            return candidate
    raise ValueError(
        f"Could not detect shot ID column. Expected one of {SHOT_ID_CANDIDATES}. Found: {list(df.columns)}"
    )


# ---------------------------------------------------------------------------
# BackendConfig — standardised config object passed to every backend.
# ---------------------------------------------------------------------------


@dataclass
class BackendConfig:
    """All configuration a backend might need.

    Built-in backends read only the fields relevant to them. Custom backends
    can use ``options`` for any extra key/value pairs declared in config.yaml
    under ``backend_options:``.
    """

    shot_data_path: str = ""
    data_dir: str = ""
    signals: list[str] = field(default_factory=list)
    min_time: float = 0.0
    max_time: float = 1.0
    timebase_hz: float | None = None
    options: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# ShotDataBackend — abstract base for tabular shot-statistics loaders.
# ---------------------------------------------------------------------------


class ShotDataBackend(ABC):
    """Base class for loaders that read shot-statistics files.

    Implementations must return a DataFrame whose first column (or a column
    detected via :data:`SHOT_ID_CANDIDATES`) is renamed to ``shot_id``.
    """

    def __init__(self, config: BackendConfig) -> None:
        self.config = config

    @abstractmethod
    def load(self, path: str) -> pd.DataFrame:
        """Load shot statistics from *path* and return a normalised DataFrame."""

    # ------------------------------------------------------------------
    # Shared helpers available to all subclasses.
    # ------------------------------------------------------------------

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """Coerce object columns to numeric and normalise the shot ID column."""
        obj_cols = df.select_dtypes(include="object").columns
        if len(obj_cols):
            coerced = df[obj_cols].apply(pd.to_numeric, errors="coerce")
            converted = [c for c in obj_cols if coerced[c].notna().any()]
            if converted:
                log.info("Coerced %d object column(s) to numeric: %s", len(converted), converted)
            df[obj_cols] = coerced

        shot_col = detect_shot_col(df)
        if shot_col != "shot_id":
            log.info("Renaming shot ID column '%s' -> 'shot_id'", shot_col)
            df = df.rename(columns={shot_col: "shot_id"})
        return df


# ---------------------------------------------------------------------------
# Built-in shot data backends.
# ---------------------------------------------------------------------------


class CsvShotDataBackend(ShotDataBackend):
    """Loads shot statistics from a CSV file."""

    def load(self, path: str) -> pd.DataFrame:
        log.info("Loading %s (CSV)...", path)
        return self._prepare(pd.read_csv(path))


class ParquetShotDataBackend(ShotDataBackend):
    """Loads shot statistics from a Parquet file."""

    def load(self, path: str) -> pd.DataFrame:
        log.info("Loading %s (Parquet)...", path)
        return self._prepare(pd.read_parquet(path))


# ---------------------------------------------------------------------------
# TraceBackend — abstract base for per-shot time-series loaders.
# ---------------------------------------------------------------------------


class TraceBackend(ABC):
    """Base class for backends that load per-shot time-series traces.

    :meth:`load` is called on demand (on scatter-plot click or during
    cluster/outlier computations). It should return a DataFrame with a
    ``time`` column and one column per configured signal, or ``None`` if
    the shot cannot be found.

    :meth:`is_available` is called once at startup to decide whether the
    time-trace panel is shown in the UI.
    """

    def __init__(self, config: BackendConfig) -> None:
        self.config = config

    @abstractmethod
    def load(self, shot_id: int) -> pd.DataFrame | None:
        """Return traces for *shot_id*, or ``None`` if not found."""

    @abstractmethod
    def is_available(self) -> bool:
        """Return ``True`` if this backend has data to serve."""


# ---------------------------------------------------------------------------
# Built-in trace backends.
# ---------------------------------------------------------------------------


class LocalParquetTraceBackend(TraceBackend):
    """Loads per-shot traces from local Parquet / CSV files.

    Files are expected at::

        <data_dir>/<any-subdir>/<shot_id>.parquet
        <data_dir>/<any-subdir>/<shot_id>.csv
    """

    def is_available(self) -> bool:
        d = self.config.data_dir
        return os.path.isdir(d) and bool(os.listdir(d))

    def find_shot_file(self, shot_id: int) -> str | None:
        """Return the path to the per-shot file, or ``None`` if not found."""
        data_dir = self.config.data_dir
        for subdir in sorted(os.listdir(data_dir)):
            for ext in (".parquet", ".csv"):
                path = os.path.join(data_dir, subdir, f"{int(shot_id)}{ext}")
                if os.path.exists(path):
                    return path
        return None

    def load(self, shot_id: int) -> pd.DataFrame | None:
        import duckdb

        path = self.find_shot_file(shot_id)
        if path is None:
            return None

        min_t, max_t = self.config.min_time, self.config.max_time
        ext = os.path.splitext(path)[1].lower()

        if ext == ".csv":
            result = pd.read_csv(path)
            return result[(result["time"] >= min_t) & (result["time"] <= max_t)].reset_index(drop=True)

        con = duckdb.connect()
        result = con.execute(f"SELECT * FROM '{path}' WHERE time >= {min_t} AND time <= {max_t}").df()
        con.close()
        return result


class _RemoteTraceBackend(TraceBackend):
    """Shared implementation for remote (UDA / SAL) backends."""

    def is_available(self) -> bool:
        return True  # live connection — assume available

    def _url(self, signal: str, shot_id: int) -> str:
        raise NotImplementedError

    def _engine(self) -> str:
        raise NotImplementedError

    def load(self, shot_id: int) -> pd.DataFrame | None:
        import xarray as xr

        cfg = self.config
        engine = self._engine()
        min_t, max_t = cfg.min_time, cfg.max_time

        if cfg.timebase_hz is not None:
            n = int(round((max_t - min_t) * cfg.timebase_hz))
            time_ref: np.ndarray | None = np.linspace(min_t, max_t, n)
        else:
            time_ref = None

        signal_data: dict[str, np.ndarray] = {}
        for signal in cfg.signals:
            try:
                ds = xr.open_dataset(self._url(signal, shot_id), engine=engine)
                if time_ref is None:
                    time_ref = ds.coords["time"].values.astype(float)
                signal_data[signal] = ds["data"].interp(time=time_ref).values
            except Exception as exc:
                log.error("[%s] Could not load '%s' for shot %d: %s", engine.upper(), signal, shot_id, exc)

        if time_ref is None:
            return None

        result = pd.DataFrame({"time": time_ref, **signal_data})
        return result[(result["time"] >= min_t) & (result["time"] <= max_t)].reset_index(drop=True)


class UdaTraceBackend(_RemoteTraceBackend):
    """Loads traces from a live UDA server via ``uda-xarray``."""

    def _engine(self) -> str:
        return "uda"

    def _url(self, signal: str, shot_id: int) -> str:
        return f"uda://{signal}:{shot_id}"


class SalTraceBackend(_RemoteTraceBackend):
    """Loads traces from a live SAL server via ``sal-xarray``."""

    def _engine(self) -> str:
        return "sal"

    def _url(self, signal: str, shot_id: int) -> str:
        return f"sal://pulse/{shot_id}/{signal}"


# ---------------------------------------------------------------------------
# Registry + factory functions.
# ---------------------------------------------------------------------------

_shot_data_registry: dict[str, type[ShotDataBackend]] = {}
_trace_registry: dict[str, type[TraceBackend]] = {}


def register_shot_data_backend(ext: str, cls: type[ShotDataBackend]) -> None:
    """Register *cls* as the shot-data backend for file extension *ext*.

    *ext* must include the leading dot, e.g. ``".csv"``.
    Existing registrations are silently overwritten, which allows plugins to
    replace built-in implementations.
    """
    _shot_data_registry[ext.lower()] = cls


def register_trace_backend(name: str, cls: type[TraceBackend]) -> None:
    """Register *cls* as the trace backend for config key *name*.

    *name* must match the value of ``backend:`` in config.yaml, e.g.
    ``"my_backend"``. Existing registrations are silently overwritten.
    """
    _trace_registry[name] = cls


def create_shot_data_backend(path: str, config: BackendConfig) -> ShotDataBackend:
    """Return an instantiated :class:`ShotDataBackend` for *path*.

    The backend is chosen by file extension. Raises :exc:`ValueError` if no
    backend is registered for the extension.
    """
    ext = os.path.splitext(path)[1].lower()
    cls = _shot_data_registry.get(ext)
    if cls is None:
        raise ValueError(
            f"No shot data backend registered for extension '{ext}'. Registered: {list(_shot_data_registry)}"
        )
    return cls(config)


def create_trace_backend(name: str, config: BackendConfig) -> TraceBackend:
    """Return an instantiated :class:`TraceBackend` for *name*.

    *name* is the value of ``backend:`` in config.yaml. Raises
    :exc:`ValueError` if no backend is registered under that name.
    """
    cls = _trace_registry.get(name)
    if cls is None:
        raise ValueError(f"No trace backend registered for name '{name}'. Registered: {list(_trace_registry)}")
    return cls(config)


# ---------------------------------------------------------------------------
# Built-in registrations.
# ---------------------------------------------------------------------------

register_shot_data_backend(".csv", CsvShotDataBackend)
register_shot_data_backend(".parquet", ParquetShotDataBackend)
register_shot_data_backend(".pq", ParquetShotDataBackend)
register_trace_backend("parquet", LocalParquetTraceBackend)
register_trace_backend("uda", UdaTraceBackend)
register_trace_backend("sal", SalTraceBackend)
