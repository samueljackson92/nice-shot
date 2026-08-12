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

    def _prepare(self, df: pd.DataFrame, coerce_objects: bool = True) -> pd.DataFrame:
        """Coerce object columns to numeric and normalise the shot ID column.

        Only columns that successfully parse as numbers are converted; columns
        with genuine string values (e.g. machine names) are kept as-is so they
        remain available as discrete colour hues in the UI.

        Set *coerce_objects* to ``False`` for already-typed sources (e.g. Parquet)
        where the dtypes must be derivable from the file schema alone, without
        inspecting any rows.
        """
        obj_cols = df.select_dtypes(include="object").columns if coerce_objects else []
        if len(obj_cols):
            coerced = df[obj_cols].apply(pd.to_numeric, errors="coerce")
            converted = [c for c in obj_cols if coerced[c].notna().any()]
            if converted:
                log.info("Coerced %d object column(s) to numeric: %s", len(converted), converted)
                df[converted] = coerced[converted]

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
# VariableShotDataBackend — abstract base for long-format shot statistics.
# ---------------------------------------------------------------------------


class VariableShotDataBackend(ShotDataBackend):
    """Base class for *long-format* shot statistics.

    A long-format file holds one row per ``(shot, variable)`` pair: the same set
    of statistic columns is repeated for every variable, distinguished by a
    variable column named in ``config.options["variable_column"]``.

    Implementations must be able to list the available variables and read the
    rows for a single variable **without** reading the whole file, so the UI can
    defer loading until the user picks one.
    """

    @property
    def variable_column(self) -> str:
        return self.config.options["variable_column"]

    @abstractmethod
    def variables(self, path: str) -> list[str]:
        """Return the sorted, unique variable names available in *path*."""

    @abstractmethod
    def schema(self, path: str) -> pd.DataFrame:
        """Return an empty DataFrame with the columns and dtypes of a loaded variable.

        Must not read row data — the UI builds its widgets from this at startup.
        """

    @abstractmethod
    def load_variable(self, path: str, variable: str) -> pd.DataFrame:
        """Load only the rows for *variable* and return a normalised DataFrame."""

    def load(self, path: str) -> pd.DataFrame:
        raise NotImplementedError(
            "Long-format sources are read one variable at a time — use load_variable() instead of load()."
        )


class LongParquetShotDataBackend(VariableShotDataBackend):
    """Reads a long-format Parquet file one variable at a time.

    Uses Parquet predicate pushdown so only the row groups holding the requested
    variable are read, and derives the column schema from file metadata alone.
    """

    def _normalise(self, df: pd.DataFrame) -> pd.DataFrame:
        """Flatten the index and drop redundant columns.

        Index levels are promoted to columns so the shot ID becomes addressable.
        The variable column itself is dropped: it is constant within a single
        variable's slice and so carries no information. A label may appear both
        as an index level and as a data column (pandas writes MultiIndex levels
        as columns), in which case the duplicate is removed first.
        """
        df = df.loc[:, ~df.columns.duplicated()]
        index_names = [n for n in (df.index.names or []) if n is not None]
        if index_names:
            df = df.drop(columns=[c for c in index_names if c in df.columns], errors="ignore")
            df = df.reset_index()
        df = df.drop(columns=[self.variable_column], errors="ignore")
        return self._prepare(df, coerce_objects=False)

    def variables(self, path: str) -> list[str]:
        col = self.variable_column
        values = pd.read_parquet(path, columns=[col])[col]
        if isinstance(values, pd.DataFrame):  # column also present as an index level
            values = values.iloc[:, 0]
        result = sorted(str(v) for v in values.dropna().unique())
        log.info("Found %d variables in %s (column '%s')", len(result), path, col)
        return result

    def schema(self, path: str) -> pd.DataFrame:
        import pyarrow.parquet as pq

        empty = pq.ParquetFile(path).schema_arrow.empty_table().to_pandas()
        return self._normalise(empty)

    def load_variable(self, path: str, variable: str) -> pd.DataFrame:
        log.info("Loading variable '%s' from %s (Parquet)...", variable, path)
        df = pd.read_parquet(path, filters=[(self.variable_column, "==", variable)])
        result = self._normalise(df)
        log.info("Loaded %d rows for variable '%s'", len(result), variable)
        return result


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


class _SuppressAsyncCleanupRaceFilter(logging.Filter):
    """Silences a known benign zarr-v3/s3fs/aiobotocore async cleanup race.

    zarr v3's Fsspec async store and s3fs each run their own background
    event loop/thread. Closing an idle S3 session can occasionally race the
    closing of the *other* loop's selector, raising a stray ValueError deep
    inside aiohttp/aiobotocore transport teardown. asyncio logs this via its
    default exception handler (logger 'asyncio') rather than raising it into
    calling code, so it never affects data already returned — just noise.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        exc = record.exc_info[1] if record.exc_info else None
        return not (isinstance(exc, ValueError) and "closed kqueue" in str(exc))


logging.getLogger("asyncio").addFilter(_SuppressAsyncCleanupRaceFilter())


class FairMastTraceBackend(TraceBackend):
    """Loads per-shot time-series traces from FAIR MAST level2 data via xarray.

    Supports local directories and any fsspec-addressable remote URL (e.g.
    ``s3://...``), and both Zarr and netCDF-4 (via h5netcdf) formats.

    Shot files are expected at ``<data_dir>/<shot_id>.<ext>`` where ``ext``
    is ``zarr`` or ``nc`` depending on ``format`` below. Signals in
    ``signals`` are ``"<group>/<variable>"`` strings (e.g.
    ``"thomson_scattering/t_e"``); a bare name with no ``/`` is read from
    the store's root group.

    Only scalar (time-only) variables are supported — profile/multi-
    dimensional variables (e.g. Thomson scattering channel profiles) are
    skipped with a logged error, matching the tolerant per-signal failure
    behaviour of the rest of this backend.

    Optional options:
      ``format``          — ``"zarr"`` (default) or ``"netcdf"``.
      ``storage_options`` — dict passed through to fsspec/the xarray engine
                             for remote stores, e.g. for FAIR MAST's public
                             S3 endpoint:
                             ``{"anon": true, "client_kwargs": {"endpoint_url": "https://s3.echo.stfc.ac.uk"}}``.
                             Ignored for local paths.
    """

    _EXT = {"zarr": "zarr", "netcdf": "nc"}
    _ENGINE = {"zarr": "zarr", "netcdf": "h5netcdf"}

    def _format(self) -> str:
        fmt = self.config.options.get("format", "zarr")
        if fmt not in self._EXT:
            raise ValueError(f"fairmast: unknown format '{fmt}', expected 'zarr' or 'netcdf'")
        return fmt

    def _storage_options(self) -> dict[str, Any]:
        return self.config.options.get("storage_options", {})

    def _shot_url(self, shot_id: int) -> str:
        base = self.config.data_dir.rstrip("/")
        return f"{base}/{int(shot_id)}.{self._EXT[self._format()]}"

    @staticmethod
    def _split_signal(signal: str) -> tuple[str | None, str]:
        """Split 'group/variable' into (group, variable); bare names -> (None, name)."""
        if "/" in signal:
            group, _, var = signal.rpartition("/")
            return (group or None), var
        return None, signal

    def is_available(self) -> bool:
        try:
            import fsspec

            storage_options = self._storage_options()
            fs, path = fsspec.core.url_to_fs(self.config.data_dir, **storage_options)
            return fs.isdir(path) or fs.exists(path)
        except ImportError:
            log.warning('FAIR MAST backend requires optional dependencies: pip install "nice-shot[fairmast]"')
            return False
        except Exception as exc:
            log.warning("FAIR MAST backend unavailable: %s", exc)
            return False

    def _open_group(self, url: str, group: str | None):
        import xarray as xr

        engine = self._ENGINE[self._format()]
        storage_options = self._storage_options()
        return xr.open_dataset(url, group=group, engine=engine, storage_options=storage_options or None)

    def load(self, shot_id: int) -> pd.DataFrame | None:
        cfg = self.config
        url = self._shot_url(shot_id)
        min_t, max_t = cfg.min_time, cfg.max_time

        if cfg.timebase_hz is not None:
            n = int(round((max_t - min_t) * cfg.timebase_hz))
            time_ref: np.ndarray | None = np.linspace(min_t, max_t, n)
        else:
            time_ref = None

        opened_groups: dict[str | None, Any] = {}
        signal_data: dict[str, np.ndarray] = {}

        for signal in cfg.signals:
            group, var = self._split_signal(signal)
            try:
                if group not in opened_groups:
                    opened_groups[group] = self._open_group(url, group)
                ds = opened_groups[group]

                if time_ref is None:
                    time_ref = ds.coords["time"].values.astype(float)

                da = ds[var].interp(time=time_ref)
                if da.ndim != 1:
                    raise ValueError(
                        f"'{signal}' has shape {da.shape} (dims {da.dims}) — only scalar "
                        f"(time-only) variables are supported"
                    )
                signal_data[signal] = da.values
            except Exception as exc:
                log.error("[fairmast] Could not load '%s' for shot %d: %s", signal, shot_id, exc)

        for ds in opened_groups.values():
            try:
                ds.close()
            except Exception:
                pass

        if time_ref is None:
            return None

        result = pd.DataFrame({"time": time_ref, **signal_data})
        return result[(result["time"] >= min_t) & (result["time"] <= max_t)].reset_index(drop=True)


class PostgresShotDataBackend(ShotDataBackend):
    """Loads shot statistics from a PostgreSQL table via DuckDB's postgres extension.

    Required option:
      ``dsn`` — libpq connection string, e.g. ``postgresql://user:pass@host/db``.

    Optional options:
      ``shot_table`` — table name (default: stem of *path*, e.g. ``shots`` from ``shots.pg``).
      ``schema``     — PostgreSQL schema (default: ``public``).
    """

    def load(self, path: str) -> pd.DataFrame:
        import duckdb

        dsn = self.config.options["dsn"]
        table = self.config.options.get("shot_table", os.path.splitext(os.path.basename(path))[0])
        schema = self.config.options.get("schema", "public")

        log.info("Loading shot data from PostgreSQL '%s.%s'...", schema, table)
        con = duckdb.connect()
        try:
            con.execute("INSTALL postgres; LOAD postgres;")
            con.execute(f"ATTACH '{dsn}' AS pg (TYPE POSTGRES, READ_ONLY)")
            df = con.execute(f"SELECT * FROM pg.{schema}.{table}").df()
        finally:
            con.close()
        return self._prepare(df)


class PostgresTraceBackend(TraceBackend):
    """Loads per-shot time-series traces from a PostgreSQL table via DuckDB.

    Required option:
      ``dsn`` — libpq connection string, e.g. ``postgresql://user:pass@host/db``.

    Optional options:
      ``trace_table`` — table name (default: ``traces``).
      ``schema``      — PostgreSQL schema (default: ``public``).
      ``shot_col``    — column holding the shot ID (default: ``shot_id``).
      ``time_col``    — column holding the time axis (default: ``time``).
    """

    def is_available(self) -> bool:
        import duckdb

        try:
            con = duckdb.connect()
            con.execute("INSTALL postgres; LOAD postgres;")
            con.execute(f"ATTACH '{self.config.options['dsn']}' AS pg (TYPE POSTGRES, READ_ONLY)")
            con.execute("SELECT 1")
            con.close()
            return True
        except Exception as exc:
            log.warning("PostgreSQL backend unavailable: %s", exc)
            return False

    def load(self, shot_id: int) -> pd.DataFrame | None:
        import duckdb

        dsn = self.config.options["dsn"]
        schema = self.config.options.get("schema", "public")
        table = self.config.options.get("trace_table", "traces")
        shot_col = self.config.options.get("shot_col", "shot_id")
        time_col = self.config.options.get("time_col", "time")
        min_t, max_t = self.config.min_time, self.config.max_time

        try:
            con = duckdb.connect()
            con.execute("INSTALL postgres; LOAD postgres;")
            con.execute(f"ATTACH '{dsn}' AS pg (TYPE POSTGRES, READ_ONLY)")
            query = (
                f"SELECT * FROM pg.{schema}.{table} "
                f"WHERE {shot_col} = {shot_id} "
                f"AND {time_col} >= {min_t} AND {time_col} <= {max_t}"
            )
            df = con.execute(query).df()
            con.close()
        except Exception as exc:
            log.error("Failed to load traces for shot %d: %s", shot_id, exc)
            return None

        if df.empty:
            return None

        if time_col != "time":
            df = df.rename(columns={time_col: "time"})
        return df


# ---------------------------------------------------------------------------
# Registry + factory functions.
# ---------------------------------------------------------------------------

_shot_data_registry: dict[str, type[ShotDataBackend]] = {}
_variable_shot_data_registry: dict[str, type[VariableShotDataBackend]] = {}
_trace_registry: dict[str, type[TraceBackend]] = {}


def register_shot_data_backend(ext: str, cls: type[ShotDataBackend]) -> None:
    """Register *cls* as the shot-data backend for file extension *ext*.

    *ext* must include the leading dot, e.g. ``".csv"``.
    Existing registrations are silently overwritten, which allows plugins to
    replace built-in implementations.
    """
    _shot_data_registry[ext.lower()] = cls


def register_variable_shot_data_backend(ext: str, cls: type[VariableShotDataBackend]) -> None:
    """Register *cls* as the long-format shot-data backend for extension *ext*.

    Used when ``variable_column`` is set in config. *ext* must include the
    leading dot, e.g. ``".parquet"``.
    """
    _variable_shot_data_registry[ext.lower()] = cls


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


def create_variable_shot_data_backend(path: str, config: BackendConfig) -> VariableShotDataBackend:
    """Return an instantiated :class:`VariableShotDataBackend` for *path*.

    The backend is chosen by file extension. Raises :exc:`ValueError` if no
    long-format backend is registered for the extension.
    """
    ext = os.path.splitext(path)[1].lower()
    cls = _variable_shot_data_registry.get(ext)
    if cls is None:
        raise ValueError(
            f"variable_column is set, but no long-format shot data backend is registered "
            f"for extension '{ext}'. Registered: {list(_variable_shot_data_registry)}"
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
register_shot_data_backend(".pg", PostgresShotDataBackend)
register_variable_shot_data_backend(".parquet", LongParquetShotDataBackend)
register_variable_shot_data_backend(".pq", LongParquetShotDataBackend)
register_trace_backend("parquet", LocalParquetTraceBackend)
register_trace_backend("uda", UdaTraceBackend)
register_trace_backend("sal", SalTraceBackend)
register_trace_backend("postgres", PostgresTraceBackend)
register_trace_backend("fairmast", FairMastTraceBackend)
