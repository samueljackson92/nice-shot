# Custom Backends

NiceShot! has a registry of backends. You can add a new data source without a change to the app code. There are two extension points:

- **`TraceBackend`** — loads the time traces of one shot. Use it for a new instrument, a new file format or a remote API.
- **`ShotDataBackend`** — loads the shot statistics table. Use it for a new file format, for example HDF5.

---

## Write a trace backend

```python
# my_package/my_backends.py
import pandas as pd
from nice_shot.backends import BackendConfig, TraceBackend, register_trace_backend

class MdsTraceBackend(TraceBackend):
    def __init__(self, config: BackendConfig) -> None:
        super().__init__(config)
        self._server = config.options.get("server", "localhost")

    def load(self, shot_id: int) -> pd.DataFrame | None:
        # fetch data for shot_id and return a DataFrame with a
        # 'time' column and one column per signal, or None
        ...

    def is_available(self) -> bool:
        return True  # or check connectivity

register_trace_backend("mds", MdsTraceBackend)
```

Then, in the config file:

```yaml
backend: mds

plugins:
  - my_package.my_backends

backend_options:
  server: "mds.mylab.ac.uk"
```

The app gives `backend_options` to the backend as `config.options`. You can also set an option with `--backend-option KEY=VALUE`, or in the [Configuration tab](configuration.md#configuration-tab).

---

## Write a shot data backend

```python
# my_package/my_backends.py
import pandas as pd
from nice_shot.backends import BackendConfig, ShotDataBackend, register_shot_data_backend

class HDF5ShotDataBackend(ShotDataBackend):
    def load(self, path: str) -> pd.DataFrame:
        import h5py
        # load from HDF5 and return a normalised DataFrame
        # (must have a 'shot_id' column — call self._prepare(df) to auto-detect)
        ...

register_shot_data_backend(".h5", HDF5ShotDataBackend)
register_shot_data_backend(".hdf5", HDF5ShotDataBackend)
```

The app selects the shot data backend from the file extension of `SHOT_DATA`.

---

## Plugin load order

The app imports the modules in `plugins` in their list order, before it makes the backends. Thus all the `register_*` calls take effect in time.

For security, you can set `plugins` only in the config file or with `--plugins`. The [Configuration tab](configuration.md#what-the-tab-does-not-change) cannot change it.
