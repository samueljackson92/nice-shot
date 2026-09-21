"""Column plumbing for a projection with more than two components.

The first two components are always ``umap_x``/``umap_y`` so the plots can read
a fixed pair of axis names. Components three and up are extra numeric columns,
which must reach the axis pickers but must stay out of the data table, the
similarity index and the lineage comparison.

The refresh path is the reason this file exists. ``_project_new_rows`` used to
build exactly two columns and ``refresh_dataset`` used to re-save exactly two,
so a wider projection would give every refreshed shot NaN coordinates and
silently narrow the cache on disk while its hash still claimed the full width.
Nothing would surface, because the plots only ever read the first two columns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from conftest import import_app

from nice_shot.analysis import reference_compare_columns


@pytest.fixture(scope="module")
def wide_app(tmp_path_factory):
    """An import of ``app.py`` whose projection has four components."""
    tmp_path = tmp_path_factory.mktemp("app_wide")
    rng = np.random.default_rng(0)
    n = 30
    frame = pd.DataFrame(
        {
            "shot_id": np.arange(5000, 5000 + n),
            **{f"feature_{i}": rng.normal(size=n) for i in range(6)},
        }
    )
    shot_data_path = tmp_path / "shots.parquet"
    frame.to_parquet(shot_data_path, index=False)

    config_path = tmp_path / "config.yaml"
    config_path.write_text("projection_method: pca\nprojection_options:\n  n_components: 4\n")

    return import_app(
        "nice_shot_app_wide",
        [
            str(shot_data_path),
            "--config",
            str(config_path),
            "--umap-cache",
            str(tmp_path / "projection.npy"),
        ],
    )


def _coords(columns) -> list[str]:
    return sorted(c for c in columns if str(c).startswith("umap"))


def test_extra_components_become_columns(wide_app):
    ds = wide_app.get_dataset(None)
    assert _coords(ds.df.columns) == ["umap_3", "umap_4", "umap_x", "umap_y"]


def test_extra_components_are_selectable_as_plot_axes(wide_app):
    """They are ordinary numeric columns, which is the point: a user can plot
    component 3 against component 4."""
    assert _coords(wide_app._pair_axis_cols) == ["umap_3", "umap_4", "umap_x", "umap_y"]
    colour_values = [o["value"] for o in wide_app._color_col_options]
    assert _coords(colour_values) == ["umap_3", "umap_4", "umap_x", "umap_y"]


def test_coordinates_stay_out_of_the_data_table(wide_app):
    assert _coords(wide_app._table_cols) == []


def test_coordinates_stay_out_of_the_similarity_index(wide_app):
    """Feeding the embedding back in as similarity features would weight it
    against the real measurements, and more components would weight it more."""
    assert _coords(wide_app.get_dataset(None).search_cols) == []


def test_coordinates_stay_out_of_the_lineage_comparison(wide_app):
    """Otherwise the tab reports "component 4 changed by 0.3 sigma"."""
    ds = wide_app.get_dataset(None)
    assert _coords(reference_compare_columns(ds.df)) == []


def test_cache_on_disk_keeps_the_full_width(wide_app):
    cached = np.load(wide_app._umap_cache_path(None))
    assert cached.shape[1] == 4


def test_model_reports_its_own_width(wide_app):
    assert wide_app._model_n_components(wide_app.get_dataset(None).model) == 4


def test_refreshed_shots_get_every_coordinate(wide_app):
    """Runs last: it appends to the backing file and replaces the cached dataset."""
    before = wide_app.get_dataset(None)
    n_before = len(before.df)

    existing = pd.read_parquet(wide_app.SHOT_DATA_PATH)
    rng = np.random.default_rng(1)
    added = pd.DataFrame(
        {
            "shot_id": np.arange(6000, 6003),
            **{f"feature_{i}": rng.normal(size=3) for i in range(6)},
        }
    )
    pd.concat([existing, added], ignore_index=True).to_parquet(wide_app.SHOT_DATA_PATH, index=False)

    latest = wide_app.refresh_dataset(None)
    assert latest == 6002

    after = wide_app.get_dataset(None)
    assert len(after.df) == n_before + 3
    coords = _coords(after.df.columns)
    assert coords == ["umap_3", "umap_4", "umap_x", "umap_y"]

    # The new shots must have a real value in every component, not just the two
    # the plots happen to read.
    new_rows = after.df[after.df["shot_id"] >= 6000][coords]
    assert len(new_rows) == 3
    assert not new_rows.isna().any().any()

    # And the cache must still be as wide as the model, not narrowed to two.
    assert np.load(wide_app._umap_cache_path(None)).shape == (n_before + 3, 4)
