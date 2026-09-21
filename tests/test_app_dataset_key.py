"""DatasetKey: the cache key a Dataset is built from.

The key round-trips through a ``dcc.Store``, which is JSON. JSON has no tuple,
so ``from_store`` has to put the tuples back. If it does not, the key becomes
unhashable, every cache lookup silently misses, and the projection is refitted
on every callback -- with no error to show why the app got slow. That is what
most of this file is guarding.
"""

from __future__ import annotations

import json
from dataclasses import replace

import pytest


@pytest.fixture
def key(app_module):
    """The key described by the config file."""
    return app_module.default_dataset_key(None)


def _through_a_store(app_module, key):
    """Exactly what a dcc.Store does to a value."""
    return app_module.DatasetKey.from_store(json.loads(json.dumps(key.to_store())))


def test_round_trip_through_a_store_is_equal(app_module, key):
    assert _through_a_store(app_module, key) == key


def test_round_trip_through_a_store_keeps_the_hash(app_module, key):
    """Equality is not enough: the key indexes a dict, so the hash must survive."""
    assert hash(_through_a_store(app_module, key)) == hash(key)


def test_round_trip_restores_the_tuples(app_module):
    """A list would make the key unhashable and every lookup a miss."""
    original = replace(
        app_module.default_dataset_key("v1"),
        umap_features=("ip_max", "ne_max"),
        umap_exclude_features=("shot_id",),
    )
    restored = _through_a_store(app_module, original)
    assert isinstance(restored.umap_features, tuple)
    assert isinstance(restored.umap_exclude_features, tuple)
    assert restored == original
    assert hash(restored) == hash(original)


def test_a_restored_key_works_as_a_dict_key(app_module, key):
    assert {_through_a_store(app_module, key): "hit"}.get(key) == "hit"


def test_key_is_hashable_even_with_features_set(app_module):
    wide = replace(app_module.default_dataset_key(None), umap_features=tuple(f"e{i:03d}" for i in range(192)))
    assert {wide: 1}[wide] == 1


# ---------------------------------------------------------------------------
# What belongs in the projection hash, and what does not
# ---------------------------------------------------------------------------


def test_reference_column_changes_the_key_but_not_the_projection(app_module, key):
    """The reference graph is part of the Dataset, so it belongs in the key.
    It moves no point, so refitting the projection for it would be waste."""
    other = replace(key, reference_shot_col="some_other_column")
    assert other != key
    assert app_module._umap_cache_hash(other) == app_module._umap_cache_hash(key)


def _other_method(key) -> str:
    """A projection method that is not the one *key* already uses."""
    return "umap" if key.projection_method == "pca" else "pca"


@pytest.mark.parametrize(
    "change",
    [
        "method",
        {"umap_features": ("feature_1",)},
        {"umap_exclude_features": ("feature_2",)},
        {"variable": "v2"},
    ],
)
def test_every_projection_field_changes_the_hash(app_module, key, change):
    if change == "method":
        change = {"projection_method": _other_method(key)}
    assert app_module._umap_cache_hash(replace(key, **change)) != app_module._umap_cache_hash(key)


@pytest.mark.parametrize(
    "attribute",
    ["n_components", "random_state", "n_neighbors", "min_dist", "metric"],
)
def test_every_projection_option_changes_the_hash(app_module, key, attribute):
    """A missed option here means a changed setting silently reuses the old
    embedding, and new shots get transformed onto a projection fitted for
    other settings."""
    changed = key.projection_options().model_copy(
        update={attribute: {"metric": "manhattan"}.get(attribute, 3 if attribute != "min_dist" else 0.4)}
    )
    other = replace(key, projection_options_json=json.dumps(changed.model_dump(), sort_keys=True))
    assert app_module._umap_cache_hash(other) != app_module._umap_cache_hash(key)


def test_cache_paths_differ_for_different_settings(app_module, key):
    """Two settings must not fight over one file on disk."""
    other = replace(key, projection_method=_other_method(key))
    assert app_module._umap_cache_path(other) != app_module._umap_cache_path(key)
    assert app_module._umap_model_path(other) != app_module._umap_model_path(key)


def test_cache_path_separates_variables(app_module):
    a = app_module._umap_cache_path(app_module.default_dataset_key("v1"))
    b = app_module._umap_cache_path(app_module.default_dataset_key("v2"))
    assert a != b
    assert "v1" in a and "v2" in b


# ---------------------------------------------------------------------------
# Coercion: the old call shapes must keep working
# ---------------------------------------------------------------------------


def test_none_means_the_config_file(app_module, key):
    assert app_module._coerce_key(None) == key


def test_a_bare_string_is_read_as_a_variable_name(app_module):
    assert app_module._coerce_key("v1").variable == "v1"


def test_a_store_dict_is_accepted(app_module, key):
    assert app_module._coerce_key(key.to_store()) == key


def test_a_key_passes_through(app_module, key):
    assert app_module._coerce_key(key) is key


def test_get_dataset_still_accepts_none(app_module):
    """Flat mode's original call shape."""
    assert app_module.get_dataset(None) is not None


# ---------------------------------------------------------------------------
# Building the key from the Configuration tab's stores
# ---------------------------------------------------------------------------


def test_empty_stores_fall_back_to_the_config_file(app_module, key):
    """A page that never opens the Configuration tab must behave as before."""
    assert app_module._dataset_key_from_stores(None, None, None) == key


def test_clearing_the_reference_column_actually_clears_it(app_module):
    """A dict holding null is "the user cleared this", which must not be read as
    "nothing applied" -- otherwise the config value comes straight back."""
    built = app_module._dataset_key_from_stores(None, None, {"value": None})
    assert built.reference_shot_col is None


def test_setting_the_reference_column_overrides_the_config(app_module):
    built = app_module._dataset_key_from_stores(None, None, {"value": "ref_shot"})
    assert built.reference_shot_col == "ref_shot"


def test_applied_projection_settings_reach_the_key(app_module):
    built = app_module._dataset_key_from_stores(
        None,
        {
            "projection_method": "pca",
            "umap_features": ["feature_1", "feature_2"],
            "umap_exclude_features": ["ip_max"],
            "projection_options": {
                "n_components": 3,
                "random_state": 1,
                "n_neighbors": 5,
                "min_dist": 0.2,
                "metric": "cosine",
            },
        },
        None,
    )
    assert built.projection_method == "pca"
    assert built.umap_features == ("feature_1", "feature_2")
    assert built.umap_exclude_features == ("ip_max",)
    options = built.projection_options()
    assert options.n_components == 3
    assert options.metric == "cosine"


def test_an_empty_feature_list_means_all_numeric_columns(app_module):
    """umap_features is `list[str] | None`, and the dropdown cannot express
    None -- an empty selection has to mean "use every numeric column"."""
    built = app_module._dataset_key_from_stores(None, {"umap_features": []}, None)
    assert built.umap_features is None


def test_the_selected_variable_reaches_the_key(app_module):
    assert app_module._dataset_key_from_stores("v7", None, None).variable == "v7"


def test_compute_dataset_key_returns_a_store_value(app_module, key):
    """The callback writes a dict, because that is what a dcc.Store holds."""
    value = app_module.compute_dataset_key(None, None, None)
    assert isinstance(value, dict)
    assert app_module.DatasetKey.from_store(value) == key
