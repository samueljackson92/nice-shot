"""Setting the reference shot column while the app runs.

Dash registers callbacks once, at import, and cannot add one afterwards. The
Lineage tab's callbacks therefore have to exist whether or not a reference
column is configured, and the tab and the reference-graph toggle have to stay
in the layout so those callbacks always have somewhere to write.

This file pins both halves: the feature is off but intact at startup, and it
comes on when the column is applied, with no restart.
"""

from __future__ import annotations

import pytest
from conftest import find_tab, layout_of


@pytest.fixture(scope="module")
def unset_ref_app(app_variant):
    """Started with no reference column, against data that has one.

    That is the situation a user is in before they set the column in the
    Configuration tab.
    """
    return app_variant("runtime_ref", {"projection_method": "pca"}, reference=True)


def test_the_column_exists_in_the_data_but_is_not_configured(unset_ref_app):
    assert unset_ref_app.SHOW_REF_TOGGLE is False
    assert unset_ref_app.REFERENCE_SHOT_COL is None
    assert "ref_shot" in unset_ref_app.get_dataset(None).df.columns


def test_the_feature_is_off_at_startup(unset_ref_app):
    disabled, style = unset_ref_app.update_reference_feature_visibility(None)
    assert disabled is True
    assert style.get("display") == "none"


def test_the_lineage_tab_is_present_but_disabled_at_startup(unset_ref_app):
    tab = find_tab(layout_of(unset_ref_app), "lineage")
    assert tab is not None
    assert tab.disabled is True


def test_the_disabled_lineage_tab_is_hidden_not_greyed_out(unset_ref_app):
    """Disabled always means "no reference column" for this tab, and then the
    tab bar must not show it at all. dcc.Tab uses disabled_style in place of
    style while disabled, so the hiding rides on the same flag."""
    tab = find_tab(layout_of(unset_ref_app), "lineage")
    assert tab.disabled_style.get("display") == "none"


def test_lineage_explains_what_is_missing_rather_than_failing(unset_ref_app):
    """Every lineage callback goes through _lin_resolve, so one guard there
    gives all of them the same clear empty state."""
    view, message = unset_ref_app._lin_resolve(None, 4000, "chain", "zscore", [], None)
    assert view is None
    assert "reference shot column" in message


def _applied_key(module):
    """The dataset key the Configuration tab produces for ref_shot."""
    return module._dataset_key_from_stores(None, None, {"value": "ref_shot"}).to_store()


def test_applying_the_column_builds_the_reference_graph(unset_ref_app):
    ds = unset_ref_app.get_dataset(_applied_key(unset_ref_app))
    assert ds.ref_adjacency, "applying the column must build the graph"


def test_applying_the_column_turns_the_feature_on(unset_ref_app):
    """The whole point: no restart."""
    disabled, style = unset_ref_app.update_reference_feature_visibility(_applied_key(unset_ref_app))
    assert disabled is False
    assert style.get("display") != "none"


def test_lineage_resolves_once_the_column_is_applied(unset_ref_app):
    key = _applied_key(unset_ref_app)
    ds = unset_ref_app.get_dataset(key)
    subject = max(ds.ref_adjacency)
    view, message = unset_ref_app._lin_resolve(key, subject, "chain", "zscore", [], None)
    assert view is not None, f"expected a lineage, got: {message}"


def test_clearing_the_column_turns_the_feature_off_again(unset_ref_app):
    cleared = unset_ref_app._dataset_key_from_stores(None, None, {"value": None}).to_store()
    disabled, _style = unset_ref_app.update_reference_feature_visibility(cleared)
    assert disabled is True


def test_changing_the_column_does_not_refit_the_projection(unset_ref_app):
    """The reference graph is part of the Dataset, so it belongs in the cache
    key -- but it moves no point, so it must not invalidate the embedding."""
    module = unset_ref_app
    plain = module._coerce_key(None)
    with_ref = module._coerce_key(_applied_key(module))
    assert with_ref != plain
    assert module._umap_cache_hash(with_ref) == module._umap_cache_hash(plain)
    assert module._umap_cache_path(with_ref) == module._umap_cache_path(plain)
