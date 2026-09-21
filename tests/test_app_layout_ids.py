"""Every id a callback refers to must exist in the layout.

Dash raises "A nonexistent object was used in an Output" the moment a callback
with a missing Output fires, and most callbacks here fire on ``selected-shot``,
which is always present. The check therefore cannot be left to the browser: a
tab that is omitted rather than disabled breaks its callbacks only once a user
clicks a point.

These tests take the place of that front-end check, and they are what make it
safe to keep a feature's widgets in the tree and disable them instead.
"""

from __future__ import annotations

import re

from conftest import component_ids, find_tab, layout_of, tab_values


def _dep_ids(spec) -> set[str]:
    """Plain string component ids named by one callback's inputs and state."""
    found: set[str] = set()
    for dep in [*spec.get("inputs", []), *spec.get("state", [])]:
        component = dep["id"] if isinstance(dep, dict) else dep.component_id
        if isinstance(component, str) and not component.startswith("{"):
            found.add(component)
    return found


def _output_ids(output) -> set[str]:
    """Plain string component ids named by one callback's output spec.

    Dash stores this as an ``Output``, a list of them, or a packed string: a
    single output is ``"store-id.data"`` and several are joined, as
    ``"..first.data...second.figure.."``. A pattern-matching id serialises as
    JSON and is matched by shape, so it is skipped.
    """
    if isinstance(output, (list, tuple)):
        found: set[str] = set()
        for item in output:
            found |= _output_ids(item)
        return found

    component = getattr(output, "component_id", None)
    if component is not None:
        return {component} if isinstance(component, str) else set()

    parts = output[2:-2].split("...") if output.startswith("..") else [output]
    found = set()
    for part in parts:
        if not part or part.startswith("{"):
            continue
        name = part.rsplit(".", 1)[0]
        if name and not name.startswith("{"):
            found.add(name)
    return found


def _callback_id_strings(module) -> set[str]:
    """Every plain string component id that any registered callback refers to."""
    found: set[str] = set()
    for spec in module.app.callback_map.values():
        found |= _dep_ids(spec)
        found |= _output_ids(spec["output"])
    return found


def _assert_ids_cover_callbacks(module) -> None:
    present = set(component_ids(layout_of(module)))
    referenced = _callback_id_strings(module)
    missing = sorted(referenced - present)
    assert not missing, f"callbacks refer to ids that the layout does not contain: {missing}"


def test_layout_contains_every_callback_id(app_module):
    """The default flat-mode layout covers every registered callback."""
    _assert_ids_cover_callbacks(app_module)


def test_layout_contains_every_callback_id_with_a_reference_column(app_variant):
    """With a reference column the Lineage widgets and callbacks both exist."""
    module = app_variant(
        "ids_ref",
        {"projection_method": "pca", "reference_shot_col": "ref_shot"},
        reference=True,
    )
    _assert_ids_cover_callbacks(module)
    assert "lineage" in tab_values(layout_of(module))


def test_layout_contains_every_callback_id_without_traces(app_variant):
    """A missing data directory disables the trace panes but breaks no callback."""
    module = app_variant(
        "ids_no_traces",
        {"projection_method": "pca"},
        extra_argv=["--data-dir", "/nonexistent/niceshot/traces"],
    )
    assert module.SHOW_TRACES is False
    _assert_ids_cover_callbacks(module)


def test_time_traces_tab_is_disabled_rather_than_removed(app_variant):
    """The established pattern: keep the tab, disable it.

    This is the precedent the Lineage and SHAP tabs are meant to follow, so it
    is pinned here rather than left implicit.
    """
    module = app_variant(
        "ids_no_traces",
        {"projection_method": "pca"},
        extra_argv=["--data-dir", "/nonexistent/niceshot/traces"],
    )
    tab = find_tab(layout_of(module), "traces")
    assert tab is not None, "the Time Traces tab must stay in the tree"
    assert tab.disabled is True


def test_no_duplicate_component_ids(app_module):
    """A repeated id makes Dash address the wrong component."""
    ids = component_ids(layout_of(app_module))
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    assert not duplicates, f"duplicated component ids: {duplicates}"


def test_id_inventory_snapshot(app_module):
    """Guard against silently dropping a widget while restructuring the layout.

    Update the expected set deliberately when adding or removing a component.
    The projection coordinate columns are excluded because their count depends
    on ``projection_options.n_components``.
    """
    ids = set(component_ids(layout_of(app_module)))
    # Spot-check the ids that other features depend on, rather than freezing
    # the whole list, which would make every layout edit a two-file change.
    required = {
        "tabs",
        "umap-plot",
        "umap-color-col",
        "shot-table",
        "selected-shot",
        "selected-variable",
        "dataset-version",
        "active-filters",
        "refresh-interval",
        "cfg-signals",
        "cfg-time-window",
        "cfg-signal-select",
        "cfg-min-time",
        "cfg-max-time",
        "cfg-apply-btn",
        "cfg-reset-btn",
        "cfg-status",
    }
    assert required <= ids, f"missing expected ids: {sorted(required - ids)}"


def test_store_ids_are_not_nested_inside_a_tab(app_module):
    """Stores must sit at the app root.

    ``dcc.Tabs`` unmounts the unselected tab, and an unmounted Store forgets
    its value while a callback whose Output is not rendered never dispatches.
    """
    layout = layout_of(app_module)
    tab_store_ids: list[str] = []
    for value in tab_values(layout):
        tab = find_tab(layout, value)
        for node in component_ids(tab):
            if node.endswith(("-signals", "-time-window")) or node in {"selected-shot", "active-filters"}:
                tab_store_ids.append(node)
    assert not tab_store_ids, f"these stores are inside a tab: {sorted(set(tab_store_ids))}"


def test_callback_map_is_not_empty(app_module):
    """A sanity check: the id comparison above is vacuous if nothing registered."""
    assert len(app_module.app.callback_map) > 30
    assert re.search(r"umap-plot", " ".join(app_module.app.callback_map))
