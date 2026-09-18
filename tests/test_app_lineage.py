"""Callback-level tests for the Lineage tab.

The Lineage callbacks are registered only when a reference column is
configured, and the shared ``app_module`` fixture deliberately has none. This
module therefore imports ``nice_shot/app.py`` a **second** time, under a
different module name, against a dataset that does have one.

Importing the same file under a different name works because ``app.py`` reads
``sys.argv`` at import time and Python caches modules by name: the alias gets
its own module object and its own globals. ``importlib.reload`` would not do --
it rebinds globals in the module object that ``test_app_callbacks.py`` and
``test_cli.py`` already hold, and those tests depend on its accumulated state
(a grown parquet, a warm dataset cache, a monkeypatched projection path).
"""

from __future__ import annotations

import contextlib
import importlib.util
import pathlib
import sys

import dash
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
import yaml
from dash._callback_context import context_value
from dash._utils import AttributeDict

import nice_shot

SUBJECT = 3013  # newest shot, tip of the 3007..3013 chain
ROOT = 3000  # a shot with no reference at all


@pytest.fixture(scope="module")
def ref_app_module(tmp_path_factory):
    """A second, independent import of ``nice_shot/app.py`` with a reference column.

    ``SHOW_REF_TOGGLE`` is therefore True here and the Lineage callbacks are
    registered, while the session-scoped ``app_module`` fixture is untouched.
    """
    tmp_path = tmp_path_factory.mktemp("app_lineage")
    n = 14
    shot_id = np.arange(3000, 3000 + n)
    frame = pd.DataFrame(
        {
            "shot_id": shot_id,
            "feature_1": np.linspace(0.0, 1.0, n),
            "feature_2": np.linspace(1.0, 0.0, n),
            "ip_max": np.linspace(600.0, 800.0, n),
            # A numeric column stored as object strings with the literal "nan"
            # for missing -- the shape real parquet shot tables use.
            "q95": ["nan"] + [f"{v:.4f}" for v in np.linspace(3.0, 5.0, n - 1)],
            "steady": np.full(n, 7.0),  # zero variance
            "scenario": ["H-mode"] * 7 + ["L-mode"] * 7,
            "objective": [f"try setting {i}" for i in range(n)],
            # A string reference column, exercising the pd.to_numeric path in
            # _build_reference_graph that real data hits.
            "ref_shot": [None] * 8 + [str(s) for s in range(3007, 3013)],
        }
    )
    shot_data_path = tmp_path / "shots.parquet"
    frame.to_parquet(shot_data_path, index=False)

    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({"projection_method": "pca", "reference_shot_col": "ref_shot"}))

    old_argv = sys.argv
    sys.argv = [
        "niceshot",
        str(shot_data_path),
        "--config",
        str(config_path),
        # Its own cache path: _umap_cache_hash keys on the config only, never on
        # the data, so sharing a path with app_module would collide on disk.
        "--umap-cache",
        str(tmp_path / "projection.npy"),
    ]
    try:
        app_path = pathlib.Path(nice_shot.__file__).parent / "app.py"
        spec = importlib.util.spec_from_file_location("nice_shot_app_ref", app_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules["nice_shot_app_ref"] = module
        spec.loader.exec_module(module)
    finally:
        sys.argv = old_argv
    return module


def _tab_values(node, found=None) -> list[str]:
    """Every ``dcc.Tab`` value in a Dash layout tree."""
    found = [] if found is None else found
    if type(node).__name__ == "Tab":
        value = getattr(node, "value", None)
        if isinstance(value, str):
            found.append(value)
    children = getattr(node, "children", None)
    if isinstance(children, (list, tuple)):
        for child in children:
            _tab_values(child, found)
    elif children is not None and not isinstance(children, str):
        _tab_values(children, found)
    return found


def _find_tab(node, value: str):
    """The ``dcc.Tab`` in a layout tree whose value is *value*."""
    if type(node).__name__ == "Tab" and getattr(node, "value", None) == value:
        return node
    children = getattr(node, "children", None)
    candidates = children if isinstance(children, (list, tuple)) else [children]
    for child in candidates:
        if child is None or isinstance(child, str):
            continue
        found = _find_tab(child, value)
        if found is not None:
            return found
    return None


def _ids(node, found=None) -> list[str]:
    """Every component id in a layout subtree."""
    found = [] if found is None else found
    component_id = getattr(node, "id", None)
    if isinstance(component_id, str):
        found.append(component_id)
    children = getattr(node, "children", None)
    candidates = children if isinstance(children, (list, tuple)) else [children]
    for child in candidates:
        if child is not None and not isinstance(child, str):
            _ids(child, found)
    return found


def _texts(node, found=None) -> list[str]:
    """Flatten the visible strings out of a Dash component tree."""
    found = [] if found is None else found
    if isinstance(node, str):
        found.append(node)
        return found
    children = getattr(node, "children", None)
    if isinstance(children, (list, tuple)):
        for child in children:
            _texts(child, found)
    elif children is not None:
        _texts(children, found)
    return found


def _default_columns(mod, subject=SUBJECT, scope="chain"):
    return mod._lin_default_columns(None, subject, scope, "zscore")


def _top_ranked(mod, metric="zscore", subject=SUBJECT, scope="chain"):
    """The head of the summary card's ranking — the Sparklines default."""
    return mod._lin_card_ranked_columns(None, subject, scope, metric)[: mod._LIN_SPARK_PANELS]


@contextlib.contextmanager
def _triggered_by(component_id: str | None):
    """Fake the callback context for a callback that branches on ``dash.ctx``."""
    triggered = [{"prop_id": f"{component_id}.n_clicks", "value": 1}] if component_id else []
    token = context_value.set(AttributeDict(triggered_inputs=triggered))
    try:
        yield
    finally:
        context_value.reset(token)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


class TestRegistration:
    def test_a_reference_column_enables_the_lineage_tab(self, ref_app_module):
        assert ref_app_module.SHOW_REF_TOGGLE is True
        assert "lineage" in _tab_values(ref_app_module.app.layout)
        assert hasattr(ref_app_module, "update_lineage_history")

    def test_every_sub_view_is_present(self, ref_app_module):
        values = _tab_values(ref_app_module.app.layout)
        assert {"lin-history", "lin-notes", "lin-tree", "lin-spark"} <= set(values)
        assert "lin-parcoords" not in values

    def test_every_sub_view_has_a_loading_indicator(self, ref_app_module):
        # A wide History table takes seconds to build, and without a spinner
        # the tab looks broken rather than busy.
        for value in ("lin-history", "lin-notes", "lin-tree", "lin-spark"):
            tab = _find_tab(ref_app_module.app.layout, value)
            assert any(type(child).__name__ == "Loading" for child in tab.children), value

    def test_the_sparkline_controls_stay_outside_the_overlay(self, ref_app_module):
        # A dimmed selector that cannot be clicked while the figure it drives
        # redraws is worse than no feedback at all.
        tab = _find_tab(ref_app_module.app.layout, "lin-spark")
        spinner = next(c for c in tab.children if type(c).__name__ == "Loading")
        assert "lin-spark-columns" not in _ids(spinner)
        assert "lin-spark-plot" in _ids(spinner)

    def test_the_lineage_tab_sits_before_configuration(self, ref_app_module):
        values = _tab_values(ref_app_module.app.layout)
        assert values.index("lineage") < values.index("config")

    def test_the_two_imports_do_not_share_state(self, ref_app_module, app_module):
        assert ref_app_module is not app_module
        assert ref_app_module.REFERENCE_SHOT_COL == "ref_shot"
        assert app_module.REFERENCE_SHOT_COL is None

    def test_a_string_reference_column_still_builds_a_graph(self, ref_app_module):
        # ref_shot is written as strings; _build_reference_graph coerces it.
        ds = ref_app_module.get_dataset(None)
        assert ds.ref_parent[SUBJECT] == 3012
        assert ds.ref_adjacency

    def test_numeric_columns_stored_as_strings_are_comparable(self, ref_app_module):
        # q95 is written as strings with the literal "nan" for missing. The flat
        # Parquet backend coerces object columns on load (backends.py:202), so
        # here it arrives already numeric; the long-format backend does not
        # (backends.py:283), which is why the tab classifies columns itself
        # rather than trusting dtype. Either way it must compare as a number.
        ds = ref_app_module.get_dataset(None)
        assert "q95" in ds.ref_numeric_cols

    def test_text_columns_are_offered_but_not_as_numbers(self, ref_app_module):
        ds = ref_app_module.get_dataset(None)
        assert "scenario" in ds.ref_compare_cols
        assert "scenario" not in ds.ref_numeric_cols

    def test_a_zero_variance_column_has_no_usable_spread(self, ref_app_module):
        ds = ref_app_module.get_dataset(None)
        assert pd.isna(ds.ref_stds["steady"])

    def test_the_reference_column_is_not_offered_for_comparison(self, ref_app_module):
        assert "ref_shot" not in ref_app_module.get_dataset(None).ref_compare_cols


# ---------------------------------------------------------------------------
# Subject resolution
# ---------------------------------------------------------------------------


class TestSubject:
    def test_prefers_the_selected_shot(self, ref_app_module):
        assert ref_app_module.resolve_lineage_subject(3010, SUBJECT, None, 0) == 3010

    def test_falls_back_to_the_latest_shot(self, ref_app_module):
        assert ref_app_module.resolve_lineage_subject(None, SUBJECT, None, 0) == SUBJECT

    def test_ignores_a_shot_that_is_not_in_the_table(self, ref_app_module):
        assert ref_app_module.resolve_lineage_subject(999999, SUBJECT, None, 0) == SUBJECT

    def test_returns_none_without_any_shot(self, ref_app_module):
        assert ref_app_module.resolve_lineage_subject(None, None, None, 0) is None

    def test_the_display_names_the_fallback_source(self, ref_app_module):
        text = ref_app_module.update_lineage_subject_display(SUBJECT, "chain", None, 0, None)
        assert "(latest)" in text and "7 shots in lineage" in text and "reference 3012" in text

    def test_the_display_names_the_selected_source(self, ref_app_module):
        text = ref_app_module.update_lineage_subject_display(3010, "chain", None, 0, 3010)
        assert "(selected)" in text

    def test_the_display_says_when_there_is_no_reference(self, ref_app_module):
        assert "no reference shot" in ref_app_module.update_lineage_subject_display(ROOT, "chain", None, 0, ROOT)


# ---------------------------------------------------------------------------
# Column selection
# ---------------------------------------------------------------------------


class TestColumnSelection:
    def test_options_always_include_the_selection(self, ref_app_module):
        # Dash clears a value that is absent from options, so a search that
        # dropped the selection would silently wipe the user's picks.
        options = ref_app_module.update_lineage_column_options("q9", None, ["ip_max"])
        assert "ip_max" in [o["value"] for o in options]
        assert "q95" in [o["value"] for o in options]

    def test_options_are_capped(self, ref_app_module, monkeypatch):
        monkeypatch.setattr(ref_app_module, "_LIN_OPTION_LIMIT", 2)
        assert len(ref_app_module.update_lineage_column_options(None, None, [])) == 2

    def test_options_drop_a_stale_selection(self, ref_app_module):
        options = ref_app_module.update_lineage_column_options(None, None, ["not_a_column"])
        assert "not_a_column" not in [o["value"] for o in options]

    def test_defaults_to_the_columns_that_changed(self, ref_app_module):
        columns = _default_columns(ref_app_module)
        assert columns
        assert "steady" not in columns  # zero variance, so it never changes

    def test_a_manual_selection_is_not_overwritten(self, ref_app_module):
        # Re-seeding on every click would discard hand-picked columns, which
        # makes comparing one variable across shots impossible.
        with _triggered_by("lin-subject-shot"):
            result = ref_app_module.seed_lineage_columns(0, 0, 0, SUBJECT, "chain", None, "zscore", ["ip_max"])
        assert isinstance(result, dash._callback.NoUpdate)

    def test_an_empty_selection_is_seeded_with_every_projection_feature(self, ref_app_module):
        features = ref_app_module._lin_projection_features(ref_app_module.get_dataset(None))
        with _triggered_by("lin-subject-shot"):
            result = ref_app_module.seed_lineage_columns(0, 0, 0, SUBJECT, "chain", None, "zscore", [])
        assert result == features
        # Every one of them, not a truncated head: a shortened default reads as
        # "these are the features". The projection coordinates and the reference
        # column are numeric too, but the tab never compares them.
        ds = ref_app_module.get_dataset(None)
        assert set(result) == set(ds.search_cols) & set(ds.ref_compare_cols)
        assert len(result) > ref_app_module._LIN_DEFAULT_N or len(result) == len(features)

    def test_the_default_keeps_a_feature_that_did_not_change(self, ref_app_module):
        # "steady" has zero variance, so a changed-only default drops it. The
        # projection uses it, so the lineage tracks it.
        with _triggered_by("lin-subject-shot"):
            result = ref_app_module.seed_lineage_columns(0, 0, 0, SUBJECT, "chain", None, "zscore", [])
        assert "steady" in result

    def test_the_top_changed_button_reseeds_over_a_manual_selection(self, ref_app_module):
        with _triggered_by("lin-cols-changed-btn"):
            result = ref_app_module.seed_lineage_columns(1, 0, 0, SUBJECT, "chain", None, "zscore", ["ip_max"])
        assert result == _default_columns(ref_app_module)

    def test_the_clear_button_empties_the_selection(self, ref_app_module):
        with _triggered_by("lin-cols-clear-btn"):
            result = ref_app_module.seed_lineage_columns(0, 0, 1, SUBJECT, "chain", None, "zscore", ["ip_max"])
        assert result == []

    def test_the_features_button_uses_the_projection_columns(self, ref_app_module):
        with _triggered_by("lin-cols-features-btn"):
            result = ref_app_module.seed_lineage_columns(0, 1, 0, SUBJECT, "chain", None, "zscore", ["ip_max"])
        candidates = set(ref_app_module.get_dataset(None).ref_compare_cols)
        assert result and set(result) <= candidates
        assert "ref_shot" not in result

    def test_projection_features_are_never_columns_the_tab_cannot_compare(self, ref_app_module):
        features = ref_app_module._lin_projection_features(ref_app_module.get_dataset(None))
        assert set(features) <= set(ref_app_module.get_dataset(None).ref_compare_cols)

    def test_the_count_reports_the_candidate_total(self, ref_app_module):
        candidates = ref_app_module.get_dataset(None).ref_compare_cols
        text = ref_app_module.update_lineage_column_count(["ip_max"], None)
        assert text == f"1 of {len(candidates)} variables"


# ---------------------------------------------------------------------------
# The ranked change card
# ---------------------------------------------------------------------------


class TestChangeCard:
    def test_ranks_the_biggest_change_first(self, ref_app_module):
        card = ref_app_module.update_lineage_cards(SUBJECT, "chain", "zscore", [], None, None, 0)
        text = " ".join(_texts(card))
        assert f"Shot {SUBJECT}" in text and "reference 3012" in text
        assert "variable(s) changed" in text

    def test_reports_the_tally_over_every_candidate_not_the_selection(self, ref_app_module):
        card = ref_app_module.update_lineage_cards(SUBJECT, "chain", "zscore", [], None, None, 0)
        total = len(ref_app_module.get_dataset(None).ref_compare_cols)
        assert f"of {total} variable(s) changed" in " ".join(_texts(card))

    def test_names_the_measure_it_ranks_by(self, ref_app_module):
        for metric, needle in [("zscore", "column spread"), ("percent", "percentage of the previous value")]:
            card = ref_app_module.update_lineage_cards(SUBJECT, "chain", metric, [], None, None, 0)
            assert needle in " ".join(_texts(card))

    def test_the_change_column_is_headed_with_its_unit(self, ref_app_module):
        zscored = " ".join(_texts(ref_app_module.update_lineage_cards(SUBJECT, "chain", "zscore", [], None, None, 0)))
        percentage = " ".join(
            _texts(ref_app_module.update_lineage_cards(SUBJECT, "chain", "percent", [], None, None, 0))
        )
        assert "change (σ)" in zscored
        assert "change (σ)" not in percentage

    def test_lists_every_variable_not_only_the_changed_ones(self, ref_app_module):
        # The card is the tab's whole summary, so "did anything else move" must
        # be answerable without widening the selection below it.
        card = ref_app_module.update_lineage_cards(SUBJECT, "chain", "zscore", [], None, None, 0)
        rendered = _texts(card)
        for column in ref_app_module.get_dataset(None).ref_compare_cols:
            assert column in rendered
        assert "unchanged" in rendered  # "steady" has zero variance

    def test_orders_the_rows_by_the_size_of_the_change(self, ref_app_module):
        from nice_shot.analysis import rank_lineage_changes

        view, _message = ref_app_module._lin_resolve(None, SUBJECT, "chain", "zscore", [], None)
        items = rank_lineage_changes(view.matrix, top_n=ref_app_module._LIN_CARD_MAX, changed_only=False)
        magnitudes = [i.magnitude for i in items if i.kind == "numeric" and np.isfinite(i.magnitude)]
        assert magnitudes == sorted(magnitudes, reverse=True)
        # Every compared variable is present, in one order or another.
        assert len(items) == len(view.matrix.columns)

    def test_the_percentage_ranking_shows_a_percentage_beside_the_bar(self, ref_app_module):
        # The change column holds the variable's own units, and those say
        # nothing about whether a change is large. A percentage does.
        percentage = " ".join(
            _texts(ref_app_module.update_lineage_cards(SUBJECT, "chain", "percent", [], None, None, 0))
        )
        zscored = " ".join(_texts(ref_app_module.update_lineage_cards(SUBJECT, "chain", "zscore", [], None, None, 0)))
        assert "% change" in percentage and "% change" not in zscored
        assert "+2.0%" in percentage  # ip_max rises by 15.4 from 784.6
        assert "%" not in zscored

    def test_the_ranking_measure_changes_the_numbers_shown(self, ref_app_module):
        zscored = " ".join(_texts(ref_app_module.update_lineage_cards(SUBJECT, "chain", "zscore", [], None, None, 0)))
        percentage = " ".join(
            _texts(ref_app_module.update_lineage_cards(SUBJECT, "chain", "percent", [], None, None, 0))
        )
        # ip_max moves by ~15.4 in its own units and by ~0.25 of a sigma.
        assert "+15.38" in percentage and "+15.38" not in zscored

    def test_the_percentage_ranking_orders_by_the_percentage(self, ref_app_module):
        from nice_shot.analysis import rank_lineage_changes

        view, _message = ref_app_module._lin_resolve(None, SUBJECT, "chain", "percent", [], None)
        items = rank_lineage_changes(view.matrix, top_n=None, changed_only=False)
        percentages = [abs(i.pct) for i in items if i.kind == "numeric" and i.pct is not None]
        assert len(percentages) > 1
        assert percentages == sorted(percentages, reverse=True)

    def test_the_offered_measures_are_the_two_comparable_ones(self, ref_app_module):
        assert [o["value"] for o in ref_app_module._LIN_CARD_METRIC_OPTIONS] == ["zscore", "percent"]
        assert [o["label"].strip() for o in ref_app_module._LIN_CARD_METRIC_OPTIONS] == ["z-scored", "percentage"]

    def test_says_so_when_the_subject_has_no_reference(self, ref_app_module):
        card = ref_app_module.update_lineage_cards(ROOT, "chain", "zscore", [], None, None, 0)
        assert "no reference shot" in " ".join(_texts(card))

    def test_says_so_without_a_subject(self, ref_app_module):
        card = ref_app_module.update_lineage_cards(None, "chain", "zscore", [], None, None, 0)
        assert "Select a shot" in " ".join(_texts(card))


# ---------------------------------------------------------------------------
# The history table
# ---------------------------------------------------------------------------


def _history(mod, subject=SUBJECT, scope="chain", metric="zscore", columns=None, respect=None, filters=None):
    return mod.update_lineage_history(
        subject,
        scope,
        metric,
        columns if columns is not None else _default_columns(mod, subject, scope),
        respect or [],
        filters,
        None,
        0,
    )


class TestHistoryTable:
    def test_returns_the_newest_row_first(self, ref_app_module):
        data, _columns, _styles, _tips, _msg = _history(ref_app_module)
        assert [row["shot_id"] for row in data] == [3013, 3012, 3011, 3010, 3009, 3008, 3007]
        assert data[0]["_rel"] == "subject"
        assert data[1]["_rel"] == "-1"

    def test_numeric_cells_are_numbers_so_the_table_can_format_them(self, ref_app_module):
        data, columns, _styles, _tips, _msg = _history(ref_app_module, columns=["ip_max"])
        spec = next(c for c in columns if c["id"] == "ip_max")
        assert spec["type"] == "numeric" and spec["format"] == {"specifier": ".4g"}
        assert isinstance(data[0]["ip_max"], float)

    def test_string_columns_stay_text(self, ref_app_module):
        data, columns, _styles, _tips, _msg = _history(ref_app_module, columns=["scenario"])
        assert next(c for c in columns if c["id"] == "scenario").get("type") is None
        assert data[0]["scenario"] == "L-mode"

    def test_colours_cells_not_rows(self, ref_app_module):
        _data, _columns, styles, _tips, _msg = _history(ref_app_module)
        cell_styles = [s for s in styles if "column_id" in s["if"] and isinstance(s["if"].get("row_index"), int)]
        assert cell_styles
        assert not any("filter_query" in s["if"] for s in styles)

    def test_the_broad_rules_come_before_the_cell_rules(self, ref_app_module):
        _data, _columns, styles, _tips, _msg = _history(ref_app_module)
        first_cell = next(
            i for i, s in enumerate(styles) if "column_id" in s["if"] and isinstance(s["if"].get("row_index"), int)
        )
        assert styles[0]["if"] == {"row_index": "odd"}
        assert first_cell > 0

    def test_the_subject_row_is_marked(self, ref_app_module):
        _data, _columns, styles, _tips, _msg = _history(ref_app_module)
        assert any("borderTop" in s for s in styles)

    def test_tooltips_carry_the_change_the_colour_stands_for(self, ref_app_module):
        _data, _columns, _styles, tips, _msg = _history(ref_app_module, columns=["ip_max"])
        assert "Δ" in tips[0]["ip_max"] and "z =" in tips[0]["ip_max"]
        assert tips[-1]["ip_max"] == "baseline"  # the oldest shot has nothing before it

    def test_is_empty_without_a_subject(self, ref_app_module):
        data, columns, styles, tips, msg = _history(ref_app_module, subject=None)
        assert (data, columns, styles, tips) == ([], [], [], [])
        assert "Select a shot" in " ".join(_texts(msg))

    def test_is_empty_without_columns(self, ref_app_module):
        data, _columns, _styles, _tips, msg = _history(ref_app_module, columns=[])
        assert data == []
        assert "Select at least one variable" in " ".join(_texts(msg))

    def test_says_so_for_a_lineage_of_one(self, ref_app_module):
        data, _columns, _styles, _tips, msg = _history(ref_app_module, subject=ROOT)
        assert data == []
        assert "no reference shot" in " ".join(_texts(msg))

    @pytest.mark.parametrize("scope,expected", [("chain", 4), ("component", 7), ("siblings", 2)])
    def test_each_scope_selects_its_own_shots(self, ref_app_module, scope, expected):
        data, _columns, _styles, _tips, _msg = _history(ref_app_module, subject=3010, scope=scope)
        assert len(data) == expected

    @pytest.mark.parametrize("metric", ["zscore", "percent", "absolute"])
    def test_every_metric_renders(self, ref_app_module, metric):
        data, _columns, styles, tips, _msg = _history(ref_app_module, metric=metric)
        assert len(data) == 7 and styles and tips

    def test_the_absolute_metric_admits_its_scale_is_local(self, ref_app_module):
        _data, _columns, _styles, _tips, msg = _history(ref_app_module, metric="absolute")
        assert "scaled to this lineage only" in " ".join(_texts(msg))


class TestHistoryFilters:
    def test_ignores_active_filters_by_default(self, ref_app_module):
        without = _history(ref_app_module, filters=None)[0]
        with_filter = _history(ref_app_module, filters=[3013, 3012])[0]
        assert [r["shot_id"] for r in without] == [r["shot_id"] for r in with_filter]

    def test_marks_filtered_shots_when_opted_in(self, ref_app_module):
        kept = [3013, 3012, 3010, 3009, 3008, 3007]  # 3011 filtered out
        data, _columns, styles, _tips, msg = _history(ref_app_module, respect=["respect"], filters=kept)
        # The shot is greyed, never dropped: removing it would make the delta
        # either side of it a comparison between two unlinked shots.
        assert len(data) == 7
        assert any(s.get("fontStyle") == "italic" for s in styles)
        assert "hidden by the active filters" in " ".join(_texts(msg))

    def test_the_zscore_denominator_is_unaffected_by_filters(self, ref_app_module):
        plain = _history(ref_app_module, columns=["ip_max"])[3]
        filtered = _history(ref_app_module, columns=["ip_max"], respect=["respect"], filters=[3013, 3012])[3]
        assert plain[0]["ip_max"] == filtered[0]["ip_max"]


# ---------------------------------------------------------------------------
# The other sub-views
# ---------------------------------------------------------------------------


class TestSubViews:
    def test_the_tree_draws_nodes_and_edges(self, ref_app_module):
        fig = ref_app_module.update_lineage_tree(SUBJECT, "chain", [], None, None, 0)
        assert isinstance(fig, go.Figure)
        assert any(t.name == "_lin_subject" for t in fig.data)
        assert any(t.name == "_lin_edge" for t in fig.data)

    def test_the_tree_reports_a_lone_shot(self, ref_app_module):
        fig = ref_app_module.update_lineage_tree(ROOT, "chain", [], None, None, 0)
        assert "no reference shot" in fig.layout.annotations[0].text

    def test_clicking_a_tree_node_selects_that_shot(self, ref_app_module):
        assert ref_app_module.select_shot_from_lineage_tree({"points": [{"hovertext": "3010"}]}, None) == 3010

    def test_clicking_nothing_changes_nothing(self, ref_app_module):
        result = ref_app_module.select_shot_from_lineage_tree(None, None)
        assert isinstance(result, dash._callback.NoUpdate)

    def test_notes_show_one_block_per_lineage_shot(self, ref_app_module):
        panel = ref_app_module.update_lineage_notes(SUBJECT, "chain", None, 0)
        text = " ".join(_texts(panel))
        assert "try setting 13" in text and "try setting 12" in text
        assert "scenario" in text

    def test_sparklines_are_skipped_unless_their_sub_view_is_open(self, ref_app_module):
        figure, message = ref_app_module.update_lineage_sparklines(
            "lin-history", SUBJECT, "chain", "zscore", ["ip_max"], None, 0
        )
        assert isinstance(figure, dash._callback.NoUpdate)
        assert isinstance(message, dash._callback.NoUpdate)

    def test_sparklines_draw_one_panel_per_selected_variable(self, ref_app_module):
        fig, message = ref_app_module.update_lineage_sparklines(
            "lin-spark", SUBJECT, "chain", "zscore", ["ip_max", "q95"], None, 0
        )
        assert len(fig.data) == 2
        assert "2 of 2 selected variable(s) drawn" in message

    def test_sparklines_keep_a_selected_variable_that_did_not_change(self, ref_app_module):
        # "steady" has zero variance. Ranking it out of the view would make the
        # panels disagree with the selector above them.
        fig, _message = ref_app_module.update_lineage_sparklines(
            "lin-spark", SUBJECT, "chain", "zscore", ["ip_max", "steady"], None, 0
        )
        assert {t.name for t in fig.data} == {"ip_max", "steady"}

    def test_sparklines_are_ordered_the_way_the_card_ranks_them(self, ref_app_module):
        columns = ["steady", "ip_max", "q95"]
        fig, _message = ref_app_module.update_lineage_sparklines(
            "lin-spark", SUBJECT, "chain", "zscore", columns, None, 0
        )
        ranked = ref_app_module._lin_card_ranked_columns(None, SUBJECT, "chain", "zscore")
        assert [t.name for t in fig.data] == [c for c in ranked if c in set(columns)]

    def test_sparklines_stop_at_the_safety_limit_and_say_so(self, ref_app_module, monkeypatch):
        monkeypatch.setattr(ref_app_module, "_LIN_SPARK_HARD_MAX", 1)
        fig, message = ref_app_module.update_lineage_sparklines(
            "lin-spark", SUBJECT, "chain", "zscore", ["ip_max", "q95", "feature_1"], None, 0
        )
        assert len(fig.data) == 1
        assert "at most 1 panels are drawn" in message

    def test_sparklines_account_for_a_text_variable(self, ref_app_module):
        fig, message = ref_app_module.update_lineage_sparklines(
            "lin-spark", SUBJECT, "chain", "zscore", ["ip_max", "scenario"], None, 0
        )
        assert len(fig.data) == 1
        assert "1 text variable(s) have no sparkline" in message

    def test_the_sparkline_selection_starts_at_the_cards_top_variables(self, ref_app_module):
        ranked = ref_app_module._lin_card_ranked_columns(None, SUBJECT, "chain", "zscore")
        with _triggered_by("lin-subject-shot"):
            seeded, remembered = ref_app_module.seed_lineage_spark_columns(
                0, SUBJECT, "chain", "zscore", None, [], None
            )
        assert seeded == ranked[: ref_app_module._LIN_SPARK_PANELS]
        assert remembered == seeded  # so a later subject can refresh it
        # Numeric only: a text column ranks in the card but cannot be a line.
        assert "scenario" not in seeded and "objective" not in seeded

    def test_the_sparkline_default_follows_the_ranking_measure(self, ref_app_module):
        with _triggered_by("lin-card-metric"):
            zscored, _ = ref_app_module.seed_lineage_spark_columns(0, SUBJECT, "chain", "zscore", None, [], None)
            percentage, _ = ref_app_module.seed_lineage_spark_columns(0, SUBJECT, "chain", "percent", None, [], None)
        assert zscored and percentage
        assert percentage == _top_ranked(ref_app_module, "percent")

    def test_a_new_subject_refreshes_a_default_sparkline_selection(self, ref_app_module):
        # The default is the card's top variables and the card describes one
        # shot, so a default left alone would name another shot's variables.
        stale = ref_app_module._lin_card_ranked_columns(None, 3010, "chain", "zscore")[
            : ref_app_module._LIN_SPARK_PANELS
        ]
        with _triggered_by("lin-subject-shot"):
            refreshed, _ = ref_app_module.seed_lineage_spark_columns(0, SUBJECT, "chain", "zscore", None, stale, stale)
        assert refreshed == _top_ranked(ref_app_module, "zscore")

    def test_a_hand_picked_sparkline_selection_survives_a_shot_change(self, ref_app_module):
        with _triggered_by("lin-subject-shot"):
            value, remembered = ref_app_module.seed_lineage_spark_columns(
                0, SUBJECT, "chain", "zscore", None, ["q95"], ["ip_max"]
            )
        assert isinstance(value, dash._callback.NoUpdate)
        assert isinstance(remembered, dash._callback.NoUpdate)

    def test_the_top_button_reseeds_over_a_hand_picked_selection(self, ref_app_module):
        with _triggered_by("lin-spark-top-btn"):
            result, _ = ref_app_module.seed_lineage_spark_columns(
                1, SUBJECT, "chain", "zscore", None, ["q95"], ["ip_max"]
            )
        assert result == _top_ranked(ref_app_module, "zscore")

    def test_the_sparkline_selector_offers_the_whole_column_pool(self, ref_app_module):
        options = ref_app_module.update_lineage_spark_options("q9", None, ["ip_max"])
        values = [o["value"] for o in options]
        assert "q95" in values and "ip_max" in values


# ---------------------------------------------------------------------------
# Long-format empty state
# ---------------------------------------------------------------------------


def test_every_lineage_view_shows_the_select_variable_message(ref_app_module, monkeypatch):
    """Long-format mode has no dataset until a variable is picked.

    Every Lineage view reaches data through ``get_dataset``, so one monkeypatch
    exercises that branch across all of them.
    """
    monkeypatch.setattr(ref_app_module, "get_dataset", lambda _variable: None)
    message = ref_app_module.SELECT_VARIABLE_MSG

    assert ref_app_module.resolve_lineage_subject(SUBJECT, SUBJECT, None, 0) is None
    assert ref_app_module.update_lineage_subject_display(SUBJECT, "chain", None, 0, None) == message
    assert message in " ".join(
        _texts(ref_app_module.update_lineage_cards(SUBJECT, "chain", "zscore", [], None, None, 0))
    )
    assert message in " ".join(_texts(ref_app_module.update_lineage_notes(SUBJECT, "chain", None, 0)))
    assert message in " ".join(_texts(_history(ref_app_module, columns=["ip_max"])[4]))
    assert ref_app_module.update_lineage_tree(SUBJECT, "chain", [], None, None, 0).layout.annotations[0].text == message
    spark, _spark_msg = ref_app_module.update_lineage_sparklines(
        "lin-spark", SUBJECT, "chain", "zscore", ["ip_max"], None, 0
    )
    assert spark.layout.annotations[0].text == message
    assert ref_app_module._lin_card_ranked_columns(None, SUBJECT, "chain", "zscore") == []
    assert isinstance(
        ref_app_module.select_shot_from_lineage_tree({"points": [{"hovertext": "3010"}]}, None),
        dash._callback.NoUpdate,
    )


# ---------------------------------------------------------------------------
# Long-format mode.
#
# A third import, because SHOW_REF_TOGGLE takes a different branch here
# (app.py reads the file schema instead of a loaded dataset, since long-format
# mode loads no rows until a variable is picked) and that branch decides
# whether the tab exists at all. It is also the mode where the tab's own column
# classification is load-bearing: LongParquetShotDataBackend prepares rows with
# coerce_objects=False, so numeric columns keep dtype=object.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def long_app_module(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("app_lineage_long")
    frames = []
    for variable, gain in (("p1_i", 1.0), ("p4_i", 3.0)):
        n = 12
        frames.append(
            pd.DataFrame(
                {
                    "shot_id": np.arange(4000, 4000 + n),
                    "variable_name": variable,
                    "reference__number": [None] * 6 + [float(s) for s in range(4005, 4011)],
                    "peak": np.linspace(1.0, 2.0, n) * gain,
                    # Object strings with the literal "nan": the long-format
                    # backend does not coerce these, so only the tab's own
                    # classification makes them comparable as numbers.
                    "rms": ["nan"] + [f"{v:.3f}" for v in np.linspace(0.5, 1.5, n - 1)],
                    "flat": np.full(n, 2.0),
                    "scenario": ["A"] * 6 + ["B"] * 6,
                    "preshot": [f"{variable} attempt {i}" for i in range(n)],
                }
            )
        )
    shot_data_path = tmp_path / "long.parquet"
    pd.concat(frames, ignore_index=True).to_parquet(shot_data_path, index=False)

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "projection_method": "pca",
                "variable_column": "variable_name",
                "reference_shot_col": "reference__number",
            }
        )
    )

    old_argv = sys.argv
    sys.argv = [
        "niceshot",
        str(shot_data_path),
        "--config",
        str(config_path),
        "--umap-cache",
        str(tmp_path / "projection.npy"),
    ]
    try:
        app_path = pathlib.Path(nice_shot.__file__).parent / "app.py"
        spec = importlib.util.spec_from_file_location("nice_shot_app_long", app_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules["nice_shot_app_long"] = module
        spec.loader.exec_module(module)
    finally:
        sys.argv = old_argv
    return module


class TestLongFormat:
    def test_the_tab_is_enabled_from_the_file_schema(self, long_app_module):
        # No rows are loaded at import, so the gate can only look at the columns.
        assert long_app_module.VARIABLE_MODE is True
        assert long_app_module.SHOW_REF_TOGGLE is True
        assert "lineage" in _tab_values(long_app_module.app.layout)

    def test_the_views_wait_for_a_variable(self, long_app_module):
        message = long_app_module.SELECT_VARIABLE_MSG
        assert long_app_module.resolve_lineage_subject(None, 4011, None, 0) is None
        assert long_app_module.update_lineage_subject_display(4011, "chain", None, 0, None) == message
        _data, _cols, _styles, _tips, msg = long_app_module.update_lineage_history(
            4011, "chain", "zscore", ["peak"], [], None, None, 0
        )
        assert message in " ".join(_texts(msg))

    def test_object_dtype_columns_are_still_compared_as_numbers(self, long_app_module):
        ds = long_app_module.get_dataset("p1_i")
        assert ds.df["rms"].dtype == object  # the backend leaves it alone here
        assert "rms" in ds.ref_numeric_cols
        assert "scenario" not in ds.ref_numeric_cols

    def test_the_lineage_is_built_per_variable(self, long_app_module):
        ds = long_app_module.get_dataset("p1_i")
        assert ds.ref_parent[4011] == 4010
        subject = long_app_module.resolve_lineage_subject(None, 4011, "p1_i", 0)
        one = long_app_module.update_lineage_history(subject, "chain", "zscore", ["peak"], [], None, "p1_i", 0)[0]
        four = long_app_module.update_lineage_history(subject, "chain", "zscore", ["peak"], [], None, "p4_i", 0)[0]
        assert [r["shot_id"] for r in one] == [4011, 4010, 4009, 4008, 4007, 4006, 4005]
        assert four[0]["peak"] == pytest.approx(one[0]["peak"] * 3.0)

    def test_a_string_stored_column_gets_real_deltas(self, long_app_module):
        subject = long_app_module.resolve_lineage_subject(None, 4011, "p1_i", 0)
        data, _cols, _styles, tips, _msg = long_app_module.update_lineage_history(
            subject, "chain", "zscore", ["rms"], [], None, "p1_i", 0
        )
        assert data[0]["rms"] == pytest.approx(1.5)
        assert "Δ = +0.1" in tips[0]["rms"]

    def test_a_zero_variance_column_is_not_offered_by_default(self, long_app_module):
        subject = long_app_module.resolve_lineage_subject(None, 4011, "p1_i", 0)
        assert "flat" not in long_app_module._lin_default_columns("p1_i", subject, "chain", "zscore")
