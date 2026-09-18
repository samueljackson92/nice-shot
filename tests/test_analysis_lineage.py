"""Tests for the reference-lineage change analysis in nice_shot/analysis.py.

These are pure-function tests: nothing here imports ``nice_shot.app``, so none
of its import-time CLI parsing, config loading or dataset building runs. Most of
the Lineage tab's behaviour is covered here rather than through a Dash callback.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nice_shot.analysis import (
    _build_reference_graph,
    _classify_reference_columns,
    _coerce_reference_numeric,
    column_stds,
    get_reference_lineage,
    lineage_change_matrix,
    rank_lineage_changes,
    reference_compare_columns,
    select_changed_columns,
)

CHAIN = {10: 9, 9: 8, 8: 7}
COLUMNS = ["ip", "const", "sparse", "ip_str", "scenario", "comment"]


@pytest.fixture
def parent(lineage_df) -> dict[int, int]:
    return _build_reference_graph(lineage_df, "ref_shot")[1]


@pytest.fixture
def adjacency(lineage_df) -> dict[int, list[int]]:
    return _build_reference_graph(lineage_df, "ref_shot")[0]


def _matrix(df, lineage, columns=None, metric="zscore", **kwargs):
    return lineage_change_matrix(df, lineage, columns or COLUMNS, metric=metric, **kwargs)


class TestCoerceReferenceNumeric:
    def test_nan_strings_become_missing_not_text(self):
        out = _coerce_reference_numeric(pd.Series(["nan", "None", "", "743.2998", "-"]))
        assert out.isna().tolist() == [True, True, True, False, True]
        assert out.iloc[3] == pytest.approx(743.2998)

    def test_already_numeric_passes_through(self):
        out = _coerce_reference_numeric(pd.Series([1.0, np.nan, 3.5]))
        assert out.tolist()[0] == 1.0 and out.tolist()[2] == 3.5

    def test_genuine_text_becomes_all_missing(self):
        assert _coerce_reference_numeric(pd.Series(["H-mode", "L-mode"])).isna().all()


class TestClassifyReferenceColumns:
    def test_numeric_stored_as_object_strings_is_numeric(self, lineage_df):
        numeric, _ = _classify_reference_columns(lineage_df, COLUMNS)
        assert "ip_str" in numeric

    def test_nan_strings_do_not_defeat_the_coercion_threshold(self):
        # 2 of 10 values are the literal string "nan"; dropna() would keep them
        # and the parse rate would fall to 0.8, below the threshold.
        df = pd.DataFrame({"x": ["nan", "nan"] + [f"{v}" for v in range(8)]})
        numeric, categorical = _classify_reference_columns(df, ["x"])
        assert numeric == ["x"] and categorical == []

    def test_genuine_text_is_categorical(self, lineage_df):
        _, categorical = _classify_reference_columns(lineage_df, COLUMNS)
        assert set(categorical) == {"scenario", "comment"}

    def test_datetime_is_categorical(self):
        df = pd.DataFrame({"when": pd.to_datetime(["2026-01-01", "2026-01-02"])})
        assert _classify_reference_columns(df, ["when"]) == ([], ["when"])

    def test_bool_is_numeric(self):
        df = pd.DataFrame({"flag": [True, False, True]})
        assert _classify_reference_columns(df, ["flag"]) == (["flag"], [])

    def test_all_null_column_is_categorical(self):
        df = pd.DataFrame({"empty": [None, None]})
        assert _classify_reference_columns(df, ["empty"]) == ([], ["empty"])


class TestColumnStds:
    def test_zero_variance_column_becomes_nan(self, lineage_df):
        assert pd.isna(column_stds(lineage_df, ["const"])["const"])

    def test_a_single_real_value_gives_no_spread(self, lineage_df):
        # "sparse" holds one inf and one real value; once the inf is dropped a
        # single observation has no standard deviation.
        assert pd.isna(column_stds(lineage_df, ["sparse"])["sparse"])

    def test_inf_does_not_poison_the_std(self):
        df = pd.DataFrame({"shot_id": [1, 2, 3, 4], "x": [1.0, 2.0, 3.0, np.inf]})
        spread = column_stds(df, ["x"])["x"]
        assert np.isfinite(spread)
        assert spread == pytest.approx(pd.Series([1.0, 2.0, 3.0]).std())

    def test_object_string_column_is_coerced(self, lineage_df):
        assert np.isfinite(column_stds(lineage_df, ["ip_str"])["ip_str"])

    def test_missing_columns_give_an_empty_series(self, lineage_df):
        assert column_stds(lineage_df, ["nope"]).empty


class TestGetReferenceLineage:
    def test_chain_walks_parents_newest_first(self, parent):
        assert get_reference_lineage(10, parent)[0] == [10, 9, 8, 7]

    def test_chain_stops_at_a_shot_with_no_parent(self, parent):
        assert get_reference_lineage(7, parent)[0] == [7]

    def test_orphan_returns_only_itself(self, parent):
        assert get_reference_lineage(1, parent)[0] == [1]

    def test_self_reference_and_dangling_reference_are_not_edges(self, parent):
        assert get_reference_lineage(2, parent)[0] == [2]
        assert get_reference_lineage(3, parent)[0] == [3]

    def test_chain_terminates_on_a_reference_cycle(self, caplog):
        # _build_reference_graph rejects only self-references, so A->B->A is
        # representable in real data and a naive walk would never return.
        ids, _ = get_reference_lineage(5, {5: 6, 6: 5})
        assert ids == [5, 6]
        assert "cycle" in caplog.text

    def test_chain_respects_max_shots(self, parent):
        assert get_reference_lineage(10, parent, max_shots=2)[0] == [10, 9]

    def test_chain_preserves_graph_order_for_an_inverted_parent(self):
        # The one real dataset with an inverted pair must still compare the two
        # shots that are actually linked, not sort them apart.
        assert get_reference_lineage(5, {5: 9})[0] == [5, 9]

    def test_component_matches_get_reference_graph(self, parent, adjacency):
        ids, _ = get_reference_lineage(10, parent, adjacency, scope="component")
        assert ids == [10, 9, 8, 7]

    def test_component_includes_siblings_of_the_subject(self, parent, adjacency):
        ids, _ = get_reference_lineage(5, parent, adjacency, scope="component")
        assert ids == [6, 5, 4]

    def test_component_with_no_edges_returns_the_subject(self, parent):
        assert get_reference_lineage(1, parent, {}, scope="component")[0] == [1]

    def test_component_cap_retains_the_subject(self):
        # Clicking a shot must never produce a view that omits it.
        adjacency = {i: [1] for i in range(2, 60)} | {1: list(range(2, 60))}
        ids, _ = get_reference_lineage(3, {}, adjacency, scope="component", max_shots=5)
        assert len(ids) == 5 and 3 in ids

    def test_siblings_share_the_parent_and_put_it_last(self, parent):
        assert get_reference_lineage(5, parent, scope="siblings")[0] == [6, 5, 4]

    def test_siblings_can_exclude_the_parent(self, parent):
        assert get_reference_lineage(5, parent, scope="siblings", include_parent=False)[0] == [6, 5]

    def test_siblings_without_a_parent_returns_the_subject(self, parent):
        assert get_reference_lineage(7, parent, scope="siblings")[0] == [7]

    def test_unknown_scope_falls_back_to_chain(self, parent, caplog):
        assert get_reference_lineage(10, parent, scope="bogus")[0] == [10, 9, 8, 7]
        assert "unknown scope" in caplog.text

    @pytest.mark.parametrize("scope", ["chain", "component", "siblings"])
    def test_empty_graph_returns_the_subject_for_every_scope(self, scope):
        assert get_reference_lineage(42, {}, {}, scope=scope)[0] == [42]

    def test_restrict_to_keeps_the_chain_and_reports_exclusions(self, parent):
        ids, excluded = get_reference_lineage(10, parent, restrict_to={10, 9, 7})
        assert ids == [10, 9, 8, 7]  # the chain is intact, so deltas stay real
        assert excluded == frozenset({8})

    def test_the_subject_is_never_excluded(self, parent):
        _, excluded = get_reference_lineage(10, parent, restrict_to={7})
        assert 10 not in excluded


class TestLineageChangeMatrix:
    def test_empty_lineage_returns_an_empty_frame_with_the_schema(self, lineage_df):
        m = _matrix(lineage_df, [])
        assert m.values.empty and list(m.columns) == COLUMNS and m.metric == "zscore"

    def test_no_columns_returns_an_empty_frame(self, lineage_df):
        m = lineage_change_matrix(lineage_df, [10, 9], [])
        assert m.values.empty and m.columns == []

    def test_single_shot_lineage_is_all_baseline(self, lineage_df):
        m = _matrix(lineage_df, [10])
        assert (m.note.iloc[0] == "baseline").all()
        assert not m.changed.to_numpy().any()
        assert m.delta.isna().to_numpy().all()

    def test_delta_is_against_the_next_older_shot(self, lineage_df):
        # The lineage is newest-first, so the previous shot is the NEXT row.
        m = _matrix(lineage_df, [10, 9, 8, 7])
        assert m.delta.at[10, "ip"] == pytest.approx(10.0)  # 100 - 90
        assert m.delta.at[9, "ip"] == pytest.approx(10.0)  # 90 - 80

    def test_oldest_row_is_the_baseline_with_no_delta(self, lineage_df):
        m = _matrix(lineage_df, [10, 9, 8, 7])
        assert pd.isna(m.delta.at[7, "ip"])
        assert not m.changed.at[7, "ip"]
        assert m.note.at[7, "ip"] == "baseline"

    def test_row_order_matches_the_lineage(self, lineage_df):
        assert _matrix(lineage_df, [8, 10, 9]).shot_ids == [8, 10, 9]
        assert list(_matrix(lineage_df, [8, 10, 9]).values.index) == [8, 10, 9]

    def test_column_order_is_preserved(self, lineage_df):
        cols = ["scenario", "ip", "const"]
        assert list(lineage_change_matrix(lineage_df, [10, 9], cols).values.columns) == cols

    def test_zscore_uses_the_whole_dataset_std(self, lineage_df):
        spread = column_stds(lineage_df, ["ip"])["ip"]
        m = _matrix(lineage_df, [10, 9])
        assert m.metric_value.at[10, "ip"] == pytest.approx(10.0 / spread)

    def test_zero_variance_column_is_nan_not_inf(self, lineage_df):
        m = _matrix(lineage_df, [10, 9])
        assert pd.isna(m.metric_value.at[10, "const"])
        assert m.note.at[10, "const"] == "zero_variance"
        assert not m.changed.at[10, "const"]

    def test_percent_uses_the_absolute_baseline_so_signs_survive(self):
        df = pd.DataFrame({"shot_id": [1, 2], "x": [-1.0, -2.0]})
        m = lineage_change_matrix(df, [1, 2], ["x"], metric="percent")
        # -2 -> -1 is a rise, so the percentage must be positive.
        assert m.metric_value.at[1, "x"] == pytest.approx(50.0)

    def test_percent_zero_to_zero_is_unchanged(self):
        df = pd.DataFrame({"shot_id": [1, 2], "x": [0.0, 0.0]})
        m = lineage_change_matrix(df, [1, 2], ["x"], metric="percent")
        assert m.metric_value.at[1, "x"] == 0.0
        assert not m.changed.at[1, "x"]

    def test_percent_zero_baseline_with_a_real_change_is_undefined(self):
        df = pd.DataFrame({"shot_id": [1, 2], "x": [5.0, 0.0]})
        m = lineage_change_matrix(df, [1, 2], ["x"], metric="percent")
        assert pd.isna(m.metric_value.at[1, "x"])
        assert m.note.at[1, "x"] == "zero_baseline"
        assert m.changed.at[1, "x"]

    def test_absolute_metric_equals_the_delta(self, lineage_df):
        m = _matrix(lineage_df, [10, 9], metric="absolute")
        assert m.metric_value.at[10, "ip"] == pytest.approx(m.delta.at[10, "ip"])

    def test_unknown_metric_falls_back_to_zscore(self, lineage_df, caplog):
        assert _matrix(lineage_df, [10, 9], metric="bogus").metric == "zscore"
        assert "unknown metric" in caplog.text

    def test_numeric_stored_as_strings_gets_a_real_delta(self, lineage_df):
        m = _matrix(lineage_df, [10, 9])
        assert m.delta.at[10, "ip_str"] == pytest.approx(20.0)  # 200 - 180

    def test_string_column_equal_and_changed(self, lineage_df):
        m = _matrix(lineage_df, [10, 9, 5])
        assert not m.changed.at[10, "scenario"]  # both L-mode
        assert m.changed.at[9, "scenario"]  # L-mode vs H-mode
        assert pd.isna(m.magnitude.at[9, "scenario"])
        assert m.kinds["scenario"] == "categorical"

    def test_null_like_strings_compare_as_missing(self):
        df = pd.DataFrame({"shot_id": [1, 2, 3], "s": ["", None, "nan"]})
        m = lineage_change_matrix(df, [1, 2, 3], ["s"])
        assert not m.changed.at[1, "s"] and not m.changed.at[2, "s"]

    def test_inf_is_shown_but_excluded_from_the_delta(self, lineage_df):
        m = _matrix(lineage_df, [1, 2])
        assert np.isinf(m.values.at[1, "sparse"])
        assert pd.isna(m.delta.at[1, "sparse"])
        assert m.note.at[1, "sparse"] == "non_finite"

    def test_nan_vs_nan_is_unchanged_but_one_sided_nan_changes(self, lineage_df):
        m = _matrix(lineage_df, [10, 9])
        assert not m.changed.at[10, "sparse"]  # missing in both
        m2 = _matrix(lineage_df, [3, 2])
        assert m2.changed.at[3, "sparse"]  # missing in 3, present in 2
        assert m2.note.at[3, "sparse"] == "missing"
        assert pd.isna(m2.magnitude.at[3, "sparse"])  # ranks last, never first

    def test_missing_lineage_shot_yields_a_nan_row(self, lineage_df):
        m = _matrix(lineage_df, [10, 999])
        assert 999 in m.values.index
        assert pd.isna(m.values.at[999, "ip"])

    def test_duplicate_shot_id_does_not_raise(self, lineage_df):
        doubled = pd.concat([lineage_df, lineage_df.iloc[[9]]], ignore_index=True)
        m = lineage_change_matrix(doubled, [10, 9], COLUMNS)
        assert m.values.at[10, "ip"] == pytest.approx(100.0)

    def test_unknown_column_is_dropped(self, lineage_df, caplog):
        m = lineage_change_matrix(lineage_df, [10, 9], ["ip", "nope"])
        assert m.columns == ["ip"]
        assert "not found in data" in caplog.text

    def test_near_equal_floats_are_unchanged(self):
        df = pd.DataFrame({"shot_id": [1, 2, 3], "x": [0.1, 0.1 + 1e-15, 0.1001]})
        m = lineage_change_matrix(df, [1, 2, 3], ["x"], metric="absolute")
        assert not m.changed.at[1, "x"]  # parse noise
        assert m.changed.at[2, "x"]  # a real change

    def test_zero_to_tiny_counts_as_a_change(self):
        df = pd.DataFrame({"shot_id": [1, 2], "x": [1e-12, 0.0]})
        m = lineage_change_matrix(df, [1, 2], ["x"], metric="absolute")
        assert m.changed.at[1, "x"]

    def test_excluded_is_carried_through(self, lineage_df):
        m = _matrix(lineage_df, [10, 9], excluded=frozenset({9}))
        assert m.excluded == frozenset({9})


class TestRankLineageChanges:
    def test_ordered_by_magnitude_descending(self, lineage_df):
        items = rank_lineage_changes(_matrix(lineage_df, [10, 9]))
        magnitudes = [i.magnitude for i in items if np.isfinite(i.magnitude)]
        assert magnitudes == sorted(magnitudes, reverse=True)

    def test_empty_for_a_lineage_of_one(self, lineage_df):
        assert rank_lineage_changes(_matrix(lineage_df, [10])) == []

    def test_reports_old_new_and_percent(self, lineage_df):
        item = next(i for i in rank_lineage_changes(_matrix(lineage_df, [10, 9])) if i.column == "ip")
        assert (item.old, item.new) == (90.0, 100.0)
        assert item.delta == pytest.approx(10.0)
        assert item.pct == pytest.approx(100.0 * 10.0 / 90.0)

    def test_percent_is_none_for_a_zero_baseline(self):
        df = pd.DataFrame({"shot_id": [1, 2], "x": [5.0, 0.0]})
        item = rank_lineage_changes(lineage_change_matrix(df, [1, 2], ["x"], metric="absolute"))[0]
        assert item.pct is None

    def test_unusable_magnitude_ranks_last(self, lineage_df):
        # A one-sided NaN is a real change, but "not recorded here" must never
        # outrank a change that can be measured.
        items = rank_lineage_changes(_matrix(lineage_df, [3, 2]))
        assert np.isfinite(items[0].magnitude)
        assert not np.isfinite(items[-1].magnitude)

    def test_free_text_columns_rank_last(self, lineage_df):
        items = rank_lineage_changes(_matrix(lineage_df, [10, 9]), top_n=None)
        assert items[-1].column == "comment"

    def test_respects_top_n(self, lineage_df):
        assert len(rank_lineage_changes(_matrix(lineage_df, [10, 9]), top_n=2)) == 2

    def test_unchanged_columns_are_dropped_by_default(self, lineage_df):
        items = rank_lineage_changes(_matrix(lineage_df, [10, 9]), top_n=None)
        assert "const" not in [i.column for i in items]  # zero variance

    def test_every_column_can_be_kept_with_the_unchanged_ones_last(self, lineage_df):
        m = _matrix(lineage_df, [10, 9])
        items = rank_lineage_changes(m, top_n=None, changed_only=False)
        assert [i.column for i in items] and set(i.column for i in items) == set(m.columns)
        assert not items[-1].changed or not np.isfinite(items[-1].magnitude)
        assert next(i for i in items if i.column == "const").changed is False

    def test_unchanged_columns_sort_below_every_change(self, lineage_df):
        # An unchanged column has a perfectly usable magnitude of 0, so without
        # an explicit rule "nothing happened here" would outrank "this
        # diagnostic was recorded for the first time".
        items = rank_lineage_changes(_matrix(lineage_df, [3, 2]), top_n=None, changed_only=False)
        flags = [item.changed for item in items]
        assert flags == sorted(flags, reverse=True)

    def test_carries_the_signed_metric_value_it_was_ranked_by(self, lineage_df):
        m = _matrix(lineage_df, [10, 9], metric="zscore")
        item = next(i for i in rank_lineage_changes(m) if i.column == "ip")
        assert item.metric_value == pytest.approx(m.metric_value.at[10, "ip"])
        assert item.magnitude == pytest.approx(abs(item.metric_value))
        assert item.changed is True

    def test_agrees_with_the_matrix_top_row(self, lineage_df):
        m = _matrix(lineage_df, [10, 9, 8])
        for item in rank_lineage_changes(m, top_n=None):
            assert m.changed.at[10, item.column]
            assert item.new == m.values.at[10, item.column]


class TestReferenceCompareColumns:
    def test_excludes_shot_id_and_projection_coords(self, lineage_df):
        df = lineage_df.assign(umap_x=0.0, umap_y=1.0)
        cols = reference_compare_columns(df)
        assert not {"shot_id", "umap_x", "umap_y"} & set(cols)

    def test_excludes_the_reference_column_when_asked(self, lineage_df):
        assert "ref_shot" not in reference_compare_columns(lineage_df, exclude=["ref_shot"])

    def test_search_cols_are_offered_first_but_nothing_is_dropped(self, lineage_df):
        cols = reference_compare_columns(lineage_df, search_cols=["sparse"], exclude=["ref_shot"])
        assert cols[0] == "sparse"
        assert "ip" in cols and "scenario" in cols

    def test_free_text_columns_come_last(self, lineage_df):
        assert reference_compare_columns(lineage_df, exclude=["ref_shot"])[-1] == "comment"

    def test_categoricals_can_be_left_out(self, lineage_df):
        cols = reference_compare_columns(lineage_df, exclude=["ref_shot"], include_categorical=False)
        assert "scenario" not in cols and "ip" in cols


class TestSelectChangedColumns:
    def test_drops_unchanged_columns(self, lineage_df):
        selected = select_changed_columns(_matrix(lineage_df, [10, 9]))
        assert "const" not in selected  # zero variance, never changes
        assert "ip" in selected

    def test_can_keep_unchanged_columns(self, lineage_df):
        assert "const" in select_changed_columns(_matrix(lineage_df, [10, 9]), changed_only=False)

    def test_kept_unchanged_columns_come_last(self, lineage_df):
        # So a cap still keeps the columns that moved.
        m = _matrix(lineage_df, [10, 9])
        columns = select_changed_columns(m, max_columns=None, changed_only=False)
        assert set(columns) == set(m.columns)
        flags = [bool(m.changed[c].any()) for c in columns]
        assert flags == sorted(flags, reverse=True)

    def test_cap_keeps_the_largest_magnitudes(self, lineage_df):
        m = _matrix(lineage_df, [10, 9])
        assert select_changed_columns(m, max_columns=1) == select_changed_columns(m)[:1]

    def test_empty_when_nothing_changed(self, lineage_df):
        assert select_changed_columns(_matrix(lineage_df, [10])) == []

    def test_empty_matrix_gives_no_columns(self, lineage_df):
        assert select_changed_columns(lineage_change_matrix(lineage_df, [10], [])) == []
