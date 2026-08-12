"""Tests for reference-graph, filter, and clickData-parsing logic in nice_shot/analysis.py."""

from __future__ import annotations

import pandas as pd
import pytest

from nice_shot.analysis import (
    _apply_filter_mask,
    _build_reference_graph,
    _extract_shot_id,
    compute_active_filter_ids,
    get_reference_graph,
)


class TestBuildReferenceGraph:
    def test_simple_chain(self):
        data = pd.DataFrame({"shot_id": [1, 2, 3], "ref_shot": [None, 1, 2]})
        adjacency, parent = _build_reference_graph(data, "ref_shot")
        assert parent == {2: 1, 3: 2}
        assert set(adjacency[2]) == {1, 3}

    def test_branching(self):
        data = pd.DataFrame({"shot_id": [1, 2, 3], "ref_shot": [None, 1, 1]})
        adjacency, parent = _build_reference_graph(data, "ref_shot")
        assert parent == {2: 1, 3: 1}
        assert set(adjacency[1]) == {2, 3}

    def test_missing_reference_is_ignored(self):
        data = pd.DataFrame({"shot_id": [1, 2], "ref_shot": [None, 999]})
        adjacency, parent = _build_reference_graph(data, "ref_shot")
        assert parent == {}
        assert adjacency == {}

    def test_self_reference_is_ignored(self):
        data = pd.DataFrame({"shot_id": [1, 2], "ref_shot": [1, 1]})
        adjacency, parent = _build_reference_graph(data, "ref_shot")
        assert 1 not in parent


class TestGetReferenceGraph:
    def test_bfs_over_chain(self):
        adjacency = {1: [2], 2: [1, 3], 3: [2]}
        assert get_reference_graph(adjacency, 1) == {1, 2, 3}

    def test_bfs_with_cycle(self):
        adjacency = {1: [2, 3], 2: [1, 3], 3: [2, 1]}
        assert get_reference_graph(adjacency, 1) == {1, 2, 3}

    def test_empty_adjacency_returns_empty_set(self):
        assert get_reference_graph({}, 1) == set()

    def test_isolated_node_returns_itself(self):
        adjacency = {1: [2], 2: [1]}
        assert get_reference_graph(adjacency, 99) == {99}


class TestApplyFilterMask:
    def test_none_returns_full_dataframe(self):
        df = pd.DataFrame({"shot_id": [1, 2, 3]})
        result = _apply_filter_mask(df, None)
        assert result is df

    def test_filters_by_shot_id_list(self):
        df = pd.DataFrame({"shot_id": [1, 2, 3]})
        result = _apply_filter_mask(df, [1, 3])
        assert list(result["shot_id"]) == [1, 3]


class TestComputeActiveFilterIds:
    @pytest.fixture
    def df(self):
        return pd.DataFrame({"shot_id": [1, 2, 3, 4], "value": [1.0, 2.0, 3.0, 4.0], "machine": ["a", "b", "a", "c"]})

    @pytest.mark.parametrize(
        "op,val,expected",
        [
            (">=", "2", [2, 3, 4]),
            ("<=", "2", [1, 2]),
            (">", "2", [3, 4]),
            ("<", "2", [1]),
            ("==", "2", [2]),
            ("!=", "2", [1, 3, 4]),
        ],
    )
    def test_numeric_operators(self, df, op, val, expected):
        result = compute_active_filter_ids(df, ["value"], [op], [val], "AND")
        assert result == expected

    def test_contains_operator(self, df):
        result = compute_active_filter_ids(df, ["machine"], ["contains"], ["a"], "AND")
        assert result == [1, 3]

    def test_and_combination(self, df):
        result = compute_active_filter_ids(df, ["value", "machine"], [">=", "contains"], ["2", "a"], "AND")
        assert result == [3]

    def test_or_combination(self, df):
        result = compute_active_filter_ids(df, ["value", "machine"], [">=", "contains"], ["2", "a"], "OR")
        assert result == [1, 2, 3, 4]

    def test_no_active_filters_returns_none(self, df):
        assert compute_active_filter_ids(df, [None], [None], [None], "AND") is None
        assert compute_active_filter_ids(df, ["value"], [">="], [""], "AND") is None


class TestExtractShotId:
    @pytest.fixture
    def df(self):
        return pd.DataFrame({"shot_id": [10, 20, 30]})

    def test_no_click_data_returns_none(self, df):
        assert _extract_shot_id(df, {}) is None
        assert _extract_shot_id(df, None) is None

    def test_hovertext_path(self, df):
        click_data = {"points": [{"hovertext": "20"}]}
        assert _extract_shot_id(df, click_data) == 20

    def test_customdata_list_path(self, df):
        click_data = {"points": [{"customdata": [30]}]}
        assert _extract_shot_id(df, click_data) == 30

    def test_customdata_scalar_path(self, df):
        click_data = {"points": [{"customdata": 30}]}
        assert _extract_shot_id(df, click_data) == 30

    def test_point_index_fallback(self, df):
        click_data = {"points": [{"pointIndex": 1}]}
        assert _extract_shot_id(df, click_data) == 20

    def test_point_index_fallback_skipped_when_color_present(self, df):
        click_data = {"points": [{"pointIndex": 1}], "color": "cluster"}
        assert _extract_shot_id(df, click_data) is None
