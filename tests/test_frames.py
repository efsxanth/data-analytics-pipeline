"""Small dataframe and label helpers shared across the pipeline."""

from __future__ import annotations

import polars as pl

from da_pipeline.constants import columns as col
from da_pipeline.frames import (
    format_count,
    get_condition_order,
    get_condition_plot_labels,
    get_condition_run_counts,
    reorder_dataframe,
    sort_dataframe,
)

CONFIG_DICT = {
    "Condition_A": ["a0.h5", "a1.h5"],
    "Condition_B": ["b0.h5"],
}


class TestFormatCount:
    def test_singular_and_plural_are_chosen_by_value(self):
        # A condition built from one file reads '1 test', not '1 tests'.
        assert format_count(1, "test", "tests") == "1 test"
        assert format_count(2, "test", "tests") == "2 tests"

    def test_zero_is_plural(self):
        assert format_count(0, "test", "tests") == "0 tests"

    def test_large_counts_are_grouped(self):
        assert format_count(24262, "obs", "obs") == "24,262 obs"


class TestConditionMetadata:
    def test_order_follows_the_config_file(self):
        assert get_condition_order(CONFIG_DICT) == ["Condition_A", "Condition_B"]

    def test_test_counts_count_files(self):
        assert get_condition_run_counts(CONFIG_DICT) == {"Condition_A": 2, "Condition_B": 1}

    def test_labels_carry_the_count_when_asked(self):
        labels = get_condition_plot_labels(CONFIG_DICT, include_counts=True)

        assert labels["Condition_A"] == "Condition_A (2 runs)"
        assert labels["Condition_B"] == "Condition_B (1 run)"

    def test_labels_fall_back_to_the_raw_name(self):
        # Always a complete mapping, so the plotting layer never has to branch.
        labels = get_condition_plot_labels(CONFIG_DICT, include_counts=False)

        assert labels == {"Condition_A": "Condition_A", "Condition_B": "Condition_B"}

    def test_raw_names_are_never_modified(self):
        # The 'condition' column keeps clean names; counts live only in labels.
        get_condition_plot_labels(CONFIG_DICT, include_counts=True)

        assert list(CONFIG_DICT) == ["Condition_A", "Condition_B"]


class TestSorting:
    def test_sort_dataframe_orders_by_the_given_columns(self):
        df = pl.DataFrame({col.CONDITION: ["B", "A", "B"], col.GROUP_ID: [1, 2, 0]})

        result = sort_dataframe(df, [col.CONDITION, col.GROUP_ID])

        assert result[col.CONDITION].to_list() == ["A", "B", "B"]
        assert result[col.GROUP_ID].to_list() == [2, 0, 1]


class TestReorder:
    def test_rows_follow_the_requested_group_order(self):
        # --group_ids 3 1 means "3 first", even though 1 sorts lower.
        df = pl.DataFrame({col.GROUP_ID: [1, 3, 1, 3], "value": [10, 30, 11, 31]})

        result = reorder_dataframe(df, column=col.GROUP_ID, group_id_order=[3, 1])

        assert result[col.GROUP_ID].to_list() == [3, 3, 1, 1]

    def test_relative_order_within_a_group_is_preserved(self):
        df = pl.DataFrame({col.GROUP_ID: [1, 3, 1, 3], "value": [10, 30, 11, 31]})

        result = reorder_dataframe(df, column=col.GROUP_ID, group_id_order=[3, 1])

        assert result["value"].to_list() == [30, 31, 10, 11]

    def test_helper_columns_do_not_leak_into_the_output(self):
        df = pl.DataFrame({col.GROUP_ID: [1, 2], "value": [1, 2]})

        result = reorder_dataframe(df, column=col.GROUP_ID, group_id_order=[2, 1])

        assert result.columns == [col.GROUP_ID, "value"]

    def test_groups_absent_from_the_order_are_dropped(self):
        # The join is inner: the requested order is also a filter.
        df = pl.DataFrame({col.GROUP_ID: [1, 2, 3], "value": [1, 2, 3]})

        result = reorder_dataframe(df, column=col.GROUP_ID, group_id_order=[3, 1])

        assert result[col.GROUP_ID].to_list() == [3, 1]
