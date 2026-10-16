
import polars as pl

from .constants import display as disp


def format_count(value: int, singular: str, plural: str) -> str:
    """'1 run' / '2 runs'. Naming the unit avoids a bare, ambiguous 'N'."""
    return f"{value:,} {singular if value == 1 else plural}"


def get_condition_order(config_dict: dict) -> list:
    """
    Returns the order of conditions in the config file.
    """

    condition_order = list(config_dict.keys())

    return condition_order


def get_condition_run_counts(config_dict: dict) -> dict:
    """
    {condition: number of files}.

    The plotting layer builds its own tick labels from this rather than parsing
    a formatted string, so the count stays a number until the moment it is drawn.
    """

    return {cond: len(paths) for cond, paths in config_dict.items()}


def get_condition_plot_labels(config_dict: dict, include_counts: bool) -> dict:
    """
    Build a mapping from each raw condition name to the label used in plots.

    The config_dict keys (the raw condition names) are left untouched so that
    everything downstream — the condition order and the 'Condition' column in
    the dataframes — keeps the clean names. Only the *values* of the returned
    mapping carry the optional ' (<n> runs)' suffix, and it is this mapping
    that the plotting modules should use for axis / legend labels.

    The suffix counts *runs* (files) and says so. Observation counts appear
    elsewhere in the figures and are labelled 'obs', so the two can never
    be confused for one another.

    Args:
        config_dict (dict): {condition_name: [file_paths, ...]}.
        include_counts (bool): If True, append ' (<n> runs)' to each label,
            where <n> is the number of file paths for that condition.

    Returns:
        dict: {raw_condition_name: display_label}. When include_counts is False
            the label is identical to the raw name, so the plotting layer can
            always look up a label without branching.

    Example (include_counts=True):
        {"Control": ["fp1", "fp2"]} → {"Control": "Control (2 runs)"}
    Example (include_counts=False):
        {"Control": ["fp1", "fp2"]} → {"Control": "Control"}
    """

    if include_counts:
        return {
            cond: f"{cond} ({format_count(len(paths), disp.UNIT_RUN_SINGULAR, disp.UNIT_RUN_PLURAL)})"
            for cond, paths in config_dict.items()
        }

    return {cond: cond for cond in config_dict}


def sort_dataframe(
    df: pl.DataFrame,
    columns_to_sort: list[str],
) -> pl.DataFrame:
    return df.sort(columns_to_sort)


def reorder_dataframe(
    df: pl.DataFrame,
    column: str,
    group_id_order: list[int],
) -> pl.DataFrame:
    """
    Reorder rows using a user-provided group ordering
    while preserving the relative ordering of records
    within each group.
    """

    ordering_df = (
        pl.DataFrame({column: group_id_order})
        .with_row_index("desired_order")
    )

    return (
        df
        .with_row_index("_row_order")
        .join(ordering_df, on=column)
        .sort(
            ["desired_order", "_row_order"]
        )
        .drop(
            ["desired_order", "_row_order"]
        )
    )

