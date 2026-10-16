"""Encode statistical evidence compactly for LLM prompts."""

from collections.abc import Mapping

_MISSING = object()

# Leading column added to every encoded row when row indices are written out.
ROW_INDEX_COLUMN = "row_index"


def encode_table(
    rows: list[dict],
    *,
    hoist_constants: bool = False,
    include_row_index: bool = False,
) -> dict:
    """Represent table rows without repeating column names.

    With ``hoist_constants``, a column whose value is identical in every row
    (and present in every row) is written once under ``constants`` instead of
    once per row. Settings such as ``ci_method`` or ``p_adjust_method`` repeat
    on every pairwise row, and on a free-tier token budget that repetition is
    what pushes the largest prompt over the per-minute limit. Row order, and so
    every zero-based row index, is unchanged.

    With ``include_row_index``, each row starts with its own zero-based index
    under a leading ``row_index`` column. A model cannot reliably count to row
    47 of a 78-row array; it can copy a number written at the start of the row.
    """

    columns = list(dict.fromkeys(
        key
        for row in rows
        for key in row
    ))

    constants: dict = {}
    if hoist_constants and len(rows) > 1:
        for column in columns:
            first = rows[0].get(column, _MISSING)
            if first is _MISSING:
                continue
            if all(
                column in row and _same(row[column], first)
                for row in rows[1:]
            ):
                constants[column] = first
        columns = [column for column in columns if column not in constants]

    encoded_rows = [[row.get(column) for column in columns] for row in rows]
    if include_row_index:
        columns = [ROW_INDEX_COLUMN, *columns]
        encoded_rows = [[index, *values] for index, values in enumerate(encoded_rows)]

    encoded = {
        "columns": columns,
        "rows": encoded_rows,
    }
    if constants:
        encoded["constants"] = constants
    return encoded


def _same(left, right) -> bool:
    """Equal and of the same type, so 1 and 1.0 or 1 and True never merge."""
    return type(left) is type(right) and left == right


def encode_evidence(
    selected_evidence: Mapping,
    *,
    hoist_constants: bool = True,
    include_row_index: bool = True,
) -> dict:
    """Encode selected evidence while preserving original row order."""

    def encode_tables(tables: Mapping) -> dict:
        result = {}

        for name, value in tables.items():
            if isinstance(value, list):
                result[name] = encode_table(
                    value,
                    hoist_constants=hoist_constants,
                    include_row_index=include_row_index,
                )
            elif isinstance(value, Mapping):
                result[name] = encode_tables(value)
            else:
                raise ValueError(
                    f"Unsupported evidence table structure: {name}"
                )

        return result

    return {
        **selected_evidence,
        "tables": encode_tables(selected_evidence["tables"]),
    }
