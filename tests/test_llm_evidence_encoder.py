
"""Tests for compact statistical evidence encoding."""

from copy import deepcopy

import pytest

from da_pipeline.llm.evidence_encoder import encode_evidence, encode_table


def test_encode_table_preserves_values_and_row_order():
    rows = [
        {"metric": "Accuracy", "eta_squared": 0.4},
        {"metric": "degradation_1", "eta_squared": 0.2},
    ]

    encoded = encode_table(rows)

    assert encoded["columns"] == ["metric", "eta_squared"]
    assert encoded["rows"] == [
        ["Accuracy", 0.4],
        ["degradation_1", 0.2],
    ]


def test_encode_table_handles_different_row_keys():
    rows = [
        {"metric": "Accuracy", "effect": 0.4},
        {"metric": "Accuracy", "p_value": 0.03},
    ]

    encoded = encode_table(rows)

    assert encoded["columns"] == ["metric", "effect", "p_value"]
    assert encoded["rows"] == [
        ["Accuracy", 0.4, None],
        ["Accuracy", None, 0.03],
    ]


def test_encode_table_handles_empty_rows():
    assert encode_table([]) == {
        "columns": [],
        "rows": [],
    }


def test_encode_evidence_preserves_nested_correlation_views():
    selected = {
        "question_id": "relationships",
        "tables": {
            "correlations": {
                "correlations": [
                    {"var_x": "temperature", "pearson_r": 0.6}
                ],
                "correlations_within_run": [
                    {"var_x": "temperature", "pearson_r_median": 0.5}
                ],
                "correlations_between_run": [
                    {"var_x": "temperature", "pearson_r": 0.4}
                ],
            }
        },
    }

    encoded = encode_evidence(selected)
    correlation_tables = encoded["tables"]["correlations"]

    assert set(correlation_tables) == {
        "correlations",
        "correlations_within_run",
        "correlations_between_run",
    }

    assert correlation_tables["correlations"]["rows"] == [
        [0, "temperature", 0.6]
    ]
    assert correlation_tables["correlations_within_run"]["rows"] == [
        [0, "temperature", 0.5]
    ]
    assert correlation_tables["correlations_between_run"]["rows"] == [
        [0, "temperature", 0.4]
    ]


def test_encoding_does_not_modify_original_evidence():
    selected = {
        "tables": {
            "effect_size": [
                {"metric": "Accuracy", "eta_squared": 0.4}
            ]
        }
    }

    original = deepcopy(selected)

    encode_evidence(selected)

    assert selected == original


def test_unsupported_table_structure_is_rejected():
    selected = {
        "tables": {
            "effect_size": "invalid table",
        }
    }

    with pytest.raises(
        ValueError,
        match="Unsupported evidence table structure",
    ):
        encode_evidence(selected)


def test_encoded_rows_reconstruct_original_tables():
    original_rows = [
        {"metric": "Accuracy", "eta_squared": 0.4},
        {"metric": "degradation_1", "eta_squared": 0.2},
    ]

    encoded = encode_table(original_rows)

    reconstructed = [
        dict(zip(encoded["columns"], row, strict=True))
        for row in encoded["rows"]
    ]

    assert reconstructed == original_rows


def test_constant_columns_are_hoisted_and_rows_still_reconstruct():
    rows = [
        {"metric": "Accuracy", "p": 0.01, "ci_method": "bootstrap", "level": None, "n": 3},
        {"metric": "degradation_1", "p": 0.2, "ci_method": "bootstrap", "level": None, "n": 3.0},
        {"metric": "degradation_2", "p": 0.3, "ci_method": "bootstrap", "level": None, "n": 3},
    ]

    encoded = encode_table(rows, hoist_constants=True)

    assert encoded["constants"] == {"ci_method": "bootstrap", "level": None}
    assert encoded["columns"] == ["metric", "p", "n"]
    reconstructed = [
        {**encoded["constants"], **dict(zip(encoded["columns"], row, strict=True))}
        for row in encoded["rows"]
    ]
    assert reconstructed == rows


def test_column_missing_from_a_row_is_not_hoisted():
    rows = [
        {"metric": "Accuracy", "effect": 0.4},
        {"metric": "Accuracy"},
    ]

    encoded = encode_table(rows, hoist_constants=True)

    assert encoded["constants"] == {"metric": "Accuracy"}
    assert encoded["rows"] == [[0.4], [None]]


def test_single_row_tables_are_not_hoisted():
    encoded = encode_table([{"metric": "Accuracy", "r": 0.5}], hoist_constants=True)

    assert "constants" not in encoded
    assert encoded["rows"] == [["Accuracy", 0.5]]


def test_encode_evidence_hoists_by_default():
    selected = {
        "tables": {
            "pairwise_conditions": [
                {"pair": "A-B", "p": 0.1, "ci_level": 0.95},
                {"pair": "A-C", "p": 0.2, "ci_level": 0.95},
            ]
        }
    }

    table = encode_evidence(selected)["tables"]["pairwise_conditions"]

    assert table["constants"] == {"ci_level": 0.95}
    assert table["columns"] == ["row_index", "pair", "p"]
    assert table["rows"] == [[0, "A-B", 0.1], [1, "A-C", 0.2]]
    assert encode_evidence(
        selected, hoist_constants=False, include_row_index=False
    )["tables"]["pairwise_conditions"]["columns"] == ["pair", "p", "ci_level"]
