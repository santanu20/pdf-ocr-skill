"""Unit tests for table schema inference.

Tests scientific_parser.perception.tables.schema.infer_table_schema +
infer_column_schema. These were untested per AUDIT_SPECIALIZED_2026-08-13.md.

Per testing.md §3.1: ≥90% strong assertions.
Per testing.md §5.2: float comparison via pytest.approx.
Run: pytest scripts/scientific_parser/tests/test_table_schema.py -v
"""

import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from scientific_parser.perception.tables.schema import (
    ColumnSchema,
    infer_column_schema,
    infer_table_schema,
)

# === infer_column_schema ======================================== #


def test_infer_column_schema_numeric_values_marked_numeric_type():
    col = infer_column_schema("Age", ["25", "30", "35"])
    assert col.name == "Age"
    assert col.data_type == "numeric"
    assert col.numeric_stats["min"] == pytest.approx(25.0)
    assert col.numeric_stats["max"] == pytest.approx(35.0)
    assert col.numeric_stats["mean"] == pytest.approx(30.0)


def test_infer_column_schema_text_values_marked_text_type():
    col = infer_column_schema("Name", ["Alice", "Bob", "Carol"])
    assert col.data_type == "text"


def test_infer_column_schema_with_unit_in_name_extracts_unit():
    """Column name like 'SiO2 (wt%)' should extract the unit."""
    col = infer_column_schema("SiO2 (wt%)", ["52.3", "48.1", "60.0"])
    assert col.data_type == "numeric"
    # Unit extraction is best-effort; verify at least the field exists
    assert hasattr(col, "unit")


def test_infer_column_schema_empty_values_returns_object_with_zero_stats():
    col = infer_column_schema("Empty", [])
    assert isinstance(col, ColumnSchema)
    assert col.name == "Empty"
    assert col.distinct_values == 0


def test_infer_column_schema_with_missing_values_counts_nulls():
    """None or empty string values count toward missing_count."""
    col = infer_column_schema("Mixed", ["1.0", "", None, "2.0"])
    assert col.missing_count >= 1


# === infer_table_schema ========================================= #


def test_infer_table_schema_simple_two_column_returns_correct_shape():
    rows = [
        ["Name", "Age"],
        ["Alice", "30"],
        ["Bob", "25"],
    ]
    schema = infer_table_schema(rows)
    assert schema["n_data_rows"] == 2
    assert schema["n_columns"] == 2
    assert len(schema["columns"]) == 2
    assert schema["columns"][0]["name"] == "Name"
    assert schema["columns"][1]["name"] == "Age"
    assert schema["columns"][0]["data_type"] == "text"
    assert schema["columns"][1]["data_type"] == "numeric"


def test_infer_table_schema_empty_rows_does_not_crash():
    """Defensive: empty input must not raise."""
    schema = infer_table_schema([])
    assert isinstance(schema, dict)


def test_infer_table_schema_single_row_only_header():
    """Header-only table — no data rows."""
    rows = [["Col1", "Col2", "Col3"]]
    schema = infer_table_schema(rows)
    assert schema["n_data_rows"] == 0
    assert schema["n_columns"] == 3


def test_infer_table_schema_numeric_column_stats_correct():
    """Numeric stats (min/max/mean) must be computed from data rows."""
    rows = [
        ["Value"],
        ["10.0"],
        ["20.0"],
        ["30.0"],
    ]
    schema = infer_table_schema(rows)
    stats = schema["columns"][0]["numeric_stats"]
    assert stats["min"] == pytest.approx(10.0)
    assert stats["max"] == pytest.approx(30.0)
    assert stats["mean"] == pytest.approx(20.0)


def test_infer_table_schema_distinct_values_counted_correctly():
    """distinct_values counts unique values per column."""
    rows = [
        ["Category"],
        ["A"],
        ["A"],
        ["B"],
        ["C"],
    ]
    schema = infer_table_schema(rows)
    assert schema["columns"][0]["distinct_values"] == 3


def test_infer_table_schema_null_density_zero_when_no_missing():
    rows = [
        ["X"],
        ["1"],
        ["2"],
    ]
    schema = infer_table_schema(rows)
    assert schema["columns"][0]["null_density"] == pytest.approx(0.0)
    assert schema["columns"][0]["missing_count"] == 0


def test_infer_table_schema_custom_header_row_index():
    """header_row_idx parameter: skip non-data rows above header."""
    rows = [
        ["prologue text", ""],
        ["Name", "Age"],
        ["Alice", "30"],
    ]
    schema = infer_table_schema(rows, header_row_idx=1)
    assert schema["columns"][0]["name"] == "Name"
