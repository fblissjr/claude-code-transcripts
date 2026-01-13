"""Export star schema DuckDB to JSON directory structure."""

import json
from datetime import datetime
from pathlib import Path


# Tables organized by type
DIMENSION_TABLES = [
    "dim_tool",
    "dim_model",
    "dim_project",
    "dim_session",
    "dim_date",
    "dim_time",
    "dim_message_type",
    "dim_content_block_type",
    "dim_file",
    "dim_programming_language",
    "dim_error_type",
    "dim_entity_type",
    "dim_intent",
    "dim_topic",
    "dim_sentiment",
]

FACT_TABLES = [
    "fact_messages",
    "fact_content_blocks",
    "fact_tool_calls",
    "fact_session_summary",
    "fact_file_operations",
    "fact_code_blocks",
    "fact_errors",
    "fact_entity_mentions",
    "fact_tool_chain_steps",
    "fact_message_enrichment",
    "fact_message_topics",
    "fact_session_insights",
]

# Key relationships for the star schema
RELATIONSHIPS = [
    {
        "from_table": "fact_messages",
        "from_column": "session_key",
        "to_table": "dim_session",
        "to_column": "session_key",
    },
    {
        "from_table": "fact_messages",
        "from_column": "model_key",
        "to_table": "dim_model",
        "to_column": "model_key",
    },
    {
        "from_table": "fact_messages",
        "from_column": "date_key",
        "to_table": "dim_date",
        "to_column": "date_key",
    },
    {
        "from_table": "fact_messages",
        "from_column": "time_key",
        "to_table": "dim_time",
        "to_column": "time_key",
    },
    {
        "from_table": "fact_tool_calls",
        "from_column": "session_key",
        "to_table": "dim_session",
        "to_column": "session_key",
    },
    {
        "from_table": "fact_tool_calls",
        "from_column": "tool_key",
        "to_table": "dim_tool",
        "to_column": "tool_key",
    },
    {
        "from_table": "dim_session",
        "from_column": "project_key",
        "to_table": "dim_project",
        "to_column": "project_key",
    },
    {
        "from_table": "fact_file_operations",
        "from_column": "file_key",
        "to_table": "dim_file",
        "to_column": "file_key",
    },
    {
        "from_table": "fact_code_blocks",
        "from_column": "language_key",
        "to_table": "dim_programming_language",
        "to_column": "language_key",
    },
]


def export_star_schema_to_json(conn, output_dir):
    """Export star schema DuckDB tables to JSON directory structure.

    Creates:
        output_dir/
            meta.json           - Schema metadata and relationships
            dimensions/         - One JSON file per dimension table
            facts/              - One JSON file per fact table

    Args:
        conn: DuckDB connection with star schema data
        output_dir: Directory to write JSON files to
    """
    output_dir = Path(output_dir)
    dimensions_dir = output_dir / "dimensions"
    facts_dir = output_dir / "facts"

    dimensions_dir.mkdir(parents=True, exist_ok=True)
    facts_dir.mkdir(parents=True, exist_ok=True)

    table_manifest = {"dimensions": [], "facts": []}

    # Export dimension tables
    for table_name in DIMENSION_TABLES:
        rows = _export_table(conn, table_name, dimensions_dir)
        table_manifest["dimensions"].append(
            {
                "name": table_name,
                "file": f"dimensions/{table_name}.json",
                "row_count": rows,
            }
        )

    # Export fact tables
    for table_name in FACT_TABLES:
        rows = _export_table(conn, table_name, facts_dir)
        table_manifest["facts"].append(
            {
                "name": table_name,
                "file": f"facts/{table_name}.json",
                "row_count": rows,
            }
        )

    # Write meta.json
    meta = {
        "version": "1.0",
        "schema_type": "star",
        "exported_at": datetime.now().astimezone().isoformat(),
        "tables": table_manifest,
        "relationships": RELATIONSHIPS,
    }

    meta_path = output_dir / "meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)


def _export_table(conn, table_name, output_dir):
    """Export a single table to JSON.

    Args:
        conn: DuckDB connection
        table_name: Name of the table to export
        output_dir: Directory to write to

    Returns:
        Number of rows exported
    """
    try:
        # Get column names
        columns_result = conn.execute(f"DESCRIBE {table_name}").fetchall()
        column_names = [row[0] for row in columns_result]

        # Get all rows
        rows = conn.execute(f"SELECT * FROM {table_name}").fetchall()

        # Convert to list of dicts
        data = []
        for row in rows:
            record = {}
            for i, col_name in enumerate(column_names):
                value = row[i]
                # Handle special types
                if hasattr(value, "isoformat"):
                    value = value.isoformat()
                record[col_name] = value
            data.append(record)

        # Write JSON file
        output_path = output_dir / f"{table_name}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

        return len(data)

    except Exception:
        # Table might not exist or be empty - write empty array
        output_path = output_dir / f"{table_name}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump([], f)
        return 0
