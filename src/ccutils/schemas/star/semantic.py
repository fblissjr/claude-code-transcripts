"""Semantic model generation for data exploration."""


def create_semantic_model(conn):
    """Create and populate the meta_semantic_model table with schema metadata.

    This function introspects the star schema and creates a metadata table
    that describes:
    - Table types (dimension, fact, staging)
    - Column types (key, attribute, measure)
    - Data types (normalized)
    - Relationships between tables
    - Default aggregations for measures
    - Display names for UI

    The semantic model enables data exploration tools to understand
    the schema without hardcoded knowledge.

    Args:
        conn: DuckDB connection to a database with star schema tables
    """
    conn.execute("DROP TABLE IF EXISTS meta_semantic_model")
    conn.execute(
        """
        CREATE TABLE meta_semantic_model (
            table_name VARCHAR,
            table_type VARCHAR,
            table_display_name VARCHAR,
            column_name VARCHAR,
            column_type VARCHAR,
            data_type VARCHAR,
            display_name VARCHAR,
            default_aggregation VARCHAR,
            related_table VARCHAR,
            related_column VARCHAR,
            is_visible BOOLEAN DEFAULT TRUE,
            is_filterable BOOLEAN DEFAULT TRUE,
            sort_order INTEGER
        )
    """
    )

    tables = conn.execute(
        """SELECT table_name
           FROM information_schema.tables
           WHERE table_schema = 'main'
           ORDER BY table_name"""
    ).fetchall()
    table_names = [t[0] for t in tables]

    dim_tables = {t for t in table_names if t.startswith("dim_")}

    sort_order = 0
    for table_name in table_names:
        if table_name == "meta_semantic_model":
            continue

        table_type = _get_table_type(table_name)
        table_display_name = _generate_display_name(table_name)

        columns = conn.execute(
            f"""SELECT column_name, data_type
               FROM information_schema.columns
               WHERE table_name = '{table_name}'
               ORDER BY ordinal_position"""
        ).fetchall()

        for col_name, raw_data_type in columns:
            sort_order += 1

            data_type = _normalize_data_type(raw_data_type)
            column_type = _detect_column_type(col_name, data_type, table_type)
            display_name = _generate_column_display_name(col_name)

            default_aggregation = None
            if column_type == "measure":
                default_aggregation = _infer_aggregation(col_name, data_type)

            related_table = None
            related_column = None
            if column_type == "key" and table_type in ("fact", "staging"):
                related_table, related_column = _find_relationship(
                    col_name, table_name, dim_tables
                )

            is_visible = column_type != "key" or table_type == "dimension"
            is_filterable = column_type in ("key", "attribute")

            conn.execute(
                """INSERT INTO meta_semantic_model
                   (table_name, table_type, table_display_name, column_name,
                    column_type, data_type, display_name, default_aggregation,
                    related_table, related_column, is_visible, is_filterable, sort_order)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    table_name,
                    table_type,
                    table_display_name,
                    col_name,
                    column_type,
                    data_type,
                    display_name,
                    default_aggregation,
                    related_table,
                    related_column,
                    is_visible,
                    is_filterable,
                    sort_order,
                ],
            )


def _get_table_type(table_name):
    """Determine table type from naming convention."""
    if table_name.startswith("dim_"):
        return "dimension"
    elif table_name.startswith("fact_"):
        return "fact"
    elif table_name.startswith("stg_"):
        return "staging"
    elif table_name.startswith("semantic_"):
        return "semantic"
    return "other"


def _generate_display_name(table_name):
    """Generate human-readable display name from table name."""
    name = table_name
    for prefix in ("dim_", "fact_", "stg_"):
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break
    return name.replace("_", " ").title()


def _generate_column_display_name(col_name):
    """Generate human-readable display name from column name."""
    return col_name.replace("_", " ").title()


def _normalize_data_type(raw_type):
    """Normalize DuckDB data types to standard categories."""
    raw_type = raw_type.upper()

    if "VARCHAR" in raw_type or "TEXT" in raw_type or "CHAR" in raw_type:
        return "varchar"
    elif "INT" in raw_type and "BIGINT" not in raw_type:
        return "integer"
    elif "BIGINT" in raw_type:
        return "integer"
    elif "FLOAT" in raw_type or "DOUBLE" in raw_type or "DECIMAL" in raw_type:
        return "float"
    elif "TIMESTAMP" in raw_type:
        return "timestamp"
    elif "DATE" in raw_type and "TIMESTAMP" not in raw_type:
        return "date"
    elif "BOOL" in raw_type:
        return "boolean"
    elif "JSON" in raw_type:
        return "json"
    return "varchar"


def _detect_column_type(col_name, data_type, table_type):
    """Detect the semantic type of a column."""
    if col_name.endswith("_key") or col_name.endswith("_id"):
        return "key"

    # Semantic views inherit measure detection from facts
    is_fact_like = table_type in ("fact", "semantic")

    if is_fact_like and data_type in ("integer", "float"):
        measure_suffixes = (
            "_count",
            "_length",
            "_score",
            "_seconds",
            "_size",
            "_chars",
            "_tokens",
            "_depth",
        )
        if any(col_name.endswith(suffix) for suffix in measure_suffixes):
            return "measure"
        if col_name.startswith("total_") or col_name.startswith("estimated_"):
            return "measure"

    if is_fact_like and data_type == "boolean":
        if col_name.startswith("is_") or col_name.startswith("has_"):
            return "measure"

    return "attribute"


def _infer_aggregation(col_name, data_type):
    """Infer the default aggregation for a measure column."""
    if "_count" in col_name or col_name.startswith("total_"):
        return "sum"
    if "_length" in col_name or "_size" in col_name or "_chars" in col_name:
        return "sum"
    if "_tokens" in col_name:
        return "sum"
    if "_seconds" in col_name or "_time" in col_name:
        return "avg"
    if "_score" in col_name:
        return "avg"
    if "_depth" in col_name:
        return "max"
    if data_type == "boolean":
        return "sum"
    return "sum"


def _find_relationship(col_name, table_name, dim_tables):
    """Find the dimension table that a key column relates to."""
    base_name = col_name
    if base_name.endswith("_key"):
        base_name = base_name[:-4]
    elif base_name.endswith("_id"):
        base_name = base_name[:-3]

    candidate = f"dim_{base_name}"
    if candidate in dim_tables:
        return (candidate, col_name)

    for dim_table in dim_tables:
        dim_base = dim_table[4:]
        if dim_base == base_name:
            return (dim_table, col_name)

    return (None, None)
