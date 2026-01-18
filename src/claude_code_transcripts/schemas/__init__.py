"""Schema definitions for Claude Code transcripts.

This package provides two schemas:
- simple: 4 tables (sessions, messages, tool_calls, thinking)
- star: 25+ dimensional tables for analytics

Use resolve_schema_format() to handle CLI schema/format combinations.
"""

from .simple import (
    create_duckdb_schema,
    export_session_to_duckdb,
    export_sessions_to_json,
    _extract_session_data,
)

from .star import (
    create_star_schema,
    create_semantic_model,
    run_star_schema_etl,
    export_star_schema_to_json,
    run_llm_enrichment,
    run_session_insights_enrichment,
    generate_dimension_key,
    get_tool_category,
    get_model_family,
    get_time_of_day,
    TOOL_CATEGORIES,
)


def resolve_schema_format(schema, output_format):
    """Resolve schema and format from potentially compound format names.

    Supports hybrid CLI: explicit --schema flag or compound format names like
    'duckdb-star' or 'json-star'.

    Args:
        schema: Explicit schema ('simple' or 'star') or None
        output_format: Format string ('html', 'duckdb', 'duckdb-star', 'json', 'json-star')

    Returns:
        Tuple of (resolved_schema, resolved_format)
    """
    # Handle compound format names
    if output_format.endswith("-star"):
        inferred_schema = "star"
        actual_format = output_format.replace("-star", "")
    else:
        inferred_schema = "simple"
        actual_format = output_format

    # Explicit --schema overrides inference
    final_schema = schema if schema else inferred_schema

    return final_schema, actual_format


__all__ = [
    # Simple schema
    "create_duckdb_schema",
    "export_session_to_duckdb",
    "export_sessions_to_json",
    "_extract_session_data",
    # Star schema
    "create_star_schema",
    "create_semantic_model",
    "run_star_schema_etl",
    "export_star_schema_to_json",
    "run_llm_enrichment",
    "run_session_insights_enrichment",
    "generate_dimension_key",
    "get_tool_category",
    "get_model_family",
    "get_time_of_day",
    "TOOL_CATEGORIES",
    # Utilities
    "resolve_schema_format",
]
