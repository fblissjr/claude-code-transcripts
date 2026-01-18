"""Star Schema package for Claude Code Transcripts.

This package provides:
- DuckDB star schema creation for transcript analytics
- Semantic model generation for data exploration
- ETL pipeline for loading session data
- LLM enrichment pipeline for message classification
- JSON export for star schema data
"""

from .enrichment import run_llm_enrichment, run_session_insights_enrichment
from .etl import run_star_schema_etl
from .json_export import export_star_schema_to_json
from .schema import create_star_schema
from .semantic import create_semantic_model
from .utils import (
    TOOL_CATEGORIES,
    generate_dimension_key,
    get_model_family,
    get_time_of_day,
    get_tool_category,
)

__all__ = [
    # Schema creation
    "create_star_schema",
    # Semantic model
    "create_semantic_model",
    # ETL
    "run_star_schema_etl",
    # JSON export
    "export_star_schema_to_json",
    # LLM enrichment
    "run_llm_enrichment",
    "run_session_insights_enrichment",
    # Utilities
    "generate_dimension_key",
    "get_tool_category",
    "get_model_family",
    "get_time_of_day",
    "TOOL_CATEGORIES",
]
