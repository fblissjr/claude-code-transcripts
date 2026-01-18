"""Simple 4-table schema for Claude Code transcripts.

This package provides a simple schema with 4 tables:
- sessions: Session metadata
- messages: All messages (user and assistant)
- tool_calls: Tool invocations and results
- thinking: Extended thinking blocks (optional)
"""

from .schema import create_duckdb_schema
from .etl import (
    export_session_to_duckdb,
    export_sessions_to_json,
    _extract_session_data,
)

__all__ = [
    "create_duckdb_schema",
    "export_session_to_duckdb",
    "export_sessions_to_json",
    "_extract_session_data",
]
