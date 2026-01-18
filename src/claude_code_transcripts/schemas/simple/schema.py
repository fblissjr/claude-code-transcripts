"""Simple 4-table schema DDL for Claude Code transcripts.

This module defines the simple schema with 4 tables:
- sessions: Session metadata
- messages: All messages (user and assistant)
- tool_calls: Tool invocations and results
- thinking: Extended thinking blocks (optional)
"""

import duckdb


def create_duckdb_schema(db_path):
    """Create DuckDB database with schema for transcript data.

    Args:
        db_path: Path to the DuckDB database file

    Returns:
        duckdb.Connection to the database
    """
    conn = duckdb.connect(str(db_path))

    # Sessions table
    conn.execute(
        """
        CREATE OR REPLACE TABLE sessions (
            session_id VARCHAR,
            project_path VARCHAR,
            project_name VARCHAR,
            first_timestamp TIMESTAMP,
            last_timestamp TIMESTAMP,
            message_count INTEGER,
            user_message_count INTEGER,
            assistant_message_count INTEGER,
            tool_use_count INTEGER,
            cwd VARCHAR,
            git_branch VARCHAR,
            version VARCHAR,
            is_agent BOOLEAN DEFAULT FALSE,
            agent_id VARCHAR,
            parent_session_id VARCHAR,
            depth_level INTEGER DEFAULT 0
        )
    """
    )

    # Messages table
    conn.execute(
        """
        CREATE OR REPLACE TABLE messages (
            id VARCHAR,
            session_id VARCHAR,
            parent_id VARCHAR,
            type VARCHAR,
            timestamp TIMESTAMP,
            model VARCHAR,
            content TEXT,
            content_json JSON,
            has_tool_use BOOLEAN,
            has_tool_result BOOLEAN,
            has_thinking BOOLEAN,
            is_sidechain BOOLEAN DEFAULT FALSE
        )
    """
    )

    # Tool calls table
    conn.execute(
        """
        CREATE OR REPLACE TABLE tool_calls (
            tool_use_id VARCHAR,
            session_id VARCHAR,
            message_id VARCHAR,
            result_message_id VARCHAR,
            tool_name VARCHAR,
            input_json JSON,
            input_summary TEXT,
            output_text TEXT,
            timestamp TIMESTAMP
        )
    """
    )

    # Thinking table
    conn.execute(
        """
        CREATE OR REPLACE TABLE thinking (
            id INTEGER,
            session_id VARCHAR,
            message_id VARCHAR,
            thinking_text TEXT,
            timestamp TIMESTAMP
        )
    """
    )

    return conn
