"""Star schema DDL - creates the dimensional model tables."""

import duckdb

from .utils import generate_dimension_key


def create_star_schema(db_path):
    """Create DuckDB database with star schema for transcript analytics.

    This creates a dimensional model with:
    - Staging table for raw data
    - Dimension tables with hash-based surrogate keys
    - Fact tables for messages, tool calls, content blocks, and session summaries

    No hard PK/FK constraints are used - relies on soft business rules.

    Args:
        db_path: Path to the DuckDB database file

    Returns:
        duckdb.Connection to the database
    """
    conn = duckdb.connect(str(db_path))

    # =========================================================================
    # Staging Table
    # =========================================================================
    conn.execute(
        """
        CREATE OR REPLACE TABLE stg_raw_messages (
            session_id VARCHAR,
            project_name VARCHAR,
            project_path VARCHAR,
            message_id VARCHAR,
            parent_id VARCHAR,
            message_type VARCHAR,
            timestamp TIMESTAMP,
            model VARCHAR,
            cwd VARCHAR,
            git_branch VARCHAR,
            version VARCHAR,
            content_json JSON,
            content_text TEXT
        )
    """
    )

    # =========================================================================
    # Core Dimension Tables
    # =========================================================================

    conn.execute(
        """
        CREATE OR REPLACE TABLE dim_tool (
            tool_key VARCHAR,
            tool_name VARCHAR,
            tool_category VARCHAR
        )
    """
    )

    conn.execute(
        """
        CREATE OR REPLACE TABLE dim_model (
            model_key VARCHAR,
            model_name VARCHAR,
            model_family VARCHAR
        )
    """
    )

    conn.execute(
        """
        CREATE OR REPLACE TABLE dim_project (
            project_key VARCHAR,
            project_path VARCHAR,
            project_name VARCHAR
        )
    """
    )

    conn.execute(
        """
        CREATE OR REPLACE TABLE dim_session (
            session_key VARCHAR,
            session_id VARCHAR,
            project_key VARCHAR,
            cwd VARCHAR,
            git_branch VARCHAR,
            version VARCHAR,
            first_timestamp TIMESTAMP,
            last_timestamp TIMESTAMP,
            is_agent BOOLEAN DEFAULT FALSE,
            agent_id VARCHAR,
            parent_session_key VARCHAR,
            depth_level INTEGER DEFAULT 0
        )
    """
    )

    conn.execute(
        """
        CREATE OR REPLACE TABLE dim_date (
            date_key INTEGER,
            full_date DATE,
            year INTEGER,
            month INTEGER,
            day INTEGER,
            day_of_week INTEGER,
            day_name VARCHAR,
            month_name VARCHAR,
            quarter INTEGER,
            is_weekend BOOLEAN
        )
    """
    )

    conn.execute(
        """
        CREATE OR REPLACE TABLE dim_time (
            time_key INTEGER,
            hour INTEGER,
            minute INTEGER,
            time_of_day VARCHAR
        )
    """
    )

    conn.execute(
        """
        CREATE OR REPLACE TABLE dim_message_type (
            message_type_key VARCHAR,
            message_type VARCHAR
        )
    """
    )

    conn.execute(
        """
        CREATE OR REPLACE TABLE dim_content_block_type (
            content_block_type_key VARCHAR,
            block_type VARCHAR
        )
    """
    )

    # =========================================================================
    # Core Fact Tables
    # =========================================================================

    conn.execute(
        """
        CREATE OR REPLACE TABLE fact_messages (
            message_id VARCHAR,
            session_key VARCHAR,
            project_key VARCHAR,
            message_type_key VARCHAR,
            model_key VARCHAR,
            date_key INTEGER,
            time_key INTEGER,
            parent_message_id VARCHAR,
            timestamp TIMESTAMP,
            content_length INTEGER,
            content_block_count INTEGER,
            has_tool_use BOOLEAN,
            has_tool_result BOOLEAN,
            has_thinking BOOLEAN,
            word_count INTEGER,
            estimated_tokens INTEGER,
            response_time_seconds FLOAT,
            conversation_depth INTEGER,
            content_text TEXT,
            content_json JSON,
            is_sidechain BOOLEAN DEFAULT FALSE
        )
    """
    )

    conn.execute(
        """
        CREATE OR REPLACE TABLE fact_content_blocks (
            content_block_id VARCHAR,
            message_id VARCHAR,
            session_key VARCHAR,
            content_block_type_key VARCHAR,
            date_key INTEGER,
            time_key INTEGER,
            block_index INTEGER,
            content_length INTEGER,
            content_text TEXT,
            content_json JSON
        )
    """
    )

    conn.execute(
        """
        CREATE OR REPLACE TABLE fact_tool_calls (
            tool_call_id VARCHAR,
            session_key VARCHAR,
            tool_key VARCHAR,
            date_key INTEGER,
            time_key INTEGER,
            invoke_message_id VARCHAR,
            result_message_id VARCHAR,
            timestamp TIMESTAMP,
            input_char_count INTEGER,
            output_char_count INTEGER,
            is_error BOOLEAN,
            input_json JSON,
            input_summary TEXT,
            output_text TEXT
        )
    """
    )

    conn.execute(
        """
        CREATE OR REPLACE TABLE fact_session_summary (
            session_key VARCHAR,
            project_key VARCHAR,
            date_key INTEGER,
            total_messages INTEGER,
            user_messages INTEGER,
            assistant_messages INTEGER,
            total_tool_calls INTEGER,
            total_thinking_blocks INTEGER,
            total_content_blocks INTEGER,
            session_duration_seconds INTEGER,
            first_timestamp TIMESTAMP,
            last_timestamp TIMESTAMP
        )
    """
    )

    # =========================================================================
    # Granular Dimensions
    # =========================================================================

    conn.execute(
        """
        CREATE OR REPLACE TABLE dim_file (
            file_key VARCHAR,
            file_path VARCHAR,
            file_name VARCHAR,
            file_extension VARCHAR,
            directory_path VARCHAR
        )
    """
    )

    conn.execute(
        """
        CREATE OR REPLACE TABLE dim_programming_language (
            language_key VARCHAR,
            language_name VARCHAR,
            file_extensions VARCHAR
        )
    """
    )

    conn.execute(
        """
        CREATE OR REPLACE TABLE dim_error_type (
            error_type_key VARCHAR,
            error_type VARCHAR,
            error_category VARCHAR
        )
    """
    )

    # =========================================================================
    # Granular Fact Tables
    # =========================================================================

    conn.execute(
        """
        CREATE OR REPLACE TABLE fact_file_operations (
            file_operation_id VARCHAR,
            tool_call_id VARCHAR,
            session_key VARCHAR,
            file_key VARCHAR,
            tool_key VARCHAR,
            date_key INTEGER,
            time_key INTEGER,
            operation_type VARCHAR,
            file_size_chars INTEGER,
            timestamp TIMESTAMP
        )
    """
    )

    conn.execute(
        """
        CREATE OR REPLACE TABLE fact_code_blocks (
            code_block_id VARCHAR,
            message_id VARCHAR,
            session_key VARCHAR,
            language_key VARCHAR,
            date_key INTEGER,
            time_key INTEGER,
            block_index INTEGER,
            line_count INTEGER,
            char_count INTEGER,
            code_text TEXT
        )
    """
    )

    conn.execute(
        """
        CREATE OR REPLACE TABLE fact_errors (
            error_id VARCHAR,
            tool_call_id VARCHAR,
            session_key VARCHAR,
            tool_key VARCHAR,
            error_type_key VARCHAR,
            date_key INTEGER,
            time_key INTEGER,
            error_message TEXT,
            timestamp TIMESTAMP
        )
    """
    )

    # =========================================================================
    # Entity Extraction Tables
    # =========================================================================

    conn.execute(
        """
        CREATE OR REPLACE TABLE dim_entity_type (
            entity_type_key VARCHAR,
            entity_type VARCHAR,
            extraction_method VARCHAR
        )
    """
    )

    conn.execute(
        """
        CREATE OR REPLACE TABLE fact_entity_mentions (
            mention_id VARCHAR,
            message_id VARCHAR,
            session_key VARCHAR,
            entity_type_key VARCHAR,
            entity_text VARCHAR,
            entity_normalized VARCHAR,
            context_snippet TEXT,
            position_start INTEGER,
            position_end INTEGER
        )
    """
    )

    # =========================================================================
    # Tool Chain Tracking
    # =========================================================================

    conn.execute(
        """
        CREATE OR REPLACE TABLE fact_tool_chain_steps (
            chain_step_id VARCHAR,
            session_key VARCHAR,
            chain_id VARCHAR,
            tool_call_id VARCHAR,
            tool_key VARCHAR,
            step_position INTEGER,
            prev_tool_key VARCHAR,
            time_since_prev_seconds FLOAT
        )
    """
    )

    # =========================================================================
    # LLM Enrichment Tables
    # =========================================================================

    conn.execute(
        """
        CREATE OR REPLACE TABLE dim_intent (
            intent_key VARCHAR,
            intent_name VARCHAR,
            intent_category VARCHAR,
            description TEXT
        )
    """
    )

    conn.execute(
        """
        CREATE OR REPLACE TABLE dim_topic (
            topic_key VARCHAR,
            topic_name VARCHAR,
            topic_category VARCHAR
        )
    """
    )

    conn.execute(
        """
        CREATE OR REPLACE TABLE dim_sentiment (
            sentiment_key VARCHAR,
            sentiment_name VARCHAR,
            valence FLOAT
        )
    """
    )

    conn.execute(
        """
        CREATE OR REPLACE TABLE fact_message_enrichment (
            enrichment_id VARCHAR,
            message_id VARCHAR,
            session_key VARCHAR,
            intent_key VARCHAR,
            sentiment_key VARCHAR,
            complexity_score FLOAT,
            confidence_score FLOAT,
            enrichment_model VARCHAR,
            enriched_at TIMESTAMP
        )
    """
    )

    conn.execute(
        """
        CREATE OR REPLACE TABLE fact_message_topics (
            message_topic_id VARCHAR,
            message_id VARCHAR,
            topic_key VARCHAR,
            relevance_score FLOAT
        )
    """
    )

    conn.execute(
        """
        CREATE OR REPLACE TABLE fact_session_insights (
            insight_id VARCHAR,
            session_key VARCHAR,
            summary_text TEXT,
            key_decisions TEXT,
            outcome_status VARCHAR,
            task_completed BOOLEAN,
            primary_intent_key VARCHAR,
            complexity_score FLOAT,
            enrichment_model VARCHAR,
            enriched_at TIMESTAMP
        )
    """
    )

    # =========================================================================
    # Pre-populate Reference Data
    # =========================================================================

    _populate_reference_data(conn)

    return conn


def _populate_reference_data(conn):
    """Pre-populate dimension tables with reference data."""

    # Entity types
    entity_types = [
        ("file_path", "regex"),
        ("url", "regex"),
        ("function_name", "regex"),
        ("class_name", "regex"),
        ("variable_name", "regex"),
        ("package_name", "regex"),
        ("error_code", "regex"),
        ("git_ref", "regex"),
    ]
    for entity_type, method in entity_types:
        key = generate_dimension_key(entity_type)
        conn.execute(
            "INSERT INTO dim_entity_type VALUES (?, ?, ?)",
            [key, entity_type, method],
        )

    # Intents
    intents = [
        ("bug_fix", "problem_solving", "Fix a bug or error"),
        ("feature", "development", "Add new functionality"),
        ("refactor", "development", "Improve code structure"),
        ("question", "inquiry", "Ask about code or concepts"),
        ("explain", "inquiry", "Request explanation"),
        ("review", "analysis", "Review or analyze code"),
        ("test", "quality", "Write or run tests"),
        ("debug", "problem_solving", "Debug an issue"),
        ("config", "setup", "Configuration or setup"),
        ("docs", "documentation", "Documentation work"),
    ]
    for intent_name, category, desc in intents:
        key = generate_dimension_key(intent_name)
        conn.execute(
            "INSERT INTO dim_intent VALUES (?, ?, ?, ?)",
            [key, intent_name, category, desc],
        )

    # Sentiments
    sentiments = [
        ("neutral", 0.0),
        ("positive", 0.5),
        ("negative", -0.5),
        ("frustrated", -0.8),
        ("satisfied", 0.8),
        ("confused", -0.3),
        ("curious", 0.3),
    ]
    for sentiment_name, valence in sentiments:
        key = generate_dimension_key(sentiment_name)
        conn.execute(
            "INSERT INTO dim_sentiment VALUES (?, ?, ?)",
            [key, sentiment_name, valence],
        )

    # Topics
    topics = [
        ("frontend", "domain"),
        ("backend", "domain"),
        ("database", "domain"),
        ("api", "domain"),
        ("auth", "domain"),
        ("testing", "practice"),
        ("deployment", "practice"),
        ("security", "concern"),
        ("performance", "concern"),
        ("architecture", "design"),
    ]
    for topic_name, category in topics:
        key = generate_dimension_key(topic_name)
        conn.execute(
            "INSERT INTO dim_topic VALUES (?, ?, ?)",
            [key, topic_name, category],
        )

    # Message types
    for msg_type in ["user", "assistant"]:
        key = generate_dimension_key(msg_type)
        conn.execute(
            "INSERT INTO dim_message_type VALUES (?, ?)",
            [key, msg_type],
        )

    # Content block types
    for block_type in ["text", "tool_use", "tool_result", "thinking", "image"]:
        key = generate_dimension_key(block_type)
        conn.execute(
            "INSERT INTO dim_content_block_type VALUES (?, ?)",
            [key, block_type],
        )

    # =========================================================================
    # Semantic Views - Pre-joined views for easy exploration
    # =========================================================================

    # Main semantic view: Sessions with all key metrics
    conn.execute(
        """
        CREATE OR REPLACE VIEW semantic_sessions AS
        SELECT
            -- Session info
            ds.session_id,
            ds.cwd,
            ds.git_branch,
            ds.version,
            ds.first_timestamp,
            ds.last_timestamp,
            -- Project info
            dp.project_name,
            dp.project_path,
            -- Metrics
            fss.total_messages,
            fss.user_messages,
            fss.assistant_messages,
            fss.total_tool_calls,
            fss.total_thinking_blocks,
            fss.session_duration_seconds,
            -- Date info
            dd.full_date,
            dd.day_name,
            dd.month_name,
            dd.year,
            dd.is_weekend
        FROM fact_session_summary fss
        JOIN dim_session ds ON fss.session_key = ds.session_key
        JOIN dim_project dp ON fss.project_key = dp.project_key
        LEFT JOIN dim_date dd ON fss.date_key = dd.date_key
    """
    )

    # Messages with full context
    conn.execute(
        """
        CREATE OR REPLACE VIEW semantic_messages AS
        SELECT
            -- Message info
            fm.message_id,
            fm.timestamp,
            fm.content_text,
            fm.content_length,
            fm.word_count,
            fm.estimated_tokens,
            fm.has_tool_use,
            fm.has_thinking,
            fm.response_time_seconds,
            fm.conversation_depth,
            -- Type
            dmt.message_type,
            -- Model
            dm.model_name,
            dm.model_family,
            -- Session
            ds.session_id,
            ds.cwd,
            -- Project
            dp.project_name,
            -- Date/Time
            dd.full_date,
            dd.day_name,
            dt.hour,
            dt.time_of_day
        FROM fact_messages fm
        JOIN dim_message_type dmt ON fm.message_type_key = dmt.message_type_key
        LEFT JOIN dim_model dm ON fm.model_key = dm.model_key
        JOIN dim_session ds ON fm.session_key = ds.session_key
        LEFT JOIN dim_project dp ON fm.project_key = dp.project_key
        LEFT JOIN dim_date dd ON fm.date_key = dd.date_key
        LEFT JOIN dim_time dt ON fm.time_key = dt.time_key
    """
    )

    # Tool calls with full context
    conn.execute(
        """
        CREATE OR REPLACE VIEW semantic_tool_calls AS
        SELECT
            -- Tool call info
            ftc.tool_call_id,
            ftc.timestamp,
            ftc.input_char_count,
            ftc.output_char_count,
            ftc.is_error,
            ftc.input_summary,
            ftc.output_text,
            -- Tool
            dt.tool_name,
            dt.tool_category,
            -- Session
            ds.session_id,
            ds.cwd,
            -- Project
            dp.project_name,
            -- Date/Time
            dd.full_date,
            dti.hour,
            dti.time_of_day
        FROM fact_tool_calls ftc
        JOIN dim_tool dt ON ftc.tool_key = dt.tool_key
        JOIN dim_session ds ON ftc.session_key = ds.session_key
        LEFT JOIN dim_project dp ON ds.project_key = dp.project_key
        LEFT JOIN dim_date dd ON ftc.date_key = dd.date_key
        LEFT JOIN dim_time dti ON ftc.time_key = dti.time_key
    """
    )

    # File operations with full context
    conn.execute(
        """
        CREATE OR REPLACE VIEW semantic_file_operations AS
        SELECT
            -- Operation info
            ffo.operation_type,
            ffo.file_size_chars,
            ffo.timestamp,
            -- File
            df.file_path,
            df.file_name,
            df.file_extension,
            df.directory_path,
            -- Tool
            dt.tool_name,
            dt.tool_category,
            -- Session
            ds.session_id,
            -- Project
            dp.project_name
        FROM fact_file_operations ffo
        JOIN dim_file df ON ffo.file_key = df.file_key
        JOIN dim_tool dt ON ffo.tool_key = dt.tool_key
        JOIN dim_session ds ON ffo.session_key = ds.session_key
        LEFT JOIN dim_project dp ON ds.project_key = dp.project_key
    """
    )
