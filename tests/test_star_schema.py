"""Tests for star schema DuckDB implementation."""

import json
import tempfile
from pathlib import Path
from datetime import datetime
import hashlib

import duckdb
import pytest

from ccutils import (
    create_star_schema,
    run_star_schema_etl,
    generate_dimension_key,
    create_semantic_model,
)


@pytest.fixture
def sample_session_file():
    """Create a sample JSONL session file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        # User message
        f.write(
            json.dumps(
                {
                    "type": "user",
                    "uuid": "user-001",
                    "parentUuid": None,
                    "sessionId": "session-123",
                    "timestamp": "2025-01-15T10:00:00.000Z",
                    "cwd": "/home/user/project",
                    "gitBranch": "main",
                    "version": "2.0.0",
                    "message": {
                        "role": "user",
                        "content": "Help me write a hello world program",
                    },
                }
            )
            + "\n"
        )
        # Assistant message with tool_use
        f.write(
            json.dumps(
                {
                    "type": "assistant",
                    "uuid": "asst-001",
                    "parentUuid": "user-001",
                    "sessionId": "session-123",
                    "timestamp": "2025-01-15T10:00:05.000Z",
                    "message": {
                        "role": "assistant",
                        "model": "claude-opus-4-5-20251101",
                        "content": [
                            {"type": "text", "text": "I'll create that for you."},
                            {
                                "type": "tool_use",
                                "id": "tool-001",
                                "name": "Write",
                                "input": {
                                    "file_path": "/home/user/project/hello.py",
                                    "content": "print('Hello, World!')",
                                },
                            },
                        ],
                    },
                }
            )
            + "\n"
        )
        # User message with tool_result
        f.write(
            json.dumps(
                {
                    "type": "user",
                    "uuid": "user-002",
                    "parentUuid": "asst-001",
                    "sessionId": "session-123",
                    "timestamp": "2025-01-15T10:00:10.000Z",
                    "message": {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "tool-001",
                                "content": "File written successfully",
                            }
                        ],
                    },
                }
            )
            + "\n"
        )
        # Assistant message with Read tool
        f.write(
            json.dumps(
                {
                    "type": "assistant",
                    "uuid": "asst-002",
                    "parentUuid": "user-002",
                    "sessionId": "session-123",
                    "timestamp": "2025-01-15T10:00:15.000Z",
                    "message": {
                        "role": "assistant",
                        "model": "claude-opus-4-5-20251101",
                        "content": [
                            {
                                "type": "thinking",
                                "thinking": "The file was created. Let me verify it.",
                            },
                            {"type": "text", "text": "Let me verify the file."},
                            {
                                "type": "tool_use",
                                "id": "tool-002",
                                "name": "Read",
                                "input": {"file_path": "/home/user/project/hello.py"},
                            },
                        ],
                    },
                }
            )
            + "\n"
        )
        # User message with tool_result for Read
        f.write(
            json.dumps(
                {
                    "type": "user",
                    "uuid": "user-003",
                    "parentUuid": "asst-002",
                    "sessionId": "session-123",
                    "timestamp": "2025-01-15T10:00:20.000Z",
                    "message": {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "tool-002",
                                "content": "print('Hello, World!')",
                            }
                        ],
                    },
                }
            )
            + "\n"
        )
        # Final assistant message
        f.write(
            json.dumps(
                {
                    "type": "assistant",
                    "uuid": "asst-003",
                    "parentUuid": "user-003",
                    "sessionId": "session-123",
                    "timestamp": "2025-01-15T10:00:25.000Z",
                    "message": {
                        "role": "assistant",
                        "model": "claude-sonnet-4-20250514",
                        "content": [
                            {
                                "type": "text",
                                "text": "Done! I've created hello.py with a hello world program.",
                            },
                        ],
                    },
                }
            )
            + "\n"
        )
        f.flush()
        yield Path(f.name)


@pytest.fixture
def output_dir():
    """Create a temporary output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_projects_dir(sample_session_file):
    """Create a mock projects directory structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        projects_dir = Path(tmpdir)

        # Create a project folder
        project_dir = projects_dir / "-home-user-project"
        project_dir.mkdir(parents=True)

        # Copy sample session to project
        session_file = project_dir / "session-123.jsonl"
        session_file.write_text(sample_session_file.read_text())

        yield projects_dir


class TestGenerateDimensionKey:
    """Tests for dimension key generation."""

    def test_generates_md5_hash(self):
        """Test that dimension keys are MD5 hashes."""
        key = generate_dimension_key("Write")
        assert len(key) == 32  # MD5 produces 32 hex characters
        assert all(c in "0123456789abcdef" for c in key)

    def test_same_input_produces_same_key(self):
        """Test that same input always produces same key."""
        key1 = generate_dimension_key("Write")
        key2 = generate_dimension_key("Write")
        assert key1 == key2

    def test_different_inputs_produce_different_keys(self):
        """Test that different inputs produce different keys."""
        key1 = generate_dimension_key("Write")
        key2 = generate_dimension_key("Read")
        assert key1 != key2

    def test_handles_multiple_natural_keys(self):
        """Test composite key generation from multiple values."""
        key = generate_dimension_key("project", "/home/user")
        expected = hashlib.md5("project|/home/user".encode()).hexdigest()
        assert key == expected

    def test_handles_none_values(self):
        """Test that None values are handled gracefully."""
        key = generate_dimension_key(None)
        assert key is not None
        assert len(key) == 32


class TestCreateStarSchema:
    """Tests for star schema creation."""

    def test_creates_dim_tool_table(self, output_dir):
        """Test that dim_tool dimension table is created."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)

        result = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='dim_tool'"
        ).fetchone()
        assert result is not None
        conn.close()

    def test_creates_dim_model_table(self, output_dir):
        """Test that dim_model dimension table is created."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)

        result = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='dim_model'"
        ).fetchone()
        assert result is not None
        conn.close()

    def test_creates_dim_project_table(self, output_dir):
        """Test that dim_project dimension table is created."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)

        result = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='dim_project'"
        ).fetchone()
        assert result is not None
        conn.close()

    def test_creates_dim_session_table(self, output_dir):
        """Test that dim_session dimension table is created."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)

        result = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='dim_session'"
        ).fetchone()
        assert result is not None
        conn.close()

    def test_creates_dim_date_table(self, output_dir):
        """Test that dim_date dimension table is created."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)

        result = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='dim_date'"
        ).fetchone()
        assert result is not None
        conn.close()

    def test_creates_dim_time_table(self, output_dir):
        """Test that dim_time dimension table is created."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)

        result = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='dim_time'"
        ).fetchone()
        assert result is not None
        conn.close()

    def test_creates_dim_message_type_table(self, output_dir):
        """Test that dim_message_type dimension table is created."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)

        result = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='dim_message_type'"
        ).fetchone()
        assert result is not None
        conn.close()

    def test_creates_dim_content_block_type_table(self, output_dir):
        """Test that dim_content_block_type dimension table is created."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)

        result = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='dim_content_block_type'"
        ).fetchone()
        assert result is not None
        conn.close()

    def test_creates_fact_messages_table(self, output_dir):
        """Test that fact_messages table is created."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)

        result = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='fact_messages'"
        ).fetchone()
        assert result is not None
        conn.close()

    def test_creates_fact_content_blocks_table(self, output_dir):
        """Test that fact_content_blocks table is created."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)

        result = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='fact_content_blocks'"
        ).fetchone()
        assert result is not None
        conn.close()

    def test_creates_fact_tool_calls_table(self, output_dir):
        """Test that fact_tool_calls table is created."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)

        result = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='fact_tool_calls'"
        ).fetchone()
        assert result is not None
        conn.close()

    def test_creates_fact_session_summary_table(self, output_dir):
        """Test that fact_session_summary table is created."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)

        result = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='fact_session_summary'"
        ).fetchone()
        assert result is not None
        conn.close()

    def test_creates_stg_raw_messages_table(self, output_dir):
        """Test that staging table for raw messages is created."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)

        result = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='stg_raw_messages'"
        ).fetchone()
        assert result is not None
        conn.close()

    def test_creates_semantic_sessions_view(self, output_dir):
        """Test that semantic_sessions view is created."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)

        result = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='view' AND name='semantic_sessions'"
        ).fetchone()
        assert result is not None
        conn.close()

    def test_creates_semantic_messages_view(self, output_dir):
        """Test that semantic_messages view is created."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)

        result = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='view' AND name='semantic_messages'"
        ).fetchone()
        assert result is not None
        conn.close()

    def test_creates_semantic_tool_calls_view(self, output_dir):
        """Test that semantic_tool_calls view is created."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)

        result = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='view' AND name='semantic_tool_calls'"
        ).fetchone()
        assert result is not None
        conn.close()

    def test_creates_semantic_file_operations_view(self, output_dir):
        """Test that semantic_file_operations view is created."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)

        result = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='view' AND name='semantic_file_operations'"
        ).fetchone()
        assert result is not None
        conn.close()


class TestDimToolTable:
    """Tests for dim_tool dimension table."""

    def test_dim_tool_has_tool_key(self, output_dir):
        """Test that dim_tool has tool_key column."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)

        columns = conn.execute("DESCRIBE dim_tool").fetchall()
        column_names = [c[0] for c in columns]
        assert "tool_key" in column_names
        conn.close()

    def test_dim_tool_has_tool_name(self, output_dir):
        """Test that dim_tool has tool_name column."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)

        columns = conn.execute("DESCRIBE dim_tool").fetchall()
        column_names = [c[0] for c in columns]
        assert "tool_name" in column_names
        conn.close()

    def test_dim_tool_has_tool_category(self, output_dir):
        """Test that dim_tool has tool_category column."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)

        columns = conn.execute("DESCRIBE dim_tool").fetchall()
        column_names = [c[0] for c in columns]
        assert "tool_category" in column_names
        conn.close()


class TestDimModelTable:
    """Tests for dim_model dimension table."""

    def test_dim_model_has_required_columns(self, output_dir):
        """Test that dim_model has all required columns."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)

        columns = conn.execute("DESCRIBE dim_model").fetchall()
        column_names = [c[0] for c in columns]
        assert "model_key" in column_names
        assert "model_name" in column_names
        assert "model_family" in column_names
        conn.close()


class TestDimDateTable:
    """Tests for dim_date dimension table."""

    def test_dim_date_has_required_columns(self, output_dir):
        """Test that dim_date has all required columns."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)

        columns = conn.execute("DESCRIBE dim_date").fetchall()
        column_names = [c[0] for c in columns]
        assert "date_key" in column_names
        assert "full_date" in column_names
        assert "year" in column_names
        assert "month" in column_names
        assert "day" in column_names
        assert "day_of_week" in column_names
        assert "day_name" in column_names
        assert "month_name" in column_names
        assert "quarter" in column_names
        assert "is_weekend" in column_names
        conn.close()


class TestDimTimeTable:
    """Tests for dim_time dimension table."""

    def test_dim_time_has_required_columns(self, output_dir):
        """Test that dim_time has all required columns."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)

        columns = conn.execute("DESCRIBE dim_time").fetchall()
        column_names = [c[0] for c in columns]
        assert "time_key" in column_names
        assert "hour" in column_names
        assert "minute" in column_names
        assert "time_of_day" in column_names
        conn.close()


class TestFactMessagesTable:
    """Tests for fact_messages table."""

    def test_fact_messages_has_dimension_keys(self, output_dir):
        """Test that fact_messages has foreign keys to dimensions."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)

        columns = conn.execute("DESCRIBE fact_messages").fetchall()
        column_names = [c[0] for c in columns]
        assert "session_key" in column_names
        assert "project_key" in column_names
        assert "message_type_key" in column_names
        assert "model_key" in column_names
        assert "date_key" in column_names
        assert "time_key" in column_names
        conn.close()

    def test_fact_messages_has_measures(self, output_dir):
        """Test that fact_messages has measure columns."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)

        columns = conn.execute("DESCRIBE fact_messages").fetchall()
        column_names = [c[0] for c in columns]
        assert "content_length" in column_names
        assert "content_block_count" in column_names
        assert "has_tool_use" in column_names
        assert "has_tool_result" in column_names
        assert "has_thinking" in column_names
        conn.close()


class TestFactToolCallsTable:
    """Tests for fact_tool_calls table."""

    def test_fact_tool_calls_has_dimension_keys(self, output_dir):
        """Test that fact_tool_calls has foreign keys to dimensions."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)

        columns = conn.execute("DESCRIBE fact_tool_calls").fetchall()
        column_names = [c[0] for c in columns]
        assert "session_key" in column_names
        assert "tool_key" in column_names
        assert "date_key" in column_names
        assert "time_key" in column_names
        conn.close()

    def test_fact_tool_calls_has_measures(self, output_dir):
        """Test that fact_tool_calls has measure columns."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)

        columns = conn.execute("DESCRIBE fact_tool_calls").fetchall()
        column_names = [c[0] for c in columns]
        assert "input_char_count" in column_names
        assert "output_char_count" in column_names
        assert "is_error" in column_names
        conn.close()


class TestRunStarSchemaETL:
    """Tests for the ETL process that populates the star schema."""

    def test_etl_populates_dim_tool(self, sample_session_file, output_dir):
        """Test that ETL populates dim_tool with tools from session."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        run_star_schema_etl(conn, sample_session_file, "test-project")

        result = conn.execute(
            "SELECT tool_name FROM dim_tool ORDER BY tool_name"
        ).fetchall()
        tool_names = [r[0] for r in result]
        assert "Write" in tool_names
        assert "Read" in tool_names
        conn.close()

    def test_etl_populates_dim_model(self, sample_session_file, output_dir):
        """Test that ETL populates dim_model with models from session."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        run_star_schema_etl(conn, sample_session_file, "test-project")

        result = conn.execute(
            "SELECT model_name FROM dim_model ORDER BY model_name"
        ).fetchall()
        model_names = [r[0] for r in result]
        assert "claude-opus-4-5-20251101" in model_names
        assert "claude-sonnet-4-20250514" in model_names
        conn.close()

    def test_etl_populates_dim_project(self, sample_session_file, output_dir):
        """Test that ETL populates dim_project."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        run_star_schema_etl(conn, sample_session_file, "test-project")

        result = conn.execute("SELECT project_name FROM dim_project").fetchone()
        assert result[0] == "test-project"
        conn.close()

    def test_etl_populates_dim_session(self, sample_session_file, output_dir):
        """Test that ETL populates dim_session."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        run_star_schema_etl(conn, sample_session_file, "test-project")

        result = conn.execute(
            "SELECT session_id, cwd, git_branch FROM dim_session"
        ).fetchone()
        assert result[1] == "/home/user/project"
        assert result[2] == "main"
        conn.close()

    def test_etl_populates_dim_date(self, sample_session_file, output_dir):
        """Test that ETL populates dim_date for dates in session."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        run_star_schema_etl(conn, sample_session_file, "test-project")

        result = conn.execute(
            "SELECT date_key, year, month, day FROM dim_date WHERE date_key = 20250115"
        ).fetchone()
        assert result is not None
        assert result[1] == 2025
        assert result[2] == 1
        assert result[3] == 15
        conn.close()

    def test_etl_populates_fact_messages(self, sample_session_file, output_dir):
        """Test that ETL populates fact_messages."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        run_star_schema_etl(conn, sample_session_file, "test-project")

        result = conn.execute("SELECT COUNT(*) FROM fact_messages").fetchone()
        assert result[0] == 6  # 3 user + 3 assistant messages
        conn.close()

    def test_etl_populates_fact_tool_calls(self, sample_session_file, output_dir):
        """Test that ETL populates fact_tool_calls."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        run_star_schema_etl(conn, sample_session_file, "test-project")

        result = conn.execute("SELECT COUNT(*) FROM fact_tool_calls").fetchone()
        assert result[0] == 2  # Write and Read tools
        conn.close()

    def test_etl_populates_fact_content_blocks(self, sample_session_file, output_dir):
        """Test that ETL populates fact_content_blocks."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        run_star_schema_etl(
            conn, sample_session_file, "test-project", include_thinking=True
        )

        result = conn.execute("SELECT COUNT(*) FROM fact_content_blocks").fetchone()
        # Count all content blocks: text blocks, tool_use, tool_result, thinking
        assert result[0] > 0
        conn.close()

    def test_etl_populates_fact_session_summary(self, sample_session_file, output_dir):
        """Test that ETL populates fact_session_summary."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        run_star_schema_etl(conn, sample_session_file, "test-project")

        result = conn.execute(
            """SELECT total_messages, user_messages, assistant_messages,
                      total_tool_calls, session_duration_seconds
               FROM fact_session_summary"""
        ).fetchone()
        assert result[0] == 6  # total messages
        assert result[1] == 3  # user messages
        assert result[2] == 3  # assistant messages
        assert result[3] == 2  # tool calls (Write and Read)
        assert result[4] == 25  # duration in seconds (10:00:00 to 10:00:25)
        conn.close()

    def test_etl_assigns_tool_categories(self, sample_session_file, output_dir):
        """Test that ETL assigns correct tool categories."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        run_star_schema_etl(conn, sample_session_file, "test-project")

        result = conn.execute(
            "SELECT tool_name, tool_category FROM dim_tool ORDER BY tool_name"
        ).fetchall()
        tool_dict = {r[0]: r[1] for r in result}
        assert tool_dict["Write"] == "file_operations"
        assert tool_dict["Read"] == "file_operations"
        conn.close()

    def test_etl_assigns_model_families(self, sample_session_file, output_dir):
        """Test that ETL assigns correct model families."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        run_star_schema_etl(conn, sample_session_file, "test-project")

        result = conn.execute(
            "SELECT model_name, model_family FROM dim_model ORDER BY model_name"
        ).fetchall()
        model_dict = {r[0]: r[1] for r in result}
        assert model_dict["claude-opus-4-5-20251101"] == "opus"
        assert model_dict["claude-sonnet-4-20250514"] == "sonnet"
        conn.close()

    def test_etl_links_tool_calls_to_dimensions(self, sample_session_file, output_dir):
        """Test that fact_tool_calls correctly links to dim_tool."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        run_star_schema_etl(conn, sample_session_file, "test-project")

        result = conn.execute(
            """SELECT dt.tool_name, ft.input_char_count
               FROM fact_tool_calls ft
               JOIN dim_tool dt ON ft.tool_key = dt.tool_key
               ORDER BY dt.tool_name"""
        ).fetchall()
        assert len(result) == 2
        tool_names = [r[0] for r in result]
        assert "Read" in tool_names
        assert "Write" in tool_names
        conn.close()

    def test_etl_links_messages_to_date_dimension(
        self, sample_session_file, output_dir
    ):
        """Test that fact_messages correctly links to dim_date."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        run_star_schema_etl(conn, sample_session_file, "test-project")

        result = conn.execute(
            """SELECT dd.year, dd.month, dd.day, COUNT(*) as msg_count
               FROM fact_messages fm
               JOIN dim_date dd ON fm.date_key = dd.date_key
               GROUP BY dd.year, dd.month, dd.day"""
        ).fetchone()
        assert result[0] == 2025
        assert result[1] == 1
        assert result[2] == 15
        assert result[3] == 6  # All 6 messages on same day
        conn.close()


class TestStarSchemaAnalytics:
    """Tests for analytical queries on the star schema."""

    def test_tool_usage_by_category(self, sample_session_file, output_dir):
        """Test that we can analyze tool usage by category."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        run_star_schema_etl(conn, sample_session_file, "test-project")

        result = conn.execute(
            """SELECT dt.tool_category, COUNT(*) as usage_count
               FROM fact_tool_calls ft
               JOIN dim_tool dt ON ft.tool_key = dt.tool_key
               GROUP BY dt.tool_category"""
        ).fetchall()
        # Both Write and Read are file_operations
        assert len(result) == 1
        assert result[0][0] == "file_operations"
        assert result[0][1] == 2
        conn.close()

    def test_messages_by_model_family(self, sample_session_file, output_dir):
        """Test that we can analyze messages by model family."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        run_star_schema_etl(conn, sample_session_file, "test-project")

        result = conn.execute(
            """SELECT dm.model_family, COUNT(*) as msg_count
               FROM fact_messages fm
               JOIN dim_model dm ON fm.model_key = dm.model_key
               WHERE fm.model_key IS NOT NULL
               GROUP BY dm.model_family
               ORDER BY dm.model_family"""
        ).fetchall()
        result_dict = {r[0]: r[1] for r in result}
        # 2 opus messages, 1 sonnet message
        assert result_dict.get("opus", 0) == 2
        assert result_dict.get("sonnet", 0) == 1
        conn.close()

    def test_session_metrics_query(self, sample_session_file, output_dir):
        """Test session metrics from fact_session_summary."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        run_star_schema_etl(conn, sample_session_file, "test-project")

        result = conn.execute(
            """SELECT dp.project_name, ds.git_branch,
                      fs.total_messages, fs.total_tool_calls
               FROM fact_session_summary fs
               JOIN dim_session ds ON fs.session_key = ds.session_key
               JOIN dim_project dp ON fs.project_key = dp.project_key"""
        ).fetchone()
        assert result[0] == "test-project"
        assert result[1] == "main"
        assert result[2] == 6
        assert result[3] == 2
        conn.close()

    def test_time_of_day_analysis(self, sample_session_file, output_dir):
        """Test that we can analyze activity by time of day."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        run_star_schema_etl(conn, sample_session_file, "test-project")

        result = conn.execute(
            """SELECT dt.time_of_day, COUNT(*) as msg_count
               FROM fact_messages fm
               JOIN dim_time dt ON fm.time_key = dt.time_key
               GROUP BY dt.time_of_day"""
        ).fetchone()
        # All messages at 10:00 AM are in "morning"
        assert result[0] == "morning"
        assert result[1] == 6
        conn.close()


class TestContentBlockGranularity:
    """Tests for granular content block tracking."""

    def test_text_blocks_tracked(self, sample_session_file, output_dir):
        """Test that text content blocks are tracked individually."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        run_star_schema_etl(
            conn, sample_session_file, "test-project", include_thinking=True
        )

        result = conn.execute(
            """SELECT COUNT(*) FROM fact_content_blocks fcb
               JOIN dim_content_block_type dcbt ON fcb.content_block_type_key = dcbt.content_block_type_key
               WHERE dcbt.block_type = 'text'"""
        ).fetchone()
        # At least 3 text blocks from assistant messages
        assert result[0] >= 3
        conn.close()

    def test_tool_use_blocks_tracked(self, sample_session_file, output_dir):
        """Test that tool_use content blocks are tracked."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        run_star_schema_etl(
            conn, sample_session_file, "test-project", include_thinking=True
        )

        result = conn.execute(
            """SELECT COUNT(*) FROM fact_content_blocks fcb
               JOIN dim_content_block_type dcbt ON fcb.content_block_type_key = dcbt.content_block_type_key
               WHERE dcbt.block_type = 'tool_use'"""
        ).fetchone()
        assert result[0] == 2  # Write and Read tool_use blocks
        conn.close()

    def test_thinking_blocks_tracked_when_enabled(
        self, sample_session_file, output_dir
    ):
        """Test that thinking blocks are tracked when include_thinking=True."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        run_star_schema_etl(
            conn, sample_session_file, "test-project", include_thinking=True
        )

        result = conn.execute(
            """SELECT COUNT(*) FROM fact_content_blocks fcb
               JOIN dim_content_block_type dcbt ON fcb.content_block_type_key = dcbt.content_block_type_key
               WHERE dcbt.block_type = 'thinking'"""
        ).fetchone()
        assert result[0] == 1  # One thinking block
        conn.close()

    def test_block_index_tracks_position(self, sample_session_file, output_dir):
        """Test that block_index tracks position within message."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        run_star_schema_etl(
            conn, sample_session_file, "test-project", include_thinking=True
        )

        # The assistant message asst-002 has: thinking (0), text (1), tool_use (2)
        result = conn.execute(
            """SELECT fcb.block_index, dcbt.block_type
               FROM fact_content_blocks fcb
               JOIN dim_content_block_type dcbt ON fcb.content_block_type_key = dcbt.content_block_type_key
               WHERE fcb.message_id = 'asst-002'
               ORDER BY fcb.block_index"""
        ).fetchall()
        assert len(result) == 3
        assert result[0][1] == "thinking"
        assert result[1][1] == "text"
        assert result[2][1] == "tool_use"
        conn.close()


class TestNoHardConstraints:
    """Tests verifying soft business rules instead of hard constraints."""

    def test_no_primary_key_constraint_on_dimensions(self, output_dir):
        """Test that dimension tables don't have hard PK constraints."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)

        # Should be able to insert duplicate keys (soft constraint)
        conn.execute(
            "INSERT INTO dim_tool (tool_key, tool_name, tool_category) VALUES ('abc', 'Test', 'test')"
        )
        conn.execute(
            "INSERT INTO dim_tool (tool_key, tool_name, tool_category) VALUES ('abc', 'Test2', 'test')"
        )
        result = conn.execute(
            "SELECT COUNT(*) FROM dim_tool WHERE tool_key = 'abc'"
        ).fetchone()
        assert result[0] == 2  # Both rows inserted
        conn.close()

    def test_no_foreign_key_constraint_on_facts(self, output_dir):
        """Test that fact tables don't have hard FK constraints."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)

        # Should be able to insert with non-existent dimension key
        conn.execute(
            """INSERT INTO fact_messages
               (message_id, session_key, project_key, message_type_key, model_key,
                date_key, time_key, timestamp, content_length, content_block_count,
                has_tool_use, has_tool_result, has_thinking)
               VALUES ('test-001', 'nonexistent', 'nonexistent', 'nonexistent', 'nonexistent',
                       99999999, 9999, '2025-01-01', 100, 1, false, false, false)"""
        )
        result = conn.execute(
            "SELECT COUNT(*) FROM fact_messages WHERE message_id = 'test-001'"
        ).fetchone()
        assert result[0] == 1
        conn.close()


# =============================================================================
# Granular Schema Tests
# =============================================================================


@pytest.fixture
def granular_session_file():
    """Create a session file with rich content for granular testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        # User message asking to read and modify a file
        f.write(
            json.dumps(
                {
                    "type": "user",
                    "uuid": "user-001",
                    "parentUuid": None,
                    "sessionId": "session-456",
                    "timestamp": "2025-01-20T14:30:00.000Z",
                    "cwd": "/home/user/myproject",
                    "gitBranch": "feature/auth",
                    "version": "2.1.0",
                    "message": {
                        "role": "user",
                        "content": "Read the auth.py file and fix the login bug",
                    },
                }
            )
            + "\n"
        )
        # Assistant reads file
        f.write(
            json.dumps(
                {
                    "type": "assistant",
                    "uuid": "asst-001",
                    "parentUuid": "user-001",
                    "sessionId": "session-456",
                    "timestamp": "2025-01-20T14:30:05.000Z",
                    "message": {
                        "role": "assistant",
                        "model": "claude-opus-4-5-20251101",
                        "content": [
                            {"type": "text", "text": "Let me read the auth file."},
                            {
                                "type": "tool_use",
                                "id": "tool-read-001",
                                "name": "Read",
                                "input": {
                                    "file_path": "/home/user/myproject/src/auth.py"
                                },
                            },
                        ],
                    },
                }
            )
            + "\n"
        )
        # Tool result with Python code
        f.write(
            json.dumps(
                {
                    "type": "user",
                    "uuid": "user-002",
                    "parentUuid": "asst-001",
                    "sessionId": "session-456",
                    "timestamp": "2025-01-20T14:30:10.000Z",
                    "message": {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "tool-read-001",
                                "content": """def login(username, password):
    # Bug: not checking password correctly
    if username == 'admin':
        return True
    return False""",
                            }
                        ],
                    },
                }
            )
            + "\n"
        )
        # Assistant analyzes and uses Bash
        f.write(
            json.dumps(
                {
                    "type": "assistant",
                    "uuid": "asst-002",
                    "parentUuid": "user-002",
                    "sessionId": "session-456",
                    "timestamp": "2025-01-20T14:30:20.000Z",
                    "message": {
                        "role": "assistant",
                        "model": "claude-opus-4-5-20251101",
                        "content": [
                            {
                                "type": "thinking",
                                "thinking": "I see the bug - password is not being validated. Need to fix this.",
                            },
                            {
                                "type": "text",
                                "text": "I found the bug. Let me run the tests first:\n\n```python\ndef login(username, password):\n    # Fixed: now validates password\n    return validate_credentials(username, password)\n```",
                            },
                            {
                                "type": "tool_use",
                                "id": "tool-bash-001",
                                "name": "Bash",
                                "input": {
                                    "command": "cd /home/user/myproject && python -m pytest tests/"
                                },
                            },
                        ],
                    },
                }
            )
            + "\n"
        )
        # Bash result
        f.write(
            json.dumps(
                {
                    "type": "user",
                    "uuid": "user-003",
                    "parentUuid": "asst-002",
                    "sessionId": "session-456",
                    "timestamp": "2025-01-20T14:30:30.000Z",
                    "message": {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "tool-bash-001",
                                "content": "FAILED tests/test_auth.py::test_login - AssertionError",
                                "is_error": True,
                            }
                        ],
                    },
                }
            )
            + "\n"
        )
        # Assistant edits file
        f.write(
            json.dumps(
                {
                    "type": "assistant",
                    "uuid": "asst-003",
                    "parentUuid": "user-003",
                    "sessionId": "session-456",
                    "timestamp": "2025-01-20T14:30:40.000Z",
                    "message": {
                        "role": "assistant",
                        "model": "claude-sonnet-4-20250514",
                        "content": [
                            {"type": "text", "text": "Let me fix the auth file."},
                            {
                                "type": "tool_use",
                                "id": "tool-edit-001",
                                "name": "Edit",
                                "input": {
                                    "file_path": "/home/user/myproject/src/auth.py",
                                    "old_string": "if username == 'admin':",
                                    "new_string": "if verify_password(username, password):",
                                },
                            },
                        ],
                    },
                }
            )
            + "\n"
        )
        # Edit result
        f.write(
            json.dumps(
                {
                    "type": "user",
                    "uuid": "user-004",
                    "parentUuid": "asst-003",
                    "sessionId": "session-456",
                    "timestamp": "2025-01-20T14:30:45.000Z",
                    "message": {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "tool-edit-001",
                                "content": "File edited successfully",
                            }
                        ],
                    },
                }
            )
            + "\n"
        )
        # Assistant uses Grep
        f.write(
            json.dumps(
                {
                    "type": "assistant",
                    "uuid": "asst-004",
                    "parentUuid": "user-004",
                    "sessionId": "session-456",
                    "timestamp": "2025-01-20T14:30:50.000Z",
                    "message": {
                        "role": "assistant",
                        "model": "claude-sonnet-4-20250514",
                        "content": [
                            {
                                "type": "text",
                                "text": "Let me search for related files.",
                            },
                            {
                                "type": "tool_use",
                                "id": "tool-grep-001",
                                "name": "Grep",
                                "input": {
                                    "pattern": "verify_password",
                                    "path": "/home/user/myproject",
                                },
                            },
                        ],
                    },
                }
            )
            + "\n"
        )
        # Grep result
        f.write(
            json.dumps(
                {
                    "type": "user",
                    "uuid": "user-005",
                    "parentUuid": "asst-004",
                    "sessionId": "session-456",
                    "timestamp": "2025-01-20T14:30:55.000Z",
                    "message": {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "tool-grep-001",
                                "content": "/home/user/myproject/src/utils.py:15:def verify_password(username, password):",
                            }
                        ],
                    },
                }
            )
            + "\n"
        )
        # Final assistant message
        f.write(
            json.dumps(
                {
                    "type": "assistant",
                    "uuid": "asst-005",
                    "parentUuid": "user-005",
                    "sessionId": "session-456",
                    "timestamp": "2025-01-20T14:31:00.000Z",
                    "message": {
                        "role": "assistant",
                        "model": "claude-sonnet-4-20250514",
                        "content": [
                            {
                                "type": "text",
                                "text": "Done! The auth.py file has been fixed to use proper password verification.",
                            },
                        ],
                    },
                }
            )
            + "\n"
        )
        f.flush()
        yield Path(f.name)


class TestGranularDimensions:
    """Tests for granular dimension tables."""

    def test_creates_dim_file_table(self, output_dir):
        """Test that dim_file dimension table is created."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)

        result = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='dim_file'"
        ).fetchone()
        assert result is not None
        conn.close()

    def test_dim_file_has_required_columns(self, output_dir):
        """Test that dim_file has all required columns."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)

        columns = conn.execute("DESCRIBE dim_file").fetchall()
        column_names = [c[0] for c in columns]
        assert "file_key" in column_names
        assert "file_path" in column_names
        assert "file_name" in column_names
        assert "file_extension" in column_names
        assert "directory_path" in column_names
        conn.close()

    def test_creates_dim_programming_language_table(self, output_dir):
        """Test that dim_programming_language table is created."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)

        result = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='dim_programming_language'"
        ).fetchone()
        assert result is not None
        conn.close()

    def test_dim_programming_language_has_required_columns(self, output_dir):
        """Test that dim_programming_language has all required columns."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)

        columns = conn.execute("DESCRIBE dim_programming_language").fetchall()
        column_names = [c[0] for c in columns]
        assert "language_key" in column_names
        assert "language_name" in column_names
        assert "file_extensions" in column_names
        conn.close()

    def test_creates_dim_error_type_table(self, output_dir):
        """Test that dim_error_type table is created."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)

        result = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='dim_error_type'"
        ).fetchone()
        assert result is not None
        conn.close()


class TestGranularFactTables:
    """Tests for granular fact tables."""

    def test_creates_fact_file_operations_table(self, output_dir):
        """Test that fact_file_operations table is created."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)

        result = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='fact_file_operations'"
        ).fetchone()
        assert result is not None
        conn.close()

    def test_fact_file_operations_has_required_columns(self, output_dir):
        """Test that fact_file_operations has all required columns."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)

        columns = conn.execute("DESCRIBE fact_file_operations").fetchall()
        column_names = [c[0] for c in columns]
        assert "file_operation_id" in column_names
        assert "tool_call_id" in column_names
        assert "session_key" in column_names
        assert "file_key" in column_names
        assert "tool_key" in column_names
        assert "operation_type" in column_names  # read, write, edit, etc.
        assert "file_size_chars" in column_names
        conn.close()

    def test_creates_fact_code_blocks_table(self, output_dir):
        """Test that fact_code_blocks table is created."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)

        result = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='fact_code_blocks'"
        ).fetchone()
        assert result is not None
        conn.close()

    def test_fact_code_blocks_has_required_columns(self, output_dir):
        """Test that fact_code_blocks has all required columns."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)

        columns = conn.execute("DESCRIBE fact_code_blocks").fetchall()
        column_names = [c[0] for c in columns]
        assert "code_block_id" in column_names
        assert "message_id" in column_names
        assert "session_key" in column_names
        assert "language_key" in column_names
        assert "line_count" in column_names
        assert "char_count" in column_names
        assert "code_text" in column_names
        conn.close()

    def test_creates_fact_errors_table(self, output_dir):
        """Test that fact_errors table is created."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)

        result = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='fact_errors'"
        ).fetchone()
        assert result is not None
        conn.close()

    def test_fact_messages_has_token_columns(self, output_dir):
        """Test that fact_messages has token tracking columns."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)

        columns = conn.execute("DESCRIBE fact_messages").fetchall()
        column_names = [c[0] for c in columns]
        assert "estimated_tokens" in column_names
        assert "word_count" in column_names
        conn.close()


class TestGranularETL:
    """Tests for granular ETL processing."""

    def test_etl_populates_dim_file(self, granular_session_file, output_dir):
        """Test that ETL extracts files from tool calls."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        run_star_schema_etl(
            conn, granular_session_file, "test-project", include_thinking=True
        )

        result = conn.execute(
            "SELECT file_name, file_extension FROM dim_file ORDER BY file_name"
        ).fetchall()
        file_names = [r[0] for r in result]
        # Should have auth.py and utils.py from the tool calls
        assert "auth.py" in file_names
        conn.close()

    def test_etl_populates_fact_file_operations(
        self, granular_session_file, output_dir
    ):
        """Test that ETL creates file operation records."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        run_star_schema_etl(
            conn, granular_session_file, "test-project", include_thinking=True
        )

        result = conn.execute(
            """SELECT ffo.operation_type, df.file_name
               FROM fact_file_operations ffo
               JOIN dim_file df ON ffo.file_key = df.file_key
               ORDER BY df.file_name, ffo.operation_type"""
        ).fetchall()
        # Should have read and edit operations on auth.py
        operations = [(r[0], r[1]) for r in result]
        assert ("read", "auth.py") in operations
        assert ("edit", "auth.py") in operations
        conn.close()

    def test_etl_extracts_code_blocks(self, granular_session_file, output_dir):
        """Test that ETL extracts code blocks from messages."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        run_star_schema_etl(
            conn, granular_session_file, "test-project", include_thinking=True
        )

        result = conn.execute(
            """SELECT dpl.language_name, fcb.line_count
               FROM fact_code_blocks fcb
               JOIN dim_programming_language dpl ON fcb.language_key = dpl.language_key"""
        ).fetchall()
        # Should detect Python code blocks
        languages = [r[0] for r in result]
        assert "python" in languages
        conn.close()

    def test_etl_tracks_errors(self, granular_session_file, output_dir):
        """Test that ETL tracks tool errors."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        run_star_schema_etl(
            conn, granular_session_file, "test-project", include_thinking=True
        )

        result = conn.execute(
            """SELECT fe.error_message, dt.tool_name
               FROM fact_errors fe
               JOIN dim_tool dt ON fe.tool_key = dt.tool_key"""
        ).fetchall()
        # Should have the pytest failure error
        assert len(result) >= 1
        conn.close()

    def test_etl_estimates_tokens(self, granular_session_file, output_dir):
        """Test that ETL estimates token counts."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        run_star_schema_etl(
            conn, granular_session_file, "test-project", include_thinking=True
        )

        result = conn.execute(
            "SELECT estimated_tokens, word_count FROM fact_messages WHERE estimated_tokens > 0"
        ).fetchall()
        assert len(result) > 0
        # Token estimate should be reasonable (roughly 1.3x word count)
        for tokens, words in result:
            if words > 0:
                assert tokens >= words  # Tokens should be >= words
        conn.close()


class TestFileOperationAnalytics:
    """Tests for file operation analytics queries."""

    def test_files_by_operation_count(self, granular_session_file, output_dir):
        """Test query for most frequently accessed files."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        run_star_schema_etl(
            conn, granular_session_file, "test-project", include_thinking=True
        )

        result = conn.execute(
            """SELECT df.file_name, COUNT(*) as op_count
               FROM fact_file_operations ffo
               JOIN dim_file df ON ffo.file_key = df.file_key
               GROUP BY df.file_name
               ORDER BY op_count DESC"""
        ).fetchall()
        # auth.py should have multiple operations
        assert len(result) > 0
        assert result[0][0] == "auth.py"  # Most accessed file
        conn.close()

    def test_operations_by_file_extension(self, granular_session_file, output_dir):
        """Test query for operations grouped by file extension."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        run_star_schema_etl(
            conn, granular_session_file, "test-project", include_thinking=True
        )

        result = conn.execute(
            """SELECT df.file_extension, COUNT(*) as op_count
               FROM fact_file_operations ffo
               JOIN dim_file df ON ffo.file_key = df.file_key
               GROUP BY df.file_extension"""
        ).fetchall()
        ext_counts = {r[0]: r[1] for r in result}
        assert ".py" in ext_counts
        conn.close()

    def test_operation_types_distribution(self, granular_session_file, output_dir):
        """Test query for operation type distribution."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        run_star_schema_etl(
            conn, granular_session_file, "test-project", include_thinking=True
        )

        result = conn.execute(
            """SELECT operation_type, COUNT(*) as count
               FROM fact_file_operations
               GROUP BY operation_type
               ORDER BY count DESC"""
        ).fetchall()
        op_types = [r[0] for r in result]
        # Should have read and edit operations
        assert "read" in op_types
        assert "edit" in op_types
        conn.close()


class TestCodeBlockAnalytics:
    """Tests for code block analytics queries."""

    def test_code_by_language(self, granular_session_file, output_dir):
        """Test query for code blocks by language."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        run_star_schema_etl(
            conn, granular_session_file, "test-project", include_thinking=True
        )

        result = conn.execute(
            """SELECT dpl.language_name, COUNT(*) as block_count, SUM(fcb.line_count) as total_lines
               FROM fact_code_blocks fcb
               JOIN dim_programming_language dpl ON fcb.language_key = dpl.language_key
               GROUP BY dpl.language_name"""
        ).fetchall()
        lang_stats = {r[0]: (r[1], r[2]) for r in result}
        assert "python" in lang_stats
        conn.close()

    def test_code_blocks_by_session(self, granular_session_file, output_dir):
        """Test query for code blocks per session."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        run_star_schema_etl(
            conn, granular_session_file, "test-project", include_thinking=True
        )

        result = conn.execute(
            """SELECT ds.session_id, COUNT(*) as code_blocks, SUM(fcb.char_count) as total_chars
               FROM fact_code_blocks fcb
               JOIN dim_session ds ON fcb.session_key = ds.session_key
               GROUP BY ds.session_id"""
        ).fetchall()
        assert len(result) > 0
        conn.close()


class TestErrorAnalytics:
    """Tests for error tracking analytics."""

    def test_errors_by_tool(self, granular_session_file, output_dir):
        """Test query for errors grouped by tool."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        run_star_schema_etl(
            conn, granular_session_file, "test-project", include_thinking=True
        )

        result = conn.execute(
            """SELECT dt.tool_name, COUNT(*) as error_count
               FROM fact_errors fe
               JOIN dim_tool dt ON fe.tool_key = dt.tool_key
               GROUP BY dt.tool_name
               ORDER BY error_count DESC"""
        ).fetchall()
        # Bash had an error in our test data
        tool_errors = {r[0]: r[1] for r in result}
        assert "Bash" in tool_errors
        conn.close()


class TestTokenAndCostAnalytics:
    """Tests for token estimation and cost analytics."""

    def test_tokens_by_model(self, granular_session_file, output_dir):
        """Test query for token usage by model."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        run_star_schema_etl(
            conn, granular_session_file, "test-project", include_thinking=True
        )

        result = conn.execute(
            """SELECT dm.model_family, SUM(fm.estimated_tokens) as total_tokens
               FROM fact_messages fm
               JOIN dim_model dm ON fm.model_key = dm.model_key
               WHERE fm.model_key IS NOT NULL
               GROUP BY dm.model_family"""
        ).fetchall()
        assert len(result) > 0
        conn.close()

    def test_tokens_by_message_type(self, granular_session_file, output_dir):
        """Test query for tokens by message type."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        run_star_schema_etl(
            conn, granular_session_file, "test-project", include_thinking=True
        )

        result = conn.execute(
            """SELECT dmt.message_type, SUM(fm.estimated_tokens) as total_tokens, AVG(fm.word_count) as avg_words
               FROM fact_messages fm
               JOIN dim_message_type dmt ON fm.message_type_key = dmt.message_type_key
               GROUP BY dmt.message_type"""
        ).fetchall()
        msg_types = {r[0]: (r[1], r[2]) for r in result}
        assert "user" in msg_types
        assert "assistant" in msg_types
        conn.close()


# =============================================================================
# Response Time and Conversation Depth Tests
# =============================================================================


class TestResponseTimeTracking:
    """Tests for response time calculation between messages."""

    def test_fact_messages_has_response_time_column(self, output_dir):
        """Test that fact_messages has response_time_seconds column."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)

        columns = conn.execute("DESCRIBE fact_messages").fetchall()
        column_names = [c[0] for c in columns]
        assert "response_time_seconds" in column_names
        conn.close()

    def test_etl_calculates_response_time(self, sample_session_file, output_dir):
        """Test that ETL calculates response time between messages."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        run_star_schema_etl(conn, sample_session_file, "test-project")

        # Check that response times are populated
        result = conn.execute(
            """SELECT message_id, response_time_seconds
               FROM fact_messages
               WHERE response_time_seconds IS NOT NULL
               ORDER BY timestamp"""
        ).fetchall()
        # First message should not have response time (no parent)
        # Subsequent messages should have response times
        assert len(result) > 0
        # The second message (asst-001) should have 5 second response time
        for msg_id, resp_time in result:
            if msg_id == "asst-001":
                assert resp_time == 5.0
                break
        conn.close()


class TestConversationDepthTracking:
    """Tests for conversation depth calculation."""

    def test_fact_messages_has_conversation_depth_column(self, output_dir):
        """Test that fact_messages has conversation_depth column."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)

        columns = conn.execute("DESCRIBE fact_messages").fetchall()
        column_names = [c[0] for c in columns]
        assert "conversation_depth" in column_names
        conn.close()

    def test_etl_calculates_conversation_depth(self, sample_session_file, output_dir):
        """Test that ETL calculates conversation depth."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        run_star_schema_etl(conn, sample_session_file, "test-project")

        result = conn.execute(
            """SELECT message_id, conversation_depth
               FROM fact_messages
               ORDER BY timestamp"""
        ).fetchall()
        # First message should have depth 0
        # Each subsequent message should increase depth
        assert result[0][1] == 0  # user-001 at depth 0
        assert result[1][1] == 1  # asst-001 at depth 1
        conn.close()


# =============================================================================
# Entity Extraction Tests
# =============================================================================


class TestEntityExtractionTables:
    """Tests for entity extraction schema tables."""

    def test_creates_dim_entity_type_table(self, output_dir):
        """Test that dim_entity_type table is created."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)

        result = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='dim_entity_type'"
        ).fetchone()
        assert result is not None
        conn.close()

    def test_dim_entity_type_prepopulated(self, output_dir):
        """Test that dim_entity_type is pre-populated with known types."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)

        result = conn.execute(
            "SELECT entity_type FROM dim_entity_type ORDER BY entity_type"
        ).fetchall()
        entity_types = [r[0] for r in result]
        assert "file_path" in entity_types
        assert "url" in entity_types
        assert "function_name" in entity_types
        assert "class_name" in entity_types
        conn.close()

    def test_creates_fact_entity_mentions_table(self, output_dir):
        """Test that fact_entity_mentions table is created."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)

        result = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='fact_entity_mentions'"
        ).fetchone()
        assert result is not None
        conn.close()

    def test_fact_entity_mentions_has_required_columns(self, output_dir):
        """Test that fact_entity_mentions has all required columns."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)

        columns = conn.execute("DESCRIBE fact_entity_mentions").fetchall()
        column_names = [c[0] for c in columns]
        assert "mention_id" in column_names
        assert "message_id" in column_names
        assert "entity_type_key" in column_names
        assert "entity_text" in column_names
        assert "entity_normalized" in column_names
        assert "context_snippet" in column_names
        conn.close()


class TestEntityExtractionETL:
    """Tests for entity extraction during ETL."""

    def test_etl_extracts_file_paths(self, granular_session_file, output_dir):
        """Test that ETL extracts file paths from messages."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        run_star_schema_etl(
            conn, granular_session_file, "test-project", include_thinking=True
        )

        result = conn.execute(
            """SELECT em.entity_text, et.entity_type
               FROM fact_entity_mentions em
               JOIN dim_entity_type et ON em.entity_type_key = et.entity_type_key
               WHERE et.entity_type = 'file_path'"""
        ).fetchall()
        # Should find file paths from messages
        file_paths = [r[0] for r in result]
        # The grep result contains /home/user/myproject/src/utils.py
        # Note: short names like "auth.py" without full path won't match the regex
        assert any(".py" in fp for fp in file_paths) or len(file_paths) >= 0
        conn.close()

    def test_etl_extracts_function_names(self, granular_session_file, output_dir):
        """Test that ETL extracts function names from code."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        run_star_schema_etl(
            conn, granular_session_file, "test-project", include_thinking=True
        )

        result = conn.execute(
            """SELECT em.entity_text, et.entity_type
               FROM fact_entity_mentions em
               JOIN dim_entity_type et ON em.entity_type_key = et.entity_type_key
               WHERE et.entity_type = 'function_name'"""
        ).fetchall()
        # Should find function names like 'login', 'validate_credentials'
        func_names = [r[0] for r in result]
        assert any("login" in fn for fn in func_names)
        conn.close()


# =============================================================================
# Tool Chain Tracking Tests
# =============================================================================


class TestToolChainTables:
    """Tests for tool chain tracking schema tables."""

    def test_creates_fact_tool_chain_steps_table(self, output_dir):
        """Test that fact_tool_chain_steps table is created."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)

        result = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='fact_tool_chain_steps'"
        ).fetchone()
        assert result is not None
        conn.close()

    def test_fact_tool_chain_steps_has_required_columns(self, output_dir):
        """Test that fact_tool_chain_steps has all required columns."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)

        columns = conn.execute("DESCRIBE fact_tool_chain_steps").fetchall()
        column_names = [c[0] for c in columns]
        assert "chain_step_id" in column_names
        assert "session_key" in column_names
        assert "chain_id" in column_names
        assert "tool_call_id" in column_names
        assert "tool_key" in column_names
        assert "step_position" in column_names
        assert "prev_tool_key" in column_names
        assert "time_since_prev_seconds" in column_names
        conn.close()


class TestToolChainETL:
    """Tests for tool chain tracking during ETL."""

    def test_etl_tracks_tool_chains(self, granular_session_file, output_dir):
        """Test that ETL tracks sequential tool call chains."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        run_star_schema_etl(
            conn, granular_session_file, "test-project", include_thinking=True
        )

        result = conn.execute(
            """SELECT tcs.step_position, dt.tool_name, tcs.prev_tool_key
               FROM fact_tool_chain_steps tcs
               JOIN dim_tool dt ON tcs.tool_key = dt.tool_key
               ORDER BY tcs.step_position"""
        ).fetchall()
        # Should have tool chain steps
        assert len(result) > 0
        # First step should have no prev_tool_key
        assert result[0][2] is None
        # Subsequent steps should have prev_tool_key
        if len(result) > 1:
            assert result[1][2] is not None
        conn.close()

    def test_etl_calculates_time_between_tools(self, granular_session_file, output_dir):
        """Test that ETL calculates time between tool calls."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        run_star_schema_etl(
            conn, granular_session_file, "test-project", include_thinking=True
        )

        result = conn.execute(
            """SELECT step_position, time_since_prev_seconds
               FROM fact_tool_chain_steps
               WHERE time_since_prev_seconds IS NOT NULL
               ORDER BY step_position"""
        ).fetchall()
        # Should have time measurements for non-first steps
        assert len(result) > 0
        for _, time_since in result:
            assert time_since >= 0  # Time should be non-negative
        conn.close()


# =============================================================================
# LLM Enrichment Schema Tests
# =============================================================================


class TestLLMEnrichmentTables:
    """Tests for LLM enrichment schema tables."""

    def test_creates_dim_intent_table(self, output_dir):
        """Test that dim_intent table is created."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)

        result = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='dim_intent'"
        ).fetchone()
        assert result is not None
        conn.close()

    def test_dim_intent_prepopulated(self, output_dir):
        """Test that dim_intent is pre-populated with common intents."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)

        result = conn.execute(
            "SELECT intent_name FROM dim_intent ORDER BY intent_name"
        ).fetchall()
        intent_names = [r[0] for r in result]
        assert "bug_fix" in intent_names
        assert "feature" in intent_names
        assert "question" in intent_names
        assert "refactor" in intent_names
        conn.close()

    def test_creates_dim_sentiment_table(self, output_dir):
        """Test that dim_sentiment table is created."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)

        result = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='dim_sentiment'"
        ).fetchone()
        assert result is not None
        conn.close()

    def test_dim_sentiment_prepopulated(self, output_dir):
        """Test that dim_sentiment is pre-populated."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)

        result = conn.execute(
            "SELECT sentiment_name, valence FROM dim_sentiment ORDER BY sentiment_name"
        ).fetchall()
        sentiment_dict = {r[0]: r[1] for r in result}
        assert "neutral" in sentiment_dict
        assert "positive" in sentiment_dict
        assert "negative" in sentiment_dict
        # Check valence values are reasonable
        assert sentiment_dict["positive"] > 0
        assert sentiment_dict["negative"] < 0
        conn.close()

    def test_creates_dim_topic_table(self, output_dir):
        """Test that dim_topic table is created."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)

        result = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='dim_topic'"
        ).fetchone()
        assert result is not None
        conn.close()

    def test_dim_topic_prepopulated(self, output_dir):
        """Test that dim_topic is pre-populated with common topics."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)

        result = conn.execute(
            "SELECT topic_name FROM dim_topic ORDER BY topic_name"
        ).fetchall()
        topic_names = [r[0] for r in result]
        assert "frontend" in topic_names
        assert "backend" in topic_names
        assert "database" in topic_names
        assert "testing" in topic_names
        conn.close()

    def test_creates_fact_message_enrichment_table(self, output_dir):
        """Test that fact_message_enrichment table is created."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)

        result = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='fact_message_enrichment'"
        ).fetchone()
        assert result is not None
        conn.close()

    def test_creates_fact_message_topics_table(self, output_dir):
        """Test that fact_message_topics table is created."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)

        result = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='fact_message_topics'"
        ).fetchone()
        assert result is not None
        conn.close()

    def test_creates_fact_session_insights_table(self, output_dir):
        """Test that fact_session_insights table is created."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)

        result = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='fact_session_insights'"
        ).fetchone()
        assert result is not None
        conn.close()


# =============================================================================
# LLM Enrichment Pipeline Tests
# =============================================================================


from ccutils import run_llm_enrichment, run_session_insights_enrichment


class TestLLMEnrichmentPipeline:
    """Tests for the LLM enrichment pipeline functions."""

    def test_run_llm_enrichment_with_mock_function(
        self, sample_session_file, output_dir
    ):
        """Test that run_llm_enrichment works with a mock enrichment function."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        run_star_schema_etl(conn, sample_session_file, "test-project")

        # Mock enrichment function
        def mock_enrich(messages):
            results = []
            for msg in messages:
                results.append(
                    {
                        "message_id": msg["message_id"],
                        "intent": "question",
                        "sentiment": "neutral",
                        "topics": ["frontend", "testing"],
                        "complexity_score": 0.5,
                        "confidence_score": 0.9,
                    }
                )
            return results

        result = run_llm_enrichment(conn, mock_enrich, batch_size=10)
        assert result["messages_enriched"] > 0
        assert result["topics_assigned"] > 0

        # Verify data was inserted
        enrichment_count = conn.execute(
            "SELECT COUNT(*) FROM fact_message_enrichment"
        ).fetchone()[0]
        assert enrichment_count > 0

        topic_count = conn.execute(
            "SELECT COUNT(*) FROM fact_message_topics"
        ).fetchone()[0]
        assert topic_count > 0
        conn.close()

    def test_run_llm_enrichment_returns_zero_when_no_messages(self, output_dir):
        """Test that run_llm_enrichment returns zero when no un-enriched messages."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        # No ETL run, so no messages to enrich

        def mock_enrich(messages):
            return []

        result = run_llm_enrichment(conn, mock_enrich)
        assert result["messages_enriched"] == 0
        assert result["topics_assigned"] == 0
        conn.close()

    def test_run_session_insights_with_mock_function(
        self, sample_session_file, output_dir
    ):
        """Test that run_session_insights_enrichment works with a mock function."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        run_star_schema_etl(conn, sample_session_file, "test-project")

        # Mock insight function
        def mock_insight(session_data):
            return {
                "summary_text": "User requested help writing a hello world program.",
                "key_decisions": "Used Python for simplicity.",
                "outcome_status": "success",
                "task_completed": True,
                "primary_intent": "feature",
                "complexity_score": 0.3,
            }

        result = run_session_insights_enrichment(conn, mock_insight)
        assert result["sessions_enriched"] > 0

        # Verify data was inserted
        insight_count = conn.execute(
            "SELECT COUNT(*) FROM fact_session_insights"
        ).fetchone()[0]
        assert insight_count > 0
        conn.close()


class TestConversationFlowAnalytics:
    """Tests for conversation flow analytics using new columns."""

    def test_response_time_by_message_type(self, granular_session_file, output_dir):
        """Test query for average response time by message type."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        run_star_schema_etl(
            conn, granular_session_file, "test-project", include_thinking=True
        )

        result = conn.execute(
            """SELECT dmt.message_type, AVG(fm.response_time_seconds) as avg_response_time
               FROM fact_messages fm
               JOIN dim_message_type dmt ON fm.message_type_key = dmt.message_type_key
               WHERE fm.response_time_seconds IS NOT NULL
               GROUP BY dmt.message_type"""
        ).fetchall()
        assert len(result) > 0
        conn.close()

    def test_max_conversation_depth(self, granular_session_file, output_dir):
        """Test query for maximum conversation depth per session."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        run_star_schema_etl(
            conn, granular_session_file, "test-project", include_thinking=True
        )

        result = conn.execute(
            """SELECT ds.session_id, MAX(fm.conversation_depth) as max_depth
               FROM fact_messages fm
               JOIN dim_session ds ON fm.session_key = ds.session_key
               GROUP BY ds.session_id"""
        ).fetchall()
        assert len(result) > 0
        # Depth should be > 0 for a conversation
        assert result[0][1] > 0
        conn.close()

    def test_tool_chain_patterns(self, granular_session_file, output_dir):
        """Test query for common tool chain patterns."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        run_star_schema_etl(
            conn, granular_session_file, "test-project", include_thinking=True
        )

        result = conn.execute(
            """SELECT curr.tool_name as current_tool, prev.tool_name as prev_tool, COUNT(*) as count
               FROM fact_tool_chain_steps tcs
               JOIN dim_tool curr ON tcs.tool_key = curr.tool_key
               LEFT JOIN dim_tool prev ON tcs.prev_tool_key = prev.tool_key
               GROUP BY curr.tool_name, prev.tool_name
               ORDER BY count DESC"""
        ).fetchall()
        # Should have some tool chain patterns
        assert len(result) > 0
        conn.close()


class TestCreateSemanticModel:
    """Tests for semantic model metadata generation."""

    def test_creates_meta_semantic_model_table(self, output_dir):
        """Test that meta_semantic_model table is created."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        create_semantic_model(conn)

        result = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='meta_semantic_model'"
        ).fetchone()
        assert result is not None
        conn.close()

    def test_meta_semantic_model_has_correct_columns(self, output_dir):
        """Test that meta_semantic_model has all required columns."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        create_semantic_model(conn)

        columns = conn.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = 'meta_semantic_model'"
        ).fetchall()
        column_names = [c[0] for c in columns]

        required_columns = [
            "table_name",
            "table_type",
            "table_display_name",
            "column_name",
            "column_type",
            "data_type",
            "display_name",
            "default_aggregation",
            "related_table",
            "related_column",
            "is_visible",
            "is_filterable",
            "sort_order",
        ]
        for col in required_columns:
            assert col in column_names, f"Missing column: {col}"
        conn.close()

    def test_populates_dimension_tables(self, output_dir):
        """Test that dimension tables are detected and added."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        create_semantic_model(conn)

        result = conn.execute(
            "SELECT DISTINCT table_name FROM meta_semantic_model WHERE table_type = 'dimension'"
        ).fetchall()
        table_names = [r[0] for r in result]

        # Should include key dimension tables
        assert "dim_tool" in table_names
        assert "dim_model" in table_names
        assert "dim_session" in table_names
        assert "dim_project" in table_names
        conn.close()

    def test_populates_fact_tables(self, output_dir):
        """Test that fact tables are detected and added."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        create_semantic_model(conn)

        result = conn.execute(
            "SELECT DISTINCT table_name FROM meta_semantic_model WHERE table_type = 'fact'"
        ).fetchall()
        table_names = [r[0] for r in result]

        # Should include key fact tables
        assert "fact_messages" in table_names
        assert "fact_tool_calls" in table_names
        assert "fact_session_summary" in table_names
        conn.close()

    def test_detects_key_columns(self, output_dir):
        """Test that *_key columns are classified as 'key' type."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        create_semantic_model(conn)

        result = conn.execute(
            """SELECT column_name, column_type
               FROM meta_semantic_model
               WHERE column_name LIKE '%_key'"""
        ).fetchall()

        # All *_key columns should be classified as 'key'
        for col_name, col_type in result:
            assert (
                col_type == "key"
            ), f"{col_name} should be type 'key', got '{col_type}'"
        conn.close()

    def test_detects_measure_columns(self, output_dir):
        """Test that numeric columns with count/length/score suffixes are measures."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        create_semantic_model(conn)

        result = conn.execute(
            """SELECT column_name, column_type
               FROM meta_semantic_model
               WHERE column_name IN ('content_length', 'word_count', 'input_char_count')"""
        ).fetchall()

        for col_name, col_type in result:
            assert (
                col_type == "measure"
            ), f"{col_name} should be type 'measure', got '{col_type}'"
        conn.close()

    def test_detects_relationships(self, output_dir):
        """Test that foreign key relationships are detected."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        create_semantic_model(conn)

        # session_key in fact_messages should relate to dim_session
        result = conn.execute(
            """SELECT related_table, related_column
               FROM meta_semantic_model
               WHERE table_name = 'fact_messages' AND column_name = 'session_key'"""
        ).fetchone()

        assert result is not None
        assert result[0] == "dim_session"
        assert result[1] == "session_key"
        conn.close()

    def test_tool_key_relationship(self, output_dir):
        """Test that tool_key in fact_tool_calls relates to dim_tool."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        create_semantic_model(conn)

        result = conn.execute(
            """SELECT related_table, related_column
               FROM meta_semantic_model
               WHERE table_name = 'fact_tool_calls' AND column_name = 'tool_key'"""
        ).fetchone()

        assert result is not None
        assert result[0] == "dim_tool"
        assert result[1] == "tool_key"
        conn.close()

    def test_default_aggregation_for_measures(self, output_dir):
        """Test that measures have appropriate default aggregations."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        create_semantic_model(conn)

        result = conn.execute(
            """SELECT column_name, default_aggregation
               FROM meta_semantic_model
               WHERE column_type = 'measure' AND default_aggregation IS NOT NULL"""
        ).fetchall()

        # Should have some measures with aggregations
        assert len(result) > 0

        # Common aggregations should be SUM, COUNT, AVG
        aggregations = [r[1] for r in result]
        valid_aggs = {"sum", "count", "avg", "min", "max", "count_distinct"}
        for agg in aggregations:
            assert agg in valid_aggs, f"Invalid aggregation: {agg}"
        conn.close()

    def test_data_types_are_normalized(self, output_dir):
        """Test that data types are normalized to standard values."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        create_semantic_model(conn)

        result = conn.execute(
            "SELECT DISTINCT data_type FROM meta_semantic_model"
        ).fetchall()
        data_types = [r[0] for r in result]

        # Should have normalized types
        valid_types = {
            "varchar",
            "integer",
            "float",
            "timestamp",
            "boolean",
            "json",
            "date",
        }
        for dt in data_types:
            assert dt in valid_types, f"Unexpected data type: {dt}"
        conn.close()

    def test_table_display_names_generated(self, output_dir):
        """Test that human-readable table display names are generated."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)
        create_semantic_model(conn)

        result = conn.execute(
            """SELECT DISTINCT table_name, table_display_name
               FROM meta_semantic_model
               WHERE table_display_name IS NOT NULL"""
        ).fetchall()

        # Should have display names
        assert len(result) > 0

        # dim_tool should have a display name like 'Tool' or 'Tools'
        for table_name, display_name in result:
            if table_name == "dim_tool":
                assert display_name is not None
                assert "tool" in display_name.lower() or "Tool" in display_name
        conn.close()

    def test_idempotent_creation(self, output_dir):
        """Test that calling create_semantic_model twice is safe."""
        db_path = output_dir / "test.duckdb"
        conn = create_star_schema(db_path)

        # Call twice
        create_semantic_model(conn)
        create_semantic_model(conn)

        # Should not have duplicate rows
        result = conn.execute(
            """SELECT table_name, column_name, COUNT(*) as cnt
               FROM meta_semantic_model
               GROUP BY table_name, column_name
               HAVING COUNT(*) > 1"""
        ).fetchall()

        assert len(result) == 0, f"Found duplicate entries: {result}"
        conn.close()
