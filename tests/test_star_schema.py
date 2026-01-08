"""Tests for star schema DuckDB implementation."""

import json
import tempfile
from pathlib import Path
from datetime import datetime
import hashlib

import duckdb
import pytest

from claude_code_transcripts import (
    create_star_schema,
    run_star_schema_etl,
    generate_dimension_key,
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
