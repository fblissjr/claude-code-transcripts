"""Tests for DuckDB export functionality."""

import json
import tempfile
from pathlib import Path

import duckdb
import pytest
from click.testing import CliRunner

from claude_code_transcripts import (
    cli,
    create_duckdb_schema,
    export_session_to_duckdb,
    generate_duckdb_archive,
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
                    "timestamp": "2025-01-01T10:00:00.000Z",
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
                    "timestamp": "2025-01-01T10:00:05.000Z",
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
                    "timestamp": "2025-01-01T10:00:10.000Z",
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
        # Assistant with thinking
        f.write(
            json.dumps(
                {
                    "type": "assistant",
                    "uuid": "asst-002",
                    "parentUuid": "user-002",
                    "sessionId": "session-123",
                    "timestamp": "2025-01-01T10:00:15.000Z",
                    "message": {
                        "role": "assistant",
                        "model": "claude-opus-4-5-20251101",
                        "content": [
                            {
                                "type": "thinking",
                                "thinking": "The file was created successfully. I should confirm this to the user.",
                            },
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


class TestCreateDuckdbSchema:
    """Tests for DuckDB schema creation."""

    def test_creates_sessions_table(self, output_dir):
        """Test that sessions table is created."""
        db_path = output_dir / "test.duckdb"
        conn = create_duckdb_schema(db_path)

        result = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='sessions'"
        ).fetchone()
        assert result is not None
        conn.close()

    def test_creates_messages_table(self, output_dir):
        """Test that messages table is created."""
        db_path = output_dir / "test.duckdb"
        conn = create_duckdb_schema(db_path)

        result = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='messages'"
        ).fetchone()
        assert result is not None
        conn.close()

    def test_creates_tool_calls_table(self, output_dir):
        """Test that tool_calls table is created."""
        db_path = output_dir / "test.duckdb"
        conn = create_duckdb_schema(db_path)

        result = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='tool_calls'"
        ).fetchone()
        assert result is not None
        conn.close()

    def test_creates_thinking_table(self, output_dir):
        """Test that thinking table is created."""
        db_path = output_dir / "test.duckdb"
        conn = create_duckdb_schema(db_path)

        result = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='thinking'"
        ).fetchone()
        assert result is not None
        conn.close()


class TestExportSessionToDuckdb:
    """Tests for exporting a single session to DuckDB."""

    def test_exports_user_messages(self, sample_session_file, output_dir):
        """Test that user messages are exported."""
        db_path = output_dir / "test.duckdb"
        conn = create_duckdb_schema(db_path)

        export_session_to_duckdb(conn, sample_session_file, "test-project")

        result = conn.execute(
            "SELECT COUNT(*) FROM messages WHERE type = 'user'"
        ).fetchone()
        assert result[0] == 2  # Two user messages
        conn.close()

    def test_exports_assistant_messages(self, sample_session_file, output_dir):
        """Test that assistant messages are exported."""
        db_path = output_dir / "test.duckdb"
        conn = create_duckdb_schema(db_path)

        export_session_to_duckdb(conn, sample_session_file, "test-project")

        result = conn.execute(
            "SELECT COUNT(*) FROM messages WHERE type = 'assistant'"
        ).fetchone()
        assert result[0] == 2  # Two assistant messages
        conn.close()

    def test_exports_tool_calls(self, sample_session_file, output_dir):
        """Test that tool calls are exported."""
        db_path = output_dir / "test.duckdb"
        conn = create_duckdb_schema(db_path)

        export_session_to_duckdb(conn, sample_session_file, "test-project")

        result = conn.execute("SELECT tool_name FROM tool_calls").fetchone()
        assert result[0] == "Write"
        conn.close()

    def test_exports_session_metadata(self, sample_session_file, output_dir):
        """Test that session metadata is exported."""
        db_path = output_dir / "test.duckdb"
        conn = create_duckdb_schema(db_path)

        export_session_to_duckdb(conn, sample_session_file, "test-project")

        result = conn.execute(
            "SELECT project_name, cwd, git_branch FROM sessions"
        ).fetchone()
        assert result[0] == "test-project"
        assert result[1] == "/home/user/project"
        assert result[2] == "main"
        conn.close()

    def test_does_not_export_thinking_by_default(self, sample_session_file, output_dir):
        """Test that thinking blocks are not exported by default."""
        db_path = output_dir / "test.duckdb"
        conn = create_duckdb_schema(db_path)

        export_session_to_duckdb(
            conn, sample_session_file, "test-project", include_thinking=False
        )

        result = conn.execute("SELECT COUNT(*) FROM thinking").fetchone()
        assert result[0] == 0
        conn.close()

    def test_exports_thinking_when_enabled(self, sample_session_file, output_dir):
        """Test that thinking blocks are exported when enabled."""
        db_path = output_dir / "test.duckdb"
        conn = create_duckdb_schema(db_path)

        export_session_to_duckdb(
            conn, sample_session_file, "test-project", include_thinking=True
        )

        result = conn.execute("SELECT COUNT(*) FROM thinking").fetchone()
        assert result[0] == 1
        conn.close()

    def test_extracts_text_content(self, sample_session_file, output_dir):
        """Test that text content is properly extracted."""
        db_path = output_dir / "test.duckdb"
        conn = create_duckdb_schema(db_path)

        export_session_to_duckdb(conn, sample_session_file, "test-project")

        result = conn.execute(
            "SELECT content FROM messages WHERE type = 'user' ORDER BY timestamp LIMIT 1"
        ).fetchone()
        assert "hello world" in result[0].lower()
        conn.close()


class TestGenerateDuckdbArchive:
    """Tests for full archive generation."""

    def test_generates_duckdb_file(self, mock_projects_dir, output_dir):
        """Test that DuckDB file is generated."""
        generate_duckdb_archive(mock_projects_dir, output_dir)

        db_path = output_dir / "archive.duckdb"
        assert db_path.exists()

    def test_archive_contains_sessions(self, mock_projects_dir, output_dir):
        """Test that archive contains session data."""
        generate_duckdb_archive(mock_projects_dir, output_dir)

        db_path = output_dir / "archive.duckdb"
        conn = duckdb.connect(str(db_path))
        result = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()
        assert result[0] >= 1
        conn.close()


class TestDuckdbCliOptions:
    """Tests for DuckDB CLI options."""

    def test_format_duckdb_option(self, mock_projects_dir, output_dir):
        """Test --format duckdb creates DuckDB file."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "all",
                "--source",
                str(mock_projects_dir),
                "--output",
                str(output_dir),
                "--format",
                "duckdb",
            ],
        )

        assert result.exit_code == 0
        assert (output_dir / "archive.duckdb").exists()
        # HTML should NOT be generated
        assert not (output_dir / "index.html").exists()

    def test_format_html_is_default(self, mock_projects_dir, output_dir):
        """Test that --format html is the default."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "all",
                "--source",
                str(mock_projects_dir),
                "--output",
                str(output_dir),
            ],
        )

        assert result.exit_code == 0
        assert (output_dir / "index.html").exists()
        # DuckDB should NOT be generated by default
        assert not (output_dir / "archive.duckdb").exists()

    def test_format_both_creates_both(self, mock_projects_dir, output_dir):
        """Test --format both creates both HTML and DuckDB."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "all",
                "--source",
                str(mock_projects_dir),
                "--output",
                str(output_dir),
                "--format",
                "both",
            ],
        )

        assert result.exit_code == 0
        assert (output_dir / "index.html").exists()
        assert (output_dir / "archive.duckdb").exists()

    def test_include_thinking_flag(self, mock_projects_dir, output_dir):
        """Test --include-thinking flag."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "all",
                "--source",
                str(mock_projects_dir),
                "--output",
                str(output_dir),
                "--format",
                "duckdb",
                "--include-thinking",
            ],
        )

        assert result.exit_code == 0
        db_path = output_dir / "archive.duckdb"
        conn = duckdb.connect(str(db_path))
        result = conn.execute("SELECT COUNT(*) FROM thinking").fetchone()
        assert result[0] >= 1
        conn.close()


class TestDuckdbFullTextSearch:
    """Tests for full-text search functionality."""

    def test_can_search_message_content(self, sample_session_file, output_dir):
        """Test that messages can be searched with LIKE."""
        db_path = output_dir / "test.duckdb"
        conn = create_duckdb_schema(db_path)
        export_session_to_duckdb(conn, sample_session_file, "test-project")

        result = conn.execute(
            "SELECT content FROM messages WHERE content LIKE '%hello world%'"
        ).fetchall()
        assert len(result) >= 1
        conn.close()

    def test_can_search_tool_names(self, sample_session_file, output_dir):
        """Test that tool calls can be searched by name."""
        db_path = output_dir / "test.duckdb"
        conn = create_duckdb_schema(db_path)
        export_session_to_duckdb(conn, sample_session_file, "test-project")

        result = conn.execute(
            "SELECT tool_name, input_summary FROM tool_calls WHERE tool_name = 'Write'"
        ).fetchall()
        assert len(result) == 1
        conn.close()
