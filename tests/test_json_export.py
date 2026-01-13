"""Tests for JSON export functionality."""

import json
import tempfile
from pathlib import Path

import pytest

from claude_code_transcripts import (
    resolve_schema_format,
    export_sessions_to_json,
    create_star_schema,
    run_star_schema_etl,
    create_semantic_model,
)
from claude_code_transcripts.star_schema import export_star_schema_to_json


class TestResolveSchemaFormat:
    """Tests for schema/format resolution logic."""

    def test_simple_duckdb_inferred(self):
        """Test that duckdb format infers simple schema."""
        schema, fmt = resolve_schema_format(None, "duckdb")
        assert schema == "simple"
        assert fmt == "duckdb"

    def test_star_duckdb_inferred(self):
        """Test that duckdb-star format infers star schema."""
        schema, fmt = resolve_schema_format(None, "duckdb-star")
        assert schema == "star"
        assert fmt == "duckdb"

    def test_simple_json_inferred(self):
        """Test that json format infers simple schema."""
        schema, fmt = resolve_schema_format(None, "json")
        assert schema == "simple"
        assert fmt == "json"

    def test_star_json_inferred(self):
        """Test that json-star format infers star schema."""
        schema, fmt = resolve_schema_format(None, "json-star")
        assert schema == "star"
        assert fmt == "json"

    def test_html_infers_simple(self):
        """Test that html format infers simple schema."""
        schema, fmt = resolve_schema_format(None, "html")
        assert schema == "simple"
        assert fmt == "html"

    def test_explicit_schema_overrides_inference(self):
        """Test that explicit --schema overrides format inference."""
        schema, fmt = resolve_schema_format("star", "duckdb")
        assert schema == "star"
        assert fmt == "duckdb"

    def test_explicit_simple_with_star_format(self):
        """Test explicit simple with -star format prefers explicit."""
        schema, fmt = resolve_schema_format("simple", "duckdb-star")
        assert schema == "simple"
        assert fmt == "duckdb"

    def test_explicit_star_with_json(self):
        """Test explicit star schema with json format."""
        schema, fmt = resolve_schema_format("star", "json")
        assert schema == "star"
        assert fmt == "json"


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
                                "thinking": "The file was created successfully.",
                            },
                            {
                                "type": "text",
                                "text": "Done! I've created hello.py.",
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


class TestExportSessionsToJson:
    """Tests for simple schema JSON export."""

    def test_creates_json_file(self, sample_session_file, output_dir):
        """Test that JSON file is created."""
        output_path = output_dir / "export.json"
        export_sessions_to_json([sample_session_file], output_path)
        assert output_path.exists()

    def test_output_is_valid_json(self, sample_session_file, output_dir):
        """Test that output is valid JSON."""
        output_path = output_dir / "export.json"
        export_sessions_to_json([sample_session_file], output_path)

        with open(output_path) as f:
            data = json.load(f)
        assert isinstance(data, dict)

    def test_has_schema_type(self, sample_session_file, output_dir):
        """Test that output has schema_type field."""
        output_path = output_dir / "export.json"
        export_sessions_to_json([sample_session_file], output_path)

        with open(output_path) as f:
            data = json.load(f)
        assert data.get("schema_type") == "simple"

    def test_has_version(self, sample_session_file, output_dir):
        """Test that output has version field."""
        output_path = output_dir / "export.json"
        export_sessions_to_json([sample_session_file], output_path)

        with open(output_path) as f:
            data = json.load(f)
        assert "version" in data

    def test_has_tables_object(self, sample_session_file, output_dir):
        """Test that output has tables object."""
        output_path = output_dir / "export.json"
        export_sessions_to_json([sample_session_file], output_path)

        with open(output_path) as f:
            data = json.load(f)
        assert "tables" in data
        assert isinstance(data["tables"], dict)

    def test_has_sessions_table(self, sample_session_file, output_dir):
        """Test that tables contains sessions."""
        output_path = output_dir / "export.json"
        export_sessions_to_json([sample_session_file], output_path)

        with open(output_path) as f:
            data = json.load(f)
        assert "sessions" in data["tables"]
        assert len(data["tables"]["sessions"]) == 1

    def test_has_messages_table(self, sample_session_file, output_dir):
        """Test that tables contains messages."""
        output_path = output_dir / "export.json"
        export_sessions_to_json([sample_session_file], output_path)

        with open(output_path) as f:
            data = json.load(f)
        assert "messages" in data["tables"]
        assert len(data["tables"]["messages"]) == 4  # 2 user + 2 assistant

    def test_has_tool_calls_table(self, sample_session_file, output_dir):
        """Test that tables contains tool_calls."""
        output_path = output_dir / "export.json"
        export_sessions_to_json([sample_session_file], output_path)

        with open(output_path) as f:
            data = json.load(f)
        assert "tool_calls" in data["tables"]
        assert len(data["tables"]["tool_calls"]) == 1  # One Write tool call

    def test_has_thinking_table(self, sample_session_file, output_dir):
        """Test that tables contains thinking (empty by default)."""
        output_path = output_dir / "export.json"
        export_sessions_to_json([sample_session_file], output_path)

        with open(output_path) as f:
            data = json.load(f)
        assert "thinking" in data["tables"]
        # By default, thinking is not included
        assert len(data["tables"]["thinking"]) == 0

    def test_includes_thinking_when_enabled(self, sample_session_file, output_dir):
        """Test that thinking is included when enabled."""
        output_path = output_dir / "export.json"
        export_sessions_to_json(
            [sample_session_file], output_path, include_thinking=True
        )

        with open(output_path) as f:
            data = json.load(f)
        assert len(data["tables"]["thinking"]) == 1

    def test_session_has_required_fields(self, sample_session_file, output_dir):
        """Test that session records have required fields."""
        output_path = output_dir / "export.json"
        export_sessions_to_json([sample_session_file], output_path)

        with open(output_path) as f:
            data = json.load(f)

        session = data["tables"]["sessions"][0]
        assert "session_id" in session
        assert "project_name" in session
        assert "cwd" in session
        assert "git_branch" in session
        assert "message_count" in session

    def test_message_has_required_fields(self, sample_session_file, output_dir):
        """Test that message records have required fields."""
        output_path = output_dir / "export.json"
        export_sessions_to_json([sample_session_file], output_path)

        with open(output_path) as f:
            data = json.load(f)

        message = data["tables"]["messages"][0]
        assert "id" in message
        assert "session_id" in message
        assert "type" in message
        assert "timestamp" in message
        assert "content" in message

    def test_tool_call_has_required_fields(self, sample_session_file, output_dir):
        """Test that tool_call records have required fields."""
        output_path = output_dir / "export.json"
        export_sessions_to_json([sample_session_file], output_path)

        with open(output_path) as f:
            data = json.load(f)

        tool_call = data["tables"]["tool_calls"][0]
        assert "tool_use_id" in tool_call
        assert "session_id" in tool_call
        assert "tool_name" in tool_call
        assert "input_json" in tool_call

    def test_multiple_sessions(self, sample_session_file, output_dir):
        """Test exporting multiple sessions."""
        # Create a second session file
        second_session = output_dir / "session2.jsonl"
        second_session.write_text(
            json.dumps(
                {
                    "type": "user",
                    "uuid": "user-101",
                    "sessionId": "session-456",
                    "timestamp": "2025-01-02T10:00:00.000Z",
                    "cwd": "/home/user/other",
                    "message": {"role": "user", "content": "Another session"},
                }
            )
            + "\n"
        )

        output_path = output_dir / "export.json"
        export_sessions_to_json([sample_session_file, second_session], output_path)

        with open(output_path) as f:
            data = json.load(f)

        assert len(data["tables"]["sessions"]) == 2


class TestExportStarSchemaToJson:
    """Tests for star schema JSON export."""

    def test_creates_directory_structure(self, sample_session_file, output_dir):
        """Test that directory structure is created."""
        star_output = output_dir / "star_export"

        conn = create_star_schema(":memory:")
        run_star_schema_etl(conn, sample_session_file, "test-project")
        create_semantic_model(conn)
        export_star_schema_to_json(conn, star_output)
        conn.close()

        assert star_output.exists()
        assert (star_output / "meta.json").exists()
        assert (star_output / "dimensions").is_dir()
        assert (star_output / "facts").is_dir()

    def test_creates_meta_json(self, sample_session_file, output_dir):
        """Test that meta.json is created with correct structure."""
        star_output = output_dir / "star_export"

        conn = create_star_schema(":memory:")
        run_star_schema_etl(conn, sample_session_file, "test-project")
        create_semantic_model(conn)
        export_star_schema_to_json(conn, star_output)
        conn.close()

        with open(star_output / "meta.json") as f:
            meta = json.load(f)

        assert meta["version"] == "1.0"
        assert meta["schema_type"] == "star"
        assert "exported_at" in meta
        assert "tables" in meta
        assert "relationships" in meta

    def test_meta_has_table_manifest(self, sample_session_file, output_dir):
        """Test that meta.json contains table manifest."""
        star_output = output_dir / "star_export"

        conn = create_star_schema(":memory:")
        run_star_schema_etl(conn, sample_session_file, "test-project")
        create_semantic_model(conn)
        export_star_schema_to_json(conn, star_output)
        conn.close()

        with open(star_output / "meta.json") as f:
            meta = json.load(f)

        assert "dimensions" in meta["tables"]
        assert "facts" in meta["tables"]
        assert len(meta["tables"]["dimensions"]) > 0
        assert len(meta["tables"]["facts"]) > 0

    def test_creates_dimension_files(self, sample_session_file, output_dir):
        """Test that dimension JSON files are created."""
        star_output = output_dir / "star_export"

        conn = create_star_schema(":memory:")
        run_star_schema_etl(conn, sample_session_file, "test-project")
        create_semantic_model(conn)
        export_star_schema_to_json(conn, star_output)
        conn.close()

        assert (star_output / "dimensions" / "dim_tool.json").exists()
        assert (star_output / "dimensions" / "dim_model.json").exists()
        assert (star_output / "dimensions" / "dim_session.json").exists()

    def test_creates_fact_files(self, sample_session_file, output_dir):
        """Test that fact JSON files are created."""
        star_output = output_dir / "star_export"

        conn = create_star_schema(":memory:")
        run_star_schema_etl(conn, sample_session_file, "test-project")
        create_semantic_model(conn)
        export_star_schema_to_json(conn, star_output)
        conn.close()

        assert (star_output / "facts" / "fact_messages.json").exists()
        assert (star_output / "facts" / "fact_tool_calls.json").exists()

    def test_dimension_file_is_valid_json(self, sample_session_file, output_dir):
        """Test that dimension files contain valid JSON."""
        star_output = output_dir / "star_export"

        conn = create_star_schema(":memory:")
        run_star_schema_etl(conn, sample_session_file, "test-project")
        create_semantic_model(conn)
        export_star_schema_to_json(conn, star_output)
        conn.close()

        with open(star_output / "dimensions" / "dim_tool.json") as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert len(data) > 0  # Should have Write tool

    def test_fact_file_is_valid_json(self, sample_session_file, output_dir):
        """Test that fact files contain valid JSON."""
        star_output = output_dir / "star_export"

        conn = create_star_schema(":memory:")
        run_star_schema_etl(conn, sample_session_file, "test-project")
        create_semantic_model(conn)
        export_star_schema_to_json(conn, star_output)
        conn.close()

        with open(star_output / "facts" / "fact_messages.json") as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert len(data) > 0  # Should have messages

    def test_dim_tool_has_expected_data(self, sample_session_file, output_dir):
        """Test that dim_tool contains Write tool."""
        star_output = output_dir / "star_export"

        conn = create_star_schema(":memory:")
        run_star_schema_etl(conn, sample_session_file, "test-project")
        create_semantic_model(conn)
        export_star_schema_to_json(conn, star_output)
        conn.close()

        with open(star_output / "dimensions" / "dim_tool.json") as f:
            data = json.load(f)

        tool_names = [t["tool_name"] for t in data]
        assert "Write" in tool_names

    def test_relationships_defined(self, sample_session_file, output_dir):
        """Test that relationships are defined in meta.json."""
        star_output = output_dir / "star_export"

        conn = create_star_schema(":memory:")
        run_star_schema_etl(conn, sample_session_file, "test-project")
        create_semantic_model(conn)
        export_star_schema_to_json(conn, star_output)
        conn.close()

        with open(star_output / "meta.json") as f:
            meta = json.load(f)

        relationships = meta["relationships"]
        assert len(relationships) > 0

        # Check structure of relationships
        rel = relationships[0]
        assert "from_table" in rel
        assert "from_column" in rel
        assert "to_table" in rel
        assert "to_column" in rel
