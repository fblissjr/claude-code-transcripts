"""Tests for search index generation."""

import json
import tempfile
from pathlib import Path

import pytest
from click.testing import CliRunner

from ccutils import (
    cli,
    extract_searchable_content,
    extract_snippet,
    generate_batch_html,
)


@pytest.fixture
def mock_projects_dir():
    """Create a mock ~/.claude/projects structure with test sessions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        projects_dir = Path(tmpdir)

        # Create project-a with sessions
        project_a = projects_dir / "-home-user-projects-project-a"
        project_a.mkdir(parents=True)

        session_a1 = project_a / "abc123.jsonl"
        session_a1.write_text(
            '{"type": "user", "timestamp": "2025-01-01T10:00:00.000Z", "message": {"role": "user", "content": "Hello from project A"}}\n'
            '{"type": "assistant", "timestamp": "2025-01-01T10:00:05.000Z", "message": {"role": "assistant", "content": [{"type": "text", "text": "Hi there! I can help you."}]}}\n'
        )

        yield projects_dir


@pytest.fixture
def output_dir():
    """Create a temporary output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestExtractSnippet:
    """Tests for snippet extraction."""

    def test_short_content_unchanged(self):
        """Test that short content is returned unchanged."""
        content = "This is short."
        snippet = extract_snippet(content, max_length=200)
        assert snippet == content

    def test_long_content_truncated(self):
        """Test that long content is truncated with ellipsis."""
        content = "x" * 500
        snippet = extract_snippet(content, max_length=200)
        assert len(snippet) <= 203  # 200 + "..."
        assert snippet.endswith("...")

    def test_centers_on_query_match(self):
        """Test that snippet centers around query match."""
        content = "A" * 100 + "TARGET" + "B" * 100
        snippet = extract_snippet(content, max_length=50, query="TARGET")
        assert "TARGET" in snippet
        assert "..." in snippet

    def test_query_not_found_returns_start(self):
        """Test that if query not found, returns start of content."""
        content = "x" * 500
        snippet = extract_snippet(content, max_length=50, query="NOTFOUND")
        assert snippet.startswith("x")
        assert snippet.endswith("...")


class TestExtractSearchableContent:
    """Tests for extracting searchable content from loglines."""

    def test_extracts_user_prompts(self):
        """Test that user prompts are extracted."""
        loglines = [
            {
                "type": "user",
                "timestamp": "2025-01-01T10:00:00.000Z",
                "message": {"content": "Build a hello world app"},
            }
        ]

        docs = extract_searchable_content(loglines, "project", "session")

        assert len(docs) >= 1
        user_docs = [d for d in docs if d["type"] == "user"]
        assert len(user_docs) == 1
        assert "hello world" in user_docs[0]["content"].lower()

    def test_extracts_assistant_text(self):
        """Test that assistant text blocks are extracted."""
        loglines = [
            {
                "type": "assistant",
                "timestamp": "2025-01-01T10:00:05.000Z",
                "message": {
                    "content": [{"type": "text", "text": "I'll create that for you."}]
                },
            }
        ]

        docs = extract_searchable_content(loglines, "project", "session")

        assert len(docs) >= 1
        assistant_docs = [d for d in docs if d["type"] == "assistant"]
        assert len(assistant_docs) == 1
        assert "create" in assistant_docs[0]["content"].lower()

    def test_truncates_tool_outputs(self):
        """Test that tool outputs are truncated at 500 chars."""
        long_output = "x" * 1000
        loglines = [
            {
                "type": "user",
                "timestamp": "2025-01-01T10:00:10.000Z",
                "message": {
                    "content": [{"type": "tool_result", "content": long_output}]
                },
            }
        ]

        docs = extract_searchable_content(loglines, "project", "session")

        # Find tool_result doc
        tool_docs = [d for d in docs if d["type"] == "tool_result"]
        assert len(tool_docs) == 1
        # Should be truncated
        assert len(tool_docs[0]["content"]) <= 500

    def test_indexes_tool_use_metadata(self):
        """Test that tool_use blocks capture name and input."""
        loglines = [
            {
                "type": "assistant",
                "timestamp": "2025-01-01T10:00:05.000Z",
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "name": "Bash",
                            "input": {
                                "command": "npm test",
                                "description": "Run tests",
                            },
                        }
                    ]
                },
            }
        ]

        docs = extract_searchable_content(loglines, "project", "session")

        tool_docs = [d for d in docs if d["type"] == "tool_use"]
        assert len(tool_docs) == 1
        assert "Bash" in tool_docs[0]["content"]
        assert "npm test" in tool_docs[0]["content"]

    def test_generates_correct_page_references(self):
        """Test that documents reference correct page numbers."""
        # Create 6 user prompts (spans 2 pages with PROMPTS_PER_PAGE=5)
        loglines = []
        for i in range(6):
            loglines.append(
                {
                    "type": "user",
                    "timestamp": f"2025-01-01T{10+i:02d}:00:00.000Z",
                    "message": {"content": f"Prompt {i+1}"},
                }
            )

        docs = extract_searchable_content(loglines, "project", "session")

        user_docs = [d for d in docs if d["type"] == "user"]
        # First 5 should reference page-001
        assert all("page-001" in d["page"] for d in user_docs[:5])
        # 6th should reference page-002
        assert "page-002" in user_docs[5]["page"]

    def test_includes_project_and_session(self):
        """Test that documents include project and session names."""
        loglines = [
            {
                "type": "user",
                "timestamp": "2025-01-01T10:00:00.000Z",
                "message": {"content": "Hello"},
            }
        ]

        docs = extract_searchable_content(loglines, "my-project", "abc123")

        assert docs[0]["project"] == "my-project"
        assert docs[0]["session"] == "abc123"

    def test_includes_anchor_for_linking(self):
        """Test that documents include anchor IDs for linking."""
        loglines = [
            {
                "type": "user",
                "timestamp": "2025-01-01T10:00:00.000Z",
                "message": {"content": "Hello"},
            }
        ]

        docs = extract_searchable_content(loglines, "project", "session")

        # Should have an anchor derived from timestamp
        assert "anchor" in docs[0]
        assert docs[0]["anchor"]  # Not empty


class TestGenerateSearchIndex:
    """Tests for full index generation."""

    def test_generates_search_index_file(self, mock_projects_dir, output_dir):
        """Test that search-index.js is generated."""
        generate_batch_html(mock_projects_dir, output_dir)

        index_path = output_dir / "search-index.js"
        assert index_path.exists()

    def test_search_index_is_valid_javascript(self, mock_projects_dir, output_dir):
        """Test that generated index is valid JavaScript variable assignment."""
        generate_batch_html(mock_projects_dir, output_dir)

        index_path = output_dir / "search-index.js"
        content = index_path.read_text()

        # Should be a JavaScript variable assignment
        assert content.startswith("var SEARCH_INDEX = ")
        assert content.rstrip().endswith(";")

        # Extract JSON and validate
        json_str = content.replace("var SEARCH_INDEX = ", "").rstrip(";")
        index_data = json.loads(json_str)

        assert "version" in index_data
        assert "documents" in index_data
        assert isinstance(index_data["documents"], list)

    def test_index_contains_session_content(self, mock_projects_dir, output_dir):
        """Test that index includes content from sessions."""
        generate_batch_html(mock_projects_dir, output_dir)

        index_path = output_dir / "search-index.js"
        content = index_path.read_text()
        json_str = content.replace("var SEARCH_INDEX = ", "").rstrip(";")
        index_data = json.loads(json_str)

        # Should have documents from the test session
        assert len(index_data["documents"]) > 0

        # Should include content from test session
        all_content = " ".join(d["content"] for d in index_data["documents"])
        assert "Hello from project A" in all_content or "project A" in all_content


class TestSearchIndexCliOptions:
    """Tests for search index CLI options."""

    def test_no_search_index_flag_skips_generation(self, mock_projects_dir, output_dir):
        """Test that --no-search-index skips index generation."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "all",
                "--source",
                str(mock_projects_dir),
                "--output",
                str(output_dir),
                "--no-search-index",
            ],
        )

        assert result.exit_code == 0
        # HTML should still be generated
        assert (output_dir / "index.html").exists()
        # But search index should NOT exist
        assert not (output_dir / "search-index.js").exists()

    def test_search_index_generated_by_default(self, mock_projects_dir, output_dir):
        """Test that search index is generated by default."""
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
        assert (output_dir / "search-index.js").exists()
