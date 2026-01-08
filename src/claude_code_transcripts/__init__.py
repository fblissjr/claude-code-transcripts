"""Convert Claude Code session JSON to a clean mobile-friendly HTML page with pagination."""

import json
import html
import os
import platform
import re
import shutil
import subprocess
import tempfile
import webbrowser
from datetime import datetime
from pathlib import Path

import click
from click_default_group import DefaultGroup
import duckdb
import httpx
from jinja2 import Environment, PackageLoader
import markdown
import questionary

# Set up Jinja2 environment
_jinja_env = Environment(
    loader=PackageLoader("claude_code_transcripts", "templates"),
    autoescape=True,
)

# Load macros template and expose macros
_macros_template = _jinja_env.get_template("macros.html")
_macros = _macros_template.module


def get_template(name):
    """Get a Jinja2 template by name."""
    return _jinja_env.get_template(name)


# Regex to match git commit output: [branch hash] message
COMMIT_PATTERN = re.compile(r"\[[\w\-/]+ ([a-f0-9]{7,})\] (.+?)(?:\n|$)")

# Regex to detect GitHub repo from git push output (e.g., github.com/owner/repo/pull/new/branch)
GITHUB_REPO_PATTERN = re.compile(
    r"github\.com/([a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+)/pull/new/"
)

PROMPTS_PER_PAGE = 5
LONG_TEXT_THRESHOLD = (
    300  # Characters - text blocks longer than this are shown in index
)


def extract_text_from_content(content):
    """Extract plain text from message content.

    Handles both string content (older format) and array content (newer format).

    Args:
        content: Either a string or a list of content blocks like
                 [{"type": "text", "text": "..."}, {"type": "image", ...}]

    Returns:
        The extracted text as a string, or empty string if no text found.
    """
    if isinstance(content, str):
        return content.strip()
    elif isinstance(content, list):
        # Extract text from content blocks of type "text"
        texts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text", "")
                if text:
                    texts.append(text)
        return " ".join(texts).strip()
    return ""


def extract_snippet(content, max_length=200, query=None):
    """Extract a relevant snippet from content.

    If query is provided, centers snippet around first match.

    Args:
        content: The text content to extract a snippet from
        max_length: Maximum length of the snippet
        query: Optional query to center the snippet around

    Returns:
        A snippet string, potentially with "..." added
    """
    if len(content) <= max_length:
        return content

    if query:
        # Find first occurrence of query (case-insensitive)
        pos = content.lower().find(query.lower())
        if pos != -1:
            # Center window around match
            start = max(0, pos - max_length // 2)
            end = min(len(content), start + max_length)
            # Adjust start if we're near the end
            if end == len(content):
                start = max(0, end - max_length)
            snippet = content[start:end]
            if start > 0:
                snippet = "..." + snippet
            if end < len(content):
                snippet = snippet + "..."
            return snippet

    # Default: first N characters
    return content[:max_length] + "..."


def extract_searchable_content(loglines, project_name, session_name):
    """Extract searchable documents from a session's loglines.

    Args:
        loglines: List of log entries from parse_session_file()
        project_name: Name of the project for document IDs
        session_name: Name of the session for document IDs

    Returns:
        List of document dicts for the search index, each containing:
        - id: Unique document ID
        - project: Project name
        - session: Session name
        - page: Page filename (e.g., "page-001.html")
        - anchor: Anchor ID for linking
        - type: Document type (user, assistant, tool_use, tool_result)
        - timestamp: ISO timestamp
        - content: Searchable text content
        - snippet: Short preview text
    """
    documents = []
    page_num = 1
    user_prompt_count = 0

    def make_anchor(timestamp):
        """Create anchor ID from timestamp."""
        if not timestamp:
            return ""
        # Convert to safe ID: msg-2025-01-01T10-00-00-000Z
        return "msg-" + timestamp.replace(":", "-").replace(".", "-")

    def add_document(doc_type, timestamp, content):
        """Helper to add a document to the list."""
        if not content or not content.strip():
            return
        documents.append(
            {
                "id": f"{project_name}/{session_name}/page-{page_num:03d}#{make_anchor(timestamp)}",
                "project": project_name,
                "session": session_name,
                "page": f"page-{page_num:03d}.html",
                "anchor": make_anchor(timestamp),
                "type": doc_type,
                "timestamp": timestamp,
                "content": content.strip(),
                "snippet": extract_snippet(content.strip()),
            }
        )

    for entry in loglines:
        log_type = entry.get("type")
        timestamp = entry.get("timestamp", "")
        message_data = entry.get("message", {})
        content = message_data.get("content", "")

        if log_type == "user":
            # Track page number based on user prompts
            user_prompt_count += 1
            if user_prompt_count > PROMPTS_PER_PAGE:
                page_num += 1
                user_prompt_count = 1

            # Handle different content formats
            if isinstance(content, str):
                add_document("user", timestamp, content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        block_type = block.get("type")
                        if block_type == "text":
                            add_document("user", timestamp, block.get("text", ""))
                        elif block_type == "tool_result":
                            # Tool results - truncate to 500 chars
                            result_content = block.get("content", "")
                            if isinstance(result_content, str):
                                truncated = result_content[:500]
                                add_document("tool_result", timestamp, truncated)

        elif log_type == "assistant":
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        block_type = block.get("type")
                        if block_type == "text":
                            add_document("assistant", timestamp, block.get("text", ""))
                        elif block_type == "tool_use":
                            # Index tool name and input
                            tool_name = block.get("name", "")
                            tool_input = block.get("input", {})
                            if isinstance(tool_input, dict):
                                input_str = json.dumps(tool_input)[:1000]
                            else:
                                input_str = str(tool_input)[:1000]
                            tool_content = f"{tool_name}: {input_str}"
                            add_document("tool_use", timestamp, tool_content)

    return documents


# Module-level variable for GitHub repo (set by generate_html)
_github_repo = None

# API constants
API_BASE_URL = "https://api.anthropic.com/v1"
ANTHROPIC_VERSION = "2023-06-01"


def get_session_summary(filepath, max_length=200):
    """Extract a human-readable summary from a session file.

    Supports both JSON and JSONL formats.
    Returns a summary string or "(no summary)" if none found.
    """
    filepath = Path(filepath)
    try:
        if filepath.suffix == ".jsonl":
            return _get_jsonl_summary(filepath, max_length)
        else:
            # For JSON files, try to get first user message
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            loglines = data.get("loglines", [])
            for entry in loglines:
                if entry.get("type") == "user":
                    msg = entry.get("message", {})
                    content = msg.get("content", "")
                    text = extract_text_from_content(content)
                    if text:
                        if len(text) > max_length:
                            return text[: max_length - 3] + "..."
                        return text
            return "(no summary)"
    except Exception:
        return "(no summary)"


def _get_jsonl_summary(filepath, max_length=200):
    """Extract summary from JSONL file."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    # First priority: summary type entries
                    if obj.get("type") == "summary" and obj.get("summary"):
                        summary = obj["summary"]
                        if len(summary) > max_length:
                            return summary[: max_length - 3] + "..."
                        return summary
                except json.JSONDecodeError:
                    continue

        # Second pass: find first non-meta user message
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if (
                        obj.get("type") == "user"
                        and not obj.get("isMeta")
                        and obj.get("message", {}).get("content")
                    ):
                        content = obj["message"]["content"]
                        text = extract_text_from_content(content)
                        if text and not text.startswith("<"):
                            if len(text) > max_length:
                                return text[: max_length - 3] + "..."
                            return text
                except json.JSONDecodeError:
                    continue
    except Exception:
        pass

    return "(no summary)"


def find_local_sessions(folder, limit=10):
    """Find recent JSONL session files in the given folder.

    Returns a list of (Path, summary) tuples sorted by modification time.
    Excludes agent files and warmup/empty sessions.
    """
    folder = Path(folder)
    if not folder.exists():
        return []

    results = []
    for f in folder.glob("**/*.jsonl"):
        if f.name.startswith("agent-"):
            continue
        summary = get_session_summary(f)
        # Skip boring/empty sessions
        if summary.lower() == "warmup" or summary == "(no summary)":
            continue
        results.append((f, summary))

    # Sort by modification time, most recent first
    results.sort(key=lambda x: x[0].stat().st_mtime, reverse=True)
    return results[:limit]


def get_project_display_name(folder_name):
    """Convert encoded folder name to readable project name.

    Claude Code stores projects in folders like:
    - -home-user-projects-myproject -> myproject
    - -mnt-c-Users-name-Projects-app -> app

    For nested paths under common roots (home, projects, code, Users, etc.),
    extracts the meaningful project portion.
    """
    # Common path prefixes to strip
    prefixes_to_strip = [
        "-home-",
        "-mnt-c-Users-",
        "-mnt-c-users-",
        "-Users-",
    ]

    name = folder_name
    for prefix in prefixes_to_strip:
        if name.lower().startswith(prefix.lower()):
            name = name[len(prefix) :]
            break

    # Split on dashes and find meaningful parts
    parts = name.split("-")

    # Common intermediate directories to skip
    skip_dirs = {"projects", "code", "repos", "src", "dev", "work", "documents"}

    # Find the first meaningful part (after skipping username and common dirs)
    meaningful_parts = []
    found_project = False

    for i, part in enumerate(parts):
        if not part:
            continue
        # Skip the first part if it looks like a username (before common dirs)
        if i == 0 and not found_project:
            # Check if next parts contain common dirs
            remaining = [p.lower() for p in parts[i + 1 :]]
            if any(d in remaining for d in skip_dirs):
                continue
        if part.lower() in skip_dirs:
            found_project = True
            continue
        meaningful_parts.append(part)
        found_project = True

    if meaningful_parts:
        return "-".join(meaningful_parts)

    # Fallback: return last non-empty part or original
    for part in reversed(parts):
        if part:
            return part
    return folder_name


def find_all_sessions(folder, include_agents=False):
    """Find all sessions in a Claude projects folder, grouped by project.

    Returns a list of project dicts, each containing:
    - name: display name for the project
    - path: Path to the project folder
    - sessions: list of session dicts with path, summary, mtime, size

    Sessions are sorted by modification time (most recent first) within each project.
    Projects are sorted by their most recent session.
    """
    folder = Path(folder)
    if not folder.exists():
        return []

    projects = {}

    for session_file in folder.glob("**/*.jsonl"):
        # Skip agent files unless requested
        if not include_agents and session_file.name.startswith("agent-"):
            continue

        # Get summary and skip boring sessions
        summary = get_session_summary(session_file)
        if summary.lower() == "warmup" or summary == "(no summary)":
            continue

        # Get project folder
        project_folder = session_file.parent
        project_key = project_folder.name

        if project_key not in projects:
            projects[project_key] = {
                "name": get_project_display_name(project_key),
                "path": project_folder,
                "sessions": [],
            }

        stat = session_file.stat()
        projects[project_key]["sessions"].append(
            {
                "path": session_file,
                "summary": summary,
                "mtime": stat.st_mtime,
                "size": stat.st_size,
            }
        )

    # Sort sessions within each project by mtime (most recent first)
    for project in projects.values():
        project["sessions"].sort(key=lambda s: s["mtime"], reverse=True)

    # Convert to list and sort projects by most recent session
    result = list(projects.values())
    result.sort(
        key=lambda p: p["sessions"][0]["mtime"] if p["sessions"] else 0, reverse=True
    )

    return result


def generate_batch_html(
    source_folder,
    output_dir,
    include_agents=False,
    progress_callback=None,
    no_search_index=False,
):
    """Generate HTML archive for all sessions in a Claude projects folder.

    Creates:
    - Master index.html listing all projects
    - Per-project directories with index.html listing sessions
    - Per-session directories with transcript pages
    - search-index.js for full-text search (unless no_search_index=True)

    Args:
        source_folder: Path to the Claude projects folder
        output_dir: Path for output archive
        include_agents: Whether to include agent-* session files
        progress_callback: Optional callback(project_name, session_name, current, total)
            called after each session is processed
        no_search_index: If True, skip generating the search index

    Returns statistics dict with total_projects, total_sessions, failed_sessions, output_dir.
    """
    source_folder = Path(source_folder)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all sessions
    projects = find_all_sessions(source_folder, include_agents=include_agents)

    # Calculate total for progress tracking
    total_session_count = sum(len(p["sessions"]) for p in projects)
    processed_count = 0
    successful_sessions = 0
    failed_sessions = []

    # Process each project
    for project in projects:
        project_dir = output_dir / project["name"]
        project_dir.mkdir(exist_ok=True)

        # Process each session
        for session in project["sessions"]:
            session_name = session["path"].stem
            session_dir = project_dir / session_name

            # Generate transcript HTML with error handling
            try:
                generate_html(session["path"], session_dir)
                successful_sessions += 1
            except Exception as e:
                failed_sessions.append(
                    {
                        "project": project["name"],
                        "session": session_name,
                        "error": str(e),
                    }
                )

            processed_count += 1

            # Call progress callback if provided
            if progress_callback:
                progress_callback(
                    project["name"], session_name, processed_count, total_session_count
                )

        # Generate project index
        _generate_project_index(project, project_dir)

    # Generate master index (with search UI if search index will be generated)
    has_search_index = not no_search_index
    _generate_master_index(projects, output_dir, has_search_index=has_search_index)

    # Generate search index (unless disabled)
    if has_search_index:
        _generate_search_index(projects, output_dir)

    return {
        "total_projects": len(projects),
        "total_sessions": successful_sessions,
        "failed_sessions": failed_sessions,
        "output_dir": output_dir,
    }


def _generate_project_index(project, output_dir):
    """Generate index.html for a single project."""
    template = get_template("project_index.html")

    # Format sessions for template
    sessions_data = []
    for session in project["sessions"]:
        mod_time = datetime.fromtimestamp(session["mtime"])
        sessions_data.append(
            {
                "name": session["path"].stem,
                "summary": session["summary"],
                "date": mod_time.strftime("%Y-%m-%d %H:%M"),
                "size_kb": session["size"] / 1024,
            }
        )

    html_content = template.render(
        project_name=project["name"],
        sessions=sessions_data,
        session_count=len(sessions_data),
        css=CSS,
        js=JS,
    )

    output_path = output_dir / "index.html"
    output_path.write_text(html_content, encoding="utf-8")


def _generate_master_index(projects, output_dir, has_search_index=False):
    """Generate master index.html listing all projects."""
    template = get_template("master_index.html")

    # Format projects for template
    projects_data = []
    total_sessions = 0

    for project in projects:
        session_count = len(project["sessions"])
        total_sessions += session_count

        # Get most recent session date
        if project["sessions"]:
            most_recent = datetime.fromtimestamp(project["sessions"][0]["mtime"])
            recent_date = most_recent.strftime("%Y-%m-%d")
        else:
            recent_date = "N/A"

        projects_data.append(
            {
                "name": project["name"],
                "session_count": session_count,
                "recent_date": recent_date,
            }
        )

    # Load global search JavaScript if search index is enabled
    global_search_js = ""
    if has_search_index:
        global_search_template = get_template("global_search.js")
        global_search_js = global_search_template.render()

    html_content = template.render(
        projects=projects_data,
        total_projects=len(projects),
        total_sessions=total_sessions,
        css=CSS,
        js=JS,
        has_search_index=has_search_index,
        global_search_js=global_search_js,
    )

    output_path = output_dir / "index.html"
    output_path.write_text(html_content, encoding="utf-8")


def _generate_search_index(projects, output_dir):
    """Generate search-index.js containing all searchable content.

    Creates a JavaScript file with a SEARCH_INDEX variable containing
    all indexed documents for client-side full-text search.

    Args:
        projects: List of project dicts from find_all_sessions()
        output_dir: Path to the output directory
    """
    all_documents = []

    for project in projects:
        project_name = project["name"]

        for session in project["sessions"]:
            session_name = session["path"].stem

            try:
                # Parse the session file
                data = parse_session_file(session["path"])
                loglines = data.get("loglines", [])

                # Extract searchable content
                documents = extract_searchable_content(
                    loglines, project_name, session_name
                )
                all_documents.extend(documents)
            except Exception:
                # Skip sessions that fail to parse
                continue

    # Build the index structure
    index_data = {
        "version": 1,
        "generated": datetime.now().astimezone().isoformat(),
        "documents": all_documents,
    }

    # Write as JavaScript variable assignment
    js_content = (
        "var SEARCH_INDEX = " + json.dumps(index_data, ensure_ascii=False) + ";"
    )
    output_path = output_dir / "search-index.js"
    output_path.write_text(js_content, encoding="utf-8")


# =============================================================================
# DuckDB Export Functions
# =============================================================================


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
            version VARCHAR
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
            has_thinking BOOLEAN
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


def export_session_to_duckdb(
    conn, session_path, project_name, include_thinking=False, truncate_output=2000
):
    """Export a single session to DuckDB.

    Args:
        conn: DuckDB connection
        session_path: Path to the JSONL session file
        project_name: Name of the project
        include_thinking: Whether to export thinking blocks
        truncate_output: Max characters for tool output (default 2000)
    """
    session_path = Path(session_path)
    session_id = None
    cwd = None
    git_branch = None
    version = None
    first_timestamp = None
    last_timestamp = None
    user_count = 0
    assistant_count = 0
    tool_use_count = 0

    # Maps to link tool_use to tool_result
    tool_use_map = (
        {}
    )  # tool_use_id -> {message_id, tool_name, input_json, input_summary, timestamp}
    thinking_id = 0

    with open(session_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            entry_type = entry.get("type")
            if entry_type not in ("user", "assistant"):
                continue

            # Extract metadata from first entry
            # Always use file stem as unique ID (agent files share parent sessionId)
            if session_id is None:
                session_id = session_path.stem
                cwd = entry.get("cwd")
                git_branch = entry.get("gitBranch")
                version = entry.get("version")

            uuid = entry.get("uuid", "")
            parent_uuid = entry.get("parentUuid")
            timestamp_str = entry.get("timestamp", "")
            message_data = entry.get("message", {})

            # Parse timestamp
            timestamp = None
            if timestamp_str:
                try:
                    timestamp = datetime.fromisoformat(
                        timestamp_str.replace("Z", "+00:00")
                    )
                    if first_timestamp is None:
                        first_timestamp = timestamp
                    last_timestamp = timestamp
                except ValueError:
                    pass

            # Extract content
            content = message_data.get("content", "")
            model = message_data.get("model")
            has_tool_use = False
            has_tool_result = False
            has_thinking = False
            text_content = ""

            if isinstance(content, str):
                text_content = content
            elif isinstance(content, list):
                text_parts = []
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    block_type = block.get("type")

                    if block_type == "text":
                        text_parts.append(block.get("text", ""))

                    elif block_type == "tool_use":
                        has_tool_use = True
                        tool_use_count += 1
                        tool_id = block.get("id", "")
                        tool_name = block.get("name", "")
                        tool_input = block.get("input", {})

                        # Create summary of input
                        if isinstance(tool_input, dict):
                            input_summary = json.dumps(tool_input)[:truncate_output]
                        else:
                            input_summary = str(tool_input)[:truncate_output]

                        tool_use_map[tool_id] = {
                            "message_id": uuid,
                            "tool_name": tool_name,
                            "input_json": json.dumps(tool_input),
                            "input_summary": input_summary,
                            "timestamp": timestamp,
                        }

                    elif block_type == "tool_result":
                        has_tool_result = True
                        tool_id = block.get("tool_use_id", "")
                        result_content = block.get("content", "")
                        if isinstance(result_content, str):
                            output_text = result_content[:truncate_output]
                        else:
                            output_text = str(result_content)[:truncate_output]

                        # Link to tool_use and insert
                        if tool_id in tool_use_map:
                            tool_info = tool_use_map[tool_id]
                            conn.execute(
                                """
                                INSERT INTO tool_calls (
                                    tool_use_id, session_id, message_id,
                                    result_message_id, tool_name, input_json,
                                    input_summary, output_text, timestamp
                                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                                [
                                    tool_id,
                                    session_id,
                                    tool_info["message_id"],
                                    uuid,
                                    tool_info["tool_name"],
                                    tool_info["input_json"],
                                    tool_info["input_summary"],
                                    output_text,
                                    tool_info["timestamp"],
                                ],
                            )

                    elif block_type == "thinking":
                        has_thinking = True
                        if include_thinking:
                            thinking_text = block.get("thinking", "")
                            thinking_id += 1
                            conn.execute(
                                """
                                INSERT INTO thinking (id, session_id, message_id, thinking_text, timestamp)
                                VALUES (?, ?, ?, ?, ?)
                            """,
                                [
                                    thinking_id,
                                    session_id,
                                    uuid,
                                    thinking_text,
                                    timestamp,
                                ],
                            )

                text_content = " ".join(text_parts)

            # Count messages
            if entry_type == "user":
                user_count += 1
            else:
                assistant_count += 1

            # Insert message
            conn.execute(
                """
                INSERT INTO messages (
                    id, session_id, parent_id, type, timestamp, model,
                    content, content_json, has_tool_use, has_tool_result, has_thinking
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                [
                    uuid,
                    session_id,
                    parent_uuid,
                    entry_type,
                    timestamp,
                    model,
                    text_content,
                    json.dumps(content) if isinstance(content, list) else None,
                    has_tool_use,
                    has_tool_result,
                    has_thinking,
                ],
            )

    # Insert session metadata
    if session_id:
        conn.execute(
            """
            INSERT INTO sessions (
                session_id, project_path, project_name, first_timestamp, last_timestamp,
                message_count, user_message_count, assistant_message_count,
                tool_use_count, cwd, git_branch, version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            [
                session_id,
                str(session_path),
                project_name,
                first_timestamp,
                last_timestamp,
                user_count + assistant_count,
                user_count,
                assistant_count,
                tool_use_count,
                cwd,
                git_branch,
                version,
            ],
        )


def generate_duckdb_archive(
    source_folder,
    output_dir,
    include_agents=False,
    include_thinking=False,
    truncate_output=2000,
    progress_callback=None,
):
    """Generate DuckDB archive for all sessions.

    Args:
        source_folder: Path to Claude projects folder
        output_dir: Path for output
        include_agents: Whether to include agent sessions
        include_thinking: Whether to include thinking blocks
        truncate_output: Max chars for tool output
        progress_callback: Optional progress callback

    Returns:
        dict with statistics
    """
    source_folder = Path(source_folder)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    db_path = output_dir / "archive.duckdb"
    conn = create_duckdb_schema(db_path)

    projects = find_all_sessions(source_folder, include_agents=include_agents)

    total_session_count = sum(len(p["sessions"]) for p in projects)
    processed_count = 0
    successful_sessions = 0
    failed_sessions = []

    for project in projects:
        project_name = project["name"]

        for session in project["sessions"]:
            session_path = session["path"]

            try:
                export_session_to_duckdb(
                    conn,
                    session_path,
                    project_name,
                    include_thinking=include_thinking,
                    truncate_output=truncate_output,
                )
                successful_sessions += 1
            except Exception as e:
                failed_sessions.append(
                    {
                        "project": project_name,
                        "session": session_path.stem,
                        "error": str(e),
                    }
                )

            processed_count += 1
            if progress_callback:
                progress_callback(
                    project_name,
                    session_path.stem,
                    processed_count,
                    total_session_count,
                )

    conn.close()

    return {
        "total_projects": len(projects),
        "total_sessions": successful_sessions,
        "failed_sessions": failed_sessions,
        "output_dir": output_dir,
        "db_path": db_path,
    }


# =============================================================================
# Star Schema DuckDB Implementation
# =============================================================================

import hashlib


def generate_dimension_key(*natural_keys):
    """Generate a dimension key from natural key(s) using MD5 hash.

    This creates a consistent surrogate key for dimension tables based on
    the natural business key(s). Using a hash allows for:
    - Deterministic key generation (same input = same key)
    - No sequence coordination needed
    - Natural deduplication in dimensions

    Args:
        *natural_keys: One or more values that form the natural key.
                       Multiple values are joined with '|' separator.

    Returns:
        32-character lowercase hex string (MD5 hash)
    """
    # Convert all values to strings, handle None
    key_parts = [str(k) if k is not None else "NULL" for k in natural_keys]
    combined = "|".join(key_parts)
    return hashlib.md5(combined.encode("utf-8")).hexdigest()


# Tool category mapping for dim_tool
TOOL_CATEGORIES = {
    # File operations
    "Read": "file_operations",
    "Write": "file_operations",
    "Edit": "file_operations",
    "MultiEdit": "file_operations",
    "NotebookEdit": "file_operations",
    "Glob": "file_operations",
    # Search tools
    "Grep": "search",
    "WebSearch": "search",
    # Execution tools
    "Bash": "execution",
    "BashOutput": "execution",
    "KillShell": "execution",
    # Web tools
    "WebFetch": "web",
    # Task management
    "Task": "task_management",
    "TodoWrite": "task_management",
    # Planning tools
    "EnterPlanMode": "planning",
    "ExitPlanMode": "planning",
    # Other
    "Skill": "other",
    "SlashCommand": "other",
    "AskUserQuestion": "interaction",
}


def get_tool_category(tool_name):
    """Get the category for a tool name."""
    return TOOL_CATEGORIES.get(tool_name, "other")


def get_model_family(model_name):
    """Extract the model family from a model name.

    Args:
        model_name: Full model name like 'claude-opus-4-5-20251101'

    Returns:
        Model family: 'opus', 'sonnet', 'haiku', or 'unknown'
    """
    if model_name is None:
        return "unknown"
    model_lower = model_name.lower()
    if "opus" in model_lower:
        return "opus"
    elif "sonnet" in model_lower:
        return "sonnet"
    elif "haiku" in model_lower:
        return "haiku"
    return "unknown"


def get_time_of_day(hour):
    """Get time of day label from hour.

    Args:
        hour: Hour of day (0-23)

    Returns:
        Time of day label: 'night', 'morning', 'afternoon', 'evening'
    """
    if hour < 6:
        return "night"
    elif hour < 12:
        return "morning"
    elif hour < 18:
        return "afternoon"
    else:
        return "evening"


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
    # Staging Table - holds raw extracted data before ETL transforms
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
    # Dimension Tables
    # =========================================================================

    # dim_tool - Tool dimension with category classification
    conn.execute(
        """
        CREATE OR REPLACE TABLE dim_tool (
            tool_key VARCHAR,
            tool_name VARCHAR,
            tool_category VARCHAR
        )
    """
    )

    # dim_model - Model dimension with family classification
    conn.execute(
        """
        CREATE OR REPLACE TABLE dim_model (
            model_key VARCHAR,
            model_name VARCHAR,
            model_family VARCHAR
        )
    """
    )

    # dim_project - Project dimension
    conn.execute(
        """
        CREATE OR REPLACE TABLE dim_project (
            project_key VARCHAR,
            project_path VARCHAR,
            project_name VARCHAR
        )
    """
    )

    # dim_session - Session dimension with metadata
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
            last_timestamp TIMESTAMP
        )
    """
    )

    # dim_date - Standard date dimension
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

    # dim_time - Time of day dimension (granularity: minute)
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

    # dim_message_type - User vs Assistant dimension
    conn.execute(
        """
        CREATE OR REPLACE TABLE dim_message_type (
            message_type_key VARCHAR,
            message_type VARCHAR
        )
    """
    )

    # dim_content_block_type - Types of content blocks
    conn.execute(
        """
        CREATE OR REPLACE TABLE dim_content_block_type (
            content_block_type_key VARCHAR,
            block_type VARCHAR
        )
    """
    )

    # =========================================================================
    # Fact Tables
    # =========================================================================

    # fact_messages - One row per message
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
            content_text TEXT,
            content_json JSON
        )
    """
    )

    # fact_content_blocks - One row per content block within a message
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

    # fact_tool_calls - One row per tool invocation (linked to result)
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

    # fact_session_summary - Aggregate facts per session
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
    # Enhanced Granular Dimensions
    # =========================================================================

    # dim_file - File dimension for file operations
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

    # dim_programming_language - Language dimension for code blocks
    conn.execute(
        """
        CREATE OR REPLACE TABLE dim_programming_language (
            language_key VARCHAR,
            language_name VARCHAR,
            file_extensions VARCHAR
        )
    """
    )

    # dim_error_type - Error classification dimension
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
    # Enhanced Granular Fact Tables
    # =========================================================================

    # fact_file_operations - File-level tool interactions
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

    # fact_code_blocks - Code snippets extracted from messages
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

    # fact_errors - Error tracking for tool calls
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

    # Pre-populate dim_message_type with known values
    for msg_type in ["user", "assistant"]:
        key = generate_dimension_key(msg_type)
        conn.execute(
            "INSERT INTO dim_message_type (message_type_key, message_type) VALUES (?, ?)",
            [key, msg_type],
        )

    # Pre-populate dim_content_block_type with known values
    for block_type in ["text", "tool_use", "tool_result", "thinking", "image"]:
        key = generate_dimension_key(block_type)
        conn.execute(
            "INSERT INTO dim_content_block_type (content_block_type_key, block_type) VALUES (?, ?)",
            [key, block_type],
        )

    return conn


# =============================================================================
# Enhanced ETL Helper Functions
# =============================================================================

# Language detection patterns for code blocks
LANGUAGE_EXTENSIONS = {
    "python": [".py", ".pyw", ".pyi"],
    "javascript": [".js", ".jsx", ".mjs"],
    "typescript": [".ts", ".tsx"],
    "java": [".java"],
    "c": [".c", ".h"],
    "cpp": [".cpp", ".cc", ".cxx", ".hpp", ".hh"],
    "csharp": [".cs"],
    "go": [".go"],
    "rust": [".rs"],
    "ruby": [".rb"],
    "php": [".php"],
    "swift": [".swift"],
    "kotlin": [".kt", ".kts"],
    "scala": [".scala"],
    "r": [".r", ".R"],
    "sql": [".sql"],
    "html": [".html", ".htm"],
    "css": [".css", ".scss", ".sass", ".less"],
    "json": [".json"],
    "yaml": [".yaml", ".yml"],
    "xml": [".xml"],
    "markdown": [".md", ".markdown"],
    "shell": [".sh", ".bash", ".zsh"],
    "powershell": [".ps1", ".psm1"],
    "dockerfile": ["Dockerfile"],
    "toml": [".toml"],
}

# Code block regex pattern: ```language\ncode\n```
import re

CODE_BLOCK_PATTERN = re.compile(r"```(\w*)\n(.*?)```", re.DOTALL)


def extract_file_info(file_path):
    """Extract file information from a file path.

    Args:
        file_path: Full path to a file

    Returns:
        dict with file_key, file_path, file_name, file_extension, directory_path
    """
    if not file_path:
        return None

    path = Path(file_path)
    file_name = path.name
    file_extension = path.suffix if path.suffix else ""
    directory_path = str(path.parent)

    return {
        "file_key": generate_dimension_key(file_path),
        "file_path": file_path,
        "file_name": file_name,
        "file_extension": file_extension,
        "directory_path": directory_path,
    }


def detect_language_from_extension(file_path):
    """Detect programming language from file extension.

    Args:
        file_path: Path to a file

    Returns:
        Language name or 'unknown'
    """
    if not file_path:
        return "unknown"

    path = Path(file_path)
    ext = path.suffix.lower()
    name = path.name

    # Check for Dockerfile (no extension)
    if name == "Dockerfile":
        return "dockerfile"

    for lang, extensions in LANGUAGE_EXTENSIONS.items():
        if ext in extensions:
            return lang

    return "unknown"


def detect_language_from_hint(hint):
    """Detect programming language from code block hint.

    Args:
        hint: Language hint from code block (e.g., 'python', 'js')

    Returns:
        Normalized language name or 'unknown'
    """
    if not hint:
        return "unknown"

    hint = hint.lower().strip()

    # Direct matches
    if hint in LANGUAGE_EXTENSIONS:
        return hint

    # Common aliases
    aliases = {
        "py": "python",
        "js": "javascript",
        "ts": "typescript",
        "rb": "ruby",
        "sh": "shell",
        "bash": "shell",
        "zsh": "shell",
        "yml": "yaml",
        "md": "markdown",
        "c++": "cpp",
        "c#": "csharp",
        "jsx": "javascript",
        "tsx": "typescript",
    }

    return aliases.get(hint, hint if hint else "unknown")


def extract_code_blocks(text):
    """Extract code blocks from text content.

    Args:
        text: Text that may contain markdown code blocks

    Returns:
        List of dicts with language, code, line_count, char_count
    """
    if not text:
        return []

    blocks = []
    for match in CODE_BLOCK_PATTERN.finditer(text):
        hint = match.group(1)
        code = match.group(2)

        language = detect_language_from_hint(hint)
        line_count = code.count("\n") + 1 if code.strip() else 0
        char_count = len(code)

        blocks.append(
            {
                "language": language,
                "code": code,
                "line_count": line_count,
                "char_count": char_count,
            }
        )

    return blocks


def estimate_tokens(text):
    """Estimate token count for text.

    Uses a simple heuristic: ~1.3 tokens per word for English text.
    This is a rough approximation that works reasonably well for most content.

    Args:
        text: Text to estimate tokens for

    Returns:
        Estimated token count
    """
    if not text:
        return 0

    # Count words (split on whitespace)
    words = len(text.split())

    # Estimate tokens (roughly 1.3x words for English, more for code)
    # Code tends to have more tokens per "word" due to punctuation
    has_code = "```" in text or "def " in text or "function " in text
    multiplier = 1.5 if has_code else 1.3

    return int(words * multiplier)


def count_words(text):
    """Count words in text.

    Args:
        text: Text to count words in

    Returns:
        Word count
    """
    if not text:
        return 0
    return len(text.split())


def get_operation_type(tool_name):
    """Map tool name to file operation type.

    Args:
        tool_name: Name of the tool

    Returns:
        Operation type: 'read', 'write', 'edit', 'search', or 'other'
    """
    tool_lower = tool_name.lower() if tool_name else ""

    if tool_lower == "read":
        return "read"
    elif tool_lower == "write":
        return "write"
    elif tool_lower in ("edit", "multiedit"):
        return "edit"
    elif tool_lower == "glob":
        return "list"
    elif tool_lower == "grep":
        return "search"
    else:
        return "other"


def extract_file_path_from_tool(tool_name, tool_input):
    """Extract file path from tool input.

    Args:
        tool_name: Name of the tool
        tool_input: Tool input dictionary

    Returns:
        File path string or None
    """
    if not isinstance(tool_input, dict):
        return None

    # Common file path parameter names
    path_keys = ["file_path", "path", "filepath", "notebook_path"]

    for key in path_keys:
        if key in tool_input:
            return tool_input[key]

    return None


def run_star_schema_etl(
    conn, session_path, project_name, include_thinking=False, truncate_output=2000
):
    """Run ETL to populate star schema from a session file.

    This function:
    1. Extracts raw data from the session file
    2. Transforms and loads dimension tables (with deduplication)
    3. Transforms and loads fact tables

    Args:
        conn: DuckDB connection
        session_path: Path to the JSONL session file
        project_name: Name of the project
        include_thinking: Whether to include thinking blocks
        truncate_output: Max characters for tool output (default 2000)
    """
    session_path = Path(session_path)

    # ==========================================================================
    # Phase 1: Extract - Parse session file and collect raw data
    # ==========================================================================
    session_id = session_path.stem
    session_key = generate_dimension_key(session_id)
    project_path = str(session_path.parent)
    project_key = generate_dimension_key(project_path)

    # Session metadata (from first entry)
    cwd = None
    git_branch = None
    version = None
    first_timestamp = None
    last_timestamp = None

    # Counters
    user_count = 0
    assistant_count = 0
    total_content_blocks = 0
    thinking_count = 0

    # Tracking structures
    messages_data = []
    content_blocks_data = []
    tool_use_map = {}  # tool_use_id -> tool info
    tool_calls_data = []
    models_seen = set()
    tools_seen = set()
    dates_seen = set()  # Set of date_key integers

    # Enhanced tracking structures
    files_seen = {}  # file_path -> file_info dict
    file_operations_data = []
    code_blocks_data = []
    errors_data = []
    languages_seen = set()

    with open(session_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            entry_type = entry.get("type")
            if entry_type not in ("user", "assistant"):
                continue

            # Extract metadata from first entry
            if cwd is None:
                cwd = entry.get("cwd")
                git_branch = entry.get("gitBranch")
                version = entry.get("version")

            message_id = entry.get("uuid", "")
            parent_id = entry.get("parentUuid")
            timestamp_str = entry.get("timestamp", "")
            message_data = entry.get("message", {})
            model = message_data.get("model")

            # Parse timestamp
            timestamp = None
            date_key = None
            time_key = None
            if timestamp_str:
                try:
                    timestamp = datetime.fromisoformat(
                        timestamp_str.replace("Z", "+00:00")
                    )
                    if first_timestamp is None:
                        first_timestamp = timestamp
                    last_timestamp = timestamp

                    # Generate date and time keys
                    date_key = int(timestamp.strftime("%Y%m%d"))
                    time_key = int(timestamp.strftime("%H%M"))
                    dates_seen.add(date_key)
                except (ValueError, TypeError):
                    pass

            # Track model
            if model:
                models_seen.add(model)

            # Process content
            content = message_data.get("content", "")
            has_tool_use = False
            has_tool_result = False
            has_thinking = False
            text_content = ""
            content_json = json.dumps(content)
            content_block_count = 0

            if isinstance(content, str):
                text_content = content
                content_block_count = 1
                # Single text block
                block_id = f"{message_id}-0"
                content_blocks_data.append(
                    {
                        "content_block_id": block_id,
                        "message_id": message_id,
                        "session_key": session_key,
                        "content_block_type_key": generate_dimension_key("text"),
                        "date_key": date_key,
                        "time_key": time_key,
                        "block_index": 0,
                        "content_length": len(content),
                        "content_text": content[:truncate_output] if content else "",
                        "content_json": json.dumps({"type": "text", "text": content}),
                    }
                )
                total_content_blocks += 1

            elif isinstance(content, list):
                texts = []
                for idx, block in enumerate(content):
                    if not isinstance(block, dict):
                        continue

                    block_type = block.get("type")
                    content_block_count += 1

                    # Determine if we should track this block
                    should_track = True
                    if block_type == "thinking" and not include_thinking:
                        should_track = False

                    if block_type == "text":
                        text = block.get("text", "")
                        texts.append(text)
                        if should_track:
                            block_id = f"{message_id}-{idx}"
                            content_blocks_data.append(
                                {
                                    "content_block_id": block_id,
                                    "message_id": message_id,
                                    "session_key": session_key,
                                    "content_block_type_key": generate_dimension_key(
                                        "text"
                                    ),
                                    "date_key": date_key,
                                    "time_key": time_key,
                                    "block_index": idx,
                                    "content_length": len(text),
                                    "content_text": (
                                        text[:truncate_output] if text else ""
                                    ),
                                    "content_json": json.dumps(block),
                                }
                            )
                            total_content_blocks += 1

                    elif block_type == "tool_use":
                        has_tool_use = True
                        tool_use_id = block.get("id")
                        tool_name = block.get("name", "unknown")
                        tool_input = block.get("input", {})
                        tools_seen.add(tool_name)

                        # Store for later linking
                        input_json = json.dumps(tool_input)
                        input_summary = input_json[:truncate_output]
                        tool_use_map[tool_use_id] = {
                            "message_id": message_id,
                            "tool_name": tool_name,
                            "tool_key": generate_dimension_key(tool_name),
                            "input_json": input_json,
                            "input_summary": input_summary,
                            "input_char_count": len(input_json),
                            "timestamp": timestamp,
                            "date_key": date_key,
                            "time_key": time_key,
                        }

                        # Track file operations for file-related tools
                        file_path = extract_file_path_from_tool(tool_name, tool_input)
                        if file_path:
                            file_info = extract_file_info(file_path)
                            if file_info and file_path not in files_seen:
                                files_seen[file_path] = file_info

                            # Track the file operation
                            operation_type = get_operation_type(tool_name)
                            file_content = tool_input.get("content", "")
                            file_size = (
                                len(file_content)
                                if isinstance(file_content, str)
                                else 0
                            )

                            file_operations_data.append(
                                {
                                    "file_operation_id": f"{tool_use_id}-file",
                                    "tool_call_id": tool_use_id,
                                    "session_key": session_key,
                                    "file_key": (
                                        file_info["file_key"] if file_info else None
                                    ),
                                    "tool_key": generate_dimension_key(tool_name),
                                    "date_key": date_key,
                                    "time_key": time_key,
                                    "operation_type": operation_type,
                                    "file_size_chars": file_size,
                                    "timestamp": timestamp,
                                }
                            )

                        if should_track:
                            block_id = f"{message_id}-{idx}"
                            content_blocks_data.append(
                                {
                                    "content_block_id": block_id,
                                    "message_id": message_id,
                                    "session_key": session_key,
                                    "content_block_type_key": generate_dimension_key(
                                        "tool_use"
                                    ),
                                    "date_key": date_key,
                                    "time_key": time_key,
                                    "block_index": idx,
                                    "content_length": len(input_json),
                                    "content_text": input_summary,
                                    "content_json": json.dumps(block),
                                }
                            )
                            total_content_blocks += 1

                    elif block_type == "tool_result":
                        has_tool_result = True
                        tool_use_id = block.get("tool_use_id")
                        result_content = block.get("content", "")
                        is_error = block.get("is_error", False)

                        if isinstance(result_content, list):
                            result_text = " ".join(
                                str(item.get("text", ""))
                                for item in result_content
                                if isinstance(item, dict)
                            )
                        else:
                            result_text = str(result_content)

                        output_text = result_text[:truncate_output]
                        output_char_count = len(result_text)

                        # Link to tool_use and create tool call record
                        if tool_use_id and tool_use_id in tool_use_map:
                            tool_info = tool_use_map[tool_use_id]
                            tool_calls_data.append(
                                {
                                    "tool_call_id": tool_use_id,
                                    "session_key": session_key,
                                    "tool_key": tool_info["tool_key"],
                                    "date_key": tool_info["date_key"],
                                    "time_key": tool_info["time_key"],
                                    "invoke_message_id": tool_info["message_id"],
                                    "result_message_id": message_id,
                                    "timestamp": tool_info["timestamp"],
                                    "input_char_count": tool_info["input_char_count"],
                                    "output_char_count": output_char_count,
                                    "is_error": is_error,
                                    "input_json": tool_info["input_json"],
                                    "input_summary": tool_info["input_summary"],
                                    "output_text": output_text,
                                }
                            )

                            # Track errors
                            if is_error:
                                errors_data.append(
                                    {
                                        "error_id": f"{tool_use_id}-error",
                                        "tool_call_id": tool_use_id,
                                        "session_key": session_key,
                                        "tool_key": tool_info["tool_key"],
                                        "error_type_key": generate_dimension_key(
                                            "tool_error"
                                        ),
                                        "date_key": tool_info["date_key"],
                                        "time_key": tool_info["time_key"],
                                        "error_message": output_text,
                                        "timestamp": tool_info["timestamp"],
                                    }
                                )

                        if should_track:
                            block_id = f"{message_id}-{idx}"
                            content_blocks_data.append(
                                {
                                    "content_block_id": block_id,
                                    "message_id": message_id,
                                    "session_key": session_key,
                                    "content_block_type_key": generate_dimension_key(
                                        "tool_result"
                                    ),
                                    "date_key": date_key,
                                    "time_key": time_key,
                                    "block_index": idx,
                                    "content_length": output_char_count,
                                    "content_text": output_text,
                                    "content_json": json.dumps(block),
                                }
                            )
                            total_content_blocks += 1

                    elif block_type == "thinking":
                        has_thinking = True
                        thinking_count += 1
                        thinking_text = block.get("thinking", "")

                        if should_track:
                            block_id = f"{message_id}-{idx}"
                            content_blocks_data.append(
                                {
                                    "content_block_id": block_id,
                                    "message_id": message_id,
                                    "session_key": session_key,
                                    "content_block_type_key": generate_dimension_key(
                                        "thinking"
                                    ),
                                    "date_key": date_key,
                                    "time_key": time_key,
                                    "block_index": idx,
                                    "content_length": len(thinking_text),
                                    "content_text": (
                                        thinking_text[:truncate_output]
                                        if thinking_text
                                        else ""
                                    ),
                                    "content_json": json.dumps(block),
                                }
                            )
                            total_content_blocks += 1

                    elif block_type == "image":
                        if should_track:
                            block_id = f"{message_id}-{idx}"
                            content_blocks_data.append(
                                {
                                    "content_block_id": block_id,
                                    "message_id": message_id,
                                    "session_key": session_key,
                                    "content_block_type_key": generate_dimension_key(
                                        "image"
                                    ),
                                    "date_key": date_key,
                                    "time_key": time_key,
                                    "block_index": idx,
                                    "content_length": 0,
                                    "content_text": "[image]",
                                    "content_json": json.dumps(
                                        {"type": "image", "note": "content omitted"}
                                    ),
                                }
                            )
                            total_content_blocks += 1

                text_content = " ".join(texts)

            # Extract code blocks from text content
            if text_content:
                extracted_blocks = extract_code_blocks(text_content)
                for cb_idx, cb in enumerate(extracted_blocks):
                    language = cb["language"]
                    languages_seen.add(language)
                    code_blocks_data.append(
                        {
                            "code_block_id": f"{message_id}-code-{cb_idx}",
                            "message_id": message_id,
                            "session_key": session_key,
                            "language_key": generate_dimension_key(language),
                            "date_key": date_key,
                            "time_key": time_key,
                            "block_index": cb_idx,
                            "line_count": cb["line_count"],
                            "char_count": cb["char_count"],
                            "code_text": cb["code"][:truncate_output],
                        }
                    )

            # Update counters
            if entry_type == "user":
                user_count += 1
            else:
                assistant_count += 1

            # Calculate token and word counts
            word_cnt = count_words(text_content)
            token_est = estimate_tokens(text_content)

            # Build message record
            message_type_key = generate_dimension_key(entry_type)
            model_key = generate_dimension_key(model) if model else None

            messages_data.append(
                {
                    "message_id": message_id,
                    "session_key": session_key,
                    "project_key": project_key,
                    "message_type_key": message_type_key,
                    "model_key": model_key,
                    "date_key": date_key,
                    "time_key": time_key,
                    "parent_message_id": parent_id,
                    "timestamp": timestamp,
                    "content_length": len(text_content),
                    "content_block_count": content_block_count,
                    "has_tool_use": has_tool_use,
                    "has_tool_result": has_tool_result,
                    "has_thinking": has_thinking,
                    "word_count": word_cnt,
                    "estimated_tokens": token_est,
                    "content_text": (
                        text_content[:truncate_output] if text_content else ""
                    ),
                    "content_json": content_json,
                }
            )

    # ==========================================================================
    # Phase 2: Transform & Load - Populate dimensions and facts
    # ==========================================================================

    # Load dim_project (upsert pattern - check if exists first)
    existing = conn.execute(
        "SELECT 1 FROM dim_project WHERE project_key = ?", [project_key]
    ).fetchone()
    if not existing:
        conn.execute(
            """INSERT INTO dim_project (project_key, project_path, project_name)
               VALUES (?, ?, ?)""",
            [project_key, project_path, project_name],
        )

    # Load dim_session
    existing = conn.execute(
        "SELECT 1 FROM dim_session WHERE session_key = ?", [session_key]
    ).fetchone()
    if not existing:
        conn.execute(
            """INSERT INTO dim_session
               (session_key, session_id, project_key, cwd, git_branch, version,
                first_timestamp, last_timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                session_key,
                session_id,
                project_key,
                cwd,
                git_branch,
                version,
                first_timestamp,
                last_timestamp,
            ],
        )

    # Load dim_tool
    for tool_name in tools_seen:
        tool_key = generate_dimension_key(tool_name)
        existing = conn.execute(
            "SELECT 1 FROM dim_tool WHERE tool_key = ?", [tool_key]
        ).fetchone()
        if not existing:
            category = get_tool_category(tool_name)
            conn.execute(
                """INSERT INTO dim_tool (tool_key, tool_name, tool_category)
                   VALUES (?, ?, ?)""",
                [tool_key, tool_name, category],
            )

    # Load dim_model
    for model_name in models_seen:
        model_key = generate_dimension_key(model_name)
        existing = conn.execute(
            "SELECT 1 FROM dim_model WHERE model_key = ?", [model_key]
        ).fetchone()
        if not existing:
            family = get_model_family(model_name)
            conn.execute(
                """INSERT INTO dim_model (model_key, model_name, model_family)
                   VALUES (?, ?, ?)""",
                [model_key, model_name, family],
            )

    # Load dim_date
    day_names = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    month_names = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]

    for date_key in dates_seen:
        existing = conn.execute(
            "SELECT 1 FROM dim_date WHERE date_key = ?", [date_key]
        ).fetchone()
        if not existing:
            # Parse date_key back to date
            year = date_key // 10000
            month = (date_key // 100) % 100
            day = date_key % 100
            try:
                full_date = datetime(year, month, day)
                day_of_week = full_date.weekday()
                quarter = (month - 1) // 3 + 1
                is_weekend = day_of_week >= 5

                conn.execute(
                    """INSERT INTO dim_date
                       (date_key, full_date, year, month, day, day_of_week,
                        day_name, month_name, quarter, is_weekend)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    [
                        date_key,
                        full_date.date(),
                        year,
                        month,
                        day,
                        day_of_week,
                        day_names[day_of_week],
                        month_names[month - 1],
                        quarter,
                        is_weekend,
                    ],
                )
            except ValueError:
                pass

    # Load dim_time (for all unique times seen)
    times_seen = set()
    for msg in messages_data:
        if msg["time_key"]:
            times_seen.add(msg["time_key"])

    for time_key in times_seen:
        existing = conn.execute(
            "SELECT 1 FROM dim_time WHERE time_key = ?", [time_key]
        ).fetchone()
        if not existing:
            hour = time_key // 100
            minute = time_key % 100
            time_of_day = get_time_of_day(hour)
            conn.execute(
                """INSERT INTO dim_time (time_key, hour, minute, time_of_day)
                   VALUES (?, ?, ?, ?)""",
                [time_key, hour, minute, time_of_day],
            )

    # Load fact_messages
    for msg in messages_data:
        conn.execute(
            """INSERT INTO fact_messages
               (message_id, session_key, project_key, message_type_key, model_key,
                date_key, time_key, parent_message_id, timestamp, content_length,
                content_block_count, has_tool_use, has_tool_result, has_thinking,
                word_count, estimated_tokens, content_text, content_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                msg["message_id"],
                msg["session_key"],
                msg["project_key"],
                msg["message_type_key"],
                msg["model_key"],
                msg["date_key"],
                msg["time_key"],
                msg["parent_message_id"],
                msg["timestamp"],
                msg["content_length"],
                msg["content_block_count"],
                msg["has_tool_use"],
                msg["has_tool_result"],
                msg["has_thinking"],
                msg["word_count"],
                msg["estimated_tokens"],
                msg["content_text"],
                msg["content_json"],
            ],
        )

    # Load fact_content_blocks
    for block in content_blocks_data:
        conn.execute(
            """INSERT INTO fact_content_blocks
               (content_block_id, message_id, session_key, content_block_type_key,
                date_key, time_key, block_index, content_length, content_text, content_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                block["content_block_id"],
                block["message_id"],
                block["session_key"],
                block["content_block_type_key"],
                block["date_key"],
                block["time_key"],
                block["block_index"],
                block["content_length"],
                block["content_text"],
                block["content_json"],
            ],
        )

    # Load fact_tool_calls
    for tc in tool_calls_data:
        conn.execute(
            """INSERT INTO fact_tool_calls
               (tool_call_id, session_key, tool_key, date_key, time_key,
                invoke_message_id, result_message_id, timestamp, input_char_count,
                output_char_count, is_error, input_json, input_summary, output_text)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                tc["tool_call_id"],
                tc["session_key"],
                tc["tool_key"],
                tc["date_key"],
                tc["time_key"],
                tc["invoke_message_id"],
                tc["result_message_id"],
                tc["timestamp"],
                tc["input_char_count"],
                tc["output_char_count"],
                tc["is_error"],
                tc["input_json"],
                tc["input_summary"],
                tc["output_text"],
            ],
        )

    # Load fact_session_summary
    session_duration = 0
    if first_timestamp and last_timestamp:
        session_duration = int((last_timestamp - first_timestamp).total_seconds())

    first_date_key = None
    if first_timestamp:
        first_date_key = int(first_timestamp.strftime("%Y%m%d"))

    conn.execute(
        """INSERT INTO fact_session_summary
           (session_key, project_key, date_key, total_messages, user_messages,
            assistant_messages, total_tool_calls, total_thinking_blocks,
            total_content_blocks, session_duration_seconds, first_timestamp, last_timestamp)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        [
            session_key,
            project_key,
            first_date_key,
            user_count + assistant_count,
            user_count,
            assistant_count,
            len(tool_calls_data),
            thinking_count,
            total_content_blocks,
            session_duration,
            first_timestamp,
            last_timestamp,
        ],
    )

    # ==========================================================================
    # Phase 3: Load Enhanced Granular Tables
    # ==========================================================================

    # Load dim_file
    for file_path, file_info in files_seen.items():
        existing = conn.execute(
            "SELECT 1 FROM dim_file WHERE file_key = ?", [file_info["file_key"]]
        ).fetchone()
        if not existing:
            conn.execute(
                """INSERT INTO dim_file
                   (file_key, file_path, file_name, file_extension, directory_path)
                   VALUES (?, ?, ?, ?, ?)""",
                [
                    file_info["file_key"],
                    file_info["file_path"],
                    file_info["file_name"],
                    file_info["file_extension"],
                    file_info["directory_path"],
                ],
            )

    # Load dim_programming_language
    for language in languages_seen:
        lang_key = generate_dimension_key(language)
        existing = conn.execute(
            "SELECT 1 FROM dim_programming_language WHERE language_key = ?", [lang_key]
        ).fetchone()
        if not existing:
            # Get file extensions for this language
            extensions = LANGUAGE_EXTENSIONS.get(language, [])
            ext_str = ",".join(extensions) if extensions else ""
            conn.execute(
                """INSERT INTO dim_programming_language
                   (language_key, language_name, file_extensions)
                   VALUES (?, ?, ?)""",
                [lang_key, language, ext_str],
            )

    # Load dim_error_type (just tool_error for now)
    error_type_key = generate_dimension_key("tool_error")
    existing = conn.execute(
        "SELECT 1 FROM dim_error_type WHERE error_type_key = ?", [error_type_key]
    ).fetchone()
    if not existing:
        conn.execute(
            """INSERT INTO dim_error_type
               (error_type_key, error_type, error_category)
               VALUES (?, ?, ?)""",
            [error_type_key, "tool_error", "execution"],
        )

    # Load fact_file_operations
    for fop in file_operations_data:
        conn.execute(
            """INSERT INTO fact_file_operations
               (file_operation_id, tool_call_id, session_key, file_key, tool_key,
                date_key, time_key, operation_type, file_size_chars, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                fop["file_operation_id"],
                fop["tool_call_id"],
                fop["session_key"],
                fop["file_key"],
                fop["tool_key"],
                fop["date_key"],
                fop["time_key"],
                fop["operation_type"],
                fop["file_size_chars"],
                fop["timestamp"],
            ],
        )

    # Load fact_code_blocks
    for cb in code_blocks_data:
        conn.execute(
            """INSERT INTO fact_code_blocks
               (code_block_id, message_id, session_key, language_key,
                date_key, time_key, block_index, line_count, char_count, code_text)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                cb["code_block_id"],
                cb["message_id"],
                cb["session_key"],
                cb["language_key"],
                cb["date_key"],
                cb["time_key"],
                cb["block_index"],
                cb["line_count"],
                cb["char_count"],
                cb["code_text"],
            ],
        )

    # Load fact_errors
    for err in errors_data:
        conn.execute(
            """INSERT INTO fact_errors
               (error_id, tool_call_id, session_key, tool_key, error_type_key,
                date_key, time_key, error_message, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                err["error_id"],
                err["tool_call_id"],
                err["session_key"],
                err["tool_key"],
                err["error_type_key"],
                err["date_key"],
                err["time_key"],
                err["error_message"],
                err["timestamp"],
            ],
        )


def parse_session_file(filepath):
    """Parse a session file and return normalized data.

    Supports both JSON and JSONL formats.
    Returns a dict with 'loglines' key containing the normalized entries.
    """
    filepath = Path(filepath)

    if filepath.suffix == ".jsonl":
        return _parse_jsonl_file(filepath)
    else:
        # Standard JSON format
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)


def _parse_jsonl_file(filepath):
    """Parse JSONL file and convert to standard format."""
    loglines = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                entry_type = obj.get("type")

                # Skip non-message entries
                if entry_type not in ("user", "assistant"):
                    continue

                # Convert to standard format
                entry = {
                    "type": entry_type,
                    "timestamp": obj.get("timestamp", ""),
                    "message": obj.get("message", {}),
                }

                # Preserve isCompactSummary if present
                if obj.get("isCompactSummary"):
                    entry["isCompactSummary"] = True

                loglines.append(entry)
            except json.JSONDecodeError:
                continue

    return {"loglines": loglines}


class CredentialsError(Exception):
    """Raised when credentials cannot be obtained."""

    pass


def get_access_token_from_keychain():
    """Get access token from macOS keychain.

    Returns the access token or None if not found.
    Raises CredentialsError with helpful message on failure.
    """
    if platform.system() != "Darwin":
        return None

    try:
        result = subprocess.run(
            [
                "security",
                "find-generic-password",
                "-a",
                os.environ.get("USER", ""),
                "-s",
                "Claude Code-credentials",
                "-w",
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return None

        # Parse the JSON to get the access token
        creds = json.loads(result.stdout.strip())
        return creds.get("claudeAiOauth", {}).get("accessToken")
    except (json.JSONDecodeError, subprocess.SubprocessError):
        return None


def get_org_uuid_from_config():
    """Get organization UUID from ~/.claude.json.

    Returns the organization UUID or None if not found.
    """
    config_path = Path.home() / ".claude.json"
    if not config_path.exists():
        return None

    try:
        with open(config_path) as f:
            config = json.load(f)
        return config.get("oauthAccount", {}).get("organizationUuid")
    except (json.JSONDecodeError, IOError):
        return None


def get_api_headers(token, org_uuid):
    """Build API request headers."""
    return {
        "Authorization": f"Bearer {token}",
        "anthropic-version": ANTHROPIC_VERSION,
        "Content-Type": "application/json",
        "x-organization-uuid": org_uuid,
    }


def fetch_sessions(token, org_uuid):
    """Fetch list of sessions from the API.

    Returns the sessions data as a dict.
    Raises httpx.HTTPError on network/API errors.
    """
    headers = get_api_headers(token, org_uuid)
    response = httpx.get(f"{API_BASE_URL}/sessions", headers=headers, timeout=30.0)
    response.raise_for_status()
    return response.json()


def fetch_session(token, org_uuid, session_id):
    """Fetch a specific session from the API.

    Returns the session data as a dict.
    Raises httpx.HTTPError on network/API errors.
    """
    headers = get_api_headers(token, org_uuid)
    response = httpx.get(
        f"{API_BASE_URL}/session_ingress/session/{session_id}",
        headers=headers,
        timeout=60.0,
    )
    response.raise_for_status()
    return response.json()


def detect_github_repo(loglines):
    """
    Detect GitHub repo from git push output in tool results.

    Looks for patterns like:
    - github.com/owner/repo/pull/new/branch (from git push messages)

    Returns the first detected repo (owner/name) or None.
    """
    for entry in loglines:
        message = entry.get("message", {})
        content = message.get("content", [])
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_result":
                result_content = block.get("content", "")
                if isinstance(result_content, str):
                    match = GITHUB_REPO_PATTERN.search(result_content)
                    if match:
                        return match.group(1)
    return None


def format_json(obj):
    try:
        if isinstance(obj, str):
            obj = json.loads(obj)
        formatted = json.dumps(obj, indent=2, ensure_ascii=False)
        return f'<pre class="json">{html.escape(formatted)}</pre>'
    except (json.JSONDecodeError, TypeError):
        return f"<pre>{html.escape(str(obj))}</pre>"


def render_markdown_text(text):
    if not text:
        return ""
    return markdown.markdown(text, extensions=["fenced_code", "tables"])


def is_json_like(text):
    if not text or not isinstance(text, str):
        return False
    text = text.strip()
    return (text.startswith("{") and text.endswith("}")) or (
        text.startswith("[") and text.endswith("]")
    )


def render_todo_write(tool_input, tool_id):
    todos = tool_input.get("todos", [])
    if not todos:
        return ""
    return _macros.todo_list(todos, tool_id)


def render_write_tool(tool_input, tool_id):
    """Render Write tool calls with file path header and content preview."""
    file_path = tool_input.get("file_path", "Unknown file")
    content = tool_input.get("content", "")
    return _macros.write_tool(file_path, content, tool_id)


def render_edit_tool(tool_input, tool_id):
    """Render Edit tool calls with diff-like old/new display."""
    file_path = tool_input.get("file_path", "Unknown file")
    old_string = tool_input.get("old_string", "")
    new_string = tool_input.get("new_string", "")
    replace_all = tool_input.get("replace_all", False)
    return _macros.edit_tool(file_path, old_string, new_string, replace_all, tool_id)


def render_bash_tool(tool_input, tool_id):
    """Render Bash tool calls with command as plain text."""
    command = tool_input.get("command", "")
    description = tool_input.get("description", "")
    return _macros.bash_tool(command, description, tool_id)


def render_content_block(block):
    if not isinstance(block, dict):
        return f"<p>{html.escape(str(block))}</p>"
    block_type = block.get("type", "")
    if block_type == "image":
        source = block.get("source", {})
        media_type = source.get("media_type", "image/png")
        data = source.get("data", "")
        return _macros.image_block(media_type, data)
    elif block_type == "thinking":
        content_html = render_markdown_text(block.get("thinking", ""))
        return _macros.thinking(content_html)
    elif block_type == "text":
        content_html = render_markdown_text(block.get("text", ""))
        return _macros.assistant_text(content_html)
    elif block_type == "tool_use":
        tool_name = block.get("name", "Unknown tool")
        tool_input = block.get("input", {})
        tool_id = block.get("id", "")
        if tool_name == "TodoWrite":
            return render_todo_write(tool_input, tool_id)
        if tool_name == "Write":
            return render_write_tool(tool_input, tool_id)
        if tool_name == "Edit":
            return render_edit_tool(tool_input, tool_id)
        if tool_name == "Bash":
            return render_bash_tool(tool_input, tool_id)
        description = tool_input.get("description", "")
        display_input = {k: v for k, v in tool_input.items() if k != "description"}
        input_json = json.dumps(display_input, indent=2, ensure_ascii=False)
        return _macros.tool_use(tool_name, description, input_json, tool_id)
    elif block_type == "tool_result":
        content = block.get("content", "")
        is_error = block.get("is_error", False)

        # Check for git commits and render with styled cards
        if isinstance(content, str):
            commits_found = list(COMMIT_PATTERN.finditer(content))
            if commits_found:
                # Build commit cards + remaining content
                parts = []
                last_end = 0
                for match in commits_found:
                    # Add any content before this commit
                    before = content[last_end : match.start()].strip()
                    if before:
                        parts.append(f"<pre>{html.escape(before)}</pre>")

                    commit_hash = match.group(1)
                    commit_msg = match.group(2)
                    parts.append(
                        _macros.commit_card(commit_hash, commit_msg, _github_repo)
                    )
                    last_end = match.end()

                # Add any remaining content after last commit
                after = content[last_end:].strip()
                if after:
                    parts.append(f"<pre>{html.escape(after)}</pre>")

                content_html = "".join(parts)
            else:
                content_html = f"<pre>{html.escape(content)}</pre>"
        elif isinstance(content, list) or is_json_like(content):
            content_html = format_json(content)
        else:
            content_html = format_json(content)
        return _macros.tool_result(content_html, is_error)
    else:
        return format_json(block)


def render_user_message_content(message_data):
    content = message_data.get("content", "")
    if isinstance(content, str):
        if is_json_like(content):
            return _macros.user_content(format_json(content))
        return _macros.user_content(render_markdown_text(content))
    elif isinstance(content, list):
        return "".join(render_content_block(block) for block in content)
    return f"<p>{html.escape(str(content))}</p>"


def render_assistant_message(message_data):
    content = message_data.get("content", [])
    if not isinstance(content, list):
        return f"<p>{html.escape(str(content))}</p>"
    return "".join(render_content_block(block) for block in content)


def make_msg_id(timestamp):
    return f"msg-{timestamp.replace(':', '-').replace('.', '-')}"


def analyze_conversation(messages):
    """Analyze messages in a conversation to extract stats and long texts."""
    tool_counts = {}  # tool_name -> count
    long_texts = []
    commits = []  # list of (hash, message, timestamp)

    for log_type, message_json, timestamp in messages:
        if not message_json:
            continue
        try:
            message_data = json.loads(message_json)
        except json.JSONDecodeError:
            continue

        content = message_data.get("content", [])
        if not isinstance(content, list):
            continue

        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type", "")

            if block_type == "tool_use":
                tool_name = block.get("name", "Unknown")
                tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1
            elif block_type == "tool_result":
                # Check for git commit output
                result_content = block.get("content", "")
                if isinstance(result_content, str):
                    for match in COMMIT_PATTERN.finditer(result_content):
                        commits.append((match.group(1), match.group(2), timestamp))
            elif block_type == "text":
                text = block.get("text", "")
                if len(text) >= LONG_TEXT_THRESHOLD:
                    long_texts.append(text)

    return {
        "tool_counts": tool_counts,
        "long_texts": long_texts,
        "commits": commits,
    }


def format_tool_stats(tool_counts):
    """Format tool counts into a concise summary string."""
    if not tool_counts:
        return ""

    # Abbreviate common tool names
    abbrev = {
        "Bash": "bash",
        "Read": "read",
        "Write": "write",
        "Edit": "edit",
        "Glob": "glob",
        "Grep": "grep",
        "Task": "task",
        "TodoWrite": "todo",
        "WebFetch": "fetch",
        "WebSearch": "search",
    }

    parts = []
    for name, count in sorted(tool_counts.items(), key=lambda x: -x[1]):
        short_name = abbrev.get(name, name.lower())
        parts.append(f"{count} {short_name}")

    return " · ".join(parts)


def is_tool_result_message(message_data):
    """Check if a message contains only tool_result blocks."""
    content = message_data.get("content", [])
    if not isinstance(content, list):
        return False
    if not content:
        return False
    return all(
        isinstance(block, dict) and block.get("type") == "tool_result"
        for block in content
    )


def render_message(log_type, message_json, timestamp):
    if not message_json:
        return ""
    try:
        message_data = json.loads(message_json)
    except json.JSONDecodeError:
        return ""
    if log_type == "user":
        content_html = render_user_message_content(message_data)
        # Check if this is a tool result message
        if is_tool_result_message(message_data):
            role_class, role_label = "tool-reply", "Tool reply"
        else:
            role_class, role_label = "user", "User"
    elif log_type == "assistant":
        content_html = render_assistant_message(message_data)
        role_class, role_label = "assistant", "Assistant"
    else:
        return ""
    if not content_html.strip():
        return ""
    msg_id = make_msg_id(timestamp)
    return _macros.message(role_class, role_label, msg_id, timestamp, content_html)


CSS = """
:root { --bg-color: #f5f5f5; --card-bg: #ffffff; --user-bg: #e3f2fd; --user-border: #1976d2; --assistant-bg: #f5f5f5; --assistant-border: #9e9e9e; --thinking-bg: #fff8e1; --thinking-border: #ffc107; --thinking-text: #666; --tool-bg: #f3e5f5; --tool-border: #9c27b0; --tool-result-bg: #e8f5e9; --tool-error-bg: #ffebee; --text-color: #212121; --text-muted: #757575; --code-bg: #263238; --code-text: #aed581; }
* { box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: var(--bg-color); color: var(--text-color); margin: 0; padding: 16px; line-height: 1.6; }
.container { max-width: 800px; margin: 0 auto; }
h1 { font-size: 1.5rem; margin-bottom: 24px; padding-bottom: 8px; border-bottom: 2px solid var(--user-border); }
.header-row { display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px; border-bottom: 2px solid var(--user-border); padding-bottom: 8px; margin-bottom: 24px; }
.header-row h1 { border-bottom: none; padding-bottom: 0; margin-bottom: 0; flex: 1; min-width: 200px; }
.message { margin-bottom: 16px; border-radius: 12px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
.message.user { background: var(--user-bg); border-left: 4px solid var(--user-border); }
.message.assistant { background: var(--card-bg); border-left: 4px solid var(--assistant-border); }
.message.tool-reply { background: #fff8e1; border-left: 4px solid #ff9800; }
.tool-reply .role-label { color: #e65100; }
.tool-reply .tool-result { background: transparent; padding: 0; margin: 0; }
.tool-reply .tool-result .truncatable.truncated::after { background: linear-gradient(to bottom, transparent, #fff8e1); }
.message-header { display: flex; justify-content: space-between; align-items: center; padding: 8px 16px; background: rgba(0,0,0,0.03); font-size: 0.85rem; }
.role-label { font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }
.user .role-label { color: var(--user-border); }
time { color: var(--text-muted); font-size: 0.8rem; }
.timestamp-link { color: inherit; text-decoration: none; }
.timestamp-link:hover { text-decoration: underline; }
.message:target { animation: highlight 2s ease-out; }
@keyframes highlight { 0% { background-color: rgba(25, 118, 210, 0.2); } 100% { background-color: transparent; } }
.message-content { padding: 16px; }
.message-content p { margin: 0 0 12px 0; }
.message-content p:last-child { margin-bottom: 0; }
.thinking { background: var(--thinking-bg); border: 1px solid var(--thinking-border); border-radius: 8px; padding: 12px; margin: 12px 0; font-size: 0.9rem; color: var(--thinking-text); }
.thinking-label { font-size: 0.75rem; font-weight: 600; text-transform: uppercase; color: #f57c00; margin-bottom: 8px; }
.thinking p { margin: 8px 0; }
.assistant-text { margin: 8px 0; }
.tool-use { background: var(--tool-bg); border: 1px solid var(--tool-border); border-radius: 8px; padding: 12px; margin: 12px 0; }
.tool-header { font-weight: 600; color: var(--tool-border); margin-bottom: 8px; display: flex; align-items: center; gap: 8px; }
.tool-icon { font-size: 1.1rem; }
.tool-description { font-size: 0.9rem; color: var(--text-muted); margin-bottom: 8px; font-style: italic; }
.tool-result { background: var(--tool-result-bg); border-radius: 8px; padding: 12px; margin: 12px 0; }
.tool-result.tool-error { background: var(--tool-error-bg); }
.file-tool { border-radius: 8px; padding: 12px; margin: 12px 0; }
.write-tool { background: linear-gradient(135deg, #e3f2fd 0%, #e8f5e9 100%); border: 1px solid #4caf50; }
.edit-tool { background: linear-gradient(135deg, #fff3e0 0%, #fce4ec 100%); border: 1px solid #ff9800; }
.file-tool-header { font-weight: 600; margin-bottom: 4px; display: flex; align-items: center; gap: 8px; font-size: 0.95rem; }
.write-header { color: #2e7d32; }
.edit-header { color: #e65100; }
.file-tool-icon { font-size: 1rem; }
.file-tool-path { font-family: monospace; background: rgba(0,0,0,0.08); padding: 2px 8px; border-radius: 4px; }
.file-tool-fullpath { font-family: monospace; font-size: 0.8rem; color: var(--text-muted); margin-bottom: 8px; word-break: break-all; }
.file-content { margin: 0; }
.edit-section { display: flex; margin: 4px 0; border-radius: 4px; overflow: hidden; }
.edit-label { padding: 8px 12px; font-weight: bold; font-family: monospace; display: flex; align-items: flex-start; }
.edit-old { background: #fce4ec; }
.edit-old .edit-label { color: #b71c1c; background: #f8bbd9; }
.edit-old .edit-content { color: #880e4f; }
.edit-new { background: #e8f5e9; }
.edit-new .edit-label { color: #1b5e20; background: #a5d6a7; }
.edit-new .edit-content { color: #1b5e20; }
.edit-content { margin: 0; flex: 1; background: transparent; font-size: 0.85rem; }
.edit-replace-all { font-size: 0.75rem; font-weight: normal; color: var(--text-muted); }
.write-tool .truncatable.truncated::after { background: linear-gradient(to bottom, transparent, #e6f4ea); }
.edit-tool .truncatable.truncated::after { background: linear-gradient(to bottom, transparent, #fff0e5); }
.todo-list { background: linear-gradient(135deg, #e8f5e9 0%, #f1f8e9 100%); border: 1px solid #81c784; border-radius: 8px; padding: 12px; margin: 12px 0; }
.todo-header { font-weight: 600; color: #2e7d32; margin-bottom: 10px; display: flex; align-items: center; gap: 8px; font-size: 0.95rem; }
.todo-items { list-style: none; margin: 0; padding: 0; }
.todo-item { display: flex; align-items: flex-start; gap: 10px; padding: 6px 0; border-bottom: 1px solid rgba(0,0,0,0.06); font-size: 0.9rem; }
.todo-item:last-child { border-bottom: none; }
.todo-icon { flex-shrink: 0; width: 20px; height: 20px; display: flex; align-items: center; justify-content: center; font-weight: bold; border-radius: 50%; }
.todo-completed .todo-icon { color: #2e7d32; background: rgba(46, 125, 50, 0.15); }
.todo-completed .todo-content { color: #558b2f; text-decoration: line-through; }
.todo-in-progress .todo-icon { color: #f57c00; background: rgba(245, 124, 0, 0.15); }
.todo-in-progress .todo-content { color: #e65100; font-weight: 500; }
.todo-pending .todo-icon { color: #757575; background: rgba(0,0,0,0.05); }
.todo-pending .todo-content { color: #616161; }
pre { background: var(--code-bg); color: var(--code-text); padding: 12px; border-radius: 6px; overflow-x: auto; font-size: 0.85rem; line-height: 1.5; margin: 8px 0; white-space: pre-wrap; word-wrap: break-word; }
pre.json { color: #e0e0e0; }
code { background: rgba(0,0,0,0.08); padding: 2px 6px; border-radius: 4px; font-size: 0.9em; }
pre code { background: none; padding: 0; }
.user-content { margin: 0; }
.truncatable { position: relative; }
.truncatable.truncated .truncatable-content { max-height: 200px; overflow: hidden; }
.truncatable.truncated::after { content: ''; position: absolute; bottom: 32px; left: 0; right: 0; height: 60px; background: linear-gradient(to bottom, transparent, var(--card-bg)); pointer-events: none; }
.message.user .truncatable.truncated::after { background: linear-gradient(to bottom, transparent, var(--user-bg)); }
.message.tool-reply .truncatable.truncated::after { background: linear-gradient(to bottom, transparent, #fff8e1); }
.tool-use .truncatable.truncated::after { background: linear-gradient(to bottom, transparent, var(--tool-bg)); }
.tool-result .truncatable.truncated::after { background: linear-gradient(to bottom, transparent, var(--tool-result-bg)); }
.expand-btn { display: none; width: 100%; padding: 8px 16px; margin-top: 4px; background: rgba(0,0,0,0.05); border: 1px solid rgba(0,0,0,0.1); border-radius: 6px; cursor: pointer; font-size: 0.85rem; color: var(--text-muted); }
.expand-btn:hover { background: rgba(0,0,0,0.1); }
.truncatable.truncated .expand-btn, .truncatable.expanded .expand-btn { display: block; }
.pagination { display: flex; justify-content: center; gap: 8px; margin: 24px 0; flex-wrap: wrap; }
.pagination a, .pagination span { padding: 5px 10px; border-radius: 6px; text-decoration: none; font-size: 0.85rem; }
.pagination a { background: var(--card-bg); color: var(--user-border); border: 1px solid var(--user-border); }
.pagination a:hover { background: var(--user-bg); }
.pagination .current { background: var(--user-border); color: white; }
.pagination .disabled { color: var(--text-muted); border: 1px solid #ddd; }
.pagination .index-link { background: var(--user-border); color: white; }
details.continuation { margin-bottom: 16px; }
details.continuation summary { cursor: pointer; padding: 12px 16px; background: var(--user-bg); border-left: 4px solid var(--user-border); border-radius: 12px; font-weight: 500; color: var(--text-muted); }
details.continuation summary:hover { background: rgba(25, 118, 210, 0.15); }
details.continuation[open] summary { border-radius: 12px 12px 0 0; margin-bottom: 0; }
.index-item { margin-bottom: 16px; border-radius: 12px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.1); background: var(--user-bg); border-left: 4px solid var(--user-border); }
.index-item a { display: block; text-decoration: none; color: inherit; }
.index-item a:hover { background: rgba(25, 118, 210, 0.1); }
.index-item-header { display: flex; justify-content: space-between; align-items: center; padding: 8px 16px; background: rgba(0,0,0,0.03); font-size: 0.85rem; }
.index-item-number { font-weight: 600; color: var(--user-border); }
.index-item-content { padding: 16px; }
.index-item-stats { padding: 8px 16px 12px 32px; font-size: 0.85rem; color: var(--text-muted); border-top: 1px solid rgba(0,0,0,0.06); }
.index-item-commit { margin-top: 6px; padding: 4px 8px; background: #fff3e0; border-radius: 4px; font-size: 0.85rem; color: #e65100; }
.index-item-commit code { background: rgba(0,0,0,0.08); padding: 1px 4px; border-radius: 3px; font-size: 0.8rem; margin-right: 6px; }
.commit-card { margin: 8px 0; padding: 10px 14px; background: #fff3e0; border-left: 4px solid #ff9800; border-radius: 6px; }
.commit-card a { text-decoration: none; color: #5d4037; display: block; }
.commit-card a:hover { color: #e65100; }
.commit-card-hash { font-family: monospace; color: #e65100; font-weight: 600; margin-right: 8px; }
.index-commit { margin-bottom: 12px; padding: 10px 16px; background: #fff3e0; border-left: 4px solid #ff9800; border-radius: 8px; box-shadow: 0 1px 2px rgba(0,0,0,0.05); }
.index-commit a { display: block; text-decoration: none; color: inherit; }
.index-commit a:hover { background: rgba(255, 152, 0, 0.1); margin: -10px -16px; padding: 10px 16px; border-radius: 8px; }
.index-commit-header { display: flex; justify-content: space-between; align-items: center; font-size: 0.85rem; margin-bottom: 4px; }
.index-commit-hash { font-family: monospace; color: #e65100; font-weight: 600; }
.index-commit-msg { color: #5d4037; }
.index-item-long-text { margin-top: 8px; padding: 12px; background: var(--card-bg); border-radius: 8px; border-left: 3px solid var(--assistant-border); }
.index-item-long-text .truncatable.truncated::after { background: linear-gradient(to bottom, transparent, var(--card-bg)); }
.index-item-long-text-content { color: var(--text-color); }
#search-box { display: none; align-items: center; gap: 8px; }
#search-box input { padding: 6px 12px; border: 1px solid var(--assistant-border); border-radius: 6px; font-size: 16px; width: 180px; }
#search-box button, #modal-search-btn, #modal-close-btn { background: var(--user-border); color: white; border: none; border-radius: 6px; padding: 6px 10px; cursor: pointer; display: flex; align-items: center; justify-content: center; }
#search-box button:hover, #modal-search-btn:hover { background: #1565c0; }
#modal-close-btn { background: var(--text-muted); margin-left: 8px; }
#modal-close-btn:hover { background: #616161; }
#search-modal[open] { border: none; border-radius: 12px; box-shadow: 0 4px 24px rgba(0,0,0,0.2); padding: 0; width: 90vw; max-width: 900px; height: 80vh; max-height: 80vh; display: flex; flex-direction: column; }
#search-modal::backdrop { background: rgba(0,0,0,0.5); }
.search-modal-header { display: flex; align-items: center; gap: 8px; padding: 16px; border-bottom: 1px solid var(--assistant-border); background: var(--bg-color); border-radius: 12px 12px 0 0; }
.search-modal-header input { flex: 1; padding: 8px 12px; border: 1px solid var(--assistant-border); border-radius: 6px; font-size: 16px; }
#search-status { padding: 8px 16px; font-size: 0.85rem; color: var(--text-muted); border-bottom: 1px solid rgba(0,0,0,0.06); }
#search-results { flex: 1; overflow-y: auto; padding: 16px; }
.search-result { margin-bottom: 16px; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
.search-result a { display: block; text-decoration: none; color: inherit; }
.search-result a:hover { background: rgba(25, 118, 210, 0.05); }
.search-result-page { padding: 6px 12px; background: rgba(0,0,0,0.03); font-size: 0.8rem; color: var(--text-muted); border-bottom: 1px solid rgba(0,0,0,0.06); }
.search-result-content { padding: 12px; }
.search-result mark { background: #fff59d; padding: 1px 2px; border-radius: 2px; }
#global-search-box { display: none; align-items: center; gap: 8px; }
#global-search-box input { padding: 6px 12px; border: 1px solid var(--assistant-border); border-radius: 6px; font-size: 16px; width: 180px; }
#global-search-box button { background: var(--user-border); color: white; border: none; border-radius: 6px; padding: 6px 10px; cursor: pointer; display: flex; align-items: center; justify-content: center; }
#global-search-box button:hover { background: #1565c0; }
#global-search-modal[open] { border: none; border-radius: 12px; box-shadow: 0 4px 24px rgba(0,0,0,0.2); padding: 0; width: 90vw; max-width: 900px; height: 80vh; max-height: 80vh; display: flex; flex-direction: column; }
#global-search-modal::backdrop { background: rgba(0,0,0,0.5); }
#global-search-status { padding: 8px 16px; font-size: 0.85rem; color: var(--text-muted); border-bottom: 1px solid rgba(0,0,0,0.06); }
#global-search-results { flex: 1; overflow-y: auto; padding: 16px; }
.search-result-meta { display: flex; align-items: center; gap: 8px; padding: 6px 12px; background: rgba(0,0,0,0.03); font-size: 0.8rem; color: var(--text-muted); border-bottom: 1px solid rgba(0,0,0,0.06); flex-wrap: wrap; }
.search-result-project { font-weight: 600; color: var(--user-border); }
.search-result-type { background: var(--assistant-border); padding: 2px 6px; border-radius: 4px; font-size: 0.75rem; }
.search-result-snippet { padding: 12px; font-size: 0.9rem; line-height: 1.5; }
.search-more { padding: 12px; text-align: center; color: var(--text-muted); font-style: italic; }
@media (max-width: 600px) { body { padding: 8px; } .message, .index-item { border-radius: 8px; } .message-content, .index-item-content { padding: 12px; } pre { font-size: 0.8rem; padding: 8px; } #search-box input, #global-search-box input { width: 100%; min-width: 100px; } .header-row { flex-direction: column; align-items: stretch; } .header-row h1 { text-align: center; } #global-search-box { width: 100%; justify-content: center; } #search-modal[open], #global-search-modal[open] { width: 95vw; height: 90vh; } .search-modal-header { flex-wrap: wrap; } .search-modal-header input { min-width: 150px; } .search-result-meta { font-size: 0.75rem; } .search-result-snippet { font-size: 0.85rem; padding: 8px; } }
"""

JS = """
document.querySelectorAll('time[data-timestamp]').forEach(function(el) {
    const timestamp = el.getAttribute('data-timestamp');
    const date = new Date(timestamp);
    const now = new Date();
    const isToday = date.toDateString() === now.toDateString();
    const timeStr = date.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' });
    if (isToday) { el.textContent = timeStr; }
    else { el.textContent = date.toLocaleDateString(undefined, { month: 'short', day: 'numeric' }) + ' ' + timeStr; }
});
document.querySelectorAll('pre.json').forEach(function(el) {
    let text = el.textContent;
    text = text.replace(/"([^"]+)":/g, '<span style="color: #ce93d8">"$1"</span>:');
    text = text.replace(/: "([^"]*)"/g, ': <span style="color: #81d4fa">"$1"</span>');
    text = text.replace(/: (\\d+)/g, ': <span style="color: #ffcc80">$1</span>');
    text = text.replace(/: (true|false|null)/g, ': <span style="color: #f48fb1">$1</span>');
    el.innerHTML = text;
});
document.querySelectorAll('.truncatable').forEach(function(wrapper) {
    const content = wrapper.querySelector('.truncatable-content');
    const btn = wrapper.querySelector('.expand-btn');
    if (content.scrollHeight > 250) {
        wrapper.classList.add('truncated');
        btn.addEventListener('click', function() {
            if (wrapper.classList.contains('truncated')) { wrapper.classList.remove('truncated'); wrapper.classList.add('expanded'); btn.textContent = 'Show less'; }
            else { wrapper.classList.remove('expanded'); wrapper.classList.add('truncated'); btn.textContent = 'Show more'; }
        });
    }
});
"""

# JavaScript to fix relative URLs when served via gisthost.github.io or gistpreview.github.io
# Fixes issue #26: Pagination links broken on gisthost.github.io
GIST_PREVIEW_JS = r"""
(function() {
    var hostname = window.location.hostname;
    if (hostname !== 'gisthost.github.io' && hostname !== 'gistpreview.github.io') return;
    // URL format: https://gisthost.github.io/?GIST_ID/filename.html
    var match = window.location.search.match(/^\?([^/]+)/);
    if (!match) return;
    var gistId = match[1];

    function rewriteLinks(root) {
        (root || document).querySelectorAll('a[href]').forEach(function(link) {
            var href = link.getAttribute('href');
            // Skip already-rewritten links (issue #26 fix)
            if (href.startsWith('?')) return;
            // Skip external links and anchors
            if (href.startsWith('http') || href.startsWith('#') || href.startsWith('//')) return;
            // Handle anchor in relative URL (e.g., page-001.html#msg-123)
            var parts = href.split('#');
            var filename = parts[0];
            var anchor = parts.length > 1 ? '#' + parts[1] : '';
            link.setAttribute('href', '?' + gistId + '/' + filename + anchor);
        });
    }

    // Run immediately
    rewriteLinks();

    // Also run on DOMContentLoaded in case DOM isn't ready yet
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', function() { rewriteLinks(); });
    }

    // Use MutationObserver to catch dynamically added content
    // gistpreview.github.io may add content after initial load
    var observer = new MutationObserver(function(mutations) {
        mutations.forEach(function(mutation) {
            mutation.addedNodes.forEach(function(node) {
                if (node.nodeType === 1) { // Element node
                    rewriteLinks(node);
                    // Also check if the node itself is a link
                    if (node.tagName === 'A' && node.getAttribute('href')) {
                        var href = node.getAttribute('href');
                        if (!href.startsWith('?') && !href.startsWith('http') &&
                            !href.startsWith('#') && !href.startsWith('//')) {
                            var parts = href.split('#');
                            var filename = parts[0];
                            var anchor = parts.length > 1 ? '#' + parts[1] : '';
                            node.setAttribute('href', '?' + gistId + '/' + filename + anchor);
                        }
                    }
                }
            });
        });
    });

    // Start observing once body exists
    function startObserving() {
        if (document.body) {
            observer.observe(document.body, { childList: true, subtree: true });
        } else {
            setTimeout(startObserving, 10);
        }
    }
    startObserving();

    // Handle fragment navigation after dynamic content loads
    // gisthost.github.io/gistpreview.github.io loads content dynamically, so the browser's
    // native fragment navigation fails because the element doesn't exist yet
    function scrollToFragment() {
        var hash = window.location.hash;
        if (!hash) return false;
        var targetId = hash.substring(1);
        var target = document.getElementById(targetId);
        if (target) {
            target.scrollIntoView({ behavior: 'smooth', block: 'start' });
            return true;
        }
        return false;
    }

    // Try immediately in case content is already loaded
    if (!scrollToFragment()) {
        // Retry with increasing delays to handle dynamic content loading
        var delays = [100, 300, 500, 1000, 2000];
        delays.forEach(function(delay) {
            setTimeout(scrollToFragment, delay);
        });
    }
})();
"""


def inject_gist_preview_js(output_dir):
    """Inject gist preview JavaScript into all HTML files in the output directory."""
    output_dir = Path(output_dir)
    for html_file in output_dir.glob("*.html"):
        content = html_file.read_text(encoding="utf-8")
        # Insert the gist preview JS before the closing </body> tag
        if "</body>" in content:
            content = content.replace(
                "</body>", f"<script>{GIST_PREVIEW_JS}</script>\n</body>"
            )
            html_file.write_text(content, encoding="utf-8")


def create_gist(output_dir, public=False):
    """Create a GitHub gist from the HTML files in output_dir.

    Returns the gist ID on success, or raises click.ClickException on failure.
    """
    output_dir = Path(output_dir)
    html_files = list(output_dir.glob("*.html"))
    if not html_files:
        raise click.ClickException("No HTML files found to upload to gist.")

    # Build the gh gist create command
    # gh gist create file1 file2 ... --public/--private
    cmd = ["gh", "gist", "create"]
    cmd.extend(str(f) for f in sorted(html_files))
    if public:
        cmd.append("--public")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        # Output is the gist URL, e.g., https://gist.github.com/username/GIST_ID
        gist_url = result.stdout.strip()
        # Extract gist ID from URL
        gist_id = gist_url.rstrip("/").split("/")[-1]
        return gist_id, gist_url
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.strip() if e.stderr else str(e)
        raise click.ClickException(f"Failed to create gist: {error_msg}")
    except FileNotFoundError:
        raise click.ClickException(
            "gh CLI not found. Install it from https://cli.github.com/ and run 'gh auth login'."
        )


def generate_pagination_html(current_page, total_pages):
    return _macros.pagination(current_page, total_pages)


def generate_index_pagination_html(total_pages):
    """Generate pagination for index page where Index is current (first page)."""
    return _macros.index_pagination(total_pages)


def generate_html(json_path, output_dir, github_repo=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Load session file (supports both JSON and JSONL)
    data = parse_session_file(json_path)

    loglines = data.get("loglines", [])

    # Auto-detect GitHub repo if not provided
    if github_repo is None:
        github_repo = detect_github_repo(loglines)
        if github_repo:
            print(f"Auto-detected GitHub repo: {github_repo}")
        else:
            print(
                "Warning: Could not auto-detect GitHub repo. Commit links will be disabled."
            )

    # Set module-level variable for render functions
    global _github_repo
    _github_repo = github_repo

    conversations = []
    current_conv = None
    for entry in loglines:
        log_type = entry.get("type")
        timestamp = entry.get("timestamp", "")
        is_compact_summary = entry.get("isCompactSummary", False)
        message_data = entry.get("message", {})
        if not message_data:
            continue
        # Convert message dict to JSON string for compatibility with existing render functions
        message_json = json.dumps(message_data)
        is_user_prompt = False
        user_text = None
        if log_type == "user":
            content = message_data.get("content", "")
            text = extract_text_from_content(content)
            if text:
                is_user_prompt = True
                user_text = text
        if is_user_prompt:
            if current_conv:
                conversations.append(current_conv)
            current_conv = {
                "user_text": user_text,
                "timestamp": timestamp,
                "messages": [(log_type, message_json, timestamp)],
                "is_continuation": bool(is_compact_summary),
            }
        elif current_conv:
            current_conv["messages"].append((log_type, message_json, timestamp))
    if current_conv:
        conversations.append(current_conv)

    total_convs = len(conversations)
    total_pages = (total_convs + PROMPTS_PER_PAGE - 1) // PROMPTS_PER_PAGE

    for page_num in range(1, total_pages + 1):
        start_idx = (page_num - 1) * PROMPTS_PER_PAGE
        end_idx = min(start_idx + PROMPTS_PER_PAGE, total_convs)
        page_convs = conversations[start_idx:end_idx]
        messages_html = []
        for conv in page_convs:
            is_first = True
            for log_type, message_json, timestamp in conv["messages"]:
                msg_html = render_message(log_type, message_json, timestamp)
                if msg_html:
                    # Wrap continuation summaries in collapsed details
                    if is_first and conv.get("is_continuation"):
                        msg_html = f'<details class="continuation"><summary>Session continuation summary</summary>{msg_html}</details>'
                    messages_html.append(msg_html)
                is_first = False
        pagination_html = generate_pagination_html(page_num, total_pages)
        page_template = get_template("page.html")
        page_content = page_template.render(
            css=CSS,
            js=JS,
            page_num=page_num,
            total_pages=total_pages,
            pagination_html=pagination_html,
            messages_html="".join(messages_html),
        )
        (output_dir / f"page-{page_num:03d}.html").write_text(
            page_content, encoding="utf-8"
        )
        print(f"Generated page-{page_num:03d}.html")

    # Calculate overall stats and collect all commits for timeline
    total_tool_counts = {}
    total_messages = 0
    all_commits = []  # (timestamp, hash, message, page_num, conv_index)
    for i, conv in enumerate(conversations):
        total_messages += len(conv["messages"])
        stats = analyze_conversation(conv["messages"])
        for tool, count in stats["tool_counts"].items():
            total_tool_counts[tool] = total_tool_counts.get(tool, 0) + count
        page_num = (i // PROMPTS_PER_PAGE) + 1
        for commit_hash, commit_msg, commit_ts in stats["commits"]:
            all_commits.append((commit_ts, commit_hash, commit_msg, page_num, i))
    total_tool_calls = sum(total_tool_counts.values())
    total_commits = len(all_commits)

    # Build timeline items: prompts and commits merged by timestamp
    timeline_items = []

    # Add prompts
    prompt_num = 0
    for i, conv in enumerate(conversations):
        if conv.get("is_continuation"):
            continue
        if conv["user_text"].startswith("Stop hook feedback:"):
            continue
        prompt_num += 1
        page_num = (i // PROMPTS_PER_PAGE) + 1
        msg_id = make_msg_id(conv["timestamp"])
        link = f"page-{page_num:03d}.html#{msg_id}"
        rendered_content = render_markdown_text(conv["user_text"])

        # Collect all messages including from subsequent continuation conversations
        # This ensures long_texts from continuations appear with the original prompt
        all_messages = list(conv["messages"])
        for j in range(i + 1, len(conversations)):
            if not conversations[j].get("is_continuation"):
                break
            all_messages.extend(conversations[j]["messages"])

        # Analyze conversation for stats (excluding commits from inline display now)
        stats = analyze_conversation(all_messages)
        tool_stats_str = format_tool_stats(stats["tool_counts"])

        long_texts_html = ""
        for lt in stats["long_texts"]:
            rendered_lt = render_markdown_text(lt)
            long_texts_html += _macros.index_long_text(rendered_lt)

        stats_html = _macros.index_stats(tool_stats_str, long_texts_html)

        item_html = _macros.index_item(
            prompt_num, link, conv["timestamp"], rendered_content, stats_html
        )
        timeline_items.append((conv["timestamp"], "prompt", item_html))

    # Add commits as separate timeline items
    for commit_ts, commit_hash, commit_msg, page_num, conv_idx in all_commits:
        item_html = _macros.index_commit(
            commit_hash, commit_msg, commit_ts, _github_repo
        )
        timeline_items.append((commit_ts, "commit", item_html))

    # Sort by timestamp
    timeline_items.sort(key=lambda x: x[0])
    index_items = [item[2] for item in timeline_items]

    index_pagination = generate_index_pagination_html(total_pages)
    index_template = get_template("index.html")
    index_content = index_template.render(
        css=CSS,
        js=JS,
        pagination_html=index_pagination,
        prompt_num=prompt_num,
        total_messages=total_messages,
        total_tool_calls=total_tool_calls,
        total_commits=total_commits,
        total_pages=total_pages,
        index_items_html="".join(index_items),
    )
    index_path = output_dir / "index.html"
    index_path.write_text(index_content, encoding="utf-8")
    print(
        f"Generated {index_path.resolve()} ({total_convs} prompts, {total_pages} pages)"
    )


@click.group(cls=DefaultGroup, default="local", default_if_no_args=True)
@click.version_option(None, "-v", "--version", package_name="claude-code-transcripts")
def cli():
    """Convert Claude Code session JSON to mobile-friendly HTML pages."""
    pass


@cli.command("local")
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    help="Output directory. If not specified, writes to temp dir and opens in browser.",
)
@click.option(
    "-a",
    "--output-auto",
    is_flag=True,
    help="Auto-name output subdirectory based on session filename (uses -o as parent, or current dir).",
)
@click.option(
    "--repo",
    help="GitHub repo (owner/name) for commit links. Auto-detected from git push output if not specified.",
)
@click.option(
    "--gist",
    is_flag=True,
    help="Upload to GitHub Gist and output a gisthost.github.io URL.",
)
@click.option(
    "--json",
    "include_json",
    is_flag=True,
    help="Include the original JSONL session file in the output directory.",
)
@click.option(
    "--open",
    "open_browser",
    is_flag=True,
    help="Open the generated index.html in your default browser (default if no -o specified).",
)
@click.option(
    "--limit",
    default=10,
    help="Maximum number of sessions to show (default: 10)",
)
def local_cmd(output, output_auto, repo, gist, include_json, open_browser, limit):
    """Select and convert a local Claude Code session to HTML."""
    projects_folder = Path.home() / ".claude" / "projects"

    if not projects_folder.exists():
        click.echo(f"Projects folder not found: {projects_folder}")
        click.echo("No local Claude Code sessions available.")
        return

    click.echo("Loading local sessions...")
    results = find_local_sessions(projects_folder, limit=limit)

    if not results:
        click.echo("No local sessions found.")
        return

    # Build choices for questionary
    choices = []
    for filepath, summary in results:
        stat = filepath.stat()
        mod_time = datetime.fromtimestamp(stat.st_mtime)
        size_kb = stat.st_size / 1024
        date_str = mod_time.strftime("%Y-%m-%d %H:%M")
        # Truncate summary if too long
        if len(summary) > 50:
            summary = summary[:47] + "..."
        display = f"{date_str}  {size_kb:5.0f} KB  {summary}"
        choices.append(questionary.Choice(title=display, value=filepath))

    selected = questionary.select(
        "Select a session to convert:",
        choices=choices,
    ).ask()

    if selected is None:
        click.echo("No session selected.")
        return

    session_file = selected

    # Determine output directory and whether to open browser
    # If no -o specified, use temp dir and open browser by default
    auto_open = output is None and not gist and not output_auto
    if output_auto:
        # Use -o as parent dir (or current dir), with auto-named subdirectory
        parent_dir = Path(output) if output else Path(".")
        output = parent_dir / session_file.stem
    elif output is None:
        output = Path(tempfile.gettempdir()) / f"claude-session-{session_file.stem}"

    output = Path(output)
    generate_html(session_file, output, github_repo=repo)

    # Show output directory
    click.echo(f"Output: {output.resolve()}")

    # Copy JSONL file to output directory if requested
    if include_json:
        output.mkdir(exist_ok=True)
        json_dest = output / session_file.name
        shutil.copy(session_file, json_dest)
        json_size_kb = json_dest.stat().st_size / 1024
        click.echo(f"JSONL: {json_dest} ({json_size_kb:.1f} KB)")

    if gist:
        # Inject gist preview JS and create gist
        inject_gist_preview_js(output)
        click.echo("Creating GitHub gist...")
        gist_id, gist_url = create_gist(output)
        preview_url = f"https://gisthost.github.io/?{gist_id}/index.html"
        click.echo(f"Gist: {gist_url}")
        click.echo(f"Preview: {preview_url}")

    if open_browser or auto_open:
        index_url = (output / "index.html").resolve().as_uri()
        webbrowser.open(index_url)


def is_url(path):
    """Check if a path is a URL (starts with http:// or https://)."""
    return path.startswith("http://") or path.startswith("https://")


def fetch_url_to_tempfile(url):
    """Fetch a URL and save to a temporary file.

    Returns the Path to the temporary file.
    Raises click.ClickException on network errors.
    """
    try:
        response = httpx.get(url, timeout=60.0, follow_redirects=True)
        response.raise_for_status()
    except httpx.RequestError as e:
        raise click.ClickException(f"Failed to fetch URL: {e}")
    except httpx.HTTPStatusError as e:
        raise click.ClickException(
            f"Failed to fetch URL: {e.response.status_code} {e.response.reason_phrase}"
        )

    # Determine file extension from URL
    url_path = url.split("?")[0]  # Remove query params
    if url_path.endswith(".jsonl"):
        suffix = ".jsonl"
    elif url_path.endswith(".json"):
        suffix = ".json"
    else:
        suffix = ".jsonl"  # Default to JSONL

    # Extract a name from the URL for the temp file
    url_name = Path(url_path).stem or "session"

    temp_dir = Path(tempfile.gettempdir())
    temp_file = temp_dir / f"claude-url-{url_name}{suffix}"
    temp_file.write_text(response.text, encoding="utf-8")
    return temp_file


@cli.command("json")
@click.argument("json_file", type=click.Path())
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    help="Output directory. If not specified, writes to temp dir and opens in browser.",
)
@click.option(
    "-a",
    "--output-auto",
    is_flag=True,
    help="Auto-name output subdirectory based on filename (uses -o as parent, or current dir).",
)
@click.option(
    "--repo",
    help="GitHub repo (owner/name) for commit links. Auto-detected from git push output if not specified.",
)
@click.option(
    "--gist",
    is_flag=True,
    help="Upload to GitHub Gist and output a gisthost.github.io URL.",
)
@click.option(
    "--json",
    "include_json",
    is_flag=True,
    help="Include the original JSON session file in the output directory.",
)
@click.option(
    "--open",
    "open_browser",
    is_flag=True,
    help="Open the generated index.html in your default browser (default if no -o specified).",
)
def json_cmd(json_file, output, output_auto, repo, gist, include_json, open_browser):
    """Convert a Claude Code session JSON/JSONL file or URL to HTML."""
    # Handle URL input
    if is_url(json_file):
        click.echo(f"Fetching {json_file}...")
        temp_file = fetch_url_to_tempfile(json_file)
        json_file_path = temp_file
        # Use URL path for naming
        url_name = Path(json_file.split("?")[0]).stem or "session"
    else:
        # Validate that local file exists
        json_file_path = Path(json_file)
        if not json_file_path.exists():
            raise click.ClickException(f"File not found: {json_file}")
        url_name = None

    # Determine output directory and whether to open browser
    # If no -o specified, use temp dir and open browser by default
    auto_open = output is None and not gist and not output_auto
    if output_auto:
        # Use -o as parent dir (or current dir), with auto-named subdirectory
        parent_dir = Path(output) if output else Path(".")
        output = parent_dir / (url_name or json_file_path.stem)
    elif output is None:
        output = (
            Path(tempfile.gettempdir())
            / f"claude-session-{url_name or json_file_path.stem}"
        )

    output = Path(output)
    generate_html(json_file_path, output, github_repo=repo)

    # Show output directory
    click.echo(f"Output: {output.resolve()}")

    # Copy JSON file to output directory if requested
    if include_json:
        output.mkdir(exist_ok=True)
        json_dest = output / json_file_path.name
        shutil.copy(json_file_path, json_dest)
        json_size_kb = json_dest.stat().st_size / 1024
        click.echo(f"JSON: {json_dest} ({json_size_kb:.1f} KB)")

    if gist:
        # Inject gist preview JS and create gist
        inject_gist_preview_js(output)
        click.echo("Creating GitHub gist...")
        gist_id, gist_url = create_gist(output)
        preview_url = f"https://gisthost.github.io/?{gist_id}/index.html"
        click.echo(f"Gist: {gist_url}")
        click.echo(f"Preview: {preview_url}")

    if open_browser or auto_open:
        index_url = (output / "index.html").resolve().as_uri()
        webbrowser.open(index_url)


def resolve_credentials(token, org_uuid):
    """Resolve token and org_uuid from arguments or auto-detect.

    Returns (token, org_uuid) tuple.
    Raises click.ClickException if credentials cannot be resolved.
    """
    # Get token
    if token is None:
        token = get_access_token_from_keychain()
        if token is None:
            if platform.system() == "Darwin":
                raise click.ClickException(
                    "Could not retrieve access token from macOS keychain. "
                    "Make sure you are logged into Claude Code, or provide --token."
                )
            else:
                raise click.ClickException(
                    "On non-macOS platforms, you must provide --token with your access token."
                )

    # Get org UUID
    if org_uuid is None:
        org_uuid = get_org_uuid_from_config()
        if org_uuid is None:
            raise click.ClickException(
                "Could not find organization UUID in ~/.claude.json. "
                "Provide --org-uuid with your organization UUID."
            )

    return token, org_uuid


def format_session_for_display(session_data):
    """Format a session for display in the list or picker.

    Returns a formatted string.
    """
    session_id = session_data.get("id", "unknown")
    title = session_data.get("title", "Untitled")
    created_at = session_data.get("created_at", "")
    # Truncate title if too long
    if len(title) > 60:
        title = title[:57] + "..."
    return f"{session_id}  {created_at[:19] if created_at else 'N/A':19}  {title}"


def generate_html_from_session_data(session_data, output_dir, github_repo=None):
    """Generate HTML from session data dict (instead of file path)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    loglines = session_data.get("loglines", [])

    # Auto-detect GitHub repo if not provided
    if github_repo is None:
        github_repo = detect_github_repo(loglines)
        if github_repo:
            click.echo(f"Auto-detected GitHub repo: {github_repo}")

    # Set module-level variable for render functions
    global _github_repo
    _github_repo = github_repo

    conversations = []
    current_conv = None
    for entry in loglines:
        log_type = entry.get("type")
        timestamp = entry.get("timestamp", "")
        is_compact_summary = entry.get("isCompactSummary", False)
        message_data = entry.get("message", {})
        if not message_data:
            continue
        # Convert message dict to JSON string for compatibility with existing render functions
        message_json = json.dumps(message_data)
        is_user_prompt = False
        user_text = None
        if log_type == "user":
            content = message_data.get("content", "")
            text = extract_text_from_content(content)
            if text:
                is_user_prompt = True
                user_text = text
        if is_user_prompt:
            if current_conv:
                conversations.append(current_conv)
            current_conv = {
                "user_text": user_text,
                "timestamp": timestamp,
                "messages": [(log_type, message_json, timestamp)],
                "is_continuation": bool(is_compact_summary),
            }
        elif current_conv:
            current_conv["messages"].append((log_type, message_json, timestamp))
    if current_conv:
        conversations.append(current_conv)

    total_convs = len(conversations)
    total_pages = (total_convs + PROMPTS_PER_PAGE - 1) // PROMPTS_PER_PAGE

    for page_num in range(1, total_pages + 1):
        start_idx = (page_num - 1) * PROMPTS_PER_PAGE
        end_idx = min(start_idx + PROMPTS_PER_PAGE, total_convs)
        page_convs = conversations[start_idx:end_idx]
        messages_html = []
        for conv in page_convs:
            is_first = True
            for log_type, message_json, timestamp in conv["messages"]:
                msg_html = render_message(log_type, message_json, timestamp)
                if msg_html:
                    # Wrap continuation summaries in collapsed details
                    if is_first and conv.get("is_continuation"):
                        msg_html = f'<details class="continuation"><summary>Session continuation summary</summary>{msg_html}</details>'
                    messages_html.append(msg_html)
                is_first = False
        pagination_html = generate_pagination_html(page_num, total_pages)
        page_template = get_template("page.html")
        page_content = page_template.render(
            css=CSS,
            js=JS,
            page_num=page_num,
            total_pages=total_pages,
            pagination_html=pagination_html,
            messages_html="".join(messages_html),
        )
        (output_dir / f"page-{page_num:03d}.html").write_text(
            page_content, encoding="utf-8"
        )
        click.echo(f"Generated page-{page_num:03d}.html")

    # Calculate overall stats and collect all commits for timeline
    total_tool_counts = {}
    total_messages = 0
    all_commits = []  # (timestamp, hash, message, page_num, conv_index)
    for i, conv in enumerate(conversations):
        total_messages += len(conv["messages"])
        stats = analyze_conversation(conv["messages"])
        for tool, count in stats["tool_counts"].items():
            total_tool_counts[tool] = total_tool_counts.get(tool, 0) + count
        page_num = (i // PROMPTS_PER_PAGE) + 1
        for commit_hash, commit_msg, commit_ts in stats["commits"]:
            all_commits.append((commit_ts, commit_hash, commit_msg, page_num, i))
    total_tool_calls = sum(total_tool_counts.values())
    total_commits = len(all_commits)

    # Build timeline items: prompts and commits merged by timestamp
    timeline_items = []

    # Add prompts
    prompt_num = 0
    for i, conv in enumerate(conversations):
        if conv.get("is_continuation"):
            continue
        if conv["user_text"].startswith("Stop hook feedback:"):
            continue
        prompt_num += 1
        page_num = (i // PROMPTS_PER_PAGE) + 1
        msg_id = make_msg_id(conv["timestamp"])
        link = f"page-{page_num:03d}.html#{msg_id}"
        rendered_content = render_markdown_text(conv["user_text"])

        # Collect all messages including from subsequent continuation conversations
        # This ensures long_texts from continuations appear with the original prompt
        all_messages = list(conv["messages"])
        for j in range(i + 1, len(conversations)):
            if not conversations[j].get("is_continuation"):
                break
            all_messages.extend(conversations[j]["messages"])

        # Analyze conversation for stats (excluding commits from inline display now)
        stats = analyze_conversation(all_messages)
        tool_stats_str = format_tool_stats(stats["tool_counts"])

        long_texts_html = ""
        for lt in stats["long_texts"]:
            rendered_lt = render_markdown_text(lt)
            long_texts_html += _macros.index_long_text(rendered_lt)

        stats_html = _macros.index_stats(tool_stats_str, long_texts_html)

        item_html = _macros.index_item(
            prompt_num, link, conv["timestamp"], rendered_content, stats_html
        )
        timeline_items.append((conv["timestamp"], "prompt", item_html))

    # Add commits as separate timeline items
    for commit_ts, commit_hash, commit_msg, page_num, conv_idx in all_commits:
        item_html = _macros.index_commit(
            commit_hash, commit_msg, commit_ts, _github_repo
        )
        timeline_items.append((commit_ts, "commit", item_html))

    # Sort by timestamp
    timeline_items.sort(key=lambda x: x[0])
    index_items = [item[2] for item in timeline_items]

    index_pagination = generate_index_pagination_html(total_pages)
    index_template = get_template("index.html")
    index_content = index_template.render(
        css=CSS,
        js=JS,
        pagination_html=index_pagination,
        prompt_num=prompt_num,
        total_messages=total_messages,
        total_tool_calls=total_tool_calls,
        total_commits=total_commits,
        total_pages=total_pages,
        index_items_html="".join(index_items),
    )
    index_path = output_dir / "index.html"
    index_path.write_text(index_content, encoding="utf-8")
    click.echo(
        f"Generated {index_path.resolve()} ({total_convs} prompts, {total_pages} pages)"
    )


@cli.command("web")
@click.argument("session_id", required=False)
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    help="Output directory. If not specified, writes to temp dir and opens in browser.",
)
@click.option(
    "-a",
    "--output-auto",
    is_flag=True,
    help="Auto-name output subdirectory based on session ID (uses -o as parent, or current dir).",
)
@click.option("--token", help="API access token (auto-detected from keychain on macOS)")
@click.option(
    "--org-uuid", help="Organization UUID (auto-detected from ~/.claude.json)"
)
@click.option(
    "--repo",
    help="GitHub repo (owner/name) for commit links. Auto-detected from git push output if not specified.",
)
@click.option(
    "--gist",
    is_flag=True,
    help="Upload to GitHub Gist and output a gisthost.github.io URL.",
)
@click.option(
    "--json",
    "include_json",
    is_flag=True,
    help="Include the JSON session data in the output directory.",
)
@click.option(
    "--open",
    "open_browser",
    is_flag=True,
    help="Open the generated index.html in your default browser (default if no -o specified).",
)
def web_cmd(
    session_id,
    output,
    output_auto,
    token,
    org_uuid,
    repo,
    gist,
    include_json,
    open_browser,
):
    """Select and convert a web session from the Claude API to HTML.

    If SESSION_ID is not provided, displays an interactive picker to select a session.
    """
    try:
        token, org_uuid = resolve_credentials(token, org_uuid)
    except click.ClickException:
        raise

    # If no session ID provided, show interactive picker
    if session_id is None:
        try:
            sessions_data = fetch_sessions(token, org_uuid)
        except httpx.HTTPStatusError as e:
            raise click.ClickException(
                f"API request failed: {e.response.status_code} {e.response.text}"
            )
        except httpx.RequestError as e:
            raise click.ClickException(f"Network error: {e}")

        sessions = sessions_data.get("data", [])
        if not sessions:
            raise click.ClickException("No sessions found.")

        # Build choices for questionary
        choices = []
        for s in sessions:
            sid = s.get("id", "unknown")
            title = s.get("title", "Untitled")
            created_at = s.get("created_at", "")
            # Truncate title if too long
            if len(title) > 50:
                title = title[:47] + "..."
            display = f"{created_at[:19] if created_at else 'N/A':19}  {title}"
            choices.append(questionary.Choice(title=display, value=sid))

        selected = questionary.select(
            "Select a session to import:",
            choices=choices,
        ).ask()

        if selected is None:
            # User cancelled
            raise click.ClickException("No session selected.")

        session_id = selected

    # Fetch the session
    click.echo(f"Fetching session {session_id}...")
    try:
        session_data = fetch_session(token, org_uuid, session_id)
    except httpx.HTTPStatusError as e:
        raise click.ClickException(
            f"API request failed: {e.response.status_code} {e.response.text}"
        )
    except httpx.RequestError as e:
        raise click.ClickException(f"Network error: {e}")

    # Determine output directory and whether to open browser
    # If no -o specified, use temp dir and open browser by default
    auto_open = output is None and not gist and not output_auto
    if output_auto:
        # Use -o as parent dir (or current dir), with auto-named subdirectory
        parent_dir = Path(output) if output else Path(".")
        output = parent_dir / session_id
    elif output is None:
        output = Path(tempfile.gettempdir()) / f"claude-session-{session_id}"

    output = Path(output)
    click.echo(f"Generating HTML in {output}/...")
    generate_html_from_session_data(session_data, output, github_repo=repo)

    # Show output directory
    click.echo(f"Output: {output.resolve()}")

    # Save JSON session data if requested
    if include_json:
        output.mkdir(exist_ok=True)
        json_dest = output / f"{session_id}.json"
        with open(json_dest, "w") as f:
            json.dump(session_data, f, indent=2)
        json_size_kb = json_dest.stat().st_size / 1024
        click.echo(f"JSON: {json_dest} ({json_size_kb:.1f} KB)")

    if gist:
        # Inject gist preview JS and create gist
        inject_gist_preview_js(output)
        click.echo("Creating GitHub gist...")
        gist_id, gist_url = create_gist(output)
        preview_url = f"https://gisthost.github.io/?{gist_id}/index.html"
        click.echo(f"Gist: {gist_url}")
        click.echo(f"Preview: {preview_url}")

    if open_browser or auto_open:
        index_url = (output / "index.html").resolve().as_uri()
        webbrowser.open(index_url)


@cli.command("all")
@click.option(
    "-s",
    "--source",
    type=click.Path(exists=True),
    help="Source directory containing Claude projects (default: ~/.claude/projects).",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    default="./claude-archive",
    help="Output directory for the archive (default: ./claude-archive).",
)
@click.option(
    "--include-agents",
    is_flag=True,
    help="Include agent-* session files (excluded by default).",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would be converted without creating files.",
)
@click.option(
    "--open",
    "open_browser",
    is_flag=True,
    help="Open the generated archive in your default browser.",
)
@click.option(
    "-q",
    "--quiet",
    is_flag=True,
    help="Suppress all output except errors.",
)
@click.option(
    "--no-search-index",
    is_flag=True,
    help="Skip generating the search index (faster, smaller output).",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["html", "duckdb", "both"]),
    default="html",
    help="Output format: html (default), duckdb, or both.",
)
@click.option(
    "--include-thinking",
    is_flag=True,
    help="Include thinking blocks in DuckDB export (can be large).",
)
def all_cmd(
    source,
    output,
    include_agents,
    dry_run,
    open_browser,
    quiet,
    no_search_index,
    output_format,
    include_thinking,
):
    """Convert all local Claude Code sessions to a browsable HTML archive.

    Creates a directory structure with:
    - Master index listing all projects
    - Per-project pages listing sessions
    - Individual session transcripts
    """
    # Default source folder
    if source is None:
        source = Path.home() / ".claude" / "projects"
    else:
        source = Path(source)

    if not source.exists():
        raise click.ClickException(f"Source directory not found: {source}")

    output = Path(output)

    if not quiet:
        click.echo(f"Scanning {source}...")

    projects = find_all_sessions(source, include_agents=include_agents)

    if not projects:
        if not quiet:
            click.echo("No sessions found.")
        return

    # Calculate totals
    total_sessions = sum(len(p["sessions"]) for p in projects)

    if not quiet:
        click.echo(f"Found {len(projects)} projects with {total_sessions} sessions")

    if dry_run:
        # Dry-run always outputs (it's the point of dry-run), but respects --quiet
        if not quiet:
            click.echo("\nDry run - would convert:")
            for project in projects:
                click.echo(
                    f"\n  {project['name']} ({len(project['sessions'])} sessions)"
                )
                for session in project["sessions"][:3]:  # Show first 3
                    mod_time = datetime.fromtimestamp(session["mtime"])
                    click.echo(
                        f"    - {session['path'].stem} ({mod_time.strftime('%Y-%m-%d')})"
                    )
                if len(project["sessions"]) > 3:
                    click.echo(f"    ... and {len(project['sessions']) - 3} more")
        return

    if not quiet:
        click.echo(f"\nGenerating archive in {output}...")

    # Progress callback for non-quiet mode
    def on_progress(project_name, session_name, current, total):
        if not quiet and current % 10 == 0:
            click.echo(f"  Processed {current}/{total} sessions...")

    stats = None
    duckdb_stats = None

    # Generate HTML if requested
    if output_format in ("html", "both"):
        stats = generate_batch_html(
            source,
            output,
            include_agents=include_agents,
            progress_callback=on_progress,
            no_search_index=no_search_index,
        )

    # Generate DuckDB if requested
    if output_format in ("duckdb", "both"):
        if not quiet:
            if output_format == "both":
                click.echo("\nGenerating DuckDB archive...")
        duckdb_stats = generate_duckdb_archive(
            source,
            output,
            include_agents=include_agents,
            include_thinking=include_thinking,
            progress_callback=on_progress if output_format == "duckdb" else None,
        )
        if stats is None:
            stats = duckdb_stats

    # Report any failures
    if stats and stats["failed_sessions"]:
        click.echo(f"\nWarning: {len(stats['failed_sessions'])} session(s) failed:")
        for failure in stats["failed_sessions"]:
            click.echo(
                f"  {failure['project']}/{failure['session']}: {failure['error']}"
            )

    if not quiet and stats:
        click.echo(
            f"\nGenerated archive with {stats['total_projects']} projects, "
            f"{stats['total_sessions']} sessions"
        )
        click.echo(f"Output: {output.resolve()}")
        if output_format in ("duckdb", "both") and duckdb_stats:
            click.echo(f"DuckDB: {duckdb_stats['db_path']}")

    if open_browser and output_format in ("html", "both"):
        index_url = (output / "index.html").resolve().as_uri()
        webbrowser.open(index_url)


def main():
    cli()
