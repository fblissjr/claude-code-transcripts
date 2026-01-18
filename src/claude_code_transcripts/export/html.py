"""HTML generation for Claude Code session transcripts.

This module provides functions for rendering Claude Code sessions as HTML,
including message rendering, CSS styling, and pagination.
"""

import html
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path

import markdown
from jinja2 import Environment, PackageLoader

from ..parsers import (
    extract_searchable_content,
    extract_text_from_content,
    find_all_sessions,
    get_session_summary,
    parse_session_file,
    PROMPTS_PER_PAGE,
)

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

# Regex to detect GitHub repo from git push output
GITHUB_REPO_PATTERN = re.compile(
    r"github\.com/([a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+)/pull/new/"
)

LONG_TEXT_THRESHOLD = (
    300  # Characters - text blocks longer than this are shown in index
)

# Module-level variable for GitHub repo (set by generate_html)
_github_repo = None


def set_github_repo(repo):
    """Set the module-level GitHub repo for rendering commit links."""
    global _github_repo
    _github_repo = repo


def get_github_repo():
    """Get the current GitHub repo setting."""
    return _github_repo


def detect_github_repo_from_cwd():
    """Detect GitHub repo from current working directory's git remote.

    Runs `git remote get-url origin` and parses the GitHub URL.
    Supports both HTTPS and SSH URL formats.

    Returns repo in format "owner/repo" or None if not detected.
    """
    try:
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            url = result.stdout.strip()
            # HTTPS: https://github.com/owner/repo.git
            if "github.com" in url:
                # Remove .git suffix if present
                url = url.rstrip("/").removesuffix(".git")
                # Extract owner/repo
                if "github.com/" in url:
                    return url.split("github.com/")[1]
                elif "github.com:" in url:  # SSH format: git@github.com:owner/repo
                    return url.split("github.com:")[1]
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass
    return None


def detect_github_repo(loglines):
    """Detect GitHub repo from session loglines.

    Looks for GitHub URLs in git push output within tool results.
    Returns repo in format "owner/repo" or None if not detected.
    """
    for entry in loglines:
        message = entry.get("message", {})
        content = message.get("content", [])
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                result_content = block.get("content", "")
                if isinstance(result_content, str):
                    match = GITHUB_REPO_PATTERN.search(result_content)
                    if match:
                        return match.group(1)
    return None


def format_json(obj):
    """Format an object as JSON HTML."""
    try:
        if isinstance(obj, str):
            obj = json.loads(obj)
        formatted = json.dumps(obj, indent=2, ensure_ascii=False)
        return f'<pre class="json">{html.escape(formatted)}</pre>'
    except (json.JSONDecodeError, TypeError):
        return f"<pre>{html.escape(str(obj))}</pre>"


def render_markdown_text(text):
    """Render markdown text to HTML."""
    if not text:
        return ""
    return markdown.markdown(text, extensions=["fenced_code", "tables"])


def is_json_like(text):
    """Check if text appears to be JSON."""
    if not text or not isinstance(text, str):
        return False
    text = text.strip()
    return (text.startswith("{") and text.endswith("}")) or (
        text.startswith("[") and text.endswith("]")
    )


def render_todo_write(tool_input, tool_id):
    """Render TodoWrite tool calls."""
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
    """Render a single content block to HTML."""
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
        has_images = False

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
        elif isinstance(content, list):
            # Handle tool result content that contains multiple blocks
            parts = []
            for item in content:
                if isinstance(item, dict):
                    item_type = item.get("type", "")
                    if item_type == "text":
                        text = item.get("text", "")
                        if text:
                            parts.append(f"<pre>{html.escape(text)}</pre>")
                    elif item_type == "image":
                        source = item.get("source", {})
                        media_type = source.get("media_type", "image/png")
                        data = source.get("data", "")
                        if data:
                            parts.append(_macros.image_block(media_type, data))
                            has_images = True
                    else:
                        # Unknown type, render as JSON
                        parts.append(format_json(item))
                else:
                    # Non-dict item, escape as text
                    parts.append(f"<pre>{html.escape(str(item))}</pre>")
            content_html = "".join(parts) if parts else format_json(content)
        elif is_json_like(content):
            content_html = format_json(content)
        else:
            content_html = format_json(content)
        return _macros.tool_result(content_html, is_error, has_images)
    else:
        return format_json(block)


def render_user_message_content(message_data):
    """Render user message content to HTML."""
    content = message_data.get("content", "")
    if isinstance(content, str):
        if is_json_like(content):
            return _macros.user_content(format_json(content))
        return _macros.user_content(render_markdown_text(content))
    elif isinstance(content, list):
        return "".join(render_content_block(block) for block in content)
    return f"<p>{html.escape(str(content))}</p>"


def render_assistant_message(message_data):
    """Render assistant message content to HTML."""
    content = message_data.get("content", [])
    if not isinstance(content, list):
        return f"<p>{html.escape(str(content))}</p>"
    return "".join(render_content_block(block) for block in content)


def make_msg_id(timestamp):
    """Create a message ID from a timestamp."""
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

    return " . ".join(parts)


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
    """Render a single message to HTML."""
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


def generate_pagination_html(current_page, total_pages):
    """Generate pagination HTML for a transcript page."""
    return _macros.pagination(current_page, total_pages)


def generate_index_pagination_html(total_pages):
    """Generate pagination HTML for the index page."""
    return _macros.index_pagination(total_pages)


# CSS constant
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
.thinking p:first-child { margin-top: 0; }
.thinking p:last-child { margin-bottom: 0; }
.tool-use { background: var(--tool-bg); border: 1px solid var(--tool-border); border-radius: 8px; padding: 12px; margin: 12px 0; }
.tool-name { font-size: 0.9rem; font-weight: 600; color: var(--tool-border); margin-bottom: 8px; display: flex; align-items: center; gap: 8px; }
.tool-name .tool-icon { width: 16px; height: 16px; }
.tool-description { font-size: 0.85rem; color: var(--text-muted); margin-bottom: 8px; font-style: italic; }
.tool-use pre { background: var(--code-bg); color: var(--code-text); padding: 12px; border-radius: 6px; overflow-x: auto; margin: 0; font-size: 0.85rem; }
.tool-result { background: var(--tool-result-bg); border: 1px solid #4caf50; border-radius: 8px; padding: 12px; margin: 12px 0; }
.tool-result.error { background: var(--tool-error-bg); border-color: #f44336; }
.tool-result pre { background: var(--code-bg); color: var(--code-text); padding: 12px; border-radius: 6px; overflow-x: auto; margin: 0; font-size: 0.85rem; white-space: pre-wrap; word-wrap: break-word; }
.tool-result.has-images pre { margin-bottom: 12px; }
.tool-result img { max-width: 100%; height: auto; border-radius: 6px; margin-top: 8px; }
.message-content pre { background: var(--code-bg); color: var(--code-text); padding: 12px; border-radius: 6px; overflow-x: auto; margin: 12px 0; font-size: 0.85rem; white-space: pre-wrap; word-wrap: break-word; }
.message-content code { background: rgba(0,0,0,0.08); padding: 2px 6px; border-radius: 4px; font-size: 0.9em; }
.message-content pre code { background: transparent; padding: 0; }
.message-content img { max-width: 100%; height: auto; border-radius: 8px; margin: 12px 0; }
.message-content ul, .message-content ol { margin: 12px 0; padding-left: 24px; }
.message-content li { margin: 4px 0; }
.message-content table { border-collapse: collapse; margin: 12px 0; width: 100%; }
.message-content th, .message-content td { border: 1px solid #ddd; padding: 8px 12px; text-align: left; }
.message-content th { background: rgba(0,0,0,0.05); font-weight: 600; }
.pagination { display: flex; justify-content: center; gap: 8px; margin: 24px 0; flex-wrap: wrap; }
.pagination a, .pagination span { padding: 8px 16px; border-radius: 6px; text-decoration: none; font-size: 0.9rem; }
.pagination a { background: var(--card-bg); color: var(--user-border); border: 1px solid var(--user-border); }
.pagination a:hover { background: var(--user-bg); }
.pagination span.current { background: var(--user-border); color: white; }
.pagination span.ellipsis { background: transparent; border: none; color: var(--text-muted); }
@media (max-width: 600px) { body { padding: 8px; } .message-content { padding: 12px; } .pagination a, .pagination span { padding: 6px 12px; } }
.truncatable { position: relative; max-height: 200px; overflow: hidden; transition: max-height 0.3s ease; }
.truncatable.truncated::after { content: ''; position: absolute; bottom: 0; left: 0; right: 0; height: 50px; background: linear-gradient(to bottom, transparent, var(--tool-result-bg)); pointer-events: none; }
.truncatable.expanded { max-height: none; }
.truncatable.expanded::after { display: none; }
.expand-btn { display: block; margin-top: 8px; padding: 4px 8px; background: transparent; border: 1px solid var(--text-muted); border-radius: 4px; color: var(--text-muted); font-size: 0.8rem; cursor: pointer; }
.expand-btn:hover { background: rgba(0,0,0,0.05); }
.todo-list { padding: 0; margin: 0; list-style: none; }
.todo-item { display: flex; align-items: flex-start; gap: 8px; padding: 6px 0; border-bottom: 1px solid rgba(0,0,0,0.1); }
.todo-item:last-child { border-bottom: none; }
.todo-status { font-size: 1rem; flex-shrink: 0; }
.todo-content { flex: 1; }
.todo-pending .todo-status::before { content: 'O'; color: #9e9e9e; }
.todo-in_progress .todo-status::before { content: '>'; color: #1976d2; }
.todo-completed .todo-status::before { content: 'X'; color: #4caf50; }
.file-op { margin: 12px 0; }
.file-op-header { display: flex; align-items: center; gap: 8px; padding: 8px 12px; background: rgba(0,0,0,0.05); border-radius: 6px 6px 0 0; font-family: monospace; font-size: 0.85rem; }
.file-op-icon { width: 16px; height: 16px; }
.file-op-path { color: var(--tool-border); font-weight: 500; }
.file-op-content { background: var(--code-bg); color: var(--code-text); padding: 12px; border-radius: 0 0 6px 6px; overflow-x: auto; font-size: 0.85rem; }
.file-op-content pre { margin: 0; white-space: pre-wrap; word-wrap: break-word; }
.diff-old { background: rgba(244, 67, 54, 0.2); padding: 2px 4px; border-radius: 2px; }
.diff-new { background: rgba(76, 175, 80, 0.2); padding: 2px 4px; border-radius: 2px; }
.diff-arrow { color: var(--text-muted); margin: 8px 0; font-size: 1.2rem; text-align: center; }
.commit-card { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); border: 1px solid #0f3460; border-radius: 8px; padding: 12px 16px; margin: 8px 0; display: flex; align-items: center; gap: 12px; }
.commit-icon { color: #4caf50; font-size: 1.2rem; }
.commit-info { flex: 1; }
.commit-hash { font-family: monospace; font-size: 0.85rem; color: #e94560; }
.commit-hash a { color: #e94560; text-decoration: none; }
.commit-hash a:hover { text-decoration: underline; }
.commit-msg { color: #eee; font-size: 0.9rem; margin-top: 4px; }
.index-header { margin-bottom: 24px; }
.index-stats { display: flex; gap: 16px; flex-wrap: wrap; margin-top: 12px; font-size: 0.9rem; color: var(--text-muted); }
.index-stats span { display: flex; align-items: center; gap: 4px; }
.index-list { list-style: none; padding: 0; margin: 0; }
.index-item { padding: 16px; background: var(--card-bg); border-radius: 8px; margin-bottom: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
.index-item-header { display: flex; justify-content: space-between; align-items: flex-start; gap: 12px; margin-bottom: 8px; }
.index-item-num { font-weight: 600; color: var(--user-border); font-size: 0.85rem; }
.index-item-time { color: var(--text-muted); font-size: 0.8rem; }
.index-item-preview { color: var(--text-color); font-size: 0.95rem; }
.index-item-preview p { margin: 0; }
.index-item-stats { margin-top: 8px; font-size: 0.8rem; color: var(--text-muted); }
.index-long-text { margin-top: 8px; padding: 8px; background: rgba(0,0,0,0.03); border-radius: 4px; font-size: 0.85rem; max-height: 100px; overflow: hidden; position: relative; }
.index-long-text::after { content: ''; position: absolute; bottom: 0; left: 0; right: 0; height: 30px; background: linear-gradient(to bottom, transparent, var(--card-bg)); }
.index-item.commit { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); border: 1px solid #0f3460; }
.index-item.commit .index-item-header { margin-bottom: 4px; }
.index-item.commit .commit-hash { font-family: monospace; font-size: 0.85rem; color: #e94560; }
.index-item.commit .commit-hash a { color: #e94560; text-decoration: none; }
.index-item.commit .commit-hash a:hover { text-decoration: underline; }
.index-item.commit .commit-msg { color: #eee; font-size: 0.9rem; }
.index-item.commit .index-item-time { color: #aaa; }
#search-box { display: none; }
#search-box.visible { display: flex; gap: 8px; align-items: center; }
#search-input { padding: 6px 12px; border: 1px solid var(--text-muted); border-radius: 4px; font-size: 0.9rem; width: 200px; }
#search-btn { padding: 6px 10px; background: var(--user-border); color: white; border: none; border-radius: 4px; cursor: pointer; display: flex; align-items: center; }
#search-btn:hover { background: #1565c0; }
#search-modal { border: none; border-radius: 12px; padding: 20px; max-width: 600px; width: 90vw; max-height: 80vh; box-shadow: 0 4px 20px rgba(0,0,0,0.3); }
#search-modal::backdrop { background: rgba(0,0,0,0.5); }
.search-modal-header { display: flex; gap: 8px; margin-bottom: 16px; }
#modal-search-input { flex: 1; padding: 10px 14px; border: 1px solid var(--text-muted); border-radius: 6px; font-size: 1rem; }
#modal-search-btn, #modal-close-btn { padding: 10px 12px; border: none; border-radius: 6px; cursor: pointer; display: flex; align-items: center; }
#modal-search-btn { background: var(--user-border); color: white; }
#modal-close-btn { background: var(--bg-color); }
#search-status { color: var(--text-muted); font-size: 0.9rem; margin-bottom: 12px; }
#search-results { max-height: 50vh; overflow-y: auto; }
.search-result { padding: 12px; border-bottom: 1px solid var(--bg-color); cursor: pointer; }
.search-result:hover { background: var(--bg-color); }
.search-result-title { font-weight: 500; margin-bottom: 4px; }
.search-result-snippet { font-size: 0.85rem; color: var(--text-muted); }
.search-result-snippet mark { background: #fff59d; padding: 0 2px; }
"""

# JavaScript constant (existing code - uses innerHTML for JSON syntax highlighting only)
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


class GistError(Exception):
    """Raised when gist creation fails."""

    pass


def create_gist(output_dir, public=False):
    """Create a GitHub gist from the HTML files in output_dir.

    Returns tuple of (gist_id, gist_url) on success.
    Raises GistError on failure.
    """
    output_dir = Path(output_dir)
    html_files = list(output_dir.glob("*.html"))
    if not html_files:
        raise GistError("No HTML files found to upload to gist.")

    # Build the gh gist create command
    cmd = ["gh", "gist", "create"]
    cmd.extend(str(f) for f in sorted(html_files))
    if public:
        cmd.append("--public")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            error_msg = result.stderr.strip() or "Unknown error"
            raise GistError(f"gh gist create failed: {error_msg}")

        # Parse the gist URL from output
        gist_url = result.stdout.strip()
        if not gist_url:
            raise GistError("No gist URL returned from gh gist create")

        # Extract gist ID from URL
        gist_id = gist_url.rstrip("/").split("/")[-1]

        return gist_id, gist_url

    except subprocess.TimeoutExpired:
        raise GistError("gh gist create timed out")
    except FileNotFoundError:
        raise GistError("gh CLI not found. Install from https://cli.github.com/")


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

    content = template.render(
        css=CSS,
        project_name=project["name"],
        sessions=sessions_data,
    )
    (output_dir / "index.html").write_text(content, encoding="utf-8")


def _generate_master_index(projects, output_dir, has_search_index=False):
    """Generate master index.html listing all projects."""
    template = get_template("master_index.html")

    # Format projects for template
    projects_data = []
    for project in projects:
        if not project["sessions"]:
            continue
        most_recent = datetime.fromtimestamp(project["sessions"][0]["mtime"])
        projects_data.append(
            {
                "name": project["name"],
                "session_count": len(project["sessions"]),
                "most_recent": most_recent.strftime("%Y-%m-%d %H:%M"),
            }
        )

    content = template.render(
        css=CSS,
        projects=projects_data,
        has_search_index=has_search_index,
    )
    (output_dir / "index.html").write_text(content, encoding="utf-8")


def _generate_search_index(projects, output_dir):
    """Generate search-index.js with searchable content from all sessions."""
    all_documents = []

    for project in projects:
        for session in project["sessions"]:
            session_path = session["path"]
            try:
                # Parse session file and extract documents
                data = parse_session_file(session_path)
                loglines = data.get("loglines", [])
                session_docs = extract_searchable_content(
                    loglines, project["name"], session_path.stem
                )
                if session_docs:
                    all_documents.extend(session_docs)
            except Exception:
                # Skip sessions that fail to parse
                pass

    # Build index with version info
    search_index = {
        "version": 1,
        "documents": all_documents,
    }

    # Write as JavaScript file
    js_content = f"var SEARCH_INDEX = {json.dumps(search_index, ensure_ascii=False)};"
    (output_dir / "search-index.js").write_text(js_content, encoding="utf-8")


def generate_multi_session_index(
    output_dir,
    sessions,
    agent_map=None,
    title="Sessions",
):
    """Generate an index page for multiple sessions.

    Args:
        output_dir: Directory to write index.html
        sessions: List of session Paths
        agent_map: Optional dict mapping parent session Path to list of agent Paths
        title: Page title

    Returns:
        Path to generated index.html
    """
    output_dir = Path(output_dir)
    agent_map = agent_map or {}
    template = get_template("multi_session_index.html")

    # Format sessions for template
    sessions_data = []
    for session_path in sessions:
        session_path = Path(session_path)
        stat = session_path.stat()
        mod_time = datetime.fromtimestamp(stat.st_mtime)
        summary = get_session_summary(session_path)

        # Check if this is an agent session
        is_agent = session_path.name.startswith("agent-")

        # Get agent count for parent sessions
        agent_count = len(agent_map.get(session_path, []))

        sessions_data.append(
            {
                "name": session_path.stem,
                "summary": summary,
                "date": mod_time.strftime("%Y-%m-%d %H:%M"),
                "size_kb": stat.st_size / 1024,
                "is_agent": is_agent,
                "agent_count": agent_count,
            }
        )

    content = template.render(
        css=CSS,
        js=JS,
        title=title,
        sessions=sessions_data,
    )

    index_path = output_dir / "index.html"
    index_path.write_text(content, encoding="utf-8")
    return index_path


def generate_html(json_path, output_dir, github_repo=None):
    """Generate HTML transcript from a session file.

    Args:
        json_path: Path to JSON/JSONL session file
        output_dir: Directory to write HTML files
        github_repo: Optional GitHub repo for commit links (format: "owner/repo")

    Returns:
        Path to output directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Load session file (supports both JSON and JSONL)
    data = parse_session_file(json_path)

    loglines = data.get("loglines", [])

    # Auto-detect GitHub repo if not provided
    if github_repo is None:
        # First try to detect from session content (git push output)
        github_repo = detect_github_repo(loglines)
        if not github_repo:
            # Fallback: detect from current working directory's git remote
            github_repo = detect_github_repo_from_cwd()

    # Set module-level variable for render functions
    set_github_repo(github_repo)

    # Group messages into conversations (each starting with a user prompt)
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

        # Check if this is a new user prompt
        is_user_prompt = False
        user_text = None

        if log_type == "user":
            content = message_data.get("content", "")
            text = extract_text_from_content(content)
            if text:
                is_user_prompt = True
                user_text = text

        if is_user_prompt:
            # Start a new conversation
            if current_conv:
                conversations.append(current_conv)
            current_conv = {
                "user_text": user_text,
                "timestamp": timestamp,
                "messages": [(log_type, message_json, timestamp)],
                "is_continuation": bool(is_compact_summary),
            }
        elif current_conv:
            # Add to current conversation
            current_conv["messages"].append((log_type, message_json, timestamp))

    # Don't forget the last conversation
    if current_conv:
        conversations.append(current_conv)

    # Calculate pagination
    total_convs = len(conversations)
    total_pages = (total_convs + PROMPTS_PER_PAGE - 1) // PROMPTS_PER_PAGE

    # Generate each page
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
        all_messages = list(conv["messages"])
        for j in range(i + 1, len(conversations)):
            if not conversations[j].get("is_continuation"):
                break
            all_messages.extend(conversations[j]["messages"])

        # Analyze conversation for stats
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
            commit_hash, commit_msg, commit_ts, get_github_repo()
        )
        timeline_items.append((commit_ts, "commit", item_html))

    # Sort by timestamp
    timeline_items.sort(key=lambda x: x[0])
    index_items = [item[2] for item in timeline_items]

    # Generate index page
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

    return output_dir
