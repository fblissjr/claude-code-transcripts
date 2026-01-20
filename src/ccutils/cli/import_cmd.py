"""Import command for Claude.ai account exports."""

import html
import json
import tempfile
import webbrowser
from pathlib import Path

import click

from ..parsers.claude_ai import parse_claude_ai_export, load_export_files
from ..export import generate_html
from ..schemas.simple import create_duckdb_schema


@click.command("import")
@click.argument("export_path", type=click.Path(exists=True))
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    help="Output path. For HTML: directory. For DuckDB: .duckdb file.",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["html", "duckdb"]),
    default="html",
    help="Output format: html (default) or duckdb.",
)
@click.option(
    "--conversation-id",
    "-c",
    "conversation_ids",
    multiple=True,
    help="Filter by conversation UUID (can specify multiple).",
)
@click.option(
    "--include-thinking/--no-thinking",
    default=True,
    help="Include thinking blocks (default: yes).",
)
@click.option(
    "--interactive",
    "-i",
    is_flag=True,
    help="Interactively select conversations to export.",
)
@click.option(
    "--open",
    "open_browser",
    is_flag=True,
    help="Open the result in browser after export (HTML only).",
)
@click.option(
    "--list",
    "list_only",
    is_flag=True,
    help="List conversations in the export without converting.",
)
def import_cmd(
    export_path,
    output,
    output_format,
    conversation_ids,
    include_thinking,
    interactive,
    open_browser,
    list_only,
):
    """Import a Claude.ai account export (from Settings > Privacy).

    EXPORT_PATH should be the directory containing conversations.json,
    projects.json, etc.

    Examples:

    \b
      # Export to HTML (opens in browser)
      ccutils import ./my-claude-export --open

    \b
      # Export specific conversations to DuckDB
      ccutils import ./export -c abc123 -c def456 --format duckdb -o data.duckdb

    \b
      # Interactive selection
      ccutils import ./export --interactive
    """
    export_path = Path(export_path)

    # Validate export directory
    if not (export_path / "conversations.json").exists():
        raise click.ClickException(
            f"conversations.json not found in {export_path}. "
            "This doesn't appear to be a valid Claude.ai export."
        )

    # Load export data for listing/interactive modes
    if list_only or interactive:
        data = load_export_files(export_path)
        conversations = data["conversations"]

        if not conversations:
            click.echo("No conversations found in export.")
            return

        if list_only:
            _list_conversations(conversations)
            return

        if interactive:
            conversation_ids = _interactive_select(conversations)
            if not conversation_ids:
                click.echo("No conversations selected.")
                return

    # Convert conversation_ids tuple to list (or None)
    conv_filter = list(conversation_ids) if conversation_ids else None

    # Parse the export
    click.echo(f"Parsing Claude.ai export from {export_path}...")
    parsed = parse_claude_ai_export(
        export_path,
        conversation_ids=conv_filter,
        include_thinking=include_thinking,
    )

    loglines = parsed["loglines"]
    metadata = parsed["_metadata"]

    if not loglines:
        click.echo("No messages found to export.")
        return

    # Count conversations
    session_ids = set(ll.get("sessionId") for ll in loglines)
    click.echo(
        f"Found {len(loglines)} messages across {len(session_ids)} conversations"
    )

    if output_format == "html":
        _export_to_html(parsed, output, open_browser)
    elif output_format == "duckdb":
        _export_to_duckdb(parsed, output, include_thinking)


def _list_conversations(conversations):
    """List all conversations in the export."""
    click.echo(f"\nFound {len(conversations)} conversations:\n")
    for conv in sorted(
        conversations, key=lambda c: c.get("updated_at", ""), reverse=True
    ):
        name = conv.get("name", "(untitled)")
        uuid = conv.get("uuid", "")
        msg_count = len(conv.get("chat_messages", []))
        updated = conv.get("updated_at", "")[:10]  # Just the date
        click.echo(f"  {uuid[:8]}  {updated}  ({msg_count:3d} msgs)  {name[:60]}")


def _interactive_select(conversations):
    """Interactively select conversations using questionary."""
    try:
        import questionary
    except ImportError:
        raise click.ClickException(
            "Interactive mode requires questionary. Install with: uv add questionary"
        )

    # Build choices
    choices = []
    for conv in sorted(
        conversations, key=lambda c: c.get("updated_at", ""), reverse=True
    ):
        name = conv.get("name", "(untitled)")
        uuid = conv.get("uuid", "")
        msg_count = len(conv.get("chat_messages", []))
        updated = conv.get("updated_at", "")[:10]
        label = f"{updated} ({msg_count:3d} msgs) {name[:50]}"
        choices.append(questionary.Choice(title=label, value=uuid))

    # Multi-select
    selected = questionary.checkbox(
        "Select conversations to export:",
        choices=choices,
    ).ask()

    return selected if selected else []


def _export_to_html(parsed, output, open_browser):
    """Export parsed data to HTML."""
    loglines = parsed["loglines"]

    # Group loglines by session for multi-conversation exports
    sessions = {}
    for ll in loglines:
        sid = ll.get("sessionId", "unknown")
        if sid not in sessions:
            sessions[sid] = []
        sessions[sid].append(ll)

    # Determine output directory
    if output is None:
        output = Path(tempfile.gettempdir()) / "claude-ai-export"
        auto_open = True
    else:
        output = Path(output)
        auto_open = False

    output.mkdir(parents=True, exist_ok=True)

    # For each session, write temp file and generate HTML
    for session_id, session_loglines in sessions.items():
        # Create temp JSON file in ccutils format
        session_data = {"loglines": session_loglines}

        # Use session name from first message's conversation, or truncated ID
        session_name = session_id[:8]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
            json.dump(session_data, tmp)
            tmp_path = Path(tmp.name)

        try:
            # Generate HTML to session subdirectory
            session_output = output / session_name
            generate_html(tmp_path, session_output)
            click.echo(f"  Generated: {session_output}")
        finally:
            tmp_path.unlink()  # Clean up temp file

    # Create index if multiple sessions
    if len(sessions) > 1:
        _create_multi_session_index(output, sessions, parsed["_metadata"])

    click.echo(f"\nOutput: {output.resolve()}")

    if open_browser or auto_open:
        if len(sessions) == 1:
            # Single session - open its index
            session_name = list(sessions.keys())[0][:8]
            index_url = (output / session_name / "index.html").resolve().as_uri()
        else:
            # Multiple sessions - open master index
            index_url = (output / "index.html").resolve().as_uri()
        webbrowser.open(index_url)


def _create_multi_session_index(output_dir, sessions, metadata):
    """Create an index.html linking to all session directories."""
    html_content = """<!DOCTYPE html>
<html>
<head>
    <title>Claude.ai Export</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; max-width: 800px; margin: 40px auto; padding: 0 20px; }
        h1 { color: #1a1a2e; }
        .session { padding: 12px; margin: 8px 0; background: #f5f5f5; border-radius: 8px; }
        .session a { color: #4a4a6a; text-decoration: none; font-weight: 500; }
        .session a:hover { color: #1a1a2e; }
        .meta { color: #666; font-size: 0.9em; margin-top: 4px; }
        .stats { color: #888; font-size: 0.85em; margin-top: 20px; }
    </style>
</head>
<body>
    <h1>Claude.ai Export</h1>
    <p class="stats">Source: Claude.ai account export | Conversations: {conv_count}</p>
    <div class="sessions">
""".format(
        conv_count=len(sessions)
    )

    for session_id, loglines in sessions.items():
        session_name = html.escape(session_id[:8])
        msg_count = len(loglines)
        # Try to get conversation name from metadata or first message
        conv_name = html.escape(session_id)  # fallback
        html_content += f"""        <div class="session">
            <a href="{session_name}/index.html">{conv_name}</a>
            <div class="meta">{msg_count} messages</div>
        </div>
"""

    html_content += """    </div>
</body>
</html>"""

    (output_dir / "index.html").write_text(html_content)


def _export_to_duckdb(parsed, output, include_thinking):
    """Export parsed data to DuckDB."""
    from datetime import datetime

    loglines = parsed["loglines"]
    metadata = parsed["_metadata"]

    # Determine output path
    if output is None:
        output = Path("claude-ai-export.duckdb")
    else:
        output = Path(output)
        if not output.suffix:
            output = output.with_suffix(".duckdb")

    output.parent.mkdir(parents=True, exist_ok=True)

    click.echo(f"Creating DuckDB database: {output}")

    # Create schema
    conn = create_duckdb_schema(output)

    # Group by session
    sessions = {}
    for ll in loglines:
        sid = ll.get("sessionId", "unknown")
        if sid not in sessions:
            sessions[sid] = []
        sessions[sid].append(ll)

    # Process each session
    tool_use_map = {}  # Global map for tool use -> result linking
    thinking_id = 0

    for session_id, session_loglines in sessions.items():
        first_ts = None
        last_ts = None
        user_count = 0
        assistant_count = 0
        tool_count = 0

        for ll in session_loglines:
            msg_type = ll.get("type")
            timestamp_str = ll.get("timestamp", "")
            msg_uuid = ll.get("uuid", "")
            message_data = ll.get("message", {})
            content = message_data.get("content", [])

            # Parse timestamp
            timestamp = None
            if timestamp_str:
                try:
                    timestamp = datetime.fromisoformat(
                        timestamp_str.replace("Z", "+00:00")
                    )
                    if first_ts is None:
                        first_ts = timestamp
                    last_ts = timestamp
                except ValueError:
                    pass

            # Count by type
            if msg_type == "user":
                user_count += 1
            else:
                assistant_count += 1

            # Process content blocks
            has_tool_use = False
            has_tool_result = False
            has_thinking = False
            text_parts = []

            for block in content:
                if not isinstance(block, dict):
                    continue
                block_type = block.get("type")

                if block_type == "text":
                    text_parts.append(block.get("text", ""))

                elif block_type == "tool_use":
                    has_tool_use = True
                    tool_count += 1
                    tool_id = block.get("id") or f"tool_{session_id}_{msg_uuid}"
                    tool_name = block.get("name", "")
                    tool_input = block.get("input", {})
                    input_summary = json.dumps(tool_input)[:2000]

                    tool_use_map[tool_id] = {
                        "message_id": msg_uuid,
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
                        output_text = result_content[:2000]
                    else:
                        output_text = str(result_content)[:2000]

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
                                msg_uuid,
                                tool_info["tool_name"],
                                tool_info["input_json"],
                                tool_info["input_summary"],
                                output_text,
                                tool_info["timestamp"],
                            ],
                        )
                        del tool_use_map[tool_id]

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
                                msg_uuid,
                                thinking_text,
                                timestamp,
                            ],
                        )

            text_content = " ".join(text_parts)

            # Insert message
            conn.execute(
                """
                INSERT INTO messages (
                    id, session_id, parent_id, type, timestamp, model,
                    content, content_json, has_tool_use, has_tool_result, has_thinking,
                    is_sidechain
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                [
                    msg_uuid,
                    session_id,
                    None,  # parent_id not available in Claude.ai export
                    msg_type,
                    timestamp,
                    message_data.get("model"),
                    text_content,
                    json.dumps(content) if content else None,
                    has_tool_use,
                    has_tool_result,
                    has_thinking,
                    False,  # is_sidechain
                ],
            )

        # Insert session
        conn.execute(
            """
            INSERT INTO sessions (
                session_id, project_path, project_name, first_timestamp, last_timestamp,
                message_count, user_message_count, assistant_message_count,
                tool_use_count, cwd, git_branch, version,
                is_agent, agent_id, parent_session_id, depth_level
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            [
                session_id,
                "claude.ai",  # project_path
                "Claude.ai Import",  # project_name
                first_ts,
                last_ts,
                user_count + assistant_count,
                user_count,
                assistant_count,
                tool_count,
                None,  # cwd
                None,  # git_branch
                None,  # version
                False,  # is_agent
                None,  # agent_id
                None,  # parent_session_id
                0,  # depth_level
            ],
        )

    conn.close()
    click.echo(f"Exported {len(sessions)} conversations to {output}")
