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

# Star schema imports (from modular package)
from .schemas.star import (
    create_semantic_model,
    create_star_schema,
    generate_dimension_key,
    get_model_family,
    get_time_of_day,
    get_tool_category,
    run_llm_enrichment,
    run_session_insights_enrichment,
    run_star_schema_etl,
    export_star_schema_to_json,
    TOOL_CATEGORIES,
)

# Session parsing imports (from modular package)
from .parsers import (
    extract_searchable_content,
    extract_session_metadata,
    extract_session_slug,
    extract_snippet,
    extract_text_from_content,
    get_session_summary,
    parse_session_file,
    PROMPTS_PER_PAGE,
    # Session discovery
    build_session_choices,
    find_agent_sessions,
    find_all_sessions,
    find_local_sessions,
    flatten_selected_sessions,
    get_project_display_name,
    matches_project_filter,
)

# Simple schema imports (from modular package)
from .schemas import (
    create_duckdb_schema,
    export_session_to_duckdb,
    export_sessions_to_json,
    _extract_session_data,
    resolve_schema_format,
)

# API client imports (from modular package)
from .api import (
    API_BASE_URL,
    ANTHROPIC_VERSION,
    CredentialsError,
    get_access_token_from_keychain,
    get_org_uuid_from_config,
    get_api_headers,
    fetch_sessions,
    fetch_session,
)

# Export format imports (from modular package)
from .export import (
    generate_duckdb_archive,
    # HTML generation
    CSS,
    JS,
    COMMIT_PATTERN,
    GITHUB_REPO_PATTERN,
    GIST_PREVIEW_JS,
    GistError,
    LONG_TEXT_THRESHOLD,
    _macros,  # Template macros for CLI functions
    analyze_conversation,
    create_gist,
    detect_github_repo,
    detect_github_repo_from_cwd,
    format_json,
    format_tool_stats,
    generate_batch_html,
    generate_html,
    generate_index_pagination_html,
    generate_multi_session_index,
    generate_pagination_html,
    get_github_repo,
    get_template,
    inject_gist_preview_js,
    is_json_like,
    is_tool_result_message,
    make_msg_id,
    render_assistant_message,
    render_bash_tool,
    render_content_block,
    render_edit_tool,
    render_markdown_text,
    render_message,
    render_todo_write,
    render_user_message_content,
    render_write_tool,
    set_github_repo,
)


@click.group(cls=DefaultGroup, default="local", default_if_no_args=True)
@click.version_option(None, "-v", "--version", package_name="claude-code-transcripts")
def cli():
    """Convert Claude Code sessions to HTML pages or DuckDB databases.

    Export individual sessions to HTML, or batch export all sessions
    to a browsable HTML archive or DuckDB database for analytics.
    """
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
    default=100,
    help="Maximum number of sessions to show (default: 100).",
)
@click.option(
    "-p",
    "--project",
    "project_filter",
    help="Filter by project name (partial match, case-insensitive).",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["html", "duckdb", "duckdb-star", "json", "json-star"]),
    default="html",
    help="Output format: html (default), duckdb[-star], or json[-star].",
)
@click.option(
    "--schema",
    "schema_type",
    type=click.Choice(["simple", "star"]),
    default=None,
    help="Data schema: simple (4 tables) or star (dimensional). Auto-inferred from format.",
)
@click.option(
    "--include-subagents",
    is_flag=True,
    help="Auto-include related agent sessions (recursive by default).",
)
@click.option(
    "--include-thinking",
    is_flag=True,
    help="Include thinking blocks in DuckDB/JSON export (can be large).",
)
@click.option(
    "--expand-chains",
    is_flag=True,
    help="Show individual sessions in chains instead of collapsed view.",
)
def local_cmd(
    output,
    output_auto,
    repo,
    gist,
    include_json,
    open_browser,
    limit,
    project_filter,
    output_format,
    schema_type,
    include_subagents,
    include_thinking,
    expand_chains,
):
    """Select and convert local Claude Code sessions to HTML or DuckDB.

    Supports multi-select: use SPACE to select multiple sessions, ENTER to confirm.
    """
    projects_folder = Path.home() / ".claude" / "projects"

    if not projects_folder.exists():
        click.echo(f"Projects folder not found: {projects_folder}")
        click.echo("No local Claude Code sessions available.")
        return

    click.echo("Loading local sessions...")
    results = find_local_sessions(
        projects_folder, limit=limit, project_filter=project_filter
    )

    if not results:
        click.echo("No local sessions found.")
        return

    # Count related agents for each session
    agent_counts = {}
    if include_subagents:
        session_paths = [filepath for filepath, _, _ in results]
        agent_map = find_agent_sessions(session_paths, recursive=True)
        for filepath, agents in agent_map.items():
            agent_counts[filepath] = len(agents)

    # Group sessions by project for better organization
    sessions_by_project = {}
    for filepath, summary, slug in results:
        project_key = filepath.parent.name
        if project_key not in sessions_by_project:
            sessions_by_project[project_key] = []
        sessions_by_project[project_key].append((filepath, summary, slug))

    # Build choices for questionary with project separators
    # Default: chains are collapsed (selecting a chain selects all sessions)
    # With --expand-chains: individual sessions shown with chain headers
    choices = build_session_choices(
        sessions_by_project,
        expand_chains=expand_chains,
        agent_counts=agent_counts if include_subagents else None,
    )

    # Multi-select with checkbox
    selected = questionary.checkbox(
        "Select sessions to convert (SPACE to select, ENTER to confirm):",
        choices=choices,
    ).ask()

    if not selected:
        click.echo("No sessions selected.")
        return

    # Flatten selection: chains return lists of paths, standalone return single paths
    selected = flatten_selected_sessions(selected)
    click.echo(f"Selected {len(selected)} session(s)")

    # Auto-include subagents if requested
    agent_map = {}
    if include_subagents:
        agent_map = find_agent_sessions(selected, recursive=True)
        agent_count = sum(len(agents) for agents in agent_map.values())
        if agent_count > 0:
            click.echo(f"Including {agent_count} related agent session(s)")
            for parent, agents in agent_map.items():
                for agent_path in agents:
                    if agent_path not in selected:
                        selected.append(agent_path)

    # Determine output path - default to ./claude-archive (not temp dir)
    if output_auto:
        parent_dir = Path(output) if output else Path(".")
        if len(selected) == 1:
            output = parent_dir / selected[0].stem
        else:
            output = (
                parent_dir / f"multi-session-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            )
    elif output is None:
        # Default to local ./claude-archive directory
        output = Path("./claude-archive")

    output = Path(output)

    # Resolve schema and format from potentially compound format names
    schema, fmt = resolve_schema_format(schema_type, output_format)

    # Execute based on format
    if fmt == "html":
        if len(selected) == 1 and not agent_map:
            # Single session, no agents - use existing simple path
            generate_html(selected[0], output, github_repo=repo)
        else:
            # Multiple sessions or has agents - use batch structure with master index
            output.mkdir(parents=True, exist_ok=True)
            for idx, session_file in enumerate(selected, 1):
                session_output = output / session_file.stem
                click.echo(f"[{idx}/{len(selected)}] {session_file.name}")
                generate_html(session_file, session_output, github_repo=repo)
            # Generate master index with agent relationships
            generate_multi_session_index(output, selected, agent_map=agent_map)
            click.echo(f"Generated {len(selected)} session(s) with master index")

    elif fmt == "duckdb":
        db_path = (
            output.with_suffix(".duckdb") if output.suffix != ".duckdb" else output
        )
        db_path.parent.mkdir(parents=True, exist_ok=True)

        if schema == "simple":
            conn = create_duckdb_schema(db_path)
            for idx, session_file in enumerate(selected, 1):
                click.echo(f"[{idx}/{len(selected)}] {session_file.name}")
                export_session_to_duckdb(
                    conn,
                    session_file,
                    session_file.parent.name,
                    include_thinking=include_thinking,
                )
            conn.close()
        else:  # star schema
            conn = create_star_schema(db_path)
            for idx, session_file in enumerate(selected, 1):
                click.echo(f"[{idx}/{len(selected)}] {session_file.name}")
                run_star_schema_etl(
                    conn,
                    session_file,
                    session_file.parent.name,
                    include_thinking=include_thinking,
                )
            # Generate semantic model metadata after all ETL is complete
            create_semantic_model(conn)
            conn.close()

        click.echo(f"Exported to {db_path}")
        return  # Skip browser open for DuckDB

    elif fmt == "json":
        if schema == "simple":
            # Handle directory paths (like ".") - generate default filename
            if output.is_dir() or output.name == "" or output.suffix == "":
                json_path = output / "sessions.json"
            elif output.suffix != ".json":
                json_path = output.with_suffix(".json")
            else:
                json_path = output
            json_path.parent.mkdir(parents=True, exist_ok=True)
            click.echo(f"Exporting {len(selected)} session(s) to JSON...")
            export_sessions_to_json(
                selected, json_path, include_thinking=include_thinking
            )
            click.echo(f"Exported to {json_path}")
        else:  # star schema
            # Star schema JSON exports to a directory
            # Handle paths like "." or paths with extensions
            if output.is_dir() or output.name in ("", "."):
                output_dir = output / "star_schema"
            elif output.suffix != "":
                output_dir = output.with_suffix("")
            else:
                output_dir = output
            output_dir.mkdir(parents=True, exist_ok=True)
            click.echo(f"Exporting {len(selected)} session(s) to star schema JSON...")
            # First create DuckDB in memory, then export to JSON
            from .star_schema import export_star_schema_to_json

            conn = create_star_schema(":memory:")
            for idx, session_file in enumerate(selected, 1):
                click.echo(f"[{idx}/{len(selected)}] {session_file.name}")
                run_star_schema_etl(
                    conn,
                    session_file,
                    session_file.parent.name,
                    include_thinking=include_thinking,
                )
            create_semantic_model(conn)
            export_star_schema_to_json(conn, output_dir)
            conn.close()
            click.echo(f"Exported to {output_dir}/")
        return  # Skip browser open for JSON

    # Show output directory
    click.echo(f"Output: {output.resolve()}")

    # Copy JSONL file to output directory if requested
    if include_json and fmt == "html":
        output.mkdir(exist_ok=True)
        for session_file in selected:
            json_dest = output / session_file.name
            shutil.copy(session_file, json_dest)
        click.echo(f"Copied {len(selected)} JSONL file(s)")

    if gist and fmt == "html" and len(selected) == 1:
        # Inject gist preview JS and create gist
        inject_gist_preview_js(output)
        click.echo("Creating GitHub gist...")
        gist_id, gist_url = create_gist(output)
        preview_url = f"https://gisthost.github.io/?{gist_id}/index.html"
        click.echo(f"Gist: {gist_url}")
        click.echo(f"Preview: {preview_url}")
    elif gist:
        click.echo("Warning: --gist only supported for single HTML session export")

    # Open browser if requested
    if open_browser and fmt == "html":
        # For multiple sessions or agents, open the master index
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
        # First try to detect from session content (git push output)
        github_repo = detect_github_repo(loglines)
        if github_repo:
            click.echo(f"Auto-detected GitHub repo from session: {github_repo}")
        else:
            # Fallback: detect from current working directory's git remote
            github_repo = detect_github_repo_from_cwd()
            if github_repo:
                click.echo(f"Auto-detected GitHub repo from cwd: {github_repo}")

    # Set module-level variable for render functions
    set_github_repo(github_repo)

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
            commit_hash, commit_msg, commit_ts, get_github_repo()
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
    "-p",
    "--project",
    "project_filter",
    help="Filter by project name (partial match, case-insensitive).",
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
    project_filter,
    dry_run,
    open_browser,
    quiet,
    no_search_index,
    output_format,
    include_thinking,
):
    """Convert all local Claude Code sessions to HTML archive or DuckDB.

    Use --format to choose output:
    - html: Browsable HTML archive with master index and per-project pages
    - duckdb: DuckDB database with sessions, messages, and tool_calls tables
    - both: Generate both HTML archive and DuckDB database
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

    projects = find_all_sessions(
        source, include_agents=include_agents, project_filter=project_filter
    )

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


@cli.command("explore")
@click.option(
    "-p",
    "--port",
    default=8765,
    help="Port to serve on (default: 8765).",
)
@click.option(
    "--no-open",
    is_flag=True,
    help="Don't open browser automatically.",
)
@click.argument("database", required=False, type=click.Path(exists=True))
def explore_cmd(port, no_open, database):
    """Launch the Data Explorer to analyze DuckDB star schema databases.

    Starts a local web server and opens the Data Explorer in your browser.
    Optionally specify a DATABASE file to have it ready to load.

    Examples:

        claude-code-transcripts explore

        claude-code-transcripts explore my-archive.duckdb
    """
    import http.server
    import socketserver
    import threading

    # Get data explorer directory
    explorer_dir = Path(__file__).parent / "explorer"
    if not explorer_dir.exists():
        raise click.ClickException(f"Data explorer directory not found: {explorer_dir}")

    explorer_file = explorer_dir / "index.html"
    if not explorer_file.exists():
        raise click.ClickException(f"Data explorer not found: {explorer_file}")

    # Create a simple HTTP server with proper cleanup
    class QuietHandler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(explorer_dir), **kwargs)

        def log_message(self, format, *args):
            pass  # Suppress request logging

    class ReusableTCPServer(socketserver.TCPServer):
        allow_reuse_address = True

    httpd = None
    try:
        httpd = ReusableTCPServer(("", port), QuietHandler)
        url = f"http://localhost:{port}/index.html"
        click.echo(f"Data Explorer running at: {url}")
        if database:
            db_path = Path(database).resolve()
            click.echo(f"Load database: {db_path}")
        click.echo("Press Ctrl+C to stop the server.")

        if not no_open:
            webbrowser.open(url)

        httpd.serve_forever()
    except KeyboardInterrupt:
        click.echo("\nStopping server...")
    except OSError as e:
        if e.errno == 48:  # Address already in use
            raise click.ClickException(
                f"Port {port} is already in use. Try a different port with -p/--port."
            )
        raise
    finally:
        if httpd:
            httpd.shutdown()
            httpd.server_close()
            click.echo("Server stopped.")


def main():
    cli()
