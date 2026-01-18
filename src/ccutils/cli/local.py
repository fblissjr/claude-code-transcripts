"""Local session selection and conversion command."""

import shutil
import webbrowser
from datetime import datetime
from pathlib import Path

import click
import questionary

from ..parsers import (
    build_session_choices,
    find_agent_sessions,
    find_local_sessions,
    flatten_selected_sessions,
)
from ..schemas import (
    create_duckdb_schema,
    export_session_to_duckdb,
    export_sessions_to_json,
    resolve_schema_format,
)
from ..schemas.star import (
    create_semantic_model,
    create_star_schema,
    export_star_schema_to_json,
    run_star_schema_etl,
)
from ..export import (
    create_gist,
    generate_html,
    generate_multi_session_index,
    inject_gist_preview_js,
)


@click.command("local")
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
