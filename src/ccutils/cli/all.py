"""Batch conversion command for all sessions."""

import webbrowser
from datetime import datetime
from pathlib import Path

import click

from ..parsers import find_all_sessions
from ..export import generate_batch_html, generate_duckdb_archive


@click.command("all")
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
