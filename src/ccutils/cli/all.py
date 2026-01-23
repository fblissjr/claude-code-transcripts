"""Batch conversion command for all sessions."""

import webbrowser
from datetime import datetime
from pathlib import Path

import click

from ..parsers import find_all_sessions
from ..schemas import resolve_schema_format
from ..export import (
    generate_batch_html,
    generate_duckdb_archive,
    generate_star_json_archive,
)


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
    type=click.Choice(["html", "duckdb", "duckdb-star", "json", "json-star", "both"]),
    default="html",
    help="Output format: html (default), duckdb, duckdb-star, json, json-star, or both.",
)
@click.option(
    "--schema",
    "schema_type",
    type=click.Choice(["simple", "star"]),
    default=None,
    help="Data schema: simple (4 tables) or star (dimensional). Auto-inferred from format.",
)
@click.option(
    "-j",
    "--jobs",
    default=1,
    type=int,
    help="Number of parallel workers (default: 1, sequential).",
)
@click.option(
    "--batch-size",
    default=10,
    type=int,
    help="Sessions per transaction batch (default: 10).",
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
    schema_type,
    jobs,
    batch_size,
    include_thinking,
):
    """Convert all local Claude Code sessions to HTML, DuckDB, or JSON archives.

    Use --format to choose output:

    \b
    - html: Browsable HTML archive with master index and per-project pages
    - duckdb: DuckDB database with simple schema (4 tables)
    - duckdb-star: DuckDB database with star schema (25+ dimensional tables)
    - json: JSON files with simple schema
    - json-star: JSON directory with star schema (dimensions/ + facts/)
    - both: Generate both HTML archive and simple DuckDB database

    Star schema provides richer analytics with dimensional modeling, including
    time dimensions, tool categories, file operations, and pre-aggregated summaries.
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

    # Resolve schema type from format if not explicitly provided
    resolved_schema, resolved_format = resolve_schema_format(schema_type, output_format)

    # Progress callback for non-quiet mode with enhanced stats
    def on_progress(project_name, session_name, current, total, stats=None):
        if quiet:
            return
        if stats and current % 5 == 0:
            # Enhanced progress with stats
            rate = stats.get("rate", 0)
            db_size = stats.get("db_size_mb", 0)
            rows = stats.get("rows_inserted", 0)
            click.echo(
                f"  [{current}/{total}] {project_name}/{session_name[:8]}... "
                f"({rows} rows, {db_size:.1f} MB, {rate:.1f} sess/sec)"
            )
        elif current % 10 == 0:
            click.echo(f"  Processed {current}/{total} sessions...")

    stats = None
    duckdb_stats = None

    # Generate HTML if requested
    if output_format in ("html", "both"):
        # HTML progress callback has different signature (no stats)
        def html_progress(proj, sess, cur, tot):
            on_progress(proj, sess, cur, tot, None)

        stats = generate_batch_html(
            source,
            output,
            include_agents=include_agents,
            progress_callback=html_progress,
            no_search_index=no_search_index,
        )

    # Generate DuckDB if requested (simple or star schema)
    if output_format in ("duckdb", "duckdb-star", "both"):
        if not quiet:
            if output_format == "both":
                click.echo("\nGenerating DuckDB archive...")
            elif output_format == "duckdb-star":
                click.echo(f"Using star schema ({resolved_schema})")

        duckdb_stats = generate_duckdb_archive(
            source,
            output,
            schema_type=resolved_schema,
            include_agents=include_agents,
            include_thinking=include_thinking,
            progress_callback=on_progress if output_format != "both" else None,
            max_workers=jobs,
            batch_size=batch_size,
        )
        if stats is None:
            stats = duckdb_stats

    # Generate JSON star schema if requested
    if output_format == "json-star":
        if not quiet:
            click.echo("Generating JSON star schema archive...")
        duckdb_stats = generate_star_json_archive(
            source,
            output,
            include_agents=include_agents,
            include_thinking=include_thinking,
            progress_callback=on_progress,
            max_workers=jobs,
            batch_size=batch_size,
        )
        if stats is None:
            stats = duckdb_stats

    # Generate simple JSON if requested
    if output_format == "json":
        if not quiet:
            click.echo("Generating JSON archive...")
        from ..schemas import export_sessions_to_json

        # Collect all session paths
        session_paths = []
        for project in projects:
            for session in project["sessions"]:
                session_paths.append(session["path"])

        output.mkdir(parents=True, exist_ok=True)
        json_path = output / "sessions.json"
        export_sessions_to_json(
            session_paths,
            json_path,
            include_thinking=include_thinking,
        )
        stats = {
            "total_projects": len(projects),
            "total_sessions": total_sessions,
            "failed_sessions": [],
            "output_dir": output,
            "db_path": None,
        }

    # Report any failures
    if stats and stats.get("failed_sessions"):
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
        if duckdb_stats:
            if duckdb_stats.get("db_path"):
                click.echo(f"DuckDB: {duckdb_stats['db_path']}")
            if duckdb_stats.get("rows_inserted"):
                click.echo(f"Rows: {duckdb_stats['rows_inserted']}")
            if duckdb_stats.get("db_size_mb"):
                click.echo(f"Size: {duckdb_stats['db_size_mb']:.2f} MB")

    if open_browser and output_format in ("html", "both"):
        index_url = (output / "index.html").resolve().as_uri()
        webbrowser.open(index_url)
