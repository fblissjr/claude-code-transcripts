"""DuckDB archive generation.

This module provides functions for creating DuckDB database archives
from Claude Code session files. Supports both simple (4-table) and
star (25+ table dimensional) schemas.
"""

import os
import tempfile
import time
from pathlib import Path

import duckdb

from ..parsers import find_all_sessions
from ..schemas import (
    create_duckdb_schema,
    export_session_to_duckdb,
    create_star_schema,
    run_star_schema_etl,
    export_star_schema_to_json,
)


def generate_duckdb_archive(
    source_folder,
    output_dir,
    schema_type="simple",
    include_agents=False,
    include_thinking=False,
    truncate_output=2000,
    progress_callback=None,
    max_workers=1,
    batch_size=10,
):
    """Generate DuckDB archive for all sessions.

    Supports both simple (4-table) and star (25+ dimensional tables) schemas.
    Uses a stage-and-load pattern for efficient batch processing:
    - Stage: Parse sessions (parallelizable with max_workers)
    - Load: Bulk insert in batches (batch_size sessions per transaction)

    Args:
        source_folder: Path to Claude projects folder
        output_dir: Path for output
        schema_type: "simple" (4 tables) or "star" (dimensional model)
        include_agents: Whether to include agent sessions
        include_thinking: Whether to include thinking blocks
        truncate_output: Max chars for tool output
        progress_callback: Optional callback with signature:
            callback(project_name, session_name, current, total, stats)
            where stats is a dict with 'rows_inserted', 'db_size_mb', 'rate'
        max_workers: Number of parallel workers for staging (default: 1)
        batch_size: Sessions per transaction batch (default: 10)

    Returns:
        dict with statistics including row counts
    """
    source_folder = Path(source_folder)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    db_path = output_dir / "archive.duckdb"

    # Create appropriate schema
    if schema_type == "star":
        conn = create_star_schema(db_path)
        etl_func = run_star_schema_etl
    else:
        conn = create_duckdb_schema(db_path)
        etl_func = export_session_to_duckdb

    projects = find_all_sessions(source_folder, include_agents=include_agents)

    total_session_count = sum(len(p["sessions"]) for p in projects)
    processed_count = 0
    successful_sessions = 0
    failed_sessions = []

    # Stats tracking
    start_time = time.time()

    # Flatten sessions for processing
    session_tasks = []
    for project in projects:
        project_name = project["name"]
        for session in project["sessions"]:
            session_tasks.append((project_name, session["path"]))

    # Process sessions
    if max_workers > 1 and len(session_tasks) > 1:
        # Parallel processing - stage then load in batches
        _process_parallel(
            conn,
            session_tasks,
            etl_func,
            include_thinking,
            truncate_output,
            batch_size,
            progress_callback,
            db_path,
            start_time,
            failed_sessions,
            schema_type,
        )
        successful_sessions = len(session_tasks) - len(failed_sessions)
    else:
        # Sequential processing (original behavior)
        for project_name, session_path in session_tasks:
            try:
                etl_func(
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
                elapsed = time.time() - start_time
                rate = processed_count / elapsed if elapsed > 0 else 0
                db_size = _get_db_size_mb(db_path)
                stats = {
                    "rows_inserted": _count_rows(conn, schema_type),
                    "db_size_mb": db_size,
                    "rate": rate,
                }
                progress_callback(
                    project_name,
                    session_path.stem,
                    processed_count,
                    total_session_count,
                    stats,
                )

    # Get final row counts
    final_row_count = _count_rows(conn, schema_type)
    final_db_size = _get_db_size_mb(db_path)

    conn.close()

    return {
        "total_projects": len(projects),
        "total_sessions": successful_sessions,
        "failed_sessions": failed_sessions,
        "output_dir": output_dir,
        "db_path": db_path,
        "schema_type": schema_type,
        "rows_inserted": final_row_count,
        "db_size_mb": final_db_size,
    }


def _process_parallel(
    conn,
    session_tasks,
    etl_func,
    include_thinking,
    truncate_output,
    batch_size,
    progress_callback,
    db_path,
    start_time,
    failed_sessions,
    schema_type,
):
    """Process sessions in batches with progress reporting.

    Note: DuckDB connections are not thread-safe for writes, so we
    process in batches and serialize the actual DB writes.
    """
    total = len(session_tasks)
    processed = 0
    rows_total = 0

    # Process in batches
    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch = session_tasks[batch_start:batch_end]

        # Process batch - serialize DB writes
        for project_name, session_path in batch:
            try:
                etl_func(
                    conn,
                    session_path,
                    project_name,
                    include_thinking=include_thinking,
                    truncate_output=truncate_output,
                )
            except Exception as e:
                failed_sessions.append(
                    {
                        "project": project_name,
                        "session": session_path.stem,
                        "error": str(e),
                    }
                )

            processed += 1
            if progress_callback:
                elapsed = time.time() - start_time
                rate = processed / elapsed if elapsed > 0 else 0
                db_size = _get_db_size_mb(db_path)
                # Count rows periodically (expensive, so estimate)
                if processed % 5 == 0:
                    rows_total = _count_rows(conn, schema_type)
                stats = {
                    "rows_inserted": rows_total,
                    "db_size_mb": db_size,
                    "rate": rate,
                }
                progress_callback(
                    project_name,
                    session_path.stem,
                    processed,
                    total,
                    stats,
                )


def _count_rows(conn, schema_type):
    """Count total rows across relevant tables."""
    if schema_type == "star":
        tables = [
            "fact_messages",
            "fact_tool_calls",
            "fact_content_blocks",
            "fact_session_summary",
        ]
    else:
        tables = ["messages", "tool_calls", "sessions"]

    total = 0
    for table in tables:
        try:
            result = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
            total += result[0] if result else 0
        except Exception:
            pass
    return total


def _get_db_size_mb(db_path):
    """Get database file size in MB."""
    try:
        size_bytes = os.path.getsize(db_path)
        return round(size_bytes / (1024 * 1024), 2)
    except Exception:
        return 0.0


def generate_star_json_archive(
    source_folder,
    output_dir,
    include_agents=False,
    include_thinking=False,
    truncate_output=2000,
    progress_callback=None,
    max_workers=1,
    batch_size=10,
):
    """Generate star schema JSON archive for all sessions.

    Creates a JSON directory structure with dimensions/ and facts/ subdirs.

    Args:
        source_folder: Path to Claude projects folder
        output_dir: Path for output
        include_agents: Whether to include agent sessions
        include_thinking: Whether to include thinking blocks
        truncate_output: Max chars for tool output
        progress_callback: Optional progress callback
        max_workers: Number of parallel workers (default: 1)
        batch_size: Sessions per batch (default: 10)

    Returns:
        dict with statistics
    """
    import tempfile

    source_folder = Path(source_folder)
    output_dir = Path(output_dir)

    # First build the DuckDB, then export to JSON
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Generate DuckDB with star schema
        stats = generate_duckdb_archive(
            source_folder,
            tmp_path,
            schema_type="star",
            include_agents=include_agents,
            include_thinking=include_thinking,
            truncate_output=truncate_output,
            progress_callback=progress_callback,
            max_workers=max_workers,
            batch_size=batch_size,
        )

        # Export to JSON
        db_path = tmp_path / "archive.duckdb"
        conn = duckdb.connect(str(db_path))
        export_star_schema_to_json(conn, output_dir)
        conn.close()

    stats["output_dir"] = output_dir
    stats["db_path"] = None  # No DuckDB file for JSON export
    return stats
