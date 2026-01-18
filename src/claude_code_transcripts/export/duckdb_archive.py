"""DuckDB archive generation.

This module provides functions for creating DuckDB database archives
from Claude Code session files.
"""

from pathlib import Path

from ..parsers import find_all_sessions
from ..schemas import create_duckdb_schema, export_session_to_duckdb


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
