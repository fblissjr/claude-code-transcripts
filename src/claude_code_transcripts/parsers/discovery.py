"""Session discovery and project management utilities.

This module provides functions for finding and organizing Claude Code sessions
across project directories.
"""

from datetime import datetime
from pathlib import Path

import questionary

from .session import extract_session_metadata, extract_session_slug, get_session_summary


def find_local_sessions(folder, limit=10, project_filter=None):
    """Find recent JSONL session files in the given folder.

    Returns a list of (Path, summary, slug) tuples sorted by modification time.
    Excludes agent files and warmup/empty sessions.
    Sessions with the same slug are part of the same conversation chain (resumed sessions).

    Args:
        folder: Path to the projects folder
        limit: Maximum number of sessions to return
        project_filter: Optional filter for project names (partial, case-insensitive)
    """
    folder = Path(folder)
    if not folder.exists():
        return []

    results = []
    for f in folder.glob("**/*.jsonl"):
        if f.name.startswith("agent-"):
            continue
        if project_filter and not matches_project_filter(f.parent.name, project_filter):
            continue
        summary = get_session_summary(f)
        # Skip boring/empty sessions
        if summary.lower() == "warmup" or summary == "(no summary)":
            continue
        slug = extract_session_slug(f)
        results.append((f, summary, slug))

    # Sort by modification time, most recent first
    results.sort(key=lambda x: x[0].stat().st_mtime, reverse=True)
    return results[:limit]


def flatten_selected_sessions(selected):
    """Flatten selected sessions which may include chains (lists) or single paths.

    In collapsed chain mode, selecting a chain returns a list of all session paths.
    This function flattens such mixed selections into a single list of paths.

    Args:
        selected: List of items - each item is either a Path or a list of Paths

    Returns:
        Flattened list of all session Paths.
    """
    result = []
    for item in selected:
        if isinstance(item, list):
            result.extend(item)
        else:
            result.append(item)
    return result


def build_session_choices(sessions_by_project, expand_chains=False, agent_counts=None):
    """Build questionary choices from sessions, with chain grouping support.

    Args:
        sessions_by_project: Dict mapping project_key to list of (filepath, summary, slug) tuples
        expand_chains: If False (default), group sessions with same slug into single choice.
                      If True, show individual sessions with chain headers.
        agent_counts: Optional dict mapping filepath to agent count for display

    Returns:
        List of questionary.Choice and questionary.Separator objects.
    """
    agent_counts = agent_counts or {}
    choices = []

    for project_key, sessions in sessions_by_project.items():
        # Add project separator with full display name
        project_name = get_project_display_name(project_key)
        choices.append(questionary.Separator(f"--- {project_name} ---"))

        # Group sessions by slug
        slug_groups = {}  # slug -> list of (filepath, summary, slug)
        standalone = []  # sessions without slug

        for filepath, summary, slug in sessions:
            if slug:
                if slug not in slug_groups:
                    slug_groups[slug] = []
                slug_groups[slug].append((filepath, summary, slug))
            else:
                standalone.append((filepath, summary, slug))

        if expand_chains:
            # Expanded mode: show individual sessions with chain headers
            for slug, chain_sessions in slug_groups.items():
                if len(chain_sessions) > 1:
                    # Add a separator for the chain
                    choices.append(
                        questionary.Separator(
                            f"  -- {slug} ({len(chain_sessions)} sessions) --"
                        )
                    )
                # Add individual sessions
                for filepath, summary, _ in chain_sessions:
                    display = _format_session_display(
                        filepath, summary, agent_counts.get(filepath, 0)
                    )
                    choices.append(questionary.Choice(title=display, value=filepath))

            # Add standalone sessions
            for filepath, summary, _ in standalone:
                display = _format_session_display(
                    filepath, summary, agent_counts.get(filepath, 0)
                )
                choices.append(questionary.Choice(title=display, value=filepath))

        else:
            # Collapsed mode: group chains into single choice
            for slug, chain_sessions in slug_groups.items():
                if len(chain_sessions) > 1:
                    # Create a single choice for the entire chain
                    paths = [s[0] for s in chain_sessions]
                    total_size = sum(p.stat().st_size for p in paths) / 1024

                    # Get date range with times
                    session_stats = [(p, p.stat().st_mtime) for p in paths]
                    session_stats.sort(key=lambda x: x[1])  # Sort by mtime
                    oldest_time = datetime.fromtimestamp(session_stats[0][1])
                    newest_time = datetime.fromtimestamp(session_stats[-1][1])

                    # Find summary from most recent session
                    newest_path = session_stats[-1][0]
                    latest_summary = None
                    for filepath, summary, _ in chain_sessions:
                        if filepath == newest_path:
                            latest_summary = summary
                            break

                    # Format date range
                    if oldest_time.date() == newest_time.date():
                        date_range = f"{oldest_time.strftime('%b %d %H:%M')} - {newest_time.strftime('%H:%M')}"
                    else:
                        date_range = f"{oldest_time.strftime('%b %d %H:%M')} - {newest_time.strftime('%b %d %H:%M')}"

                    # Truncate summary for display
                    max_summary = 50
                    if latest_summary and len(latest_summary) > max_summary:
                        latest_summary = latest_summary[: max_summary - 3] + "..."

                    # Multi-line display for better readability
                    line1 = f"[{len(chain_sessions)} sessions] {slug}"
                    line2 = f"    {total_size:,.0f} KB | {date_range}"
                    if latest_summary:
                        line2 += f' | "{latest_summary}"'

                    display = f"{line1}\n{line2}"
                    choices.append(questionary.Choice(title=display, value=paths))
                else:
                    # Single session with slug - treat as standalone
                    filepath, summary, _ = chain_sessions[0]
                    display = _format_session_display(
                        filepath, summary, agent_counts.get(filepath, 0)
                    )
                    choices.append(questionary.Choice(title=display, value=filepath))

            # Add standalone sessions
            for filepath, summary, _ in standalone:
                display = _format_session_display(
                    filepath, summary, agent_counts.get(filepath, 0)
                )
                choices.append(questionary.Choice(title=display, value=filepath))

    return choices


def _format_session_display(filepath, summary, agent_count=0):
    """Format a single session for display in the selection list.

    Args:
        filepath: Path to the session file
        summary: Session summary text
        agent_count: Number of related agent sessions

    Returns:
        Formatted display string.
    """
    stat = filepath.stat()
    mod_time = datetime.fromtimestamp(stat.st_mtime)
    size_kb = stat.st_size / 1024
    date_str = mod_time.strftime("%Y-%m-%d %H:%M")

    # Truncate summary
    max_summary = 45
    if len(summary) > max_summary:
        summary = summary[: max_summary - 3] + "..."

    # Build suffix for agents
    suffix = ""
    if agent_count > 0:
        suffix = f" (+{agent_count} agents)"

    return f"{date_str}  {size_kb:5.0f} KB  {summary}{suffix}"


def find_agent_sessions(session_paths, recursive=True):
    """Find all agent sessions related to given parent sessions.

    Agent sessions are identified by:
    - Filename pattern: agent-{agentId}.jsonl
    - Contains sessionId field linking to parent session
    - Has isSidechain: true flag

    Args:
        session_paths: List of parent session Paths
        recursive: If True, also discover agents spawned by agents

    Returns:
        Dict mapping parent session Path to list of agent session Paths.
        When recursive=True, nested agents are flattened under the original parent.
    """
    if not session_paths:
        return {}

    session_paths = [Path(p) for p in session_paths]
    original_set = set(session_paths)
    result = {p: [] for p in session_paths}

    # Build a map of sessionId -> session_path for quick lookup
    session_id_map = {}
    for p in session_paths:
        meta = extract_session_metadata(p)
        if meta.get("sessionId"):
            session_id_map[meta["sessionId"]] = p
        # Also map by file stem
        session_id_map[p.stem] = p

    # Track which original parent each path traces back to
    # (for recursive flattening)
    root_parent_map = {p: p for p in session_paths}

    # Get all directories containing the sessions
    dirs = set(p.parent for p in session_paths)

    # Find all agent files in those directories
    agent_files = []
    for d in dirs:
        agent_files.extend(d.glob("agent-*.jsonl"))

    # Multiple passes to handle recursive discovery
    found_new = True
    processed_agents = set()

    while found_new:
        found_new = False

        for agent_path in agent_files:
            if agent_path in processed_agents:
                continue

            meta = extract_session_metadata(agent_path)
            parent_session_id = meta.get("sessionId")

            if not parent_session_id:
                processed_agents.add(agent_path)
                continue

            # Find the parent session
            parent_path = session_id_map.get(parent_session_id)

            if parent_path is not None:
                processed_agents.add(agent_path)
                found_new = True

                # Find the root parent (original session, not an agent)
                root_parent = root_parent_map.get(parent_path, parent_path)

                # Add agent to the appropriate parent
                if recursive and root_parent in original_set:
                    # Flatten to original parent
                    if agent_path not in result[root_parent]:
                        result[root_parent].append(agent_path)
                    # Track this agent's root parent
                    root_parent_map[agent_path] = root_parent
                else:
                    # Non-recursive: add to immediate parent only
                    if parent_path in result:
                        if agent_path not in result[parent_path]:
                            result[parent_path].append(agent_path)

                # Register this agent in session_id_map so its children can find it
                if recursive:
                    agent_stem = agent_path.stem
                    if agent_stem not in session_id_map:
                        session_id_map[agent_stem] = agent_path

    return result


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


def matches_project_filter(folder_name: str, project_filter: str | None) -> bool:
    """Check if project folder matches filter (partial, case-insensitive).

    Matches against both the display name AND the raw folder name for
    better discoverability (e.g., searching "claude-code" will match
    even if display name is "workspace-claude-transcripts").

    Args:
        folder_name: The raw folder name (e.g., "-home-user-projects-myproject")
        project_filter: Filter string to match against, or None for no filtering

    Returns:
        True if the filter matches or is None/empty, False otherwise
    """
    if not project_filter:
        return True
    filter_lower = project_filter.lower()
    display_name = get_project_display_name(folder_name)
    # Match against display name OR raw folder name
    return filter_lower in display_name.lower() or filter_lower in folder_name.lower()


def find_all_sessions(folder, include_agents=False, project_filter=None):
    """Find all sessions in a Claude projects folder, grouped by project.

    Returns a list of project dicts, each containing:
    - name: display name for the project
    - path: Path to the project folder
    - sessions: list of session dicts with path, summary, mtime, size

    Sessions are sorted by modification time (most recent first) within each project.
    Projects are sorted by their most recent session.

    Args:
        folder: Path to the projects folder
        include_agents: Whether to include agent-* session files
        project_filter: Optional filter for project names (partial, case-insensitive)
    """
    folder = Path(folder)
    if not folder.exists():
        return []

    projects = {}

    for session_file in folder.glob("**/*.jsonl"):
        # Skip agent files unless requested
        if not include_agents and session_file.name.startswith("agent-"):
            continue

        # Skip projects that don't match filter
        if project_filter and not matches_project_filter(
            session_file.parent.name, project_filter
        ):
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
