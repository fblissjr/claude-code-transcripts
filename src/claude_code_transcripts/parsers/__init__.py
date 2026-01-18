"""Session and content parsing utilities.

This package provides functions for parsing Claude Code session files,
extracting content from messages, and discovering sessions across projects.
"""

from .session import (
    extract_searchable_content,
    extract_session_metadata,
    extract_session_slug,
    extract_snippet,
    extract_text_from_content,
    get_session_summary,
    parse_session_file,
    PROMPTS_PER_PAGE,
)

from .discovery import (
    build_session_choices,
    find_agent_sessions,
    find_all_sessions,
    find_local_sessions,
    flatten_selected_sessions,
    get_project_display_name,
    matches_project_filter,
)

__all__ = [
    # Session parsing
    "extract_searchable_content",
    "extract_session_metadata",
    "extract_session_slug",
    "extract_snippet",
    "extract_text_from_content",
    "get_session_summary",
    "parse_session_file",
    "PROMPTS_PER_PAGE",
    # Session discovery
    "build_session_choices",
    "find_agent_sessions",
    "find_all_sessions",
    "find_local_sessions",
    "flatten_selected_sessions",
    "get_project_display_name",
    "matches_project_filter",
]
