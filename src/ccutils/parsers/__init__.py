"""Session and content parsing utilities.

This package provides functions for parsing Claude Code session files,
extracting content from messages, and discovering sessions across projects.
"""

from .session import (
    extract_repo_from_session,
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
    get_terminal_width,
    matches_project_filter,
)

from .claude_ai import (
    convert_content_block,
    convert_conversation_to_loglines,
    convert_message_to_logline,
    load_export_files,
    parse_claude_ai_export,
)

from .schema_inspector import (
    format_schema,
    infer_schema,
    inspect_export_directory,
    inspect_json_file,
)

__all__ = [
    # Session parsing
    "extract_repo_from_session",
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
    "get_terminal_width",
    "matches_project_filter",
    # Claude.ai export parsing
    "convert_content_block",
    "convert_conversation_to_loglines",
    "convert_message_to_logline",
    "load_export_files",
    "parse_claude_ai_export",
    # Schema inspection
    "format_schema",
    "infer_schema",
    "inspect_export_directory",
    "inspect_json_file",
]
