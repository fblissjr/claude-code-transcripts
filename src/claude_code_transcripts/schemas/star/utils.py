"""Shared utilities for star schema operations."""

import hashlib


def generate_dimension_key(*natural_keys):
    """Generate a dimension key from natural key(s) using MD5 hash.

    This creates a consistent surrogate key for dimension tables based on
    the natural business key(s). Using a hash allows for:
    - Deterministic key generation (same input = same key)
    - No sequence coordination needed
    - Natural deduplication in dimensions

    Args:
        *natural_keys: One or more values that form the natural key.
                       Multiple values are joined with '|' separator.

    Returns:
        32-character lowercase hex string (MD5 hash)
    """
    key_parts = [str(k) if k is not None else "NULL" for k in natural_keys]
    combined = "|".join(key_parts)
    return hashlib.md5(combined.encode("utf-8")).hexdigest()


# Tool category mapping for dim_tool
TOOL_CATEGORIES = {
    # File operations
    "Read": "file_operations",
    "Write": "file_operations",
    "Edit": "file_operations",
    "MultiEdit": "file_operations",
    "NotebookEdit": "file_operations",
    "Glob": "file_operations",
    # Search tools
    "Grep": "search",
    "WebSearch": "search",
    # Execution tools
    "Bash": "execution",
    "BashOutput": "execution",
    "KillShell": "execution",
    # Web tools
    "WebFetch": "web",
    # Task management
    "Task": "task_management",
    "TodoWrite": "task_management",
    # Planning tools
    "EnterPlanMode": "planning",
    "ExitPlanMode": "planning",
    # Other
    "Skill": "other",
    "SlashCommand": "other",
    "AskUserQuestion": "interaction",
}


def get_tool_category(tool_name):
    """Get the category for a tool name."""
    return TOOL_CATEGORIES.get(tool_name, "other")


def get_model_family(model_name):
    """Extract the model family from a model name.

    Args:
        model_name: Full model name like 'claude-opus-4-5-20251101'

    Returns:
        Model family: 'opus', 'sonnet', 'haiku', or 'unknown'
    """
    if model_name is None:
        return "unknown"
    model_lower = model_name.lower()
    if "opus" in model_lower:
        return "opus"
    elif "sonnet" in model_lower:
        return "sonnet"
    elif "haiku" in model_lower:
        return "haiku"
    return "unknown"


def get_time_of_day(hour):
    """Get time of day label from hour.

    Args:
        hour: Hour of day (0-23)

    Returns:
        Time of day label: 'night', 'morning', 'afternoon', 'evening'
    """
    if hour < 6:
        return "night"
    elif hour < 12:
        return "morning"
    elif hour < 18:
        return "afternoon"
    else:
        return "evening"
