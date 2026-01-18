"""Data extractors for ETL pipeline - code blocks, entities, file info."""

import re
from pathlib import Path

from .utils import generate_dimension_key


# Language detection patterns for code blocks
LANGUAGE_EXTENSIONS = {
    "python": [".py", ".pyw", ".pyi"],
    "javascript": [".js", ".jsx", ".mjs"],
    "typescript": [".ts", ".tsx"],
    "java": [".java"],
    "c": [".c", ".h"],
    "cpp": [".cpp", ".cc", ".cxx", ".hpp", ".hh"],
    "csharp": [".cs"],
    "go": [".go"],
    "rust": [".rs"],
    "ruby": [".rb"],
    "php": [".php"],
    "swift": [".swift"],
    "kotlin": [".kt", ".kts"],
    "scala": [".scala"],
    "r": [".r", ".R"],
    "sql": [".sql"],
    "html": [".html", ".htm"],
    "css": [".css", ".scss", ".sass", ".less"],
    "json": [".json"],
    "yaml": [".yaml", ".yml"],
    "xml": [".xml"],
    "markdown": [".md", ".markdown"],
    "shell": [".sh", ".bash", ".zsh"],
    "powershell": [".ps1", ".psm1"],
    "dockerfile": ["Dockerfile"],
    "toml": [".toml"],
}

# Code block regex pattern
CODE_BLOCK_PATTERN = re.compile(r"```(\w*)\n(.*?)```", re.DOTALL)

# Entity extraction patterns
ENTITY_PATTERNS = {
    "file_path": re.compile(
        r'(?:^|[\s"\'])(/[a-zA-Z0-9_\-./]+\.[a-zA-Z0-9]+|'
        r"[a-zA-Z]:\\[a-zA-Z0-9_\-\\./]+\.[a-zA-Z0-9]+|"
        r"\./[a-zA-Z0-9_\-./]+\.[a-zA-Z0-9]+)"
    ),
    "url": re.compile(r"https?://[a-zA-Z0-9\-._~:/?#\[\]@!$&'()*+,;=%]+"),
    "function_name": re.compile(
        r"\bdef\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(|"
        r"\bfunction\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(|"
        r"\bconst\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(?:async\s*)?\(|"
        r"\bfunc\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\("
    ),
    "class_name": re.compile(
        r"\bclass\s+([A-Z][a-zA-Z0-9_]*)|"
        r"\binterface\s+([A-Z][a-zA-Z0-9_]*)|"
        r"\bstruct\s+([A-Z][a-zA-Z0-9_]*)"
    ),
    "package_name": re.compile(
        r"\bimport\s+([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)|"
        r"\bfrom\s+([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)\s+import|"
        r'\brequire\s*\(\s*["\']([a-zA-Z0-9@/_\-]+)["\']\s*\)'
    ),
    "error_code": re.compile(
        r"\b(E[0-9]{4}|ERR_[A-Z_]+|[A-Z]+Error|[A-Z]+Exception)\b"
    ),
    "git_ref": re.compile(
        r"\b([a-f0-9]{7,40})\b|"
        r"\b(HEAD~?\d*|main|master|develop|feature/[a-zA-Z0-9_\-]+|"
        r"release/[a-zA-Z0-9._\-]+|hotfix/[a-zA-Z0-9_\-]+)\b"
    ),
}


def extract_entities(text, message_id, session_key):
    """Extract entities from text using regex patterns.

    Args:
        text: Text content to extract entities from
        message_id: ID of the message containing this text
        session_key: Session key for linking

    Returns:
        List of entity mention dicts
    """
    if not text:
        return []

    mentions = []
    for entity_type, pattern in ENTITY_PATTERNS.items():
        entity_type_key = generate_dimension_key(entity_type)

        for match in pattern.finditer(text):
            matched_text = match.group(0)
            groups = [g for g in match.groups() if g]
            if groups:
                matched_text = groups[0]

            if len(matched_text) < 2:
                continue
            if entity_type == "git_ref" and matched_text in ("HEAD", "main", "master"):
                if not re.search(
                    r"git|branch|checkout|merge|rebase",
                    text[: match.start()].split("\n")[-1],
                    re.I,
                ):
                    continue

            start = max(0, match.start() - 50)
            end = min(len(text), match.end() + 50)
            context = text[start:end]

            mention_id = generate_dimension_key(
                message_id, entity_type, str(match.start())
            )
            mentions.append(
                {
                    "mention_id": mention_id,
                    "message_id": message_id,
                    "session_key": session_key,
                    "entity_type_key": entity_type_key,
                    "entity_text": matched_text[:500],
                    "entity_normalized": matched_text.lower()[:500],
                    "context_snippet": context[:200],
                    "position_start": match.start(),
                    "position_end": match.end(),
                }
            )

    return mentions


def extract_file_info(file_path):
    """Extract file information from a file path."""
    if not file_path:
        return None

    path = Path(file_path)
    return {
        "file_key": generate_dimension_key(file_path),
        "file_path": file_path,
        "file_name": path.name,
        "file_extension": path.suffix if path.suffix else "",
        "directory_path": str(path.parent),
    }


def detect_language_from_extension(file_path):
    """Detect programming language from file extension."""
    if not file_path:
        return "unknown"

    path = Path(file_path)
    ext = path.suffix.lower()
    name = path.name

    if name == "Dockerfile":
        return "dockerfile"

    for lang, extensions in LANGUAGE_EXTENSIONS.items():
        if ext in extensions:
            return lang

    return "unknown"


def detect_language_from_hint(hint):
    """Detect programming language from code block hint."""
    if not hint:
        return "unknown"

    hint = hint.lower().strip()

    if hint in LANGUAGE_EXTENSIONS:
        return hint

    aliases = {
        "py": "python",
        "js": "javascript",
        "ts": "typescript",
        "rb": "ruby",
        "sh": "shell",
        "bash": "shell",
        "zsh": "shell",
        "yml": "yaml",
        "md": "markdown",
        "c++": "cpp",
        "c#": "csharp",
        "jsx": "javascript",
        "tsx": "typescript",
    }

    return aliases.get(hint, hint if hint else "unknown")


def extract_code_blocks(text):
    """Extract code blocks from text content."""
    if not text:
        return []

    blocks = []
    for match in CODE_BLOCK_PATTERN.finditer(text):
        hint = match.group(1)
        code = match.group(2)

        language = detect_language_from_hint(hint)
        line_count = code.count("\n") + 1 if code.strip() else 0
        char_count = len(code)

        blocks.append(
            {
                "language": language,
                "code": code,
                "line_count": line_count,
                "char_count": char_count,
            }
        )

    return blocks


def estimate_tokens(text):
    """Estimate token count for text."""
    if not text:
        return 0

    words = len(text.split())
    has_code = "```" in text or "def " in text or "function " in text
    multiplier = 1.5 if has_code else 1.3

    return int(words * multiplier)


def count_words(text):
    """Count words in text."""
    if not text:
        return 0
    return len(text.split())


def get_operation_type(tool_name):
    """Map tool name to file operation type."""
    tool_lower = tool_name.lower() if tool_name else ""

    if tool_lower == "read":
        return "read"
    elif tool_lower == "write":
        return "write"
    elif tool_lower in ("edit", "multiedit"):
        return "edit"
    elif tool_lower == "glob":
        return "list"
    elif tool_lower == "grep":
        return "search"
    return "other"


def extract_file_path_from_tool(tool_name, tool_input):
    """Extract file path from tool input."""
    if not isinstance(tool_input, dict):
        return None

    path_keys = ["file_path", "path", "filepath", "notebook_path"]
    for key in path_keys:
        if key in tool_input:
            return tool_input[key]

    return None


def calculate_conversation_depth(message_id, parent_id, depth_map):
    """Calculate conversation depth for a message."""
    if parent_id is None or parent_id not in depth_map:
        return 0
    return depth_map[parent_id] + 1
