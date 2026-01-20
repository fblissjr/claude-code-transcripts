# ccutils

Claude utilities for session transcripts, star schema analytics, data exploration, and probably more as it comes up as a use case in my day to day.

> **Origin:** This project began as a fork of Simon Willison's [claude-code-transcripts](https://github.com/simonw/claude-code-transcripts). It has since diverged significantly with star schema analytics, a visual data explorer, modular architecture, and as a broader Claude utility.

## Installation

```bash
uv tool install ccutils
```

Or run without installing:

```bash
uvx ccutils --help
```

## Quick Start

```bash
# Interactive session picker - select and view as HTML
ccutils

# Export to DuckDB for SQL analytics
ccutils local --format duckdb -o ./archive

# Export with star schema (25+ dimensional tables)
ccutils local --format duckdb-star -o ./analytics

# Launch visual data explorer
ccutils explore ./analytics/archive.duckdb
```

## Commands

| Command | Description |
|---------|-------------|
| `local` | Select from local sessions (~/.claude/projects) - **default** |
| `web` | Import from Claude API |
| `json` | Convert specific JSON/JSONL file or URL |
| `all` | Batch convert all sessions |
| `explore` | Launch Data Explorer web UI |
| `import` | Import Claude.ai account exports (Settings > Privacy) |
| `schema` | Inspect JSON structure without exposing content |

## Export Formats

### HTML Transcripts

Clean, mobile-friendly HTML with pagination, commit timeline, and tool stats.

```bash
ccutils local -o ./transcript --open
ccutils json session.jsonl -o ./output
```

### DuckDB Analytics

#### Simple Schema (4 tables)

```bash
ccutils local --format duckdb -o ./archive
```

Tables: `sessions`, `messages`, `tool_calls`, `thinking`

#### Star Schema (25+ tables)

```bash
ccutils local --format duckdb-star -o ./analytics
```

Dimensional model with:
- **Dimensions:** tool categories, model families, dates/times, files, languages, entities
- **Facts:** messages, tool calls, code blocks, file operations, errors
- **Pre-aggregated:** session summaries, tool chains

```sql
-- Tool usage by category
SELECT dt.tool_category, COUNT(*) as uses
FROM fact_tool_calls ftc
JOIN dim_tool dt ON ftc.tool_key = dt.tool_key
GROUP BY dt.tool_category ORDER BY uses DESC;

-- Activity by time of day
SELECT dt.time_of_day, COUNT(*) as messages
FROM fact_messages fm
JOIN dim_time dt ON fm.time_key = dt.time_key
GROUP BY dt.time_of_day;
```

### JSON Export

```bash
# Simple schema - single file
ccutils local --format json -o ./sessions.json

# Star schema - directory structure
ccutils local --format json-star -o ./star-export/
```

## Data Explorer

Visual query builder for star schema databases:

```bash
ccutils explore ./analytics/archive.duckdb
```

Features: type-aware columns, automatic joins, filter autocomplete, live SQL preview.

## Common Options

```bash
-o, --output PATH      # Output directory or file
-a, --output-auto      # Auto-name based on session
--format FORMAT        # html, duckdb, duckdb-star, json, json-star
--schema SCHEMA        # simple or star (auto-inferred from format)
--include-thinking     # Include thinking blocks (can be large)
--include-subagents    # Include related agent sessions
--open                 # Open result in browser
```

## Documentation

- [Star Schema Reference](docs/STAR_SCHEMA.md) - Complete schema docs, queries, Python API
- [Data Explorer Guide](docs/DATA_EXPLORER.md) - Visual explorer features

## Development

```bash
uv run pytest           # Run tests
uv run ccutils --help   # Run development version
uv run black .          # Format code
```

## License

Apache-2.0
