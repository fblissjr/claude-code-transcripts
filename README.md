# claude-code-transcripts fork
fork of simonw's [claude-code-transcripts](https://github.com/simonw/claude-code-transcripts) that aims to add structured data creation in an analytics friendly star schema, using duckdb, and a lightweight HTML-based data explorer using duckdb-wasm

---

Convert Claude Code session files (JSON or JSONL) to clean, mobile-friendly HTML pages with pagination.

[Example transcript](https://static.simonwillison.net/static/2025/claude-code-microjs/index.html) produced using this tool.

Read [A new way to extract detailed transcripts from Claude Code](https://simonwillison.net/2025/Dec/25/claude-code-transcripts/) for background on this project.

## Installation

Install this tool using `uv`:
```bash
uv tool install claude-code-transcripts
```
Or run it without installing:
```bash
uvx claude-code-transcripts --help
```

## Usage

This tool converts Claude Code session files into browseable multi-page HTML transcripts.

There are four commands available:

- `local` (default) - select from local Claude Code sessions stored in `~/.claude/projects`
- `web` - select from web sessions via the Claude API
- `json` - convert a specific JSON or JSONL session file
- `all` - convert all local sessions to a browsable HTML archive

The quickest way to view a recent local session:

```bash
claude-code-transcripts
```

This shows an interactive picker to select a session, generates HTML, and opens it in your default browser.

### Output options

All commands support these options:

- `-o, --output DIRECTORY` - output directory (default: writes to temp dir and opens browser)
- `-a, --output-auto` - auto-name output subdirectory based on session ID or filename
- `--repo OWNER/NAME` - GitHub repo for commit links (auto-detected from git push output if not specified)
- `--open` - open the generated `index.html` in your default browser (default if no `-o` specified)
- `--gist` - upload the generated HTML files to a GitHub Gist and output a preview URL
- `--json` - include the original session file in the output directory

The generated output includes:
- `index.html` - an index page with a timeline of prompts and commits
- `page-001.html`, `page-002.html`, etc. - paginated transcript pages

### Local sessions

Local Claude Code sessions are stored as JSONL files in `~/.claude/projects`. Run with no arguments to select from recent sessions:

```bash
claude-code-transcripts
# or explicitly:
claude-code-transcripts local
```

Use `--limit` to control how many sessions are shown (default: 10):

```bash
claude-code-transcripts local --limit 20
```

### Web sessions

Import sessions directly from the Claude API:

```bash
# Interactive session picker
claude-code-transcripts web

# Import a specific session by ID
claude-code-transcripts web SESSION_ID

# Import and publish to gist
claude-code-transcripts web SESSION_ID --gist
```

On macOS, API credentials are automatically retrieved from your keychain (requires being logged into Claude Code). On other platforms, provide `--token` and `--org-uuid` manually.

### Publishing to GitHub Gist

Use the `--gist` option to automatically upload your transcript to a GitHub Gist and get a shareable preview URL:

```bash
claude-code-transcripts --gist
claude-code-transcripts web --gist
claude-code-transcripts json session.json --gist
```

This will output something like:
```
Gist: https://gist.github.com/username/abc123def456
Preview: https://gisthost.github.io/?abc123def456/index.html
Files: /var/folders/.../session-id
```

The preview URL uses [gisthost.github.io](https://gisthost.github.io/) to render your HTML gist. The tool automatically injects JavaScript to fix relative links when served through gisthost.

Combine with `-o` to keep a local copy:

```bash
claude-code-transcripts json session.json -o ./my-transcript --gist
```

**Requirements:** The `--gist` option requires the [GitHub CLI](https://cli.github.com/) (`gh`) to be installed and authenticated (`gh auth login`).

### Auto-naming output directories

Use `-a/--output-auto` to automatically create a subdirectory named after the session:

```bash
# Creates ./session_ABC123/ subdirectory
claude-code-transcripts web SESSION_ABC123 -a

# Creates ./transcripts/session_ABC123/ subdirectory
claude-code-transcripts web SESSION_ABC123 -o ./transcripts -a
```

### Including the source file

Use the `--json` option to include the original session file in the output directory:

```bash
claude-code-transcripts json session.json -o ./my-transcript --json
```

This will output:
```
JSON: ./my-transcript/session_ABC.json (245.3 KB)
```

This is useful for archiving the source data alongside the HTML output.

### Converting from JSON/JSONL files

Convert a specific session file directly:

```bash
claude-code-transcripts json session.json -o output-directory/
claude-code-transcripts json session.jsonl --open
```
This works with both JSONL files in the `~/.claude/projects/` folder and JSON session files extracted from Claude Code for web.

The `json` command can take a URL to a JSON or JSONL file as an alternative to a path on disk.

### Converting all sessions

Convert all your local Claude Code sessions to a browsable HTML archive:

```bash
claude-code-transcripts all
```

This creates a directory structure with:
- A master index listing all projects
- Per-project pages listing sessions
- Individual session transcripts

Options:

- `-s, --source DIRECTORY` - source directory (default: `~/.claude/projects`)
- `-o, --output DIRECTORY` - output directory (default: `./claude-archive`)
- `--include-agents` - include agent session files (excluded by default)
- `--dry-run` - show what would be converted without creating files
- `--open` - open the generated archive in your default browser
- `-q, --quiet` - suppress all output except errors
- `--no-search-index` - skip generating the search index for faster/smaller output
- `--format FORMAT` - output format: `html` (default), `duckdb`, `duckdb-star`, `json`, `json-star`, or `both`
- `--schema SCHEMA` - data schema: `simple` (4 tables) or `star` (dimensional). Auto-inferred from format
- `--include-thinking` - include thinking blocks in DuckDB/JSON export (opt-in)

Examples:

```bash
# Preview what would be converted
claude-code-transcripts all --dry-run

# Convert all sessions and open in browser
claude-code-transcripts all --open

# Convert to a specific directory
claude-code-transcripts all -o ./my-archive

# Include agent sessions
claude-code-transcripts all --include-agents

# Export to DuckDB for SQL analytics
claude-code-transcripts all --format duckdb -o ./my-archive

# Export both HTML and DuckDB
claude-code-transcripts all --format both -o ./my-archive

# Export to JSON (simple schema)
claude-code-transcripts local --format json -o ./sessions.json

# Export to JSON (star schema - directory structure)
claude-code-transcripts local --format json-star -o ./star-export/
```

### DuckDB export

The `--format duckdb` option exports your transcripts to a DuckDB database (`archive.duckdb`) for SQL-based analytics. This is useful for querying patterns across sessions, analyzing tool usage, or building custom reports.

The database contains four tables:

- `sessions` - session metadata (project, timestamps, message counts)
- `messages` - all user and assistant messages with content
- `tool_calls` - tool invocations with inputs and outputs
- `thinking` - Claude's thinking blocks (only with `--include-thinking`)

Example queries:

```sql
-- Connect to the database
-- duckdb ./my-archive/archive.duckdb

-- Count messages by type
SELECT type, COUNT(*) FROM messages GROUP BY type;

-- Most used tools
SELECT tool_name, COUNT(*) as uses
FROM tool_calls
GROUP BY tool_name
ORDER BY uses DESC
LIMIT 10;

-- Search message content
SELECT session_id, type, LEFT(content, 100)
FROM messages
WHERE content LIKE '%error%';

-- Sessions with most tool calls
SELECT s.session_id, s.project_name, COUNT(t.tool_use_id) as tool_count
FROM sessions s
JOIN tool_calls t ON s.session_id = t.session_id
GROUP BY s.session_id, s.project_name
ORDER BY tool_count DESC
LIMIT 10;
```

Use `--include-thinking` to also export Claude's thinking blocks (these can be 10KB+ each, so they're opt-in).

### Star Schema Analytics

A comprehensive dimensional model with 25+ tables for advanced analytics. Use `--format duckdb-star` to generate:

```bash
# Generate star schema from selected session
claude-code-transcripts local --format duckdb-star -o ./analytics

# Or from all sessions
claude-code-transcripts all --format duckdb-star -o ./analytics

# Launch the visual Data Explorer
claude-code-transcripts explore ./analytics/archive.duckdb

# Or query directly with DuckDB CLI
duckdb ./analytics/archive.duckdb
```

**Key differences from simple format:**

| Feature | Simple (`--format duckdb`) | Star Schema (`--format duckdb-star`) |
|---------|---------------------------|--------------------------------------|
| Tables | 4 (flat) | 25+ (dimensional) |
| Time analysis | Timestamps only | Date/time dimensions |
| Tool analysis | Tool names | Categories, chains |
| File tracking | In tool inputs | Dedicated dimension |
| Code blocks | In message content | Extracted, by language |
| Aggregations | Query-time | Pre-computed summaries |

**Example queries:**

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

**Data Explorer features:** Type-aware columns, automatic joins, filter autocomplete, live SQL preview.

See [docs/STAR_SCHEMA.md](docs/STAR_SCHEMA.md) for complete schema documentation, all example queries, Python API reference, and advanced use cases. See [docs/DATA_EXPLORER.md](docs/DATA_EXPLORER.md) for the visual explorer.

### JSON export

Export sessions to JSON format for integration with other tools, data pipelines, or archival purposes.

**Simple schema** (`--format json`): Single JSON file with 4 tables:

```bash
claude-code-transcripts local --format json -o ./sessions.json
```

Output structure:
```json
{
  "version": "1.0",
  "schema_type": "simple",
  "exported_at": "2025-01-13T10:30:00Z",
  "tables": {
    "sessions": [...],
    "messages": [...],
    "tool_calls": [...],
    "thinking": [...]
  }
}
```

**Star schema** (`--format json-star`): Directory structure with separate files per table:

```bash
claude-code-transcripts local --format json-star -o ./star-export/
```

Output structure:
```
star-export/
  meta.json           # Schema metadata, table manifest, relationships
  dimensions/
    dim_tool.json
    dim_model.json
    dim_session.json
    ...               # 15 dimension tables
  facts/
    fact_messages.json
    fact_tool_calls.json
    ...               # 12 fact tables
```

The `meta.json` file includes:
- Table manifest with row counts
- Relationship definitions (foreign keys)
- Schema version for compatibility

**Hybrid CLI**: You can also use the explicit `--schema` flag:

```bash
# These are equivalent:
claude-code-transcripts local --format json-star
claude-code-transcripts local --schema star --format json
```

## Development

To contribute to this tool, first checkout the code. You can run the tests using `uv run`:
```bash
cd claude-code-transcripts
uv run pytest
```
And run your local development copy of the tool like this:
```bash
uv run claude-code-transcripts --help
```
