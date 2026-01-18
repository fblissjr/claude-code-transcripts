# Data Explorer

A web application for exploring DuckDB star schema databases. Single HTML file with modular JavaScript, designed to work with databases created by the star schema ETL.

## Quick Start (CLI)

```bash
# Generate a star schema database from local sessions
claude-code-transcripts local --format duckdb-star -o ./my-analytics

# Launch the explorer and load the database
claude-code-transcripts explore ./my-analytics/archive.duckdb

# Or just launch the explorer (load file manually)
claude-code-transcripts explore
```

## Quick Start (Python API)

```python
from claude_code_transcripts import create_star_schema, run_star_schema_etl

conn = create_star_schema("analytics.duckdb")
run_star_schema_etl(conn, "session.jsonl", project_name="My Project")
conn.close()
```

Then launch: `claude-code-transcripts explore analytics.duckdb`

## Features

- **Spreadsheet-first UX**: Data grid as primary canvas with sortable columns
- **Type-aware columns**: Visual indicators (123/Abc/cal) for data types
- **Per-column filtering**: Click column headers with Shift to filter, autocomplete from distinct values
- **Automatic joins**: Semantic model detects fact-to-dimension relationships
- **Live SQL preview**: See the generated query as you build it
- **Pagination**: Handle large result sets efficiently

## Architecture

```
explorer/
├── index.html          # Main HTML shell
├── css/
│   └── styles.css      # Custom styles (Tailwind via CDN)
└── js/
    ├── app.js          # Application entry point
    ├── state.js        # Centralized state with pub/sub
    ├── duckdb.js       # DuckDB WASM integration
    ├── query-builder.js # SQL generation with semantic joins
    └── ui.js           # DOM rendering functions
```

### Module Responsibilities

| Module | Purpose |
|--------|---------|
| `state.js` | Application state, type icons, table/column helpers |
| `duckdb.js` | Initialize DuckDB WASM, load databases, execute queries |
| `query-builder.js` | Build SELECT/JOIN/WHERE/ORDER clauses from state |
| `ui.js` | Render table list, data grid, filters, SQL panel |
| `app.js` | Wire up state subscriptions, initialize on load |

## Semantic Model Integration

The explorer reads from `meta_semantic_model` table (created by `create_semantic_model()`):

```sql
SELECT * FROM meta_semantic_model WHERE table_type = 'fact';
```

This provides:
- Table types (dimension/fact/staging)
- Column types (key/attribute/measure)
- Relationships between tables
- Default aggregations for measures
- Display names for UI

## UI Interactions

| Action | Result |
|--------|--------|
| Click fact table | Set as base table, show columns |
| Check dimension column | Auto-join and add to SELECT |
| Click column header | Toggle sort (asc/desc/none) |
| Shift+click column header | Open filter dropdown |
| Type in filter | Autocomplete from distinct values |

## Dependencies

- DuckDB WASM 1.29.0 (CDN)
- Tailwind CSS (CDN)
- No build step required

## Security

- Table names escaped with `escapeIdentifier()` to prevent SQL injection
- Filter values escaped with `escapeSQL()` for string literals
- No innerHTML usage - all DOM manipulation via safe methods
