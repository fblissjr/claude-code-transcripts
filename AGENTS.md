Uses uv. Run tests like this:

    uv run pytest

Run the development version of the tool like this:

    uv run claude-code-transcripts --help

Always practice TDD: write a failing test, watch it fail, then make it pass.

Commit early and often. Commits should bundle the test, implementation, and documentation changes together.

Run Black to format code before you commit:

    uv run black .

## Project Structure

```
claude-code-transcripts/
├── src/claude_code_transcripts/
│   ├── __init__.py           # CLI, orchestration, re-exports modular APIs
│   ├── parsers/              # Session file parsing utilities
│   │   ├── __init__.py       # Public API exports
│   │   └── session.py        # JSONL/JSON session parsing
│   ├── schemas/              # Schema definitions
│   │   ├── __init__.py       # Unified exports for both schemas
│   │   ├── simple/           # Simple 4-table schema
│   │   │   ├── __init__.py
│   │   │   ├── schema.py     # DDL for simple schema
│   │   │   └── etl.py        # Simple schema ETL
│   │   └── star/             # Star schema (25+ tables)
│   │       ├── __init__.py   # Public API exports
│   │       ├── schema.py     # DDL for star schema tables
│   │       ├── etl.py        # Main ETL pipeline
│   │       ├── semantic.py   # Semantic model generation
│   │       ├── extractors.py # Code blocks, entities, file extraction
│   │       ├── enrichment.py # LLM enrichment functions
│   │       ├── json_export.py# JSON export for star schema
│   │       └── utils.py      # Key generation, tool/model classification
│   ├── explorer/             # Data Explorer SPA
│   │   ├── index.html
│   │   ├── css/styles.css
│   │   └── js/{app,state,duckdb,query-builder,ui}.js
│   └── templates/            # Jinja2 templates for HTML export
│       ├── base.html
│       ├── page.html
│       └── star_schema_dashboard.html
├── tests/
│   ├── test_claude_code_transcripts.py  # Core functionality tests
│   ├── test_star_schema.py              # Star schema & ETL tests
│   └── test_json_export.py              # JSON export tests
├── docs/
│   ├── STAR_SCHEMA.md        # Star schema documentation
│   └── DATA_EXPLORER.md      # Data explorer documentation
└── README.md
```

## Key Components

### 1. CLI Commands
- `local` - Select from local sessions (~/.claude/projects)
- `web` - Import from Claude API
- `json` - Convert specific JSON/JSONL files
- `all` - Batch convert all sessions
- `explore` - Launch Data Explorer web server

### 2. Export Formats
Three output formats with two schema types:

**Simple schema** (4 tables: `sessions`, `messages`, `tool_calls`, `thinking`):
- `--format duckdb` - DuckDB database file
- `--format json` - Single JSON file with nested tables

**Star schema** (25+ dimensional tables):
- `--format duckdb-star` - DuckDB database file
- `--format json-star` - Directory with meta.json + dimensions/*.json + facts/*.json
- Modular package at `schemas/star/` (schema, etl, semantic, extractors, enrichment, json_export, utils)
- See `create_star_schema()`, `run_star_schema_etl()`, `export_star_schema_to_json()` functions
- Visual explorer at `explorer/`
- Full documentation in docs/STAR_SCHEMA.md and docs/DATA_EXPLORER.md

**Hybrid CLI**: Use `--schema simple|star` with `--format duckdb|json` for explicit control.

### 3. Star Schema Tables

**Dimensions:**
- dim_tool (with category classification)
- dim_model (with family: opus/sonnet/haiku)
- dim_date, dim_time (time dimensions)
- dim_session, dim_project
- dim_file, dim_programming_language
- dim_error_type
- dim_entity_type (for extracted entities)
- dim_intent, dim_sentiment, dim_topic (for LLM enrichment)

**Facts:**
- fact_messages (with response_time_seconds, conversation_depth)
- fact_tool_calls
- fact_content_blocks
- fact_session_summary (pre-aggregated)
- fact_file_operations
- fact_code_blocks
- fact_errors
- fact_entity_mentions
- fact_tool_chain_steps
- fact_message_enrichment, fact_message_topics, fact_session_insights (LLM enrichment)

### 4. LLM Enrichment Pipeline

For optional LLM-based classification:
```python
from claude_code_transcripts import run_llm_enrichment, run_session_insights_enrichment

# Enrich messages with intent, sentiment, topics
run_llm_enrichment(conn, my_enrich_func)

# Generate session-level insights
run_session_insights_enrichment(conn, my_insight_func)
```

## Testing

Run all tests:

    uv run pytest

Run star schema tests specifically:

    uv run pytest tests/test_star_schema.py -v

Run with coverage:

    uv run pytest --cov=claude_code_transcripts

## Common Workflows

### Adding a new dimension
1. Add CREATE TABLE statement in `schemas/star/schema.py`
2. Add ETL logic in `schemas/star/etl.py` to populate the dimension
3. Write tests in `test_star_schema.py`
4. Update docs/STAR_SCHEMA.md

### Adding a new fact table
1. Add CREATE TABLE statement in `schemas/star/schema.py`
2. Add data collection logic in ETL extraction phase
3. Add INSERT statement in ETL loading phase
4. Write tests covering schema and ETL
5. Update documentation
