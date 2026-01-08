# Star Schema DuckDB Implementation

This document describes the star schema data model for Claude Code transcript analytics. The schema is designed for efficient analytical queries across sessions, messages, tools, and time dimensions.

## Overview

The star schema follows dimensional modeling best practices:

- **Dimension tables** contain descriptive attributes (who, what, when, where)
- **Fact tables** contain measurable events and metrics
- **Hash-based surrogate keys** link facts to dimensions (no hard PK/FK constraints)
- **Soft business rules** for referential integrity

## Architecture

```
                    ┌─────────────┐
                    │  dim_date   │
                    └──────┬──────┘
                           │
    ┌───────────┐   ┌──────┴──────┐   ┌─────────────┐
    │ dim_tool  │───│fact_messages│───│ dim_model   │
    └───────────┘   └──────┬──────┘   └─────────────┘
                           │
    ┌───────────┐   ┌──────┴──────┐   ┌─────────────────┐
    │dim_session│───│fact_tool_   │───│dim_content_block│
    └───────────┘   │   calls     │   │     _type       │
                    └─────────────┘   └─────────────────┘
    ┌───────────┐   ┌─────────────┐   ┌─────────────┐
    │dim_project│───│fact_session │───│  dim_time   │
    └───────────┘   │  _summary   │   └─────────────┘
                    └─────────────┘
                    ┌─────────────┐
                    │fact_content │
                    │   _blocks   │
                    └─────────────┘

    Granular Analytics Tables:

    ┌───────────┐   ┌─────────────┐   ┌───────────────────┐
    │ dim_file  │───│fact_file_   │───│dim_programming_   │
    └───────────┘   │ operations  │   │   language        │
                    └─────────────┘   └───────────────────┘
                    ┌─────────────┐
    ┌─────────────┐ │fact_code_   │
    │dim_error_   │ │  _blocks    │
    │   type      │ └─────────────┘
    └──────┬──────┘
           │
    ┌──────┴──────┐
    │ fact_errors │
    └─────────────┘
```

## Dimension Tables

### dim_tool

Tool dimension with category classification for analyzing tool usage patterns.

| Column | Type | Description |
|--------|------|-------------|
| tool_key | VARCHAR | MD5 hash of tool_name (surrogate key) |
| tool_name | VARCHAR | Tool name (e.g., "Write", "Bash", "Read") |
| tool_category | VARCHAR | Category classification |

**Tool Categories:**
- `file_operations`: Read, Write, Edit, MultiEdit, NotebookEdit, Glob
- `search`: Grep, WebSearch
- `execution`: Bash, BashOutput, KillShell
- `web`: WebFetch
- `task_management`: Task, TodoWrite
- `planning`: EnterPlanMode, ExitPlanMode
- `interaction`: AskUserQuestion
- `other`: Skill, SlashCommand, and unknown tools

### dim_model

Claude model dimension with family classification.

| Column | Type | Description |
|--------|------|-------------|
| model_key | VARCHAR | MD5 hash of model_name |
| model_name | VARCHAR | Full model name (e.g., "claude-opus-4-5-20251101") |
| model_family | VARCHAR | Model family: "opus", "sonnet", "haiku", or "unknown" |

### dim_project

Project dimension for grouping sessions.

| Column | Type | Description |
|--------|------|-------------|
| project_key | VARCHAR | MD5 hash of project_path |
| project_path | VARCHAR | Full path to project directory |
| project_name | VARCHAR | Display name of project |

### dim_session

Session dimension with metadata.

| Column | Type | Description |
|--------|------|-------------|
| session_key | VARCHAR | MD5 hash of session_id |
| session_id | VARCHAR | Session filename stem |
| project_key | VARCHAR | Reference to dim_project |
| cwd | VARCHAR | Working directory |
| git_branch | VARCHAR | Git branch name |
| version | VARCHAR | Claude Code version |
| first_timestamp | TIMESTAMP | Session start time |
| last_timestamp | TIMESTAMP | Session end time |

### dim_date

Standard date dimension for time-based analysis.

| Column | Type | Description |
|--------|------|-------------|
| date_key | INTEGER | YYYYMMDD format (e.g., 20250115) |
| full_date | DATE | Full date value |
| year | INTEGER | Year (e.g., 2025) |
| month | INTEGER | Month (1-12) |
| day | INTEGER | Day of month (1-31) |
| day_of_week | INTEGER | Day of week (0=Monday, 6=Sunday) |
| day_name | VARCHAR | Day name (e.g., "Monday") |
| month_name | VARCHAR | Month name (e.g., "January") |
| quarter | INTEGER | Quarter (1-4) |
| is_weekend | BOOLEAN | True if Saturday or Sunday |

### dim_time

Time of day dimension for intraday analysis.

| Column | Type | Description |
|--------|------|-------------|
| time_key | INTEGER | HHMM format (e.g., 1430 for 2:30 PM) |
| hour | INTEGER | Hour (0-23) |
| minute | INTEGER | Minute (0-59) |
| time_of_day | VARCHAR | Period: "night", "morning", "afternoon", "evening" |

**Time of Day Classifications:**
- `night`: 00:00 - 05:59
- `morning`: 06:00 - 11:59
- `afternoon`: 12:00 - 17:59
- `evening`: 18:00 - 23:59

### dim_message_type

Message type dimension (pre-populated).

| Column | Type | Description |
|--------|------|-------------|
| message_type_key | VARCHAR | MD5 hash of message_type |
| message_type | VARCHAR | "user" or "assistant" |

### dim_content_block_type

Content block type dimension (pre-populated).

| Column | Type | Description |
|--------|------|-------------|
| content_block_type_key | VARCHAR | MD5 hash of block_type |
| block_type | VARCHAR | "text", "tool_use", "tool_result", "thinking", "image" |

### dim_file

File dimension for tracking file operations.

| Column | Type | Description |
|--------|------|-------------|
| file_key | VARCHAR | MD5 hash of file_path |
| file_path | VARCHAR | Full path to file |
| file_name | VARCHAR | Base filename |
| file_extension | VARCHAR | File extension (e.g., ".py", ".ts") |
| directory_path | VARCHAR | Parent directory path |

### dim_programming_language

Programming language dimension for code block analysis.

| Column | Type | Description |
|--------|------|-------------|
| language_key | VARCHAR | MD5 hash of language_name |
| language_name | VARCHAR | Language name (e.g., "python", "javascript") |
| file_extensions | VARCHAR | Associated file extensions |

**Supported Languages:**
- python, javascript, typescript, java, cpp, c, csharp, go, rust, ruby, php, swift, kotlin, scala, sql, html, css, markdown, json, yaml, xml, bash, shell, powershell, dockerfile, makefile, toml, unknown

### dim_error_type

Error type dimension for error tracking and analysis.

| Column | Type | Description |
|--------|------|-------------|
| error_type_key | VARCHAR | MD5 hash of error_type |
| error_type | VARCHAR | Error type (e.g., "FileNotFound", "PermissionDenied") |
| error_category | VARCHAR | Category: "file_system", "network", "permission", "syntax", "runtime", "other" |

## Fact Tables

### fact_messages

One row per message - the core fact table.

| Column | Type | Description |
|--------|------|-------------|
| message_id | VARCHAR | Message UUID (degenerate dimension) |
| session_key | VARCHAR | FK to dim_session |
| project_key | VARCHAR | FK to dim_project |
| message_type_key | VARCHAR | FK to dim_message_type |
| model_key | VARCHAR | FK to dim_model (NULL for user messages) |
| date_key | INTEGER | FK to dim_date |
| time_key | INTEGER | FK to dim_time |
| parent_message_id | VARCHAR | Parent message UUID |
| timestamp | TIMESTAMP | Message timestamp |
| content_length | INTEGER | Character count of text content |
| content_block_count | INTEGER | Number of content blocks |
| has_tool_use | BOOLEAN | Contains tool_use blocks |
| has_tool_result | BOOLEAN | Contains tool_result blocks |
| has_thinking | BOOLEAN | Contains thinking blocks |
| word_count | INTEGER | Word count in text content |
| estimated_tokens | INTEGER | Estimated token count (~1.3x words) |
| response_time_seconds | FLOAT | Time since parent message (seconds) |
| conversation_depth | INTEGER | Depth in conversation tree (0=root) |
| content_text | TEXT | Extracted text content (truncated) |
| content_json | JSON | Full content as JSON |

### fact_content_blocks

Granular content block tracking - one row per block within a message.

| Column | Type | Description |
|--------|------|-------------|
| content_block_id | VARCHAR | Generated ID (message_id-index) |
| message_id | VARCHAR | Parent message UUID |
| session_key | VARCHAR | FK to dim_session |
| content_block_type_key | VARCHAR | FK to dim_content_block_type |
| date_key | INTEGER | FK to dim_date |
| time_key | INTEGER | FK to dim_time |
| block_index | INTEGER | Position within message (0-based) |
| content_length | INTEGER | Character count |
| content_text | TEXT | Block content (truncated) |
| content_json | JSON | Full block as JSON |

### fact_tool_calls

Tool invocation facts - links tool_use to tool_result.

| Column | Type | Description |
|--------|------|-------------|
| tool_call_id | VARCHAR | tool_use_id (degenerate dimension) |
| session_key | VARCHAR | FK to dim_session |
| tool_key | VARCHAR | FK to dim_tool |
| date_key | INTEGER | FK to dim_date |
| time_key | INTEGER | FK to dim_time |
| invoke_message_id | VARCHAR | Message with tool_use |
| result_message_id | VARCHAR | Message with tool_result |
| timestamp | TIMESTAMP | When tool was invoked |
| input_char_count | INTEGER | Size of input JSON |
| output_char_count | INTEGER | Size of output text |
| is_error | BOOLEAN | Tool returned error |
| input_json | JSON | Full tool input |
| input_summary | TEXT | Truncated input |
| output_text | TEXT | Truncated output |

### fact_session_summary

Pre-aggregated session metrics for dashboard queries.

| Column | Type | Description |
|--------|------|-------------|
| session_key | VARCHAR | FK to dim_session |
| project_key | VARCHAR | FK to dim_project |
| date_key | INTEGER | FK to dim_date (session start date) |
| total_messages | INTEGER | Total message count |
| user_messages | INTEGER | User message count |
| assistant_messages | INTEGER | Assistant message count |
| total_tool_calls | INTEGER | Tool invocation count |
| total_thinking_blocks | INTEGER | Thinking block count |
| total_content_blocks | INTEGER | Content block count |
| session_duration_seconds | INTEGER | Duration in seconds |
| first_timestamp | TIMESTAMP | Session start |
| last_timestamp | TIMESTAMP | Session end |

### fact_file_operations

File-level operations tracking for read/write/edit activities.

| Column | Type | Description |
|--------|------|-------------|
| file_operation_id | VARCHAR | Generated ID (tool_call_id-file_key) |
| tool_call_id | VARCHAR | FK to fact_tool_calls |
| session_key | VARCHAR | FK to dim_session |
| file_key | VARCHAR | FK to dim_file |
| tool_key | VARCHAR | FK to dim_tool |
| date_key | INTEGER | FK to dim_date |
| time_key | INTEGER | FK to dim_time |
| operation_type | VARCHAR | "read", "write", "edit", "search", "list" |
| file_size_chars | INTEGER | Size of file content in characters |
| timestamp | TIMESTAMP | When operation occurred |

### fact_code_blocks

Code block extraction from messages for language analysis.

| Column | Type | Description |
|--------|------|-------------|
| code_block_id | VARCHAR | Generated ID (message_id-index) |
| message_id | VARCHAR | Parent message UUID |
| session_key | VARCHAR | FK to dim_session |
| language_key | VARCHAR | FK to dim_programming_language |
| date_key | INTEGER | FK to dim_date |
| time_key | INTEGER | FK to dim_time |
| block_index | INTEGER | Position within message (0-based) |
| line_count | INTEGER | Number of lines in code block |
| char_count | INTEGER | Character count |
| code_text | TEXT | Code block content (truncated) |

### fact_errors

Error tracking from tool calls.

| Column | Type | Description |
|--------|------|-------------|
| error_id | VARCHAR | Generated ID (tool_call_id) |
| tool_call_id | VARCHAR | FK to fact_tool_calls |
| session_key | VARCHAR | FK to dim_session |
| tool_key | VARCHAR | FK to dim_tool |
| error_type_key | VARCHAR | FK to dim_error_type |
| date_key | INTEGER | FK to dim_date |
| time_key | INTEGER | FK to dim_time |
| error_message | TEXT | Error message content |
| timestamp | TIMESTAMP | When error occurred |

### fact_entity_mentions

Entities extracted from message text using regex patterns.

| Column | Type | Description |
|--------|------|-------------|
| mention_id | VARCHAR | Generated ID |
| message_id | VARCHAR | Parent message UUID |
| session_key | VARCHAR | FK to dim_session |
| entity_type_key | VARCHAR | FK to dim_entity_type |
| entity_text | VARCHAR | Extracted entity text |
| entity_normalized | VARCHAR | Normalized (lowercase) entity |
| context_snippet | TEXT | Surrounding context |
| position_start | INTEGER | Start position in text |
| position_end | INTEGER | End position in text |

### fact_tool_chain_steps

Sequential tool call patterns for workflow analysis.

| Column | Type | Description |
|--------|------|-------------|
| chain_step_id | VARCHAR | Generated ID |
| session_key | VARCHAR | FK to dim_session |
| chain_id | VARCHAR | Chain identifier |
| tool_call_id | VARCHAR | FK to fact_tool_calls |
| tool_key | VARCHAR | FK to dim_tool |
| step_position | INTEGER | Position in chain (0-based) |
| prev_tool_key | VARCHAR | Previous tool's key (NULL for first) |
| time_since_prev_seconds | FLOAT | Time since previous tool call |

## Entity Extraction Dimensions

### dim_entity_type

Types of entities extracted from text (pre-populated).

| Column | Type | Description |
|--------|------|-------------|
| entity_type_key | VARCHAR | MD5 hash of entity_type |
| entity_type | VARCHAR | Type name |
| extraction_method | VARCHAR | "regex" for pattern-based extraction |

**Pre-populated Entity Types:**
- `file_path`: File paths (/path/to/file.py, ./relative/path.ts)
- `url`: HTTP/HTTPS URLs
- `function_name`: Function definitions (def, function, const =>)
- `class_name`: Class definitions (class, interface, struct)
- `package_name`: Import statements (import, from, require)
- `error_code`: Error codes (E1234, ERR_*, *Error, *Exception)
- `git_ref`: Git references (commit hashes, branch names)

## LLM Enrichment Tables

These tables support optional LLM-based classification and summarization.

### dim_intent

User intent classifications (pre-populated).

| Column | Type | Description |
|--------|------|-------------|
| intent_key | VARCHAR | MD5 hash of intent_name |
| intent_name | VARCHAR | Intent name |
| intent_category | VARCHAR | Category grouping |
| description | TEXT | Human-readable description |

**Pre-populated Intents:**
- `bug_fix` (problem_solving): Fix a bug or error
- `feature` (development): Add new functionality
- `refactor` (development): Improve code structure
- `question` (inquiry): Ask about code or concepts
- `explain` (inquiry): Request explanation
- `review` (analysis): Review or analyze code
- `test` (quality): Write or run tests
- `debug` (problem_solving): Debug an issue
- `config` (setup): Configuration or setup
- `docs` (documentation): Documentation work

### dim_sentiment

Sentiment/tone classifications (pre-populated).

| Column | Type | Description |
|--------|------|-------------|
| sentiment_key | VARCHAR | MD5 hash of sentiment_name |
| sentiment_name | VARCHAR | Sentiment name |
| valence | FLOAT | Emotional valence (-1 to +1) |

**Pre-populated Sentiments:**
- `neutral` (0.0)
- `positive` (0.5)
- `negative` (-0.5)
- `frustrated` (-0.8)
- `satisfied` (0.8)
- `confused` (-0.3)
- `curious` (0.3)

### dim_topic

Domain/topic tags (pre-populated).

| Column | Type | Description |
|--------|------|-------------|
| topic_key | VARCHAR | MD5 hash of topic_name |
| topic_name | VARCHAR | Topic name |
| topic_category | VARCHAR | Category grouping |

**Pre-populated Topics:**
- Domain: frontend, backend, database, api, auth
- Practice: testing, deployment
- Concern: security, performance
- Design: architecture

### fact_message_enrichment

LLM-assigned labels per message.

| Column | Type | Description |
|--------|------|-------------|
| enrichment_id | VARCHAR | Generated ID |
| message_id | VARCHAR | Parent message UUID |
| session_key | VARCHAR | FK to dim_session |
| intent_key | VARCHAR | FK to dim_intent |
| sentiment_key | VARCHAR | FK to dim_sentiment |
| complexity_score | FLOAT | Message complexity (0-1) |
| confidence_score | FLOAT | Classification confidence (0-1) |
| enrichment_model | VARCHAR | Model used for enrichment |
| enriched_at | TIMESTAMP | When enrichment was performed |

### fact_message_topics

Many-to-many message-topic associations.

| Column | Type | Description |
|--------|------|-------------|
| message_topic_id | VARCHAR | Generated ID |
| message_id | VARCHAR | Parent message UUID |
| topic_key | VARCHAR | FK to dim_topic |
| relevance_score | FLOAT | Topic relevance (0-1) |

### fact_session_insights

LLM-generated session summaries.

| Column | Type | Description |
|--------|------|-------------|
| insight_id | VARCHAR | Generated ID |
| session_key | VARCHAR | FK to dim_session |
| summary_text | TEXT | Session summary |
| key_decisions | TEXT | Key decisions made |
| outcome_status | VARCHAR | success, partial, failed, unknown |
| task_completed | BOOLEAN | Whether task was completed |
| primary_intent_key | VARCHAR | FK to dim_intent |
| complexity_score | FLOAT | Session complexity (0-1) |
| enrichment_model | VARCHAR | Model used for enrichment |
| enriched_at | TIMESTAMP | When enrichment was performed |

## Staging Table

### stg_raw_messages

Staging table for raw extracted data (used during ETL processing).

| Column | Type | Description |
|--------|------|-------------|
| session_id | VARCHAR | Session identifier |
| project_name | VARCHAR | Project name |
| project_path | VARCHAR | Project path |
| message_id | VARCHAR | Message UUID |
| parent_id | VARCHAR | Parent message UUID |
| message_type | VARCHAR | user/assistant |
| timestamp | TIMESTAMP | Message timestamp |
| model | VARCHAR | Model name |
| cwd | VARCHAR | Working directory |
| git_branch | VARCHAR | Git branch |
| version | VARCHAR | Claude Code version |
| content_json | JSON | Raw content |
| content_text | TEXT | Extracted text |

## Key Generation

Dimension keys are generated using MD5 hashes of natural keys:

```python
from claude_code_transcripts import generate_dimension_key

# Single natural key
tool_key = generate_dimension_key("Write")
# Returns: "9a79be611e0267e1d943da0737c6c51c"

# Composite natural key
project_key = generate_dimension_key("project_name", "/home/user/myproject")
# Returns: MD5 hash of "project_name|/home/user/myproject"
```

This approach provides:
- **Deterministic keys**: Same input always produces same key
- **No coordination needed**: Keys can be generated independently
- **Natural deduplication**: Duplicates automatically resolve to same key

## Example Queries

### Tool Usage by Category

```sql
SELECT
    dt.tool_category,
    COUNT(*) as usage_count,
    AVG(ftc.output_char_count) as avg_output_size
FROM fact_tool_calls ftc
JOIN dim_tool dt ON ftc.tool_key = dt.tool_key
GROUP BY dt.tool_category
ORDER BY usage_count DESC;
```

### Messages by Model Family Over Time

```sql
SELECT
    dd.year,
    dd.month,
    dm.model_family,
    COUNT(*) as message_count
FROM fact_messages fm
JOIN dim_date dd ON fm.date_key = dd.date_key
JOIN dim_model dm ON fm.model_key = dm.model_key
WHERE fm.model_key IS NOT NULL
GROUP BY dd.year, dd.month, dm.model_family
ORDER BY dd.year, dd.month;
```

### Activity by Time of Day

```sql
SELECT
    dt.time_of_day,
    COUNT(*) as message_count,
    SUM(CASE WHEN fm.has_tool_use THEN 1 ELSE 0 END) as tool_uses
FROM fact_messages fm
JOIN dim_time dt ON fm.time_key = dt.time_key
GROUP BY dt.time_of_day
ORDER BY
    CASE dt.time_of_day
        WHEN 'morning' THEN 1
        WHEN 'afternoon' THEN 2
        WHEN 'evening' THEN 3
        WHEN 'night' THEN 4
    END;
```

### Session Productivity Metrics

```sql
SELECT
    dp.project_name,
    ds.git_branch,
    fss.total_messages,
    fss.total_tool_calls,
    ROUND(fss.total_tool_calls::FLOAT / NULLIF(fss.total_messages, 0), 2) as tools_per_message,
    fss.session_duration_seconds / 60 as duration_minutes
FROM fact_session_summary fss
JOIN dim_session ds ON fss.session_key = ds.session_key
JOIN dim_project dp ON fss.project_key = dp.project_key
ORDER BY fss.first_timestamp DESC
LIMIT 20;
```

### Content Block Analysis

```sql
SELECT
    dcbt.block_type,
    COUNT(*) as block_count,
    AVG(fcb.content_length) as avg_length,
    MAX(fcb.content_length) as max_length
FROM fact_content_blocks fcb
JOIN dim_content_block_type dcbt ON fcb.content_block_type_key = dcbt.content_block_type_key
GROUP BY dcbt.block_type
ORDER BY block_count DESC;
```

### Weekend vs Weekday Activity

```sql
SELECT
    CASE WHEN dd.is_weekend THEN 'Weekend' ELSE 'Weekday' END as period,
    COUNT(*) as message_count,
    COUNT(DISTINCT fss.session_key) as session_count
FROM fact_session_summary fss
JOIN dim_date dd ON fss.date_key = dd.date_key
GROUP BY dd.is_weekend;
```

### File Operations by Extension

```sql
SELECT
    df.file_extension,
    ffo.operation_type,
    COUNT(*) as operation_count,
    AVG(ffo.file_size_chars) as avg_file_size
FROM fact_file_operations ffo
JOIN dim_file df ON ffo.file_key = df.file_key
GROUP BY df.file_extension, ffo.operation_type
ORDER BY operation_count DESC
LIMIT 20;
```

### Code Blocks by Language

```sql
SELECT
    dpl.language_name,
    COUNT(*) as block_count,
    SUM(fcb.line_count) as total_lines,
    AVG(fcb.line_count) as avg_lines_per_block
FROM fact_code_blocks fcb
JOIN dim_programming_language dpl ON fcb.language_key = dpl.language_key
GROUP BY dpl.language_name
ORDER BY block_count DESC;
```

### Error Analysis by Tool

```sql
SELECT
    dt.tool_name,
    det.error_type,
    COUNT(*) as error_count
FROM fact_errors fe
JOIN dim_tool dt ON fe.tool_key = dt.tool_key
JOIN dim_error_type det ON fe.error_type_key = det.error_type_key
GROUP BY dt.tool_name, det.error_type
ORDER BY error_count DESC
LIMIT 15;
```

### Token Estimation by Model

```sql
SELECT
    dm.model_family,
    COUNT(*) as message_count,
    SUM(fm.word_count) as total_words,
    SUM(fm.estimated_tokens) as total_tokens,
    AVG(fm.estimated_tokens) as avg_tokens_per_message
FROM fact_messages fm
JOIN dim_model dm ON fm.model_key = dm.model_key
WHERE fm.model_key IS NOT NULL
GROUP BY dm.model_family
ORDER BY total_tokens DESC;
```

### Response Time Analysis

```sql
SELECT
    dmt.message_type,
    AVG(fm.response_time_seconds) as avg_response_time,
    MAX(fm.response_time_seconds) as max_response_time,
    COUNT(*) as message_count
FROM fact_messages fm
JOIN dim_message_type dmt ON fm.message_type_key = dmt.message_type_key
WHERE fm.response_time_seconds IS NOT NULL
GROUP BY dmt.message_type;
```

### Conversation Depth Analysis

```sql
SELECT
    ds.session_id,
    MAX(fm.conversation_depth) as max_depth,
    AVG(fm.conversation_depth) as avg_depth,
    COUNT(*) as message_count
FROM fact_messages fm
JOIN dim_session ds ON fm.session_key = ds.session_key
GROUP BY ds.session_id
ORDER BY max_depth DESC;
```

### Entity Mentions by Type

```sql
SELECT
    det.entity_type,
    COUNT(*) as mention_count,
    COUNT(DISTINCT fem.entity_normalized) as unique_entities
FROM fact_entity_mentions fem
JOIN dim_entity_type det ON fem.entity_type_key = det.entity_type_key
GROUP BY det.entity_type
ORDER BY mention_count DESC;
```

### Tool Chain Patterns

```sql
SELECT
    curr.tool_name as current_tool,
    prev.tool_name as previous_tool,
    COUNT(*) as transition_count,
    AVG(tcs.time_since_prev_seconds) as avg_time_between
FROM fact_tool_chain_steps tcs
JOIN dim_tool curr ON tcs.tool_key = curr.tool_key
LEFT JOIN dim_tool prev ON tcs.prev_tool_key = prev.tool_key
GROUP BY curr.tool_name, prev.tool_name
ORDER BY transition_count DESC
LIMIT 20;
```

### Messages by Intent (after LLM enrichment)

```sql
SELECT
    di.intent_name,
    di.intent_category,
    COUNT(*) as message_count,
    AVG(fme.complexity_score) as avg_complexity
FROM fact_message_enrichment fme
JOIN dim_intent di ON fme.intent_key = di.intent_key
GROUP BY di.intent_name, di.intent_category
ORDER BY message_count DESC;
```

## Usage

### Creating a Star Schema Database

```python
from pathlib import Path
from claude_code_transcripts import create_star_schema, run_star_schema_etl

# Create schema
db_path = Path("./analytics.duckdb")
conn = create_star_schema(db_path)

# Load a session
session_path = Path("~/.claude/projects/myproject/session-123.jsonl")
run_star_schema_etl(
    conn,
    session_path,
    project_name="My Project",
    include_thinking=True,  # Include thinking blocks
    truncate_output=5000    # Max chars for content
)

# Query the data
result = conn.execute("""
    SELECT dp.project_name, COUNT(*) as sessions
    FROM fact_session_summary fss
    JOIN dim_project dp ON fss.project_key = dp.project_key
    GROUP BY dp.project_name
""").fetchall()

conn.close()
```

### LLM Enrichment Pipeline

The schema includes tables for LLM-enriched classifications that are populated separately from the main ETL:

```python
from claude_code_transcripts import (
    create_star_schema,
    run_star_schema_etl,
    run_llm_enrichment,
    run_session_insights_enrichment
)

# First, run the main ETL
conn = create_star_schema(db_path)
run_star_schema_etl(conn, session_path, "My Project")

# Define an enrichment function (your custom LLM integration)
def my_enrich_func(messages):
    """Call your LLM to classify messages."""
    results = []
    for msg in messages:
        # Call your LLM API here...
        results.append({
            "message_id": msg["message_id"],
            "intent": "bug_fix",  # from dim_intent
            "sentiment": "neutral",  # from dim_sentiment
            "topics": ["backend", "testing"],  # from dim_topic
            "complexity_score": 0.7,
            "confidence_score": 0.9,
        })
    return results

# Run message enrichment
result = run_llm_enrichment(
    conn,
    my_enrich_func,
    model_name="claude-3-haiku-20240307",
    batch_size=10
)
print(f"Enriched {result['messages_enriched']} messages")

# Define a session insight function
def my_insight_func(session_data):
    """Generate session-level insights."""
    # Call your LLM to summarize the session...
    return {
        "summary_text": "User debugged an authentication issue...",
        "key_decisions": "Switched from JWT to session tokens...",
        "outcome_status": "success",
        "task_completed": True,
        "primary_intent": "bug_fix",
        "complexity_score": 0.6,
    }

# Run session enrichment
result = run_session_insights_enrichment(conn, my_insight_func)
print(f"Enriched {result['sessions_enriched']} sessions")
```

## Design Decisions

### Why Hash-Based Keys?

1. **No Sequence Management**: No need to coordinate auto-increment sequences
2. **Deterministic**: Same natural key always produces same surrogate key
3. **Idempotent ETL**: Re-running ETL produces same results
4. **Distributed Friendly**: Keys can be generated anywhere

### Why Soft Business Rules?

1. **Flexibility**: Can load facts before dimensions if needed
2. **Error Tolerance**: Partial loads don't break referential integrity
3. **Performance**: No constraint checking overhead
4. **Recovery**: Easier to fix data issues

### Why Pre-Aggregated Summary Facts?

1. **Dashboard Performance**: Fast queries for common metrics
2. **Reduced Complexity**: Simpler queries for common patterns
3. **Session-Level Analysis**: Natural grain for many analytics

## ETL Process

The `run_star_schema_etl` function performs a three-phase ETL:

1. **Extract**: Parse JSONL session file, collect raw data
2. **Transform**:
   - Generate dimension keys
   - Classify tools and models
   - Parse timestamps into date/time keys
   - Link tool_use to tool_result
3. **Load**:
   - Upsert dimensions (check exists before insert)
   - Insert facts

The ETL is designed to be idempotent - running it multiple times with the same input produces the same result.

## Use Cases

Beyond basic analytics, the star schema enables several advanced applications:

### 1. Context Retrieval for New Sessions

Use prior session data as context for new Claude conversations. Query relevant past interactions based on files, topics, or patterns:

```sql
-- Find sessions that worked on similar files
SELECT DISTINCT ds.session_id, ds.cwd, dp.project_name,
       fm.content_text, fm.timestamp
FROM fact_file_operations ffo
JOIN dim_file df ON ffo.file_key = df.file_key
JOIN dim_session ds ON ffo.session_key = ds.session_key
JOIN dim_project dp ON ds.project_key = dp.project_key
JOIN fact_messages fm ON ffo.session_key = fm.session_key
WHERE df.file_path LIKE '%auth%'
  AND fm.content_text IS NOT NULL
ORDER BY fm.timestamp DESC
LIMIT 20;
```

```python
# Build context from past sessions for a new conversation
def get_relevant_context(conn, file_patterns, limit=5):
    """Retrieve relevant past interactions for context injection."""
    context_messages = []
    for pattern in file_patterns:
        result = conn.execute("""
            SELECT fm.content_text, dmt.message_type
            FROM fact_messages fm
            JOIN dim_message_type dmt ON fm.message_type_key = dmt.message_type_key
            JOIN fact_file_operations ffo ON fm.session_key = ffo.session_key
            JOIN dim_file df ON ffo.file_key = df.file_key
            WHERE df.file_path LIKE ?
            ORDER BY fm.timestamp DESC
            LIMIT ?
        """, [f"%{pattern}%", limit]).fetchall()
        context_messages.extend(result)
    return context_messages
```

### 2. Tool Pattern Optimization

Analyze which tool sequences are most effective for different task types:

```sql
-- Find successful tool patterns (sessions that completed quickly)
WITH session_metrics AS (
    SELECT session_key,
           session_duration_seconds,
           total_tool_calls,
           NTILE(4) OVER (ORDER BY session_duration_seconds) as speed_quartile
    FROM fact_session_summary
)
SELECT curr.tool_name, prev.tool_name as prev_tool,
       COUNT(*) as transitions,
       AVG(sm.session_duration_seconds) as avg_session_duration
FROM fact_tool_chain_steps tcs
JOIN dim_tool curr ON tcs.tool_key = curr.tool_key
LEFT JOIN dim_tool prev ON tcs.prev_tool_key = prev.tool_key
JOIN session_metrics sm ON tcs.session_key = sm.session_key
WHERE sm.speed_quartile = 1  -- Fastest sessions
GROUP BY curr.tool_name, prev.tool_name
HAVING COUNT(*) > 5
ORDER BY transitions DESC;
```

### 3. Session Continuation / Handoff

Resume or hand off sessions by reconstructing state:

```sql
-- Get full session state for continuation
SELECT ds.session_id, ds.cwd, ds.git_branch,
       fm.message_id, fm.content_text, fm.conversation_depth,
       dmt.message_type, dm.model_name
FROM fact_messages fm
JOIN dim_session ds ON fm.session_key = ds.session_key
JOIN dim_message_type dmt ON fm.message_type_key = dmt.message_type_key
LEFT JOIN dim_model dm ON fm.model_key = dm.model_key
WHERE ds.session_id = 'target-session-id'
ORDER BY fm.timestamp;

-- Find files modified in session for handoff context
SELECT df.file_path, ffo.operation_type,
       COUNT(*) as operations
FROM fact_file_operations ffo
JOIN dim_file df ON ffo.file_key = df.file_key
JOIN dim_session ds ON ffo.session_key = ds.session_key
WHERE ds.session_id = 'target-session-id'
GROUP BY df.file_path, ffo.operation_type;
```

### 4. Training Data Extraction

Extract high-quality interaction patterns for fine-tuning or analysis:

```sql
-- Extract successful debugging sessions (LLM enrichment required)
SELECT fm.content_text, dmt.message_type, di.intent_name
FROM fact_messages fm
JOIN dim_message_type dmt ON fm.message_type_key = dmt.message_type_key
JOIN fact_message_enrichment fme ON fm.message_id = fme.message_id
JOIN dim_intent di ON fme.intent_key = di.intent_key
JOIN fact_session_insights fsi ON fm.session_key = fsi.session_key
WHERE di.intent_name = 'debug'
  AND fsi.task_completed = true
  AND fme.confidence_score > 0.8
ORDER BY fm.timestamp;
```

### 5. Error Pattern Analysis

Identify and learn from error patterns:

```sql
-- Find error-prone file/tool combinations
SELECT df.file_extension, dt.tool_name, det.error_type,
       COUNT(*) as error_count,
       COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY dt.tool_name) as error_pct
FROM fact_errors fe
JOIN fact_file_operations ffo ON fe.tool_call_id = ffo.tool_call_id
JOIN dim_file df ON ffo.file_key = df.file_key
JOIN dim_tool dt ON fe.tool_key = dt.tool_key
JOIN dim_error_type det ON fe.error_type_key = det.error_type_key
GROUP BY df.file_extension, dt.tool_name, det.error_type
HAVING COUNT(*) > 3
ORDER BY error_count DESC;
```

### 6. Cost Estimation and Optimization

Track token usage for cost analysis:

```sql
-- Estimate costs by model and project
SELECT dp.project_name, dm.model_family,
       SUM(fm.estimated_tokens) as total_tokens,
       -- Rough cost estimate (adjust rates as needed)
       CASE dm.model_family
           WHEN 'opus' THEN SUM(fm.estimated_tokens) * 0.000015
           WHEN 'sonnet' THEN SUM(fm.estimated_tokens) * 0.000003
           WHEN 'haiku' THEN SUM(fm.estimated_tokens) * 0.00000025
           ELSE 0
       END as estimated_cost_usd
FROM fact_messages fm
JOIN dim_model dm ON fm.model_key = dm.model_key
JOIN dim_session ds ON fm.session_key = ds.session_key
JOIN dim_project dp ON ds.project_key = dp.project_key
GROUP BY dp.project_name, dm.model_family
ORDER BY total_tokens DESC;
```

### 7. Knowledge Base Construction

Build a searchable knowledge base from past sessions:

```sql
-- Extract entities and their contexts for a knowledge graph
SELECT det.entity_type, fem.entity_normalized, fem.context_snippet,
       dp.project_name, ds.session_id, fm.timestamp
FROM fact_entity_mentions fem
JOIN dim_entity_type det ON fem.entity_type_key = det.entity_type_key
JOIN fact_messages fm ON fem.message_id = fm.message_id
JOIN dim_session ds ON fem.session_key = ds.session_key
JOIN dim_project dp ON ds.project_key = dp.project_key
WHERE det.entity_type IN ('function_name', 'class_name', 'file_path')
ORDER BY fem.entity_normalized, fm.timestamp;
```

### 8. Workflow Replay

Reconstruct and potentially replay successful workflows:

```sql
-- Get ordered tool sequence for a successful session
SELECT tcs.step_position, dt.tool_name, dt.tool_category,
       ftc.input_summary, tcs.time_since_prev_seconds
FROM fact_tool_chain_steps tcs
JOIN dim_tool dt ON tcs.tool_key = dt.tool_key
JOIN fact_tool_calls ftc ON tcs.tool_call_id = ftc.tool_call_id
JOIN fact_session_insights fsi ON tcs.session_key = fsi.session_key
WHERE fsi.task_completed = true
  AND fsi.outcome_status = 'success'
ORDER BY tcs.chain_id, tcs.step_position;
```

These use cases can be combined - for example, using context retrieval + tool pattern analysis to pre-populate a new session with relevant history and suggest optimal tool sequences based on the task type.
