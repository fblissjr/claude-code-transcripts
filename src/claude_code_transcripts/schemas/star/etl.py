"""ETL pipeline for loading session data into star schema."""

import json
from datetime import datetime
from pathlib import Path

from .extractors import (
    LANGUAGE_EXTENSIONS,
    calculate_conversation_depth,
    count_words,
    estimate_tokens,
    extract_code_blocks,
    extract_entities,
    extract_file_info,
    extract_file_path_from_tool,
    get_operation_type,
)
from .utils import (
    generate_dimension_key,
    get_model_family,
    get_time_of_day,
    get_tool_category,
)


def run_star_schema_etl(
    conn, session_path, project_name, include_thinking=False, truncate_output=2000
):
    """Run ETL to populate star schema from a session file.

    This function:
    1. Extracts raw data from the session file
    2. Transforms and loads dimension tables (with deduplication)
    3. Transforms and loads fact tables

    Args:
        conn: DuckDB connection
        session_path: Path to the JSONL session file
        project_name: Name of the project
        include_thinking: Whether to include thinking blocks
        truncate_output: Max characters for tool output (default 2000)
    """
    session_path = Path(session_path)

    # ==========================================================================
    # Phase 1: Extract - Parse session file and collect raw data
    # ==========================================================================
    session_id = session_path.stem
    session_key = generate_dimension_key(session_id)
    project_path = str(session_path.parent)
    project_key = generate_dimension_key(project_path)

    # Session metadata
    cwd = None
    git_branch = None
    version = None
    first_timestamp = None
    last_timestamp = None

    # Agent metadata
    is_agent = False
    agent_id = None
    parent_session_id = None
    is_sidechain = False

    # Counters
    user_count = 0
    assistant_count = 0
    total_content_blocks = 0
    thinking_count = 0

    # Tracking structures
    messages_data = []
    content_blocks_data = []
    tool_use_map = {}
    tool_calls_data = []
    models_seen = set()
    tools_seen = set()
    dates_seen = set()

    # Granular tracking
    files_seen = {}
    file_operations_data = []
    code_blocks_data = []
    errors_data = []
    languages_seen = set()

    # Conversation tracking
    message_timestamps = {}
    depth_map = {}
    entity_mentions_data = []

    # Tool chain tracking
    tool_chain_data = []
    prev_tool_call = None

    with open(session_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            entry_type = entry.get("type")
            if entry_type not in ("user", "assistant"):
                continue

            # Extract metadata from first entry
            if cwd is None:
                cwd = entry.get("cwd")
                git_branch = entry.get("gitBranch")
                version = entry.get("version")
                agent_id = entry.get("agentId")
                is_sidechain = entry.get("isSidechain", False)
                is_agent = agent_id is not None
                if is_agent:
                    parent_session_id = entry.get("sessionId")

            message_id = entry.get("uuid", "")
            parent_id = entry.get("parentUuid")
            timestamp_str = entry.get("timestamp", "")
            message_data = entry.get("message", {})
            model = message_data.get("model")

            # Parse timestamp
            timestamp = None
            date_key = None
            time_key = None
            if timestamp_str:
                try:
                    timestamp = datetime.fromisoformat(
                        timestamp_str.replace("Z", "+00:00")
                    )
                    if first_timestamp is None:
                        first_timestamp = timestamp
                    last_timestamp = timestamp
                    date_key = int(timestamp.strftime("%Y%m%d"))
                    time_key = int(timestamp.strftime("%H%M"))
                    dates_seen.add(date_key)
                except (ValueError, TypeError):
                    pass

            if model:
                models_seen.add(model)

            # Process content
            content = message_data.get("content", "")
            has_tool_use = False
            has_tool_result = False
            has_thinking = False
            text_content = ""
            content_json = json.dumps(content)
            content_block_count = 0

            if isinstance(content, str):
                text_content = content
                content_block_count = 1
                block_id = f"{message_id}-0"
                content_blocks_data.append(
                    {
                        "content_block_id": block_id,
                        "message_id": message_id,
                        "session_key": session_key,
                        "content_block_type_key": generate_dimension_key("text"),
                        "date_key": date_key,
                        "time_key": time_key,
                        "block_index": 0,
                        "content_length": len(content),
                        "content_text": content[:truncate_output] if content else "",
                        "content_json": json.dumps({"type": "text", "text": content}),
                    }
                )
                total_content_blocks += 1

            elif isinstance(content, list):
                texts = []
                for idx, block in enumerate(content):
                    if not isinstance(block, dict):
                        continue

                    block_type = block.get("type")
                    content_block_count += 1

                    should_track = True
                    if block_type == "thinking" and not include_thinking:
                        should_track = False

                    if block_type == "text":
                        text = block.get("text", "")
                        texts.append(text)
                        if should_track:
                            _add_content_block(
                                content_blocks_data,
                                message_id,
                                session_key,
                                "text",
                                idx,
                                date_key,
                                time_key,
                                text,
                                truncate_output,
                                block,
                            )
                            total_content_blocks += 1

                    elif block_type == "tool_use":
                        has_tool_use = True
                        tool_use_id = block.get("id")
                        tool_name = block.get("name", "unknown")
                        tool_input = block.get("input", {})
                        tools_seen.add(tool_name)

                        input_json = json.dumps(tool_input)
                        input_summary = input_json[:truncate_output]
                        tool_use_map[tool_use_id] = {
                            "message_id": message_id,
                            "tool_name": tool_name,
                            "tool_key": generate_dimension_key(tool_name),
                            "input_json": input_json,
                            "input_summary": input_summary,
                            "input_char_count": len(input_json),
                            "timestamp": timestamp,
                            "date_key": date_key,
                            "time_key": time_key,
                        }

                        # Track file operations
                        file_path = extract_file_path_from_tool(tool_name, tool_input)
                        if file_path:
                            file_info = extract_file_info(file_path)
                            if file_info and file_path not in files_seen:
                                files_seen[file_path] = file_info

                            operation_type = get_operation_type(tool_name)
                            file_content = tool_input.get("content", "")
                            file_size = (
                                len(file_content)
                                if isinstance(file_content, str)
                                else 0
                            )

                            file_operations_data.append(
                                {
                                    "file_operation_id": f"{tool_use_id}-file",
                                    "tool_call_id": tool_use_id,
                                    "session_key": session_key,
                                    "file_key": (
                                        file_info["file_key"] if file_info else None
                                    ),
                                    "tool_key": generate_dimension_key(tool_name),
                                    "date_key": date_key,
                                    "time_key": time_key,
                                    "operation_type": operation_type,
                                    "file_size_chars": file_size,
                                    "timestamp": timestamp,
                                }
                            )

                        # Track tool chain
                        tool_key = generate_dimension_key(tool_name)
                        chain_id = f"{session_key}-chain"
                        step_position = len(tool_chain_data)

                        time_since_prev = None
                        prev_tool_key_val = None
                        if prev_tool_call and timestamp:
                            prev_ts = prev_tool_call[2]
                            if prev_ts:
                                time_since_prev = (timestamp - prev_ts).total_seconds()
                            prev_tool_key_val = prev_tool_call[1]

                        tool_chain_data.append(
                            {
                                "chain_step_id": f"{chain_id}-{step_position}",
                                "session_key": session_key,
                                "chain_id": chain_id,
                                "tool_call_id": tool_use_id,
                                "tool_key": tool_key,
                                "step_position": step_position,
                                "prev_tool_key": prev_tool_key_val,
                                "time_since_prev_seconds": time_since_prev,
                            }
                        )
                        prev_tool_call = (tool_use_id, tool_key, timestamp)

                        if should_track:
                            _add_content_block(
                                content_blocks_data,
                                message_id,
                                session_key,
                                "tool_use",
                                idx,
                                date_key,
                                time_key,
                                input_summary,
                                truncate_output,
                                block,
                            )
                            total_content_blocks += 1

                    elif block_type == "tool_result":
                        has_tool_result = True
                        tool_use_id = block.get("tool_use_id")
                        result_content = block.get("content", "")
                        is_error = block.get("is_error", False)

                        if isinstance(result_content, list):
                            result_text = " ".join(
                                str(item.get("text", ""))
                                for item in result_content
                                if isinstance(item, dict)
                            )
                        else:
                            result_text = str(result_content)

                        output_text = result_text[:truncate_output]
                        output_char_count = len(result_text)

                        if tool_use_id and tool_use_id in tool_use_map:
                            tool_info = tool_use_map[tool_use_id]
                            tool_calls_data.append(
                                {
                                    "tool_call_id": tool_use_id,
                                    "session_key": session_key,
                                    "tool_key": tool_info["tool_key"],
                                    "date_key": tool_info["date_key"],
                                    "time_key": tool_info["time_key"],
                                    "invoke_message_id": tool_info["message_id"],
                                    "result_message_id": message_id,
                                    "timestamp": tool_info["timestamp"],
                                    "input_char_count": tool_info["input_char_count"],
                                    "output_char_count": output_char_count,
                                    "is_error": is_error,
                                    "input_json": tool_info["input_json"],
                                    "input_summary": tool_info["input_summary"],
                                    "output_text": output_text,
                                }
                            )

                            if is_error:
                                errors_data.append(
                                    {
                                        "error_id": f"{tool_use_id}-error",
                                        "tool_call_id": tool_use_id,
                                        "session_key": session_key,
                                        "tool_key": tool_info["tool_key"],
                                        "error_type_key": generate_dimension_key(
                                            "tool_error"
                                        ),
                                        "date_key": tool_info["date_key"],
                                        "time_key": tool_info["time_key"],
                                        "error_message": output_text,
                                        "timestamp": tool_info["timestamp"],
                                    }
                                )

                        if should_track:
                            _add_content_block(
                                content_blocks_data,
                                message_id,
                                session_key,
                                "tool_result",
                                idx,
                                date_key,
                                time_key,
                                output_text,
                                truncate_output,
                                block,
                            )
                            total_content_blocks += 1

                    elif block_type == "thinking":
                        has_thinking = True
                        thinking_count += 1
                        thinking_text = block.get("thinking", "")

                        if should_track:
                            _add_content_block(
                                content_blocks_data,
                                message_id,
                                session_key,
                                "thinking",
                                idx,
                                date_key,
                                time_key,
                                thinking_text,
                                truncate_output,
                                block,
                            )
                            total_content_blocks += 1

                    elif block_type == "image":
                        if should_track:
                            content_blocks_data.append(
                                {
                                    "content_block_id": f"{message_id}-{idx}",
                                    "message_id": message_id,
                                    "session_key": session_key,
                                    "content_block_type_key": generate_dimension_key(
                                        "image"
                                    ),
                                    "date_key": date_key,
                                    "time_key": time_key,
                                    "block_index": idx,
                                    "content_length": 0,
                                    "content_text": "[image]",
                                    "content_json": json.dumps(
                                        {"type": "image", "note": "content omitted"}
                                    ),
                                }
                            )
                            total_content_blocks += 1

                text_content = " ".join(texts)

            # Extract code blocks
            if text_content:
                extracted_blocks = extract_code_blocks(text_content)
                for cb_idx, cb in enumerate(extracted_blocks):
                    language = cb["language"]
                    languages_seen.add(language)
                    code_blocks_data.append(
                        {
                            "code_block_id": f"{message_id}-code-{cb_idx}",
                            "message_id": message_id,
                            "session_key": session_key,
                            "language_key": generate_dimension_key(language),
                            "date_key": date_key,
                            "time_key": time_key,
                            "block_index": cb_idx,
                            "line_count": cb["line_count"],
                            "char_count": cb["char_count"],
                            "code_text": cb["code"][:truncate_output],
                        }
                    )

                entities = extract_entities(text_content, message_id, session_key)
                entity_mentions_data.extend(entities)

            if entry_type == "user":
                user_count += 1
            else:
                assistant_count += 1

            word_cnt = count_words(text_content)
            token_est = estimate_tokens(text_content)

            response_time = None
            if parent_id and parent_id in message_timestamps and timestamp:
                parent_ts = message_timestamps[parent_id]
                if parent_ts:
                    response_time = (timestamp - parent_ts).total_seconds()

            conversation_depth = calculate_conversation_depth(
                message_id, parent_id, depth_map
            )
            depth_map[message_id] = conversation_depth

            if timestamp:
                message_timestamps[message_id] = timestamp

            message_type_key = generate_dimension_key(entry_type)
            model_key = generate_dimension_key(model) if model else None

            messages_data.append(
                {
                    "message_id": message_id,
                    "session_key": session_key,
                    "project_key": project_key,
                    "message_type_key": message_type_key,
                    "model_key": model_key,
                    "date_key": date_key,
                    "time_key": time_key,
                    "parent_message_id": parent_id,
                    "timestamp": timestamp,
                    "content_length": len(text_content),
                    "content_block_count": content_block_count,
                    "has_tool_use": has_tool_use,
                    "has_tool_result": has_tool_result,
                    "has_thinking": has_thinking,
                    "word_count": word_cnt,
                    "estimated_tokens": token_est,
                    "response_time_seconds": response_time,
                    "conversation_depth": conversation_depth,
                    "content_text": (
                        text_content[:truncate_output] if text_content else ""
                    ),
                    "content_json": content_json,
                    "is_sidechain": is_sidechain,
                }
            )

    # ==========================================================================
    # Phase 2: Load Dimensions
    # ==========================================================================

    _load_dimensions(
        conn,
        project_key,
        project_path,
        project_name,
        session_key,
        session_id,
        cwd,
        git_branch,
        version,
        first_timestamp,
        last_timestamp,
        is_agent,
        agent_id,
        parent_session_id,
        tools_seen,
        models_seen,
        dates_seen,
        messages_data,
        files_seen,
        languages_seen,
    )

    # ==========================================================================
    # Phase 3: Load Facts
    # ==========================================================================

    _load_facts(
        conn,
        session_key,
        project_key,
        messages_data,
        content_blocks_data,
        tool_calls_data,
        user_count,
        assistant_count,
        thinking_count,
        total_content_blocks,
        first_timestamp,
        last_timestamp,
        file_operations_data,
        code_blocks_data,
        errors_data,
        entity_mentions_data,
        tool_chain_data,
    )


def _add_content_block(
    blocks_data,
    message_id,
    session_key,
    block_type,
    idx,
    date_key,
    time_key,
    content_text,
    truncate_output,
    block,
):
    """Helper to add a content block to the list."""
    blocks_data.append(
        {
            "content_block_id": f"{message_id}-{idx}",
            "message_id": message_id,
            "session_key": session_key,
            "content_block_type_key": generate_dimension_key(block_type),
            "date_key": date_key,
            "time_key": time_key,
            "block_index": idx,
            "content_length": len(content_text) if content_text else 0,
            "content_text": content_text[:truncate_output] if content_text else "",
            "content_json": json.dumps(block),
        }
    )


def _load_dimensions(
    conn,
    project_key,
    project_path,
    project_name,
    session_key,
    session_id,
    cwd,
    git_branch,
    version,
    first_timestamp,
    last_timestamp,
    is_agent,
    agent_id,
    parent_session_id,
    tools_seen,
    models_seen,
    dates_seen,
    messages_data,
    files_seen,
    languages_seen,
):
    """Load all dimension tables."""

    # dim_project
    if not conn.execute(
        "SELECT 1 FROM dim_project WHERE project_key = ?", [project_key]
    ).fetchone():
        conn.execute(
            "INSERT INTO dim_project VALUES (?, ?, ?)",
            [project_key, project_path, project_name],
        )

    # dim_session
    if not conn.execute(
        "SELECT 1 FROM dim_session WHERE session_key = ?", [session_key]
    ).fetchone():
        conn.execute(
            """INSERT INTO dim_session
               (session_key, session_id, project_key, cwd, git_branch, version,
                first_timestamp, last_timestamp, is_agent, agent_id,
                parent_session_key, depth_level)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                session_key,
                session_id,
                project_key,
                cwd,
                git_branch,
                version,
                first_timestamp,
                last_timestamp,
                is_agent,
                agent_id,
                (
                    generate_dimension_key(parent_session_id)
                    if parent_session_id
                    else None
                ),
                0,
            ],
        )

    # dim_tool
    for tool_name in tools_seen:
        tool_key = generate_dimension_key(tool_name)
        if not conn.execute(
            "SELECT 1 FROM dim_tool WHERE tool_key = ?", [tool_key]
        ).fetchone():
            category = get_tool_category(tool_name)
            conn.execute(
                "INSERT INTO dim_tool VALUES (?, ?, ?)",
                [tool_key, tool_name, category],
            )

    # dim_model
    for model_name in models_seen:
        model_key = generate_dimension_key(model_name)
        if not conn.execute(
            "SELECT 1 FROM dim_model WHERE model_key = ?", [model_key]
        ).fetchone():
            family = get_model_family(model_name)
            conn.execute(
                "INSERT INTO dim_model VALUES (?, ?, ?)",
                [model_key, model_name, family],
            )

    # dim_date
    day_names = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    month_names = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]

    for date_key in dates_seen:
        if not conn.execute(
            "SELECT 1 FROM dim_date WHERE date_key = ?", [date_key]
        ).fetchone():
            year = date_key // 10000
            month = (date_key // 100) % 100
            day = date_key % 100
            try:
                full_date = datetime(year, month, day)
                day_of_week = full_date.weekday()
                quarter = (month - 1) // 3 + 1
                is_weekend = day_of_week >= 5

                conn.execute(
                    """INSERT INTO dim_date VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    [
                        date_key,
                        full_date.date(),
                        year,
                        month,
                        day,
                        day_of_week,
                        day_names[day_of_week],
                        month_names[month - 1],
                        quarter,
                        is_weekend,
                    ],
                )
            except ValueError:
                pass

    # dim_time
    times_seen = {msg["time_key"] for msg in messages_data if msg["time_key"]}
    for time_key in times_seen:
        if not conn.execute(
            "SELECT 1 FROM dim_time WHERE time_key = ?", [time_key]
        ).fetchone():
            hour = time_key // 100
            minute = time_key % 100
            time_of_day = get_time_of_day(hour)
            conn.execute(
                "INSERT INTO dim_time VALUES (?, ?, ?, ?)",
                [time_key, hour, minute, time_of_day],
            )

    # dim_file
    for file_path, file_info in files_seen.items():
        if not conn.execute(
            "SELECT 1 FROM dim_file WHERE file_key = ?", [file_info["file_key"]]
        ).fetchone():
            conn.execute(
                "INSERT INTO dim_file VALUES (?, ?, ?, ?, ?)",
                [
                    file_info["file_key"],
                    file_info["file_path"],
                    file_info["file_name"],
                    file_info["file_extension"],
                    file_info["directory_path"],
                ],
            )

    # dim_programming_language
    for language in languages_seen:
        lang_key = generate_dimension_key(language)
        if not conn.execute(
            "SELECT 1 FROM dim_programming_language WHERE language_key = ?", [lang_key]
        ).fetchone():
            extensions = LANGUAGE_EXTENSIONS.get(language, [])
            ext_str = ",".join(extensions) if extensions else ""
            conn.execute(
                "INSERT INTO dim_programming_language VALUES (?, ?, ?)",
                [lang_key, language, ext_str],
            )

    # dim_error_type
    error_type_key = generate_dimension_key("tool_error")
    if not conn.execute(
        "SELECT 1 FROM dim_error_type WHERE error_type_key = ?", [error_type_key]
    ).fetchone():
        conn.execute(
            "INSERT INTO dim_error_type VALUES (?, ?, ?)",
            [error_type_key, "tool_error", "execution"],
        )


def _load_facts(
    conn,
    session_key,
    project_key,
    messages_data,
    content_blocks_data,
    tool_calls_data,
    user_count,
    assistant_count,
    thinking_count,
    total_content_blocks,
    first_timestamp,
    last_timestamp,
    file_operations_data,
    code_blocks_data,
    errors_data,
    entity_mentions_data,
    tool_chain_data,
):
    """Load all fact tables."""

    # fact_messages
    for msg in messages_data:
        conn.execute(
            """INSERT INTO fact_messages
               (message_id, session_key, project_key, message_type_key, model_key,
                date_key, time_key, parent_message_id, timestamp, content_length,
                content_block_count, has_tool_use, has_tool_result, has_thinking,
                word_count, estimated_tokens, response_time_seconds, conversation_depth,
                content_text, content_json, is_sidechain)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                msg["message_id"],
                msg["session_key"],
                msg["project_key"],
                msg["message_type_key"],
                msg["model_key"],
                msg["date_key"],
                msg["time_key"],
                msg["parent_message_id"],
                msg["timestamp"],
                msg["content_length"],
                msg["content_block_count"],
                msg["has_tool_use"],
                msg["has_tool_result"],
                msg["has_thinking"],
                msg["word_count"],
                msg["estimated_tokens"],
                msg["response_time_seconds"],
                msg["conversation_depth"],
                msg["content_text"],
                msg["content_json"],
                msg["is_sidechain"],
            ],
        )

    # fact_content_blocks
    for block in content_blocks_data:
        conn.execute(
            """INSERT INTO fact_content_blocks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                block["content_block_id"],
                block["message_id"],
                block["session_key"],
                block["content_block_type_key"],
                block["date_key"],
                block["time_key"],
                block["block_index"],
                block["content_length"],
                block["content_text"],
                block["content_json"],
            ],
        )

    # fact_tool_calls
    for tc in tool_calls_data:
        conn.execute(
            """INSERT INTO fact_tool_calls VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                tc["tool_call_id"],
                tc["session_key"],
                tc["tool_key"],
                tc["date_key"],
                tc["time_key"],
                tc["invoke_message_id"],
                tc["result_message_id"],
                tc["timestamp"],
                tc["input_char_count"],
                tc["output_char_count"],
                tc["is_error"],
                tc["input_json"],
                tc["input_summary"],
                tc["output_text"],
            ],
        )

    # fact_session_summary
    session_duration = 0
    if first_timestamp and last_timestamp:
        session_duration = int((last_timestamp - first_timestamp).total_seconds())

    first_date_key = None
    if first_timestamp:
        first_date_key = int(first_timestamp.strftime("%Y%m%d"))

    conn.execute(
        """INSERT INTO fact_session_summary VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        [
            session_key,
            project_key,
            first_date_key,
            user_count + assistant_count,
            user_count,
            assistant_count,
            len(tool_calls_data),
            thinking_count,
            total_content_blocks,
            session_duration,
            first_timestamp,
            last_timestamp,
        ],
    )

    # fact_file_operations
    for fop in file_operations_data:
        conn.execute(
            """INSERT INTO fact_file_operations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                fop["file_operation_id"],
                fop["tool_call_id"],
                fop["session_key"],
                fop["file_key"],
                fop["tool_key"],
                fop["date_key"],
                fop["time_key"],
                fop["operation_type"],
                fop["file_size_chars"],
                fop["timestamp"],
            ],
        )

    # fact_code_blocks
    for cb in code_blocks_data:
        conn.execute(
            """INSERT INTO fact_code_blocks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                cb["code_block_id"],
                cb["message_id"],
                cb["session_key"],
                cb["language_key"],
                cb["date_key"],
                cb["time_key"],
                cb["block_index"],
                cb["line_count"],
                cb["char_count"],
                cb["code_text"],
            ],
        )

    # fact_errors
    for err in errors_data:
        conn.execute(
            """INSERT INTO fact_errors VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                err["error_id"],
                err["tool_call_id"],
                err["session_key"],
                err["tool_key"],
                err["error_type_key"],
                err["date_key"],
                err["time_key"],
                err["error_message"],
                err["timestamp"],
            ],
        )

    # fact_entity_mentions
    for em in entity_mentions_data:
        conn.execute(
            """INSERT INTO fact_entity_mentions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                em["mention_id"],
                em["message_id"],
                em["session_key"],
                em["entity_type_key"],
                em["entity_text"],
                em["entity_normalized"],
                em["context_snippet"],
                em["position_start"],
                em["position_end"],
            ],
        )

    # fact_tool_chain_steps
    for tc in tool_chain_data:
        conn.execute(
            """INSERT INTO fact_tool_chain_steps VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                tc["chain_step_id"],
                tc["session_key"],
                tc["chain_id"],
                tc["tool_call_id"],
                tc["tool_key"],
                tc["step_position"],
                tc["prev_tool_key"],
                tc["time_since_prev_seconds"],
            ],
        )
