"""LLM enrichment pipeline for star schema."""

from datetime import datetime

from .utils import generate_dimension_key


def run_llm_enrichment(
    conn,
    enrich_func,
    model_name="claude-3-haiku-20240307",
    batch_size=10,
    session_key=None,
):
    """Run LLM enrichment on messages that haven't been enriched yet.

    This function provides a framework for enriching messages with LLM-derived
    classifications like intent, sentiment, and topics.

    Args:
        conn: DuckDB connection
        enrich_func: Function(messages) -> list of enrichment results.
                    Each result should be a dict with:
                    - message_id: str
                    - intent: str (should match dim_intent.intent_name)
                    - sentiment: str (should match dim_sentiment.sentiment_name)
                    - topics: list[str] (should match dim_topic.topic_name values)
                    - complexity_score: float (0-1)
                    - confidence_score: float (0-1)
        model_name: Name of the model used for enrichment (for tracking)
        batch_size: Number of messages to process at once
        session_key: Optional session key to limit enrichment to one session

    Returns:
        dict with counts: messages_enriched, topics_assigned
    """
    query = """
        SELECT m.message_id, m.session_key, m.content_text, mt.message_type
        FROM fact_messages m
        JOIN dim_message_type mt ON m.message_type_key = mt.message_type_key
        LEFT JOIN fact_message_enrichment e ON m.message_id = e.message_id
        WHERE e.message_id IS NULL
          AND m.content_text IS NOT NULL
          AND LENGTH(m.content_text) > 0
    """
    params = []
    if session_key:
        query += " AND m.session_key = ?"
        params.append(session_key)
    query += f" LIMIT {batch_size}"

    messages = conn.execute(query, params).fetchall()

    if not messages:
        return {"messages_enriched": 0, "topics_assigned": 0}

    message_data = [
        {
            "message_id": row[0],
            "session_key": row[1],
            "content_text": row[2],
            "message_type": row[3],
        }
        for row in messages
    ]

    enrichment_results = enrich_func(message_data)

    intent_lookup = {
        row[0]: row[1]
        for row in conn.execute(
            "SELECT intent_name, intent_key FROM dim_intent"
        ).fetchall()
    }
    sentiment_lookup = {
        row[0]: row[1]
        for row in conn.execute(
            "SELECT sentiment_name, sentiment_key FROM dim_sentiment"
        ).fetchall()
    }
    topic_lookup = {
        row[0]: row[1]
        for row in conn.execute(
            "SELECT topic_name, topic_key FROM dim_topic"
        ).fetchall()
    }

    messages_enriched = 0
    topics_assigned = 0
    enriched_at = datetime.now()

    for result in enrichment_results:
        message_id = result.get("message_id")
        if not message_id:
            continue

        msg_session_key = None
        for md in message_data:
            if md["message_id"] == message_id:
                msg_session_key = md["session_key"]
                break

        intent_name = result.get("intent", "question")
        sentiment_name = result.get("sentiment", "neutral")
        intent_key = intent_lookup.get(intent_name, intent_lookup.get("question"))
        sentiment_key = sentiment_lookup.get(
            sentiment_name, sentiment_lookup.get("neutral")
        )

        enrichment_id = generate_dimension_key(message_id, "enrichment")
        conn.execute(
            """INSERT INTO fact_message_enrichment
               (enrichment_id, message_id, session_key, intent_key, sentiment_key,
                complexity_score, confidence_score, enrichment_model, enriched_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                enrichment_id,
                message_id,
                msg_session_key,
                intent_key,
                sentiment_key,
                result.get("complexity_score", 0.5),
                result.get("confidence_score", 0.5),
                model_name,
                enriched_at,
            ],
        )
        messages_enriched += 1

        topics = result.get("topics", [])
        for idx, topic_name in enumerate(topics):
            topic_key = topic_lookup.get(topic_name)
            if topic_key:
                message_topic_id = generate_dimension_key(message_id, "topic", str(idx))
                relevance = 1.0 - (idx * 0.1) if idx < 10 else 0.1
                conn.execute(
                    """INSERT INTO fact_message_topics
                       (message_topic_id, message_id, topic_key, relevance_score)
                       VALUES (?, ?, ?, ?)""",
                    [message_topic_id, message_id, topic_key, relevance],
                )
                topics_assigned += 1

    return {"messages_enriched": messages_enriched, "topics_assigned": topics_assigned}


def run_session_insights_enrichment(
    conn,
    insight_func,
    model_name="claude-3-haiku-20240307",
    session_key=None,
):
    """Generate LLM-based insights for sessions.

    Args:
        conn: DuckDB connection
        insight_func: Function(session_data) -> insight dict with:
                     - summary_text: str
                     - key_decisions: str
                     - outcome_status: str (success, partial, failed, unknown)
                     - task_completed: bool
                     - primary_intent: str (should match dim_intent.intent_name)
                     - complexity_score: float (0-1)
        model_name: Name of the model used for enrichment
        session_key: Optional session key to process only one session

    Returns:
        dict with count: sessions_enriched
    """
    query = """
        SELECT s.session_key, s.session_id,
               ss.total_messages, ss.total_tool_calls,
               ss.session_duration_seconds
        FROM dim_session s
        JOIN fact_session_summary ss ON s.session_key = ss.session_key
        LEFT JOIN fact_session_insights i ON s.session_key = i.session_key
        WHERE i.session_key IS NULL
    """
    params = []
    if session_key:
        query += " AND s.session_key = ?"
        params.append(session_key)

    sessions = conn.execute(query, params).fetchall()

    if not sessions:
        return {"sessions_enriched": 0}

    intent_lookup = {
        row[0]: row[1]
        for row in conn.execute(
            "SELECT intent_name, intent_key FROM dim_intent"
        ).fetchall()
    }

    sessions_enriched = 0
    enriched_at = datetime.now()

    for row in sessions:
        sess_key = row[0]
        session_id = row[1]
        total_messages = row[2]
        total_tool_calls = row[3]
        duration_seconds = row[4]

        messages = conn.execute(
            """SELECT content_text, mt.message_type
               FROM fact_messages m
               JOIN dim_message_type mt ON m.message_type_key = mt.message_type_key
               WHERE m.session_key = ?
               ORDER BY m.timestamp
               LIMIT 50""",
            [sess_key],
        ).fetchall()

        session_data = {
            "session_key": sess_key,
            "session_id": session_id,
            "total_messages": total_messages,
            "total_tool_calls": total_tool_calls,
            "duration_seconds": duration_seconds,
            "messages": [
                {"content": row[0], "type": row[1]} for row in messages if row[0]
            ],
        }

        insight = insight_func(session_data)

        primary_intent_name = insight.get("primary_intent", "question")
        primary_intent_key = intent_lookup.get(
            primary_intent_name, intent_lookup.get("question")
        )

        insight_id = generate_dimension_key(sess_key, "insight")
        conn.execute(
            """INSERT INTO fact_session_insights
               (insight_id, session_key, summary_text, key_decisions, outcome_status,
                task_completed, primary_intent_key, complexity_score,
                enrichment_model, enriched_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                insight_id,
                sess_key,
                insight.get("summary_text", ""),
                insight.get("key_decisions", ""),
                insight.get("outcome_status", "unknown"),
                insight.get("task_completed", False),
                primary_intent_key,
                insight.get("complexity_score", 0.5),
                model_name,
                enriched_at,
            ],
        )
        sessions_enriched += 1

    return {"sessions_enriched": sessions_enriched}
