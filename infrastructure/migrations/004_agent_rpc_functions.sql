-- Sovereign Venture Engine: Agent RPC Functions
-- Migration 004 — database functions for agent coordination.

-- ============================================================================
-- Atomic Task Claiming (prevents double-processing)
-- ============================================================================
CREATE OR REPLACE FUNCTION claim_next_task(
    p_agent_id TEXT,
    p_vertical_id TEXT DEFAULT NULL
)
RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    v_task task_queue%ROWTYPE;
BEGIN
    -- Atomically find and claim the highest-priority pending task.
    -- FOR UPDATE SKIP LOCKED ensures concurrent workers never claim
    -- the same task — critical for multi-instance deployments.
    SELECT * INTO v_task
    FROM task_queue
    WHERE target_agent_id = p_agent_id
      AND status = 'pending'
      AND (scheduled_at IS NULL OR scheduled_at <= NOW())
      AND (p_vertical_id IS NULL OR vertical_id = p_vertical_id)
    ORDER BY priority ASC, created_at ASC
    LIMIT 1
    FOR UPDATE SKIP LOCKED;

    IF v_task.id IS NULL THEN
        RETURN NULL;
    END IF;

    -- Transition: pending → claimed
    UPDATE task_queue
    SET status = 'claimed',
        claimed_at = NOW(),
        updated_at = NOW()
    WHERE id = v_task.id;

    RETURN jsonb_build_object(
        'task_id', v_task.task_id,
        'source_agent_id', v_task.source_agent_id,
        'target_agent_id', v_task.target_agent_id,
        'task_type', v_task.task_type,
        'priority', v_task.priority,
        'input_data', v_task.input_data,
        'retry_count', v_task.retry_count,
        'vertical_id', v_task.vertical_id,
        'created_at', v_task.created_at
    );
END;
$$;

-- ============================================================================
-- Agent Statistics (for dashboard and monitoring)
-- ============================================================================
CREATE OR REPLACE FUNCTION get_agent_stats(
    p_vertical_id TEXT,
    p_agent_id TEXT DEFAULT NULL,
    p_days INT DEFAULT 30
)
RETURNS TABLE (
    agent_id TEXT,
    total_runs BIGINT,
    completed_runs BIGINT,
    failed_runs BIGINT,
    success_rate FLOAT,
    avg_duration_ms FLOAT,
    total_tasks_created BIGINT,
    total_tasks_received BIGINT
)
LANGUAGE plpgsql
AS $$
DECLARE
    cutoff TIMESTAMPTZ := NOW() - (p_days || ' days')::INTERVAL;
BEGIN
    RETURN QUERY
    SELECT
        ar.agent_id,
        COUNT(*) AS total_runs,
        COUNT(*) FILTER (WHERE ar.status = 'completed') AS completed_runs,
        COUNT(*) FILTER (WHERE ar.status = 'failed') AS failed_runs,
        CASE WHEN COUNT(*) > 0
            THEN COUNT(*) FILTER (WHERE ar.status = 'completed')::FLOAT / COUNT(*)
            ELSE 0 END AS success_rate,
        AVG(ar.duration_ms)::FLOAT AS avg_duration_ms,
        (SELECT COUNT(*) FROM task_queue tq
         WHERE tq.source_agent_id = ar.agent_id
           AND tq.vertical_id = p_vertical_id
           AND tq.created_at >= cutoff) AS total_tasks_created,
        (SELECT COUNT(*) FROM task_queue tq
         WHERE tq.target_agent_id = ar.agent_id
           AND tq.vertical_id = p_vertical_id
           AND tq.created_at >= cutoff) AS total_tasks_received
    FROM agent_runs ar
    WHERE ar.vertical_id = p_vertical_id
      AND ar.created_at >= cutoff
      AND (p_agent_id IS NULL OR ar.agent_id = p_agent_id)
    GROUP BY ar.agent_id;
END;
$$;

-- ============================================================================
-- Search Shared Insights (vector similarity)
-- ============================================================================
CREATE OR REPLACE FUNCTION match_shared_insights(
    query_embedding VECTOR(1536),
    match_count INT DEFAULT 5,
    match_threshold FLOAT DEFAULT 0.7,
    p_vertical_id TEXT DEFAULT NULL,
    p_insight_type TEXT DEFAULT NULL,
    p_source_agent_id TEXT DEFAULT NULL
)
RETURNS TABLE (
    id UUID,
    source_agent_id TEXT,
    insight_type TEXT,
    title TEXT,
    content TEXT,
    confidence_score FLOAT,
    metadata JSONB,
    similarity FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        si.id,
        si.source_agent_id,
        si.insight_type,
        si.title,
        si.content,
        si.confidence_score,
        si.metadata,
        1 - (si.embedding <=> query_embedding) AS similarity
    FROM shared_insights si
    WHERE
        (p_vertical_id IS NULL OR si.vertical_id = p_vertical_id)
        AND (p_insight_type IS NULL OR si.insight_type = p_insight_type)
        AND (p_source_agent_id IS NULL OR si.source_agent_id = p_source_agent_id)
        AND si.embedding IS NOT NULL
        AND 1 - (si.embedding <=> query_embedding) > match_threshold
    ORDER BY si.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

-- ============================================================================
-- Increment Insight Usage Counter
-- ============================================================================
CREATE OR REPLACE FUNCTION increment_insight_usage(p_insight_id UUID)
RETURNS VOID
LANGUAGE plpgsql
AS $$
BEGIN
    UPDATE shared_insights
    SET usage_count = COALESCE(usage_count, 0) + 1,
        updated_at = NOW()
    WHERE id = p_insight_id;
END;
$$;
