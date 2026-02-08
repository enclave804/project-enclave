-- Project Enclave: RPC Functions
-- These are called via supabase.rpc() from the Python client.

-- ============================================================================
-- Vector Similarity Search for RAG
-- ============================================================================
CREATE OR REPLACE FUNCTION match_knowledge_chunks(
    query_embedding VECTOR(1536),
    match_count INT DEFAULT 5,
    match_threshold FLOAT DEFAULT 0.7,
    p_vertical_id TEXT DEFAULT NULL,
    p_chunk_type TEXT DEFAULT NULL
)
RETURNS TABLE (
    id UUID,
    content TEXT,
    chunk_type TEXT,
    metadata JSONB,
    source_id TEXT,
    source_type TEXT,
    similarity FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        kc.id,
        kc.content,
        kc.chunk_type,
        kc.metadata,
        kc.source_id,
        kc.source_type,
        1 - (kc.embedding <=> query_embedding) AS similarity
    FROM knowledge_chunks kc
    WHERE
        (p_vertical_id IS NULL OR kc.vertical_id = p_vertical_id)
        AND (p_chunk_type IS NULL OR kc.chunk_type = p_chunk_type)
        AND 1 - (kc.embedding <=> query_embedding) > match_threshold
    ORDER BY kc.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

-- ============================================================================
-- Outreach Statistics Aggregation
-- ============================================================================
CREATE OR REPLACE FUNCTION get_outreach_stats(
    p_vertical_id TEXT,
    p_days INT DEFAULT 30
)
RETURNS TABLE (
    total_sent BIGINT,
    total_opened BIGINT,
    total_replied BIGINT,
    total_bounced BIGINT,
    total_meetings BIGINT,
    open_rate FLOAT,
    reply_rate FLOAT,
    bounce_rate FLOAT,
    meeting_rate FLOAT
)
LANGUAGE plpgsql
AS $$
DECLARE
    cutoff TIMESTAMPTZ := NOW() - (p_days || ' days')::INTERVAL;
    v_total_sent BIGINT;
BEGIN
    SELECT COUNT(*) INTO v_total_sent
    FROM outreach_events
    WHERE vertical_id = p_vertical_id
      AND sent_at >= cutoff
      AND status != 'draft';

    RETURN QUERY
    SELECT
        v_total_sent AS total_sent,
        COUNT(*) FILTER (WHERE oe.opened_at IS NOT NULL) AS total_opened,
        COUNT(*) FILTER (WHERE oe.replied_at IS NOT NULL) AS total_replied,
        COUNT(*) FILTER (WHERE oe.status = 'bounced') AS total_bounced,
        COUNT(*) FILTER (WHERE oe.reply_intent = 'meeting_request') AS total_meetings,
        CASE WHEN v_total_sent > 0
            THEN COUNT(*) FILTER (WHERE oe.opened_at IS NOT NULL)::FLOAT / v_total_sent
            ELSE 0 END AS open_rate,
        CASE WHEN v_total_sent > 0
            THEN COUNT(*) FILTER (WHERE oe.replied_at IS NOT NULL)::FLOAT / v_total_sent
            ELSE 0 END AS reply_rate,
        CASE WHEN v_total_sent > 0
            THEN COUNT(*) FILTER (WHERE oe.status = 'bounced')::FLOAT / v_total_sent
            ELSE 0 END AS bounce_rate,
        CASE WHEN v_total_sent > 0
            THEN COUNT(*) FILTER (WHERE oe.reply_intent = 'meeting_request')::FLOAT / v_total_sent
            ELSE 0 END AS meeting_rate
    FROM outreach_events oe
    WHERE oe.vertical_id = p_vertical_id
      AND oe.sent_at >= cutoff
      AND oe.status != 'draft';
END;
$$;

-- ============================================================================
-- Template Usage Counter
-- ============================================================================
CREATE OR REPLACE FUNCTION increment_template_usage(p_template_id UUID)
RETURNS VOID
LANGUAGE plpgsql
AS $$
BEGIN
    UPDATE outreach_templates
    SET times_used = COALESCE(times_used, 0) + 1,
        updated_at = NOW()
    WHERE id = p_template_id;
END;
$$;

-- ============================================================================
-- Template Performance Recalculation
-- ============================================================================
CREATE OR REPLACE FUNCTION recalculate_template_stats(
    p_vertical_id TEXT,
    p_days INT DEFAULT 30
)
RETURNS VOID
LANGUAGE plpgsql
AS $$
DECLARE
    cutoff TIMESTAMPTZ := NOW() - (p_days || ' days')::INTERVAL;
BEGIN
    UPDATE outreach_templates ot
    SET
        times_used = stats.total_sent,
        open_rate = stats.open_rate,
        reply_rate = stats.reply_rate,
        meeting_rate = stats.meeting_rate,
        updated_at = NOW()
    FROM (
        SELECT
            oe.template_id,
            COUNT(*) AS total_sent,
            COUNT(*) FILTER (WHERE oe.opened_at IS NOT NULL)::FLOAT / NULLIF(COUNT(*), 0) AS open_rate,
            COUNT(*) FILTER (WHERE oe.replied_at IS NOT NULL)::FLOAT / NULLIF(COUNT(*), 0) AS reply_rate,
            COUNT(*) FILTER (WHERE oe.reply_intent = 'meeting_request')::FLOAT / NULLIF(COUNT(*), 0) AS meeting_rate
        FROM outreach_events oe
        WHERE oe.vertical_id = p_vertical_id
          AND oe.sent_at >= cutoff
          AND oe.template_id IS NOT NULL
        GROUP BY oe.template_id
    ) stats
    WHERE ot.id = stats.template_id
      AND ot.vertical_id = p_vertical_id;
END;
$$;

-- ============================================================================
-- Check Duplicate Contact (within cooldown period)
-- ============================================================================
CREATE OR REPLACE FUNCTION check_contact_cooldown(
    p_email TEXT,
    p_vertical_id TEXT,
    p_cooldown_days INT DEFAULT 90
)
RETURNS TABLE (
    is_duplicate BOOLEAN,
    last_contacted_at TIMESTAMPTZ,
    last_status TEXT,
    days_since_contact INT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        CASE WHEN oe.sent_at IS NOT NULL AND oe.sent_at > NOW() - (p_cooldown_days || ' days')::INTERVAL
            THEN TRUE ELSE FALSE END AS is_duplicate,
        oe.sent_at AS last_contacted_at,
        oe.status AS last_status,
        EXTRACT(DAY FROM NOW() - oe.sent_at)::INT AS days_since_contact
    FROM contacts c
    LEFT JOIN outreach_events oe ON oe.contact_id = c.id
        AND oe.vertical_id = p_vertical_id
    WHERE c.email = p_email
      AND c.vertical_id = p_vertical_id
    ORDER BY oe.sent_at DESC NULLS LAST
    LIMIT 1;
END;
$$;
