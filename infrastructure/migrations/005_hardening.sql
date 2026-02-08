-- Sovereign Venture Engine: Hardening & Reliability Fixes
-- Migration 005 — addresses zombie tasks, staleness, and Realtime.

-- ============================================================================
-- 1. ZOMBIE TASK PREVENTION
-- Add heartbeat column so running agents prove they're alive.
-- ============================================================================
ALTER TABLE task_queue ADD COLUMN IF NOT EXISTS heartbeat_at TIMESTAMPTZ;

-- Function to recover zombie tasks (called by cron or supervisor)
CREATE OR REPLACE FUNCTION recover_zombie_tasks(
    p_stale_minutes INT DEFAULT 10
)
RETURNS INT
LANGUAGE plpgsql
AS $$
DECLARE
    recovered INT;
BEGIN
    WITH zombies AS (
        UPDATE task_queue
        SET status = 'pending',
            claimed_at = NULL,
            heartbeat_at = NULL,
            retry_count = retry_count + 1,
            updated_at = NOW()
        WHERE status IN ('claimed', 'running')
          AND updated_at < NOW() - (p_stale_minutes || ' minutes')::INTERVAL
          AND retry_count < max_retries
        RETURNING id
    )
    SELECT COUNT(*) INTO recovered FROM zombies;

    -- Permanently fail tasks that exceeded max retries
    UPDATE task_queue
    SET status = 'failed',
        error_message = COALESCE(error_message, '') || ' [zombie: exceeded max retries]',
        completed_at = NOW(),
        updated_at = NOW()
    WHERE status IN ('claimed', 'running')
      AND updated_at < NOW() - (p_stale_minutes || ' minutes')::INTERVAL
      AND retry_count >= max_retries;

    RETURN recovered;
END;
$$;

-- ============================================================================
-- 2. ENABLE REALTIME on task_queue
-- Allows the supervisor/orchestrator to subscribe to new tasks
-- instead of polling every N seconds.
-- ============================================================================
ALTER PUBLICATION supabase_realtime ADD TABLE task_queue;

-- ============================================================================
-- 3. AGENT ERROR TRACKING (for Circuit Breaker)
-- Track consecutive errors per agent so we can auto-disable runaway agents.
-- ============================================================================
ALTER TABLE agents ADD COLUMN IF NOT EXISTS consecutive_errors INTEGER DEFAULT 0;
ALTER TABLE agents ADD COLUMN IF NOT EXISTS max_consecutive_errors INTEGER DEFAULT 5;
ALTER TABLE agents ADD COLUMN IF NOT EXISTS disabled_at TIMESTAMPTZ;
ALTER TABLE agents ADD COLUMN IF NOT EXISTS disabled_reason TEXT;

-- Function to record an agent error and auto-disable if threshold hit
CREATE OR REPLACE FUNCTION record_agent_error(
    p_agent_id TEXT,
    p_vertical_id TEXT,
    p_error_message TEXT DEFAULT NULL
)
RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    v_agent agents%ROWTYPE;
BEGIN
    UPDATE agents
    SET consecutive_errors = consecutive_errors + 1,
        updated_at = NOW()
    WHERE agent_id = p_agent_id
      AND vertical_id = p_vertical_id
    RETURNING * INTO v_agent;

    IF v_agent.id IS NULL THEN
        RETURN jsonb_build_object('status', 'agent_not_found');
    END IF;

    -- Circuit breaker: auto-disable if too many consecutive errors
    IF v_agent.consecutive_errors >= v_agent.max_consecutive_errors THEN
        UPDATE agents
        SET enabled = false,
            disabled_at = NOW(),
            disabled_reason = 'circuit_breaker: ' || v_agent.consecutive_errors || ' consecutive errors. Last: ' || COALESCE(p_error_message, 'unknown'),
            updated_at = NOW()
        WHERE agent_id = p_agent_id
          AND vertical_id = p_vertical_id;

        RETURN jsonb_build_object(
            'status', 'disabled',
            'consecutive_errors', v_agent.consecutive_errors,
            'reason', 'circuit_breaker'
        );
    END IF;

    RETURN jsonb_build_object(
        'status', 'recorded',
        'consecutive_errors', v_agent.consecutive_errors
    );
END;
$$;

-- Function to reset error counter on success
CREATE OR REPLACE FUNCTION reset_agent_errors(
    p_agent_id TEXT,
    p_vertical_id TEXT
)
RETURNS VOID
LANGUAGE plpgsql
AS $$
BEGIN
    UPDATE agents
    SET consecutive_errors = 0,
        updated_at = NOW()
    WHERE agent_id = p_agent_id
      AND vertical_id = p_vertical_id;
END;
$$;
