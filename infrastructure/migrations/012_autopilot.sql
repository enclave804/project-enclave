-- ============================================================
-- Migration 012: Autopilot — Self-Healing, Budget, Strategy
-- Phase 15: Autonomous Mode
-- ============================================================
-- Adds tables for:
--   1. autopilot_sessions  — Strategy session tracking
--   2. optimization_actions — Individual actions taken
--   3. budget_snapshots     — Time-series budget data
-- ============================================================

-- ── 1. Autopilot Sessions ─────────────────────────────────────

CREATE TABLE IF NOT EXISTS autopilot_sessions (
    id              UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    vertical_id     TEXT NOT NULL,
    session_type    TEXT NOT NULL CHECK (session_type IN (
                        'full_analysis', 'healing', 'budget', 'strategy'
                    )),
    status          TEXT NOT NULL DEFAULT 'running' CHECK (status IN (
                        'running', 'completed', 'failed', 'cancelled'
                    )),
    metrics_snapshot JSONB DEFAULT '{}',
    detected_issues  JSONB DEFAULT '[]',
    strategy_output  JSONB DEFAULT '{}',
    actions_taken    JSONB DEFAULT '[]',
    started_at      TIMESTAMPTZ DEFAULT now(),
    completed_at    TIMESTAMPTZ,
    created_at      TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_autopilot_sessions_vertical
    ON autopilot_sessions (vertical_id);

CREATE INDEX IF NOT EXISTS idx_autopilot_sessions_status
    ON autopilot_sessions (status);

CREATE INDEX IF NOT EXISTS idx_autopilot_sessions_type
    ON autopilot_sessions (session_type);

CREATE INDEX IF NOT EXISTS idx_autopilot_sessions_created
    ON autopilot_sessions (created_at DESC);


-- ── 2. Optimization Actions ───────────────────────────────────

CREATE TABLE IF NOT EXISTS optimization_actions (
    id              UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    session_id      UUID REFERENCES autopilot_sessions(id) ON DELETE SET NULL,
    vertical_id     TEXT NOT NULL,
    action_type     TEXT NOT NULL CHECK (action_type IN (
                        'config_fix', 'agent_restart', 'agent_disable',
                        'budget_reallocation', 'experiment_launched',
                        'schedule_adjustment', 'alert_sent'
                    )),
    target          TEXT NOT NULL,  -- agent_id, campaign_id, or experiment_id
    parameters      JSONB DEFAULT '{}',
    result          TEXT CHECK (result IN (
                        'success', 'failed', 'pending', 'rejected'
                    )),
    approved_by     TEXT,           -- user_id or 'auto'
    error_message   TEXT,
    created_at      TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_optimization_actions_session
    ON optimization_actions (session_id);

CREATE INDEX IF NOT EXISTS idx_optimization_actions_vertical
    ON optimization_actions (vertical_id);

CREATE INDEX IF NOT EXISTS idx_optimization_actions_type
    ON optimization_actions (action_type);

CREATE INDEX IF NOT EXISTS idx_optimization_actions_result
    ON optimization_actions (result);

CREATE INDEX IF NOT EXISTS idx_optimization_actions_created
    ON optimization_actions (created_at DESC);


-- ── 3. Budget Snapshots ───────────────────────────────────────

CREATE TABLE IF NOT EXISTS budget_snapshots (
    id              UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    vertical_id     TEXT NOT NULL,
    org_id          UUID,
    period          TEXT NOT NULL,   -- '2025-02' (monthly) or '2025-02-09' (daily)
    total_spend     NUMERIC DEFAULT 0,
    total_revenue   NUMERIC DEFAULT 0,
    roas            NUMERIC DEFAULT 0,
    breakdown       JSONB DEFAULT '{}',  -- per-campaign breakdown
    created_at      TIMESTAMPTZ DEFAULT now(),

    UNIQUE (vertical_id, period)    -- one snapshot per period per vertical
);

CREATE INDEX IF NOT EXISTS idx_budget_snapshots_vertical
    ON budget_snapshots (vertical_id);

CREATE INDEX IF NOT EXISTS idx_budget_snapshots_period
    ON budget_snapshots (period DESC);

CREATE INDEX IF NOT EXISTS idx_budget_snapshots_org
    ON budget_snapshots (org_id);


-- ── RLS Policies ──────────────────────────────────────────────

ALTER TABLE autopilot_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE optimization_actions ENABLE ROW LEVEL SECURITY;
ALTER TABLE budget_snapshots ENABLE ROW LEVEL SECURITY;

-- Service role bypass (application uses service key)
CREATE POLICY autopilot_sessions_service ON autopilot_sessions
    FOR ALL
    USING (true)
    WITH CHECK (true);

CREATE POLICY optimization_actions_service ON optimization_actions
    FOR ALL
    USING (true)
    WITH CHECK (true);

CREATE POLICY budget_snapshots_service ON budget_snapshots
    FOR ALL
    USING (true)
    WITH CHECK (true);


-- ── RPC Functions ─────────────────────────────────────────────

-- Get autopilot summary stats for a vertical
CREATE OR REPLACE FUNCTION get_autopilot_stats(p_vertical_id TEXT)
RETURNS JSONB
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
DECLARE
    result JSONB;
BEGIN
    SELECT jsonb_build_object(
        'total_sessions', (
            SELECT COUNT(*)
            FROM autopilot_sessions
            WHERE vertical_id = p_vertical_id
        ),
        'completed_sessions', (
            SELECT COUNT(*)
            FROM autopilot_sessions
            WHERE vertical_id = p_vertical_id
            AND status = 'completed'
        ),
        'total_actions', (
            SELECT COUNT(*)
            FROM optimization_actions
            WHERE vertical_id = p_vertical_id
        ),
        'successful_actions', (
            SELECT COUNT(*)
            FROM optimization_actions
            WHERE vertical_id = p_vertical_id
            AND result = 'success'
        ),
        'pending_actions', (
            SELECT COUNT(*)
            FROM optimization_actions
            WHERE vertical_id = p_vertical_id
            AND result = 'pending'
        ),
        'latest_session', (
            SELECT jsonb_build_object(
                'id', id,
                'session_type', session_type,
                'status', status,
                'started_at', started_at,
                'completed_at', completed_at
            )
            FROM autopilot_sessions
            WHERE vertical_id = p_vertical_id
            ORDER BY created_at DESC
            LIMIT 1
        ),
        'latest_budget', (
            SELECT jsonb_build_object(
                'period', period,
                'total_spend', total_spend,
                'total_revenue', total_revenue,
                'roas', roas
            )
            FROM budget_snapshots
            WHERE vertical_id = p_vertical_id
            ORDER BY created_at DESC
            LIMIT 1
        )
    ) INTO result;

    RETURN result;
END;
$$;

-- Get action breakdown by type for a vertical
CREATE OR REPLACE FUNCTION get_action_breakdown(p_vertical_id TEXT, p_days INTEGER DEFAULT 30)
RETURNS JSONB
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
DECLARE
    result JSONB;
BEGIN
    SELECT jsonb_agg(
        jsonb_build_object(
            'action_type', action_type,
            'total', cnt,
            'successful', success_cnt
        )
    )
    FROM (
        SELECT
            action_type,
            COUNT(*) AS cnt,
            COUNT(*) FILTER (WHERE result = 'success') AS success_cnt
        FROM optimization_actions
        WHERE vertical_id = p_vertical_id
        AND created_at >= now() - (p_days || ' days')::interval
        GROUP BY action_type
        ORDER BY cnt DESC
    ) sub
    INTO result;

    RETURN COALESCE(result, '[]'::jsonb);
END;
$$;
