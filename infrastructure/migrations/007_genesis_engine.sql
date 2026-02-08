-- ============================================================
-- Migration 007: Genesis Engine
-- Business Launcher Infrastructure
-- ============================================================
--
-- Purpose:
--   The Genesis Engine is the meta-layer that sits ABOVE agents.
--   It doesn't do work — it HIRES workers. These tables track
--   the end-to-end flow from "I have a business idea" to
--   "agents are deployed and running."
--
-- Tables:
--   1. genesis_sessions — End-to-end onboarding lifecycle
--   2. business_blueprints — Approved strategic plans (shared brain)
--   3. vertical_credentials — Encrypted per-vertical API key storage
--
-- Architecture:
--   User → ArchitectAgent Interview → BusinessBlueprint → ConfigGenerator
--   → YAML files → CredentialManager → Shadow Mode Launch
--
-- Five-Gate Safety:
--   1. Interview completeness (automated)
--   2. Blueprint human review (manual)
--   3. Config Pydantic validation (automated)
--   4. Credential collection (manual)
--   5. Shadow mode launch (all new verticals sandboxed)
-- ============================================================


-- ─── 1. Genesis Sessions ────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS genesis_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Which user initiated this session
    user_id TEXT,

    -- Current stage in the Genesis flow
    status TEXT NOT NULL DEFAULT 'interview'
        CHECK (status IN (
            'interview',            -- Stage 1: Gathering context
            'market_analysis',      -- Stage 1.5: AI analyzing market
            'blueprint_generation', -- Stage 2: Generating blueprint
            'blueprint_review',     -- Stage 2: Waiting for human approval
            'config_generation',    -- Stage 3: Generating YAML configs
            'config_review',        -- Stage 3: Waiting for human review
            'credential_collection',-- Stage 4: Collecting API keys
            'launching',            -- Stage 4: Deploying agents
            'launched',             -- Complete: Agents running in shadow mode
            'failed',               -- Error: requires manual intervention
            'cancelled'             -- User cancelled the flow
        )),

    -- Interview context (accumulates as questions are answered)
    business_context JSONB NOT NULL DEFAULT '{}',

    -- Questions tracking
    questions_asked JSONB NOT NULL DEFAULT '[]',
    interview_completion_score FLOAT DEFAULT 0.0,

    -- Reference to the generated blueprint
    blueprint_id UUID,

    -- Reference to the generated vertical
    vertical_id TEXT,

    -- Agent run ID for the ArchitectAgent execution
    agent_run_id TEXT,

    -- Error tracking
    error_message TEXT,
    error_node TEXT,

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ
);

-- Fast lookup by status for dashboard
CREATE INDEX IF NOT EXISTS idx_genesis_sessions_status
    ON genesis_sessions(status, created_at DESC);

-- Fast lookup by user
CREATE INDEX IF NOT EXISTS idx_genesis_sessions_user
    ON genesis_sessions(user_id, created_at DESC);

-- Fast lookup by vertical
CREATE INDEX IF NOT EXISTS idx_genesis_sessions_vertical
    ON genesis_sessions(vertical_id) WHERE vertical_id IS NOT NULL;


-- ─── 2. Business Blueprints ────────────────────────────────────────

CREATE TABLE IF NOT EXISTS business_blueprints (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Identity
    vertical_id TEXT NOT NULL,
    vertical_name TEXT NOT NULL,
    industry TEXT NOT NULL,

    -- The complete blueprint data (serialized BusinessBlueprint)
    blueprint_data JSONB NOT NULL,

    -- Lifecycle
    status TEXT NOT NULL DEFAULT 'draft'
        CHECK (status IN (
            'draft',
            'pending_review',
            'approved',
            'config_generated',
            'launched',
            'archived'
        )),

    -- Review tracking
    version INTEGER NOT NULL DEFAULT 1,
    approved_by TEXT,
    approved_at TIMESTAMPTZ,
    review_feedback TEXT,

    -- Which session created this
    session_id UUID REFERENCES genesis_sessions(id) ON DELETE SET NULL,

    -- AI reasoning (preserved for learning)
    strategy_reasoning TEXT,
    risk_factors JSONB DEFAULT '[]',
    success_metrics JSONB DEFAULT '[]',

    -- Metadata
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Fast lookup by vertical
CREATE INDEX IF NOT EXISTS idx_blueprints_vertical
    ON business_blueprints(vertical_id, version DESC);

-- Fast lookup by status for dashboard
CREATE INDEX IF NOT EXISTS idx_blueprints_status
    ON business_blueprints(status, created_at DESC);

-- Fast lookup by industry for shared brain learning
CREATE INDEX IF NOT EXISTS idx_blueprints_industry
    ON business_blueprints(industry);


-- ─── 3. Vertical Credentials ───────────────────────────────────────

CREATE TABLE IF NOT EXISTS vertical_credentials (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Which vertical these credentials belong to
    vertical_id TEXT NOT NULL,

    -- Credential identity
    credential_name TEXT NOT NULL,   -- Human-readable: "Apollo API Key"
    env_var_name TEXT NOT NULL,      -- Environment variable: "APOLLO_API_KEY"

    -- Encrypted value (Fernet symmetric encryption)
    -- The encryption key is derived from ENCLAVE_MASTER_KEY env var
    -- NEVER store plaintext credentials
    encrypted_value TEXT,

    -- Whether the credential has been provided
    is_set BOOLEAN NOT NULL DEFAULT FALSE,

    -- Metadata
    instructions TEXT DEFAULT '',    -- How to obtain this credential
    required BOOLEAN NOT NULL DEFAULT TRUE,
    last_validated_at TIMESTAMPTZ,   -- When we last checked it works
    validation_status TEXT DEFAULT 'unknown'
        CHECK (validation_status IN ('unknown', 'valid', 'invalid', 'expired')),

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Each vertical can only have one entry per env var
    UNIQUE(vertical_id, env_var_name)
);

-- Fast lookup by vertical
CREATE INDEX IF NOT EXISTS idx_credentials_vertical
    ON vertical_credentials(vertical_id);


-- ─── 4. Updated Timestamp Trigger ──────────────────────────────────

-- Auto-update updated_at on any row change
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply to all genesis tables
DO $$
BEGIN
    -- genesis_sessions
    IF NOT EXISTS (
        SELECT 1 FROM pg_trigger
        WHERE tgname = 'trg_genesis_sessions_updated_at'
    ) THEN
        CREATE TRIGGER trg_genesis_sessions_updated_at
            BEFORE UPDATE ON genesis_sessions
            FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    END IF;

    -- business_blueprints
    IF NOT EXISTS (
        SELECT 1 FROM pg_trigger
        WHERE tgname = 'trg_blueprints_updated_at'
    ) THEN
        CREATE TRIGGER trg_blueprints_updated_at
            BEFORE UPDATE ON business_blueprints
            FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    END IF;

    -- vertical_credentials
    IF NOT EXISTS (
        SELECT 1 FROM pg_trigger
        WHERE tgname = 'trg_credentials_updated_at'
    ) THEN
        CREATE TRIGGER trg_credentials_updated_at
            BEFORE UPDATE ON vertical_credentials
            FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    END IF;
END
$$;


-- ─── 5. RPC Functions ──────────────────────────────────────────────

-- Create or update a genesis session
CREATE OR REPLACE FUNCTION upsert_genesis_session(
    p_session_id UUID,
    p_user_id TEXT DEFAULT NULL,
    p_status TEXT DEFAULT 'interview',
    p_business_context JSONB DEFAULT '{}',
    p_questions_asked JSONB DEFAULT '[]',
    p_completion_score FLOAT DEFAULT 0.0,
    p_blueprint_id UUID DEFAULT NULL,
    p_vertical_id TEXT DEFAULT NULL,
    p_agent_run_id TEXT DEFAULT NULL,
    p_error_message TEXT DEFAULT NULL,
    p_error_node TEXT DEFAULT NULL
)
RETURNS genesis_sessions
LANGUAGE plpgsql
AS $$
DECLARE
    result genesis_sessions;
BEGIN
    INSERT INTO genesis_sessions (
        id, user_id, status, business_context,
        questions_asked, interview_completion_score,
        blueprint_id, vertical_id, agent_run_id,
        error_message, error_node
    ) VALUES (
        p_session_id, p_user_id, p_status, p_business_context,
        p_questions_asked, p_completion_score,
        p_blueprint_id, p_vertical_id, p_agent_run_id,
        p_error_message, p_error_node
    )
    ON CONFLICT (id) DO UPDATE SET
        status = COALESCE(EXCLUDED.status, genesis_sessions.status),
        business_context = COALESCE(EXCLUDED.business_context, genesis_sessions.business_context),
        questions_asked = COALESCE(EXCLUDED.questions_asked, genesis_sessions.questions_asked),
        interview_completion_score = COALESCE(EXCLUDED.interview_completion_score, genesis_sessions.interview_completion_score),
        blueprint_id = COALESCE(EXCLUDED.blueprint_id, genesis_sessions.blueprint_id),
        vertical_id = COALESCE(EXCLUDED.vertical_id, genesis_sessions.vertical_id),
        agent_run_id = COALESCE(EXCLUDED.agent_run_id, genesis_sessions.agent_run_id),
        error_message = EXCLUDED.error_message,
        error_node = EXCLUDED.error_node,
        completed_at = CASE
            WHEN EXCLUDED.status IN ('launched', 'failed', 'cancelled')
            THEN NOW()
            ELSE genesis_sessions.completed_at
        END
    RETURNING * INTO result;

    RETURN result;
END;
$$;


-- Store a business blueprint
CREATE OR REPLACE FUNCTION store_blueprint(
    p_blueprint_id UUID,
    p_vertical_id TEXT,
    p_blueprint_data JSONB,
    p_status TEXT DEFAULT 'draft',
    p_session_id UUID DEFAULT NULL
)
RETURNS business_blueprints
LANGUAGE plpgsql
AS $$
DECLARE
    result business_blueprints;
    v_name TEXT;
    v_industry TEXT;
    v_reasoning TEXT;
    v_risks JSONB;
    v_metrics JSONB;
BEGIN
    -- Extract fields from blueprint data
    v_name := p_blueprint_data->>'vertical_name';
    v_industry := p_blueprint_data->>'industry';
    v_reasoning := p_blueprint_data->>'strategy_reasoning';
    v_risks := COALESCE(p_blueprint_data->'risk_factors', '[]'::jsonb);
    v_metrics := COALESCE(p_blueprint_data->'success_metrics', '[]'::jsonb);

    INSERT INTO business_blueprints (
        id, vertical_id, vertical_name, industry,
        blueprint_data, status, session_id,
        strategy_reasoning, risk_factors, success_metrics
    ) VALUES (
        p_blueprint_id, p_vertical_id,
        COALESCE(v_name, 'Unknown'), COALESCE(v_industry, 'Unknown'),
        p_blueprint_data, p_status, p_session_id,
        v_reasoning, v_risks, v_metrics
    )
    ON CONFLICT (id) DO UPDATE SET
        blueprint_data = EXCLUDED.blueprint_data,
        status = EXCLUDED.status,
        version = business_blueprints.version + 1,
        strategy_reasoning = EXCLUDED.strategy_reasoning,
        risk_factors = EXCLUDED.risk_factors,
        success_metrics = EXCLUDED.success_metrics
    RETURNING * INTO result;

    RETURN result;
END;
$$;


-- Get blueprints by industry (for shared brain learning)
CREATE OR REPLACE FUNCTION get_blueprints_by_industry(
    p_industry TEXT,
    p_status TEXT DEFAULT 'launched',
    p_limit INTEGER DEFAULT 10
)
RETURNS SETOF business_blueprints
LANGUAGE sql STABLE
AS $$
    SELECT *
    FROM business_blueprints
    WHERE industry = p_industry
      AND status = p_status
    ORDER BY created_at DESC
    LIMIT p_limit;
$$;


-- Get active genesis sessions (for dashboard)
CREATE OR REPLACE FUNCTION get_active_genesis_sessions(
    p_user_id TEXT DEFAULT NULL,
    p_limit INTEGER DEFAULT 20
)
RETURNS SETOF genesis_sessions
LANGUAGE sql STABLE
AS $$
    SELECT *
    FROM genesis_sessions
    WHERE status NOT IN ('launched', 'failed', 'cancelled')
      AND (p_user_id IS NULL OR user_id = p_user_id)
    ORDER BY updated_at DESC
    LIMIT p_limit;
$$;


-- ─── 6. Row-Level Security ─────────────────────────────────────────

-- Enable RLS on all genesis tables
ALTER TABLE genesis_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE business_blueprints ENABLE ROW LEVEL SECURITY;
ALTER TABLE vertical_credentials ENABLE ROW LEVEL SECURITY;

-- Service role has full access (used by the platform backend)
CREATE POLICY IF NOT EXISTS genesis_sessions_service_policy
    ON genesis_sessions FOR ALL
    USING (true)
    WITH CHECK (true);

CREATE POLICY IF NOT EXISTS blueprints_service_policy
    ON business_blueprints FOR ALL
    USING (true)
    WITH CHECK (true);

CREATE POLICY IF NOT EXISTS credentials_service_policy
    ON vertical_credentials FOR ALL
    USING (true)
    WITH CHECK (true);
