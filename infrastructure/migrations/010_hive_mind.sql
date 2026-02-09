-- ============================================================
-- Migration 010: Hive Mind — Phase 13
-- ============================================================
-- Adds tables for the Experiment Engine, lead scoring, and
-- RPC functions for insight reinforcement (boost/decay).
-- Builds on existing shared_insights table (001 migration).
-- ============================================================

-- ─── Experiments ──────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS experiments (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    experiment_id   TEXT UNIQUE NOT NULL,
    vertical_id     TEXT NOT NULL,
    agent_id        TEXT NOT NULL DEFAULT '',

    name            TEXT NOT NULL,
    variants        JSONB NOT NULL DEFAULT '[]',
    metric          TEXT NOT NULL DEFAULT 'conversion',
    status          TEXT NOT NULL DEFAULT 'active'
                    CHECK (status IN ('active', 'concluded', 'paused')),

    total_observations INT NOT NULL DEFAULT 0,
    metadata        JSONB DEFAULT '{}',

    created_at      TIMESTAMPTZ DEFAULT now(),
    updated_at      TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_experiments_vertical
    ON experiments(vertical_id);
CREATE INDEX IF NOT EXISTS idx_experiments_status
    ON experiments(status);
CREATE INDEX IF NOT EXISTS idx_experiments_agent
    ON experiments(agent_id);

-- ─── Experiment Outcomes ──────────────────────────────────────

CREATE TABLE IF NOT EXISTS experiment_outcomes (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    experiment_id   TEXT NOT NULL REFERENCES experiments(experiment_id),
    vertical_id     TEXT NOT NULL,

    variant         TEXT NOT NULL,
    outcome         TEXT NOT NULL,

    created_at      TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_experiment_outcomes_exp
    ON experiment_outcomes(experiment_id);
CREATE INDEX IF NOT EXISTS idx_experiment_outcomes_vertical
    ON experiment_outcomes(vertical_id);

-- ─── Lead Scores ──────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS lead_scores (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    vertical_id     TEXT NOT NULL,
    agent_id        TEXT NOT NULL DEFAULT 'lead_scorer',

    -- Lead reference
    lead_id         TEXT DEFAULT '',
    company_name    TEXT DEFAULT '',
    contact_name    TEXT DEFAULT '',
    contact_email   TEXT DEFAULT '',

    -- Score details
    score           INT NOT NULL DEFAULT 0 CHECK (score >= 0 AND score <= 100),
    tier            TEXT NOT NULL DEFAULT 'cold'
                    CHECK (tier IN ('hot', 'warm', 'lukewarm', 'cold')),
    factors         JSONB DEFAULT '[]',

    -- Model info
    model_type      TEXT NOT NULL DEFAULT 'heuristic'
                    CHECK (model_type IN ('heuristic', 'ml')),
    features        JSONB DEFAULT '{}',

    created_at      TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_lead_scores_vertical
    ON lead_scores(vertical_id);
CREATE INDEX IF NOT EXISTS idx_lead_scores_tier
    ON lead_scores(tier);
CREATE INDEX IF NOT EXISTS idx_lead_scores_score
    ON lead_scores(score DESC);

-- ─── RPC: Boost Insight Confidence ────────────────────────────

CREATE OR REPLACE FUNCTION boost_insight_confidence(
    p_insight_id UUID,
    p_boost_amount FLOAT DEFAULT 0.05
) RETURNS VOID AS $$
BEGIN
    UPDATE shared_insights
    SET confidence_score = LEAST(confidence_score + p_boost_amount, 1.0),
        usage_count = COALESCE(usage_count, 0) + 1,
        updated_at = now()
    WHERE id = p_insight_id;
END;
$$ LANGUAGE plpgsql;

-- ─── RPC: Decay Insight Confidence ────────────────────────────

CREATE OR REPLACE FUNCTION decay_insight_confidence(
    p_insight_id UUID,
    p_decay_amount FLOAT DEFAULT 0.02
) RETURNS VOID AS $$
BEGIN
    UPDATE shared_insights
    SET confidence_score = GREATEST(confidence_score - p_decay_amount, 0.0),
        updated_at = now()
    WHERE id = p_insight_id;
END;
$$ LANGUAGE plpgsql;

-- ─── RPC: Get Brain Stats ─────────────────────────────────────

CREATE OR REPLACE FUNCTION get_brain_stats(
    p_vertical_id TEXT
) RETURNS JSONB AS $$
DECLARE
    result JSONB;
BEGIN
    SELECT jsonb_build_object(
        'total_insights', (
            SELECT COUNT(*)
            FROM shared_insights
            WHERE vertical_id = p_vertical_id
        ),
        'avg_confidence', (
            SELECT COALESCE(AVG(confidence_score), 0)
            FROM shared_insights
            WHERE vertical_id = p_vertical_id
        ),
        'total_experiments', (
            SELECT COUNT(*)
            FROM experiments
            WHERE vertical_id = p_vertical_id
        ),
        'active_experiments', (
            SELECT COUNT(*)
            FROM experiments
            WHERE vertical_id = p_vertical_id AND status = 'active'
        ),
        'total_scored_leads', (
            SELECT COUNT(*)
            FROM lead_scores
            WHERE vertical_id = p_vertical_id
        ),
        'avg_lead_score', (
            SELECT COALESCE(AVG(score), 0)
            FROM lead_scores
            WHERE vertical_id = p_vertical_id
        )
    ) INTO result;

    RETURN result;
END;
$$ LANGUAGE plpgsql;
