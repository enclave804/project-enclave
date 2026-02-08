-- ============================================================
-- Migration 006: God Mode Lite
-- RLHF Data Collection + Shadow Agent Infrastructure
-- ============================================================
--
-- Purpose:
--   1. training_examples: Save every human correction (bad_draft → good_rewrite)
--      so we can build a fine-tuning dataset over time.
--   2. Shadow mode columns on agents: allow "ghost" agents that silently
--      process the same tasks as production agents, without external effects.
--
-- Architecture:
--   - training_examples collects RLHF pairs: (model_output, human_correction)
--   - Shadow agents have shadow_mode=true and shadow_of pointing to the
--     champion agent. The task queue duplicates tasks for shadows automatically.
--   - Shadow agents MUST NOT call external tools (emails, APIs) — enforced
--     by the sandbox protocol checking shadow_mode in task metadata.
--
-- This is "God Mode Lite" — we collect the data now, optimize later (Phase 4).
-- ============================================================

-- ─── 1. RLHF Training Examples ─────────────────────────────────────

CREATE TABLE IF NOT EXISTS training_examples (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id TEXT NOT NULL,
    vertical_id TEXT NOT NULL DEFAULT 'enclave_guard',

    -- The input context that produced the output
    task_input JSONB NOT NULL DEFAULT '{}',

    -- What the agent generated (the "candidate" output)
    model_output TEXT NOT NULL,

    -- What the human corrected it to (the "gold" output)
    -- NULL if the human only scored but didn't rewrite
    human_correction TEXT,

    -- Quality score: 0 = terrible, 100 = perfect
    -- Used for filtering training data by quality threshold
    score INTEGER CHECK (score >= 0 AND score <= 100),

    -- How this example was collected
    source TEXT NOT NULL DEFAULT 'manual_review'
        CHECK (source IN (
            'manual_review',    -- Human reviewed in dashboard/Telegram
            'shadow_comparison', -- Shadow agent vs champion comparison
            'a_b_test',         -- A/B test winner/loser
            'automated_eval'    -- LLM-as-judge scored
        )),

    -- Metadata for filtering (e.g., lead industry, task type)
    metadata JSONB DEFAULT '{}',

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Fast lookup by agent + quality for training data export
CREATE INDEX IF NOT EXISTS idx_training_examples_agent_score
    ON training_examples(agent_id, score DESC);

-- Fast lookup by vertical for dashboard
CREATE INDEX IF NOT EXISTS idx_training_examples_vertical
    ON training_examples(vertical_id, created_at DESC);

-- Filter by source type
CREATE INDEX IF NOT EXISTS idx_training_examples_source
    ON training_examples(source, created_at DESC);


-- ─── 2. Shadow Mode on Agents ──────────────────────────────────────

-- shadow_mode: when true, this agent runs in parallel but NEVER
-- triggers external effects (emails, API calls, etc.)
ALTER TABLE agents
    ADD COLUMN IF NOT EXISTS shadow_mode BOOLEAN NOT NULL DEFAULT FALSE;

-- shadow_of: the agent_id of the "champion" this shadow is copying.
-- When the champion gets a task, the shadow gets a duplicate.
ALTER TABLE agents
    ADD COLUMN IF NOT EXISTS shadow_of TEXT;

-- Index for fast shadow lookup during task dispatch
CREATE INDEX IF NOT EXISTS idx_agents_shadow_of
    ON agents(shadow_of) WHERE shadow_of IS NOT NULL;


-- ─── 3. Helper RPC: Get Training Examples for Export ────────────────

CREATE OR REPLACE FUNCTION get_training_examples(
    p_agent_id TEXT DEFAULT NULL,
    p_vertical_id TEXT DEFAULT NULL,
    p_min_score INTEGER DEFAULT NULL,
    p_source TEXT DEFAULT NULL,
    p_limit INTEGER DEFAULT 1000
)
RETURNS SETOF training_examples
LANGUAGE sql STABLE
AS $$
    SELECT *
    FROM training_examples
    WHERE (p_agent_id IS NULL OR agent_id = p_agent_id)
      AND (p_vertical_id IS NULL OR vertical_id = p_vertical_id)
      AND (p_min_score IS NULL OR score >= p_min_score)
      AND (p_source IS NULL OR source = p_source)
    ORDER BY created_at DESC
    LIMIT p_limit;
$$;


-- ─── 4. Helper RPC: Get Shadow Agents for a Champion ────────────────

CREATE OR REPLACE FUNCTION get_shadow_agents(
    p_champion_agent_id TEXT,
    p_vertical_id TEXT DEFAULT NULL
)
RETURNS SETOF agents
LANGUAGE sql STABLE
AS $$
    SELECT *
    FROM agents
    WHERE shadow_of = p_champion_agent_id
      AND shadow_mode = TRUE
      AND enabled = TRUE
      AND (p_vertical_id IS NULL OR vertical_id = p_vertical_id);
$$;


-- ─── 5. Training Stats RPC ─────────────────────────────────────────

CREATE OR REPLACE FUNCTION get_training_stats(
    p_vertical_id TEXT DEFAULT NULL
)
RETURNS TABLE(
    agent_id TEXT,
    total_examples BIGINT,
    scored_examples BIGINT,
    corrected_examples BIGINT,
    avg_score NUMERIC,
    latest_example TIMESTAMPTZ
)
LANGUAGE sql STABLE
AS $$
    SELECT
        agent_id,
        COUNT(*) AS total_examples,
        COUNT(score) AS scored_examples,
        COUNT(human_correction) AS corrected_examples,
        ROUND(AVG(score)::numeric, 1) AS avg_score,
        MAX(created_at) AS latest_example
    FROM training_examples
    WHERE (p_vertical_id IS NULL OR vertical_id = p_vertical_id)
    GROUP BY agent_id
    ORDER BY total_examples DESC;
$$;
