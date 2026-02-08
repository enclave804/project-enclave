-- Sovereign Venture Engine: Agent Platform Tables
-- Migration 003 — adds multi-agent framework tables.
-- Backward compatible: existing tables are only ALTERed (new columns).

-- ============================================================================
-- AGENTS (Registry)
-- ============================================================================
CREATE TABLE agents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_id TEXT NOT NULL,             -- human-readable key (e.g. "outreach")
    agent_type TEXT NOT NULL,           -- implementation class key
    name TEXT NOT NULL,
    description TEXT DEFAULT '',
    config JSONB DEFAULT '{}',          -- full YAML config as JSON
    enabled BOOLEAN DEFAULT true,
    vertical_id TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(agent_id, vertical_id)
);

CREATE INDEX idx_agents_vertical ON agents(vertical_id);
CREATE INDEX idx_agents_type ON agents(agent_type);
CREATE INDEX idx_agents_enabled ON agents(enabled) WHERE enabled = true;

-- ============================================================================
-- AGENT RUNS (Execution History)
-- ============================================================================
CREATE TABLE agent_runs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    run_id TEXT NOT NULL UNIQUE,        -- UUID string from Python
    agent_id TEXT NOT NULL,
    agent_type TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'started',  -- started, completed, failed
    input_data JSONB DEFAULT '{}',
    output_data JSONB DEFAULT '{}',
    error_message TEXT,
    duration_ms INTEGER,
    parent_run_id TEXT,                 -- for sub-agent calls
    vertical_id TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_agent_runs_agent ON agent_runs(agent_id);
CREATE INDEX idx_agent_runs_vertical ON agent_runs(vertical_id);
CREATE INDEX idx_agent_runs_status ON agent_runs(status);
CREATE INDEX idx_agent_runs_created ON agent_runs(created_at);
CREATE INDEX idx_agent_runs_parent ON agent_runs(parent_run_id) WHERE parent_run_id IS NOT NULL;

-- ============================================================================
-- AGENT CONVERSATIONS (Multi-turn Memory)
-- ============================================================================
CREATE TABLE agent_conversations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    conversation_id TEXT NOT NULL,       -- groups messages in one thread
    agent_id TEXT NOT NULL,
    role TEXT NOT NULL,                   -- 'system', 'user', 'assistant', 'tool'
    content TEXT NOT NULL,
    related_entity_id TEXT,              -- contact_id, company_id, etc.
    related_entity_type TEXT,            -- 'contact', 'company', 'opportunity'
    metadata JSONB DEFAULT '{}',
    vertical_id TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_agent_conv_conversation ON agent_conversations(conversation_id);
CREATE INDEX idx_agent_conv_agent ON agent_conversations(agent_id);
CREATE INDEX idx_agent_conv_entity ON agent_conversations(related_entity_type, related_entity_id)
    WHERE related_entity_id IS NOT NULL;
CREATE INDEX idx_agent_conv_vertical ON agent_conversations(vertical_id);

-- ============================================================================
-- TASK QUEUE (Cross-Agent Coordination)
-- ============================================================================
CREATE TABLE task_queue (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    task_id TEXT NOT NULL UNIQUE,         -- UUID string from Python
    source_agent_id TEXT,                 -- agent that created this task (NULL = manual/external)
    target_agent_id TEXT NOT NULL,        -- agent that should execute this
    task_type TEXT NOT NULL,              -- agent-specific task category
    priority INTEGER NOT NULL DEFAULT 5,  -- 1 = highest, 10 = lowest
    status TEXT NOT NULL DEFAULT 'pending',  -- pending, claimed, running, completed, failed
    input_data JSONB DEFAULT '{}',
    output_data JSONB DEFAULT '{}',
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,
    scheduled_at TIMESTAMPTZ,            -- for delayed execution
    claimed_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    vertical_id TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_task_queue_target ON task_queue(target_agent_id);
CREATE INDEX idx_task_queue_status ON task_queue(status);
CREATE INDEX idx_task_queue_priority ON task_queue(priority);
CREATE INDEX idx_task_queue_vertical ON task_queue(vertical_id);
CREATE INDEX idx_task_queue_scheduled ON task_queue(scheduled_at)
    WHERE scheduled_at IS NOT NULL AND status = 'pending';
-- Composite index for the claim query (most critical for performance)
CREATE INDEX idx_task_queue_claim ON task_queue(target_agent_id, status, priority, created_at)
    WHERE status = 'pending';

-- ============================================================================
-- SHARED INSIGHTS (Cross-Agent Brain / Hive Mind)
-- ============================================================================
CREATE TABLE shared_insights (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_agent_id TEXT NOT NULL,
    insight_type TEXT NOT NULL,           -- 'winning_pattern', 'objection_rebuttal', 'keyword_performance', 'market_signal', 'customer_preference'
    title TEXT,
    content TEXT NOT NULL,
    embedding VECTOR(1536),
    confidence_score FLOAT DEFAULT 0.5,
    usage_count INTEGER DEFAULT 0,
    related_entity_id TEXT,
    related_entity_type TEXT,
    metadata JSONB DEFAULT '{}',
    vertical_id TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- HNSW index for fast semantic search across insights
CREATE INDEX idx_insights_embedding ON shared_insights
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX idx_insights_agent ON shared_insights(source_agent_id);
CREATE INDEX idx_insights_type ON shared_insights(insight_type);
CREATE INDEX idx_insights_vertical ON shared_insights(vertical_id);
CREATE INDEX idx_insights_confidence ON shared_insights(confidence_score);

-- ============================================================================
-- AGENT CONTENT (Generated Artifacts)
-- ============================================================================
CREATE TABLE agent_content (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_id TEXT NOT NULL,
    content_type TEXT NOT NULL,           -- 'blog_post', 'landing_page', 'case_study', 'ad_copy', 'proposal'
    title TEXT NOT NULL,
    body TEXT,
    status TEXT NOT NULL DEFAULT 'draft', -- draft, review, approved, published, archived
    seo_score FLOAT,
    meta_title TEXT,
    meta_description TEXT,
    published_url TEXT,
    published_at TIMESTAMPTZ,
    metadata JSONB DEFAULT '{}',
    vertical_id TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_agent_content_agent ON agent_content(agent_id);
CREATE INDEX idx_agent_content_type ON agent_content(content_type);
CREATE INDEX idx_agent_content_status ON agent_content(status);
CREATE INDEX idx_agent_content_vertical ON agent_content(vertical_id);

-- ============================================================================
-- ALTER EXISTING TABLES (Backward Compatible)
-- ============================================================================

-- Add agent_id to knowledge_chunks so agents can tag their contributions
ALTER TABLE knowledge_chunks ADD COLUMN IF NOT EXISTS agent_id TEXT;
CREATE INDEX IF NOT EXISTS idx_knowledge_agent ON knowledge_chunks(agent_id) WHERE agent_id IS NOT NULL;

-- Add agent_id to pipeline_runs for linking legacy pipeline runs to agents
ALTER TABLE pipeline_runs ADD COLUMN IF NOT EXISTS agent_id TEXT;
CREATE INDEX IF NOT EXISTS idx_pipeline_agent ON pipeline_runs(agent_id) WHERE agent_id IS NOT NULL;

-- ============================================================================
-- UPDATED_AT TRIGGERS for new tables
-- ============================================================================
CREATE TRIGGER agents_updated_at
    BEFORE UPDATE ON agents
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER agent_runs_updated_at
    BEFORE UPDATE ON agent_runs
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER task_queue_updated_at
    BEFORE UPDATE ON task_queue
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER shared_insights_updated_at
    BEFORE UPDATE ON shared_insights
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER agent_content_updated_at
    BEFORE UPDATE ON agent_content
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- ============================================================================
-- ROW LEVEL SECURITY (Vertical Isolation)
-- ============================================================================
ALTER TABLE agents ENABLE ROW LEVEL SECURITY;
ALTER TABLE agent_runs ENABLE ROW LEVEL SECURITY;
ALTER TABLE agent_conversations ENABLE ROW LEVEL SECURITY;
ALTER TABLE task_queue ENABLE ROW LEVEL SECURITY;
ALTER TABLE shared_insights ENABLE ROW LEVEL SECURITY;
ALTER TABLE agent_content ENABLE ROW LEVEL SECURITY;

CREATE POLICY vertical_isolation_agents ON agents
    USING (vertical_id = current_setting('app.current_vertical', true));

CREATE POLICY vertical_isolation_agent_runs ON agent_runs
    USING (vertical_id = current_setting('app.current_vertical', true));

CREATE POLICY vertical_isolation_agent_conversations ON agent_conversations
    USING (vertical_id = current_setting('app.current_vertical', true));

CREATE POLICY vertical_isolation_task_queue ON task_queue
    USING (vertical_id = current_setting('app.current_vertical', true));

CREATE POLICY vertical_isolation_shared_insights ON shared_insights
    USING (vertical_id = current_setting('app.current_vertical', true));

CREATE POLICY vertical_isolation_agent_content ON agent_content
    USING (vertical_id = current_setting('app.current_vertical', true));
