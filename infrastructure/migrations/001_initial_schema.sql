-- Project Enclave: Initial Database Schema
-- Run this migration against your Supabase project.
-- Requires the pgvector extension to be enabled first.

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Enable UUID generation
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================================================
-- COMPANIES
-- ============================================================================
CREATE TABLE companies (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name TEXT NOT NULL,
    domain TEXT,
    industry TEXT,
    employee_count INTEGER,
    tech_stack JSONB DEFAULT '{}',
    vulnerabilities JSONB DEFAULT '[]',
    apollo_id TEXT,
    linkedin_url TEXT,
    website_url TEXT,
    source TEXT,                          -- how we discovered them
    enrichment_data JSONB DEFAULT '{}',   -- raw enrichment payload
    enriched_at TIMESTAMPTZ,
    vertical_id TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(domain, vertical_id)
);

CREATE INDEX idx_companies_vertical ON companies(vertical_id);
CREATE INDEX idx_companies_domain ON companies(domain);
CREATE INDEX idx_companies_apollo ON companies(apollo_id);
CREATE INDEX idx_companies_industry ON companies(industry);

-- ============================================================================
-- CONTACTS
-- ============================================================================
CREATE TABLE contacts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    company_id UUID REFERENCES companies(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    title TEXT,
    email TEXT,
    linkedin_url TEXT,
    phone TEXT,
    role_category TEXT,                   -- 'decision_maker', 'influencer', 'end_user'
    persona_id TEXT,                      -- matches persona.id in config.yaml
    apollo_id TEXT,
    seniority TEXT,
    vertical_id TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(email, vertical_id)
);

CREATE INDEX idx_contacts_vertical ON contacts(vertical_id);
CREATE INDEX idx_contacts_company ON contacts(company_id);
CREATE INDEX idx_contacts_email ON contacts(email);
CREATE INDEX idx_contacts_persona ON contacts(persona_id);

-- ============================================================================
-- OUTREACH TEMPLATES
-- ============================================================================
CREATE TABLE outreach_templates (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name TEXT NOT NULL,
    channel TEXT NOT NULL DEFAULT 'email',
    approach_type TEXT NOT NULL,           -- 'vulnerability_alert', 'compliance_angle', etc.
    subject_template TEXT,
    body_template TEXT,
    target_persona TEXT,                   -- matches persona.id in config.yaml
    sequence_step INTEGER DEFAULT 1,
    times_used INTEGER DEFAULT 0,
    open_rate FLOAT,
    reply_rate FLOAT,
    meeting_rate FLOAT,
    active BOOLEAN DEFAULT true,
    vertical_id TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_templates_vertical ON outreach_templates(vertical_id);
CREATE INDEX idx_templates_approach ON outreach_templates(approach_type);
CREATE INDEX idx_templates_persona ON outreach_templates(target_persona);

-- ============================================================================
-- OUTREACH EVENTS
-- ============================================================================
CREATE TABLE outreach_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    contact_id UUID REFERENCES contacts(id) ON DELETE CASCADE,
    company_id UUID REFERENCES companies(id) ON DELETE SET NULL,
    channel TEXT NOT NULL DEFAULT 'email',
    direction TEXT NOT NULL DEFAULT 'outbound',  -- 'outbound', 'inbound'
    template_id UUID REFERENCES outreach_templates(id) ON DELETE SET NULL,
    sequence_name TEXT,
    sequence_step INTEGER DEFAULT 1,
    content_hash TEXT,                    -- hash of message for dedup
    subject TEXT,
    body_preview TEXT,                    -- first 500 chars of body

    -- Status tracking
    status TEXT NOT NULL DEFAULT 'draft',  -- draft, sent, opened, replied, bounced, unsubscribed
    sent_at TIMESTAMPTZ,
    delivered_at TIMESTAMPTZ,
    opened_at TIMESTAMPTZ,
    replied_at TIMESTAMPTZ,
    bounced_at TIMESTAMPTZ,

    -- Response data
    sentiment TEXT,                        -- 'positive', 'negative', 'neutral'
    reply_intent TEXT,                     -- 'meeting_request', 'objection', 'question', 'unsubscribe', 'not_interested'
    reply_text TEXT,

    -- Metadata
    sending_provider_id TEXT,             -- SendGrid/Mailgun message ID
    vertical_id TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_outreach_vertical ON outreach_events(vertical_id);
CREATE INDEX idx_outreach_contact ON outreach_events(contact_id);
CREATE INDEX idx_outreach_status ON outreach_events(status);
CREATE INDEX idx_outreach_sent_at ON outreach_events(sent_at);
CREATE INDEX idx_outreach_template ON outreach_events(template_id);

-- ============================================================================
-- OPPORTUNITIES (Sales Pipeline)
-- ============================================================================
CREATE TABLE opportunities (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    company_id UUID REFERENCES companies(id) ON DELETE CASCADE,
    contact_id UUID REFERENCES contacts(id) ON DELETE SET NULL,
    stage TEXT NOT NULL DEFAULT 'prospect',  -- prospect, qualified, proposal, negotiation, closed_won, closed_lost
    value_cents INTEGER,
    currency TEXT DEFAULT 'USD',
    notes TEXT,
    lost_reason TEXT,
    won_at TIMESTAMPTZ,
    lost_at TIMESTAMPTZ,
    vertical_id TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_opportunities_vertical ON opportunities(vertical_id);
CREATE INDEX idx_opportunities_stage ON opportunities(stage);
CREATE INDEX idx_opportunities_company ON opportunities(company_id);

-- ============================================================================
-- KNOWLEDGE CHUNKS (RAG Vector Store)
-- ============================================================================
CREATE TABLE knowledge_chunks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    content TEXT NOT NULL,
    embedding VECTOR(1536),               -- OpenAI text-embedding-3-small or equivalent
    chunk_type TEXT NOT NULL,              -- company_intel, outreach_result, winning_pattern, etc.
    metadata JSONB DEFAULT '{}',
    source_id TEXT,                        -- optional reference to source entity
    source_type TEXT,                      -- 'company', 'contact', 'outreach_event', 'manual'
    vertical_id TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- HNSW index for fast approximate nearest neighbor search
CREATE INDEX idx_knowledge_embedding ON knowledge_chunks
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX idx_knowledge_vertical ON knowledge_chunks(vertical_id);
CREATE INDEX idx_knowledge_type ON knowledge_chunks(chunk_type);
CREATE INDEX idx_knowledge_source ON knowledge_chunks(source_type, source_id);

-- ============================================================================
-- SUPPRESSION LIST (Do-not-contact)
-- ============================================================================
CREATE TABLE suppression_list (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email TEXT NOT NULL,
    reason TEXT,                           -- 'unsubscribed', 'bounced', 'complained', 'manual'
    vertical_id TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(email, vertical_id)
);

CREATE INDEX idx_suppression_email ON suppression_list(email);
CREATE INDEX idx_suppression_vertical ON suppression_list(vertical_id);

-- ============================================================================
-- PIPELINE RUNS (Audit Log)
-- ============================================================================
CREATE TABLE pipeline_runs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    lead_id TEXT,                          -- contact or company being processed
    node_name TEXT NOT NULL,
    status TEXT NOT NULL,                  -- 'started', 'completed', 'failed', 'skipped'
    input_state JSONB DEFAULT '{}',
    output_state JSONB DEFAULT '{}',
    error_message TEXT,
    duration_ms INTEGER,
    vertical_id TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_pipeline_vertical ON pipeline_runs(vertical_id);
CREATE INDEX idx_pipeline_lead ON pipeline_runs(lead_id);
CREATE INDEX idx_pipeline_node ON pipeline_runs(node_name);
CREATE INDEX idx_pipeline_status ON pipeline_runs(status);
CREATE INDEX idx_pipeline_created ON pipeline_runs(created_at);

-- ============================================================================
-- UPDATED_AT TRIGGER
-- ============================================================================
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER companies_updated_at
    BEFORE UPDATE ON companies
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER contacts_updated_at
    BEFORE UPDATE ON contacts
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER outreach_events_updated_at
    BEFORE UPDATE ON outreach_events
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER opportunities_updated_at
    BEFORE UPDATE ON opportunities
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER outreach_templates_updated_at
    BEFORE UPDATE ON outreach_templates
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER knowledge_chunks_updated_at
    BEFORE UPDATE ON knowledge_chunks
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- ============================================================================
-- ROW LEVEL SECURITY (Vertical Isolation)
-- ============================================================================
-- These policies enforce data isolation between verticals.
-- The application sets the current vertical via:
--   SET app.current_vertical = 'enclave_guard';
-- before executing queries.

ALTER TABLE companies ENABLE ROW LEVEL SECURITY;
ALTER TABLE contacts ENABLE ROW LEVEL SECURITY;
ALTER TABLE outreach_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE opportunities ENABLE ROW LEVEL SECURITY;
ALTER TABLE outreach_templates ENABLE ROW LEVEL SECURITY;
ALTER TABLE knowledge_chunks ENABLE ROW LEVEL SECURITY;
ALTER TABLE suppression_list ENABLE ROW LEVEL SECURITY;
ALTER TABLE pipeline_runs ENABLE ROW LEVEL SECURITY;

-- Service role bypasses RLS (used by the backend)
-- For the application role, create policies:
CREATE POLICY vertical_isolation_companies ON companies
    USING (vertical_id = current_setting('app.current_vertical', true));

CREATE POLICY vertical_isolation_contacts ON contacts
    USING (vertical_id = current_setting('app.current_vertical', true));

CREATE POLICY vertical_isolation_outreach ON outreach_events
    USING (vertical_id = current_setting('app.current_vertical', true));

CREATE POLICY vertical_isolation_opportunities ON opportunities
    USING (vertical_id = current_setting('app.current_vertical', true));

CREATE POLICY vertical_isolation_templates ON outreach_templates
    USING (vertical_id = current_setting('app.current_vertical', true));

CREATE POLICY vertical_isolation_knowledge ON knowledge_chunks
    USING (vertical_id = current_setting('app.current_vertical', true));

CREATE POLICY vertical_isolation_suppression ON suppression_list
    USING (vertical_id = current_setting('app.current_vertical', true));

CREATE POLICY vertical_isolation_pipeline ON pipeline_runs
    USING (vertical_id = current_setting('app.current_vertical', true));
