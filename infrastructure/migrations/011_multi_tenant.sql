-- ============================================================
-- Migration 011: Enterprise & Multi-Tenancy — Phase 14
-- ============================================================
-- Transforms the platform from single-tenant to multi-tenant
-- SaaS with organization-based isolation, API keys, and
-- role-based access control.
--
-- Adds: organizations, org_members, api_keys tables
-- Modifies: adds org_id column to 7 high-value tables
-- Seeds: default organization for backward compatibility
-- ============================================================

-- ─── Organizations ──────────────────────────────────────────

CREATE TABLE IF NOT EXISTS organizations (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name            TEXT NOT NULL,
    slug            TEXT UNIQUE NOT NULL,
    plan_tier       TEXT NOT NULL DEFAULT 'free'
                    CHECK (plan_tier IN ('free', 'starter', 'pro', 'enterprise')),
    settings        JSONB DEFAULT '{}',

    created_at      TIMESTAMPTZ DEFAULT now(),
    updated_at      TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_organizations_slug
    ON organizations(slug);
CREATE INDEX IF NOT EXISTS idx_organizations_plan_tier
    ON organizations(plan_tier);

-- ─── Organization Members ───────────────────────────────────

CREATE TABLE IF NOT EXISTS org_members (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id          UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    user_id         UUID NOT NULL,
    email           TEXT NOT NULL DEFAULT '',
    role            TEXT NOT NULL DEFAULT 'viewer'
                    CHECK (role IN ('owner', 'admin', 'editor', 'viewer')),
    invited_by      UUID,

    created_at      TIMESTAMPTZ DEFAULT now(),

    UNIQUE(org_id, user_id)
);

CREATE INDEX IF NOT EXISTS idx_org_members_org
    ON org_members(org_id);
CREATE INDEX IF NOT EXISTS idx_org_members_user
    ON org_members(user_id);
CREATE INDEX IF NOT EXISTS idx_org_members_email
    ON org_members(email);

-- ─── API Keys ───────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS api_keys (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id          UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    key_hash        TEXT NOT NULL,
    key_prefix      TEXT NOT NULL,
    name            TEXT NOT NULL DEFAULT 'Default Key',
    scopes          JSONB DEFAULT '["read"]',
    rate_limit_per_minute INT DEFAULT 60,
    expires_at      TIMESTAMPTZ,
    last_used_at    TIMESTAMPTZ,
    is_active       BOOLEAN DEFAULT true,

    created_at      TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_api_keys_org
    ON api_keys(org_id);
CREATE INDEX IF NOT EXISTS idx_api_keys_hash
    ON api_keys(key_hash);
CREATE INDEX IF NOT EXISTS idx_api_keys_prefix
    ON api_keys(key_prefix);
CREATE INDEX IF NOT EXISTS idx_api_keys_active
    ON api_keys(is_active);

-- ─── Add org_id to Existing Tables ──────────────────────────

-- Companies
ALTER TABLE companies
    ADD COLUMN IF NOT EXISTS org_id UUID REFERENCES organizations(id);
CREATE INDEX IF NOT EXISTS idx_companies_org
    ON companies(org_id);

-- Contacts
ALTER TABLE contacts
    ADD COLUMN IF NOT EXISTS org_id UUID REFERENCES organizations(id);
CREATE INDEX IF NOT EXISTS idx_contacts_org
    ON contacts(org_id);

-- Pipeline Runs
ALTER TABLE pipeline_runs
    ADD COLUMN IF NOT EXISTS org_id UUID REFERENCES organizations(id);
CREATE INDEX IF NOT EXISTS idx_pipeline_runs_org
    ON pipeline_runs(org_id);

-- Shared Insights
ALTER TABLE shared_insights
    ADD COLUMN IF NOT EXISTS org_id UUID REFERENCES organizations(id);
CREATE INDEX IF NOT EXISTS idx_shared_insights_org
    ON shared_insights(org_id);

-- Agent Content
ALTER TABLE agent_content
    ADD COLUMN IF NOT EXISTS org_id UUID REFERENCES organizations(id);
CREATE INDEX IF NOT EXISTS idx_agent_content_org
    ON agent_content(org_id);

-- Experiments
ALTER TABLE experiments
    ADD COLUMN IF NOT EXISTS org_id UUID REFERENCES organizations(id);
CREATE INDEX IF NOT EXISTS idx_experiments_org
    ON experiments(org_id);

-- Lead Scores
ALTER TABLE lead_scores
    ADD COLUMN IF NOT EXISTS org_id UUID REFERENCES organizations(id);
CREATE INDEX IF NOT EXISTS idx_lead_scores_org
    ON lead_scores(org_id);

-- ─── Seed Default Organization ──────────────────────────────

INSERT INTO organizations (name, slug, plan_tier, settings)
VALUES ('Default Organization', 'default', 'enterprise', '{"is_default": true}')
ON CONFLICT (slug) DO NOTHING;

-- ─── Backfill org_id on Existing Rows ───────────────────────

UPDATE companies
SET org_id = (SELECT id FROM organizations WHERE slug = 'default')
WHERE org_id IS NULL;

UPDATE contacts
SET org_id = (SELECT id FROM organizations WHERE slug = 'default')
WHERE org_id IS NULL;

UPDATE pipeline_runs
SET org_id = (SELECT id FROM organizations WHERE slug = 'default')
WHERE org_id IS NULL;

UPDATE shared_insights
SET org_id = (SELECT id FROM organizations WHERE slug = 'default')
WHERE org_id IS NULL;

UPDATE agent_content
SET org_id = (SELECT id FROM organizations WHERE slug = 'default')
WHERE org_id IS NULL;

UPDATE experiments
SET org_id = (SELECT id FROM organizations WHERE slug = 'default')
WHERE org_id IS NULL;

UPDATE lead_scores
SET org_id = (SELECT id FROM organizations WHERE slug = 'default')
WHERE org_id IS NULL;

-- ─── RLS Policies for New Tables ────────────────────────────

-- Organizations: members can read their own orgs
ALTER TABLE organizations ENABLE ROW LEVEL SECURITY;

CREATE POLICY org_read_policy ON organizations
    FOR SELECT
    USING (
        id IN (
            SELECT org_id FROM org_members
            WHERE user_id = auth.uid()
        )
    );

CREATE POLICY org_manage_policy ON organizations
    FOR ALL
    USING (
        id IN (
            SELECT org_id FROM org_members
            WHERE user_id = auth.uid()
            AND role IN ('owner', 'admin')
        )
    );

-- Org Members: visible to members of the same org
ALTER TABLE org_members ENABLE ROW LEVEL SECURITY;

CREATE POLICY org_members_read_policy ON org_members
    FOR SELECT
    USING (
        org_id IN (
            SELECT org_id FROM org_members AS om
            WHERE om.user_id = auth.uid()
        )
    );

CREATE POLICY org_members_manage_policy ON org_members
    FOR ALL
    USING (
        org_id IN (
            SELECT org_id FROM org_members AS om
            WHERE om.user_id = auth.uid()
            AND om.role IN ('owner', 'admin')
        )
    );

-- API Keys: visible to admins of the org
ALTER TABLE api_keys ENABLE ROW LEVEL SECURITY;

CREATE POLICY api_keys_read_policy ON api_keys
    FOR SELECT
    USING (
        org_id IN (
            SELECT org_id FROM org_members
            WHERE user_id = auth.uid()
            AND role IN ('owner', 'admin')
        )
    );

CREATE POLICY api_keys_manage_policy ON api_keys
    FOR ALL
    USING (
        org_id IN (
            SELECT org_id FROM org_members
            WHERE user_id = auth.uid()
            AND role IN ('owner', 'admin')
        )
    );

-- ─── RPC: Get Organization Stats ────────────────────────────

CREATE OR REPLACE FUNCTION get_org_stats(
    p_org_id UUID
) RETURNS JSONB AS $$
DECLARE
    result JSONB;
BEGIN
    SELECT jsonb_build_object(
        'total_members', (
            SELECT COUNT(*)
            FROM org_members
            WHERE org_id = p_org_id
        ),
        'total_api_keys', (
            SELECT COUNT(*)
            FROM api_keys
            WHERE org_id = p_org_id AND is_active = true
        ),
        'total_verticals', (
            SELECT COUNT(DISTINCT vertical_id)
            FROM companies
            WHERE org_id = p_org_id
        ),
        'total_companies', (
            SELECT COUNT(*)
            FROM companies
            WHERE org_id = p_org_id
        ),
        'total_insights', (
            SELECT COUNT(*)
            FROM shared_insights
            WHERE org_id = p_org_id
        )
    ) INTO result;

    RETURN result;
END;
$$ LANGUAGE plpgsql;
