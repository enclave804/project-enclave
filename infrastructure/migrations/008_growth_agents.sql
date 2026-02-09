-- ============================================================
-- Migration 008: Growth Agents — Phase 11
-- ============================================================
-- Adds tables for Proposal Builder, Social Media, and Ads
-- Strategy agents. These tables track proposals, social posts,
-- ad campaigns, and their performance metrics.
-- ============================================================

-- ─── Proposals ──────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS proposals (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    vertical_id     TEXT NOT NULL,
    agent_id        TEXT NOT NULL DEFAULT 'proposal_builder',

    -- Deal context
    company_name    TEXT NOT NULL,
    company_domain  TEXT DEFAULT '',
    contact_name    TEXT NOT NULL DEFAULT '',
    contact_email   TEXT NOT NULL DEFAULT '',
    contact_title   TEXT DEFAULT '',

    -- Proposal details
    proposal_type   TEXT NOT NULL DEFAULT 'full_proposal'
        CHECK (proposal_type IN ('sow', 'one_pager', 'executive_summary', 'full_proposal')),
    pricing_tier    TEXT NOT NULL DEFAULT 'professional'
        CHECK (pricing_tier IN ('starter', 'professional', 'enterprise', 'custom')),
    pricing_amount  NUMERIC(12,2) DEFAULT 0,
    timeline_weeks  INTEGER DEFAULT 0,
    deliverables    JSONB DEFAULT '[]'::JSONB,

    -- Content
    proposal_markdown TEXT DEFAULT '',
    sections          JSONB DEFAULT '[]'::JSONB,

    -- Status
    status          TEXT NOT NULL DEFAULT 'draft'
        CHECK (status IN ('draft', 'review', 'approved', 'delivered', 'accepted', 'rejected')),
    delivered_at    TIMESTAMPTZ,
    delivered_via   TEXT DEFAULT 'email',

    -- Metadata
    meeting_notes   TEXT DEFAULT '',
    pain_points     JSONB DEFAULT '[]'::JSONB,
    metadata        JSONB DEFAULT '{}'::JSONB,

    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_proposals_vertical
    ON proposals (vertical_id);
CREATE INDEX IF NOT EXISTS idx_proposals_company
    ON proposals (company_domain);
CREATE INDEX IF NOT EXISTS idx_proposals_status
    ON proposals (status);

-- ─── Social Media Posts ─────────────────────────────────────

CREATE TABLE IF NOT EXISTS social_posts (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    vertical_id     TEXT NOT NULL,
    agent_id        TEXT NOT NULL DEFAULT 'social',

    -- Post content
    platform        TEXT NOT NULL CHECK (platform IN ('twitter', 'x', 'linkedin', 'instagram')),
    content         TEXT NOT NULL DEFAULT '',
    post_type       TEXT NOT NULL DEFAULT 'thought_leadership',
    hashtags        JSONB DEFAULT '[]'::JSONB,
    media_url       TEXT DEFAULT '',

    -- External IDs (from platform API)
    external_post_id TEXT DEFAULT '',

    -- Status
    status          TEXT NOT NULL DEFAULT 'draft'
        CHECK (status IN ('draft', 'review', 'approved', 'scheduled', 'published', 'failed')),
    scheduled_at    TIMESTAMPTZ,
    published_at    TIMESTAMPTZ,

    -- Engagement metrics (updated after publishing)
    impressions     INTEGER DEFAULT 0,
    likes           INTEGER DEFAULT 0,
    shares          INTEGER DEFAULT 0,
    comments        INTEGER DEFAULT 0,
    clicks          INTEGER DEFAULT 0,

    -- Metadata
    topic           TEXT DEFAULT '',
    was_edited      BOOLEAN DEFAULT false,
    metadata        JSONB DEFAULT '{}'::JSONB,

    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_social_posts_vertical
    ON social_posts (vertical_id);
CREATE INDEX IF NOT EXISTS idx_social_posts_platform
    ON social_posts (platform, status);
CREATE INDEX IF NOT EXISTS idx_social_posts_published
    ON social_posts (published_at DESC);

-- ─── Ad Campaigns ───────────────────────────────────────────

CREATE TABLE IF NOT EXISTS ad_campaigns (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    vertical_id     TEXT NOT NULL,
    agent_id        TEXT NOT NULL DEFAULT 'ads_strategy',

    -- Campaign details
    platform        TEXT NOT NULL CHECK (platform IN ('google', 'meta', 'linkedin')),
    campaign_name   TEXT NOT NULL DEFAULT '',
    objective       TEXT NOT NULL DEFAULT 'lead_gen',
    budget_daily    NUMERIC(10,2) DEFAULT 0,
    budget_total    NUMERIC(12,2) DEFAULT 0,
    target_cpa      NUMERIC(10,2) DEFAULT 0,

    -- Targeting
    target_audience JSONB DEFAULT '{}'::JSONB,
    keywords        JSONB DEFAULT '[]'::JSONB,
    negative_keywords JSONB DEFAULT '[]'::JSONB,

    -- Ad groups and creatives
    ad_groups       JSONB DEFAULT '[]'::JSONB,

    -- Status
    status          TEXT NOT NULL DEFAULT 'draft'
        CHECK (status IN ('draft', 'review', 'approved', 'active', 'paused', 'completed')),

    -- Performance metrics
    impressions     BIGINT DEFAULT 0,
    clicks          BIGINT DEFAULT 0,
    conversions     INTEGER DEFAULT 0,
    spend           NUMERIC(12,2) DEFAULT 0,
    ctr             NUMERIC(6,4) DEFAULT 0,
    cpa             NUMERIC(10,2) DEFAULT 0,
    roas            NUMERIC(8,2) DEFAULT 0,

    -- Metadata
    metadata        JSONB DEFAULT '{}'::JSONB,

    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_ad_campaigns_vertical
    ON ad_campaigns (vertical_id);
CREATE INDEX IF NOT EXISTS idx_ad_campaigns_platform
    ON ad_campaigns (platform, status);

-- ─── Content Calendar ───────────────────────────────────────

CREATE TABLE IF NOT EXISTS content_calendar (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    vertical_id     TEXT NOT NULL,
    agent_id        TEXT NOT NULL,

    -- Schedule
    scheduled_date  DATE NOT NULL,
    platform        TEXT NOT NULL,
    post_type       TEXT NOT NULL DEFAULT 'thought_leadership',
    topic           TEXT NOT NULL DEFAULT '',

    -- Status
    status          TEXT NOT NULL DEFAULT 'planned'
        CHECK (status IN ('planned', 'drafted', 'approved', 'published', 'skipped')),

    -- References
    social_post_id  UUID REFERENCES social_posts(id) ON DELETE SET NULL,
    metadata        JSONB DEFAULT '{}'::JSONB,

    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_content_calendar_vertical
    ON content_calendar (vertical_id, scheduled_date);

-- ─── RPC: Get Growth Agent Stats ────────────────────────────

CREATE OR REPLACE FUNCTION get_growth_stats(
    p_vertical_id TEXT,
    p_days INTEGER DEFAULT 30
)
RETURNS JSONB
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
DECLARE
    result JSONB;
BEGIN
    SELECT jsonb_build_object(
        'proposals', (
            SELECT jsonb_build_object(
                'total', COUNT(*),
                'delivered', COUNT(*) FILTER (WHERE status = 'delivered'),
                'accepted', COUNT(*) FILTER (WHERE status = 'accepted'),
                'total_value', COALESCE(SUM(pricing_amount) FILTER (WHERE status IN ('delivered', 'accepted')), 0)
            )
            FROM proposals
            WHERE vertical_id = p_vertical_id
              AND created_at >= now() - (p_days || ' days')::INTERVAL
        ),
        'social_posts', (
            SELECT jsonb_build_object(
                'total', COUNT(*),
                'published', COUNT(*) FILTER (WHERE status = 'published'),
                'total_impressions', COALESCE(SUM(impressions), 0),
                'total_engagement', COALESCE(SUM(likes + shares + comments), 0)
            )
            FROM social_posts
            WHERE vertical_id = p_vertical_id
              AND created_at >= now() - (p_days || ' days')::INTERVAL
        ),
        'ad_campaigns', (
            SELECT jsonb_build_object(
                'total', COUNT(*),
                'active', COUNT(*) FILTER (WHERE status = 'active'),
                'total_spend', COALESCE(SUM(spend), 0),
                'total_conversions', COALESCE(SUM(conversions), 0)
            )
            FROM ad_campaigns
            WHERE vertical_id = p_vertical_id
              AND created_at >= now() - (p_days || ' days')::INTERVAL
        )
    ) INTO result;

    RETURN result;
END;
$$;
