-- ============================================================
-- Migration 018: Universal + Creative Agent Tables
-- Phase 22 — Universal Business + Creative Production Agents
-- ============================================================

-- projects: project tasks, milestones, status tracking
CREATE TABLE IF NOT EXISTS projects (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    vertical_id TEXT NOT NULL,
    company_id UUID REFERENCES companies(id) ON DELETE SET NULL,
    contact_id UUID REFERENCES contacts(id) ON DELETE SET NULL,
    project_name TEXT NOT NULL DEFAULT '',
    description TEXT DEFAULT '',
    assigned_to TEXT DEFAULT '',
    priority TEXT DEFAULT 'medium'
        CHECK (priority IN ('critical','high','medium','low')),
    status TEXT NOT NULL DEFAULT 'not_started'
        CHECK (status IN ('not_started','in_progress','blocked','completed','cancelled')),
    tasks JSONB DEFAULT '[]'::JSONB,
    total_tasks INTEGER DEFAULT 0,
    completed_tasks INTEGER DEFAULT 0,
    milestones JSONB DEFAULT '[]'::JSONB,
    completion_percent FLOAT DEFAULT 0.0,
    due_date DATE,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    tags JSONB DEFAULT '[]'::JSONB,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- project_plans: strategic plans, timelines, risk assessments
CREATE TABLE IF NOT EXISTS project_plans (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    vertical_id TEXT NOT NULL,
    project_id UUID REFERENCES projects(id) ON DELETE SET NULL,
    plan_name TEXT NOT NULL DEFAULT '',
    scope TEXT DEFAULT '',
    objectives JSONB DEFAULT '[]'::JSONB,
    timeline JSONB DEFAULT '[]'::JSONB,
    risks JSONB DEFAULT '[]'::JSONB,
    risk_count INTEGER DEFAULT 0,
    high_risks INTEGER DEFAULT 0,
    resources JSONB DEFAULT '[]'::JSONB,
    budget_estimate_cents INTEGER DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'draft'
        CHECK (status IN ('draft','active','on_hold','completed','archived')),
    approved_at TIMESTAMPTZ,
    approved_by TEXT DEFAULT '',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- brand_mentions: social/web mentions, sentiment tracking
CREATE TABLE IF NOT EXISTS brand_mentions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    vertical_id TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT '',
    platform TEXT DEFAULT ''
        CHECK (platform IN ('twitter','linkedin','reddit','news','blog','forum','review','other','')),
    author TEXT DEFAULT '',
    content_snippet TEXT DEFAULT '',
    mention_url TEXT DEFAULT '',
    sentiment_score FLOAT DEFAULT 0.0,
    sentiment_label TEXT DEFAULT 'neutral'
        CHECK (sentiment_label IN ('positive','negative','neutral','mixed')),
    is_competitor_mention BOOLEAN DEFAULT false,
    competitor_name TEXT DEFAULT '',
    alert_triggered BOOLEAN DEFAULT false,
    status TEXT NOT NULL DEFAULT 'new'
        CHECK (status IN ('new','analyzed','actioned','dismissed')),
    detected_at TIMESTAMPTZ DEFAULT now(),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- threat_intelligence: CVEs, IOCs, security advisories
CREATE TABLE IF NOT EXISTS threat_intelligence (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    vertical_id TEXT NOT NULL,
    threat_type TEXT NOT NULL DEFAULT 'vulnerability'
        CHECK (threat_type IN ('vulnerability','malware','phishing','data_breach','insider_threat','apt','ddos','ransomware','other')),
    severity TEXT DEFAULT 'medium'
        CHECK (severity IN ('critical','high','medium','low','informational')),
    cve_id TEXT DEFAULT '',
    cvss_score FLOAT DEFAULT 0.0,
    ioc_type TEXT DEFAULT ''
        CHECK (ioc_type IN ('ip','domain','hash','url','email','registry','file_path','')),
    ioc_value TEXT DEFAULT '',
    source_feed TEXT DEFAULT '',
    advisory_title TEXT DEFAULT '',
    advisory_text TEXT DEFAULT '',
    affected_systems JSONB DEFAULT '[]'::JSONB,
    mitigation_steps JSONB DEFAULT '[]'::JSONB,
    status TEXT NOT NULL DEFAULT 'new'
        CHECK (status IN ('new','investigating','confirmed','mitigated','false_positive')),
    risk_score FLOAT DEFAULT 0.0,
    mitigated_at TIMESTAMPTZ,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- web_designs: website design briefs, generated HTML/CSS
CREATE TABLE IF NOT EXISTS web_designs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    vertical_id TEXT NOT NULL,
    company_id UUID REFERENCES companies(id) ON DELETE SET NULL,
    project_name TEXT NOT NULL DEFAULT '',
    site_type TEXT DEFAULT 'landing_page'
        CHECK (site_type IN ('landing_page','portfolio','blog','ecommerce','corporate','documentation','custom')),
    design_brief JSONB DEFAULT '{}',
    brand_colors JSONB DEFAULT '[]'::JSONB,
    typography JSONB DEFAULT '{}',
    pages JSONB DEFAULT '[]'::JSONB,
    total_pages INTEGER DEFAULT 0,
    generated_pages INTEGER DEFAULT 0,
    html_output_path TEXT DEFAULT '',
    css_output_path TEXT DEFAULT '',
    preview_url TEXT DEFAULT '',
    responsive BOOLEAN DEFAULT true,
    framework TEXT DEFAULT 'html5',
    status TEXT NOT NULL DEFAULT 'brief'
        CHECK (status IN ('brief','wireframe','design','development','review','published')),
    feedback JSONB DEFAULT '[]'::JSONB,
    approved_at TIMESTAMPTZ,
    published_at TIMESTAMPTZ,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- graphic_designs: graphic asset briefs, generated images
CREATE TABLE IF NOT EXISTS graphic_designs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    vertical_id TEXT NOT NULL,
    company_id UUID REFERENCES companies(id) ON DELETE SET NULL,
    project_name TEXT NOT NULL DEFAULT '',
    asset_type TEXT DEFAULT 'logo'
        CHECK (asset_type IN ('logo','banner','social_post','infographic','icon','illustration','brand_kit','flyer','business_card','custom')),
    design_brief JSONB DEFAULT '{}',
    brand_colors JSONB DEFAULT '[]'::JSONB,
    style_guide JSONB DEFAULT '{}',
    dimensions TEXT DEFAULT '',
    text_content TEXT DEFAULT '',
    assets_generated JSONB DEFAULT '[]'::JSONB,
    total_assets INTEGER DEFAULT 0,
    image_paths JSONB DEFAULT '[]'::JSONB,
    variations JSONB DEFAULT '[]'::JSONB,
    export_formats JSONB DEFAULT '["png"]'::JSONB,
    status TEXT NOT NULL DEFAULT 'brief'
        CHECK (status IN ('brief','concept','design','revision','approved','delivered')),
    feedback JSONB DEFAULT '[]'::JSONB,
    approved_at TIMESTAMPTZ,
    delivered_at TIMESTAMPTZ,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_projects_vertical_status ON projects(vertical_id, status);
CREATE INDEX IF NOT EXISTS idx_projects_priority ON projects(vertical_id, priority);
CREATE INDEX IF NOT EXISTS idx_project_plans_vertical_status ON project_plans(vertical_id, status);
CREATE INDEX IF NOT EXISTS idx_brand_mentions_vertical_status ON brand_mentions(vertical_id, status);
CREATE INDEX IF NOT EXISTS idx_brand_mentions_platform ON brand_mentions(vertical_id, platform);
CREATE INDEX IF NOT EXISTS idx_threat_intelligence_vertical_status ON threat_intelligence(vertical_id, status);
CREATE INDEX IF NOT EXISTS idx_threat_intelligence_severity ON threat_intelligence(vertical_id, severity);
CREATE INDEX IF NOT EXISTS idx_threat_intelligence_cve ON threat_intelligence(cve_id) WHERE cve_id != '';
CREATE INDEX IF NOT EXISTS idx_web_designs_vertical_status ON web_designs(vertical_id, status);
CREATE INDEX IF NOT EXISTS idx_graphic_designs_vertical_status ON graphic_designs(vertical_id, status);

-- RPC: Get universal + creative agent metrics
CREATE OR REPLACE FUNCTION get_universal_creative_metrics(p_vertical_id TEXT)
RETURNS JSONB LANGUAGE plpgsql SECURITY DEFINER AS $$
DECLARE
    result JSONB;
BEGIN
    SELECT jsonb_build_object(
        'projects', (SELECT jsonb_build_object(
            'total', COUNT(*),
            'in_progress', COUNT(*) FILTER (WHERE status = 'in_progress'),
            'blocked', COUNT(*) FILTER (WHERE status = 'blocked'),
            'completed', COUNT(*) FILTER (WHERE status = 'completed')
        ) FROM projects WHERE vertical_id = p_vertical_id),
        'project_plans', (SELECT jsonb_build_object(
            'total', COUNT(*),
            'active', COUNT(*) FILTER (WHERE status = 'active'),
            'draft', COUNT(*) FILTER (WHERE status = 'draft')
        ) FROM project_plans WHERE vertical_id = p_vertical_id),
        'brand_mentions', (SELECT jsonb_build_object(
            'total', COUNT(*),
            'new', COUNT(*) FILTER (WHERE status = 'new'),
            'avg_sentiment', COALESCE(AVG(sentiment_score), 0)
        ) FROM brand_mentions WHERE vertical_id = p_vertical_id),
        'threat_intelligence', (SELECT jsonb_build_object(
            'total', COUNT(*),
            'critical', COUNT(*) FILTER (WHERE severity = 'critical'),
            'investigating', COUNT(*) FILTER (WHERE status = 'investigating')
        ) FROM threat_intelligence WHERE vertical_id = p_vertical_id),
        'web_designs', (SELECT jsonb_build_object(
            'total', COUNT(*),
            'published', COUNT(*) FILTER (WHERE status = 'published'),
            'in_progress', COUNT(*) FILTER (WHERE status IN ('design','development','review'))
        ) FROM web_designs WHERE vertical_id = p_vertical_id),
        'graphic_designs', (SELECT jsonb_build_object(
            'total', COUNT(*),
            'delivered', COUNT(*) FILTER (WHERE status = 'delivered'),
            'in_progress', COUNT(*) FILTER (WHERE status IN ('concept','design','revision'))
        ) FROM graphic_designs WHERE vertical_id = p_vertical_id)
    ) INTO result;
    RETURN result;
END;
$$;
