-- ============================================================
-- Migration 017: Universal Business Agent Tables v2
-- Phase 21 — Universal Business Agents v2
-- ============================================================

-- client_onboarding: client onboarding workflows and milestones
CREATE TABLE IF NOT EXISTS client_onboarding (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    vertical_id TEXT NOT NULL,
    company_id UUID REFERENCES companies(id) ON DELETE SET NULL,
    contact_id UUID REFERENCES contacts(id) ON DELETE SET NULL,
    opportunity_id UUID REFERENCES opportunities(id) ON DELETE SET NULL,
    company_name TEXT NOT NULL DEFAULT '',
    contact_name TEXT DEFAULT '',
    contact_email TEXT DEFAULT '',
    template_name TEXT DEFAULT 'default',
    milestones JSONB DEFAULT '[]'::JSONB,
    total_milestones INTEGER DEFAULT 0,
    completed_milestones INTEGER DEFAULT 0,
    completion_percent FLOAT DEFAULT 0.0,
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending','in_progress','completed','stalled','cancelled')),
    kickoff_scheduled_at TIMESTAMPTZ,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    welcome_package_sent BOOLEAN DEFAULT false,
    welcome_package_content TEXT DEFAULT '',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- invoices: billing and payment tracking
CREATE TABLE IF NOT EXISTS invoices (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    vertical_id TEXT NOT NULL,
    company_id UUID REFERENCES companies(id) ON DELETE SET NULL,
    contact_id UUID REFERENCES contacts(id) ON DELETE SET NULL,
    opportunity_id UUID REFERENCES opportunities(id) ON DELETE SET NULL,
    invoice_number TEXT NOT NULL DEFAULT '',
    proposal_id UUID,
    line_items JSONB DEFAULT '[]'::JSONB,
    subtotal_cents INTEGER DEFAULT 0,
    tax_cents INTEGER DEFAULT 0,
    total_cents INTEGER DEFAULT 0,
    currency TEXT DEFAULT 'usd',
    status TEXT NOT NULL DEFAULT 'draft'
        CHECK (status IN ('draft','sent','viewed','paid','overdue','void','refunded')),
    due_date DATE,
    paid_at TIMESTAMPTZ,
    payment_method TEXT DEFAULT '',
    stripe_invoice_id TEXT DEFAULT '',
    stripe_hosted_url TEXT DEFAULT '',
    reminder_count INTEGER DEFAULT 0,
    last_reminder_at TIMESTAMPTZ,
    reminder_tone TEXT DEFAULT 'polite'
        CHECK (reminder_tone IN ('polite','firm','final')),
    notes TEXT DEFAULT '',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- knowledge_articles: knowledge base content
CREATE TABLE IF NOT EXISTS knowledge_articles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    vertical_id TEXT NOT NULL,
    title TEXT NOT NULL DEFAULT '',
    slug TEXT DEFAULT '',
    body_markdown TEXT DEFAULT '',
    category TEXT DEFAULT 'general',
    tags JSONB DEFAULT '[]'::JSONB,
    source_type TEXT DEFAULT 'manual'
        CHECK (source_type IN ('manual','support_ticket','agent_learning','faq_auto','documentation')),
    source_ticket_id UUID,
    source_agent_id TEXT DEFAULT '',
    helpful_votes INTEGER DEFAULT 0,
    unhelpful_votes INTEGER DEFAULT 0,
    view_count INTEGER DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'draft'
        CHECK (status IN ('draft','review','published','archived')),
    published_at TIMESTAMPTZ,
    last_reviewed_at TIMESTAMPTZ,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- feedback_responses: NPS, CSAT, CES survey responses
CREATE TABLE IF NOT EXISTS feedback_responses (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    vertical_id TEXT NOT NULL,
    company_id UUID REFERENCES companies(id) ON DELETE SET NULL,
    contact_id UUID REFERENCES contacts(id) ON DELETE SET NULL,
    survey_type TEXT NOT NULL DEFAULT 'nps'
        CHECK (survey_type IN ('nps','csat','ces','custom')),
    touchpoint TEXT DEFAULT 'post_project',
    nps_score INTEGER CHECK (nps_score IS NULL OR (nps_score >= 0 AND nps_score <= 10)),
    csat_score INTEGER CHECK (csat_score IS NULL OR (csat_score >= 1 AND csat_score <= 5)),
    ces_score INTEGER CHECK (ces_score IS NULL OR (ces_score >= 1 AND ces_score <= 7)),
    comment TEXT DEFAULT '',
    sentiment TEXT DEFAULT 'neutral'
        CHECK (sentiment IN ('positive','neutral','negative')),
    sentiment_score FLOAT DEFAULT 0.0,
    contact_email TEXT DEFAULT '',
    contact_name TEXT DEFAULT '',
    company_name TEXT DEFAULT '',
    requires_followup BOOLEAN DEFAULT false,
    followup_status TEXT DEFAULT 'none'
        CHECK (followup_status IN ('none','pending','completed','escalated')),
    followup_notes TEXT DEFAULT '',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- referrals: client referral tracking and commissions
CREATE TABLE IF NOT EXISTS referrals (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    vertical_id TEXT NOT NULL,
    referrer_company_id UUID REFERENCES companies(id) ON DELETE SET NULL,
    referrer_contact_id UUID REFERENCES contacts(id) ON DELETE SET NULL,
    referrer_name TEXT DEFAULT '',
    referrer_email TEXT DEFAULT '',
    referrer_company_name TEXT DEFAULT '',
    referee_name TEXT DEFAULT '',
    referee_email TEXT DEFAULT '',
    referee_company_name TEXT DEFAULT '',
    referee_company_domain TEXT DEFAULT '',
    status TEXT NOT NULL DEFAULT 'submitted'
        CHECK (status IN ('submitted','contacted','qualified','converted','lost','expired')),
    deal_value_cents INTEGER DEFAULT 0,
    commission_percent FLOAT DEFAULT 0.0,
    commission_cents INTEGER DEFAULT 0,
    commission_paid BOOLEAN DEFAULT false,
    commission_paid_at TIMESTAMPTZ,
    source TEXT DEFAULT 'client_referral',
    notes TEXT DEFAULT '',
    converted_at TIMESTAMPTZ,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- deal_analyses: win/loss analysis records
CREATE TABLE IF NOT EXISTS deal_analyses (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    vertical_id TEXT NOT NULL,
    opportunity_id UUID REFERENCES opportunities(id) ON DELETE SET NULL,
    company_id UUID REFERENCES companies(id) ON DELETE SET NULL,
    outcome TEXT NOT NULL DEFAULT 'won'
        CHECK (outcome IN ('won','lost')),
    deal_value_cents INTEGER DEFAULT 0,
    deal_stage TEXT DEFAULT '',
    close_date DATE,
    win_loss_factors JSONB DEFAULT '[]'::JSONB,
    competitor_involved TEXT DEFAULT '',
    objections_faced JSONB DEFAULT '[]'::JSONB,
    sales_cycle_days INTEGER DEFAULT 0,
    decision_makers JSONB DEFAULT '[]'::JSONB,
    recommendations JSONB DEFAULT '[]'::JSONB,
    patterns_identified JSONB DEFAULT '[]'::JSONB,
    analyzed_by TEXT DEFAULT '',
    analysis_confidence FLOAT DEFAULT 0.0,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- data_quality_issues: data hygiene and enrichment tracking
CREATE TABLE IF NOT EXISTS data_quality_issues (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    vertical_id TEXT NOT NULL,
    target_table TEXT NOT NULL DEFAULT '',
    target_id UUID,
    target_field TEXT DEFAULT '',
    issue_type TEXT NOT NULL DEFAULT 'missing'
        CHECK (issue_type IN ('missing','invalid_email','duplicate','stale','inconsistent','invalid_phone','incomplete','format_error')),
    severity TEXT DEFAULT 'medium'
        CHECK (severity IN ('low','medium','high','critical')),
    description TEXT DEFAULT '',
    original_value TEXT DEFAULT '',
    suggested_value TEXT DEFAULT '',
    resolved BOOLEAN DEFAULT false,
    resolved_at TIMESTAMPTZ,
    resolution_method TEXT DEFAULT ''
        CHECK (resolution_method IN ('','auto_fix','manual','enrichment','deletion','skip')),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- compliance_records: regulatory compliance and consent tracking
CREATE TABLE IF NOT EXISTS compliance_records (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    vertical_id TEXT NOT NULL,
    company_id UUID REFERENCES companies(id) ON DELETE SET NULL,
    contact_id UUID REFERENCES contacts(id) ON DELETE SET NULL,
    regulation TEXT NOT NULL DEFAULT 'gdpr'
        CHECK (regulation IN ('gdpr','ccpa','can_spam','hipaa','sox','pci_dss','custom')),
    requirement TEXT DEFAULT '',
    record_type TEXT NOT NULL DEFAULT 'consent'
        CHECK (record_type IN ('consent','data_request','retention','breach_notification','audit','policy')),
    contact_email TEXT DEFAULT '',
    consent_given BOOLEAN DEFAULT false,
    consent_type TEXT DEFAULT '',
    consent_timestamp TIMESTAMPTZ,
    consent_method TEXT DEFAULT '',
    retention_expiry DATE,
    data_categories JSONB DEFAULT '[]'::JSONB,
    status TEXT NOT NULL DEFAULT 'active'
        CHECK (status IN ('active','expired','revoked','pending_review','archived')),
    notes TEXT DEFAULT '',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_client_onboarding_vertical_status ON client_onboarding(vertical_id, status);
CREATE INDEX IF NOT EXISTS idx_client_onboarding_company ON client_onboarding(company_id);
CREATE INDEX IF NOT EXISTS idx_client_onboarding_opportunity ON client_onboarding(opportunity_id);

CREATE INDEX IF NOT EXISTS idx_invoices_vertical_status ON invoices(vertical_id, status);
CREATE INDEX IF NOT EXISTS idx_invoices_company ON invoices(company_id);
CREATE INDEX IF NOT EXISTS idx_invoices_due_date ON invoices(due_date) WHERE status NOT IN ('paid', 'void', 'refunded');
CREATE INDEX IF NOT EXISTS idx_invoices_stripe ON invoices(stripe_invoice_id) WHERE stripe_invoice_id != '';

CREATE INDEX IF NOT EXISTS idx_knowledge_articles_vertical_status ON knowledge_articles(vertical_id, status);
CREATE INDEX IF NOT EXISTS idx_knowledge_articles_category ON knowledge_articles(vertical_id, category);
CREATE INDEX IF NOT EXISTS idx_knowledge_articles_slug ON knowledge_articles(slug) WHERE slug != '';

CREATE INDEX IF NOT EXISTS idx_feedback_responses_vertical_type ON feedback_responses(vertical_id, survey_type);
CREATE INDEX IF NOT EXISTS idx_feedback_responses_company ON feedback_responses(company_id);
CREATE INDEX IF NOT EXISTS idx_feedback_responses_followup ON feedback_responses(followup_status) WHERE requires_followup = true;

CREATE INDEX IF NOT EXISTS idx_referrals_vertical_status ON referrals(vertical_id, status);
CREATE INDEX IF NOT EXISTS idx_referrals_referrer ON referrals(referrer_company_id);
CREATE INDEX IF NOT EXISTS idx_referrals_commission ON referrals(commission_paid) WHERE commission_cents > 0;

CREATE INDEX IF NOT EXISTS idx_deal_analyses_vertical_outcome ON deal_analyses(vertical_id, outcome);
CREATE INDEX IF NOT EXISTS idx_deal_analyses_opportunity ON deal_analyses(opportunity_id);
CREATE INDEX IF NOT EXISTS idx_deal_analyses_company ON deal_analyses(company_id);

CREATE INDEX IF NOT EXISTS idx_data_quality_issues_vertical_type ON data_quality_issues(vertical_id, issue_type);
CREATE INDEX IF NOT EXISTS idx_data_quality_issues_severity ON data_quality_issues(vertical_id, severity) WHERE resolved = false;
CREATE INDEX IF NOT EXISTS idx_data_quality_issues_target ON data_quality_issues(target_table, target_id);

CREATE INDEX IF NOT EXISTS idx_compliance_records_vertical_status ON compliance_records(vertical_id, status);
CREATE INDEX IF NOT EXISTS idx_compliance_records_regulation ON compliance_records(vertical_id, regulation);
CREATE INDEX IF NOT EXISTS idx_compliance_records_contact ON compliance_records(contact_id);
CREATE INDEX IF NOT EXISTS idx_compliance_records_expiry ON compliance_records(retention_expiry) WHERE retention_expiry IS NOT NULL;

-- RPC: get_universal_business_metrics_v2
CREATE OR REPLACE FUNCTION get_universal_business_metrics_v2(p_vertical_id TEXT)
RETURNS JSONB
LANGUAGE plpgsql SECURITY DEFINER
AS $$
DECLARE
    result JSONB;
BEGIN
    SELECT jsonb_build_object(
        'onboarding_active', (SELECT COUNT(*) FROM client_onboarding WHERE vertical_id = p_vertical_id AND status IN ('pending', 'in_progress')),
        'onboarding_completed', (SELECT COUNT(*) FROM client_onboarding WHERE vertical_id = p_vertical_id AND status = 'completed'),
        'onboarding_stalled', (SELECT COUNT(*) FROM client_onboarding WHERE vertical_id = p_vertical_id AND status = 'stalled'),
        'invoices_outstanding', (SELECT COUNT(*) FROM invoices WHERE vertical_id = p_vertical_id AND status IN ('sent', 'viewed')),
        'invoices_overdue', (SELECT COUNT(*) FROM invoices WHERE vertical_id = p_vertical_id AND status = 'overdue'),
        'invoices_total_cents', (SELECT COALESCE(SUM(total_cents), 0) FROM invoices WHERE vertical_id = p_vertical_id AND status = 'paid'),
        'knowledge_articles_published', (SELECT COUNT(*) FROM knowledge_articles WHERE vertical_id = p_vertical_id AND status = 'published'),
        'knowledge_articles_draft', (SELECT COUNT(*) FROM knowledge_articles WHERE vertical_id = p_vertical_id AND status = 'draft'),
        'feedback_avg_nps', (SELECT COALESCE(AVG(nps_score), 0) FROM feedback_responses WHERE vertical_id = p_vertical_id AND nps_score IS NOT NULL),
        'feedback_total_responses', (SELECT COUNT(*) FROM feedback_responses WHERE vertical_id = p_vertical_id),
        'feedback_pending_followup', (SELECT COUNT(*) FROM feedback_responses WHERE vertical_id = p_vertical_id AND requires_followup = true AND followup_status = 'pending'),
        'referrals_active', (SELECT COUNT(*) FROM referrals WHERE vertical_id = p_vertical_id AND status IN ('submitted', 'contacted', 'qualified')),
        'referrals_converted', (SELECT COUNT(*) FROM referrals WHERE vertical_id = p_vertical_id AND status = 'converted'),
        'referrals_commission_total', (SELECT COALESCE(SUM(commission_cents), 0) FROM referrals WHERE vertical_id = p_vertical_id AND status = 'converted'),
        'deals_won', (SELECT COUNT(*) FROM deal_analyses WHERE vertical_id = p_vertical_id AND outcome = 'won'),
        'deals_lost', (SELECT COUNT(*) FROM deal_analyses WHERE vertical_id = p_vertical_id AND outcome = 'lost'),
        'deals_avg_cycle_days', (SELECT COALESCE(AVG(sales_cycle_days), 0) FROM deal_analyses WHERE vertical_id = p_vertical_id),
        'data_quality_open_issues', (SELECT COUNT(*) FROM data_quality_issues WHERE vertical_id = p_vertical_id AND resolved = false),
        'data_quality_critical', (SELECT COUNT(*) FROM data_quality_issues WHERE vertical_id = p_vertical_id AND severity = 'critical' AND resolved = false),
        'compliance_active_records', (SELECT COUNT(*) FROM compliance_records WHERE vertical_id = p_vertical_id AND status = 'active'),
        'compliance_pending_review', (SELECT COUNT(*) FROM compliance_records WHERE vertical_id = p_vertical_id AND status = 'pending_review'),
        'compliance_expiring_soon', (SELECT COUNT(*) FROM compliance_records WHERE vertical_id = p_vertical_id AND retention_expiry IS NOT NULL AND retention_expiry <= CURRENT_DATE + INTERVAL '30 days' AND status = 'active')
    )
    INTO result;

    RETURN result;
END;
$$;
