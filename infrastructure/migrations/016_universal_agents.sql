-- ============================================================
-- Migration 016: Universal Business Agent Tables
-- Phase 20 — Cross-Vertical Business Operations
-- ============================================================

-- contracts: service agreements, MSAs, NDAs
CREATE TABLE IF NOT EXISTS contracts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    vertical_id TEXT NOT NULL,
    company_id UUID REFERENCES companies(id) ON DELETE SET NULL,
    contact_id UUID REFERENCES contacts(id) ON DELETE SET NULL,
    opportunity_id UUID REFERENCES opportunities(id) ON DELETE SET NULL,
    contract_type TEXT NOT NULL DEFAULT 'service_agreement'
        CHECK (contract_type IN ('service_agreement','msa','nda','sow','addendum')),
    title TEXT NOT NULL DEFAULT '',
    content_markdown TEXT DEFAULT '',
    status TEXT NOT NULL DEFAULT 'draft'
        CHECK (status IN ('draft','pending_review','sent','signed','active','expired','cancelled')),
    value_cents INTEGER DEFAULT 0,
    start_date DATE,
    end_date DATE,
    renewal_date DATE,
    auto_renew BOOLEAN DEFAULT false,
    signed_at TIMESTAMPTZ,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- support_tickets: client support requests
CREATE TABLE IF NOT EXISTS support_tickets (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    vertical_id TEXT NOT NULL,
    company_id UUID REFERENCES companies(id) ON DELETE SET NULL,
    contact_id UUID REFERENCES contacts(id) ON DELETE SET NULL,
    ticket_number TEXT NOT NULL DEFAULT '',
    subject TEXT NOT NULL DEFAULT '',
    description TEXT DEFAULT '',
    category TEXT DEFAULT 'general',
    priority TEXT DEFAULT 'medium'
        CHECK (priority IN ('low','medium','high','urgent')),
    status TEXT NOT NULL DEFAULT 'open'
        CHECK (status IN ('open','in_progress','waiting_on_client','resolved','closed')),
    assigned_to TEXT DEFAULT '',
    resolution TEXT DEFAULT '',
    resolved_at TIMESTAMPTZ,
    first_response_at TIMESTAMPTZ,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- competitor_intel: market intelligence
CREATE TABLE IF NOT EXISTS competitor_intel (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    vertical_id TEXT NOT NULL,
    competitor_name TEXT NOT NULL DEFAULT '',
    competitor_domain TEXT DEFAULT '',
    intel_type TEXT DEFAULT 'pricing'
        CHECK (intel_type IN ('pricing','feature','review','news','hiring','technology')),
    title TEXT NOT NULL DEFAULT '',
    content TEXT DEFAULT '',
    source_url TEXT DEFAULT '',
    severity TEXT DEFAULT 'info'
        CHECK (severity IN ('info','low','medium','high','critical')),
    actionable BOOLEAN DEFAULT false,
    action_taken TEXT DEFAULT '',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- business_reports: generated analytics reports
CREATE TABLE IF NOT EXISTS business_reports (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    vertical_id TEXT NOT NULL,
    report_type TEXT NOT NULL DEFAULT 'weekly'
        CHECK (report_type IN ('daily','weekly','monthly','quarterly','annual','custom')),
    title TEXT NOT NULL DEFAULT '',
    content_markdown TEXT DEFAULT '',
    metrics JSONB DEFAULT '{}',
    period_start DATE,
    period_end DATE,
    status TEXT DEFAULT 'draft'
        CHECK (status IN ('draft','generated','reviewed','published')),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_contracts_vertical_status ON contracts(vertical_id, status);
CREATE INDEX IF NOT EXISTS idx_contracts_company ON contracts(company_id);
CREATE INDEX IF NOT EXISTS idx_contracts_renewal ON contracts(renewal_date) WHERE renewal_date IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_support_tickets_vertical_status ON support_tickets(vertical_id, status);
CREATE INDEX IF NOT EXISTS idx_support_tickets_company ON support_tickets(company_id);
CREATE INDEX IF NOT EXISTS idx_support_tickets_priority ON support_tickets(vertical_id, priority) WHERE status NOT IN ('resolved', 'closed');
CREATE INDEX IF NOT EXISTS idx_competitor_intel_vertical ON competitor_intel(vertical_id);
CREATE INDEX IF NOT EXISTS idx_competitor_intel_type ON competitor_intel(vertical_id, intel_type);
CREATE INDEX IF NOT EXISTS idx_business_reports_vertical_type ON business_reports(vertical_id, report_type);

-- RPC: get_business_metrics
CREATE OR REPLACE FUNCTION get_business_metrics(p_vertical_id TEXT)
RETURNS JSONB
LANGUAGE plpgsql SECURITY DEFINER
AS $$
DECLARE
    result JSONB;
BEGIN
    SELECT jsonb_build_object(
        'active_contracts', (SELECT COUNT(*) FROM contracts WHERE vertical_id = p_vertical_id AND status = 'active'),
        'contract_value_cents', (SELECT COALESCE(SUM(value_cents), 0) FROM contracts WHERE vertical_id = p_vertical_id AND status = 'active'),
        'open_tickets', (SELECT COUNT(*) FROM support_tickets WHERE vertical_id = p_vertical_id AND status IN ('open', 'in_progress')),
        'urgent_tickets', (SELECT COUNT(*) FROM support_tickets WHERE vertical_id = p_vertical_id AND priority = 'urgent' AND status NOT IN ('resolved', 'closed')),
        'competitor_alerts', (SELECT COUNT(*) FROM competitor_intel WHERE vertical_id = p_vertical_id AND actionable = true AND action_taken = ''),
        'reports_this_month', (SELECT COUNT(*) FROM business_reports WHERE vertical_id = p_vertical_id AND created_at >= date_trunc('month', now()))
    )
    INTO result;

    RETURN result;
END;
$$;
