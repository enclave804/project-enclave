-- ============================================================
-- Migration 009: Operations Agents — Phase 12
-- ============================================================
-- Adds tables for Finance Agent and Customer Success Agent.
-- These tables track invoices, payment reminders, client
-- records, CS interactions, and their lifecycle.
-- ============================================================

-- ─── Invoices ───────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS invoices (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    vertical_id     TEXT NOT NULL,
    agent_id        TEXT NOT NULL DEFAULT 'finance',

    -- Stripe references
    stripe_invoice_id  TEXT UNIQUE,
    stripe_customer_id TEXT DEFAULT '',

    -- Deal context
    proposal_id     TEXT DEFAULT '',
    company_name    TEXT NOT NULL,
    contact_email   TEXT NOT NULL,
    contact_name    TEXT DEFAULT '',

    -- Invoice details
    amount_cents    BIGINT NOT NULL DEFAULT 0,
    currency        TEXT NOT NULL DEFAULT 'usd',
    description     TEXT DEFAULT 'Professional Services',
    due_date        TIMESTAMPTZ,

    -- Status lifecycle: draft → open → paid | overdue | void
    status          TEXT NOT NULL DEFAULT 'draft'
        CHECK (status IN ('draft', 'open', 'paid', 'overdue', 'void', 'uncollectible')),

    -- Payment tracking
    paid_at         TIMESTAMPTZ,
    voided_at       TIMESTAMPTZ,
    days_overdue    INTEGER DEFAULT 0,

    -- URLs
    hosted_url      TEXT DEFAULT '',
    pdf_url         TEXT DEFAULT '',

    -- Metadata
    metadata        JSONB DEFAULT '{}'::JSONB,

    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_invoices_vertical
    ON invoices (vertical_id);
CREATE INDEX IF NOT EXISTS idx_invoices_status
    ON invoices (status);
CREATE INDEX IF NOT EXISTS idx_invoices_proposal
    ON invoices (proposal_id);
CREATE INDEX IF NOT EXISTS idx_invoices_stripe
    ON invoices (stripe_invoice_id);
CREATE INDEX IF NOT EXISTS idx_invoices_due
    ON invoices (due_date)
    WHERE status = 'open';

-- ─── Payment Reminders ──────────────────────────────────────

CREATE TABLE IF NOT EXISTS payment_reminders (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    vertical_id     TEXT NOT NULL,
    agent_id        TEXT NOT NULL DEFAULT 'finance',

    -- Invoice reference
    invoice_id      UUID REFERENCES invoices(id) ON DELETE CASCADE,
    stripe_invoice_id TEXT DEFAULT '',

    -- Reminder details
    contact_email   TEXT NOT NULL,
    company_name    TEXT DEFAULT '',
    amount_display  TEXT DEFAULT '',
    days_overdue    INTEGER DEFAULT 0,
    tone            TEXT NOT NULL DEFAULT 'polite'
        CHECK (tone IN ('polite', 'firm', 'final')),

    -- Content
    draft_text      TEXT NOT NULL DEFAULT '',
    final_text      TEXT DEFAULT '',

    -- Status lifecycle: draft → approved → sent | rejected
    status          TEXT NOT NULL DEFAULT 'draft'
        CHECK (status IN ('draft', 'approved', 'sent', 'rejected')),
    approved_at     TIMESTAMPTZ,
    sent_at         TIMESTAMPTZ,

    -- Metadata
    was_edited      BOOLEAN DEFAULT false,
    metadata        JSONB DEFAULT '{}'::JSONB,

    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_reminders_vertical
    ON payment_reminders (vertical_id);
CREATE INDEX IF NOT EXISTS idx_reminders_invoice
    ON payment_reminders (invoice_id);
CREATE INDEX IF NOT EXISTS idx_reminders_status
    ON payment_reminders (status);

-- ─── Client Records ─────────────────────────────────────────

CREATE TABLE IF NOT EXISTS client_records (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    vertical_id     TEXT NOT NULL,
    agent_id        TEXT NOT NULL DEFAULT 'cs',

    -- Client identity
    company_name    TEXT NOT NULL,
    company_domain  TEXT DEFAULT '',
    contact_name    TEXT NOT NULL DEFAULT '',
    contact_email   TEXT NOT NULL DEFAULT '',
    contact_title   TEXT DEFAULT '',

    -- Relationship
    status          TEXT NOT NULL DEFAULT 'active'
        CHECK (status IN ('onboarding', 'active', 'at_risk', 'churned', 'paused')),
    started_at      TIMESTAMPTZ DEFAULT now(),

    -- Health metrics
    sentiment_score NUMERIC(3,2) DEFAULT 0.00
        CHECK (sentiment_score >= 0 AND sentiment_score <= 1),
    churn_risk      NUMERIC(3,2) DEFAULT 0.00
        CHECK (churn_risk >= 0 AND churn_risk <= 1),
    last_contact_at TIMESTAMPTZ,
    next_checkin_at TIMESTAMPTZ,

    -- Onboarding
    onboarding_complete BOOLEAN DEFAULT false,
    onboarding_checklist JSONB DEFAULT '[]'::JSONB,

    -- Revenue link
    proposal_id     TEXT DEFAULT '',
    total_invoiced_cents BIGINT DEFAULT 0,
    total_paid_cents     BIGINT DEFAULT 0,

    -- Metadata
    notes           TEXT DEFAULT '',
    tags            JSONB DEFAULT '[]'::JSONB,
    metadata        JSONB DEFAULT '{}'::JSONB,

    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_clients_vertical
    ON client_records (vertical_id);
CREATE INDEX IF NOT EXISTS idx_clients_status
    ON client_records (status);
CREATE INDEX IF NOT EXISTS idx_clients_risk
    ON client_records (churn_risk DESC)
    WHERE status IN ('active', 'at_risk');
CREATE INDEX IF NOT EXISTS idx_clients_domain
    ON client_records (company_domain);

-- ─── CS Interactions ────────────────────────────────────────

CREATE TABLE IF NOT EXISTS cs_interactions (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    vertical_id     TEXT NOT NULL,
    agent_id        TEXT NOT NULL DEFAULT 'cs',

    -- Client reference
    client_id       UUID REFERENCES client_records(id) ON DELETE CASCADE,

    -- Interaction details
    interaction_type TEXT NOT NULL DEFAULT 'checkin'
        CHECK (interaction_type IN (
            'onboarding', 'checkin', 'health_check',
            'escalation', 'renewal', 'feedback'
        )),
    checkin_type    TEXT DEFAULT ''
        CHECK (checkin_type IN ('', 'onboarding', '30_day', '60_day', 'quarterly', 'health_check')),

    -- Content
    subject         TEXT DEFAULT '',
    draft_text      TEXT DEFAULT '',
    final_text      TEXT DEFAULT '',

    -- Status lifecycle: draft → approved → sent | rejected
    status          TEXT NOT NULL DEFAULT 'draft'
        CHECK (status IN ('draft', 'approved', 'sent', 'rejected')),
    approved_at     TIMESTAMPTZ,
    sent_at         TIMESTAMPTZ,

    -- Tracking
    was_edited      BOOLEAN DEFAULT false,
    response_received BOOLEAN DEFAULT false,
    sentiment_score   NUMERIC(3,2) DEFAULT 0.00,

    -- Metadata
    metadata        JSONB DEFAULT '{}'::JSONB,

    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_cs_interactions_vertical
    ON cs_interactions (vertical_id);
CREATE INDEX IF NOT EXISTS idx_cs_interactions_client
    ON cs_interactions (client_id);
CREATE INDEX IF NOT EXISTS idx_cs_interactions_type
    ON cs_interactions (interaction_type, status);

-- ─── RPC: Get Operations Stats ──────────────────────────────

CREATE OR REPLACE FUNCTION get_operations_stats(
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
        'invoices', (
            SELECT jsonb_build_object(
                'total', COUNT(*),
                'open', COUNT(*) FILTER (WHERE status = 'open'),
                'paid', COUNT(*) FILTER (WHERE status = 'paid'),
                'overdue', COUNT(*) FILTER (WHERE status = 'overdue'),
                'total_amount', COALESCE(SUM(amount_cents), 0),
                'paid_amount', COALESCE(SUM(amount_cents) FILTER (WHERE status = 'paid'), 0),
                'overdue_amount', COALESCE(SUM(amount_cents) FILTER (WHERE status = 'overdue'), 0)
            )
            FROM invoices
            WHERE vertical_id = p_vertical_id
              AND created_at >= now() - (p_days || ' days')::INTERVAL
        ),
        'reminders', (
            SELECT jsonb_build_object(
                'total', COUNT(*),
                'sent', COUNT(*) FILTER (WHERE status = 'sent'),
                'pending', COUNT(*) FILTER (WHERE status IN ('draft', 'approved'))
            )
            FROM payment_reminders
            WHERE vertical_id = p_vertical_id
              AND created_at >= now() - (p_days || ' days')::INTERVAL
        ),
        'clients', (
            SELECT jsonb_build_object(
                'total', COUNT(*),
                'active', COUNT(*) FILTER (WHERE status = 'active'),
                'onboarding', COUNT(*) FILTER (WHERE status = 'onboarding'),
                'at_risk', COUNT(*) FILTER (WHERE status = 'at_risk'),
                'churned', COUNT(*) FILTER (WHERE status = 'churned'),
                'avg_churn_risk', COALESCE(AVG(churn_risk) FILTER (WHERE status IN ('active', 'at_risk')), 0)
            )
            FROM client_records
            WHERE vertical_id = p_vertical_id
        ),
        'interactions', (
            SELECT jsonb_build_object(
                'total', COUNT(*),
                'sent', COUNT(*) FILTER (WHERE status = 'sent'),
                'pending', COUNT(*) FILTER (WHERE status IN ('draft', 'approved'))
            )
            FROM cs_interactions
            WHERE vertical_id = p_vertical_id
              AND created_at >= now() - (p_days || ' days')::INTERVAL
        )
    ) INTO result;

    RETURN result;
END;
$$;
