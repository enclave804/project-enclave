-- ============================================================
-- Migration 013: Sales Pipeline Agents — Phase 16
-- ============================================================
-- Adds tables for the Follow-Up Agent (multi-touch sequences)
-- and Meeting Scheduler Agent (calendar booking).
-- Leverages existing tables: opportunities, outreach_events,
-- proposals, invoices, client_records.
-- ============================================================

-- ─── Follow-Up Sequences ──────────────────────────────────────

CREATE TABLE IF NOT EXISTS follow_up_sequences (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    vertical_id     TEXT NOT NULL,
    contact_id      UUID REFERENCES contacts(id) ON DELETE CASCADE,
    company_id      UUID REFERENCES companies(id) ON DELETE SET NULL,
    opportunity_id  UUID REFERENCES opportunities(id) ON DELETE SET NULL,

    -- Sequence config
    sequence_name   TEXT NOT NULL DEFAULT 'default',
    current_step    INTEGER NOT NULL DEFAULT 0,
    max_steps       INTEGER NOT NULL DEFAULT 5,
    status          TEXT NOT NULL DEFAULT 'active'
        CHECK (status IN ('active', 'paused', 'completed', 'cancelled', 'replied')),

    -- Contact info (denormalized for quick access)
    contact_email   TEXT NOT NULL,
    contact_name    TEXT DEFAULT '',
    company_name    TEXT DEFAULT '',

    -- Timing
    interval_days   INTEGER NOT NULL DEFAULT 3,
    next_send_at    TIMESTAMPTZ,
    last_sent_at    TIMESTAMPTZ,
    started_at      TIMESTAMPTZ DEFAULT now(),
    completed_at    TIMESTAMPTZ,

    -- Content: [{step, subject, body, sent_at, status}]
    steps           JSONB DEFAULT '[]'::JSONB,

    -- Metadata
    metadata        JSONB DEFAULT '{}'::JSONB,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_followup_vertical
    ON follow_up_sequences (vertical_id);
CREATE INDEX IF NOT EXISTS idx_followup_status
    ON follow_up_sequences (status);
CREATE INDEX IF NOT EXISTS idx_followup_contact
    ON follow_up_sequences (contact_id);
CREATE INDEX IF NOT EXISTS idx_followup_opportunity
    ON follow_up_sequences (opportunity_id);
CREATE INDEX IF NOT EXISTS idx_followup_next_send
    ON follow_up_sequences (next_send_at)
    WHERE status = 'active';
CREATE INDEX IF NOT EXISTS idx_followup_email
    ON follow_up_sequences (contact_email);

-- ─── Scheduled Meetings ───────────────────────────────────────

CREATE TABLE IF NOT EXISTS scheduled_meetings (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    vertical_id     TEXT NOT NULL,
    contact_id      UUID REFERENCES contacts(id) ON DELETE CASCADE,
    company_id      UUID REFERENCES companies(id) ON DELETE SET NULL,
    opportunity_id  UUID REFERENCES opportunities(id) ON DELETE SET NULL,

    -- Meeting details
    meeting_type    TEXT NOT NULL DEFAULT 'discovery'
        CHECK (meeting_type IN (
            'discovery', 'demo', 'follow_up',
            'negotiation', 'kickoff'
        )),
    title           TEXT NOT NULL DEFAULT '',

    -- Participants
    contact_email   TEXT NOT NULL,
    contact_name    TEXT DEFAULT '',
    company_name    TEXT DEFAULT '',
    organizer_email TEXT DEFAULT '',

    -- Scheduling
    scheduled_at    TIMESTAMPTZ,
    duration_minutes INTEGER DEFAULT 30,
    meeting_url     TEXT DEFAULT '',
    calendar_event_id TEXT DEFAULT '',

    -- Status: proposed → confirmed → completed | cancelled | no_show
    status          TEXT NOT NULL DEFAULT 'proposed'
        CHECK (status IN (
            'proposed', 'confirmed', 'completed',
            'cancelled', 'no_show', 'rescheduled'
        )),

    -- Notes
    agenda          TEXT DEFAULT '',
    notes           TEXT DEFAULT '',
    outcome         TEXT DEFAULT '',

    -- Follow-up
    follow_up_action TEXT DEFAULT '',
    next_step       TEXT DEFAULT '',

    -- Metadata
    metadata        JSONB DEFAULT '{}'::JSONB,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_meetings_vertical
    ON scheduled_meetings (vertical_id);
CREATE INDEX IF NOT EXISTS idx_meetings_status
    ON scheduled_meetings (status);
CREATE INDEX IF NOT EXISTS idx_meetings_contact
    ON scheduled_meetings (contact_id);
CREATE INDEX IF NOT EXISTS idx_meetings_opportunity
    ON scheduled_meetings (opportunity_id);
CREATE INDEX IF NOT EXISTS idx_meetings_scheduled
    ON scheduled_meetings (scheduled_at)
    WHERE status IN ('proposed', 'confirmed');
CREATE INDEX IF NOT EXISTS idx_meetings_type
    ON scheduled_meetings (meeting_type);

-- ─── RPC: Get Pipeline Stats ──────────────────────────────────

CREATE OR REPLACE FUNCTION get_pipeline_stats(
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
        'opportunities', (
            SELECT jsonb_build_object(
                'total', COUNT(*),
                'prospect', COUNT(*) FILTER (WHERE stage = 'prospect'),
                'qualified', COUNT(*) FILTER (WHERE stage = 'qualified'),
                'proposal', COUNT(*) FILTER (WHERE stage = 'proposal'),
                'negotiation', COUNT(*) FILTER (WHERE stage = 'negotiation'),
                'closed_won', COUNT(*) FILTER (WHERE stage = 'closed_won'),
                'closed_lost', COUNT(*) FILTER (WHERE stage = 'closed_lost'),
                'total_value', COALESCE(SUM(value_cents), 0),
                'won_value', COALESCE(SUM(value_cents) FILTER (WHERE stage = 'closed_won'), 0)
            )
            FROM opportunities
            WHERE vertical_id = p_vertical_id
              AND created_at >= now() - (p_days || ' days')::INTERVAL
        ),
        'sequences', (
            SELECT jsonb_build_object(
                'total', COUNT(*),
                'active', COUNT(*) FILTER (WHERE status = 'active'),
                'completed', COUNT(*) FILTER (WHERE status = 'completed'),
                'replied', COUNT(*) FILTER (WHERE status = 'replied'),
                'paused', COUNT(*) FILTER (WHERE status = 'paused'),
                'avg_steps', COALESCE(AVG(current_step), 0)
            )
            FROM follow_up_sequences
            WHERE vertical_id = p_vertical_id
              AND created_at >= now() - (p_days || ' days')::INTERVAL
        ),
        'meetings', (
            SELECT jsonb_build_object(
                'total', COUNT(*),
                'proposed', COUNT(*) FILTER (WHERE status = 'proposed'),
                'confirmed', COUNT(*) FILTER (WHERE status = 'confirmed'),
                'completed', COUNT(*) FILTER (WHERE status = 'completed'),
                'cancelled', COUNT(*) FILTER (WHERE status = 'cancelled'),
                'no_show', COUNT(*) FILTER (WHERE status = 'no_show')
            )
            FROM scheduled_meetings
            WHERE vertical_id = p_vertical_id
              AND created_at >= now() - (p_days || ' days')::INTERVAL
        )
    ) INTO result;

    RETURN result;
END;
$$;
