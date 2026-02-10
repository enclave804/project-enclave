-- ============================================================
-- Migration 014: Domain Expert Agents — Phase 18
-- ============================================================
-- Adds tables for cybersecurity domain expert agents:
-- security_assessments, security_findings, remediation_tasks.
-- These tables power the VulnScanner, NetworkAnalyst, AppSecReviewer,
-- ComplianceMapper, RiskReporter, IAMAnalyst, IncidentReadiness,
-- CloudSecurity, SecurityTrainer, and RemediationGuide agents.
-- ============================================================

-- ─── Security Assessments ───────────────────────────────────

CREATE TABLE IF NOT EXISTS security_assessments (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    vertical_id         TEXT NOT NULL,
    company_id          UUID REFERENCES companies(id) ON DELETE CASCADE,
    contact_id          UUID REFERENCES contacts(id) ON DELETE SET NULL,
    opportunity_id      UUID REFERENCES opportunities(id) ON DELETE SET NULL,

    -- Assessment type
    assessment_type     TEXT NOT NULL
        CHECK (assessment_type IN (
            'vulnerability_scan', 'network_analysis', 'app_security',
            'compliance_mapping', 'cloud_security', 'iam_review',
            'incident_readiness', 'full_assessment'
        )),

    -- Status
    status              TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN (
            'pending', 'in_progress', 'completed', 'failed', 'cancelled'
        )),

    -- Findings: [{severity, category, title, description, recommendation, cve_id}]
    findings            JSONB DEFAULT '[]'::JSONB,

    -- Risk score: 0.0-10.0 overall risk
    risk_score          FLOAT DEFAULT 0.0,

    -- Executive summary
    executive_summary   TEXT DEFAULT '',

    -- Remediation plan
    remediation_plan    JSONB DEFAULT '[]'::JSONB,

    -- Frameworks checked: ['SOC2', 'HIPAA', 'PCI', 'ISO27001']
    frameworks_checked  JSONB DEFAULT '[]'::JSONB,

    -- Metadata
    metadata            JSONB DEFAULT '{}'::JSONB,

    -- Timing
    started_at          TIMESTAMPTZ,
    completed_at        TIMESTAMPTZ,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_security_assessments_vertical_status
    ON security_assessments (vertical_id, status);
CREATE INDEX IF NOT EXISTS idx_security_assessments_company
    ON security_assessments (company_id);
CREATE INDEX IF NOT EXISTS idx_security_assessments_type
    ON security_assessments (assessment_type);

-- ─── Security Findings ──────────────────────────────────────

CREATE TABLE IF NOT EXISTS security_findings (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    assessment_id       UUID REFERENCES security_assessments(id) ON DELETE CASCADE,
    vertical_id         TEXT NOT NULL,

    -- Severity
    severity            TEXT NOT NULL
        CHECK (severity IN (
            'critical', 'high', 'medium', 'low', 'informational'
        )),

    -- Category
    category            TEXT NOT NULL
        CHECK (category IN (
            'network', 'application', 'infrastructure', 'access_control',
            'compliance', 'cloud', 'human', 'physical'
        )),

    -- Details
    title               TEXT NOT NULL,
    description         TEXT NOT NULL DEFAULT '',
    recommendation      TEXT DEFAULT '',

    -- Affected asset: IP, domain, service, etc.
    affected_asset      TEXT DEFAULT '',

    -- CVE tracking
    cve_id              TEXT DEFAULT '',
    cvss_score          FLOAT DEFAULT 0.0,

    -- Status
    status              TEXT NOT NULL DEFAULT 'open'
        CHECK (status IN (
            'open', 'in_progress', 'remediated', 'accepted', 'false_positive'
        )),

    -- Timing
    remediated_at       TIMESTAMPTZ,
    verified_at         TIMESTAMPTZ,

    -- Evidence & metadata
    evidence            JSONB DEFAULT '{}'::JSONB,
    metadata            JSONB DEFAULT '{}'::JSONB,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_security_findings_assessment_severity
    ON security_findings (assessment_id, severity);
CREATE INDEX IF NOT EXISTS idx_security_findings_vertical_status
    ON security_findings (vertical_id, status);
CREATE INDEX IF NOT EXISTS idx_security_findings_category
    ON security_findings (category);

-- ─── Remediation Tasks ──────────────────────────────────────

CREATE TABLE IF NOT EXISTS remediation_tasks (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    finding_id          UUID REFERENCES security_findings(id) ON DELETE CASCADE,
    vertical_id         TEXT NOT NULL,

    -- Task details
    title               TEXT NOT NULL,
    description         TEXT DEFAULT '',
    priority            TEXT
        CHECK (priority IN (
            'critical', 'high', 'medium', 'low'
        )),

    -- Status
    status              TEXT DEFAULT 'pending'
        CHECK (status IN (
            'pending', 'assigned', 'in_progress', 'completed',
            'verified', 'deferred'
        )),

    -- Assignment
    assigned_to         TEXT DEFAULT '',
    due_date            TIMESTAMPTZ,

    -- Verification: how to verify the fix
    verification_method TEXT DEFAULT '',
    verified_at         TIMESTAMPTZ,

    -- Metadata
    metadata            JSONB DEFAULT '{}'::JSONB,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_remediation_tasks_finding
    ON remediation_tasks (finding_id);
CREATE INDEX IF NOT EXISTS idx_remediation_tasks_vertical_status
    ON remediation_tasks (vertical_id, status);

-- ─── RPC: Get Assessment Summary ────────────────────────────

CREATE OR REPLACE FUNCTION get_assessment_summary(
    p_vertical_id TEXT
)
RETURNS JSONB
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
DECLARE
    result JSONB;
BEGIN
    SELECT jsonb_build_object(
        'assessments', (
            SELECT jsonb_build_object(
                'total', COUNT(*),
                'pending', COUNT(*) FILTER (WHERE status = 'pending'),
                'in_progress', COUNT(*) FILTER (WHERE status = 'in_progress'),
                'completed', COUNT(*) FILTER (WHERE status = 'completed'),
                'failed', COUNT(*) FILTER (WHERE status = 'failed'),
                'cancelled', COUNT(*) FILTER (WHERE status = 'cancelled'),
                'avg_risk_score', COALESCE(AVG(risk_score) FILTER (WHERE status = 'completed'), 0)
            )
            FROM security_assessments
            WHERE vertical_id = p_vertical_id
        ),
        'findings', (
            SELECT jsonb_build_object(
                'total', COUNT(*),
                'critical', COUNT(*) FILTER (WHERE severity = 'critical'),
                'high', COUNT(*) FILTER (WHERE severity = 'high'),
                'medium', COUNT(*) FILTER (WHERE severity = 'medium'),
                'low', COUNT(*) FILTER (WHERE severity = 'low'),
                'informational', COUNT(*) FILTER (WHERE severity = 'informational'),
                'open', COUNT(*) FILTER (WHERE status = 'open'),
                'in_progress', COUNT(*) FILTER (WHERE status = 'in_progress'),
                'remediated', COUNT(*) FILTER (WHERE status = 'remediated'),
                'accepted', COUNT(*) FILTER (WHERE status = 'accepted'),
                'false_positive', COUNT(*) FILTER (WHERE status = 'false_positive')
            )
            FROM security_findings
            WHERE vertical_id = p_vertical_id
        ),
        'remediation', (
            SELECT jsonb_build_object(
                'total', COUNT(*),
                'pending', COUNT(*) FILTER (WHERE status = 'pending'),
                'assigned', COUNT(*) FILTER (WHERE status = 'assigned'),
                'in_progress', COUNT(*) FILTER (WHERE status = 'in_progress'),
                'completed', COUNT(*) FILTER (WHERE status = 'completed'),
                'verified', COUNT(*) FILTER (WHERE status = 'verified'),
                'deferred', COUNT(*) FILTER (WHERE status = 'deferred'),
                'completion_rate', CASE
                    WHEN COUNT(*) > 0 THEN
                        ROUND(
                            (COUNT(*) FILTER (WHERE status IN ('completed', 'verified')))::NUMERIC
                            / COUNT(*)::NUMERIC * 100, 1
                        )
                    ELSE 0
                END
            )
            FROM remediation_tasks
            WHERE vertical_id = p_vertical_id
        )
    ) INTO result;

    RETURN result;
END;
$$;
