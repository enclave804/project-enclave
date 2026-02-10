-- ============================================================
-- Migration 015: 3D Printing Domain Tables
-- Phase 19 — PrintBiz Domain Expert Agent Infrastructure
-- ============================================================

-- print_jobs: tracks print jobs through production
CREATE TABLE IF NOT EXISTS print_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    vertical_id TEXT NOT NULL,
    company_id UUID REFERENCES companies(id) ON DELETE SET NULL,
    contact_id UUID REFERENCES contacts(id) ON DELETE SET NULL,
    opportunity_id UUID REFERENCES opportunities(id) ON DELETE SET NULL,
    job_name TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending','file_review','quoting','approved','printing',
                          'post_processing','qc','shipping','delivered','cancelled')),
    -- File info
    file_name TEXT DEFAULT '',
    file_url TEXT DEFAULT '',
    file_format TEXT DEFAULT '',
    file_size_bytes BIGINT DEFAULT 0,
    -- Geometry
    geometry_analysis JSONB DEFAULT '{}',
    dimensions_mm JSONB DEFAULT '{}',
    volume_cm3 FLOAT DEFAULT 0,
    surface_area_cm2 FLOAT DEFAULT 0,
    is_manifold BOOLEAN DEFAULT true,
    mesh_issues JSONB DEFAULT '[]',
    -- Print config
    material TEXT DEFAULT '',
    technology TEXT DEFAULT '',
    layer_height_um INTEGER DEFAULT 200,
    infill_percent INTEGER DEFAULT 20,
    scale_factor FLOAT DEFAULT 1.0,
    -- Pricing
    estimated_cost_cents INTEGER DEFAULT 0,
    quoted_price_cents INTEGER DEFAULT 0,
    -- Timeline
    estimated_print_hours FLOAT DEFAULT 0,
    print_started_at TIMESTAMPTZ,
    print_completed_at TIMESTAMPTZ,
    shipped_at TIMESTAMPTZ,
    delivered_at TIMESTAMPTZ,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- file_analyses: geometry analysis results
CREATE TABLE IF NOT EXISTS file_analyses (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    vertical_id TEXT NOT NULL,
    print_job_id UUID REFERENCES print_jobs(id) ON DELETE CASCADE,
    file_name TEXT NOT NULL DEFAULT '',
    file_format TEXT DEFAULT '',
    vertex_count INTEGER DEFAULT 0,
    face_count INTEGER DEFAULT 0,
    is_manifold BOOLEAN DEFAULT true,
    is_watertight BOOLEAN DEFAULT true,
    has_normals BOOLEAN DEFAULT true,
    bounding_box JSONB DEFAULT '{}',
    volume_cm3 FLOAT DEFAULT 0,
    surface_area_cm2 FLOAT DEFAULT 0,
    issues JSONB DEFAULT '[]',
    warnings JSONB DEFAULT '[]',
    printability_score FLOAT DEFAULT 0,
    repairs_applied JSONB DEFAULT '[]',
    repaired_file_url TEXT DEFAULT '',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- quality_inspections: QC results
CREATE TABLE IF NOT EXISTS quality_inspections (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    vertical_id TEXT NOT NULL,
    print_job_id UUID REFERENCES print_jobs(id) ON DELETE CASCADE,
    inspector_type TEXT DEFAULT 'automated',
    dimensional_accuracy JSONB DEFAULT '{}',
    surface_quality_score FLOAT DEFAULT 0,
    structural_integrity FLOAT DEFAULT 0,
    overall_score FLOAT DEFAULT 0,
    defects JSONB DEFAULT '[]',
    passed BOOLEAN DEFAULT false,
    notes TEXT DEFAULT '',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- print_quotes: detailed cost breakdowns
CREATE TABLE IF NOT EXISTS print_quotes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    vertical_id TEXT NOT NULL,
    print_job_id UUID REFERENCES print_jobs(id) ON DELETE CASCADE,
    company_id UUID REFERENCES companies(id) ON DELETE SET NULL,
    contact_id UUID REFERENCES contacts(id) ON DELETE SET NULL,
    material_cost_cents INTEGER DEFAULT 0,
    print_time_cost_cents INTEGER DEFAULT 0,
    post_processing_cost_cents INTEGER DEFAULT 0,
    shipping_cost_cents INTEGER DEFAULT 0,
    markup_percent FLOAT DEFAULT 30,
    total_cents INTEGER DEFAULT 0,
    material TEXT DEFAULT '',
    technology TEXT DEFAULT '',
    estimated_days INTEGER DEFAULT 5,
    line_items JSONB DEFAULT '[]',
    status TEXT NOT NULL DEFAULT 'draft'
        CHECK (status IN ('draft','sent','accepted','rejected','expired')),
    sent_at TIMESTAMPTZ,
    expires_at TIMESTAMPTZ,
    accepted_at TIMESTAMPTZ,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_print_jobs_vertical_status ON print_jobs(vertical_id, status);
CREATE INDEX IF NOT EXISTS idx_print_jobs_company ON print_jobs(company_id);
CREATE INDEX IF NOT EXISTS idx_file_analyses_vertical ON file_analyses(vertical_id);
CREATE INDEX IF NOT EXISTS idx_file_analyses_job ON file_analyses(print_job_id);
CREATE INDEX IF NOT EXISTS idx_quality_inspections_vertical ON quality_inspections(vertical_id);
CREATE INDEX IF NOT EXISTS idx_quality_inspections_job ON quality_inspections(print_job_id);
CREATE INDEX IF NOT EXISTS idx_print_quotes_vertical_status ON print_quotes(vertical_id, status);
CREATE INDEX IF NOT EXISTS idx_print_quotes_job ON print_quotes(print_job_id);
CREATE INDEX IF NOT EXISTS idx_print_quotes_company ON print_quotes(company_id);

-- RPC: get_print_stats
CREATE OR REPLACE FUNCTION get_print_stats(p_vertical_id TEXT)
RETURNS JSONB
LANGUAGE plpgsql SECURITY DEFINER
AS $$
DECLARE
    result JSONB;
BEGIN
    SELECT jsonb_build_object(
        'total_jobs', COUNT(*),
        'pending', COUNT(*) FILTER (WHERE status = 'pending'),
        'printing', COUNT(*) FILTER (WHERE status = 'printing'),
        'completed', COUNT(*) FILTER (WHERE status = 'delivered'),
        'cancelled', COUNT(*) FILTER (WHERE status = 'cancelled'),
        'total_revenue_cents', COALESCE(SUM(quoted_price_cents) FILTER (WHERE status = 'delivered'), 0),
        'avg_print_hours', COALESCE(AVG(estimated_print_hours) FILTER (WHERE status = 'delivered'), 0)
    )
    INTO result
    FROM print_jobs
    WHERE vertical_id = p_vertical_id;

    RETURN result;
END;
$$;
