"""
Tests for Print Domain & Universal Agent Infrastructure — Phases 19-20.

Covers:
    - Migration 015_print_domain.sql schema validation (4 tables, indexes, RPC)
    - Migration 016_universal_agents.sql schema validation (4 tables, indexes, RPC)
    - 10 PrintBiz domain agent state TypedDicts
    - 4 Universal business agent state TypedDicts
    - 4 MCP tool stub modules (mesh, printer, shipping, measurement)
    - 10 PrintBiz domain YAML configs
    - 4 PrintBiz universal YAML configs
    - 4 Enclave Guard universal YAML configs
    - 7 Pydantic contract models
"""

from __future__ import annotations

import asyncio
import inspect
import json
from pathlib import Path
from typing import Any, get_type_hints

import pytest
import yaml


# ══════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════

MIGRATION_015_PATH = (
    Path(__file__).parent.parent.parent
    / "infrastructure"
    / "migrations"
    / "015_print_domain.sql"
)

MIGRATION_016_PATH = (
    Path(__file__).parent.parent.parent
    / "infrastructure"
    / "migrations"
    / "016_universal_agents.sql"
)

PRINT_AGENTS_DIR = (
    Path(__file__).parent.parent.parent
    / "verticals"
    / "print_biz"
    / "agents"
)

GUARD_AGENTS_DIR = (
    Path(__file__).parent.parent.parent
    / "verticals"
    / "enclave_guard"
    / "agents"
)

PRINT_DOMAIN_YAMLS = [
    "file_analyst.yaml",
    "mesh_repair.yaml",
    "scale_optimizer.yaml",
    "material_advisor.yaml",
    "quote_engine.yaml",
    "print_manager.yaml",
    "post_process.yaml",
    "qc_inspector.yaml",
    "cad_advisor.yaml",
    "logistics.yaml",
]

PRINT_UNIVERSAL_YAMLS = [
    "contract_manager.yaml",
    "support.yaml",
    "competitive_intel.yaml",
    "reporting.yaml",
]

GUARD_UNIVERSAL_YAMLS = [
    "contract_manager.yaml",
    "support.yaml",
    "competitive_intel.yaml",
    "reporting.yaml",
]


def _read_migration(path: Path) -> str:
    with open(path) as f:
        return f.read()


def _load_yaml(directory: Path, filename: str) -> dict[str, Any]:
    with open(directory / filename) as f:
        return yaml.safe_load(f)


# ══════════════════════════════════════════════════════════════════════
# 1. Migration 015 — Print Domain Tables
# ══════════════════════════════════════════════════════════════════════


class TestMigration015:
    """Validate 015_print_domain.sql migration structure."""

    def test_migration_file_exists(self):
        assert MIGRATION_015_PATH.exists(), "015_print_domain.sql not found"

    def test_migration_not_empty(self):
        sql = _read_migration(MIGRATION_015_PATH)
        assert len(sql) > 100

    # ── Tables ─────────────────────────────────────────────────────

    def test_creates_print_jobs_table(self):
        sql = _read_migration(MIGRATION_015_PATH)
        assert "CREATE TABLE" in sql
        assert "print_jobs" in sql

    def test_creates_file_analyses_table(self):
        sql = _read_migration(MIGRATION_015_PATH)
        assert "file_analyses" in sql

    def test_creates_quality_inspections_table(self):
        sql = _read_migration(MIGRATION_015_PATH)
        assert "quality_inspections" in sql

    def test_creates_print_quotes_table(self):
        sql = _read_migration(MIGRATION_015_PATH)
        assert "print_quotes" in sql

    # ── print_jobs columns ─────────────────────────────────────────

    def test_print_jobs_columns(self):
        sql = _read_migration(MIGRATION_015_PATH)
        expected_cols = [
            "vertical_id",
            "company_id",
            "contact_id",
            "opportunity_id",
            "job_name",
            "status",
            "file_name",
            "file_url",
            "file_format",
            "file_size_bytes",
            "geometry_analysis",
            "dimensions_mm",
            "volume_cm3",
            "surface_area_cm2",
            "is_manifold",
            "mesh_issues",
            "material",
            "technology",
            "layer_height_um",
            "infill_percent",
            "scale_factor",
            "estimated_cost_cents",
            "quoted_price_cents",
            "estimated_print_hours",
        ]
        for col in expected_cols:
            assert col in sql, f"Column '{col}' missing from print_jobs"

    def test_print_jobs_status_check(self):
        sql = _read_migration(MIGRATION_015_PATH)
        for status in [
            "pending", "file_review", "quoting", "approved", "printing",
            "post_processing", "qc", "shipping", "delivered", "cancelled",
        ]:
            assert status in sql, f"print_jobs status '{status}' not in CHECK"

    # ── file_analyses columns ──────────────────────────────────────

    def test_file_analyses_columns(self):
        sql = _read_migration(MIGRATION_015_PATH)
        expected_cols = [
            "vertex_count",
            "face_count",
            "is_manifold",
            "is_watertight",
            "has_normals",
            "bounding_box",
            "volume_cm3",
            "surface_area_cm2",
            "issues",
            "warnings",
            "printability_score",
            "repairs_applied",
        ]
        for col in expected_cols:
            assert col in sql, f"Column '{col}' missing from file_analyses"

    # ── quality_inspections columns ─────────────────────────────────

    def test_quality_inspections_columns(self):
        sql = _read_migration(MIGRATION_015_PATH)
        expected_cols = [
            "dimensional_accuracy",
            "surface_quality_score",
            "structural_integrity",
            "overall_score",
            "defects",
            "passed",
        ]
        for col in expected_cols:
            assert col in sql, f"Column '{col}' missing from quality_inspections"

    # ── print_quotes columns ───────────────────────────────────────

    def test_print_quotes_columns(self):
        sql = _read_migration(MIGRATION_015_PATH)
        expected_cols = [
            "material_cost_cents",
            "print_time_cost_cents",
            "post_processing_cost_cents",
            "shipping_cost_cents",
            "markup_percent",
            "total_cents",
            "line_items",
            "status",
        ]
        for col in expected_cols:
            assert col in sql, f"Column '{col}' missing from print_quotes"

    def test_print_quotes_status_check(self):
        sql = _read_migration(MIGRATION_015_PATH)
        for status in ["draft", "sent", "accepted", "rejected", "expired"]:
            assert status in sql, f"print_quotes status '{status}' not in CHECK"

    # ── Indexes ────────────────────────────────────────────────────

    def test_indexes_created(self):
        sql = _read_migration(MIGRATION_015_PATH)
        assert "CREATE INDEX" in sql

    def test_index_print_jobs_vertical_status(self):
        sql = _read_migration(MIGRATION_015_PATH)
        assert "idx_print_jobs_vertical_status" in sql

    def test_index_print_jobs_company(self):
        sql = _read_migration(MIGRATION_015_PATH)
        assert "idx_print_jobs_company" in sql

    def test_index_file_analyses_vertical(self):
        sql = _read_migration(MIGRATION_015_PATH)
        assert "idx_file_analyses_vertical" in sql

    def test_index_file_analyses_job(self):
        sql = _read_migration(MIGRATION_015_PATH)
        assert "idx_file_analyses_job" in sql

    def test_index_quality_inspections_vertical(self):
        sql = _read_migration(MIGRATION_015_PATH)
        assert "idx_quality_inspections_vertical" in sql

    def test_index_print_quotes_vertical_status(self):
        sql = _read_migration(MIGRATION_015_PATH)
        assert "idx_print_quotes_vertical_status" in sql

    # ── RPC Function ───────────────────────────────────────────────

    def test_rpc_function_get_print_stats(self):
        sql = _read_migration(MIGRATION_015_PATH)
        assert "get_print_stats" in sql

    def test_rpc_function_returns_jsonb(self):
        sql = _read_migration(MIGRATION_015_PATH)
        assert "RETURNS JSONB" in sql

    def test_rpc_function_security_definer(self):
        sql = _read_migration(MIGRATION_015_PATH)
        assert "SECURITY DEFINER" in sql


# ══════════════════════════════════════════════════════════════════════
# 2. Migration 016 — Universal Agent Tables
# ══════════════════════════════════════════════════════════════════════


class TestMigration016:
    """Validate 016_universal_agents.sql migration structure."""

    def test_migration_file_exists(self):
        assert MIGRATION_016_PATH.exists(), "016_universal_agents.sql not found"

    def test_migration_not_empty(self):
        sql = _read_migration(MIGRATION_016_PATH)
        assert len(sql) > 100

    # ── Tables ─────────────────────────────────────────────────────

    def test_creates_contracts_table(self):
        sql = _read_migration(MIGRATION_016_PATH)
        assert "CREATE TABLE" in sql
        assert "contracts" in sql

    def test_creates_support_tickets_table(self):
        sql = _read_migration(MIGRATION_016_PATH)
        assert "support_tickets" in sql

    def test_creates_competitor_intel_table(self):
        sql = _read_migration(MIGRATION_016_PATH)
        assert "competitor_intel" in sql

    def test_creates_business_reports_table(self):
        sql = _read_migration(MIGRATION_016_PATH)
        assert "business_reports" in sql

    # ── contracts columns ──────────────────────────────────────────

    def test_contracts_columns(self):
        sql = _read_migration(MIGRATION_016_PATH)
        expected_cols = [
            "contract_type",
            "title",
            "content_markdown",
            "status",
            "value_cents",
            "start_date",
            "end_date",
            "renewal_date",
            "auto_renew",
        ]
        for col in expected_cols:
            assert col in sql, f"Column '{col}' missing from contracts"

    def test_contracts_type_check(self):
        sql = _read_migration(MIGRATION_016_PATH)
        for ctype in ["service_agreement", "msa", "nda", "sow", "addendum"]:
            assert ctype in sql, f"contract_type '{ctype}' not in CHECK"

    def test_contracts_status_check(self):
        sql = _read_migration(MIGRATION_016_PATH)
        for status in ["draft", "pending_review", "sent", "signed", "active", "expired", "cancelled"]:
            assert status in sql, f"contract status '{status}' not in CHECK"

    # ── support_tickets columns ────────────────────────────────────

    def test_support_tickets_columns(self):
        sql = _read_migration(MIGRATION_016_PATH)
        expected_cols = [
            "ticket_number",
            "subject",
            "description",
            "category",
            "priority",
            "status",
            "assigned_to",
            "resolution",
        ]
        for col in expected_cols:
            assert col in sql, f"Column '{col}' missing from support_tickets"

    def test_support_tickets_priority_check(self):
        sql = _read_migration(MIGRATION_016_PATH)
        for prio in ["low", "medium", "high", "urgent"]:
            assert prio in sql, f"support priority '{prio}' not in CHECK"

    # ── competitor_intel columns ───────────────────────────────────

    def test_competitor_intel_columns(self):
        sql = _read_migration(MIGRATION_016_PATH)
        expected_cols = [
            "competitor_name",
            "competitor_domain",
            "intel_type",
            "title",
            "content",
            "source_url",
            "severity",
            "actionable",
        ]
        for col in expected_cols:
            assert col in sql, f"Column '{col}' missing from competitor_intel"

    # ── business_reports columns ───────────────────────────────────

    def test_business_reports_columns(self):
        sql = _read_migration(MIGRATION_016_PATH)
        expected_cols = [
            "report_type",
            "title",
            "content_markdown",
            "metrics",
            "period_start",
            "period_end",
            "status",
        ]
        for col in expected_cols:
            assert col in sql, f"Column '{col}' missing from business_reports"

    # ── Indexes ────────────────────────────────────────────────────

    def test_indexes_exist(self):
        sql = _read_migration(MIGRATION_016_PATH)
        assert "CREATE INDEX" in sql

    def test_index_contracts_vertical_status(self):
        sql = _read_migration(MIGRATION_016_PATH)
        assert "idx_contracts_vertical_status" in sql

    def test_index_support_tickets_vertical_status(self):
        sql = _read_migration(MIGRATION_016_PATH)
        assert "idx_support_tickets_vertical_status" in sql

    def test_index_competitor_intel_vertical(self):
        sql = _read_migration(MIGRATION_016_PATH)
        assert "idx_competitor_intel_vertical" in sql

    def test_index_business_reports_vertical_type(self):
        sql = _read_migration(MIGRATION_016_PATH)
        assert "idx_business_reports_vertical_type" in sql

    # ── RPC Function ───────────────────────────────────────────────

    def test_rpc_function_get_business_metrics(self):
        sql = _read_migration(MIGRATION_016_PATH)
        assert "get_business_metrics" in sql

    def test_rpc_function_returns_jsonb(self):
        sql = _read_migration(MIGRATION_016_PATH)
        assert "RETURNS JSONB" in sql

    def test_rpc_function_security_definer(self):
        sql = _read_migration(MIGRATION_016_PATH)
        assert "SECURITY DEFINER" in sql


# ══════════════════════════════════════════════════════════════════════
# 3. PrintBiz Domain Agent State TypedDicts
# ══════════════════════════════════════════════════════════════════════


class TestPrintDomainStates:
    """Import and validate all 10 PrintBiz domain agent state TypedDicts."""

    def test_all_print_states_importable(self):
        from core.agents.state import (
            FileAnalystAgentState,
            MeshRepairAgentState,
            ScaleOptimizerAgentState,
            MaterialAdvisorAgentState,
            QuoteEngineAgentState,
            PrintManagerAgentState,
            PostProcessAgentState,
            QCInspectorAgentState,
            CADAdvisorAgentState,
            LogisticsAgentState,
        )
        states = [
            FileAnalystAgentState,
            MeshRepairAgentState,
            ScaleOptimizerAgentState,
            MaterialAdvisorAgentState,
            QuoteEngineAgentState,
            PrintManagerAgentState,
            PostProcessAgentState,
            QCInspectorAgentState,
            CADAdvisorAgentState,
            LogisticsAgentState,
        ]
        for state_cls in states:
            assert state_cls is not None, f"{state_cls} should be importable"

    def test_all_print_states_have_annotations(self):
        from core.agents.state import (
            FileAnalystAgentState,
            MeshRepairAgentState,
            ScaleOptimizerAgentState,
            MaterialAdvisorAgentState,
            QuoteEngineAgentState,
            PrintManagerAgentState,
            PostProcessAgentState,
            QCInspectorAgentState,
            CADAdvisorAgentState,
            LogisticsAgentState,
        )
        states = [
            FileAnalystAgentState,
            MeshRepairAgentState,
            ScaleOptimizerAgentState,
            MaterialAdvisorAgentState,
            QuoteEngineAgentState,
            PrintManagerAgentState,
            PostProcessAgentState,
            QCInspectorAgentState,
            CADAdvisorAgentState,
            LogisticsAgentState,
        ]
        for state_cls in states:
            assert hasattr(state_cls, "__annotations__"), (
                f"{state_cls.__name__} missing __annotations__"
            )
            assert len(state_cls.__annotations__) > 0, (
                f"{state_cls.__name__} has no annotations"
            )

    def test_file_analyst_state_has_geometry_fields(self):
        from core.agents.state import FileAnalystAgentState
        hints = get_type_hints(FileAnalystAgentState)
        for field in [
            "vertex_count", "face_count", "is_manifold", "is_watertight",
            "volume_cm3", "surface_area_cm2", "printability_score",
        ]:
            assert field in hints, f"FileAnalystAgentState missing '{field}'"

    def test_quote_engine_state_has_pricing_fields(self):
        from core.agents.state import QuoteEngineAgentState
        hints = get_type_hints(QuoteEngineAgentState)
        for field in ["total_cents", "quote_document", "report_summary"]:
            assert field in hints, f"QuoteEngineAgentState missing '{field}'"

    def test_mesh_repair_state_key_fields(self):
        from core.agents.state import MeshRepairAgentState
        hints = get_type_hints(MeshRepairAgentState)
        for field in [
            "original_issues", "repair_plan", "repairs_applied",
            "issues_resolved", "issues_remaining", "repair_success_rate",
        ]:
            assert field in hints, f"MeshRepairAgentState missing '{field}'"

    def test_scale_optimizer_state_key_fields(self):
        from core.agents.state import ScaleOptimizerAgentState
        hints = get_type_hints(ScaleOptimizerAgentState)
        for field in [
            "target_scale", "target_scale_factor", "scaled_dimensions",
            "recommended_scale_factor", "fit_on_build_plate",
        ]:
            assert field in hints, f"ScaleOptimizerAgentState missing '{field}'"

    def test_material_advisor_state_key_fields(self):
        from core.agents.state import MaterialAdvisorAgentState
        hints = get_type_hints(MaterialAdvisorAgentState)
        for field in ["recommended_material", "report_summary"]:
            assert field in hints, f"MaterialAdvisorAgentState missing '{field}'"

    def test_print_manager_state_key_fields(self):
        from core.agents.state import PrintManagerAgentState
        hints = get_type_hints(PrintManagerAgentState)
        for field in ["pending_jobs", "report_summary"]:
            assert field in hints, f"PrintManagerAgentState missing '{field}'"

    def test_qc_inspector_state_key_fields(self):
        from core.agents.state import QCInspectorAgentState
        hints = get_type_hints(QCInspectorAgentState)
        for field in ["overall_qc_score", "qc_pass", "defects_found", "report_summary"]:
            assert field in hints, f"QCInspectorAgentState missing '{field}'"

    def test_logistics_state_key_fields(self):
        from core.agents.state import LogisticsAgentState
        hints = get_type_hints(LogisticsAgentState)
        for field in ["report_summary"]:
            assert field in hints, f"LogisticsAgentState missing '{field}'"


# ══════════════════════════════════════════════════════════════════════
# 4. Universal Agent State TypedDicts
# ══════════════════════════════════════════════════════════════════════


class TestUniversalStates:
    """Import and validate all 4 universal business agent state TypedDicts."""

    def test_all_universal_states_importable(self):
        from core.agents.state import (
            ContractManagerAgentState,
            SupportAgentState,
            CompetitiveIntelAgentState,
            ReportingAgentState,
        )
        states = [
            ContractManagerAgentState,
            SupportAgentState,
            CompetitiveIntelAgentState,
            ReportingAgentState,
        ]
        for state_cls in states:
            assert state_cls is not None

    def test_all_universal_states_have_annotations(self):
        from core.agents.state import (
            ContractManagerAgentState,
            SupportAgentState,
            CompetitiveIntelAgentState,
            ReportingAgentState,
        )
        states = [
            ContractManagerAgentState,
            SupportAgentState,
            CompetitiveIntelAgentState,
            ReportingAgentState,
        ]
        for state_cls in states:
            assert hasattr(state_cls, "__annotations__"), (
                f"{state_cls.__name__} missing __annotations__"
            )
            assert len(state_cls.__annotations__) > 0

    def test_contract_state_fields(self):
        from core.agents.state import ContractManagerAgentState
        hints = get_type_hints(ContractManagerAgentState)
        for field in [
            "expiring_contracts", "renewal_window_days", "draft_contract",
            "contract_type", "contract_saved", "signature_status",
        ]:
            assert field in hints, f"ContractManagerAgentState missing '{field}'"

    def test_support_state_fields(self):
        from core.agents.state import SupportAgentState
        hints = get_type_hints(SupportAgentState)
        for field in [
            "ticket_id", "ticket_subject", "category", "priority",
            "sentiment", "escalation_needed", "draft_response",
        ]:
            assert field in hints, f"SupportAgentState missing '{field}'"

    def test_competitive_intel_state_fields(self):
        from core.agents.state import CompetitiveIntelAgentState
        hints = get_type_hints(CompetitiveIntelAgentState)
        for field in [
            "monitored_competitors", "intel_findings", "threat_score",
            "alerts", "alerts_sent", "intel_saved",
        ]:
            assert field in hints, f"CompetitiveIntelAgentState missing '{field}'"


# ══════════════════════════════════════════════════════════════════════
# 5. MCP Tool Stubs
# ══════════════════════════════════════════════════════════════════════


class TestMeshTools:
    """Test mesh_tools module stub functions."""

    def test_module_importable(self):
        from core.mcp.tools import mesh_tools
        assert mesh_tools is not None

    def test_analyze_mesh_exists(self):
        from core.mcp.tools.mesh_tools import analyze_mesh
        assert callable(analyze_mesh)

    def test_repair_mesh_exists(self):
        from core.mcp.tools.mesh_tools import repair_mesh
        assert callable(repair_mesh)

    def test_check_manifold_exists(self):
        from core.mcp.tools.mesh_tools import check_manifold
        assert callable(check_manifold)

    def test_compute_volume_exists(self):
        from core.mcp.tools.mesh_tools import compute_volume
        assert callable(compute_volume)

    def test_all_functions_are_async(self):
        from core.mcp.tools.mesh_tools import (
            analyze_mesh,
            repair_mesh,
            check_manifold,
            compute_volume,
        )
        assert inspect.iscoroutinefunction(analyze_mesh)
        assert inspect.iscoroutinefunction(repair_mesh)
        assert inspect.iscoroutinefunction(check_manifold)
        assert inspect.iscoroutinefunction(compute_volume)


class TestPrinterTools:
    """Test printer_tools module stub functions."""

    def test_module_importable(self):
        from core.mcp.tools import printer_tools
        assert printer_tools is not None

    def test_list_printers_exists(self):
        from core.mcp.tools.printer_tools import list_printers
        assert callable(list_printers)

    def test_get_printer_status_exists(self):
        from core.mcp.tools.printer_tools import get_printer_status
        assert callable(get_printer_status)

    def test_start_print_job_exists(self):
        from core.mcp.tools.printer_tools import start_print_job
        assert callable(start_print_job)

    def test_get_print_progress_exists(self):
        from core.mcp.tools.printer_tools import get_print_progress
        assert callable(get_print_progress)

    def test_all_functions_are_async(self):
        from core.mcp.tools.printer_tools import (
            list_printers,
            get_printer_status,
            start_print_job,
            get_print_progress,
        )
        assert inspect.iscoroutinefunction(list_printers)
        assert inspect.iscoroutinefunction(get_printer_status)
        assert inspect.iscoroutinefunction(start_print_job)
        assert inspect.iscoroutinefunction(get_print_progress)


class TestShippingTools:
    """Test shipping_tools module stub functions."""

    def test_module_importable(self):
        from core.mcp.tools import shipping_tools
        assert shipping_tools is not None

    def test_get_shipping_rates_exists(self):
        from core.mcp.tools.shipping_tools import get_shipping_rates
        assert callable(get_shipping_rates)

    def test_create_shipping_label_exists(self):
        from core.mcp.tools.shipping_tools import create_shipping_label
        assert callable(create_shipping_label)

    def test_track_shipment_exists(self):
        from core.mcp.tools.shipping_tools import track_shipment
        assert callable(track_shipment)

    def test_all_functions_are_async(self):
        from core.mcp.tools.shipping_tools import (
            get_shipping_rates,
            create_shipping_label,
            track_shipment,
        )
        assert inspect.iscoroutinefunction(get_shipping_rates)
        assert inspect.iscoroutinefunction(create_shipping_label)
        assert inspect.iscoroutinefunction(track_shipment)


class TestMeasurementTools:
    """Test measurement_tools module stub functions."""

    def test_module_importable(self):
        from core.mcp.tools import measurement_tools
        assert measurement_tools is not None

    def test_check_dimensional_accuracy_exists(self):
        from core.mcp.tools.measurement_tools import check_dimensional_accuracy
        assert callable(check_dimensional_accuracy)

    def test_compute_surface_quality_exists(self):
        from core.mcp.tools.measurement_tools import compute_surface_quality
        assert callable(compute_surface_quality)

    def test_compare_geometries_exists(self):
        from core.mcp.tools.measurement_tools import compare_geometries
        assert callable(compare_geometries)

    def test_all_functions_are_async(self):
        from core.mcp.tools.measurement_tools import (
            check_dimensional_accuracy,
            compute_surface_quality,
            compare_geometries,
        )
        assert inspect.iscoroutinefunction(check_dimensional_accuracy)
        assert inspect.iscoroutinefunction(compute_surface_quality)
        assert inspect.iscoroutinefunction(compare_geometries)


# ══════════════════════════════════════════════════════════════════════
# 6. YAML Configs
# ══════════════════════════════════════════════════════════════════════


class TestPrintDomainYAMLConfigs:
    """Validate PrintBiz domain YAML configs (10 files)."""

    # ── Existence ──────────────────────────────────────────────────

    def test_print_domain_yamls_exist(self):
        for filename in PRINT_DOMAIN_YAMLS:
            path = PRINT_AGENTS_DIR / filename
            assert path.exists(), f"Missing YAML config: {filename}"

    def test_print_domain_yamls_parseable(self):
        for filename in PRINT_DOMAIN_YAMLS:
            cfg = _load_yaml(PRINT_AGENTS_DIR, filename)
            assert isinstance(cfg, dict), f"Invalid YAML in {filename}"

    # ── Required Keys ──────────────────────────────────────────────

    def test_yaml_has_agent_id(self):
        for filename in PRINT_DOMAIN_YAMLS:
            cfg = _load_yaml(PRINT_AGENTS_DIR, filename)
            assert "agent_id" in cfg, f"Missing agent_id in {filename}"
            assert isinstance(cfg["agent_id"], str)
            assert len(cfg["agent_id"]) > 0

    def test_yaml_has_agent_type(self):
        for filename in PRINT_DOMAIN_YAMLS:
            cfg = _load_yaml(PRINT_AGENTS_DIR, filename)
            assert "agent_type" in cfg, f"Missing agent_type in {filename}"
            assert isinstance(cfg["agent_type"], str)

    def test_yaml_has_name(self):
        for filename in PRINT_DOMAIN_YAMLS:
            cfg = _load_yaml(PRINT_AGENTS_DIR, filename)
            assert "name" in cfg, f"Missing name in {filename}"
            assert isinstance(cfg["name"], str)
            assert len(cfg["name"]) > 0

    def test_yaml_has_human_gates(self):
        for filename in PRINT_DOMAIN_YAMLS:
            cfg = _load_yaml(PRINT_AGENTS_DIR, filename)
            assert "human_gates" in cfg, f"Missing human_gates in {filename}"
            assert cfg["human_gates"]["enabled"] is True

    # ── Model Routing ─────────────────────────────────────────────

    def test_print_domain_yaml_has_model_routing(self):
        for filename in PRINT_DOMAIN_YAMLS:
            cfg = _load_yaml(PRINT_AGENTS_DIR, filename)
            assert "model_routing" in cfg, (
                f"{filename} should have model_routing"
            )

    # ── Agent type values ──────────────────────────────────────────

    def test_yaml_agent_types_match_expected(self):
        expected_types = [
            "file_analyst",
            "mesh_repair",
            "scale_optimizer",
            "material_advisor",
            "quote_engine",
            "print_manager",
            "post_process",
            "qc_inspector",
            "cad_advisor",
            "logistics",
        ]
        for filename, expected_type in zip(PRINT_DOMAIN_YAMLS, expected_types):
            cfg = _load_yaml(PRINT_AGENTS_DIR, filename)
            assert cfg["agent_type"] == expected_type, (
                f"{filename}: expected agent_type={expected_type}, got {cfg['agent_type']}"
            )

    # ── Uniqueness ─────────────────────────────────────────────────

    def test_all_agent_ids_unique(self):
        agent_ids = []
        for filename in PRINT_DOMAIN_YAMLS:
            cfg = _load_yaml(PRINT_AGENTS_DIR, filename)
            agent_ids.append(cfg["agent_id"])
        assert len(agent_ids) == len(set(agent_ids)), "Duplicate agent_id found in print domain"


class TestPrintUniversalYAMLConfigs:
    """Validate PrintBiz universal YAML configs (4 files)."""

    def test_print_universal_yamls_exist(self):
        for filename in PRINT_UNIVERSAL_YAMLS:
            path = PRINT_AGENTS_DIR / filename
            assert path.exists(), f"Missing YAML config: {filename}"

    def test_yaml_has_agent_id(self):
        for filename in PRINT_UNIVERSAL_YAMLS:
            cfg = _load_yaml(PRINT_AGENTS_DIR, filename)
            assert "agent_id" in cfg, f"Missing agent_id in {filename}"

    def test_yaml_has_agent_type(self):
        for filename in PRINT_UNIVERSAL_YAMLS:
            cfg = _load_yaml(PRINT_AGENTS_DIR, filename)
            assert "agent_type" in cfg, f"Missing agent_type in {filename}"

    def test_yaml_has_name(self):
        for filename in PRINT_UNIVERSAL_YAMLS:
            cfg = _load_yaml(PRINT_AGENTS_DIR, filename)
            assert "name" in cfg, f"Missing name in {filename}"

    def test_yaml_has_human_gates(self):
        for filename in PRINT_UNIVERSAL_YAMLS:
            cfg = _load_yaml(PRINT_AGENTS_DIR, filename)
            assert "human_gates" in cfg, f"Missing human_gates in {filename}"
            assert cfg["human_gates"]["enabled"] is True

    def test_yaml_has_model_routing(self):
        for filename in PRINT_UNIVERSAL_YAMLS:
            cfg = _load_yaml(PRINT_AGENTS_DIR, filename)
            assert "model_routing" in cfg, (
                f"{filename} should have model_routing"
            )

    def test_yaml_agent_types_match_expected(self):
        expected_types = [
            "contract_manager",
            "support_agent",
            "competitive_intel",
            "reporting",
        ]
        for filename, expected_type in zip(PRINT_UNIVERSAL_YAMLS, expected_types):
            cfg = _load_yaml(PRINT_AGENTS_DIR, filename)
            assert cfg["agent_type"] == expected_type, (
                f"{filename}: expected agent_type={expected_type}, got {cfg['agent_type']}"
            )


class TestGuardUniversalYAMLConfigs:
    """Validate Enclave Guard universal YAML configs (4 files)."""

    def test_guard_universal_yamls_exist(self):
        for filename in GUARD_UNIVERSAL_YAMLS:
            path = GUARD_AGENTS_DIR / filename
            assert path.exists(), f"Missing YAML config: {filename}"

    def test_yaml_has_agent_id(self):
        for filename in GUARD_UNIVERSAL_YAMLS:
            cfg = _load_yaml(GUARD_AGENTS_DIR, filename)
            assert "agent_id" in cfg, f"Missing agent_id in {filename}"

    def test_yaml_has_agent_type(self):
        for filename in GUARD_UNIVERSAL_YAMLS:
            cfg = _load_yaml(GUARD_AGENTS_DIR, filename)
            assert "agent_type" in cfg, f"Missing agent_type in {filename}"

    def test_yaml_has_name(self):
        for filename in GUARD_UNIVERSAL_YAMLS:
            cfg = _load_yaml(GUARD_AGENTS_DIR, filename)
            assert "name" in cfg, f"Missing name in {filename}"

    def test_yaml_has_human_gates(self):
        for filename in GUARD_UNIVERSAL_YAMLS:
            cfg = _load_yaml(GUARD_AGENTS_DIR, filename)
            assert "human_gates" in cfg, f"Missing human_gates in {filename}"
            assert cfg["human_gates"]["enabled"] is True

    def test_guard_universal_yaml_has_model_routing(self):
        for filename in GUARD_UNIVERSAL_YAMLS:
            cfg = _load_yaml(GUARD_AGENTS_DIR, filename)
            assert "model_routing" in cfg, (
                f"{filename} should have model_routing"
            )

    def test_yaml_agent_types_match_expected(self):
        expected_types = [
            "contract_manager",
            "support_agent",
            "competitive_intel",
            "reporting",
        ]
        for filename, expected_type in zip(GUARD_UNIVERSAL_YAMLS, expected_types):
            cfg = _load_yaml(GUARD_AGENTS_DIR, filename)
            assert cfg["agent_type"] == expected_type, (
                f"{filename}: expected agent_type={expected_type}, got {cfg['agent_type']}"
            )


# ══════════════════════════════════════════════════════════════════════
# 7. Pydantic Contracts
# ══════════════════════════════════════════════════════════════════════


class TestContracts:
    """Validate new Pydantic contract models from Phase 19-20."""

    def test_print_job_request_defaults(self):
        from core.agents.contracts import PrintJobRequest
        req = PrintJobRequest(file_name="model.stl")
        assert req.file_name == "model.stl"
        assert req.file_format == "STL"
        assert req.file_url == ""
        assert req.quantity == 1
        assert req.notes == ""
        assert req.metadata == {}

    def test_geometry_analysis_defaults(self):
        from core.agents.contracts import GeometryAnalysis
        ga = GeometryAnalysis()
        assert ga.is_manifold is True
        assert ga.is_watertight is True
        assert ga.vertex_count == 0
        assert ga.face_count == 0
        assert ga.volume_cm3 == 0.0
        assert ga.surface_area_cm2 == 0.0
        assert ga.bounding_box == {}
        assert ga.issues == []
        assert ga.printability_score == 0.0

    def test_material_recommendation_required(self):
        from core.agents.contracts import MaterialRecommendation
        rec = MaterialRecommendation(material="PLA", technology="FDM")
        assert rec.material == "PLA"
        assert rec.technology == "FDM"
        assert rec.cost_per_cm3 == 0.0
        assert rec.layer_height_um == 200
        assert rec.detail_level == "medium"
        assert rec.alternatives == []

    def test_print_quote_defaults(self):
        from core.agents.contracts import PrintQuote
        quote = PrintQuote()
        assert quote.quote_id == ""
        assert quote.total_cents == 0
        assert quote.line_items == []
        assert quote.estimated_days == 5
        assert quote.valid_until == ""
        assert quote.company_name == ""
        assert quote.contact_email == ""

    def test_contract_request_defaults(self):
        from core.agents.contracts import ContractRequest
        req = ContractRequest()
        assert req.contract_type == "service_agreement"
        assert req.company_name == ""
        assert req.contact_email == ""
        assert req.value_cents == 0
        assert req.start_date == ""
        assert req.duration_months == 12
        assert req.custom_terms == []
        assert req.metadata == {}

    def test_support_ticket_required_field(self):
        from core.agents.contracts import SupportTicketData
        ticket = SupportTicketData(subject="Can't login")
        assert ticket.subject == "Can't login"
        assert ticket.description == ""
        assert ticket.contact_email == ""
        assert ticket.category == "general"
        assert ticket.priority == "medium"
        assert ticket.metadata == {}

    def test_competitor_alert_defaults(self):
        from core.agents.contracts import CompetitorAlert
        alert = CompetitorAlert(competitor_name="Rival Inc")
        assert alert.competitor_name == "Rival Inc"
        assert alert.intel_type == "news"
        assert alert.title == ""
        assert alert.content == ""
        assert alert.source_url == ""
        assert alert.severity == "info"
        assert alert.actionable is False
        assert alert.metadata == {}
