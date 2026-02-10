"""
Tests for Domain Expert Agent Infrastructure — Phase 18.

Covers:
    - Migration 014_domain_agents.sql schema validation (tables, indexes, RPC)
    - All 10 domain agent state TypedDicts (import + field validation)
    - 4 MCP tool stub modules (ssl_scan, http_scanner, dns, compliance)
    - 10 YAML configs for Enclave Guard domain agents
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

MIGRATION_PATH = (
    Path(__file__).parent.parent.parent
    / "infrastructure"
    / "migrations"
    / "014_domain_agents.sql"
)

AGENTS_DIR = (
    Path(__file__).parent.parent.parent
    / "verticals"
    / "enclave_guard"
    / "agents"
)

DOMAIN_YAML_FILES = [
    "vuln_scanner.yaml",
    "network_analyst.yaml",
    "appsec_reviewer.yaml",
    "compliance_mapper.yaml",
    "risk_reporter.yaml",
    "iam_analyst.yaml",
    "incident_readiness.yaml",
    "cloud_security.yaml",
    "security_trainer.yaml",
    "remediation_guide.yaml",
]


def _read_migration() -> str:
    with open(MIGRATION_PATH) as f:
        return f.read()


def _load_yaml(filename: str) -> dict[str, Any]:
    with open(AGENTS_DIR / filename) as f:
        return yaml.safe_load(f)


# ══════════════════════════════════════════════════════════════════════
# 1. Migration 014 Schema Tests
# ══════════════════════════════════════════════════════════════════════


class TestMigration014:
    """Validate 014_domain_agents.sql migration structure."""

    def test_migration_file_exists(self):
        assert MIGRATION_PATH.exists(), "014_domain_agents.sql not found"

    def test_migration_not_empty(self):
        sql = _read_migration()
        assert len(sql) > 100

    # ── Tables ─────────────────────────────────────────────────────

    def test_creates_security_assessments_table(self):
        sql = _read_migration()
        assert "CREATE TABLE" in sql
        assert "security_assessments" in sql

    def test_creates_security_findings_table(self):
        sql = _read_migration()
        assert "security_findings" in sql

    def test_creates_remediation_tasks_table(self):
        sql = _read_migration()
        assert "remediation_tasks" in sql

    # ── security_assessments columns ───────────────────────────────

    def test_security_assessments_columns(self):
        sql = _read_migration()
        expected_cols = [
            "vertical_id",
            "company_id",
            "contact_id",
            "opportunity_id",
            "assessment_type",
            "status",
            "findings",
            "risk_score",
            "executive_summary",
            "remediation_plan",
            "frameworks_checked",
            "metadata",
            "started_at",
            "completed_at",
            "created_at",
            "updated_at",
        ]
        for col in expected_cols:
            assert col in sql, f"Column '{col}' missing from migration"

    def test_security_assessments_type_check(self):
        sql = _read_migration()
        for atype in [
            "vulnerability_scan",
            "network_analysis",
            "app_security",
            "compliance_mapping",
            "cloud_security",
            "iam_review",
            "incident_readiness",
            "full_assessment",
        ]:
            assert atype in sql, f"assessment_type '{atype}' not in CHECK"

    def test_security_assessments_status_check(self):
        sql = _read_migration()
        for status in ["pending", "in_progress", "completed", "failed", "cancelled"]:
            assert status in sql, f"assessment status '{status}' not in CHECK"

    # ── security_findings columns ──────────────────────────────────

    def test_security_findings_columns(self):
        sql = _read_migration()
        expected_cols = [
            "assessment_id",
            "vertical_id",
            "severity",
            "category",
            "title",
            "description",
            "recommendation",
            "affected_asset",
            "cve_id",
            "cvss_score",
            "status",
            "remediated_at",
            "verified_at",
            "evidence",
            "metadata",
        ]
        for col in expected_cols:
            assert col in sql, f"Column '{col}' missing from findings table"

    def test_security_findings_severity_check(self):
        sql = _read_migration()
        for sev in ["critical", "high", "medium", "low", "informational"]:
            assert sev in sql, f"severity '{sev}' not in CHECK"

    def test_security_findings_category_check(self):
        sql = _read_migration()
        for cat in [
            "network",
            "application",
            "infrastructure",
            "access_control",
            "compliance",
            "cloud",
            "human",
            "physical",
        ]:
            assert cat in sql, f"category '{cat}' not in CHECK"

    def test_security_findings_status_check(self):
        sql = _read_migration()
        for status in [
            "open",
            "in_progress",
            "remediated",
            "accepted",
            "false_positive",
        ]:
            assert status in sql, f"finding status '{status}' not in CHECK"

    # ── remediation_tasks columns ──────────────────────────────────

    def test_remediation_tasks_columns(self):
        sql = _read_migration()
        expected_cols = [
            "finding_id",
            "vertical_id",
            "title",
            "description",
            "priority",
            "status",
            "assigned_to",
            "due_date",
            "verification_method",
            "verified_at",
            "metadata",
        ]
        for col in expected_cols:
            assert col in sql, f"Column '{col}' missing from remediation_tasks"

    def test_remediation_tasks_priority_check(self):
        sql = _read_migration()
        for pri in ["critical", "high", "medium", "low"]:
            assert pri in sql

    def test_remediation_tasks_status_check(self):
        sql = _read_migration()
        for status in [
            "pending",
            "assigned",
            "in_progress",
            "completed",
            "verified",
            "deferred",
        ]:
            assert status in sql, f"remediation status '{status}' not in CHECK"

    # ── Indexes ────────────────────────────────────────────────────

    def test_index_assessments_vertical_status(self):
        sql = _read_migration()
        assert "idx_security_assessments_vertical_status" in sql

    def test_index_assessments_company(self):
        sql = _read_migration()
        assert "idx_security_assessments_company" in sql

    def test_index_assessments_type(self):
        sql = _read_migration()
        assert "idx_security_assessments_type" in sql

    def test_index_findings_assessment_severity(self):
        sql = _read_migration()
        assert "idx_security_findings_assessment_severity" in sql

    def test_index_findings_vertical_status(self):
        sql = _read_migration()
        assert "idx_security_findings_vertical_status" in sql

    def test_index_findings_category(self):
        sql = _read_migration()
        assert "idx_security_findings_category" in sql

    def test_index_remediation_finding(self):
        sql = _read_migration()
        assert "idx_remediation_tasks_finding" in sql

    def test_index_remediation_vertical_status(self):
        sql = _read_migration()
        assert "idx_remediation_tasks_vertical_status" in sql

    # ── RPC Function ───────────────────────────────────────────────

    def test_rpc_function_get_assessment_summary(self):
        sql = _read_migration()
        assert "get_assessment_summary" in sql

    def test_rpc_function_returns_jsonb(self):
        sql = _read_migration()
        assert "RETURNS JSONB" in sql

    def test_rpc_function_security_definer(self):
        sql = _read_migration()
        assert "SECURITY DEFINER" in sql


# ══════════════════════════════════════════════════════════════════════
# 2. Domain Agent State TypedDicts
# ══════════════════════════════════════════════════════════════════════


class TestDomainAgentStates:
    """Import and validate all 10 domain agent state TypedDicts."""

    # ── VulnScannerAgentState ──────────────────────────────────────

    def test_import_vuln_scanner_state(self):
        from core.agents.state import VulnScannerAgentState
        assert VulnScannerAgentState is not None

    def test_vuln_scanner_state_key_fields(self):
        from core.agents.state import VulnScannerAgentState
        hints = get_type_hints(VulnScannerAgentState)
        for field in ["company_domain", "scan_type", "ssl_findings", "risk_score", "assessment_id"]:
            assert field in hints, f"VulnScannerAgentState missing '{field}'"

    def test_vuln_scanner_state_minimal(self):
        from core.agents.state import VulnScannerAgentState
        state: VulnScannerAgentState = {"agent_id": "vuln_scanner_v1", "vertical_id": "enclave_guard"}
        assert state["agent_id"] == "vuln_scanner_v1"

    # ── NetworkAnalystAgentState ───────────────────────────────────

    def test_import_network_analyst_state(self):
        from core.agents.state import NetworkAnalystAgentState
        assert NetworkAnalystAgentState is not None

    def test_network_analyst_state_key_fields(self):
        from core.agents.state import NetworkAnalystAgentState
        hints = get_type_hints(NetworkAnalystAgentState)
        for field in ["exposed_services", "open_ports", "dns_records", "subdomains", "attack_surface_score"]:
            assert field in hints, f"NetworkAnalystAgentState missing '{field}'"

    # ── AppSecReviewerAgentState ───────────────────────────────────

    def test_import_appsec_reviewer_state(self):
        from core.agents.state import AppSecReviewerAgentState
        assert AppSecReviewerAgentState is not None

    def test_appsec_reviewer_state_key_fields(self):
        from core.agents.state import AppSecReviewerAgentState
        hints = get_type_hints(AppSecReviewerAgentState)
        for field in ["header_analysis", "csp_analysis", "cookie_analysis", "cors_analysis", "tls_analysis"]:
            assert field in hints, f"AppSecReviewerAgentState missing '{field}'"

    # ── ComplianceMapperAgentState ─────────────────────────────────

    def test_import_compliance_mapper_state(self):
        from core.agents.state import ComplianceMapperAgentState
        assert ComplianceMapperAgentState is not None

    def test_compliance_mapper_state_key_fields(self):
        from core.agents.state import ComplianceMapperAgentState
        hints = get_type_hints(ComplianceMapperAgentState)
        for field in ["target_frameworks", "compliance_gaps", "compliance_score", "remediation_roadmap"]:
            assert field in hints, f"ComplianceMapperAgentState missing '{field}'"

    # ── RiskReporterAgentState ─────────────────────────────────────

    def test_import_risk_reporter_state(self):
        from core.agents.state import RiskReporterAgentState
        assert RiskReporterAgentState is not None

    def test_risk_reporter_state_key_fields(self):
        from core.agents.state import RiskReporterAgentState
        hints = get_type_hints(RiskReporterAgentState)
        for field in ["overall_risk_score", "estimated_breach_cost", "executive_summary", "risk_matrix"]:
            assert field in hints, f"RiskReporterAgentState missing '{field}'"

    # ── IAMAnalystAgentState ───────────────────────────────────────

    def test_import_iam_analyst_state(self):
        from core.agents.state import IAMAnalystAgentState
        assert IAMAnalystAgentState is not None

    def test_iam_analyst_state_key_fields(self):
        from core.agents.state import IAMAnalystAgentState
        hints = get_type_hints(IAMAnalystAgentState)
        for field in ["mfa_status", "password_policy", "privileged_accounts", "iam_findings", "iam_risk_score"]:
            assert field in hints, f"IAMAnalystAgentState missing '{field}'"

    # ── IncidentReadinessAgentState ────────────────────────────────

    def test_import_incident_readiness_state(self):
        from core.agents.state import IncidentReadinessAgentState
        assert IncidentReadinessAgentState is not None

    def test_incident_readiness_state_key_fields(self):
        from core.agents.state import IncidentReadinessAgentState
        hints = get_type_hints(IncidentReadinessAgentState)
        for field in ["ir_plan_exists", "readiness_score", "readiness_grade", "gaps", "recommendations"]:
            assert field in hints, f"IncidentReadinessAgentState missing '{field}'"

    # ── CloudSecurityAgentState ────────────────────────────────────

    def test_import_cloud_security_state(self):
        from core.agents.state import CloudSecurityAgentState
        assert CloudSecurityAgentState is not None

    def test_cloud_security_state_key_fields(self):
        from core.agents.state import CloudSecurityAgentState
        hints = get_type_hints(CloudSecurityAgentState)
        for field in ["cloud_provider", "s3_buckets", "misconfigurations", "cloud_findings", "cloud_risk_score"]:
            assert field in hints, f"CloudSecurityAgentState missing '{field}'"

    # ── SecurityTrainerAgentState ──────────────────────────────────

    def test_import_security_trainer_state(self):
        from core.agents.state import SecurityTrainerAgentState
        assert SecurityTrainerAgentState is not None

    def test_security_trainer_state_key_fields(self):
        from core.agents.state import SecurityTrainerAgentState
        hints = get_type_hints(SecurityTrainerAgentState)
        for field in ["training_modules", "phishing_scenarios", "human_risk_score", "content_approved"]:
            assert field in hints, f"SecurityTrainerAgentState missing '{field}'"

    # ── RemediationGuideAgentState ─────────────────────────────────

    def test_import_remediation_guide_state(self):
        from core.agents.state import RemediationGuideAgentState
        assert RemediationGuideAgentState is not None

    def test_remediation_guide_state_key_fields(self):
        from core.agents.state import RemediationGuideAgentState
        hints = get_type_hints(RemediationGuideAgentState)
        for field in ["open_findings", "remediation_steps", "tasks_created", "findings_remediated"]:
            assert field in hints, f"RemediationGuideAgentState missing '{field}'"

    # ── BaseAgentState inheritance ─────────────────────────────────

    def test_all_states_inherit_base_fields(self):
        from core.agents.state import (
            VulnScannerAgentState,
            NetworkAnalystAgentState,
            AppSecReviewerAgentState,
            ComplianceMapperAgentState,
            RiskReporterAgentState,
            IAMAnalystAgentState,
            IncidentReadinessAgentState,
            CloudSecurityAgentState,
            SecurityTrainerAgentState,
            RemediationGuideAgentState,
        )
        states = [
            VulnScannerAgentState,
            NetworkAnalystAgentState,
            AppSecReviewerAgentState,
            ComplianceMapperAgentState,
            RiskReporterAgentState,
            IAMAnalystAgentState,
            IncidentReadinessAgentState,
            CloudSecurityAgentState,
            SecurityTrainerAgentState,
            RemediationGuideAgentState,
        ]
        base_fields = ["agent_id", "vertical_id", "run_id", "current_node", "error"]
        for state_cls in states:
            hints = get_type_hints(state_cls)
            for field in base_fields:
                assert field in hints, (
                    f"{state_cls.__name__} missing base field '{field}'"
                )

    def test_all_states_are_typed_dicts(self):
        from core.agents.state import (
            VulnScannerAgentState,
            NetworkAnalystAgentState,
            AppSecReviewerAgentState,
            ComplianceMapperAgentState,
            RiskReporterAgentState,
            IAMAnalystAgentState,
            IncidentReadinessAgentState,
            CloudSecurityAgentState,
            SecurityTrainerAgentState,
            RemediationGuideAgentState,
        )
        states = [
            VulnScannerAgentState,
            NetworkAnalystAgentState,
            AppSecReviewerAgentState,
            ComplianceMapperAgentState,
            RiskReporterAgentState,
            IAMAnalystAgentState,
            IncidentReadinessAgentState,
            CloudSecurityAgentState,
            SecurityTrainerAgentState,
            RemediationGuideAgentState,
        ]
        for state_cls in states:
            # TypedDict classes have __annotations__
            assert hasattr(state_cls, "__annotations__"), (
                f"{state_cls.__name__} is not a TypedDict"
            )


# ══════════════════════════════════════════════════════════════════════
# 3. SSL Scan MCP Tools
# ══════════════════════════════════════════════════════════════════════


class TestSSLScanTools:
    """Test ssl_scan_tools module stub functions."""

    def test_import_module(self):
        from core.mcp.tools import ssl_scan_tools
        assert ssl_scan_tools is not None

    def test_scan_ssl_certificate_exists(self):
        from core.mcp.tools.ssl_scan_tools import scan_ssl_certificate
        assert callable(scan_ssl_certificate)

    def test_check_ssl_protocols_exists(self):
        from core.mcp.tools.ssl_scan_tools import check_ssl_protocols
        assert callable(check_ssl_protocols)

    def test_get_ssl_grade_exists(self):
        from core.mcp.tools.ssl_scan_tools import get_ssl_grade
        assert callable(get_ssl_grade)

    def test_scan_ssl_certificate_is_async(self):
        from core.mcp.tools.ssl_scan_tools import scan_ssl_certificate
        assert inspect.iscoroutinefunction(scan_ssl_certificate)

    def test_check_ssl_protocols_is_async(self):
        from core.mcp.tools.ssl_scan_tools import check_ssl_protocols
        assert inspect.iscoroutinefunction(check_ssl_protocols)

    def test_get_ssl_grade_is_async(self):
        from core.mcp.tools.ssl_scan_tools import get_ssl_grade
        assert inspect.iscoroutinefunction(get_ssl_grade)

    @pytest.mark.asyncio
    async def test_scan_ssl_certificate_returns_json(self):
        from core.mcp.tools.ssl_scan_tools import scan_ssl_certificate
        result = await scan_ssl_certificate("example.com")
        data = json.loads(result)
        assert data["status"] == "success"
        assert data["domain"] == "example.com"
        assert "certificate" in data
        assert data["certificate"]["is_expired"] is False

    @pytest.mark.asyncio
    async def test_check_ssl_protocols_returns_json(self):
        from core.mcp.tools.ssl_scan_tools import check_ssl_protocols
        result = await check_ssl_protocols("example.com")
        data = json.loads(result)
        assert data["status"] == "success"
        assert "protocols" in data
        assert data["protocols"]["TLS_1.3"]["supported"] is True
        assert data["supports_forward_secrecy"] is True

    @pytest.mark.asyncio
    async def test_get_ssl_grade_returns_json(self):
        from core.mcp.tools.ssl_scan_tools import get_ssl_grade
        result = await get_ssl_grade("example.com")
        data = json.loads(result)
        assert data["status"] == "success"
        assert data["overall_grade"] in ["A+", "A", "A-", "B", "C", "D", "F"]
        assert "category_scores" in data

    @pytest.mark.asyncio
    async def test_scan_ssl_certificate_custom_port(self):
        from core.mcp.tools.ssl_scan_tools import scan_ssl_certificate
        result = await scan_ssl_certificate("example.com", port=8443)
        data = json.loads(result)
        assert data["port"] == 8443


# ══════════════════════════════════════════════════════════════════════
# 4. HTTP Scanner MCP Tools
# ══════════════════════════════════════════════════════════════════════


class TestHTTPScannerTools:
    """Test http_scanner_tools module stub functions."""

    def test_import_module(self):
        from core.mcp.tools import http_scanner_tools
        assert http_scanner_tools is not None

    def test_scan_security_headers_exists(self):
        from core.mcp.tools.http_scanner_tools import scan_security_headers
        assert callable(scan_security_headers)

    def test_check_cors_policy_exists(self):
        from core.mcp.tools.http_scanner_tools import check_cors_policy
        assert callable(check_cors_policy)

    def test_analyze_cookie_security_exists(self):
        from core.mcp.tools.http_scanner_tools import analyze_cookie_security
        assert callable(analyze_cookie_security)

    def test_check_csp_policy_exists(self):
        from core.mcp.tools.http_scanner_tools import check_csp_policy
        assert callable(check_csp_policy)

    def test_all_functions_are_async(self):
        from core.mcp.tools.http_scanner_tools import (
            scan_security_headers,
            check_cors_policy,
            analyze_cookie_security,
            check_csp_policy,
        )
        assert inspect.iscoroutinefunction(scan_security_headers)
        assert inspect.iscoroutinefunction(check_cors_policy)
        assert inspect.iscoroutinefunction(analyze_cookie_security)
        assert inspect.iscoroutinefunction(check_csp_policy)

    def test_security_headers_constant(self):
        from core.mcp.tools.http_scanner_tools import SECURITY_HEADERS
        assert isinstance(SECURITY_HEADERS, dict)
        assert "Strict-Transport-Security" in SECURITY_HEADERS
        assert "Content-Security-Policy" in SECURITY_HEADERS
        assert "X-Content-Type-Options" in SECURITY_HEADERS

    @pytest.mark.asyncio
    async def test_scan_security_headers_returns_json(self):
        from core.mcp.tools.http_scanner_tools import scan_security_headers
        result = await scan_security_headers("https://example.com")
        data = json.loads(result)
        assert data["status"] == "success"
        assert "headers" in data
        assert "summary" in data
        assert data["summary"]["total"] > 0

    @pytest.mark.asyncio
    async def test_check_cors_policy_returns_json(self):
        from core.mcp.tools.http_scanner_tools import check_cors_policy
        result = await check_cors_policy("https://api.example.com")
        data = json.loads(result)
        assert data["status"] == "success"
        assert "cors_enabled" in data
        assert "policy" in data
        assert "risk_level" in data

    @pytest.mark.asyncio
    async def test_analyze_cookie_security_returns_json(self):
        from core.mcp.tools.http_scanner_tools import analyze_cookie_security
        result = await analyze_cookie_security("https://example.com")
        data = json.loads(result)
        assert data["status"] == "success"
        assert "cookies" in data
        assert isinstance(data["cookies"], list)
        assert "summary" in data

    @pytest.mark.asyncio
    async def test_check_csp_policy_returns_json(self):
        from core.mcp.tools.http_scanner_tools import check_csp_policy
        result = await check_csp_policy("https://example.com")
        data = json.loads(result)
        assert data["status"] == "success"
        assert data["csp_present"] is True
        assert "directives" in data
        assert "grade" in data


# ══════════════════════════════════════════════════════════════════════
# 5. DNS MCP Tools
# ══════════════════════════════════════════════════════════════════════


class TestDNSTools:
    """Test dns_tools module stub functions."""

    def test_import_module(self):
        from core.mcp.tools import dns_tools
        assert dns_tools is not None

    def test_enumerate_dns_records_exists(self):
        from core.mcp.tools.dns_tools import enumerate_dns_records
        assert callable(enumerate_dns_records)

    def test_find_subdomains_exists(self):
        from core.mcp.tools.dns_tools import find_subdomains
        assert callable(find_subdomains)

    def test_check_dnssec_exists(self):
        from core.mcp.tools.dns_tools import check_dnssec
        assert callable(check_dnssec)

    def test_check_spf_dmarc_exists(self):
        from core.mcp.tools.dns_tools import check_spf_dmarc
        assert callable(check_spf_dmarc)

    def test_all_functions_are_async(self):
        from core.mcp.tools.dns_tools import (
            enumerate_dns_records,
            find_subdomains,
            check_dnssec,
            check_spf_dmarc,
        )
        assert inspect.iscoroutinefunction(enumerate_dns_records)
        assert inspect.iscoroutinefunction(find_subdomains)
        assert inspect.iscoroutinefunction(check_dnssec)
        assert inspect.iscoroutinefunction(check_spf_dmarc)

    @pytest.mark.asyncio
    async def test_enumerate_dns_records_returns_json(self):
        from core.mcp.tools.dns_tools import enumerate_dns_records
        result = await enumerate_dns_records("example.com")
        data = json.loads(result)
        assert data["status"] == "success"
        assert data["domain"] == "example.com"
        assert "records" in data
        assert "A" in data["records"]
        assert "MX" in data["records"]

    @pytest.mark.asyncio
    async def test_enumerate_dns_records_filter_types(self):
        from core.mcp.tools.dns_tools import enumerate_dns_records
        result = await enumerate_dns_records("example.com", record_types=["A", "MX"])
        data = json.loads(result)
        assert set(data["records"].keys()).issubset({"A", "MX"})

    @pytest.mark.asyncio
    async def test_find_subdomains_returns_json(self):
        from core.mcp.tools.dns_tools import find_subdomains
        result = await find_subdomains("example.com")
        data = json.loads(result)
        assert data["status"] == "success"
        assert "subdomains" in data
        assert data["total_found"] > 0
        assert data["live_count"] >= 0

    @pytest.mark.asyncio
    async def test_check_dnssec_returns_json(self):
        from core.mcp.tools.dns_tools import check_dnssec
        result = await check_dnssec("example.com")
        data = json.loads(result)
        assert data["status"] == "success"
        assert "dnssec_enabled" in data
        assert "validation" in data

    @pytest.mark.asyncio
    async def test_check_spf_dmarc_returns_json(self):
        from core.mcp.tools.dns_tools import check_spf_dmarc
        result = await check_spf_dmarc("example.com")
        data = json.loads(result)
        assert data["status"] == "success"
        assert "spf" in data
        assert "dmarc" in data
        assert "dkim" in data
        assert data["spf"]["present"] is True
        assert data["dmarc"]["present"] is True


# ══════════════════════════════════════════════════════════════════════
# 6. Compliance Framework MCP Tools
# ══════════════════════════════════════════════════════════════════════


class TestComplianceTools:
    """Test compliance_framework_tools module stub functions."""

    def test_import_module(self):
        from core.mcp.tools import compliance_framework_tools
        assert compliance_framework_tools is not None

    def test_get_framework_requirements_exists(self):
        from core.mcp.tools.compliance_framework_tools import get_framework_requirements
        assert callable(get_framework_requirements)

    def test_map_finding_to_controls_exists(self):
        from core.mcp.tools.compliance_framework_tools import map_finding_to_controls
        assert callable(map_finding_to_controls)

    def test_get_compliance_score_exists(self):
        from core.mcp.tools.compliance_framework_tools import get_compliance_score
        assert callable(get_compliance_score)

    def test_all_functions_are_async(self):
        from core.mcp.tools.compliance_framework_tools import (
            get_framework_requirements,
            map_finding_to_controls,
            get_compliance_score,
        )
        assert inspect.iscoroutinefunction(get_framework_requirements)
        assert inspect.iscoroutinefunction(map_finding_to_controls)
        assert inspect.iscoroutinefunction(get_compliance_score)

    @pytest.mark.asyncio
    async def test_get_framework_requirements_soc2(self):
        from core.mcp.tools.compliance_framework_tools import get_framework_requirements
        result = await get_framework_requirements("SOC2")
        data = json.loads(result)
        assert data["status"] == "success"
        assert data["framework"] == "SOC 2 Type II"
        assert data["total_controls"] > 0
        assert "categories" in data

    @pytest.mark.asyncio
    async def test_get_framework_requirements_unknown(self):
        from core.mcp.tools.compliance_framework_tools import get_framework_requirements
        result = await get_framework_requirements("UNKNOWN_FRAMEWORK")
        data = json.loads(result)
        assert data["status"] == "error"
        assert "supported_frameworks" in data

    @pytest.mark.asyncio
    async def test_map_finding_to_controls_returns_json(self):
        from core.mcp.tools.compliance_framework_tools import map_finding_to_controls
        finding = {
            "title": "Weak TLS configuration",
            "severity": "high",
            "category": "ssl",
            "detail": "Server supports TLS 1.0",
        }
        result = await map_finding_to_controls(finding, "SOC2")
        data = json.loads(result)
        assert data["status"] == "success"
        assert "mapped_controls" in data
        assert "controls_affected" in data

    @pytest.mark.asyncio
    async def test_get_compliance_score_returns_json(self):
        from core.mcp.tools.compliance_framework_tools import get_compliance_score
        findings = [
            {"title": "Missing MFA", "severity": "critical", "category": "iam"},
            {"title": "Weak CSP", "severity": "medium", "category": "headers"},
        ]
        result = await get_compliance_score(findings, "HIPAA")
        data = json.loads(result)
        assert data["status"] == "success"
        assert "score" in data
        assert data["score"]["percentage"] >= 0
        assert data["score"]["percentage"] <= 100
        assert data["score"]["status"] in ["passing", "warning", "failing"]

    @pytest.mark.asyncio
    async def test_get_compliance_score_all_frameworks(self):
        from core.mcp.tools.compliance_framework_tools import get_compliance_score
        findings = [{"title": "Test", "severity": "low", "category": "network"}]
        for fw in ["SOC2", "HIPAA", "PCI", "ISO27001"]:
            result = await get_compliance_score(findings, fw)
            data = json.loads(result)
            assert data["status"] == "success", f"Failed for framework {fw}"


# ══════════════════════════════════════════════════════════════════════
# 7. Domain YAML Configs
# ══════════════════════════════════════════════════════════════════════


class TestDomainYAMLConfigs:
    """Load and validate all 10 domain agent YAML configs."""

    # ── Existence ──────────────────────────────────────────────────

    def test_all_yaml_files_exist(self):
        for filename in DOMAIN_YAML_FILES:
            path = AGENTS_DIR / filename
            assert path.exists(), f"Missing YAML config: {filename}"

    def test_all_yaml_files_parseable(self):
        for filename in DOMAIN_YAML_FILES:
            cfg = _load_yaml(filename)
            assert isinstance(cfg, dict), f"Invalid YAML in {filename}"

    # ── Required Top-Level Keys ────────────────────────────────────

    def test_all_configs_have_agent_id(self):
        for filename in DOMAIN_YAML_FILES:
            cfg = _load_yaml(filename)
            assert "agent_id" in cfg, f"Missing agent_id in {filename}"
            assert isinstance(cfg["agent_id"], str)
            assert len(cfg["agent_id"]) > 0

    def test_all_configs_have_agent_type(self):
        for filename in DOMAIN_YAML_FILES:
            cfg = _load_yaml(filename)
            assert "agent_type" in cfg, f"Missing agent_type in {filename}"
            assert isinstance(cfg["agent_type"], str)

    def test_all_configs_have_name(self):
        for filename in DOMAIN_YAML_FILES:
            cfg = _load_yaml(filename)
            assert "name" in cfg, f"Missing name in {filename}"
            assert isinstance(cfg["name"], str)
            assert len(cfg["name"]) > 0

    def test_all_configs_have_description(self):
        for filename in DOMAIN_YAML_FILES:
            cfg = _load_yaml(filename)
            assert "description" in cfg, f"Missing description in {filename}"
            assert len(cfg["description"].strip()) > 0

    def test_all_configs_have_enabled(self):
        for filename in DOMAIN_YAML_FILES:
            cfg = _load_yaml(filename)
            assert "enabled" in cfg, f"Missing enabled in {filename}"
            assert cfg["enabled"] is True

    def test_all_configs_have_model(self):
        for filename in DOMAIN_YAML_FILES:
            cfg = _load_yaml(filename)
            assert "model" in cfg, f"Missing model in {filename}"
            model = cfg["model"]
            assert "provider" in model
            assert model["provider"] == "anthropic"
            assert "model" in model
            assert "temperature" in model
            assert "max_tokens" in model

    def test_all_configs_have_tools(self):
        for filename in DOMAIN_YAML_FILES:
            cfg = _load_yaml(filename)
            assert "tools" in cfg, f"Missing tools in {filename}"
            assert isinstance(cfg["tools"], list)
            assert len(cfg["tools"]) >= 2  # At minimum search_knowledge + save_insight

    def test_all_configs_have_human_gates(self):
        for filename in DOMAIN_YAML_FILES:
            cfg = _load_yaml(filename)
            assert "human_gates" in cfg, f"Missing human_gates in {filename}"
            gates = cfg["human_gates"]
            assert gates["enabled"] is True
            assert "gate_before" in gates
            assert "human_review" in gates["gate_before"]

    def test_all_configs_have_schedule(self):
        for filename in DOMAIN_YAML_FILES:
            cfg = _load_yaml(filename)
            assert "schedule" in cfg, f"Missing schedule in {filename}"
            assert "trigger" in cfg["schedule"]
            assert cfg["schedule"]["trigger"] in ("event", "scheduled")

    def test_all_configs_have_params(self):
        for filename in DOMAIN_YAML_FILES:
            cfg = _load_yaml(filename)
            assert "params" in cfg, f"Missing params in {filename}"
            assert isinstance(cfg["params"], dict)
            assert "company_name" in cfg["params"]

    # ── Agent-Specific Validation ──────────────────────────────────

    def test_vuln_scanner_config(self):
        cfg = _load_yaml("vuln_scanner.yaml")
        assert cfg["agent_id"] == "vuln_scanner_v1"
        assert cfg["agent_type"] == "vuln_scanner"
        assert "scan_ssl_certificate" in cfg["tools"]
        assert "enumerate_dns_records" in cfg["tools"]

    def test_network_analyst_config(self):
        cfg = _load_yaml("network_analyst.yaml")
        assert cfg["agent_id"] == "network_analyst_v1"
        assert cfg["agent_type"] == "network_analyst"
        assert "find_subdomains" in cfg["tools"]
        assert "check_dnssec" in cfg["tools"]
        assert "check_spf_dmarc" in cfg["tools"]

    def test_appsec_reviewer_config(self):
        cfg = _load_yaml("appsec_reviewer.yaml")
        assert cfg["agent_id"] == "appsec_reviewer_v1"
        assert cfg["agent_type"] == "appsec_reviewer"
        assert "scan_security_headers" in cfg["tools"]
        assert "check_cors_policy" in cfg["tools"]
        assert "check_csp_policy" in cfg["tools"]

    def test_compliance_mapper_config(self):
        cfg = _load_yaml("compliance_mapper.yaml")
        assert cfg["agent_id"] == "compliance_mapper_v1"
        assert cfg["agent_type"] == "compliance_mapper"
        assert "get_framework_requirements" in cfg["tools"]
        assert "map_finding_to_controls" in cfg["tools"]
        assert "get_compliance_score" in cfg["tools"]

    def test_risk_reporter_config(self):
        cfg = _load_yaml("risk_reporter.yaml")
        assert cfg["agent_id"] == "risk_reporter_v1"
        assert cfg["agent_type"] == "risk_reporter"
        assert cfg["model"]["max_tokens"] >= 8192

    def test_iam_analyst_config(self):
        cfg = _load_yaml("iam_analyst.yaml")
        assert cfg["agent_id"] == "iam_analyst_v1"
        assert cfg["agent_type"] == "iam_analyst"
        assert "mfa_required_threshold" in cfg["params"]

    def test_incident_readiness_config(self):
        cfg = _load_yaml("incident_readiness.yaml")
        assert cfg["agent_id"] == "incident_readiness_v1"
        assert cfg["agent_type"] == "incident_readiness"
        assert "scenario_categories" in cfg["params"]

    def test_cloud_security_config(self):
        cfg = _load_yaml("cloud_security.yaml")
        assert cfg["agent_id"] == "cloud_security_v1"
        assert cfg["agent_type"] == "cloud_security"
        assert "supported_providers" in cfg["params"]
        assert "aws" in cfg["params"]["supported_providers"]

    def test_security_trainer_config(self):
        cfg = _load_yaml("security_trainer.yaml")
        assert cfg["agent_id"] == "security_trainer_v1"
        assert cfg["agent_type"] == "security_trainer"
        assert cfg["schedule"]["trigger"] == "scheduled"
        assert "cron" in cfg["schedule"]

    def test_remediation_guide_config(self):
        cfg = _load_yaml("remediation_guide.yaml")
        assert cfg["agent_id"] == "remediation_guide_v1"
        assert cfg["agent_type"] == "remediation_guide"
        assert "priority_weights" in cfg["params"]
        assert "remediation_sla" in cfg["params"]

    # ── Uniqueness ─────────────────────────────────────────────────

    def test_all_agent_ids_unique(self):
        agent_ids = []
        for filename in DOMAIN_YAML_FILES:
            cfg = _load_yaml(filename)
            agent_ids.append(cfg["agent_id"])
        assert len(agent_ids) == len(set(agent_ids)), "Duplicate agent_id found"

    def test_all_agent_types_unique(self):
        agent_types = []
        for filename in DOMAIN_YAML_FILES:
            cfg = _load_yaml(filename)
            agent_types.append(cfg["agent_type"])
        assert len(agent_types) == len(set(agent_types)), "Duplicate agent_type found"
