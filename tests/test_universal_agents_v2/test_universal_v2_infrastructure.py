"""
Tests for Universal Business Agents v2 Infrastructure — Phase 21.

Covers:
    - Migration 017_universal_agents_v2.sql schema validation (8 tables, indexes, RPC)
    - 8 Universal business agent v2 state TypedDicts
    - 8 Pydantic contract models
    - 16 YAML configs (8 Enclave Guard + 8 PrintBiz)
    - 3 MCP tool stub modules (billing, survey, data_quality)
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

MIGRATION_017_PATH = (
    Path(__file__).parent.parent.parent
    / "infrastructure"
    / "migrations"
    / "017_universal_agents_v2.sql"
)

GUARD_AGENTS_DIR = (
    Path(__file__).parent.parent.parent
    / "verticals"
    / "enclave_guard"
    / "agents"
)

PRINT_AGENTS_DIR = (
    Path(__file__).parent.parent.parent
    / "verticals"
    / "print_biz"
    / "agents"
)

PHASE21_YAMLS = [
    "onboarding.yaml",
    "invoice.yaml",
    "knowledge_base.yaml",
    "feedback.yaml",
    "referral.yaml",
    "win_loss.yaml",
    "data_enrichment.yaml",
    "compliance.yaml",
]


def _read_migration(path: Path) -> str:
    with open(path) as f:
        return f.read()


def _load_yaml(directory: Path, filename: str) -> dict[str, Any]:
    with open(directory / filename) as f:
        return yaml.safe_load(f)


# ══════════════════════════════════════════════════════════════════════
# 1. Migration 017 — Universal Business Agent v2 Tables
# ══════════════════════════════════════════════════════════════════════


class TestMigration017:
    """Validate 017_universal_agents_v2.sql migration structure."""

    def test_migration_file_exists(self):
        assert MIGRATION_017_PATH.exists(), "017_universal_agents_v2.sql not found"

    def test_migration_not_empty(self):
        sql = _read_migration(MIGRATION_017_PATH)
        assert len(sql) > 100

    # ── Tables ─────────────────────────────────────────────────────

    def test_creates_client_onboarding_table(self):
        sql = _read_migration(MIGRATION_017_PATH)
        assert "CREATE TABLE" in sql
        assert "client_onboarding" in sql

    def test_creates_invoices_table(self):
        sql = _read_migration(MIGRATION_017_PATH)
        assert "invoices" in sql

    def test_creates_knowledge_articles_table(self):
        sql = _read_migration(MIGRATION_017_PATH)
        assert "knowledge_articles" in sql

    def test_creates_feedback_responses_table(self):
        sql = _read_migration(MIGRATION_017_PATH)
        assert "feedback_responses" in sql

    def test_creates_referrals_table(self):
        sql = _read_migration(MIGRATION_017_PATH)
        assert "referrals" in sql

    def test_creates_deal_analyses_table(self):
        sql = _read_migration(MIGRATION_017_PATH)
        assert "deal_analyses" in sql

    def test_creates_data_quality_issues_table(self):
        sql = _read_migration(MIGRATION_017_PATH)
        assert "data_quality_issues" in sql

    def test_creates_compliance_records_table(self):
        sql = _read_migration(MIGRATION_017_PATH)
        assert "compliance_records" in sql

    # ── Status CHECK constraints ──────────────────────────────────

    def test_client_onboarding_status_check(self):
        sql = _read_migration(MIGRATION_017_PATH)
        for status in ["pending", "in_progress", "completed", "stalled", "cancelled"]:
            assert status in sql, f"client_onboarding status '{status}' not in CHECK"

    def test_invoices_status_check(self):
        sql = _read_migration(MIGRATION_017_PATH)
        for status in ["draft", "sent", "viewed", "paid", "overdue", "void", "refunded"]:
            assert status in sql, f"invoices status '{status}' not in CHECK"

    def test_knowledge_articles_status_check(self):
        sql = _read_migration(MIGRATION_017_PATH)
        for status in ["draft", "review", "published", "archived"]:
            assert status in sql, f"knowledge_articles status '{status}' not in CHECK"

    def test_feedback_responses_survey_type_check(self):
        sql = _read_migration(MIGRATION_017_PATH)
        for stype in ["nps", "csat", "ces", "custom"]:
            assert stype in sql, f"feedback_responses survey_type '{stype}' not in CHECK"

    def test_referrals_status_check(self):
        sql = _read_migration(MIGRATION_017_PATH)
        for status in ["submitted", "contacted", "qualified", "converted", "lost", "expired"]:
            assert status in sql, f"referrals status '{status}' not in CHECK"

    def test_deal_analyses_outcome_check(self):
        sql = _read_migration(MIGRATION_017_PATH)
        for outcome in ["won", "lost"]:
            assert outcome in sql, f"deal_analyses outcome '{outcome}' not in CHECK"

    def test_data_quality_issues_issue_type_check(self):
        sql = _read_migration(MIGRATION_017_PATH)
        for issue_type in ["missing", "invalid_email", "duplicate", "stale"]:
            assert issue_type in sql, f"data_quality_issues issue_type '{issue_type}' not in CHECK"

    def test_compliance_records_status_check(self):
        sql = _read_migration(MIGRATION_017_PATH)
        for status in ["active", "expired", "revoked", "pending_review", "archived"]:
            assert status in sql, f"compliance_records status '{status}' not in CHECK"

    # ── Key columns ───────────────────────────────────────────────

    def test_client_onboarding_columns(self):
        sql = _read_migration(MIGRATION_017_PATH)
        expected_cols = [
            "vertical_id",
            "company_id",
            "contact_id",
            "opportunity_id",
            "company_name",
            "template_name",
            "milestones",
            "total_milestones",
            "completed_milestones",
            "completion_percent",
            "status",
            "welcome_package_sent",
        ]
        for col in expected_cols:
            assert col in sql, f"Column '{col}' missing from client_onboarding"

    def test_invoices_columns(self):
        sql = _read_migration(MIGRATION_017_PATH)
        expected_cols = [
            "vertical_id",
            "invoice_number",
            "line_items",
            "subtotal_cents",
            "tax_cents",
            "total_cents",
            "currency",
            "status",
            "due_date",
            "paid_at",
            "stripe_invoice_id",
            "reminder_count",
        ]
        for col in expected_cols:
            assert col in sql, f"Column '{col}' missing from invoices"

    def test_compliance_records_columns(self):
        sql = _read_migration(MIGRATION_017_PATH)
        expected_cols = [
            "vertical_id",
            "regulation",
            "record_type",
            "consent_given",
            "consent_type",
            "consent_timestamp",
            "retention_expiry",
            "data_categories",
            "status",
        ]
        for col in expected_cols:
            assert col in sql, f"Column '{col}' missing from compliance_records"

    # ── Indexes ────────────────────────────────────────────────────

    def test_indexes_exist(self):
        sql = _read_migration(MIGRATION_017_PATH)
        assert "CREATE INDEX" in sql

    def test_indexes_cover_vertical_id(self):
        sql = _read_migration(MIGRATION_017_PATH)
        # Multiple indexes should reference vertical_id
        assert sql.count("vertical_id") > 8, (
            "Expected many vertical_id references across index definitions"
        )

    # ── RPC Function ───────────────────────────────────────────────

    def test_rpc_function(self):
        sql = _read_migration(MIGRATION_017_PATH)
        assert "get_universal_business_metrics_v2" in sql


# ══════════════════════════════════════════════════════════════════════
# 2. State TypedDicts
# ══════════════════════════════════════════════════════════════════════


class TestPhase21States:
    """Import and validate all 8 Phase 21 agent state TypedDicts."""

    def test_import_onboarding_agent_state(self):
        from core.agents.state import OnboardingAgentState
        assert OnboardingAgentState is not None

    def test_import_invoice_agent_state(self):
        from core.agents.state import InvoiceAgentState
        assert InvoiceAgentState is not None

    def test_import_knowledge_base_agent_state(self):
        from core.agents.state import KnowledgeBaseAgentState
        assert KnowledgeBaseAgentState is not None

    def test_import_feedback_agent_state(self):
        from core.agents.state import FeedbackAgentState
        assert FeedbackAgentState is not None

    def test_import_referral_agent_state(self):
        from core.agents.state import ReferralAgentState
        assert ReferralAgentState is not None

    def test_import_win_loss_agent_state(self):
        from core.agents.state import WinLossAgentState
        assert WinLossAgentState is not None

    def test_import_data_enrichment_agent_state(self):
        from core.agents.state import DataEnrichmentAgentState
        assert DataEnrichmentAgentState is not None

    def test_import_compliance_agent_state(self):
        from core.agents.state import ComplianceAgentState
        assert ComplianceAgentState is not None

    def test_all_states_have_base_fields(self):
        from core.agents.state import (
            OnboardingAgentState,
            InvoiceAgentState,
            KnowledgeBaseAgentState,
            FeedbackAgentState,
            ReferralAgentState,
            WinLossAgentState,
            DataEnrichmentAgentState,
            ComplianceAgentState,
        )
        states = [
            OnboardingAgentState,
            InvoiceAgentState,
            KnowledgeBaseAgentState,
            FeedbackAgentState,
            ReferralAgentState,
            WinLossAgentState,
            DataEnrichmentAgentState,
            ComplianceAgentState,
        ]
        base_fields = ["agent_id", "vertical_id", "run_id", "current_node"]
        for state_cls in states:
            hints = get_type_hints(state_cls)
            for field in base_fields:
                assert field in hints, (
                    f"{state_cls.__name__} missing base field '{field}'"
                )

    def test_state_count_phase21(self):
        from core.agents.state import (
            OnboardingAgentState,
            InvoiceAgentState,
            KnowledgeBaseAgentState,
            FeedbackAgentState,
            ReferralAgentState,
            WinLossAgentState,
            DataEnrichmentAgentState,
            ComplianceAgentState,
        )
        states = [
            OnboardingAgentState,
            InvoiceAgentState,
            KnowledgeBaseAgentState,
            FeedbackAgentState,
            ReferralAgentState,
            WinLossAgentState,
            DataEnrichmentAgentState,
            ComplianceAgentState,
        ]
        assert len(states) == 8, "Expected 8 Phase 21 state TypedDicts"

    def test_states_are_typeddict(self):
        from core.agents.state import (
            OnboardingAgentState,
            InvoiceAgentState,
            KnowledgeBaseAgentState,
            FeedbackAgentState,
            ReferralAgentState,
            WinLossAgentState,
            DataEnrichmentAgentState,
            ComplianceAgentState,
        )
        states = [
            OnboardingAgentState,
            InvoiceAgentState,
            KnowledgeBaseAgentState,
            FeedbackAgentState,
            ReferralAgentState,
            WinLossAgentState,
            DataEnrichmentAgentState,
            ComplianceAgentState,
        ]
        for state_cls in states:
            assert hasattr(state_cls, "__annotations__"), (
                f"{state_cls.__name__} missing __annotations__"
            )
            assert len(state_cls.__annotations__) > 0, (
                f"{state_cls.__name__} has no annotations"
            )


# ══════════════════════════════════════════════════════════════════════
# 3. Pydantic Contracts
# ══════════════════════════════════════════════════════════════════════


class TestPhase21Contracts:
    """Validate Phase 21 Pydantic contract models."""

    def test_import_onboarding_request(self):
        from core.agents.contracts import OnboardingRequest
        assert OnboardingRequest is not None

    def test_import_knowledge_article_data(self):
        from core.agents.contracts import KnowledgeArticleData
        assert KnowledgeArticleData is not None

    def test_import_feedback_survey_request(self):
        from core.agents.contracts import FeedbackSurveyRequest
        assert FeedbackSurveyRequest is not None

    def test_import_feedback_response_data(self):
        from core.agents.contracts import FeedbackResponseData
        assert FeedbackResponseData is not None

    def test_import_referral_data(self):
        from core.agents.contracts import ReferralData
        assert ReferralData is not None

    def test_import_deal_analysis_data(self):
        from core.agents.contracts import DealAnalysisData
        assert DealAnalysisData is not None

    def test_import_data_quality_issue(self):
        from core.agents.contracts import DataQualityIssue
        assert DataQualityIssue is not None

    def test_import_compliance_record_data(self):
        from core.agents.contracts import ComplianceRecordData
        assert ComplianceRecordData is not None

    def test_onboarding_request_validation(self):
        from core.agents.contracts import OnboardingRequest
        req = OnboardingRequest(company_name="Acme Corp", contact_email="j@acme.com")
        assert req.company_name == "Acme Corp"
        assert req.contact_email == "j@acme.com"
        assert req.contact_name == ""
        assert req.template_name == "default"
        assert req.custom_milestones == []
        assert req.metadata == {}

    def test_referral_data_validation(self):
        from core.agents.contracts import ReferralData
        ref = ReferralData(
            referrer_email="alice@company.com",
            referee_email="bob@partner.com",
        )
        assert ref.referrer_email == "alice@company.com"
        assert ref.referee_email == "bob@partner.com"
        assert ref.referrer_name == ""
        assert ref.referee_company == ""
        assert ref.source == "client_referral"
        assert ref.notes == ""
        assert ref.metadata == {}

    def test_compliance_record_defaults(self):
        from core.agents.contracts import ComplianceRecordData
        rec = ComplianceRecordData()
        assert rec.regulation == "gdpr"
        assert rec.record_type == "consent"
        assert rec.contact_email == ""
        assert rec.consent_given is False
        assert rec.consent_type == ""
        assert rec.data_categories == []
        assert rec.metadata == {}

    def test_data_quality_issue_severity(self):
        from core.agents.contracts import DataQualityIssue
        issue = DataQualityIssue(target_table="contacts")
        assert issue.target_table == "contacts"
        assert issue.severity == "medium"
        assert issue.issue_type == "missing"
        assert issue.description == ""
        assert issue.original_value == ""
        assert issue.suggested_value == ""


# ══════════════════════════════════════════════════════════════════════
# 4. YAML Configs
# ══════════════════════════════════════════════════════════════════════


class TestEnclaveGuardV2YAMLConfigs:
    """Validate Enclave Guard Phase 21 YAML configs (8 files)."""

    @pytest.mark.parametrize("filename", PHASE21_YAMLS)
    def test_eg_yaml_exists(self, filename: str):
        path = GUARD_AGENTS_DIR / filename
        assert path.exists(), f"Missing EG YAML config: {filename}"


class TestPrintBizV2YAMLConfigs:
    """Validate PrintBiz Phase 21 YAML configs (8 files)."""

    @pytest.mark.parametrize("filename", PHASE21_YAMLS)
    def test_pb_yaml_exists(self, filename: str):
        path = PRINT_AGENTS_DIR / filename
        assert path.exists(), f"Missing PB YAML config: {filename}"


class TestYAMLConfigContent:
    """Validate YAML content across both verticals."""

    def test_yaml_has_required_keys(self):
        required_keys = ["agent_id", "agent_type", "trigger", "model_routing", "human_gates"]
        for directory in [GUARD_AGENTS_DIR, PRINT_AGENTS_DIR]:
            for filename in PHASE21_YAMLS:
                cfg = _load_yaml(directory, filename)
                for key in required_keys:
                    assert key in cfg, (
                        f"Missing key '{key}' in {directory.parent.name}/{filename}"
                    )

    def test_yaml_agent_types_match(self):
        expected_types = [
            "onboarding",
            "invoice",
            "knowledge_base",
            "feedback",
            "referral",
            "win_loss",
            "data_enrichment",
            "compliance",
        ]
        for directory in [GUARD_AGENTS_DIR, PRINT_AGENTS_DIR]:
            for filename, expected_type in zip(PHASE21_YAMLS, expected_types):
                cfg = _load_yaml(directory, filename)
                assert cfg["agent_type"] == expected_type, (
                    f"{directory.parent.name}/{filename}: "
                    f"expected agent_type={expected_type}, got {cfg['agent_type']}"
                )

    def test_eg_pb_yaml_parity(self):
        """Both verticals should have the same 8 Phase 21 YAML files."""
        for filename in PHASE21_YAMLS:
            eg_path = GUARD_AGENTS_DIR / filename
            pb_path = PRINT_AGENTS_DIR / filename
            assert eg_path.exists(), f"Enclave Guard missing {filename}"
            assert pb_path.exists(), f"PrintBiz missing {filename}"

    def test_yaml_human_gates_enabled(self):
        for directory in [GUARD_AGENTS_DIR, PRINT_AGENTS_DIR]:
            for filename in PHASE21_YAMLS:
                cfg = _load_yaml(directory, filename)
                assert cfg["human_gates"]["enabled"] is True, (
                    f"human_gates.enabled not True in {directory.parent.name}/{filename}"
                )


# ══════════════════════════════════════════════════════════════════════
# 5. MCP Tool Stubs
# ══════════════════════════════════════════════════════════════════════


class TestBillingTools:
    """Test billing_tools module stub functions."""

    def test_import_billing_tools(self):
        from core.mcp.tools import billing_tools
        assert billing_tools is not None

    def test_billing_tools_functions(self):
        from core.mcp.tools.billing_tools import (
            create_invoice,
            check_payment_status,
            send_payment_reminder,
            calculate_line_items,
        )
        assert callable(create_invoice)
        assert callable(check_payment_status)
        assert callable(send_payment_reminder)
        assert callable(calculate_line_items)
        assert inspect.iscoroutinefunction(create_invoice)
        assert inspect.iscoroutinefunction(check_payment_status)
        assert inspect.iscoroutinefunction(send_payment_reminder)
        assert inspect.iscoroutinefunction(calculate_line_items)


class TestSurveyTools:
    """Test survey_tools module stub functions."""

    def test_import_survey_tools(self):
        from core.mcp.tools import survey_tools
        assert survey_tools is not None

    def test_survey_tools_functions(self):
        from core.mcp.tools.survey_tools import (
            send_nps_survey,
            collect_survey_responses,
            calculate_nps,
            analyze_feedback_sentiment,
        )
        assert callable(send_nps_survey)
        assert callable(collect_survey_responses)
        assert callable(calculate_nps)
        assert callable(analyze_feedback_sentiment)
        assert inspect.iscoroutinefunction(send_nps_survey)
        assert inspect.iscoroutinefunction(collect_survey_responses)
        assert inspect.iscoroutinefunction(calculate_nps)
        assert inspect.iscoroutinefunction(analyze_feedback_sentiment)

    def test_calculate_nps_logic(self):
        from core.mcp.tools.survey_tools import calculate_nps
        # 5 promoters (9-10), 3 passives (7-8), 2 detractors (0-6)
        responses = [
            {"score": 10},
            {"score": 9},
            {"score": 10},
            {"score": 9},
            {"score": 10},
            {"score": 8},
            {"score": 7},
            {"score": 7},
            {"score": 5},
            {"score": 3},
        ]
        result_json = asyncio.get_event_loop().run_until_complete(
            calculate_nps(responses)
        )
        result = json.loads(result_json)
        assert result["status"] == "success"
        assert result["total_responses"] == 10
        assert result["promoters"] == 5
        assert result["passives"] == 3
        assert result["detractors"] == 2
        # NPS = 50% - 20% = 30
        assert result["nps_score"] == 30.0


class TestDataQualityTools:
    """Test data_quality_tools module stub functions."""

    def test_import_data_quality_tools(self):
        from core.mcp.tools import data_quality_tools
        assert data_quality_tools is not None

    def test_data_quality_tools_functions(self):
        from core.mcp.tools.data_quality_tools import (
            validate_email,
            find_duplicates,
            validate_phone,
            check_data_freshness,
        )
        assert callable(validate_email)
        assert callable(find_duplicates)
        assert callable(validate_phone)
        assert callable(check_data_freshness)
        assert inspect.iscoroutinefunction(validate_email)
        assert inspect.iscoroutinefunction(find_duplicates)
        assert inspect.iscoroutinefunction(validate_phone)
        assert inspect.iscoroutinefunction(check_data_freshness)

    def test_validate_email_logic(self):
        from core.mcp.tools.data_quality_tools import validate_email

        # Valid email
        result_json = asyncio.get_event_loop().run_until_complete(
            validate_email("user@example.com")
        )
        result = json.loads(result_json)
        assert result["status"] == "success"
        assert result["is_valid"] is True
        assert result["format_ok"] is True
        assert result["domain"] == "example.com"

        # Invalid email
        result_json = asyncio.get_event_loop().run_until_complete(
            validate_email("not-an-email")
        )
        result = json.loads(result_json)
        assert result["status"] == "success"
        assert result["is_valid"] is False
        assert result["format_ok"] is False
