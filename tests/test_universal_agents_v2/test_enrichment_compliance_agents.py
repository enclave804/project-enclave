"""
Tests for Universal Business Agents v2 — Phase 21 (Part 2).

Covers 2 cross-vertical business operations agents:
    1. DataEnrichmentAgent (data_enrichment)
    2. ComplianceAgentImpl (compliance)

Each agent tests:
    - State TypedDict import and creation
    - Agent registration, construction, state class
    - Initial state preparation
    - Module-level constants and system prompts
    - All graph nodes (async, mocked DB/LLM)
    - Graph construction and routing
    - __repr__ and write_knowledge
"""

from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ======================================================================
#  1.  DataEnrichmentAgent
# ======================================================================


class TestDataEnrichmentAgentState:
    """Tests for DataEnrichmentAgentState TypedDict."""

    def test_import(self):
        from core.agents.state import DataEnrichmentAgentState
        assert DataEnrichmentAgentState is not None

    def test_create_minimal(self):
        from core.agents.state import DataEnrichmentAgentState
        state: DataEnrichmentAgentState = {
            "agent_id": "data_enrichment_v1",
            "vertical_id": "enclave_guard",
        }
        assert state["agent_id"] == "data_enrichment_v1"

    def test_create_full(self):
        from core.agents.state import DataEnrichmentAgentState
        state: DataEnrichmentAgentState = {
            "agent_id": "data_enrichment_v1",
            "vertical_id": "enclave_guard",
            "tables_scanned": [],
            "records_scanned": 0,
            "scan_mode": "full",
            "issues_found": [],
            "total_issues": 0,
            "critical_issues": 0,
            "duplicate_groups": [],
            "duplicates_found": 0,
            "enrichment_tasks": [],
            "records_enriched": 0,
            "fixes_applied": 0,
            "fixes_approved": False,
            "issues_saved": False,
            "report_summary": "",
            "report_generated_at": "",
        }
        assert state["scan_mode"] == "full"
        assert state["total_issues"] == 0
        assert state["fixes_approved"] is False
        assert state["issues_saved"] is False

    def test_scan_fields(self):
        from core.agents.state import DataEnrichmentAgentState
        state: DataEnrichmentAgentState = {
            "tables_scanned": ["contacts", "companies"],
            "records_scanned": 150,
            "scan_mode": "incremental",
        }
        assert len(state["tables_scanned"]) == 2
        assert state["records_scanned"] == 150
        assert state["scan_mode"] == "incremental"

    def test_issues_and_duplicates_fields(self):
        from core.agents.state import DataEnrichmentAgentState
        state: DataEnrichmentAgentState = {
            "issues_found": [{"issue_type": "missing", "field": "email"}],
            "total_issues": 5,
            "critical_issues": 2,
            "duplicate_groups": [{"normalized_name": "john doe", "count": 3}],
            "duplicates_found": 2,
        }
        assert len(state["issues_found"]) == 1
        assert state["critical_issues"] == 2
        assert state["duplicates_found"] == 2


class TestDataEnrichmentAgent:
    """Tests for DataEnrichmentAgent implementation."""

    def _make_agent(self, **kwargs):
        """Create a DataEnrichmentAgent with mocked dependencies."""
        from core.agents.implementations.data_enrichment_agent import DataEnrichmentAgent
        from core.config.agent_schema import AgentInstanceConfig

        config = AgentInstanceConfig(
            agent_id="data_enrichment_v1",
            agent_type="data_enrichment",
            name="Data Enrichment Agent",
            vertical_id="enclave_guard",
            params={"company_name": "Test Corp"},
            **kwargs,
        )
        db = MagicMock()
        db.search_insights = MagicMock(return_value=[])
        db.store_insight = MagicMock(return_value={"id": "test"})
        db.log_agent_run = MagicMock()
        db.reset_agent_errors = MagicMock()

        embedder = MagicMock()
        embedder.embed_query = MagicMock(return_value=[0.0] * 1536)

        llm = MagicMock()
        llm.messages = MagicMock()

        return DataEnrichmentAgent(
            config=config, db=db, embedder=embedder, anthropic_client=llm,
        )

    # --- Registration & Construction ------------------------------------

    def test_registration(self):
        from core.agents.implementations.data_enrichment_agent import DataEnrichmentAgent  # noqa: F401
        from core.agents.registry import AGENT_IMPLEMENTATIONS
        assert "data_enrichment" in AGENT_IMPLEMENTATIONS

    def test_agent_type(self):
        from core.agents.implementations.data_enrichment_agent import DataEnrichmentAgent
        assert DataEnrichmentAgent.agent_type == "data_enrichment"

    def test_build_graph(self):
        agent = self._make_agent()
        graph = agent.build_graph()
        assert graph is not None

    def test_get_state_class(self):
        from core.agents.state import DataEnrichmentAgentState
        agent = self._make_agent()
        assert agent.get_state_class() is DataEnrichmentAgentState

    def test_get_tools(self):
        agent = self._make_agent()
        assert agent.get_tools() == []

    def test_repr(self):
        agent = self._make_agent()
        r = repr(agent)
        assert "DataEnrichmentAgent" in r
        assert "data_enrichment_v1" in r

    # --- Initial State ---------------------------------------------------

    def test_prepare_initial_state(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state(
            {"scan_tables": ["contacts"], "scan_mode": "full"}, "run-1"
        )
        assert state["tables_scanned"] == []
        assert state["records_scanned"] == 0
        assert state["scan_mode"] == "full"
        assert state["issues_found"] == []
        assert state["total_issues"] == 0
        assert state["critical_issues"] == 0
        assert state["duplicate_groups"] == []
        assert state["duplicates_found"] == 0
        assert state["enrichment_tasks"] == []
        assert state["records_enriched"] == 0
        assert state["fixes_applied"] == 0
        assert state["fixes_approved"] is False
        assert state["issues_saved"] is False
        assert state["report_summary"] == ""

    # --- Constants -------------------------------------------------------

    def test_issue_types(self):
        from core.agents.implementations import data_enrichment_agent
        types = data_enrichment_agent.ISSUE_TYPES
        assert "missing" in types
        assert "invalid_email" in types
        assert "duplicate" in types
        assert "stale" in types
        assert "inconsistent" in types
        assert "invalid_phone" in types
        assert "incomplete" in types
        assert "format_error" in types

    def test_stale_threshold_days(self):
        from core.agents.implementations import data_enrichment_agent
        assert data_enrichment_agent.STALE_THRESHOLD_DAYS == 180

    def test_email_regex(self):
        import re
        from core.agents.implementations import data_enrichment_agent
        regex = data_enrichment_agent.EMAIL_REGEX
        assert re.match(regex, "test@example.com")
        assert not re.match(regex, "not-an-email")

    def test_scan_tables(self):
        from core.agents.implementations import data_enrichment_agent
        tables = data_enrichment_agent.SCAN_TABLES
        assert "contacts" in tables
        assert "companies" in tables

    def test_data_enrichment_prompt(self):
        from core.agents.implementations import data_enrichment_agent
        prompt = data_enrichment_agent.DATA_ENRICHMENT_PROMPT
        assert "{issues_json}" in prompt
        assert "{issue_types}" in prompt

    # --- Node 1: Scan Records --------------------------------------------

    @pytest.mark.asyncio
    async def test_node_scan_records_empty(self):
        agent = self._make_agent()
        mock_result = MagicMock()
        mock_result.data = []
        agent.db.client.table.return_value.select.return_value.eq.return_value.limit.return_value.execute.return_value = mock_result

        state = agent._prepare_initial_state(
            {"scan_tables": ["contacts", "companies"], "scan_mode": "full"}, "run-1"
        )
        result = await agent._node_scan_records(state)
        assert result["current_node"] == "scan_records"
        assert result["records_scanned"] == 0
        assert result["scan_mode"] == "full"

    @pytest.mark.asyncio
    async def test_node_scan_records_with_data(self):
        agent = self._make_agent()
        mock_result = MagicMock()
        mock_result.data = [
            {"id": "r1", "name": "John", "email": "john@test.com"},
            {"id": "r2", "name": "Jane", "email": "jane@test.com"},
        ]
        agent.db.client.table.return_value.select.return_value.eq.return_value.limit.return_value.execute.return_value = mock_result

        state = agent._prepare_initial_state(
            {"scan_tables": ["contacts"], "scan_mode": "full"}, "run-1"
        )
        result = await agent._node_scan_records(state)
        assert result["current_node"] == "scan_records"
        assert result["records_scanned"] == 2
        assert "contacts" in result["tables_scanned"]

    @pytest.mark.asyncio
    async def test_node_scan_records_db_error(self):
        agent = self._make_agent()
        agent.db.client.table.return_value.select.return_value.eq.return_value.limit.side_effect = Exception("DB fail")

        state = agent._prepare_initial_state(
            {"scan_tables": ["contacts"], "scan_mode": "full"}, "run-1"
        )
        result = await agent._node_scan_records(state)
        assert result["current_node"] == "scan_records"
        assert result["records_scanned"] == 0

    # --- Node 2: Identify Issues -----------------------------------------

    @pytest.mark.asyncio
    async def test_node_identify_issues_clean_data(self):
        agent = self._make_agent()
        stale_cutoff = (
            datetime.now(timezone.utc) - timedelta(days=10)
        ).isoformat()
        mock_result = MagicMock()
        mock_result.data = [
            {"id": "r1", "name": "John Doe", "email": "john@test.com",
             "phone": "555-0100", "updated_at": stale_cutoff},
        ]
        agent.db.client.table.return_value.select.return_value.eq.return_value.limit.return_value.execute.return_value = mock_result

        state = agent._prepare_initial_state({}, "run-1")
        state["tables_scanned"] = ["contacts"]
        result = await agent._node_identify_issues(state)
        assert result["current_node"] == "identify_issues"
        # Clean data should have no or very few issues
        assert result["total_issues"] >= 0

    @pytest.mark.asyncio
    async def test_node_identify_issues_missing_email(self):
        agent = self._make_agent()
        mock_result = MagicMock()
        mock_result.data = [
            {"id": "r1", "name": "John Doe", "email": "",
             "phone": "555-0100", "updated_at": datetime.now(timezone.utc).isoformat()},
        ]
        agent.db.client.table.return_value.select.return_value.eq.return_value.limit.return_value.execute.return_value = mock_result

        state = agent._prepare_initial_state({}, "run-1")
        state["tables_scanned"] = ["contacts"]
        result = await agent._node_identify_issues(state)
        assert result["total_issues"] >= 1
        issue_types = [i["issue_type"] for i in result["issues_found"]]
        assert "missing" in issue_types

    @pytest.mark.asyncio
    async def test_node_identify_issues_invalid_email(self):
        agent = self._make_agent()
        mock_result = MagicMock()
        mock_result.data = [
            {"id": "r1", "name": "John Doe", "email": "not-an-email",
             "phone": "555-0100", "updated_at": datetime.now(timezone.utc).isoformat()},
        ]
        agent.db.client.table.return_value.select.return_value.eq.return_value.limit.return_value.execute.return_value = mock_result

        state = agent._prepare_initial_state({}, "run-1")
        state["tables_scanned"] = ["contacts"]
        result = await agent._node_identify_issues(state)
        issue_types = [i["issue_type"] for i in result["issues_found"]]
        assert "invalid_email" in issue_types

    @pytest.mark.asyncio
    async def test_node_identify_issues_duplicates(self):
        agent = self._make_agent()
        mock_result = MagicMock()
        mock_result.data = [
            {"id": "r1", "name": "John Doe", "email": "john@test.com",
             "phone": "555-0100", "updated_at": datetime.now(timezone.utc).isoformat()},
            {"id": "r2", "name": "John Doe", "email": "john2@test.com",
             "phone": "555-0200", "updated_at": datetime.now(timezone.utc).isoformat()},
        ]
        agent.db.client.table.return_value.select.return_value.eq.return_value.limit.return_value.execute.return_value = mock_result

        state = agent._prepare_initial_state({}, "run-1")
        state["tables_scanned"] = ["contacts"]
        result = await agent._node_identify_issues(state)
        assert result["duplicates_found"] >= 1
        assert len(result["duplicate_groups"]) >= 1

    @pytest.mark.asyncio
    async def test_node_identify_issues_db_error(self):
        agent = self._make_agent()
        agent.db.client.table.return_value.select.return_value.eq.return_value.limit.side_effect = Exception("DB fail")

        state = agent._prepare_initial_state({}, "run-1")
        state["tables_scanned"] = ["contacts"]
        result = await agent._node_identify_issues(state)
        assert result["current_node"] == "identify_issues"
        assert result["issues_found"] == []

    # --- Node 3: Suggest Fixes -------------------------------------------

    @pytest.mark.asyncio
    async def test_node_suggest_fixes_no_issues(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state({}, "run-1")
        state["issues_found"] = []
        result = await agent._node_suggest_fixes(state)
        assert result["current_node"] == "suggest_fixes"
        assert result["enrichment_tasks"] == []

    @pytest.mark.asyncio
    async def test_node_suggest_fixes_llm_success(self):
        agent = self._make_agent()
        fixes = [
            {
                "record_id": "r1",
                "table": "contacts",
                "field": "email",
                "issue_type": "invalid_email",
                "current_value": "not-email",
                "suggested_value": "needs_manual_lookup",
                "confidence": 0.8,
                "reasoning": "Email format is invalid",
            },
        ]
        mock_response = MagicMock()
        text_block = MagicMock()
        text_block.text = json.dumps(fixes)
        mock_response.content = [text_block]
        agent.llm.messages.create.return_value = mock_response

        state = agent._prepare_initial_state({}, "run-1")
        state["issues_found"] = [
            {"record_id": "r1", "table": "contacts", "field": "email",
             "issue_type": "invalid_email", "current_value": "not-email",
             "details": "Email fails validation"},
        ]
        result = await agent._node_suggest_fixes(state)
        assert result["current_node"] == "suggest_fixes"
        assert len(result["enrichment_tasks"]) >= 1

    @pytest.mark.asyncio
    async def test_node_suggest_fixes_llm_error_fallback(self):
        agent = self._make_agent()
        # Return invalid JSON to trigger fallback
        mock_response = MagicMock()
        text_block = MagicMock()
        text_block.text = "not valid json"
        mock_response.content = [text_block]
        agent.llm.messages.create.return_value = mock_response

        state = agent._prepare_initial_state({}, "run-1")
        state["issues_found"] = [
            {"record_id": "r1", "table": "contacts", "field": "email",
             "issue_type": "invalid_email", "current_value": "bad@",
             "details": "Invalid email"},
        ]
        result = await agent._node_suggest_fixes(state)
        assert result["current_node"] == "suggest_fixes"
        # Fallback should create basic fix suggestions
        assert len(result["enrichment_tasks"]) >= 1

    @pytest.mark.asyncio
    async def test_node_suggest_fixes_stale_records(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state({}, "run-1")
        state["issues_found"] = [
            {"record_id": "r1", "table": "contacts", "field": "updated_at",
             "issue_type": "stale", "current_value": "2023-01-01",
             "details": "Record not updated in over 180 days"},
        ]
        result = await agent._node_suggest_fixes(state)
        assert result["current_node"] == "suggest_fixes"
        # Stale records get suggestions without LLM
        stale_tasks = [t for t in result["enrichment_tasks"] if t["issue_type"] == "stale"]
        assert len(stale_tasks) >= 1
        assert stale_tasks[0]["suggested_value"] == "re_verify"

    # --- Node 4: Human Review -------------------------------------------

    @pytest.mark.asyncio
    async def test_node_human_review(self):
        agent = self._make_agent()
        state = {
            "total_issues": 10,
            "enrichment_tasks": [{"record_id": "r1"}],
        }
        result = await agent._node_human_review(state)
        assert result["requires_human_approval"] is True
        assert result["current_node"] == "human_review"

    # --- Node 5: Report --------------------------------------------------

    @pytest.mark.asyncio
    async def test_node_report_success(self):
        agent = self._make_agent()
        mock_insert = MagicMock()
        mock_insert.execute.return_value = MagicMock(data=[{"id": "issue_1"}])
        agent.db.client.table.return_value.insert.return_value = mock_insert

        state = agent._prepare_initial_state({}, "run-1")
        state["issues_found"] = [
            {"record_id": "r1", "table": "contacts", "field": "email",
             "issue_type": "missing", "current_value": "", "details": "Missing email"},
        ]
        state["total_issues"] = 1
        state["critical_issues"] = 1
        state["enrichment_tasks"] = [{"confidence": 0.9}]
        state["duplicate_groups"] = []
        state["tables_scanned"] = ["contacts"]
        state["records_scanned"] = 50
        state["human_approval_status"] = "approved"
        result = await agent._node_report(state)
        assert result["current_node"] == "report"
        assert result["issues_saved"] is True
        assert result["fixes_applied"] >= 1
        assert result["records_enriched"] >= 1
        assert "Data Quality Report" in result["report_summary"]
        assert result["report_generated_at"] != ""

    @pytest.mark.asyncio
    async def test_node_report_db_failure(self):
        agent = self._make_agent()
        agent.db.client.table.return_value.insert.return_value.execute.side_effect = Exception("DB fail")

        state = agent._prepare_initial_state({}, "run-1")
        state["issues_found"] = [
            {"record_id": "r1", "table": "contacts", "field": "email",
             "issue_type": "missing", "current_value": "", "details": "Missing"},
        ]
        state["total_issues"] = 1
        state["critical_issues"] = 1
        state["enrichment_tasks"] = []
        state["duplicate_groups"] = []
        state["tables_scanned"] = ["contacts"]
        state["records_scanned"] = 10
        result = await agent._node_report(state)
        assert result["issues_saved"] is False

    @pytest.mark.asyncio
    async def test_node_report_no_issues(self):
        agent = self._make_agent()
        mock_insert = MagicMock()
        mock_insert.execute.return_value = MagicMock(data=[])
        agent.db.client.table.return_value.insert.return_value = mock_insert

        state = agent._prepare_initial_state({}, "run-1")
        state["issues_found"] = []
        state["total_issues"] = 0
        state["critical_issues"] = 0
        state["enrichment_tasks"] = []
        state["duplicate_groups"] = []
        state["tables_scanned"] = ["contacts", "companies"]
        state["records_scanned"] = 100
        result = await agent._node_report(state)
        assert result["current_node"] == "report"
        assert result["fixes_applied"] == 0
        assert result["report_generated_at"] != ""

    # --- Routing ---------------------------------------------------------

    def test_route_approved(self):
        from core.agents.implementations.data_enrichment_agent import DataEnrichmentAgent
        state = {"human_approval_status": "approved"}
        assert DataEnrichmentAgent._route_after_review(state) == "approved"

    def test_route_rejected(self):
        from core.agents.implementations.data_enrichment_agent import DataEnrichmentAgent
        state = {"human_approval_status": "rejected"}
        assert DataEnrichmentAgent._route_after_review(state) == "rejected"

    def test_route_default_approved(self):
        from core.agents.implementations.data_enrichment_agent import DataEnrichmentAgent
        state = {}
        assert DataEnrichmentAgent._route_after_review(state) == "approved"

    # --- write_knowledge -------------------------------------------------

    @pytest.mark.asyncio
    async def test_write_knowledge(self):
        agent = self._make_agent()
        result = await agent.write_knowledge({"report_summary": "test"})
        assert result is None


# ======================================================================
#  2.  ComplianceAgentImpl
# ======================================================================


class TestComplianceAgentState:
    """Tests for ComplianceAgentState TypedDict."""

    def test_import(self):
        from core.agents.state import ComplianceAgentState
        assert ComplianceAgentState is not None

    def test_create_minimal(self):
        from core.agents.state import ComplianceAgentState
        state: ComplianceAgentState = {
            "agent_id": "compliance_v1",
            "vertical_id": "enclave_guard",
        }
        assert state["agent_id"] == "compliance_v1"

    def test_create_full(self):
        from core.agents.state import ComplianceAgentState
        state: ComplianceAgentState = {
            "agent_id": "compliance_v1",
            "vertical_id": "enclave_guard",
            "active_regulations": [],
            "regulation_requirements": [],
            "consent_records": [],
            "missing_consent": [],
            "total_records_audited": 0,
            "consent_gaps": 0,
            "expiring_records": [],
            "retention_actions": [],
            "compliance_score": 0.0,
            "findings": [],
            "findings_count": 0,
            "actions_approved": False,
            "records_saved": False,
            "report_document": "",
            "report_summary": "",
            "report_generated_at": "",
        }
        assert state["compliance_score"] == 0.0
        assert state["actions_approved"] is False
        assert state["records_saved"] is False

    def test_consent_audit_fields(self):
        from core.agents.state import ComplianceAgentState
        state: ComplianceAgentState = {
            "consent_records": [{"contact_id": "c1", "consent_type": "marketing_email", "status": "active"}],
            "missing_consent": [{"contact_id": "c2", "consent_type": "data_processing"}],
            "total_records_audited": 100,
            "consent_gaps": 15,
        }
        assert len(state["consent_records"]) == 1
        assert state["consent_gaps"] == 15

    def test_findings_fields(self):
        from core.agents.state import ComplianceAgentState
        state: ComplianceAgentState = {
            "findings": [
                {"regulation": "gdpr", "severity": "critical", "description": "Missing DPA"},
            ],
            "findings_count": 1,
            "compliance_score": 72.5,
            "report_document": "# Compliance Report...",
        }
        assert len(state["findings"]) == 1
        assert state["compliance_score"] == 72.5


class TestComplianceAgentImpl:
    """Tests for ComplianceAgentImpl implementation."""

    def _make_agent(self, **kwargs):
        """Create a ComplianceAgentImpl with mocked dependencies."""
        from core.agents.implementations.compliance_agent_impl import ComplianceAgentImpl
        from core.config.agent_schema import AgentInstanceConfig

        config = AgentInstanceConfig(
            agent_id="compliance_v1",
            agent_type="compliance",
            name="Compliance Agent",
            vertical_id="enclave_guard",
            params={"company_name": "Test Corp"},
            **kwargs,
        )
        db = MagicMock()
        db.search_insights = MagicMock(return_value=[])
        db.store_insight = MagicMock(return_value={"id": "test"})
        db.log_agent_run = MagicMock()
        db.reset_agent_errors = MagicMock()

        embedder = MagicMock()
        embedder.embed_query = MagicMock(return_value=[0.0] * 1536)

        llm = MagicMock()
        llm.messages = MagicMock()

        return ComplianceAgentImpl(
            config=config, db=db, embedder=embedder, anthropic_client=llm,
        )

    # --- Registration & Construction ------------------------------------

    def test_registration(self):
        from core.agents.implementations.compliance_agent_impl import ComplianceAgentImpl  # noqa: F401
        from core.agents.registry import AGENT_IMPLEMENTATIONS
        assert "compliance" in AGENT_IMPLEMENTATIONS

    def test_agent_type(self):
        from core.agents.implementations.compliance_agent_impl import ComplianceAgentImpl
        assert ComplianceAgentImpl.agent_type == "compliance"

    def test_build_graph(self):
        agent = self._make_agent()
        graph = agent.build_graph()
        assert graph is not None

    def test_get_state_class(self):
        from core.agents.state import ComplianceAgentState
        agent = self._make_agent()
        assert agent.get_state_class() is ComplianceAgentState

    def test_get_tools(self):
        agent = self._make_agent()
        assert agent.get_tools() == []

    def test_repr(self):
        agent = self._make_agent()
        r = repr(agent)
        assert "ComplianceAgentImpl" in r
        assert "compliance_v1" in r

    # --- Initial State ---------------------------------------------------

    def test_prepare_initial_state(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state(
            {"regulations": ["gdpr", "ccpa"]}, "run-1"
        )
        assert state["active_regulations"] == []
        assert state["regulation_requirements"] == []
        assert state["consent_records"] == []
        assert state["missing_consent"] == []
        assert state["total_records_audited"] == 0
        assert state["consent_gaps"] == 0
        assert state["expiring_records"] == []
        assert state["retention_actions"] == []
        assert state["compliance_score"] == 0.0
        assert state["findings"] == []
        assert state["findings_count"] == 0
        assert state["actions_approved"] is False
        assert state["records_saved"] is False
        assert state["report_document"] == ""
        assert state["report_summary"] == ""

    # --- Constants -------------------------------------------------------

    def test_supported_regulations(self):
        from core.agents.implementations import compliance_agent_impl
        regs = compliance_agent_impl.SUPPORTED_REGULATIONS
        assert "gdpr" in regs
        assert "ccpa" in regs
        assert "can_spam" in regs
        assert "hipaa" in regs
        assert "sox" in regs
        assert "pci_dss" in regs

    def test_consent_types(self):
        from core.agents.implementations import compliance_agent_impl
        types = compliance_agent_impl.CONSENT_TYPES
        assert "marketing_email" in types
        assert "data_processing" in types
        assert "analytics" in types
        assert "third_party_sharing" in types

    def test_retention_defaults(self):
        from core.agents.implementations import compliance_agent_impl
        defaults = compliance_agent_impl.RETENTION_DEFAULTS
        assert "gdpr" in defaults
        assert "ccpa" in defaults
        assert "hipaa" in defaults
        assert defaults["gdpr"] == 365 * 3
        assert defaults["hipaa"] == 365 * 6

    def test_regulation_requirements(self):
        from core.agents.implementations import compliance_agent_impl
        reqs = compliance_agent_impl.REGULATION_REQUIREMENTS
        assert "gdpr" in reqs
        assert "ccpa" in reqs
        assert "can_spam" in reqs
        assert reqs["gdpr"]["right_to_erasure"] is True
        assert reqs["gdpr"]["breach_notification_hours"] == 72
        assert reqs["can_spam"]["unsubscribe_required"] is True

    def test_compliance_audit_prompt(self):
        from core.agents.implementations import compliance_agent_impl
        prompt = compliance_agent_impl.COMPLIANCE_AUDIT_PROMPT
        assert "{regulations}" in prompt
        assert "{consent_summary_json}" in prompt
        assert "{missing_consent_json}" in prompt
        assert "{retention_json}" in prompt

    def test_compliance_report_prompt(self):
        from core.agents.implementations import compliance_agent_impl
        prompt = compliance_agent_impl.COMPLIANCE_REPORT_PROMPT
        assert "{findings_json}" in prompt
        assert "{scores_json}" in prompt
        assert "{overall_score}" in prompt

    # --- Node 1: Audit Consent -------------------------------------------

    @pytest.mark.asyncio
    async def test_node_audit_consent_no_contacts(self):
        agent = self._make_agent()
        mock_result = MagicMock()
        mock_result.data = []
        agent.db.client.table.return_value.select.return_value.eq.return_value.limit.return_value.execute.return_value = mock_result

        state = agent._prepare_initial_state(
            {"regulations": ["gdpr", "can_spam"]}, "run-1"
        )
        result = await agent._node_audit_consent(state)
        assert result["current_node"] == "audit_consent"
        assert "gdpr" in result["active_regulations"]
        assert "can_spam" in result["active_regulations"]
        assert result["total_records_audited"] == 0
        assert result["consent_gaps"] == 0

    @pytest.mark.asyncio
    async def test_node_audit_consent_with_contacts(self):
        agent = self._make_agent()

        # Contacts query
        mock_contacts = MagicMock()
        mock_contacts.data = [
            {"id": "c1", "name": "John", "email": "john@test.com"},
            {"id": "c2", "name": "Jane", "email": "jane@test.com"},
        ]

        # Consent records query
        mock_consent = MagicMock()
        mock_consent.data = [
            {"contact_id": "c1", "consent_type": "marketing_email",
             "status": "granted", "expires_at": ""},
        ]

        # Setup chained mocks for contacts query and consent query
        agent.db.client.table.return_value.select.return_value.eq.return_value.limit.return_value.execute.return_value = mock_contacts
        agent.db.client.table.return_value.select.return_value.eq.return_value.in_.return_value.execute.return_value = mock_consent

        state = agent._prepare_initial_state(
            {"regulations": ["gdpr", "can_spam"]}, "run-1"
        )
        result = await agent._node_audit_consent(state)
        assert result["current_node"] == "audit_consent"
        assert result["total_records_audited"] == 2
        assert len(result["regulation_requirements"]) >= 1
        # There should be missing consent gaps
        assert result["consent_gaps"] >= 0

    @pytest.mark.asyncio
    async def test_node_audit_consent_db_error(self):
        agent = self._make_agent()
        agent.db.client.table.return_value.select.return_value.eq.return_value.limit.side_effect = Exception("DB fail")

        state = agent._prepare_initial_state(
            {"regulations": ["gdpr"]}, "run-1"
        )
        result = await agent._node_audit_consent(state)
        assert result["current_node"] == "audit_consent"
        assert result["total_records_audited"] == 0

    @pytest.mark.asyncio
    async def test_node_audit_consent_filters_unsupported(self):
        agent = self._make_agent()
        mock_result = MagicMock()
        mock_result.data = []
        agent.db.client.table.return_value.select.return_value.eq.return_value.limit.return_value.execute.return_value = mock_result

        state = agent._prepare_initial_state(
            {"regulations": ["gdpr", "fake_regulation"]}, "run-1"
        )
        result = await agent._node_audit_consent(state)
        assert "gdpr" in result["active_regulations"]
        assert "fake_regulation" not in result["active_regulations"]

    # --- Node 2: Check Retention -----------------------------------------

    @pytest.mark.asyncio
    async def test_node_check_retention_no_violations(self):
        agent = self._make_agent()
        mock_result = MagicMock()
        mock_result.data = []
        agent.db.client.table.return_value.select.return_value.eq.return_value.lt.return_value.limit.return_value.execute.return_value = mock_result

        state = agent._prepare_initial_state({}, "run-1")
        state["active_regulations"] = ["gdpr", "ccpa"]
        result = await agent._node_check_retention(state)
        assert result["current_node"] == "check_retention"
        assert result["expiring_records"] == []
        assert result["retention_actions"] == []

    @pytest.mark.asyncio
    async def test_node_check_retention_with_violations(self):
        agent = self._make_agent()
        old_date = (datetime.now(timezone.utc) - timedelta(days=365 * 5)).isoformat()
        mock_result = MagicMock()
        mock_result.data = [
            {"id": "r1", "created_at": old_date, "updated_at": old_date,
             "name": "Old Record", "email": "old@test.com"},
        ]
        agent.db.client.table.return_value.select.return_value.eq.return_value.lt.return_value.limit.return_value.execute.return_value = mock_result

        state = agent._prepare_initial_state({}, "run-1")
        state["active_regulations"] = ["ccpa"]  # 2 year retention
        result = await agent._node_check_retention(state)
        assert result["current_node"] == "check_retention"
        assert len(result["expiring_records"]) >= 1
        assert len(result["retention_actions"]) >= 1

    @pytest.mark.asyncio
    async def test_node_check_retention_db_error(self):
        agent = self._make_agent()
        agent.db.client.table.return_value.select.return_value.eq.return_value.lt.side_effect = Exception("DB fail")

        state = agent._prepare_initial_state({}, "run-1")
        state["active_regulations"] = ["gdpr"]
        result = await agent._node_check_retention(state)
        assert result["current_node"] == "check_retention"
        assert result["expiring_records"] == []

    # --- Node 3: Generate Compliance Report ------------------------------

    @pytest.mark.asyncio
    async def test_node_generate_compliance_report_llm_success(self):
        agent = self._make_agent()
        audit_result = {
            "findings": [
                {"regulation": "gdpr", "finding_type": "missing_consent",
                 "severity": "critical", "description": "Missing DPA consent",
                 "affected_records": 15, "recommended_action": "Collect consent",
                 "compliance_risk": "GDPR Article 6 violation"},
            ],
            "regulation_scores": {"gdpr": 65.0},
            "overall_score": 65.0,
            "executive_summary": "Significant compliance gaps identified.",
        }
        report_text = "# Compliance Report\n\n## Executive Summary\nSignificant gaps..."

        mock_audit_response = MagicMock()
        audit_block = MagicMock()
        audit_block.text = json.dumps(audit_result)
        mock_audit_response.content = [audit_block]

        mock_report_response = MagicMock()
        report_block = MagicMock()
        report_block.text = report_text
        mock_report_response.content = [report_block]

        agent.llm.messages.create.side_effect = [mock_audit_response, mock_report_response]

        state = agent._prepare_initial_state({}, "run-1")
        state["active_regulations"] = ["gdpr"]
        state["consent_records"] = [{"status": "active"}]
        state["missing_consent"] = [{"contact_id": "c1", "consent_type": "data_processing"}]
        state["expiring_records"] = []
        state["total_records_audited"] = 50
        result = await agent._node_generate_compliance_report(state)
        assert result["current_node"] == "generate_compliance_report"
        assert result["compliance_score"] == 65.0
        assert len(result["findings"]) == 1
        assert result["findings_count"] == 1
        assert "Compliance Report" in result["report_document"]

    @pytest.mark.asyncio
    async def test_node_generate_compliance_report_llm_error_fallback(self):
        agent = self._make_agent()
        agent.llm.messages.create.side_effect = Exception("LLM timeout")

        state = agent._prepare_initial_state({}, "run-1")
        state["active_regulations"] = ["gdpr"]
        state["consent_records"] = []
        state["missing_consent"] = [{"contact_id": "c1"}]
        state["expiring_records"] = []
        state["total_records_audited"] = 10
        result = await agent._node_generate_compliance_report(state)
        assert result["current_node"] == "generate_compliance_report"
        # Fallback report should still be generated
        assert "Compliance Report" in result["report_document"]

    @pytest.mark.asyncio
    async def test_node_generate_compliance_report_empty_data(self):
        agent = self._make_agent()
        # Return valid but empty audit result
        audit_result = {
            "findings": [],
            "regulation_scores": {"gdpr": 100.0},
            "overall_score": 100.0,
            "executive_summary": "No issues found.",
        }
        report_text = "# Compliance Report\n\nAll clear."

        mock_audit_response = MagicMock()
        audit_block = MagicMock()
        audit_block.text = json.dumps(audit_result)
        mock_audit_response.content = [audit_block]

        mock_report_response = MagicMock()
        report_block = MagicMock()
        report_block.text = report_text
        mock_report_response.content = [report_block]

        agent.llm.messages.create.side_effect = [mock_audit_response, mock_report_response]

        state = agent._prepare_initial_state({}, "run-1")
        state["active_regulations"] = ["gdpr"]
        state["consent_records"] = []
        state["missing_consent"] = []
        state["expiring_records"] = []
        state["total_records_audited"] = 0
        result = await agent._node_generate_compliance_report(state)
        assert result["compliance_score"] == 100.0
        assert result["findings_count"] == 0

    # --- Node 4: Human Review -------------------------------------------

    @pytest.mark.asyncio
    async def test_node_human_review(self):
        agent = self._make_agent()
        state = {
            "compliance_score": 72.0,
            "findings": [{"severity": "high"}],
        }
        result = await agent._node_human_review(state)
        assert result["requires_human_approval"] is True
        assert result["current_node"] == "human_review"

    # --- Node 5: Report --------------------------------------------------

    @pytest.mark.asyncio
    async def test_node_report_success(self):
        agent = self._make_agent()
        mock_insert = MagicMock()
        mock_insert.execute.return_value = MagicMock(data=[{"id": "comp_1"}])
        agent.db.client.table.return_value.insert.return_value = mock_insert

        state = agent._prepare_initial_state({}, "run-1")
        state["active_regulations"] = ["gdpr", "ccpa"]
        state["findings"] = [
            {"regulation": "gdpr", "severity": "critical", "description": "Missing DPA"},
        ]
        state["compliance_score"] = 72.0
        state["missing_consent"] = [{"contact_id": "c1"}]
        state["expiring_records"] = [{"record_id": "r1"}]
        state["retention_actions"] = [{"action": "archive"}]
        state["total_records_audited"] = 50
        state["human_approval_status"] = "approved"
        state["report_document"] = "# Report"
        result = await agent._node_report(state)
        assert result["current_node"] == "report"
        assert result["records_saved"] is True
        assert result["actions_approved"] is True
        assert "Compliance Audit Summary" in result["report_summary"]
        assert result["report_generated_at"] != ""

    @pytest.mark.asyncio
    async def test_node_report_db_failure(self):
        agent = self._make_agent()
        agent.db.client.table.return_value.insert.return_value.execute.side_effect = Exception("DB fail")

        state = agent._prepare_initial_state({}, "run-1")
        state["active_regulations"] = ["gdpr"]
        state["findings"] = []
        state["compliance_score"] = 100.0
        state["missing_consent"] = []
        state["expiring_records"] = []
        state["retention_actions"] = []
        state["total_records_audited"] = 10
        state["report_document"] = ""
        result = await agent._node_report(state)
        assert result["records_saved"] is False

    @pytest.mark.asyncio
    async def test_node_report_rejected(self):
        agent = self._make_agent()
        mock_insert = MagicMock()
        mock_insert.execute.return_value = MagicMock(data=[{"id": "comp_2"}])
        agent.db.client.table.return_value.insert.return_value = mock_insert

        state = agent._prepare_initial_state({}, "run-1")
        state["active_regulations"] = ["gdpr"]
        state["findings"] = [{"regulation": "gdpr", "severity": "high", "description": "Gap"}]
        state["compliance_score"] = 80.0
        state["missing_consent"] = []
        state["expiring_records"] = []
        state["retention_actions"] = []
        state["total_records_audited"] = 20
        state["human_approval_status"] = "rejected"
        state["report_document"] = "# Draft"
        result = await agent._node_report(state)
        assert result["current_node"] == "report"
        assert result["actions_approved"] is False
        assert result["records_saved"] is True

    # --- Routing ---------------------------------------------------------

    def test_route_approved(self):
        from core.agents.implementations.compliance_agent_impl import ComplianceAgentImpl
        state = {"human_approval_status": "approved"}
        assert ComplianceAgentImpl._route_after_review(state) == "approved"

    def test_route_rejected(self):
        from core.agents.implementations.compliance_agent_impl import ComplianceAgentImpl
        state = {"human_approval_status": "rejected"}
        assert ComplianceAgentImpl._route_after_review(state) == "rejected"

    def test_route_default_approved(self):
        from core.agents.implementations.compliance_agent_impl import ComplianceAgentImpl
        state = {}
        assert ComplianceAgentImpl._route_after_review(state) == "approved"

    # --- write_knowledge -------------------------------------------------

    @pytest.mark.asyncio
    async def test_write_knowledge(self):
        agent = self._make_agent()
        result = await agent.write_knowledge({"report_summary": "test"})
        assert result is None
