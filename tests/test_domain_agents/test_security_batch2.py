"""
Tests for Security Domain Agents — Batch 2 (5 agents, ~150 tests).

Covers:
    1. IAMAnalystAgent        (iam_analyst)
    2. IncidentReadinessAgent (incident_readiness)
    3. CloudSecurityAgent     (cloud_security)
    4. SecurityTrainerAgent   (security_trainer)
    5. RemediationGuideAgent  (remediation_guide)

Each agent is tested for:
    - State TypedDict construction
    - Registration in AGENT_IMPLEMENTATIONS
    - Construction with mocked deps
    - State class accessor
    - Initial state preparation
    - Constants / system prompt
    - Every node (async)
    - Graph construction and node keys
    - Routing function
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock

import pytest


# ======================================================================
#  1. IAMAnalystAgent
# ======================================================================


class TestIAMAnalystState:
    """Tests for IAMAnalystAgentState TypedDict."""

    def test_import(self):
        from core.agents.state import IAMAnalystAgentState
        assert IAMAnalystAgentState is not None

    def test_create_minimal(self):
        from core.agents.state import IAMAnalystAgentState
        state: IAMAnalystAgentState = {
            "agent_id": "iam_v1",
            "vertical_id": "enclave_guard",
        }
        assert state["agent_id"] == "iam_v1"

    def test_create_full(self):
        from core.agents.state import IAMAnalystAgentState
        state: IAMAnalystAgentState = {
            "agent_id": "iam_v1",
            "vertical_id": "enclave_guard",
            "questionnaire_responses": {"mfa_adoption_rate": 0.5},
            "iam_risk_score": 72.0,
            "report_summary": "",
            "report_generated_at": "",
        }
        assert state["iam_risk_score"] == 72.0
        assert state["questionnaire_responses"]["mfa_adoption_rate"] == 0.5

    def test_risk_score_field(self):
        from core.agents.state import IAMAnalystAgentState
        state: IAMAnalystAgentState = {"iam_risk_score": 55.0}
        assert state["iam_risk_score"] == 55.0

    def test_report_fields(self):
        from core.agents.state import IAMAnalystAgentState
        state: IAMAnalystAgentState = {
            "report_summary": "Test",
            "report_generated_at": "2025-01-01T00:00:00Z",
        }
        assert state["report_summary"] == "Test"


class TestIAMAnalystAgent:
    """Tests for IAMAnalystAgent implementation."""

    def _make_agent(self, **kwargs):
        from core.agents.implementations.iam_analyst_agent import IAMAnalystAgent
        from core.config.agent_schema import AgentInstanceConfig

        config = AgentInstanceConfig(
            agent_id="iam_v1",
            agent_type="iam_analyst",
            name="IAM Analyst",
            vertical_id="enclave_guard",
            params={"company_name": "Enclave Guard"},
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

        return IAMAnalystAgent(
            config=config, db=db, embedder=embedder, anthropic_client=llm,
        )

    # ---- Registration & Construction ----

    def test_registration(self):
        from core.agents.implementations.iam_analyst_agent import IAMAnalystAgent  # noqa: F401
        from core.agents.registry import AGENT_IMPLEMENTATIONS
        assert "iam_analyst" in AGENT_IMPLEMENTATIONS

    def test_agent_type(self):
        from core.agents.implementations.iam_analyst_agent import IAMAnalystAgent
        assert IAMAnalystAgent.agent_type == "iam_analyst"

    def test_build_graph(self):
        agent = self._make_agent()
        graph = agent.build_graph()
        assert graph is not None

    def test_get_state_class(self):
        from core.agents.state import IAMAnalystAgentState
        agent = self._make_agent()
        assert agent.get_state_class() is IAMAnalystAgentState

    def test_get_tools(self):
        agent = self._make_agent()
        assert agent.get_tools() == []

    def test_repr(self):
        agent = self._make_agent()
        r = repr(agent)
        assert "IAMAnalystAgent" in r
        assert "iam_v1" in r

    # ---- Initial State ----

    def test_prepare_initial_state(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state({"questionnaire": {}}, "run-1")
        assert state["questionnaire_responses"] == {}
        assert state["access_findings"] == []
        assert state["mfa_findings"] == []
        assert state["iam_risk_score"] == 0.0
        assert state["iam_risk_level"] == "low"
        assert state["findings_approved"] is False
        assert state["report_summary"] == ""

    def test_prepare_initial_state_with_questionnaire(self):
        agent = self._make_agent()
        q = {"mfa_adoption_rate": 0.8}
        state = agent._prepare_initial_state({"questionnaire": q}, "run-2")
        assert state["questionnaire_responses"] == q

    # ---- Constants ----

    def test_constants_access_risk_levels(self):
        from core.agents.implementations import iam_analyst_agent
        assert "critical" in iam_analyst_agent.ACCESS_RISK_LEVELS
        assert "low" in iam_analyst_agent.ACCESS_RISK_LEVELS
        assert iam_analyst_agent.ACCESS_RISK_LEVELS["critical"]["min_score"] == 80

    def test_constants_mfa_types(self):
        from core.agents.implementations import iam_analyst_agent
        assert "fido2" in iam_analyst_agent.MFA_TYPES
        assert iam_analyst_agent.MFA_TYPES["fido2"]["strength"] == 1.0

    def test_constants_password_policy(self):
        from core.agents.implementations import iam_analyst_agent
        reqs = iam_analyst_agent.PASSWORD_POLICY_REQUIREMENTS
        assert "min_length" in reqs
        assert reqs["min_length"]["recommended"] == 14

    def test_system_prompt(self):
        from core.agents.implementations import iam_analyst_agent
        assert "{company_name}" in iam_analyst_agent.IAM_SYSTEM_PROMPT
        assert "{industry}" in iam_analyst_agent.IAM_SYSTEM_PROMPT

    # ---- Node 1: load_questionnaire ----

    @pytest.mark.asyncio
    async def test_node_load_questionnaire_empty(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state({}, "run-1")
        result = await agent._node_load_questionnaire(state)
        assert result["current_node"] == "load_questionnaire"
        assert result["questionnaire_responses"] == {}

    @pytest.mark.asyncio
    async def test_node_load_questionnaire_with_data(self):
        agent = self._make_agent()
        q = {"mfa_adoption_rate": 0.5, "industry": "tech"}
        state = agent._prepare_initial_state(
            {"questionnaire": q, "company_name": "Test Co"}, "run-2"
        )
        result = await agent._node_load_questionnaire(state)
        assert result["organization_profile"]["company_name"] == "Test Co"
        assert result["organization_profile"]["industry"] == "tech"

    @pytest.mark.asyncio
    async def test_node_load_questionnaire_frameworks_string(self):
        agent = self._make_agent()
        q = {"compliance_frameworks": "SOC2, ISO27001, HIPAA"}
        state = agent._prepare_initial_state({"questionnaire": q}, "run-3")
        result = await agent._node_load_questionnaire(state)
        assert "SOC2" in result["compliance_frameworks"]
        assert len(result["compliance_frameworks"]) == 3

    # ---- Node 2: analyze_access ----

    @pytest.mark.asyncio
    async def test_node_analyze_access_no_priv(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state({}, "run-1")
        state["questionnaire_responses"] = {}
        state["organization_profile"] = {"employee_count": 100, "company_name": "X"}
        result = await agent._node_analyze_access(state)
        assert result["current_node"] == "analyze_access"
        assert result["privileged_account_count"] == 0

    @pytest.mark.asyncio
    async def test_node_analyze_access_high_priv_ratio(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state({}, "run-1")
        state["questionnaire_responses"] = {
            "privileged_account_count": 20,
            "access_review_frequency": "never",
            "rbac_maturity": "none",
        }
        state["organization_profile"] = {"employee_count": 50, "company_name": "Y"}
        result = await agent._node_analyze_access(state)
        assert result["privileged_account_count"] == 20
        severities = [f["severity"] for f in result["access_findings"]]
        assert "high" in severities or "critical" in severities

    @pytest.mark.asyncio
    async def test_node_analyze_access_never_review(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state({}, "run-1")
        state["questionnaire_responses"] = {"access_review_frequency": "never"}
        state["organization_profile"] = {"employee_count": 10, "company_name": "Z"}
        result = await agent._node_analyze_access(state)
        categories = [f["category"] for f in result["access_findings"]]
        assert "access_reviews" in categories

    # ---- Node 3: assess_mfa ----

    @pytest.mark.asyncio
    async def test_node_assess_mfa_no_admin_mfa(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state({}, "run-1")
        state["questionnaire_responses"] = {
            "mfa_adoption_rate": 0.2,
            "mfa_enforced_admins": False,
        }
        state["organization_profile"] = {"company_name": "NoMFA"}
        state["access_findings"] = []
        result = await agent._node_assess_mfa(state)
        sev = [f["severity"] for f in result["mfa_findings"]]
        assert "critical" in sev

    @pytest.mark.asyncio
    async def test_node_assess_mfa_good_policy(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state({}, "run-1")
        state["questionnaire_responses"] = {
            "mfa_adoption_rate": 1.0,
            "mfa_enforced_admins": True,
            "mfa_enforced_all": True,
            "mfa_types": ["fido2", "totp"],
            "password_min_length": 16,
            "password_complexity": True,
            "breach_checking": True,
            "password_rotation_days": 0,
        }
        state["organization_profile"] = {"company_name": "GoodCo"}
        state["access_findings"] = []
        result = await agent._node_assess_mfa(state)
        assert result["password_policy_score"] > 0
        assert result["iam_risk_level"] in ("low", "medium")

    @pytest.mark.asyncio
    async def test_node_assess_mfa_sms_only(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state({}, "run-1")
        state["questionnaire_responses"] = {
            "mfa_adoption_rate": 0.9,
            "mfa_enforced_admins": True,
            "mfa_types": ["sms"],
            "password_min_length": 8,
        }
        state["organization_profile"] = {"company_name": "SMSCo"}
        state["access_findings"] = []
        result = await agent._node_assess_mfa(state)
        cats = [f["category"] for f in result["mfa_findings"]]
        assert "type_strength" in cats

    # ---- Node 4: human_review ----

    @pytest.mark.asyncio
    async def test_node_human_review(self):
        agent = self._make_agent()
        state = {"all_findings": [{"finding": "test"}], "iam_risk_score": 50}
        result = await agent._node_human_review(state)
        assert result["requires_human_approval"] is True
        assert result["current_node"] == "human_review"

    # ---- Node 5: save_findings ----

    @pytest.mark.asyncio
    async def test_node_save_findings(self):
        agent = self._make_agent()
        agent.db.client.table.return_value.insert.return_value.execute = MagicMock()
        state = agent._prepare_initial_state({}, "run-1")
        state["all_findings"] = [
            {"finding": "test finding", "severity": "high", "category": "iam"},
        ]
        state["organization_profile"] = {"company_name": "SaveCo"}
        result = await agent._node_save_findings(state)
        assert result["findings_approved"] is True
        assert result["current_node"] == "save_findings"

    @pytest.mark.asyncio
    async def test_node_save_findings_db_error(self):
        agent = self._make_agent()
        agent.db.client.table.return_value.insert.return_value.execute.side_effect = Exception("DB fail")
        state = agent._prepare_initial_state({}, "run-1")
        state["all_findings"] = [{"finding": "fail", "severity": "low"}]
        state["organization_profile"] = {"company_name": "FailCo"}
        result = await agent._node_save_findings(state)
        assert result["findings_approved"] is True

    # ---- Node 6: report ----

    @pytest.mark.asyncio
    async def test_node_report(self):
        agent = self._make_agent()
        state = {
            "organization_profile": {"company_name": "RptCo", "industry": "tech", "employee_count": 50},
            "all_findings": [{"severity": "high"}],
            "compliance_frameworks": ["SOC2"],
            "iam_risk_score": 60.0,
            "iam_risk_level": "high",
            "mfa_adoption_rate": 0.5,
            "mfa_types_deployed": ["totp"],
            "mfa_enforced_for_admins": True,
            "password_policy_score": 0.5,
            "password_min_length": 10,
            "privileged_account_count": 5,
            "service_account_count": 3,
            "access_review_frequency": "quarterly",
            "rbac_maturity": "basic",
        }
        result = await agent._node_report(state)
        assert result["current_node"] == "report"
        assert "IAM Security Assessment Report" in result["report_summary"]
        assert result["report_generated_at"] != ""

    # ---- Routing ----

    def test_route_after_review_approved(self):
        from core.agents.implementations.iam_analyst_agent import IAMAnalystAgent
        assert IAMAnalystAgent._route_after_review({"human_approval_status": "approved"}) == "approved"

    def test_route_after_review_rejected(self):
        from core.agents.implementations.iam_analyst_agent import IAMAnalystAgent
        assert IAMAnalystAgent._route_after_review({"human_approval_status": "rejected"}) == "rejected"

    def test_route_after_review_default(self):
        from core.agents.implementations.iam_analyst_agent import IAMAnalystAgent
        assert IAMAnalystAgent._route_after_review({}) == "approved"

    # ---- Graph nodes ----

    def test_graph_has_correct_nodes(self):
        agent = self._make_agent()
        graph = agent.build_graph()
        expected = {
            "load_questionnaire", "analyze_access", "assess_mfa",
            "human_review", "save_findings", "report",
        }
        node_keys = set(graph.nodes.keys()) - {"__start__", "__end__"}
        assert expected.issubset(node_keys)

    def test_write_knowledge_exists(self):
        agent = self._make_agent()
        assert hasattr(agent, "write_knowledge")


# ======================================================================
#  2. IncidentReadinessAgent
# ======================================================================


class TestIncidentReadinessState:
    """Tests for IncidentReadinessAgentState TypedDict."""

    def test_import(self):
        from core.agents.state import IncidentReadinessAgentState
        assert IncidentReadinessAgentState is not None

    def test_create_minimal(self):
        from core.agents.state import IncidentReadinessAgentState
        state: IncidentReadinessAgentState = {
            "agent_id": "ir_v1",
            "vertical_id": "enclave_guard",
        }
        assert state["agent_id"] == "ir_v1"

    def test_create_full(self):
        from core.agents.state import IncidentReadinessAgentState
        state: IncidentReadinessAgentState = {
            "agent_id": "ir_v1",
            "vertical_id": "enclave_guard",
            "readiness_score": 75.0,
            "readiness_grade": "B",
            "report_summary": "",
            "report_generated_at": "",
        }
        assert state["readiness_score"] == 75.0
        assert state["readiness_grade"] == "B"

    def test_ir_plan_field(self):
        from core.agents.state import IncidentReadinessAgentState
        state: IncidentReadinessAgentState = {"ir_plan_exists": True}
        assert state["ir_plan_exists"] is True


class TestIncidentReadinessAgent:
    """Tests for IncidentReadinessAgent implementation."""

    def _make_agent(self, **kwargs):
        from core.agents.implementations.incident_readiness_agent import IncidentReadinessAgent
        from core.config.agent_schema import AgentInstanceConfig

        config = AgentInstanceConfig(
            agent_id="ir_v1",
            agent_type="incident_readiness",
            name="Incident Readiness",
            vertical_id="enclave_guard",
            params={"company_name": "Enclave Guard"},
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

        return IncidentReadinessAgent(
            config=config, db=db, embedder=embedder, anthropic_client=llm,
        )

    # ---- Registration & Construction ----

    def test_registration(self):
        from core.agents.implementations.incident_readiness_agent import IncidentReadinessAgent  # noqa: F401
        from core.agents.registry import AGENT_IMPLEMENTATIONS
        assert "incident_readiness" in AGENT_IMPLEMENTATIONS

    def test_agent_type(self):
        from core.agents.implementations.incident_readiness_agent import IncidentReadinessAgent
        assert IncidentReadinessAgent.agent_type == "incident_readiness"

    def test_build_graph(self):
        agent = self._make_agent()
        graph = agent.build_graph()
        assert graph is not None

    def test_get_state_class(self):
        from core.agents.state import IncidentReadinessAgentState
        agent = self._make_agent()
        assert agent.get_state_class() is IncidentReadinessAgentState

    def test_get_tools(self):
        agent = self._make_agent()
        assert agent.get_tools() == []

    def test_repr(self):
        agent = self._make_agent()
        r = repr(agent)
        assert "IncidentReadinessAgent" in r
        assert "ir_v1" in r

    # ---- Initial State ----

    def test_prepare_initial_state(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state({}, "run-1")
        assert state["ir_plan_exists"] is False
        assert state["ir_plan_findings"] == []
        assert state["backup_strategy"] == "none"
        assert state["readiness_score"] == 0.0
        assert state["readiness_grade"] == "F"
        assert state["critical_gaps"] == []
        assert state["findings_approved"] is False

    # ---- Constants ----

    def test_constants_readiness_grades(self):
        from core.agents.implementations import incident_readiness_agent
        grades = incident_readiness_agent.READINESS_GRADES
        assert "A" in grades
        assert "F" in grades
        assert grades["A"]["min_score"] == 90

    def test_constants_ir_plan_requirements(self):
        from core.agents.implementations import incident_readiness_agent
        reqs = incident_readiness_agent.IR_PLAN_REQUIREMENTS
        assert "plan_exists" in reqs
        assert "tested_annually" in reqs

    def test_constants_backup_criteria(self):
        from core.agents.implementations import incident_readiness_agent
        assert "encrypted" in incident_readiness_agent.BACKUP_CRITERIA

    def test_system_prompt(self):
        from core.agents.implementations import incident_readiness_agent
        assert "{company_name}" in incident_readiness_agent.IR_SYSTEM_PROMPT

    # ---- Node 1: assess_ir_plan ----

    @pytest.mark.asyncio
    async def test_node_assess_ir_plan_no_plan(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state(
            {"questionnaire": {"ir_plan_exists": False}, "company_name": "NoPlan"},
            "run-1",
        )
        result = await agent._node_assess_ir_plan(state)
        assert result["current_node"] == "assess_ir_plan"
        assert result["ir_plan_exists"] is False
        severities = [f["severity"] for f in result["ir_plan_findings"]]
        assert "critical" in severities

    @pytest.mark.asyncio
    async def test_node_assess_ir_plan_with_plan_not_tested(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state(
            {
                "questionnaire": {
                    "ir_plan_exists": True,
                    "ir_roles_defined": True,
                    "escalation_procedures": True,
                    "runbook_count": 3,
                },
                "company_name": "HasPlan",
            },
            "run-2",
        )
        result = await agent._node_assess_ir_plan(state)
        assert result["ir_plan_exists"] is True
        assert result["runbook_count"] == 3
        # No last_tested => critical finding
        reqs = [f["requirement"] for f in result["ir_plan_findings"]]
        assert "tested_annually" in reqs

    @pytest.mark.asyncio
    async def test_node_assess_ir_plan_no_roles(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state(
            {
                "questionnaire": {
                    "ir_plan_exists": True,
                    "ir_roles_defined": False,
                    "escalation_procedures": False,
                },
                "company_name": "NoRoles",
            },
            "run-3",
        )
        result = await agent._node_assess_ir_plan(state)
        reqs = [f["requirement"] for f in result["ir_plan_findings"]]
        assert "roles_defined" in reqs
        assert "escalation_procedures" in reqs

    # ---- Node 2: evaluate_backups ----

    @pytest.mark.asyncio
    async def test_node_evaluate_backups_no_strategy(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state(
            {"questionnaire": {"backup_strategy": "none"}}, "run-1"
        )
        result = await agent._node_evaluate_backups(state)
        assert result["current_node"] == "evaluate_backups"
        assert result["backup_strategy"] == "none"
        reqs = [f["requirement"] for f in result["backup_findings"]]
        assert "strategy_exists" in reqs

    @pytest.mark.asyncio
    async def test_node_evaluate_backups_local_only(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state(
            {"questionnaire": {"backup_strategy": "local"}}, "run-2"
        )
        result = await agent._node_evaluate_backups(state)
        reqs = [f["requirement"] for f in result["backup_findings"]]
        assert "offsite_or_cloud" in reqs

    @pytest.mark.asyncio
    async def test_node_evaluate_backups_good(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state(
            {
                "questionnaire": {
                    "backup_strategy": "3-2-1",
                    "backup_tested": True,
                    "backup_encrypted": True,
                    "immutable_backups": True,
                    "rto_hours": 4,
                    "rpo_hours": 1,
                },
            },
            "run-3",
        )
        result = await agent._node_evaluate_backups(state)
        assert result["backup_encrypted"] is True
        assert result["immutable_backups"] is True
        assert result["rto_hours"] == 4

    # ---- Node 3: check_communication ----

    @pytest.mark.asyncio
    async def test_node_check_communication_none(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state({"questionnaire": {}}, "run-1")
        result = await agent._node_check_communication(state)
        assert result["current_node"] == "check_communication"
        assert result["communication_plan_exists"] is False
        reqs = [f["requirement"] for f in result["communication_findings"]]
        assert "plan_exists" in reqs

    @pytest.mark.asyncio
    async def test_node_check_communication_partial(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state(
            {
                "questionnaire": {
                    "communication_plan_exists": True,
                    "stakeholder_list_defined": True,
                    "regulatory_notification_process": False,
                },
            },
            "run-2",
        )
        result = await agent._node_check_communication(state)
        assert result["communication_plan_exists"] is True
        reqs = [f["requirement"] for f in result["communication_findings"]]
        assert "regulatory_process" in reqs

    # ---- Node 4: human_review ----

    @pytest.mark.asyncio
    async def test_node_human_review(self):
        agent = self._make_agent()
        state = {
            "ir_plan_findings": [{"finding": "a"}],
            "backup_findings": [],
            "communication_findings": [],
        }
        result = await agent._node_human_review(state)
        assert result["requires_human_approval"] is True

    # ---- Node 5: score_readiness ----

    @pytest.mark.asyncio
    async def test_node_score_readiness_perfect(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state({"company_name": "PerfectCo"}, "run-1")
        state["ir_plan_findings"] = []
        state["backup_findings"] = []
        state["communication_findings"] = []
        result = await agent._node_score_readiness(state)
        assert result["readiness_score"] == 100.0
        assert result["readiness_grade"] == "A"
        assert result["findings_approved"] is True

    @pytest.mark.asyncio
    async def test_node_score_readiness_many_critical(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state({"company_name": "BadCo"}, "run-2")
        state["ir_plan_findings"] = [
            {"severity": "critical"},
            {"severity": "critical"},
            {"severity": "critical"},
        ]
        state["backup_findings"] = [{"severity": "critical"}, {"severity": "high"}]
        state["communication_findings"] = [{"severity": "high"}]
        result = await agent._node_score_readiness(state)
        assert result["readiness_score"] < 50
        assert result["readiness_grade"] in ("D", "F")
        assert len(result["critical_gaps"]) == 4

    # ---- Node 6: report ----

    @pytest.mark.asyncio
    async def test_node_report(self):
        agent = self._make_agent()
        state = {
            "task_input": {"company_name": "RptCo"},
            "readiness_score": 65.0,
            "readiness_grade": "C",
            "all_findings": [{"severity": "medium"}],
            "ir_plan_exists": True,
            "ir_roles_defined": True,
            "escalation_procedures": False,
            "runbook_count": 2,
            "ir_plan_last_tested": None,
            "backup_strategy": "cloud",
            "backup_frequency": "daily",
            "backup_tested": True,
            "backup_encrypted": True,
            "immutable_backups": False,
            "rto_hours": 8,
            "rpo_hours": 4,
            "communication_plan_exists": True,
            "stakeholder_list_defined": True,
            "regulatory_notification_process": False,
            "critical_gaps": [],
        }
        result = await agent._node_report(state)
        assert result["current_node"] == "report"
        assert "Incident Readiness Assessment Report" in result["report_summary"]
        assert result["report_generated_at"] != ""

    # ---- Routing ----

    def test_route_after_review_approved(self):
        from core.agents.implementations.incident_readiness_agent import IncidentReadinessAgent
        assert IncidentReadinessAgent._route_after_review({"human_approval_status": "approved"}) == "approved"

    def test_route_after_review_rejected(self):
        from core.agents.implementations.incident_readiness_agent import IncidentReadinessAgent
        assert IncidentReadinessAgent._route_after_review({"human_approval_status": "rejected"}) == "rejected"

    def test_route_after_review_default(self):
        from core.agents.implementations.incident_readiness_agent import IncidentReadinessAgent
        assert IncidentReadinessAgent._route_after_review({}) == "approved"

    # ---- Graph ----

    def test_graph_has_correct_nodes(self):
        agent = self._make_agent()
        graph = agent.build_graph()
        expected = {
            "assess_ir_plan", "evaluate_backups", "check_communication",
            "human_review", "score_readiness", "report",
        }
        node_keys = set(graph.nodes.keys()) - {"__start__", "__end__"}
        assert expected.issubset(node_keys)

    def test_write_knowledge_exists(self):
        agent = self._make_agent()
        assert hasattr(agent, "write_knowledge")


# ======================================================================
#  3. CloudSecurityAgent
# ======================================================================


class TestCloudSecurityState:
    """Tests for CloudSecurityAgentState TypedDict."""

    def test_import(self):
        from core.agents.state import CloudSecurityAgentState
        assert CloudSecurityAgentState is not None

    def test_create_minimal(self):
        from core.agents.state import CloudSecurityAgentState
        state: CloudSecurityAgentState = {
            "agent_id": "cloud_v1",
            "vertical_id": "enclave_guard",
        }
        assert state["agent_id"] == "cloud_v1"

    def test_create_full(self):
        from core.agents.state import CloudSecurityAgentState
        state: CloudSecurityAgentState = {
            "agent_id": "cloud_v1",
            "vertical_id": "enclave_guard",
            "cloud_risk_score": 45.0,
            "report_summary": "",
            "report_generated_at": "",
        }
        assert state["cloud_risk_score"] == 45.0

    def test_risk_score_field(self):
        from core.agents.state import CloudSecurityAgentState
        state: CloudSecurityAgentState = {"cloud_risk_score": 80.5}
        assert state["cloud_risk_score"] == 80.5


class TestCloudSecurityAgent:
    """Tests for CloudSecurityAgent implementation."""

    def _make_agent(self, **kwargs):
        from core.agents.implementations.cloud_security_agent import CloudSecurityAgent
        from core.config.agent_schema import AgentInstanceConfig

        config = AgentInstanceConfig(
            agent_id="cloud_v1",
            agent_type="cloud_security",
            name="Cloud Security",
            vertical_id="enclave_guard",
            params={"company_name": "Enclave Guard"},
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

        return CloudSecurityAgent(
            config=config, db=db, embedder=embedder, anthropic_client=llm,
        )

    # ---- Registration & Construction ----

    def test_registration(self):
        from core.agents.implementations.cloud_security_agent import CloudSecurityAgent  # noqa: F401
        from core.agents.registry import AGENT_IMPLEMENTATIONS
        assert "cloud_security" in AGENT_IMPLEMENTATIONS

    def test_agent_type(self):
        from core.agents.implementations.cloud_security_agent import CloudSecurityAgent
        assert CloudSecurityAgent.agent_type == "cloud_security"

    def test_build_graph(self):
        agent = self._make_agent()
        graph = agent.build_graph()
        assert graph is not None

    def test_get_state_class(self):
        from core.agents.state import CloudSecurityAgentState
        agent = self._make_agent()
        assert agent.get_state_class() is CloudSecurityAgentState

    def test_get_tools(self):
        agent = self._make_agent()
        assert agent.get_tools() == []

    def test_repr(self):
        agent = self._make_agent()
        r = repr(agent)
        assert "CloudSecurityAgent" in r
        assert "cloud_v1" in r

    # ---- Initial State ----

    def test_prepare_initial_state(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state({}, "run-1")
        assert state["cloud_providers"] == []
        assert state["public_buckets"] == []
        assert state["cloud_risk_score"] == 0.0
        assert state["cloud_risk_level"] == "low"
        assert state["findings_approved"] is False

    # ---- Constants ----

    def test_constants_cloud_providers(self):
        from core.agents.implementations import cloud_security_agent
        assert "aws" in cloud_security_agent.CLOUD_PROVIDERS
        assert "azure" in cloud_security_agent.CLOUD_PROVIDERS

    def test_constants_misconfig_categories(self):
        from core.agents.implementations import cloud_security_agent
        cats = cloud_security_agent.MISCONFIG_CATEGORIES
        assert "storage" in cats
        assert "iam" in cats

    def test_constants_benchmarks(self):
        from core.agents.implementations import cloud_security_agent
        assert "cis_level_1" in cloud_security_agent.CLOUD_BENCHMARKS

    def test_system_prompt(self):
        from core.agents.implementations import cloud_security_agent
        assert "{company_name}" in cloud_security_agent.CLOUD_SYSTEM_PROMPT

    # ---- Node 1: identify_cloud ----

    @pytest.mark.asyncio
    async def test_node_identify_cloud_empty(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state(
            {"questionnaire": {}, "company_name": "Empty"}, "run-1"
        )
        result = await agent._node_identify_cloud(state)
        assert result["current_node"] == "identify_cloud"
        assert result["cloud_providers"] == []

    @pytest.mark.asyncio
    async def test_node_identify_cloud_with_providers(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state(
            {
                "questionnaire": {
                    "cloud_providers": ["aws", "gcp"],
                    "cloud_services": ["S3", "EC2", "GKE"],
                },
                "company_name": "MultiCloud",
            },
            "run-2",
        )
        result = await agent._node_identify_cloud(state)
        assert "aws" in result["cloud_providers"]
        assert len(result["cloud_services_in_use"]) == 3

    @pytest.mark.asyncio
    async def test_node_identify_cloud_string_providers(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state(
            {"questionnaire": {"cloud_providers": "aws, azure"}, "company_name": "StrCo"},
            "run-3",
        )
        result = await agent._node_identify_cloud(state)
        assert len(result["cloud_providers"]) == 2

    # ---- Node 2: scan_configs ----

    @pytest.mark.asyncio
    async def test_node_scan_configs_no_issues(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state(
            {
                "questionnaire": {
                    "storage_encryption_enabled": True,
                    "database_encryption_enabled": True,
                    "transit_encryption_enabled": True,
                    "cloud_trail_enabled": True,
                    "flow_logs_enabled": True,
                },
            },
            "run-1",
        )
        state["cloud_providers"] = ["aws"]
        result = await agent._node_scan_configs(state)
        assert result["public_buckets"] == []
        assert result["unencrypted_resources"] == []
        assert result["logging_gaps"] == []

    @pytest.mark.asyncio
    async def test_node_scan_configs_public_bucket(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state(
            {
                "questionnaire": {
                    "public_storage_buckets": [{"name": "my-public-bucket"}],
                    "storage_encryption_enabled": True,
                    "database_encryption_enabled": True,
                    "transit_encryption_enabled": True,
                    "cloud_trail_enabled": True,
                    "flow_logs_enabled": True,
                },
            },
            "run-2",
        )
        state["cloud_providers"] = ["aws"]
        result = await agent._node_scan_configs(state)
        assert len(result["public_buckets"]) == 1

    @pytest.mark.asyncio
    async def test_node_scan_configs_no_encryption(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state(
            {"questionnaire": {}}, "run-3"
        )
        state["cloud_providers"] = ["aws"]
        result = await agent._node_scan_configs(state)
        assert len(result["unencrypted_resources"]) >= 2

    # ---- Node 3: analyze_misconfigs ----

    @pytest.mark.asyncio
    async def test_node_analyze_misconfigs_clean(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state({}, "run-1")
        state["public_buckets"] = []
        state["open_security_groups"] = []
        state["unencrypted_resources"] = []
        state["logging_gaps"] = []
        result = await agent._node_analyze_misconfigs(state)
        assert result["cloud_risk_score"] == 0.0
        assert result["cloud_risk_level"] == "low"

    @pytest.mark.asyncio
    async def test_node_analyze_misconfigs_with_issues(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state({}, "run-2")
        state["public_buckets"] = [{"name": "pub-bucket"}]
        state["open_security_groups"] = [{"rule": "0.0.0.0/0"}]
        state["unencrypted_resources"] = [{"resource_type": "database", "detail": "no enc"}]
        state["logging_gaps"] = [{"service": "audit", "detail": "no logging"}]
        result = await agent._node_analyze_misconfigs(state)
        assert result["cloud_risk_score"] > 0
        assert len(result["misconfig_findings"]) == 4
        assert result["misconfig_by_severity"].get("critical", 0) >= 1

    # ---- Node 4: human_review ----

    @pytest.mark.asyncio
    async def test_node_human_review(self):
        agent = self._make_agent()
        state = {"all_findings": [{"finding": "x"}], "cloud_risk_score": 50}
        result = await agent._node_human_review(state)
        assert result["requires_human_approval"] is True

    # ---- Node 5: save_findings ----

    @pytest.mark.asyncio
    async def test_node_save_findings(self):
        agent = self._make_agent()
        agent.db.client.table.return_value.insert.return_value.execute = MagicMock()
        state = agent._prepare_initial_state({"company_name": "SaveCo"}, "run-1")
        state["all_findings"] = [{"finding": "pub bucket", "severity": "critical"}]
        state["cloud_providers"] = ["aws"]
        result = await agent._node_save_findings(state)
        assert result["findings_approved"] is True

    # ---- Node 6: report ----

    @pytest.mark.asyncio
    async def test_node_report(self):
        agent = self._make_agent()
        state = {
            "task_input": {"company_name": "RptCo"},
            "all_findings": [{"severity": "high"}],
            "cloud_providers": ["aws"],
            "cloud_services_in_use": ["S3"],
            "cloud_risk_score": 40.0,
            "cloud_risk_level": "medium",
            "misconfig_by_severity": {"high": 1},
            "misconfig_by_category": {"storage": 1},
            "benchmark_compliance": {"cis_level_1": 0.80, "cis_level_2": 0.60},
            "public_buckets": [],
            "open_security_groups": [],
            "unencrypted_resources": [],
            "logging_gaps": [],
        }
        result = await agent._node_report(state)
        assert result["current_node"] == "report"
        assert "Cloud Security Assessment Report" in result["report_summary"]
        assert result["report_generated_at"] != ""

    # ---- Routing ----

    def test_route_after_review_approved(self):
        from core.agents.implementations.cloud_security_agent import CloudSecurityAgent
        assert CloudSecurityAgent._route_after_review({"human_approval_status": "approved"}) == "approved"

    def test_route_after_review_rejected(self):
        from core.agents.implementations.cloud_security_agent import CloudSecurityAgent
        assert CloudSecurityAgent._route_after_review({"human_approval_status": "rejected"}) == "rejected"

    def test_route_after_review_default(self):
        from core.agents.implementations.cloud_security_agent import CloudSecurityAgent
        assert CloudSecurityAgent._route_after_review({}) == "approved"

    # ---- Graph ----

    def test_graph_has_correct_nodes(self):
        agent = self._make_agent()
        graph = agent.build_graph()
        expected = {
            "identify_cloud", "scan_configs", "analyze_misconfigs",
            "human_review", "save_findings", "report",
        }
        node_keys = set(graph.nodes.keys()) - {"__start__", "__end__"}
        assert expected.issubset(node_keys)

    def test_write_knowledge_exists(self):
        agent = self._make_agent()
        assert hasattr(agent, "write_knowledge")


# ======================================================================
#  4. SecurityTrainerAgent
# ======================================================================


class TestSecurityTrainerState:
    """Tests for SecurityTrainerAgentState TypedDict."""

    def test_import(self):
        from core.agents.state import SecurityTrainerAgentState
        assert SecurityTrainerAgentState is not None

    def test_create_minimal(self):
        from core.agents.state import SecurityTrainerAgentState
        state: SecurityTrainerAgentState = {
            "agent_id": "trainer_v1",
            "vertical_id": "enclave_guard",
        }
        assert state["agent_id"] == "trainer_v1"

    def test_create_full(self):
        from core.agents.state import SecurityTrainerAgentState
        state: SecurityTrainerAgentState = {
            "agent_id": "trainer_v1",
            "vertical_id": "enclave_guard",
            "human_risk_score": 55.0,
            "training_modules": [],
            "phishing_scenarios": [],
            "content_approved": False,
            "report_summary": "",
            "report_generated_at": "",
        }
        assert state["human_risk_score"] == 55.0
        assert state["content_approved"] is False

    def test_risk_score_field(self):
        from core.agents.state import SecurityTrainerAgentState
        state: SecurityTrainerAgentState = {"human_risk_score": 80.0}
        assert state["human_risk_score"] == 80.0


class TestSecurityTrainerAgent:
    """Tests for SecurityTrainerAgent implementation."""

    def _make_agent(self, **kwargs):
        from core.agents.implementations.security_trainer_agent import SecurityTrainerAgent
        from core.config.agent_schema import AgentInstanceConfig

        config = AgentInstanceConfig(
            agent_id="trainer_v1",
            agent_type="security_trainer",
            name="Security Trainer",
            vertical_id="enclave_guard",
            params={"company_name": "Enclave Guard"},
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

        return SecurityTrainerAgent(
            config=config, db=db, embedder=embedder, anthropic_client=llm,
        )

    # ---- Registration & Construction ----

    def test_registration(self):
        from core.agents.implementations.security_trainer_agent import SecurityTrainerAgent  # noqa: F401
        from core.agents.registry import AGENT_IMPLEMENTATIONS
        assert "security_trainer" in AGENT_IMPLEMENTATIONS

    def test_agent_type(self):
        from core.agents.implementations.security_trainer_agent import SecurityTrainerAgent
        assert SecurityTrainerAgent.agent_type == "security_trainer"

    def test_build_graph(self):
        agent = self._make_agent()
        graph = agent.build_graph()
        assert graph is not None

    def test_get_state_class(self):
        from core.agents.state import SecurityTrainerAgentState
        agent = self._make_agent()
        assert agent.get_state_class() is SecurityTrainerAgentState

    def test_get_tools(self):
        agent = self._make_agent()
        assert agent.get_tools() == []

    def test_repr(self):
        agent = self._make_agent()
        r = repr(agent)
        assert "SecurityTrainerAgent" in r
        assert "trainer_v1" in r

    # ---- Initial State ----

    def test_prepare_initial_state(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state({"employee_count": 150}, "run-1")
        assert state["employee_count"] == 150
        assert state["department_risks"] == []
        assert state["phishing_click_rate"] == 0.0
        assert state["training_modules"] == []
        assert state["phishing_scenarios"] == []
        assert state["content_approved"] is False

    # ---- Constants ----

    def test_constants_training_topics(self):
        from core.agents.implementations import security_trainer_agent
        topics = security_trainer_agent.TRAINING_TOPICS
        assert "phishing_awareness" in topics
        assert "password_hygiene" in topics
        assert len(topics) >= 8

    def test_constants_phishing_templates(self):
        from core.agents.implementations import security_trainer_agent
        templates = security_trainer_agent.PHISHING_TEMPLATES
        assert "credential_harvest" in templates
        assert "ceo_fraud" in templates

    def test_constants_risk_categories(self):
        from core.agents.implementations import security_trainer_agent
        cats = security_trainer_agent.RISK_CATEGORIES
        assert "critical" in cats
        assert cats["critical"]["min_score"] == 80

    def test_system_prompt(self):
        from core.agents.implementations import security_trainer_agent
        assert "{company_name}" in security_trainer_agent.TRAINER_SYSTEM_PROMPT
        assert "{employee_count}" in security_trainer_agent.TRAINER_SYSTEM_PROMPT

    # ---- Node 1: assess_human_risk ----

    @pytest.mark.asyncio
    async def test_node_assess_human_risk_default(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state(
            {"questionnaire": {}, "company_name": "TestCo", "employee_count": 100},
            "run-1",
        )
        result = await agent._node_assess_human_risk(state)
        assert result["current_node"] == "assess_human_risk"
        assert result["employee_count"] == 100
        assert result["human_risk_score"] >= 0

    @pytest.mark.asyncio
    async def test_node_assess_human_risk_high_click_rate(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state(
            {
                "questionnaire": {"phishing_click_rate": 0.8, "training_completion_rate": 0.1},
                "company_name": "HighRisk",
                "employee_count": 200,
            },
            "run-2",
        )
        result = await agent._node_assess_human_risk(state)
        assert result["human_risk_score"] >= 60
        assert result["phishing_click_rate"] == 0.8

    @pytest.mark.asyncio
    async def test_node_assess_human_risk_departments(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state(
            {
                "questionnaire": {
                    "departments": ["executive", "engineering", "finance"],
                },
                "company_name": "DeptCo",
            },
            "run-3",
        )
        result = await agent._node_assess_human_risk(state)
        dept_names = [d["department"] for d in result["department_risks"]]
        assert "executive" in dept_names
        high_risk = [d for d in result["department_risks"] if d["risk_level"] == "high"]
        assert len(high_risk) >= 2  # executive & finance

    # ---- Node 2: generate_training ----

    @pytest.mark.asyncio
    async def test_node_generate_training_low_risk(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state({}, "run-1")
        state["human_risk_score"] = 20.0
        state["phishing_click_rate"] = 0.05
        state["department_risks"] = []
        result = await agent._node_generate_training(state)
        assert result["current_node"] == "generate_training"
        assert result["module_count"] >= 1
        topics = [m["topic"] for m in result["training_modules"]]
        assert "phishing_awareness" in topics

    @pytest.mark.asyncio
    async def test_node_generate_training_high_risk(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state({}, "run-2")
        state["human_risk_score"] = 75.0
        state["phishing_click_rate"] = 0.5
        state["department_risks"] = [
            {"department": "executive", "risk_level": "high"},
        ]
        result = await agent._node_generate_training(state)
        assert result["module_count"] >= 4
        topics = result["training_topics_covered"]
        assert "password_hygiene" in topics
        assert "executive_security" in topics

    # ---- Node 3: create_phishing_sims ----

    @pytest.mark.asyncio
    async def test_node_create_phishing_sims_low_risk(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state({}, "run-1")
        state["human_risk_score"] = 20.0
        state["department_risks"] = []
        result = await agent._node_create_phishing_sims(state)
        assert result["current_node"] == "create_phishing_sims"
        assert result["simulation_count"] >= 2
        difficulties = [s["difficulty"] for s in result["phishing_scenarios"]]
        assert "easy" in difficulties

    @pytest.mark.asyncio
    async def test_node_create_phishing_sims_high_risk(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state({}, "run-2")
        state["human_risk_score"] = 70.0
        state["department_risks"] = [{"department": "finance", "risk_level": "high"}]
        result = await agent._node_create_phishing_sims(state)
        assert result["simulation_count"] >= 4
        difficulties = [s["difficulty"] for s in result["phishing_scenarios"]]
        assert "hard" in difficulties

    # ---- Node 4: human_review ----

    @pytest.mark.asyncio
    async def test_node_human_review(self):
        agent = self._make_agent()
        state = {"training_modules": [{"topic": "a"}], "phishing_scenarios": []}
        result = await agent._node_human_review(state)
        assert result["requires_human_approval"] is True

    # ---- Node 5: deliver ----

    @pytest.mark.asyncio
    async def test_node_deliver(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state({"company_name": "DeliverCo"}, "run-1")
        state["training_modules"] = [
            {"topic": "phishing", "title": "Phishing 101", "target_audience": "all"},
        ]
        state["phishing_scenarios"] = [
            {"name": "Cred Harvest", "target_department": "all"},
        ]
        state["human_risk_score"] = 50
        state["phishing_click_rate"] = 0.3
        result = await agent._node_deliver(state)
        assert result["current_node"] == "deliver"
        assert result["modules_delivered"] == 1
        assert result["simulations_launched"] == 1
        assert result["content_approved"] is True
        assert len(result["delivery_schedule"]) == 2

    @pytest.mark.asyncio
    async def test_node_deliver_empty(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state({"company_name": "EmptyCo"}, "run-2")
        state["training_modules"] = []
        state["phishing_scenarios"] = []
        state["human_risk_score"] = 0
        state["phishing_click_rate"] = 0
        result = await agent._node_deliver(state)
        assert result["modules_delivered"] == 0
        assert result["simulations_launched"] == 0

    # ---- Node 6: report ----

    @pytest.mark.asyncio
    async def test_node_report(self):
        agent = self._make_agent()
        state = {
            "task_input": {"company_name": "RptCo"},
            "employee_count": 100,
            "human_risk_score": 55.0,
            "phishing_click_rate": 0.25,
            "training_completion_rate": 0.7,
            "department_risks": [{"department": "hr", "risk_level": "high"}],
            "training_modules": [
                {"title": "Phishing 101", "difficulty": "beginner", "duration_minutes": 20, "target_audience": "all"},
            ],
            "phishing_scenarios": [
                {"name": "Cred Harvest", "difficulty": "easy", "target_department": "all"},
            ],
            "phishing_difficulty_mix": {"easy": 1, "medium": 0, "hard": 0},
        }
        result = await agent._node_report(state)
        assert result["current_node"] == "report"
        assert "Security Awareness Training Report" in result["report_summary"]
        assert result["report_generated_at"] != ""

    # ---- Routing ----

    def test_route_after_review_approved(self):
        from core.agents.implementations.security_trainer_agent import SecurityTrainerAgent
        assert SecurityTrainerAgent._route_after_review({"human_approval_status": "approved"}) == "approved"

    def test_route_after_review_rejected(self):
        from core.agents.implementations.security_trainer_agent import SecurityTrainerAgent
        assert SecurityTrainerAgent._route_after_review({"human_approval_status": "rejected"}) == "rejected"

    def test_route_after_review_default(self):
        from core.agents.implementations.security_trainer_agent import SecurityTrainerAgent
        assert SecurityTrainerAgent._route_after_review({}) == "approved"

    # ---- Graph ----

    def test_graph_has_correct_nodes(self):
        agent = self._make_agent()
        graph = agent.build_graph()
        expected = {
            "assess_human_risk", "generate_training", "create_phishing_sims",
            "human_review", "deliver", "report",
        }
        node_keys = set(graph.nodes.keys()) - {"__start__", "__end__"}
        assert expected.issubset(node_keys)

    def test_write_knowledge_exists(self):
        agent = self._make_agent()
        assert hasattr(agent, "write_knowledge")


# ======================================================================
#  5. RemediationGuideAgent
# ======================================================================


class TestRemediationGuideState:
    """Tests for RemediationGuideAgentState TypedDict."""

    def test_import(self):
        from core.agents.state import RemediationGuideAgentState
        assert RemediationGuideAgentState is not None

    def test_create_minimal(self):
        from core.agents.state import RemediationGuideAgentState
        state: RemediationGuideAgentState = {
            "agent_id": "remed_v1",
            "vertical_id": "enclave_guard",
        }
        assert state["agent_id"] == "remed_v1"

    def test_create_full(self):
        from core.agents.state import RemediationGuideAgentState
        state: RemediationGuideAgentState = {
            "agent_id": "remed_v1",
            "vertical_id": "enclave_guard",
            "verification_results": [],
            "report_summary": "",
            "report_generated_at": "",
        }
        assert state["verification_results"] == []

    def test_tasks_created_field(self):
        from core.agents.state import RemediationGuideAgentState
        state: RemediationGuideAgentState = {"tasks_created": 5}
        assert state["tasks_created"] == 5

    def test_report_fields(self):
        from core.agents.state import RemediationGuideAgentState
        state: RemediationGuideAgentState = {
            "report_summary": "Fix it",
            "report_generated_at": "2025-01-01",
        }
        assert state["report_summary"] == "Fix it"


class TestRemediationGuideAgent:
    """Tests for RemediationGuideAgent implementation."""

    def _make_agent(self, **kwargs):
        from core.agents.implementations.remediation_guide_agent import RemediationGuideAgent
        from core.config.agent_schema import AgentInstanceConfig

        config = AgentInstanceConfig(
            agent_id="remed_v1",
            agent_type="remediation_guide",
            name="Remediation Guide",
            vertical_id="enclave_guard",
            params={"company_name": "Enclave Guard"},
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

        return RemediationGuideAgent(
            config=config, db=db, embedder=embedder, anthropic_client=llm,
        )

    # ---- Registration & Construction ----

    def test_registration(self):
        from core.agents.implementations.remediation_guide_agent import RemediationGuideAgent  # noqa: F401
        from core.agents.registry import AGENT_IMPLEMENTATIONS
        assert "remediation_guide" in AGENT_IMPLEMENTATIONS

    def test_agent_type(self):
        from core.agents.implementations.remediation_guide_agent import RemediationGuideAgent
        assert RemediationGuideAgent.agent_type == "remediation_guide"

    def test_build_graph(self):
        agent = self._make_agent()
        graph = agent.build_graph()
        assert graph is not None

    def test_get_state_class(self):
        from core.agents.state import RemediationGuideAgentState
        agent = self._make_agent()
        assert agent.get_state_class() is RemediationGuideAgentState

    def test_get_tools(self):
        agent = self._make_agent()
        assert agent.get_tools() == []

    def test_repr(self):
        agent = self._make_agent()
        r = repr(agent)
        assert "RemediationGuideAgent" in r
        assert "remed_v1" in r

    # ---- Initial State ----

    def test_prepare_initial_state(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state(
            {"source_agents": ["vuln_scanner", "iam_analyst"]}, "run-1"
        )
        assert state["source_findings"] == []
        assert state["source_agents"] == ["vuln_scanner", "iam_analyst"]
        assert state["finding_count"] == 0
        assert state["remediation_plans"] == []
        assert state["tasks_created"] == []
        assert state["plans_approved"] is False

    def test_prepare_initial_state_no_sources(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state({}, "run-2")
        assert state["source_agents"] == []

    # ---- Constants ----

    def test_constants_priority_weights(self):
        from core.agents.implementations import remediation_guide_agent
        weights = remediation_guide_agent.PRIORITY_WEIGHTS
        assert "severity" in weights
        assert abs(sum(weights.values()) - 1.0) < 0.01

    def test_constants_remediation_categories(self):
        from core.agents.implementations import remediation_guide_agent
        cats = remediation_guide_agent.REMEDIATION_CATEGORIES
        assert "configuration" in cats
        assert "patch" in cats
        assert "architecture" in cats

    def test_constants_verification_methods(self):
        from core.agents.implementations import remediation_guide_agent
        methods = remediation_guide_agent.VERIFICATION_METHODS
        assert "rescan" in methods
        assert "config_review" in methods

    def test_system_prompt(self):
        from core.agents.implementations import remediation_guide_agent
        assert "{company_name}" in remediation_guide_agent.REMEDIATION_SYSTEM_PROMPT
        assert "{finding_count}" in remediation_guide_agent.REMEDIATION_SYSTEM_PROMPT

    # ---- Node 1: load_findings ----

    @pytest.mark.asyncio
    async def test_node_load_findings_from_db(self):
        agent = self._make_agent()
        mock_result = MagicMock()
        mock_result.data = [
            {"id": "f1", "finding": "Test finding", "severity": "high"},
        ]
        agent.db.client.table.return_value.select.return_value.eq.return_value.eq.return_value.execute.return_value = mock_result

        state = agent._prepare_initial_state({"company_name": "DBCo"}, "run-1")
        result = await agent._node_load_findings(state)
        assert result["current_node"] == "load_findings"
        assert result["finding_count"] == 1

    @pytest.mark.asyncio
    async def test_node_load_findings_from_task(self):
        agent = self._make_agent()
        mock_result = MagicMock()
        mock_result.data = []
        agent.db.client.table.return_value.select.return_value.eq.return_value.eq.return_value.execute.return_value = mock_result

        state = agent._prepare_initial_state(
            {
                "company_name": "TaskCo",
                "findings": [
                    {"finding": "inline finding 1", "severity": "critical"},
                    {"finding": "inline finding 2", "severity": "low"},
                ],
            },
            "run-2",
        )
        result = await agent._node_load_findings(state)
        assert result["finding_count"] == 2

    @pytest.mark.asyncio
    async def test_node_load_findings_db_error(self):
        agent = self._make_agent()
        agent.db.client.table.side_effect = Exception("DB down")

        state = agent._prepare_initial_state(
            {"company_name": "ErrCo", "findings": [{"finding": "fallback", "severity": "medium"}]},
            "run-3",
        )
        result = await agent._node_load_findings(state)
        assert result["finding_count"] == 1

    @pytest.mark.asyncio
    async def test_node_load_findings_dedup(self):
        agent = self._make_agent()
        mock_result = MagicMock()
        mock_result.data = [
            {"id": "f1", "finding": "Duplicate finding", "severity": "high"},
            {"id": "f2", "finding": "Duplicate finding", "severity": "high"},
            {"id": "f3", "finding": "Unique finding", "severity": "low"},
        ]
        agent.db.client.table.return_value.select.return_value.eq.return_value.eq.return_value.execute.return_value = mock_result

        state = agent._prepare_initial_state({"company_name": "DedupCo"}, "run-4")
        result = await agent._node_load_findings(state)
        assert result["finding_count"] == 2

    # ---- Node 2: prioritize ----

    @pytest.mark.asyncio
    async def test_node_prioritize_empty(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state({}, "run-1")
        state["source_findings"] = []
        result = await agent._node_prioritize(state)
        assert result["current_node"] == "prioritize"
        assert result["prioritized_findings"] == []
        assert result["critical_count"] == 0

    @pytest.mark.asyncio
    async def test_node_prioritize_sorted_by_score(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state({}, "run-2")
        state["source_findings"] = [
            {"id": "low1", "finding": "Low item", "severity": "low"},
            {"id": "crit1", "finding": "Critical item", "severity": "critical"},
            {"id": "med1", "finding": "Medium item", "severity": "medium"},
        ]
        result = await agent._node_prioritize(state)
        assert result["critical_count"] == 1
        assert result["low_count"] == 1
        scores = [f["priority_score"] for f in result["prioritized_findings"]]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_node_prioritize_network_exploitability(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state({}, "run-3")
        state["source_findings"] = [
            {"id": "net1", "finding": "Open port", "severity": "high", "domain": "network"},
            {"id": "iam1", "finding": "Weak password", "severity": "high", "domain": "iam"},
        ]
        result = await agent._node_prioritize(state)
        # Network finding should score higher due to exploitability
        findings = result["prioritized_findings"]
        net = next(f for f in findings if f["finding_id"] == "net1")
        iam = next(f for f in findings if f["finding_id"] == "iam1")
        assert net["priority_score"] >= iam["priority_score"]

    # ---- Node 3: generate_steps ----

    @pytest.mark.asyncio
    async def test_node_generate_steps_empty(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state({}, "run-1")
        state["prioritized_findings"] = []
        result = await agent._node_generate_steps(state)
        assert result["current_node"] == "generate_steps"
        assert result["remediation_plans"] == []
        assert result["total_steps"] == 0

    @pytest.mark.asyncio
    async def test_node_generate_steps_with_findings(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state({}, "run-2")
        state["prioritized_findings"] = [
            {
                "finding_id": "f1",
                "finding": "Public S3 bucket",
                "severity": "critical",
                "category": "storage",
                "domain": "cloud",
                "recommendation": "Enable S3 block public access",
            },
        ]
        result = await agent._node_generate_steps(state)
        assert len(result["remediation_plans"]) == 1
        plan = result["remediation_plans"][0]
        assert len(plan["steps"]) >= 3
        assert plan["verification_method"] == "rescan"
        assert result["total_steps"] >= 3
        assert result["estimated_total_hours"] > 0

    @pytest.mark.asyncio
    async def test_node_generate_steps_patch_category(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state({}, "run-3")
        state["prioritized_findings"] = [
            {
                "finding_id": "f2",
                "finding": "Outdated software",
                "severity": "high",
                "category": "compute",
                "domain": "network",
                "recommendation": "Update and patch the affected systems",
            },
        ]
        result = await agent._node_generate_steps(state)
        plan = result["remediation_plans"][0]
        assert plan["category"] == "patch"

    # ---- Node 4: human_review ----

    @pytest.mark.asyncio
    async def test_node_human_review(self):
        agent = self._make_agent()
        state = {"remediation_plans": [{"finding_id": "f1"}], "estimated_total_hours": 10}
        result = await agent._node_human_review(state)
        assert result["requires_human_approval"] is True

    # ---- Node 5: create_tasks ----

    @pytest.mark.asyncio
    async def test_node_create_tasks(self):
        agent = self._make_agent()
        agent.db.client.table.return_value.insert.return_value.execute = MagicMock()

        state = agent._prepare_initial_state({"company_name": "TaskCo"}, "run-1")
        state["remediation_plans"] = [
            {
                "finding_id": "f1",
                "finding": "Public bucket",
                "severity": "critical",
                "category": "configuration",
                "estimated_hours": 3,
                "steps": [{"step_number": 1, "action": "Fix it"}],
                "verification_method": "rescan",
            },
        ]
        result = await agent._node_create_tasks(state)
        assert result["current_node"] == "create_tasks"
        assert result["tasks_count"] == 1
        assert result["plans_approved"] is True
        task = result["tasks_created"][0]
        assert task["severity"] == "critical"
        assert "CRITICAL" in task["title"]

    @pytest.mark.asyncio
    async def test_node_create_tasks_due_dates(self):
        agent = self._make_agent()
        agent.db.client.table.return_value.insert.return_value.execute = MagicMock()

        state = agent._prepare_initial_state({"company_name": "DueCo"}, "run-2")
        state["remediation_plans"] = [
            {"finding_id": "c1", "finding": "crit", "severity": "critical",
             "category": "configuration", "estimated_hours": 2, "steps": [], "verification_method": "rescan"},
            {"finding_id": "l1", "finding": "low", "severity": "low",
             "category": "policy", "estimated_hours": 1, "steps": [], "verification_method": "attestation"},
        ]
        result = await agent._node_create_tasks(state)
        assert result["tasks_count"] == 2
        crit_task = next(t for t in result["tasks_created"] if t["severity"] == "critical")
        low_task = next(t for t in result["tasks_created"] if t["severity"] == "low")
        assert crit_task["due_date"] < low_task["due_date"]

    @pytest.mark.asyncio
    async def test_node_create_tasks_db_error(self):
        agent = self._make_agent()
        agent.db.client.table.return_value.insert.return_value.execute.side_effect = Exception("DB fail")

        state = agent._prepare_initial_state({"company_name": "ErrCo"}, "run-3")
        state["remediation_plans"] = [
            {"finding_id": "f1", "finding": "test", "severity": "medium",
             "category": "configuration", "estimated_hours": 2, "steps": [], "verification_method": "config_review"},
        ]
        result = await agent._node_create_tasks(state)
        # Should still produce the task record even if DB fails
        assert result["tasks_count"] == 1

    # ---- Node 6: verify ----

    @pytest.mark.asyncio
    async def test_node_verify(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state({"company_name": "VerifyCo"}, "run-1")
        state["tasks_created"] = [
            {"task_id": "t1", "finding_id": "f1", "verification_method": "rescan"},
        ]
        state["critical_count"] = 1
        state["high_count"] = 0
        state["medium_count"] = 0
        state["low_count"] = 0
        state["estimated_total_hours"] = 5
        result = await agent._node_verify(state)
        assert result["current_node"] == "verify"
        assert len(result["verification_results"]) == 1
        assert result["verification_results"][0]["status"] == "pending"
        assert result["verified_count"] == 0

    @pytest.mark.asyncio
    async def test_node_verify_empty_tasks(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state({"company_name": "EmptyCo"}, "run-2")
        state["tasks_created"] = []
        state["critical_count"] = 0
        state["high_count"] = 0
        state["medium_count"] = 0
        state["low_count"] = 0
        state["estimated_total_hours"] = 0
        result = await agent._node_verify(state)
        assert result["verification_results"] == []

    # ---- Node 7: report ----

    @pytest.mark.asyncio
    async def test_node_report(self):
        agent = self._make_agent()
        state = {
            "task_input": {"company_name": "RptCo"},
            "source_agents": ["vuln_scanner"],
            "finding_count": 3,
            "critical_count": 1,
            "high_count": 1,
            "medium_count": 1,
            "low_count": 0,
            "remediation_plans": [
                {"category": "configuration"},
                {"category": "patch"},
            ],
            "total_steps": 8,
            "estimated_total_hours": 12.0,
            "tasks_created": [{"task_id": "t1"}, {"task_id": "t2"}],
            "verified_count": 0,
            "failed_verification_count": 0,
            "prioritized_findings": [
                {"severity": "critical", "finding": "Public bucket"},
                {"severity": "high", "finding": "Open port"},
            ],
        }
        result = await agent._node_report(state)
        assert result["current_node"] == "report"
        assert "Remediation Guide Report" in result["report_summary"]
        assert result["report_generated_at"] != ""

    @pytest.mark.asyncio
    async def test_node_report_empty(self):
        agent = self._make_agent()
        state = {
            "task_input": {"company_name": "EmptyCo"},
            "source_agents": [],
            "finding_count": 0,
            "critical_count": 0,
            "high_count": 0,
            "medium_count": 0,
            "low_count": 0,
            "remediation_plans": [],
            "total_steps": 0,
            "estimated_total_hours": 0,
            "tasks_created": [],
            "verified_count": 0,
            "failed_verification_count": 0,
            "prioritized_findings": [],
        }
        result = await agent._node_report(state)
        assert "Remediation Guide Report" in result["report_summary"]

    # ---- Routing ----

    def test_route_after_review_approved(self):
        from core.agents.implementations.remediation_guide_agent import RemediationGuideAgent
        assert RemediationGuideAgent._route_after_review({"human_approval_status": "approved"}) == "approved"

    def test_route_after_review_rejected(self):
        from core.agents.implementations.remediation_guide_agent import RemediationGuideAgent
        assert RemediationGuideAgent._route_after_review({"human_approval_status": "rejected"}) == "rejected"

    def test_route_after_review_default(self):
        from core.agents.implementations.remediation_guide_agent import RemediationGuideAgent
        assert RemediationGuideAgent._route_after_review({}) == "approved"

    # ---- Graph ----

    def test_graph_has_correct_nodes(self):
        agent = self._make_agent()
        graph = agent.build_graph()
        expected = {
            "load_findings", "prioritize", "generate_steps",
            "human_review", "create_tasks", "verify", "report",
        }
        node_keys = set(graph.nodes.keys()) - {"__start__", "__end__"}
        assert expected.issubset(node_keys)

    def test_write_knowledge_exists(self):
        agent = self._make_agent()
        assert hasattr(agent, "write_knowledge")
