# tests/test_domain_agents/__init__.py must exist
"""
Tests for Security Domain Expert Agents — Batch 1.

Covers the first 5 Enclave Guard cybersecurity domain agents:
    1. VulnScannerAgent (vuln_scanner)
    2. NetworkAnalystAgent (network_analyst)
    3. AppSecReviewerAgent (appsec_reviewer)
    4. ComplianceMapperAgent (compliance_mapper)
    5. RiskReporterAgent (risk_reporter)

Each agent tests:
    - State TypedDict import and creation
    - Agent registration, construction, state class
    - Initial state preparation
    - Module-level constants
    - All graph nodes (async, mocked DB/LLM)
    - Graph construction and routing
    - System prompt
    - __repr__ and write_knowledge
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ══════════════════════════════════════════════════════════════════════
#  1.  VulnScannerAgent
# ══════════════════════════════════════════════════════════════════════


class TestVulnScannerState:
    """Tests for VulnScannerAgentState TypedDict."""

    def test_import(self):
        from core.agents.state import VulnScannerAgentState
        assert VulnScannerAgentState is not None

    def test_create_minimal(self):
        from core.agents.state import VulnScannerAgentState
        state: VulnScannerAgentState = {
            "agent_id": "vuln_v1",
            "vertical_id": "enclave_guard",
        }
        assert state["agent_id"] == "vuln_v1"

    def test_create_full(self):
        from core.agents.state import VulnScannerAgentState
        state: VulnScannerAgentState = {
            "agent_id": "vuln_v1",
            "vertical_id": "enclave_guard",
            "company_domain": "example.com",
            "company_name": "Example Corp",
            "scan_type": "full",
            "scan_targets": [],
            "ssl_findings": [],
            "header_findings": [],
            "network_findings": [],
            "raw_scan_data": {},
            "findings": [],
            "risk_score": 0.0,
            "executive_summary": "",
            "critical_count": 0,
            "high_count": 0,
            "medium_count": 0,
            "low_count": 0,
            "assessment_id": "",
            "findings_saved": False,
            "report_summary": "",
            "report_generated_at": "",
        }
        assert state["company_domain"] == "example.com"
        assert state["risk_score"] == 0.0
        assert state["findings_saved"] is False

    def test_severity_count_fields(self):
        from core.agents.state import VulnScannerAgentState
        state: VulnScannerAgentState = {
            "critical_count": 3,
            "high_count": 5,
            "medium_count": 7,
            "low_count": 10,
        }
        assert state["critical_count"] == 3
        assert state["high_count"] == 5

    def test_scan_data_fields(self):
        from core.agents.state import VulnScannerAgentState
        state: VulnScannerAgentState = {
            "ssl_findings": [{"severity": "high"}],
            "header_findings": [{"severity": "medium"}],
            "network_findings": [{"severity": "critical"}],
        }
        assert len(state["ssl_findings"]) == 1
        assert state["network_findings"][0]["severity"] == "critical"


class TestVulnScannerAgent:
    """Tests for VulnScannerAgent implementation."""

    def _make_agent(self, **kwargs):
        """Create a VulnScannerAgent with mocked dependencies."""
        from core.agents.implementations.vuln_scanner_agent import VulnScannerAgent
        from core.config.agent_schema import AgentInstanceConfig

        config = AgentInstanceConfig(
            agent_id="vuln_v1",
            agent_type="vuln_scanner",
            name="Vulnerability Scanner",
            vertical_id="enclave_guard",
            params={
                "company_name": "Enclave Guard",
            },
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

        return VulnScannerAgent(
            config=config, db=db, embedder=embedder, anthropic_client=llm,
        )

    # ─── Registration & Construction ─────────────────────────────────

    def test_registration(self):
        from core.agents.implementations.vuln_scanner_agent import VulnScannerAgent  # noqa: F401
        from core.agents.registry import AGENT_IMPLEMENTATIONS
        assert "vuln_scanner" in AGENT_IMPLEMENTATIONS

    def test_agent_type(self):
        from core.agents.implementations.vuln_scanner_agent import VulnScannerAgent
        assert VulnScannerAgent.agent_type == "vuln_scanner"

    def test_build_graph(self):
        agent = self._make_agent()
        graph = agent.build_graph()
        assert graph is not None

    def test_get_state_class(self):
        from core.agents.state import VulnScannerAgentState
        agent = self._make_agent()
        assert agent.get_state_class() is VulnScannerAgentState

    def test_get_tools(self):
        agent = self._make_agent()
        assert agent.get_tools() == []

    def test_repr(self):
        agent = self._make_agent()
        r = repr(agent)
        assert "VulnScannerAgent" in r
        assert "vuln_v1" in r

    # ─── Initial State ───────────────────────────────────────────────

    def test_prepare_initial_state(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state(
            {"company_domain": "example.com", "company_name": "Ex"}, "run-1"
        )
        assert state["company_domain"] == "example.com"
        assert state["scan_type"] == "full"
        assert state["ssl_findings"] == []
        assert state["header_findings"] == []
        assert state["network_findings"] == []
        assert state["findings"] == []
        assert state["risk_score"] == 0.0
        assert state["findings_saved"] is False
        assert state["report_summary"] == ""

    def test_prepare_initial_state_custom_scan_type(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state(
            {"company_domain": "test.io", "scan_type": "ssl_only"}, "run-2"
        )
        assert state["scan_type"] == "ssl_only"

    # ─── Constants ──────────────────────────────────────────────────

    def test_scan_types(self):
        from core.agents.implementations import vuln_scanner_agent
        assert "full" in vuln_scanner_agent.SCAN_TYPES
        assert "ssl_only" in vuln_scanner_agent.SCAN_TYPES
        assert "headers_only" in vuln_scanner_agent.SCAN_TYPES
        assert "network" in vuln_scanner_agent.SCAN_TYPES

    def test_severity_levels(self):
        from core.agents.implementations import vuln_scanner_agent
        levels = vuln_scanner_agent.SEVERITY_LEVELS
        assert "critical" in levels
        assert "high" in levels
        assert "medium" in levels
        assert "low" in levels
        assert "info" in levels

    def test_risk_thresholds(self):
        from core.agents.implementations import vuln_scanner_agent
        t = vuln_scanner_agent.RISK_THRESHOLDS
        assert t["critical"] == 9.0
        assert t["high"] == 7.0
        assert t["medium"] == 4.0
        assert t["low"] == 2.0

    def test_required_security_headers(self):
        from core.agents.implementations import vuln_scanner_agent
        headers = vuln_scanner_agent.REQUIRED_SECURITY_HEADERS
        assert "Strict-Transport-Security" in headers
        assert "Content-Security-Policy" in headers
        assert "X-Content-Type-Options" in headers
        assert len(headers) == 7

    def test_system_prompt_template(self):
        from core.agents.implementations import vuln_scanner_agent
        prompt = vuln_scanner_agent.VULN_SCANNER_SYSTEM_PROMPT
        assert "{company_name}" in prompt
        assert "{domain}" in prompt

    # ─── Node 1: Initialize Scan ────────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_initialize_scan_basic(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state(
            {"company_domain": "example.com"}, "run-1"
        )
        result = await agent._node_initialize_scan(state)
        assert result["current_node"] == "initialize_scan"
        assert result["scan_type"] == "full"
        assert len(result["scan_targets"]) >= 1

    @pytest.mark.asyncio
    async def test_node_initialize_scan_no_domain(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state({}, "run-1")
        result = await agent._node_initialize_scan(state)
        assert result["current_node"] == "initialize_scan"
        assert "error" in result

    @pytest.mark.asyncio
    async def test_node_initialize_scan_invalid_type_defaults(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state(
            {"company_domain": "test.io", "scan_type": "invalid_type"}, "run-1"
        )
        result = await agent._node_initialize_scan(state)
        assert result["scan_type"] == "full"

    @pytest.mark.asyncio
    async def test_node_initialize_scan_ssl_only_targets(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state(
            {"company_domain": "test.io", "scan_type": "ssl_only"}, "run-1"
        )
        result = await agent._node_initialize_scan(state)
        assert result["scan_type"] == "ssl_only"
        # ssl_only => only 443 target
        assert len(result["scan_targets"]) == 1

    # ─── Node 2: Scan Targets ──────────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_scan_targets_empty(self):
        agent = self._make_agent()
        mock_result = MagicMock()
        mock_result.data = []
        agent.db.client.table.return_value.select.return_value.eq.return_value.limit.return_value.execute.return_value = mock_result

        state = agent._prepare_initial_state(
            {"company_domain": "example.com"}, "run-1"
        )
        result = await agent._node_scan_targets(state)
        assert result["current_node"] == "scan_targets"
        assert result["ssl_findings"] == []
        # With empty DB data, headers={} so all required headers are "missing"
        # The scan correctly generates findings for each missing header
        assert isinstance(result["header_findings"], list)
        assert result["network_findings"] == []

    @pytest.mark.asyncio
    async def test_node_scan_targets_with_expired_ssl(self):
        agent = self._make_agent()
        mock_result = MagicMock()
        mock_result.data = [
            {
                "ssl_data": {"cert_expired": True},
                "headers": {},
                "shodan_data": {"ports": []},
            }
        ]
        agent.db.client.table.return_value.select.return_value.eq.return_value.limit.return_value.execute.return_value = mock_result

        state = agent._prepare_initial_state(
            {"company_domain": "example.com"}, "run-1"
        )
        result = await agent._node_scan_targets(state)
        assert any(f["severity"] == "critical" for f in result["ssl_findings"])

    @pytest.mark.asyncio
    async def test_node_scan_targets_db_error(self):
        agent = self._make_agent()
        agent.db.client.table.return_value.select.side_effect = Exception("DB fail")

        state = agent._prepare_initial_state(
            {"company_domain": "example.com"}, "run-1"
        )
        result = await agent._node_scan_targets(state)
        assert result["current_node"] == "scan_targets"
        assert result["ssl_findings"] == []

    # ─── Node 3: Analyze Results ──────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_analyze_results_no_findings(self):
        agent = self._make_agent()
        agent.llm.messages.create = MagicMock(return_value=MagicMock(
            content=[MagicMock(text='{"risk_score": 0.0, "executive_summary": "Clean scan"}')]
        ))
        state = agent._prepare_initial_state(
            {"company_domain": "example.com"}, "run-1"
        )
        result = await agent._node_analyze_results(state)
        assert result["current_node"] == "analyze_results"
        assert result["risk_score"] == 0.0
        assert result["critical_count"] == 0

    @pytest.mark.asyncio
    async def test_node_analyze_results_with_findings(self):
        agent = self._make_agent()
        agent.llm.messages.create = MagicMock(side_effect=Exception("LLM fail"))

        state = agent._prepare_initial_state(
            {"company_domain": "example.com"}, "run-1"
        )
        state["ssl_findings"] = [{"severity": "critical", "title": "Expired cert"}]
        state["header_findings"] = [{"severity": "high", "title": "No HSTS"}]
        result = await agent._node_analyze_results(state)

        assert result["critical_count"] == 1
        assert result["high_count"] == 1
        assert result["risk_score"] > 0

    @pytest.mark.asyncio
    async def test_node_analyze_results_llm_fallback(self):
        agent = self._make_agent()
        agent.llm.messages.create = MagicMock(side_effect=Exception("timeout"))

        state = agent._prepare_initial_state(
            {"company_domain": "test.io"}, "run-1"
        )
        state["ssl_findings"] = [{"severity": "medium", "title": "Cert expiring"}]
        result = await agent._node_analyze_results(state)
        # Fallback summary should mention the domain
        assert "test.io" in result["executive_summary"]

    # ─── Node 4: Human Review ────────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_human_review(self):
        agent = self._make_agent()
        state = {"findings": [{"severity": "high"}], "risk_score": 7.0}
        result = await agent._node_human_review(state)
        assert result["requires_human_approval"] is True
        assert result["current_node"] == "human_review"

    @pytest.mark.asyncio
    async def test_node_human_review_empty_findings(self):
        agent = self._make_agent()
        state = {"findings": [], "risk_score": 0.0}
        result = await agent._node_human_review(state)
        assert result["requires_human_approval"] is True

    # ─── Node 5: Save Findings ────────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_save_findings_success(self):
        agent = self._make_agent()
        mock_insert = MagicMock()
        mock_insert.execute.return_value = MagicMock(data=[{"id": "assess_1"}])
        agent.db.client.table.return_value.insert.return_value = mock_insert

        state = agent._prepare_initial_state(
            {"company_domain": "example.com"}, "run-1"
        )
        state["findings"] = [{"severity": "high", "title": "Test finding"}]
        state["risk_score"] = 7.0
        result = await agent._node_save_findings(state)

        assert result["current_node"] == "save_findings"
        assert result["assessment_id"] == "assess_1"
        assert result["findings_saved"] is True

    @pytest.mark.asyncio
    async def test_node_save_findings_db_error(self):
        agent = self._make_agent()
        agent.db.client.table.return_value.insert.return_value.execute.side_effect = Exception("insert fail")

        state = agent._prepare_initial_state(
            {"company_domain": "example.com"}, "run-1"
        )
        state["findings"] = [{"severity": "low"}]
        result = await agent._node_save_findings(state)
        assert result["findings_saved"] is False

    @pytest.mark.asyncio
    async def test_node_save_findings_empty(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state(
            {"company_domain": "example.com"}, "run-1"
        )
        result = await agent._node_save_findings(state)
        assert result["current_node"] == "save_findings"

    # ─── Node 6: Report ──────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_report(self):
        agent = self._make_agent()
        state = {
            "company_domain": "example.com",
            "scan_type": "full",
            "risk_score": 6.5,
            "critical_count": 1,
            "high_count": 2,
            "medium_count": 3,
            "low_count": 4,
            "findings": [{}, {}, {}],
            "executive_summary": "Some risks found.",
            "findings_saved": True,
            "assessment_id": "a_1",
        }
        result = await agent._node_report(state)
        assert result["current_node"] == "report"
        assert "Vulnerability Scan Report" in result["report_summary"]
        assert result["report_generated_at"] != ""

    @pytest.mark.asyncio
    async def test_node_report_no_summary(self):
        agent = self._make_agent()
        state = {
            "company_domain": "test.io",
            "scan_type": "ssl_only",
            "risk_score": 0.0,
            "critical_count": 0,
            "high_count": 0,
            "medium_count": 0,
            "low_count": 0,
            "findings": [],
            "executive_summary": "",
            "findings_saved": False,
        }
        result = await agent._node_report(state)
        assert "Vulnerability Scan Report" in result["report_summary"]

    # ─── Routing ─────────────────────────────────────────────────────

    def test_route_after_review_approved(self):
        from core.agents.implementations.vuln_scanner_agent import VulnScannerAgent
        assert VulnScannerAgent._route_after_review({"human_approval_status": "approved"}) == "approved"

    def test_route_after_review_rejected(self):
        from core.agents.implementations.vuln_scanner_agent import VulnScannerAgent
        assert VulnScannerAgent._route_after_review({"human_approval_status": "rejected"}) == "rejected"

    def test_route_after_review_default(self):
        from core.agents.implementations.vuln_scanner_agent import VulnScannerAgent
        assert VulnScannerAgent._route_after_review({}) == "approved"

    # ─── Graph Nodes ─────────────────────────────────────────────────

    def test_graph_has_correct_nodes(self):
        agent = self._make_agent()
        graph = agent.build_graph()
        expected = {"initialize_scan", "scan_targets", "analyze_results", "human_review", "save_findings", "report"}
        assert expected.issubset(set(graph.nodes.keys()))

    # ─── System Prompt / Knowledge ───────────────────────────────────

    def test_system_prompt(self):
        from core.agents.implementations.vuln_scanner_agent import VULN_SCANNER_SYSTEM_PROMPT
        assert isinstance(VULN_SCANNER_SYSTEM_PROMPT, str)
        assert len(VULN_SCANNER_SYSTEM_PROMPT) > 50

    def test_write_knowledge_exists(self):
        agent = self._make_agent()
        assert hasattr(agent, "write_knowledge")


# ══════════════════════════════════════════════════════════════════════
#  2.  NetworkAnalystAgent
# ══════════════════════════════════════════════════════════════════════


class TestNetworkAnalystState:
    """Tests for NetworkAnalystAgentState TypedDict."""

    def test_import(self):
        from core.agents.state import NetworkAnalystAgentState
        assert NetworkAnalystAgentState is not None

    def test_create_minimal(self):
        from core.agents.state import NetworkAnalystAgentState
        state: NetworkAnalystAgentState = {
            "agent_id": "net_v1",
            "vertical_id": "enclave_guard",
        }
        assert state["agent_id"] == "net_v1"

    def test_create_full(self):
        from core.agents.state import NetworkAnalystAgentState
        state: NetworkAnalystAgentState = {
            "agent_id": "net_v1",
            "vertical_id": "enclave_guard",
            "company_domain": "example.com",
            "company_name": "Example Corp",
            "dns_records": [],
            "subdomains": [],
            "open_ports": [],
            "exposed_services": [],
            "whois_data": {},
            "topology_map": {},
            "attack_surface_score": 0.0,
            "attack_vectors": [],
            "recommendations": [],
            "report_summary": "",
            "report_generated_at": "",
        }
        assert state["attack_surface_score"] == 0.0
        assert state["dns_records"] == []

    def test_network_fields(self):
        from core.agents.state import NetworkAnalystAgentState
        state: NetworkAnalystAgentState = {
            "open_ports": [{"port": 443}],
            "exposed_services": [{"service": "https"}],
            "subdomains": ["api", "www"],
        }
        assert len(state["subdomains"]) == 2


class TestNetworkAnalystAgent:
    """Tests for NetworkAnalystAgent implementation."""

    def _make_agent(self, **kwargs):
        from core.agents.implementations.network_analyst_agent import NetworkAnalystAgent
        from core.config.agent_schema import AgentInstanceConfig

        config = AgentInstanceConfig(
            agent_id="net_v1",
            agent_type="network_analyst",
            name="Network Analyst",
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

        return NetworkAnalystAgent(
            config=config, db=db, embedder=embedder, anthropic_client=llm,
        )

    # ─── Registration & Construction ─────────────────────────────────

    def test_registration(self):
        from core.agents.implementations.network_analyst_agent import NetworkAnalystAgent  # noqa: F401
        from core.agents.registry import AGENT_IMPLEMENTATIONS
        assert "network_analyst" in AGENT_IMPLEMENTATIONS

    def test_agent_type(self):
        from core.agents.implementations.network_analyst_agent import NetworkAnalystAgent
        assert NetworkAnalystAgent.agent_type == "network_analyst"

    def test_build_graph(self):
        agent = self._make_agent()
        graph = agent.build_graph()
        assert graph is not None

    def test_get_state_class(self):
        from core.agents.state import NetworkAnalystAgentState
        agent = self._make_agent()
        assert agent.get_state_class() is NetworkAnalystAgentState

    def test_get_tools(self):
        agent = self._make_agent()
        assert agent.get_tools() == []

    def test_repr(self):
        agent = self._make_agent()
        r = repr(agent)
        assert "NetworkAnalystAgent" in r
        assert "net_v1" in r

    # ─── Initial State ───────────────────────────────────────────────

    def test_prepare_initial_state(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state(
            {"company_domain": "test.io"}, "run-1"
        )
        assert state["company_domain"] == "test.io"
        assert state["dns_records"] == []
        assert state["subdomains"] == []
        assert state["open_ports"] == []
        assert state["exposed_services"] == []
        assert state["whois_data"] == {}
        assert state["attack_surface_score"] == 0.0
        assert state["attack_vectors"] == []
        assert state["recommendations"] == []

    # ─── Constants ──────────────────────────────────────────────────

    def test_high_risk_ports(self):
        from core.agents.implementations import network_analyst_agent
        ports = network_analyst_agent.HIGH_RISK_PORTS
        assert 23 in ports   # telnet
        assert 3389 in ports  # rdp
        assert 6379 in ports  # redis

    def test_service_risk_weights(self):
        from core.agents.implementations import network_analyst_agent
        w = network_analyst_agent.SERVICE_RISK_WEIGHTS
        assert w["telnet"] > w["https"]
        assert w["redis"] > w["ssh"]

    def test_dns_record_types(self):
        from core.agents.implementations import network_analyst_agent
        assert "A" in network_analyst_agent.DNS_RECORD_TYPES
        assert "MX" in network_analyst_agent.DNS_RECORD_TYPES
        assert "TXT" in network_analyst_agent.DNS_RECORD_TYPES

    def test_system_prompt_template(self):
        from core.agents.implementations import network_analyst_agent
        prompt = network_analyst_agent.NETWORK_ANALYST_SYSTEM_PROMPT
        assert "{company_name}" in prompt
        assert "{domain}" in prompt

    # ─── Node 1: Gather Data ──────────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_gather_data_empty(self):
        agent = self._make_agent()
        mock_result = MagicMock()
        mock_result.data = []
        agent.db.client.table.return_value.select.return_value.eq.return_value.execute.return_value = mock_result
        agent.db.client.table.return_value.select.return_value.eq.return_value.limit.return_value.execute.return_value = mock_result

        state = agent._prepare_initial_state(
            {"company_domain": "example.com"}, "run-1"
        )
        result = await agent._node_gather_data(state)
        assert result["current_node"] == "gather_data"
        assert result["dns_records"] == []

    @pytest.mark.asyncio
    async def test_node_gather_data_with_services(self):
        agent = self._make_agent()

        dns_result = MagicMock()
        dns_result.data = [
            {"record_type": "A", "name": "example.com", "value": "1.2.3.4", "ttl": 300},
        ]

        sub_result = MagicMock()
        sub_result.data = [{"subdomain": "api"}, {"subdomain": "www"}]

        shodan_result = MagicMock()
        shodan_result.data = [{"shodan_data": {
            "services": [
                {"port": 443, "service": "https", "version": "1.1"},
                {"port": 22, "service": "ssh", "version": "OpenSSH 8.9"},
            ]
        }}]

        whois_result = MagicMock()
        whois_result.data = [{"registrar": "GoDaddy"}]

        # Chain DB calls (different tables)
        call_count = {"n": 0}
        original_table = agent.db.client.table

        def table_side_effect(name):
            mock_chain = MagicMock()
            if name == "dns_cache":
                mock_chain.select.return_value.eq.return_value.execute.return_value = dns_result
            elif name == "subdomain_cache":
                mock_chain.select.return_value.eq.return_value.execute.return_value = sub_result
            elif name == "tech_stack_cache":
                mock_chain.select.return_value.eq.return_value.limit.return_value.execute.return_value = shodan_result
            elif name == "whois_cache":
                mock_chain.select.return_value.eq.return_value.limit.return_value.execute.return_value = whois_result
            return mock_chain

        agent.db.client.table = MagicMock(side_effect=table_side_effect)

        state = agent._prepare_initial_state(
            {"company_domain": "example.com"}, "run-1"
        )
        result = await agent._node_gather_data(state)
        assert result["current_node"] == "gather_data"
        assert len(result["dns_records"]) == 1
        assert len(result["subdomains"]) == 2
        assert len(result["open_ports"]) >= 2

    @pytest.mark.asyncio
    async def test_node_gather_data_db_error(self):
        agent = self._make_agent()
        agent.db.client.table.side_effect = Exception("DB offline")

        state = agent._prepare_initial_state(
            {"company_domain": "example.com"}, "run-1"
        )
        result = await agent._node_gather_data(state)
        assert result["current_node"] == "gather_data"

    # ─── Node 2: Analyze Topology ──────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_analyze_topology_basic(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state(
            {"company_domain": "example.com"}, "run-1"
        )
        state["dns_records"] = [{"type": "MX", "value": "mail.example.com"}]
        state["subdomains"] = ["api", "www"]
        state["open_ports"] = [{"host": "example.com", "port": 443}]
        state["exposed_services"] = [{"host": "example.com", "service": "https"}]

        result = await agent._node_analyze_topology(state)
        assert result["current_node"] == "analyze_topology"
        topo = result["topology_map"]
        assert topo["domain"] == "example.com"
        assert topo["host_count"] >= 3  # primary + 2 subdomains + MX
        assert topo["has_mail_server"] is True

    @pytest.mark.asyncio
    async def test_node_analyze_topology_empty(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state(
            {"company_domain": "test.io"}, "run-1"
        )
        result = await agent._node_analyze_topology(state)
        topo = result["topology_map"]
        assert topo["host_count"] == 1  # just primary domain

    # ─── Node 3: Assess Attack Surface ───────────────────────────

    @pytest.mark.asyncio
    async def test_node_assess_attack_surface_empty(self):
        agent = self._make_agent()
        agent.llm.messages.create = MagicMock(side_effect=Exception("LLM fail"))

        state = agent._prepare_initial_state(
            {"company_domain": "example.com"}, "run-1"
        )
        result = await agent._node_assess_attack_surface(state)
        assert result["current_node"] == "assess_attack_surface"
        assert result["attack_surface_score"] == 0.0

    @pytest.mark.asyncio
    async def test_node_assess_attack_surface_with_services(self):
        agent = self._make_agent()
        agent.llm.messages.create = MagicMock(side_effect=Exception("LLM fail"))

        state = agent._prepare_initial_state(
            {"company_domain": "example.com"}, "run-1"
        )
        state["exposed_services"] = [
            {"service": "redis", "risk_level": "critical", "host": "example.com", "port": 6379},
            {"service": "ssh", "risk_level": "high", "host": "example.com", "port": 22},
        ]
        state["subdomains"] = ["api", "www", "dev"]
        state["open_ports"] = [
            {"port": 6379, "host": "example.com"},
            {"port": 22, "host": "example.com"},
        ]

        result = await agent._node_assess_attack_surface(state)
        assert result["attack_surface_score"] > 0.0
        assert len(result["attack_vectors"]) >= 1

    # ─── Node 4: Human Review ────────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_human_review(self):
        agent = self._make_agent()
        state = {"attack_surface_score": 5.0, "attack_vectors": []}
        result = await agent._node_human_review(state)
        assert result["requires_human_approval"] is True
        assert result["current_node"] == "human_review"

    # ─── Node 5: Report ──────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_report(self):
        agent = self._make_agent()
        state = {
            "company_domain": "example.com",
            "attack_surface_score": 4.5,
            "topology_map": {
                "host_count": 3,
                "total_open_ports": 5,
                "total_services": 4,
                "subdomain_count": 2,
                "dns_record_count": 10,
            },
            "attack_vectors": [
                {"severity": "high", "vector": "Exposed SSH", "description": "Port 22 open"},
            ],
            "recommendations": [
                {"priority": "high", "action": "Close SSH to public"},
            ],
        }
        result = await agent._node_report(state)
        assert result["current_node"] == "report"
        assert "Network Analysis Report" in result["report_summary"]
        assert result["report_generated_at"] != ""

    @pytest.mark.asyncio
    async def test_node_report_empty(self):
        agent = self._make_agent()
        state = {
            "company_domain": "clean.io",
            "attack_surface_score": 0.0,
            "topology_map": {},
            "attack_vectors": [],
            "recommendations": [],
        }
        result = await agent._node_report(state)
        assert "Network Analysis Report" in result["report_summary"]

    # ─── Routing ─────────────────────────────────────────────────────

    def test_route_after_review_approved(self):
        from core.agents.implementations.network_analyst_agent import NetworkAnalystAgent
        assert NetworkAnalystAgent._route_after_review({"human_approval_status": "approved"}) == "approved"

    def test_route_after_review_rejected(self):
        from core.agents.implementations.network_analyst_agent import NetworkAnalystAgent
        assert NetworkAnalystAgent._route_after_review({"human_approval_status": "rejected"}) == "rejected"

    def test_route_after_review_default(self):
        from core.agents.implementations.network_analyst_agent import NetworkAnalystAgent
        assert NetworkAnalystAgent._route_after_review({}) == "approved"

    # ─── Graph Nodes ─────────────────────────────────────────────────

    def test_graph_has_correct_nodes(self):
        agent = self._make_agent()
        graph = agent.build_graph()
        expected = {"gather_data", "analyze_topology", "assess_attack_surface", "human_review", "report"}
        assert expected.issubset(set(graph.nodes.keys()))

    def test_write_knowledge_exists(self):
        agent = self._make_agent()
        assert hasattr(agent, "write_knowledge")


# ══════════════════════════════════════════════════════════════════════
#  3.  AppSecReviewerAgent
# ══════════════════════════════════════════════════════════════════════


class TestAppSecReviewerState:
    """Tests for AppSecReviewerAgentState TypedDict."""

    def test_import(self):
        from core.agents.state import AppSecReviewerAgentState
        assert AppSecReviewerAgentState is not None

    def test_create_minimal(self):
        from core.agents.state import AppSecReviewerAgentState
        state: AppSecReviewerAgentState = {
            "agent_id": "appsec_v1",
            "vertical_id": "enclave_guard",
        }
        assert state["agent_id"] == "appsec_v1"

    def test_create_full(self):
        from core.agents.state import AppSecReviewerAgentState
        state: AppSecReviewerAgentState = {
            "agent_id": "appsec_v1",
            "vertical_id": "enclave_guard",
            "company_domain": "example.com",
            "company_name": "Example Corp",
            "endpoints": [],
            "technologies_detected": [],
            "security_headers": {},
            "csp_analysis": {},
            "cors_analysis": {},
            "cookie_analysis": [],
            "owasp_findings": [],
            "tls_config": {},
            "findings": [],
            "appsec_score": 0.0,
            "critical_count": 0,
            "high_count": 0,
            "medium_count": 0,
            "low_count": 0,
            "assessment_id": "",
            "findings_saved": False,
            "report_summary": "",
            "report_generated_at": "",
        }
        assert state["appsec_score"] == 0.0
        assert state["cookie_analysis"] == []

    def test_header_fields(self):
        from core.agents.state import AppSecReviewerAgentState
        state: AppSecReviewerAgentState = {
            "security_headers": {"HSTS": {"present": True}},
            "csp_analysis": {"present": True, "grade": "strong"},
            "cors_analysis": {"allows_any_origin": False},
        }
        assert state["csp_analysis"]["grade"] == "strong"


class TestAppSecReviewerAgent:
    """Tests for AppSecReviewerAgent implementation."""

    def _make_agent(self, **kwargs):
        from core.agents.implementations.appsec_reviewer_agent import AppSecReviewerAgent
        from core.config.agent_schema import AgentInstanceConfig

        config = AgentInstanceConfig(
            agent_id="appsec_v1",
            agent_type="appsec_reviewer",
            name="AppSec Reviewer",
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

        return AppSecReviewerAgent(
            config=config, db=db, embedder=embedder, anthropic_client=llm,
        )

    # ─── Registration & Construction ─────────────────────────────────

    def test_registration(self):
        from core.agents.implementations.appsec_reviewer_agent import AppSecReviewerAgent  # noqa: F401
        from core.agents.registry import AGENT_IMPLEMENTATIONS
        assert "appsec_reviewer" in AGENT_IMPLEMENTATIONS

    def test_agent_type(self):
        from core.agents.implementations.appsec_reviewer_agent import AppSecReviewerAgent
        assert AppSecReviewerAgent.agent_type == "appsec_reviewer"

    def test_build_graph(self):
        agent = self._make_agent()
        graph = agent.build_graph()
        assert graph is not None

    def test_get_state_class(self):
        from core.agents.state import AppSecReviewerAgentState
        agent = self._make_agent()
        assert agent.get_state_class() is AppSecReviewerAgentState

    def test_get_tools(self):
        agent = self._make_agent()
        assert agent.get_tools() == []

    def test_repr(self):
        agent = self._make_agent()
        r = repr(agent)
        assert "AppSecReviewerAgent" in r
        assert "appsec_v1" in r

    # ─── Initial State ───────────────────────────────────────────────

    def test_prepare_initial_state(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state(
            {"company_domain": "app.io"}, "run-1"
        )
        assert state["company_domain"] == "app.io"
        assert state["endpoints"] == []
        assert state["technologies_detected"] == []
        assert state["security_headers"] == {}
        assert state["csp_analysis"] == {}
        assert state["cors_analysis"] == {}
        assert state["cookie_analysis"] == []
        assert state["owasp_findings"] == []
        assert state["appsec_score"] == 0.0
        assert state["findings_saved"] is False

    # ─── Constants ──────────────────────────────────────────────────

    def test_security_headers_spec(self):
        from core.agents.implementations import appsec_reviewer_agent
        spec = appsec_reviewer_agent.SECURITY_HEADERS_SPEC
        assert "Strict-Transport-Security" in spec
        assert "Content-Security-Policy" in spec
        assert spec["Strict-Transport-Security"]["required"] is True

    def test_owasp_top_10(self):
        from core.agents.implementations import appsec_reviewer_agent
        owasp = appsec_reviewer_agent.OWASP_TOP_10
        assert len(owasp) == 10
        assert owasp[0]["id"] == "A01"
        assert owasp[9]["id"] == "A10"

    def test_system_prompt_template(self):
        from core.agents.implementations import appsec_reviewer_agent
        prompt = appsec_reviewer_agent.APPSEC_SYSTEM_PROMPT
        assert "{company_name}" in prompt
        assert "{domain}" in prompt

    # ─── Node 1: Discover Endpoints ─────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_discover_endpoints_empty(self):
        agent = self._make_agent()
        mock_result = MagicMock()
        mock_result.data = []
        agent.db.client.table.return_value.select.return_value.eq.return_value.limit.return_value.execute.return_value = mock_result

        state = agent._prepare_initial_state(
            {"company_domain": "example.com"}, "run-1"
        )
        result = await agent._node_discover_endpoints(state)
        assert result["current_node"] == "discover_endpoints"
        # Fallback endpoint always added
        assert len(result["endpoints"]) >= 1

    @pytest.mark.asyncio
    async def test_node_discover_endpoints_with_tech(self):
        agent = self._make_agent()
        mock_result = MagicMock()
        mock_result.data = [{
            "headers": {"Server": "nginx", "X-Powered-By": "Express"},
            "technologies": ["React"],
            "status_code": 200,
        }]
        agent.db.client.table.return_value.select.return_value.eq.return_value.limit.return_value.execute.return_value = mock_result

        state = agent._prepare_initial_state(
            {"company_domain": "example.com"}, "run-1"
        )
        result = await agent._node_discover_endpoints(state)
        assert "nginx" in result["technologies_detected"]
        assert "Express" in result["technologies_detected"]
        assert "React" in result["technologies_detected"]

    @pytest.mark.asyncio
    async def test_node_discover_endpoints_db_error(self):
        agent = self._make_agent()
        agent.db.client.table.return_value.select.side_effect = Exception("fail")

        state = agent._prepare_initial_state(
            {"company_domain": "test.io"}, "run-1"
        )
        result = await agent._node_discover_endpoints(state)
        assert len(result["endpoints"]) >= 1  # fallback

    # ─── Node 2: Scan Headers ──────────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_scan_headers_empty(self):
        agent = self._make_agent()
        mock_result = MagicMock()
        mock_result.data = []
        agent.db.client.table.return_value.select.return_value.eq.return_value.limit.return_value.execute.return_value = mock_result

        state = agent._prepare_initial_state(
            {"company_domain": "example.com"}, "run-1"
        )
        result = await agent._node_scan_headers(state)
        assert result["current_node"] == "scan_headers"
        # All headers should be checked
        assert len(result["security_headers"]) == 7

    @pytest.mark.asyncio
    async def test_node_scan_headers_with_data(self):
        agent = self._make_agent()
        mock_result = MagicMock()
        mock_result.data = [{
            "headers": {
                "Strict-Transport-Security": "max-age=31536000",
                "Content-Security-Policy": "default-src 'self' 'unsafe-inline'",
                "X-Content-Type-Options": "nosniff",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Credentials": "true",
            },
            "ssl_data": {"protocol_version": "TLS 1.2", "supports_tls_1_0": True},
            "cookies": [
                {"name": "session", "secure": False, "httponly": False, "samesite": "None"},
            ],
        }]
        agent.db.client.table.return_value.select.return_value.eq.return_value.limit.return_value.execute.return_value = mock_result

        state = agent._prepare_initial_state(
            {"company_domain": "example.com"}, "run-1"
        )
        result = await agent._node_scan_headers(state)
        assert result["csp_analysis"]["present"] is True
        assert result["csp_analysis"]["allows_unsafe_inline"] is True
        assert result["cors_analysis"]["allows_any_origin"] is True
        assert result["tls_config"]["supports_tls_1_0"] is True
        assert len(result["cookie_analysis"]) == 1

    # ─── Node 3: Check OWASP ──────────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_check_owasp_with_missing_headers(self):
        agent = self._make_agent()
        agent.llm.messages.create = MagicMock(side_effect=Exception("LLM fail"))

        state = agent._prepare_initial_state(
            {"company_domain": "example.com"}, "run-1"
        )
        # Missing required headers
        state["security_headers"] = {
            "Strict-Transport-Security": {"present": False, "required": True, "severity": "high"},
            "Content-Security-Policy": {"present": False, "required": True, "severity": "high"},
        }
        state["csp_analysis"] = {"present": False}
        state["cors_analysis"] = {}
        state["cookie_analysis"] = []
        state["tls_config"] = {}

        result = await agent._node_check_owasp(state)
        assert result["current_node"] == "check_owasp"
        # Should have findings for missing HSTS, CSP
        assert len(result["findings"]) >= 2
        assert result["appsec_score"] > 0.0

    @pytest.mark.asyncio
    async def test_node_check_owasp_cors_critical(self):
        agent = self._make_agent()
        agent.llm.messages.create = MagicMock(side_effect=Exception("LLM fail"))

        state = agent._prepare_initial_state(
            {"company_domain": "example.com"}, "run-1"
        )
        state["security_headers"] = {}
        state["csp_analysis"] = {"present": True}
        state["cors_analysis"] = {"allows_any_origin": True, "allows_credentials": True}
        state["cookie_analysis"] = []
        state["tls_config"] = {}

        result = await agent._node_check_owasp(state)
        assert any(f.get("severity") == "critical" for f in result["findings"])

    @pytest.mark.asyncio
    async def test_node_check_owasp_cookie_findings(self):
        agent = self._make_agent()
        agent.llm.messages.create = MagicMock(side_effect=Exception("LLM fail"))

        state = agent._prepare_initial_state(
            {"company_domain": "example.com"}, "run-1"
        )
        state["security_headers"] = {}
        state["csp_analysis"] = {"present": True}
        state["cors_analysis"] = {}
        state["cookie_analysis"] = [
            {"name": "sess", "secure": False, "httponly": False, "samesite": "None"},
        ]
        state["tls_config"] = {}

        result = await agent._node_check_owasp(state)
        cookie_findings = [f for f in result["findings"] if "Cookie" in f.get("title", "")]
        assert len(cookie_findings) >= 1

    # ─── Node 4: Human Review ────────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_human_review(self):
        agent = self._make_agent()
        state = {"findings": [{"severity": "high"}], "appsec_score": 5.0}
        result = await agent._node_human_review(state)
        assert result["requires_human_approval"] is True
        assert result["current_node"] == "human_review"

    # ─── Node 5: Save Findings ────────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_save_findings_success(self):
        agent = self._make_agent()
        mock_insert = MagicMock()
        mock_insert.execute.return_value = MagicMock(data=[{"id": "appsec_assess_1"}])
        agent.db.client.table.return_value.insert.return_value = mock_insert

        state = agent._prepare_initial_state(
            {"company_domain": "example.com"}, "run-1"
        )
        state["findings"] = [{"severity": "high", "title": "Missing HSTS"}]
        state["appsec_score"] = 5.0
        result = await agent._node_save_findings(state)

        assert result["assessment_id"] == "appsec_assess_1"
        assert result["findings_saved"] is True

    @pytest.mark.asyncio
    async def test_node_save_findings_db_error(self):
        agent = self._make_agent()
        agent.db.client.table.return_value.insert.return_value.execute.side_effect = Exception("fail")

        state = agent._prepare_initial_state(
            {"company_domain": "example.com"}, "run-1"
        )
        state["findings"] = [{"severity": "low"}]
        result = await agent._node_save_findings(state)
        assert result["findings_saved"] is False

    # ─── Node 6: Report ──────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_report(self):
        agent = self._make_agent()
        state = {
            "company_domain": "example.com",
            "appsec_score": 5.0,
            "critical_count": 0,
            "high_count": 2,
            "medium_count": 3,
            "low_count": 1,
            "findings": [{}, {}],
            "csp_analysis": {"present": True, "grade": "weak"},
            "cors_analysis": {"allows_any_origin": False},
            "technologies_detected": ["nginx", "React"],
            "findings_saved": True,
            "assessment_id": "a_1",
        }
        result = await agent._node_report(state)
        assert result["current_node"] == "report"
        assert "Application Security Review Report" in result["report_summary"]
        assert result["report_generated_at"] != ""

    # ─── Routing ─────────────────────────────────────────────────────

    def test_route_after_review_approved(self):
        from core.agents.implementations.appsec_reviewer_agent import AppSecReviewerAgent
        assert AppSecReviewerAgent._route_after_review({"human_approval_status": "approved"}) == "approved"

    def test_route_after_review_rejected(self):
        from core.agents.implementations.appsec_reviewer_agent import AppSecReviewerAgent
        assert AppSecReviewerAgent._route_after_review({"human_approval_status": "rejected"}) == "rejected"

    def test_route_after_review_default(self):
        from core.agents.implementations.appsec_reviewer_agent import AppSecReviewerAgent
        assert AppSecReviewerAgent._route_after_review({}) == "approved"

    # ─── Graph Nodes ─────────────────────────────────────────────────

    def test_graph_has_correct_nodes(self):
        agent = self._make_agent()
        graph = agent.build_graph()
        expected = {"discover_endpoints", "scan_headers", "check_owasp", "human_review", "save_findings", "report"}
        assert expected.issubset(set(graph.nodes.keys()))

    def test_write_knowledge_exists(self):
        agent = self._make_agent()
        assert hasattr(agent, "write_knowledge")


# ══════════════════════════════════════════════════════════════════════
#  4.  ComplianceMapperAgent
# ══════════════════════════════════════════════════════════════════════


class TestComplianceMapperState:
    """Tests for ComplianceMapperAgentState TypedDict."""

    def test_import(self):
        from core.agents.state import ComplianceMapperAgentState
        assert ComplianceMapperAgentState is not None

    def test_create_minimal(self):
        from core.agents.state import ComplianceMapperAgentState
        state: ComplianceMapperAgentState = {
            "agent_id": "comp_v1",
            "vertical_id": "enclave_guard",
        }
        assert state["agent_id"] == "comp_v1"

    def test_create_full(self):
        from core.agents.state import ComplianceMapperAgentState
        state: ComplianceMapperAgentState = {
            "agent_id": "comp_v1",
            "vertical_id": "enclave_guard",
            "company_domain": "example.com",
            "company_name": "Example Corp",
            "target_frameworks": ["soc2", "hipaa"],
            "framework_requirements": {},
            "total_controls": 0,
            "control_mappings": [],
            "controls_met": 0,
            "controls_partial": 0,
            "controls_missing": 0,
            "gaps": [],
            "compliance_scores": {},
            "overall_compliance_score": 0.0,
            "roadmap_items": [],
            "roadmap_summary": "",
            "report_summary": "",
            "report_generated_at": "",
        }
        assert state["overall_compliance_score"] == 0.0
        assert len(state["target_frameworks"]) == 2

    def test_control_count_fields(self):
        from core.agents.state import ComplianceMapperAgentState
        state: ComplianceMapperAgentState = {
            "controls_met": 5,
            "controls_partial": 3,
            "controls_missing": 2,
            "total_controls": 10,
        }
        assert state["controls_met"] + state["controls_partial"] + state["controls_missing"] == 10


class TestComplianceMapperAgent:
    """Tests for ComplianceMapperAgent implementation."""

    def _make_agent(self, **kwargs):
        from core.agents.implementations.compliance_mapper_agent import ComplianceMapperAgent
        from core.config.agent_schema import AgentInstanceConfig

        config = AgentInstanceConfig(
            agent_id="comp_v1",
            agent_type="compliance_mapper",
            name="Compliance Mapper",
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

        return ComplianceMapperAgent(
            config=config, db=db, embedder=embedder, anthropic_client=llm,
        )

    # ─── Registration & Construction ─────────────────────────────────

    def test_registration(self):
        from core.agents.implementations.compliance_mapper_agent import ComplianceMapperAgent  # noqa: F401
        from core.agents.registry import AGENT_IMPLEMENTATIONS
        assert "compliance_mapper" in AGENT_IMPLEMENTATIONS

    def test_agent_type(self):
        from core.agents.implementations.compliance_mapper_agent import ComplianceMapperAgent
        assert ComplianceMapperAgent.agent_type == "compliance_mapper"

    def test_build_graph(self):
        agent = self._make_agent()
        graph = agent.build_graph()
        assert graph is not None

    def test_get_state_class(self):
        from core.agents.state import ComplianceMapperAgentState
        agent = self._make_agent()
        assert agent.get_state_class() is ComplianceMapperAgentState

    def test_get_tools(self):
        agent = self._make_agent()
        assert agent.get_tools() == []

    def test_repr(self):
        agent = self._make_agent()
        r = repr(agent)
        assert "ComplianceMapperAgent" in r
        assert "comp_v1" in r

    # ─── Initial State ───────────────────────────────────────────────

    def test_prepare_initial_state(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state(
            {"company_domain": "example.com", "target_frameworks": ["soc2", "hipaa"]}, "run-1"
        )
        assert state["company_domain"] == "example.com"
        assert "soc2" in state["target_frameworks"]
        assert "hipaa" in state["target_frameworks"]
        assert state["total_controls"] == 0
        assert state["overall_compliance_score"] == 0.0
        assert state["roadmap_items"] == []

    def test_prepare_initial_state_filters_invalid_frameworks(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state(
            {"company_domain": "test.io", "target_frameworks": ["soc2", "bogus_fw"]}, "run-1"
        )
        assert "soc2" in state["target_frameworks"]
        assert "bogus_fw" not in state["target_frameworks"]

    def test_prepare_initial_state_default_framework(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state(
            {"company_domain": "test.io"}, "run-1"
        )
        assert "soc2" in state["target_frameworks"]

    # ─── Constants ──────────────────────────────────────────────────

    def test_supported_frameworks(self):
        from core.agents.implementations import compliance_mapper_agent
        fw = compliance_mapper_agent.SUPPORTED_FRAMEWORKS
        assert "soc2" in fw
        assert "hipaa" in fw
        assert "pci_dss" in fw
        assert "iso27001" in fw

    def test_control_statuses(self):
        from core.agents.implementations import compliance_mapper_agent
        s = compliance_mapper_agent.CONTROL_STATUSES
        assert "met" in s
        assert "partial" in s
        assert "missing" in s
        assert "not_applicable" in s

    def test_framework_controls_soc2(self):
        from core.agents.implementations import compliance_mapper_agent
        controls = compliance_mapper_agent.FRAMEWORK_CONTROLS["soc2"]
        assert len(controls) == 10
        assert controls[0]["id"] == "CC1.1"

    def test_framework_controls_hipaa(self):
        from core.agents.implementations import compliance_mapper_agent
        controls = compliance_mapper_agent.FRAMEWORK_CONTROLS["hipaa"]
        assert len(controls) == 10

    def test_system_prompt_template(self):
        from core.agents.implementations import compliance_mapper_agent
        prompt = compliance_mapper_agent.COMPLIANCE_SYSTEM_PROMPT
        assert "{company_name}" in prompt
        assert "{domain}" in prompt

    # ─── Node 1: Load Requirements ──────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_load_requirements_soc2(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state(
            {"company_domain": "test.io", "target_frameworks": ["soc2"]}, "run-1"
        )
        result = await agent._node_load_requirements(state)
        assert result["current_node"] == "load_requirements"
        assert "soc2" in result["framework_requirements"]
        assert result["total_controls"] == 10

    @pytest.mark.asyncio
    async def test_node_load_requirements_multiple(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state(
            {"company_domain": "test.io", "target_frameworks": ["soc2", "hipaa"]}, "run-1"
        )
        result = await agent._node_load_requirements(state)
        assert result["total_controls"] == 20

    @pytest.mark.asyncio
    async def test_node_load_requirements_empty_framework(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state(
            {"company_domain": "test.io"}, "run-1"
        )
        # Default is ["soc2"]
        result = await agent._node_load_requirements(state)
        assert result["total_controls"] >= 10

    # ─── Node 2: Map Controls ──────────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_map_controls_no_findings(self):
        agent = self._make_agent()
        agent.llm.messages.create = MagicMock(side_effect=Exception("LLM fail"))

        mock_result = MagicMock()
        mock_result.data = []
        agent.db.client.table.return_value.select.return_value.eq.return_value.eq.return_value.execute.return_value = mock_result

        state = agent._prepare_initial_state(
            {"company_domain": "test.io", "target_frameworks": ["soc2"]}, "run-1"
        )
        state["framework_requirements"] = {"soc2": [{"id": "CC1.1", "control": "test", "description": "test", "category": "Control Environment"}]}

        result = await agent._node_map_controls(state)
        assert result["current_node"] == "map_controls"
        assert len(result["control_mappings"]) >= 1

    @pytest.mark.asyncio
    async def test_node_map_controls_db_error(self):
        agent = self._make_agent()
        agent.llm.messages.create = MagicMock(side_effect=Exception("LLM fail"))
        agent.db.client.table.return_value.select.return_value.eq.return_value.eq.return_value.execute.side_effect = Exception("DB fail")

        state = agent._prepare_initial_state(
            {"company_domain": "test.io", "target_frameworks": ["soc2"]}, "run-1"
        )
        state["framework_requirements"] = {"soc2": [{"id": "CC1.1", "control": "test", "description": "test", "category": "Other"}]}

        result = await agent._node_map_controls(state)
        assert result["current_node"] == "map_controls"

    # ─── Node 3: Identify Gaps ────────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_identify_gaps_all_missing(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state(
            {"company_domain": "test.io", "target_frameworks": ["soc2"]}, "run-1"
        )
        state["control_mappings"] = [
            {"framework": "soc2", "control_id": "CC1.1", "status": "missing", "gap": "No integrity policy"},
            {"framework": "soc2", "control_id": "CC2.1", "status": "missing", "gap": "No communication policy"},
        ]

        result = await agent._node_identify_gaps(state)
        assert result["controls_missing"] == 2
        assert result["controls_met"] == 0
        assert result["overall_compliance_score"] == 0.0
        assert len(result["gaps"]) == 2

    @pytest.mark.asyncio
    async def test_node_identify_gaps_mixed(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state(
            {"company_domain": "test.io", "target_frameworks": ["soc2"]}, "run-1"
        )
        state["control_mappings"] = [
            {"framework": "soc2", "control_id": "CC1.1", "status": "met"},
            {"framework": "soc2", "control_id": "CC2.1", "status": "partial", "gap": "Incomplete"},
            {"framework": "soc2", "control_id": "CC3.1", "status": "missing", "gap": "No risk assessment"},
        ]

        result = await agent._node_identify_gaps(state)
        assert result["controls_met"] == 1
        assert result["controls_partial"] == 1
        assert result["controls_missing"] == 1
        assert result["overall_compliance_score"] == 50.0  # (1*1.0 + 1*0.5) / 3 * 100

    # ─── Node 4: Human Review ────────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_human_review(self):
        agent = self._make_agent()
        state = {"gaps": [{"severity": "high"}], "overall_compliance_score": 45.0}
        result = await agent._node_human_review(state)
        assert result["requires_human_approval"] is True
        assert result["current_node"] == "human_review"

    # ─── Node 5: Generate Roadmap ────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_generate_roadmap_with_gaps(self):
        agent = self._make_agent()
        agent.llm.messages.create = MagicMock(side_effect=Exception("LLM fail"))

        state = agent._prepare_initial_state(
            {"company_domain": "test.io"}, "run-1"
        )
        state["gaps"] = [
            {"severity": "high", "control_id": "CC1.1", "framework": "soc2", "description": "Missing policy"},
            {"severity": "medium", "control_id": "CC2.1", "framework": "soc2", "description": "Incomplete comms"},
        ]
        state["compliance_scores"] = {"soc2": 40.0}

        result = await agent._node_generate_roadmap(state)
        assert result["current_node"] == "generate_roadmap"
        assert len(result["roadmap_items"]) == 2
        assert result["roadmap_items"][0]["priority"] == 1
        assert "test.io" in result["roadmap_summary"]

    @pytest.mark.asyncio
    async def test_node_generate_roadmap_empty_gaps(self):
        agent = self._make_agent()
        agent.llm.messages.create = MagicMock(side_effect=Exception("LLM fail"))

        state = agent._prepare_initial_state(
            {"company_domain": "clean.io"}, "run-1"
        )
        state["gaps"] = []
        state["compliance_scores"] = {"soc2": 100.0}

        result = await agent._node_generate_roadmap(state)
        assert result["roadmap_items"] == []

    # ─── Node 6: Report ──────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_report(self):
        agent = self._make_agent()
        state = {
            "company_domain": "example.com",
            "overall_compliance_score": 55.0,
            "compliance_scores": {"soc2": 60.0, "hipaa": 50.0},
            "controls_met": 6,
            "controls_partial": 4,
            "controls_missing": 10,
            "gaps": [{}] * 14,
            "roadmap_summary": "Fix critical gaps first.",
            "roadmap_items": [
                {"priority": 1, "action": "Fix CC1.1", "framework": "soc2", "effort_weeks": 2},
            ],
        }
        result = await agent._node_report(state)
        assert result["current_node"] == "report"
        assert "Compliance Gap Analysis Report" in result["report_summary"]
        assert result["report_generated_at"] != ""

    # ─── Routing ─────────────────────────────────────────────────────

    def test_route_after_review_approved(self):
        from core.agents.implementations.compliance_mapper_agent import ComplianceMapperAgent
        assert ComplianceMapperAgent._route_after_review({"human_approval_status": "approved"}) == "approved"

    def test_route_after_review_rejected(self):
        from core.agents.implementations.compliance_mapper_agent import ComplianceMapperAgent
        assert ComplianceMapperAgent._route_after_review({"human_approval_status": "rejected"}) == "rejected"

    def test_route_after_review_default(self):
        from core.agents.implementations.compliance_mapper_agent import ComplianceMapperAgent
        assert ComplianceMapperAgent._route_after_review({}) == "approved"

    # ─── Graph Nodes ─────────────────────────────────────────────────

    def test_graph_has_correct_nodes(self):
        agent = self._make_agent()
        graph = agent.build_graph()
        expected = {"load_requirements", "map_controls", "identify_gaps", "human_review", "generate_roadmap", "report"}
        assert expected.issubset(set(graph.nodes.keys()))

    def test_write_knowledge_exists(self):
        agent = self._make_agent()
        assert hasattr(agent, "write_knowledge")


# ══════════════════════════════════════════════════════════════════════
#  5.  RiskReporterAgent
# ══════════════════════════════════════════════════════════════════════


class TestRiskReporterState:
    """Tests for RiskReporterAgentState TypedDict."""

    def test_import(self):
        from core.agents.state import RiskReporterAgentState
        assert RiskReporterAgentState is not None

    def test_create_minimal(self):
        from core.agents.state import RiskReporterAgentState
        state: RiskReporterAgentState = {
            "agent_id": "risk_v1",
            "vertical_id": "enclave_guard",
        }
        assert state["agent_id"] == "risk_v1"

    def test_create_full(self):
        from core.agents.state import RiskReporterAgentState
        state: RiskReporterAgentState = {
            "agent_id": "risk_v1",
            "vertical_id": "enclave_guard",
            "company_domain": "example.com",
            "company_name": "Example Corp",
            "all_findings": [],
            "compliance_data": {},
            "network_data": {},
            "overall_risk_score": 0.0,
            "risk_by_category": {},
            "estimated_breach_cost": 0.0,
            "annualized_loss_expectancy": 0.0,
            "executive_summary": "",
            "detailed_sections": [],
            "risk_matrix": {},
            "priority_actions": [],
            "report_format": "markdown",
            "report_approved": False,
            "report_delivered": False,
            "report_summary": "",
            "report_generated_at": "",
        }
        assert state["overall_risk_score"] == 0.0
        assert state["report_approved"] is False
        assert state["report_delivered"] is False

    def test_financial_fields(self):
        from core.agents.state import RiskReporterAgentState
        state: RiskReporterAgentState = {
            "estimated_breach_cost": 4_450_000.0,
            "annualized_loss_expectancy": 356_000.0,
        }
        assert state["estimated_breach_cost"] == 4_450_000.0

    def test_report_format_field(self):
        from core.agents.state import RiskReporterAgentState
        state: RiskReporterAgentState = {
            "report_format": "pdf",
        }
        assert state["report_format"] == "pdf"


class TestRiskReporterAgent:
    """Tests for RiskReporterAgent implementation."""

    def _make_agent(self, **kwargs):
        from core.agents.implementations.risk_reporter_agent import RiskReporterAgent
        from core.config.agent_schema import AgentInstanceConfig

        config = AgentInstanceConfig(
            agent_id="risk_v1",
            agent_type="risk_reporter",
            name="Risk Reporter",
            vertical_id="enclave_guard",
            params={
                "company_name": "Enclave Guard",
                "industry": "technology",
            },
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

        return RiskReporterAgent(
            config=config, db=db, embedder=embedder, anthropic_client=llm,
        )

    # ─── Registration & Construction ─────────────────────────────────

    def test_registration(self):
        from core.agents.implementations.risk_reporter_agent import RiskReporterAgent  # noqa: F401
        from core.agents.registry import AGENT_IMPLEMENTATIONS
        assert "risk_reporter" in AGENT_IMPLEMENTATIONS

    def test_agent_type(self):
        from core.agents.implementations.risk_reporter_agent import RiskReporterAgent
        assert RiskReporterAgent.agent_type == "risk_reporter"

    def test_build_graph(self):
        agent = self._make_agent()
        graph = agent.build_graph()
        assert graph is not None

    def test_get_state_class(self):
        from core.agents.state import RiskReporterAgentState
        agent = self._make_agent()
        assert agent.get_state_class() is RiskReporterAgentState

    def test_get_tools(self):
        agent = self._make_agent()
        assert agent.get_tools() == []

    def test_repr(self):
        agent = self._make_agent()
        r = repr(agent)
        assert "RiskReporterAgent" in r
        assert "risk_v1" in r

    # ─── Initial State ───────────────────────────────────────────────

    def test_prepare_initial_state(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state(
            {"company_domain": "example.com", "company_name": "Ex"}, "run-1"
        )
        assert state["company_domain"] == "example.com"
        assert state["all_findings"] == []
        assert state["compliance_data"] == {}
        assert state["network_data"] == {}
        assert state["overall_risk_score"] == 0.0
        assert state["risk_by_category"] == {}
        assert state["estimated_breach_cost"] == 0.0
        assert state["annualized_loss_expectancy"] == 0.0
        assert state["report_approved"] is False
        assert state["report_delivered"] is False
        assert state["report_format"] == "markdown"

    def test_prepare_initial_state_custom_format(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state(
            {"company_domain": "test.io", "report_format": "pdf"}, "run-1"
        )
        assert state["report_format"] == "pdf"

    # ─── Constants ──────────────────────────────────────────────────

    def test_risk_categories(self):
        from core.agents.implementations import risk_reporter_agent
        cats = risk_reporter_agent.RISK_CATEGORIES
        assert "vulnerability_exploitation" in cats
        assert "data_breach" in cats
        assert "ransomware" in cats
        assert "compliance_penalty" in cats
        assert "business_disruption" in cats
        assert "reputation_damage" in cats
        assert len(cats) == 6

    def test_breach_cost_multipliers(self):
        from core.agents.implementations import risk_reporter_agent
        m = risk_reporter_agent.BREACH_COST_MULTIPLIERS
        assert m["healthcare"] > m["default"]
        assert "default" in m

    def test_likelihood_labels(self):
        from core.agents.implementations import risk_reporter_agent
        labels = risk_reporter_agent.LIKELIHOOD_LABELS
        assert len(labels) == 5

    def test_impact_labels(self):
        from core.agents.implementations import risk_reporter_agent
        labels = risk_reporter_agent.IMPACT_LABELS
        assert len(labels) == 5

    def test_system_prompt_template(self):
        from core.agents.implementations import risk_reporter_agent
        prompt = risk_reporter_agent.RISK_REPORT_SYSTEM_PROMPT
        assert "{company_name}" in prompt
        assert "{domain}" in prompt
        assert "{industry}" in prompt

    # ─── Node 1: Aggregate Data ──────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_aggregate_data_empty(self):
        agent = self._make_agent()
        mock_result = MagicMock()
        mock_result.data = []
        agent.db.client.table.return_value.select.return_value.eq.return_value.eq.return_value.order.return_value.limit.return_value.execute.return_value = mock_result
        agent.db.client.table.return_value.select.return_value.eq.return_value.eq.return_value.order.return_value.execute.return_value = mock_result

        state = agent._prepare_initial_state(
            {"company_domain": "example.com"}, "run-1"
        )
        result = await agent._node_aggregate_data(state)
        assert result["current_node"] == "aggregate_data"
        assert result["all_findings"] == []
        assert result["compliance_data"] == {}
        assert result["network_data"] == {}

    @pytest.mark.asyncio
    async def test_node_aggregate_data_with_findings(self):
        agent = self._make_agent()
        findings_result = MagicMock()
        findings_result.data = [
            {"severity": "high", "title": "Expired cert"},
            {"severity": "medium", "title": "Missing HSTS"},
        ]

        assess_result = MagicMock()
        assess_result.data = [
            {"id": "a_1", "assessment_type": "vulnerability_scan", "risk_score": 7.0, "finding_count": 5},
        ]

        call_count = {"n": 0}

        def table_side_effect(name):
            mock_chain = MagicMock()
            if name == "security_findings":
                mock_chain.select.return_value.eq.return_value.eq.return_value.order.return_value.limit.return_value.execute.return_value = findings_result
            elif name == "security_assessments":
                mock_chain.select.return_value.eq.return_value.eq.return_value.order.return_value.execute.return_value = assess_result
            return mock_chain

        agent.db.client.table = MagicMock(side_effect=table_side_effect)

        state = agent._prepare_initial_state(
            {"company_domain": "example.com"}, "run-1"
        )
        result = await agent._node_aggregate_data(state)
        assert len(result["all_findings"]) >= 2
        assert result["network_data"]["risk_score"] == 7.0

    @pytest.mark.asyncio
    async def test_node_aggregate_data_db_error(self):
        agent = self._make_agent()
        agent.db.client.table.return_value.select.return_value.eq.return_value.eq.return_value.order.side_effect = Exception("DB fail")

        state = agent._prepare_initial_state(
            {"company_domain": "test.io"}, "run-1"
        )
        result = await agent._node_aggregate_data(state)
        assert result["current_node"] == "aggregate_data"

    # ─── Node 2: Quantify Risk ──────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_quantify_risk_no_findings(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state(
            {"company_domain": "example.com"}, "run-1"
        )
        result = await agent._node_quantify_risk(state)
        assert result["current_node"] == "quantify_risk"
        assert result["overall_risk_score"] == 0.0
        assert result["annualized_loss_expectancy"] > 0  # Even zero-risk has minimum ARO

    @pytest.mark.asyncio
    async def test_node_quantify_risk_with_critical(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state(
            {"company_domain": "example.com"}, "run-1"
        )
        state["all_findings"] = [
            {"severity": "critical"},
            {"severity": "critical"},
            {"severity": "high"},
            {"severity": "medium"},
        ]
        state["compliance_data"] = {"overall_score": 30}
        result = await agent._node_quantify_risk(state)
        assert result["overall_risk_score"] > 5.0
        assert "vulnerability_exploitation" in result["risk_by_category"]
        assert "data_breach" in result["risk_by_category"]
        assert result["estimated_breach_cost"] > 0
        assert result["annualized_loss_expectancy"] > 0

    @pytest.mark.asyncio
    async def test_node_quantify_risk_caps_at_ten(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state(
            {"company_domain": "example.com"}, "run-1"
        )
        # Many critical findings to exceed 10
        state["all_findings"] = [{"severity": "critical"}] * 20
        result = await agent._node_quantify_risk(state)
        assert result["overall_risk_score"] == 10.0

    # ─── Node 3: Generate Report ──────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_generate_report_llm_fallback(self):
        agent = self._make_agent()
        agent.llm.messages.create = MagicMock(side_effect=Exception("LLM fail"))

        state = agent._prepare_initial_state(
            {"company_domain": "example.com"}, "run-1"
        )
        state["all_findings"] = [{"severity": "high", "title": "Test"}]
        state["overall_risk_score"] = 6.0
        state["risk_by_category"] = {"data_breach": 6.0, "ransomware": 4.8}
        state["annualized_loss_expectancy"] = 200000.0

        result = await agent._node_generate_report(state)
        assert result["current_node"] == "generate_report"
        assert "example.com" in result["executive_summary"]
        assert len(result["detailed_sections"]) == 4
        assert len(result["risk_matrix"]["entries"]) >= 1

    @pytest.mark.asyncio
    async def test_node_generate_report_with_llm(self):
        agent = self._make_agent()
        llm_response = json.dumps({
            "executive_summary": "Critical risk posture requires immediate action.",
            "risk_matrix": [
                {"category": "Data Breach", "likelihood": 0.7, "impact": 0.9, "risk_level": "critical", "dollar_exposure": 500000}
            ],
            "recommendations": [
                {"priority": 1, "action": "Patch critical vulns", "cost_estimate": "$5000", "risk_reduction": "40%"}
            ],
        })
        agent.llm.messages.create = MagicMock(return_value=MagicMock(
            content=[MagicMock(text=llm_response)]
        ))

        state = agent._prepare_initial_state(
            {"company_domain": "example.com"}, "run-1"
        )
        state["all_findings"] = [{"severity": "critical"}]
        state["overall_risk_score"] = 8.0
        state["risk_by_category"] = {"data_breach": 8.0}
        state["annualized_loss_expectancy"] = 500000.0

        result = await agent._node_generate_report(state)
        assert "immediate action" in result["executive_summary"]
        assert len(result["risk_matrix"]["entries"]) == 1
        assert len(result["priority_actions"]) == 1

    # ─── Node 4: Human Review ────────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_human_review(self):
        agent = self._make_agent()
        state = {"overall_risk_score": 8.0, "annualized_loss_expectancy": 500000.0}
        result = await agent._node_human_review(state)
        assert result["requires_human_approval"] is True
        assert result["current_node"] == "human_review"

    # ─── Node 5: Deliver ──────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_deliver_success(self):
        agent = self._make_agent()
        mock_insert = MagicMock()
        mock_insert.execute.return_value = MagicMock(data=[{"id": "report_1"}])
        agent.db.client.table.return_value.insert.return_value = mock_insert

        state = agent._prepare_initial_state(
            {"company_domain": "example.com"}, "run-1"
        )
        state["overall_risk_score"] = 7.0
        state["executive_summary"] = "High risk identified."
        state["all_findings"] = [{"severity": "high"}]
        state["annualized_loss_expectancy"] = 300000.0
        state["estimated_breach_cost"] = 4000000.0
        state["risk_by_category"] = {"data_breach": 7.0}

        result = await agent._node_deliver(state)
        assert result["current_node"] == "deliver"
        assert result["report_delivered"] is True
        assert result["report_approved"] is True

    @pytest.mark.asyncio
    async def test_node_deliver_db_error(self):
        agent = self._make_agent()
        agent.db.client.table.return_value.insert.return_value.execute.side_effect = Exception("DB fail")

        state = agent._prepare_initial_state(
            {"company_domain": "example.com"}, "run-1"
        )
        state["all_findings"] = []
        state["overall_risk_score"] = 0.0
        state["annualized_loss_expectancy"] = 0.0
        state["estimated_breach_cost"] = 0.0
        state["risk_by_category"] = {}

        result = await agent._node_deliver(state)
        assert result["report_delivered"] is False

    # ─── Node 6: Report ──────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_node_report(self):
        agent = self._make_agent()
        state = {
            "company_domain": "example.com",
            "overall_risk_score": 7.5,
            "annualized_loss_expectancy": 350000.0,
            "estimated_breach_cost": 4970000.0,
            "all_findings": [{}] * 15,
            "risk_by_category": {
                "vulnerability_exploitation": 9.0,
                "data_breach": 7.5,
                "ransomware": 6.0,
            },
            "executive_summary": "High risk posture.",
            "priority_actions": [
                {"priority": 1, "action": "Patch critical vulns"},
            ],
            "report_delivered": True,
        }
        result = await agent._node_report(state)
        assert result["current_node"] == "report"
        assert "Executive Risk Report" in result["report_summary"]
        assert "DELIVERED" in result["report_summary"]
        assert result["report_generated_at"] != ""

    @pytest.mark.asyncio
    async def test_node_report_not_delivered(self):
        agent = self._make_agent()
        state = {
            "company_domain": "test.io",
            "overall_risk_score": 3.0,
            "annualized_loss_expectancy": 50000.0,
            "estimated_breach_cost": 4450000.0,
            "all_findings": [],
            "risk_by_category": {},
            "executive_summary": "",
            "priority_actions": [],
            "report_delivered": False,
        }
        result = await agent._node_report(state)
        assert "NOT DELIVERED" in result["report_summary"]

    # ─── Routing ─────────────────────────────────────────────────────

    def test_route_after_review_approved(self):
        from core.agents.implementations.risk_reporter_agent import RiskReporterAgent
        assert RiskReporterAgent._route_after_review({"human_approval_status": "approved"}) == "approved"

    def test_route_after_review_rejected(self):
        from core.agents.implementations.risk_reporter_agent import RiskReporterAgent
        assert RiskReporterAgent._route_after_review({"human_approval_status": "rejected"}) == "rejected"

    def test_route_after_review_default(self):
        from core.agents.implementations.risk_reporter_agent import RiskReporterAgent
        assert RiskReporterAgent._route_after_review({}) == "approved"

    # ─── Graph Nodes ─────────────────────────────────────────────────

    def test_graph_has_correct_nodes(self):
        agent = self._make_agent()
        graph = agent.build_graph()
        expected = {"aggregate_data", "quantify_risk", "generate_report", "human_review", "deliver", "report"}
        assert expected.issubset(set(graph.nodes.keys()))

    def test_write_knowledge_exists(self):
        agent = self._make_agent()
        assert hasattr(agent, "write_knowledge")
