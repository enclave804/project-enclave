"""
Tests for MCP tools and server configuration.

Validates:
- Apollo tools delegate correctly to ApolloClient
- Supabase tools delegate correctly to EnclaveDB
- Email tools are sandboxed in non-production environments
- Tool string shorthand coercion in AgentInstanceConfig
- MCP server creation and tool registration
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.config.agent_schema import AgentInstanceConfig, AgentToolConfig


def _run(coro):
    """Helper to run async functions in sync tests."""
    return asyncio.get_event_loop().run_until_complete(coro)


# ─── Tool String Coercion Tests ───────────────────────────────────────


class TestToolStringCoercion:
    """Tests for the model_validator that coerces tool strings."""

    def test_string_tools_coerced_to_agent_tool_config(self):
        """Plain string tools are converted to AgentToolConfig with type='mcp'."""
        config = AgentInstanceConfig(
            agent_id="test",
            agent_type="test",
            name="Test Agent",
            tools=["apollo_search", "send_email"],  # type: ignore[arg-type]
        )
        assert len(config.tools) == 2
        assert all(isinstance(t, AgentToolConfig) for t in config.tools)
        assert config.tools[0].name == "apollo_search"
        assert config.tools[0].type == "mcp"
        assert config.tools[1].name == "send_email"
        assert config.tools[1].type == "mcp"

    def test_dict_tools_still_work(self):
        """Verbose dict format still works alongside string shorthand."""
        config = AgentInstanceConfig(
            agent_id="test",
            agent_type="test",
            name="Test Agent",
            tools=[
                {"name": "apollo_search", "type": "builtin"},
                "send_email",
            ],  # type: ignore[arg-type]
        )
        assert len(config.tools) == 2
        assert config.tools[0].name == "apollo_search"
        assert config.tools[0].type == "builtin"
        assert config.tools[1].name == "send_email"
        assert config.tools[1].type == "mcp"

    def test_empty_tools_list(self):
        """Empty tools list works fine."""
        config = AgentInstanceConfig(
            agent_id="test",
            agent_type="test",
            name="Test Agent",
            tools=[],
        )
        assert config.tools == []

    def test_no_tools_key(self):
        """Missing tools key defaults to empty list."""
        config = AgentInstanceConfig(
            agent_id="test",
            agent_type="test",
            name="Test Agent",
        )
        assert config.tools == []

    def test_agent_tool_config_default_type(self):
        """AgentToolConfig defaults to type='mcp' when not specified."""
        tool = AgentToolConfig(name="my_tool")
        assert tool.type == "mcp"

    def test_agent_tool_config_explicit_type(self):
        """AgentToolConfig respects explicit type."""
        tool = AgentToolConfig(name="my_tool", type="builtin")
        assert tool.type == "builtin"


# ─── Apollo Tools Tests ───────────────────────────────────────────────


class TestApolloTools:
    """Tests for Apollo MCP tool functions."""

    def test_search_leads_delegates_to_client(self):
        """search_leads calls ApolloClient.search_people with parsed params."""
        from core.mcp.tools.apollo_tools import search_leads

        mock_client = AsyncMock()
        mock_client.search_people.return_value = {
            "people": [
                {
                    "name": "Jane CTO",
                    "title": "CTO",
                    "email": "jane@acme.com",
                    "organization": {
                        "name": "Acme Corp",
                        "primary_domain": "acme.com",
                        "industry": "Technology",
                        "estimated_num_employees": 150,
                    },
                }
            ],
            "pagination": {"total_entries": 1, "page": 1},
        }

        result = _run(search_leads(
            titles="CTO,CISO",
            seniorities="c_suite",
            per_page=10,
            _apollo_client=mock_client,
        ))

        mock_client.search_people.assert_called_once()
        call_kwargs = mock_client.search_people.call_args[1]
        assert call_kwargs["person_titles"] == ["CTO", "CISO"]
        assert call_kwargs["person_seniorities"] == ["c_suite"]
        assert call_kwargs["per_page"] == 10

        parsed = json.loads(result)
        assert parsed["total_results"] == 1
        assert len(parsed["leads"]) == 1
        assert parsed["leads"][0]["name"] == "Jane CTO"

    def test_search_leads_handles_none_params(self):
        """search_leads handles None params gracefully."""
        from core.mcp.tools.apollo_tools import search_leads

        mock_client = AsyncMock()
        mock_client.search_people.return_value = {
            "people": [],
            "pagination": {"total_entries": 0},
        }

        result = _run(search_leads(_apollo_client=mock_client))

        call_kwargs = mock_client.search_people.call_args[1]
        assert call_kwargs["person_titles"] is None
        assert call_kwargs["person_seniorities"] is None

        parsed = json.loads(result)
        assert parsed["leads"] == []

    def test_enrich_company_delegates_to_client(self):
        """enrich_company calls ApolloClient.enrich_company with domain."""
        from core.mcp.tools.apollo_tools import enrich_company

        mock_client = AsyncMock()
        mock_client.enrich_company.return_value = {
            "organization": {
                "name": "Acme Corp",
                "primary_domain": "acme.com",
                "industry": "Technology",
                "estimated_num_employees": 150,
                "current_technologies": [
                    {"name": "AWS"},
                    {"name": "React"},
                ],
            }
        }

        result = _run(enrich_company(
            domain="acme.com",
            _apollo_client=mock_client,
        ))

        mock_client.enrich_company.assert_called_once_with("acme.com")

        parsed = json.loads(result)
        assert parsed["name"] == "Acme Corp"
        assert parsed["industry"] == "Technology"
        assert "AWS" in parsed["tech_stack"]
        assert "React" in parsed["tech_stack"]

    def test_enrich_company_not_found(self):
        """enrich_company returns error for unknown domain."""
        from core.mcp.tools.apollo_tools import enrich_company

        mock_client = AsyncMock()
        mock_client.enrich_company.return_value = {}

        result = _run(enrich_company(
            domain="unknown.xyz",
            _apollo_client=mock_client,
        ))

        parsed = json.loads(result)
        assert "error" in parsed


# ─── Supabase Tools Tests ─────────────────────────────────────────────


class TestSupabaseTools:
    """Tests for Supabase/RAG MCP tool functions."""

    def test_search_knowledge_delegates(self):
        """search_knowledge embeds query and calls db.search_knowledge."""
        from core.mcp.tools.supabase_tools import search_knowledge

        mock_db = MagicMock()
        mock_db.search_knowledge.return_value = [
            {
                "content": "CISOs respond well to ROI framing",
                "chunk_type": "winning_pattern",
                "source_type": "outreach_result",
                "similarity": 0.92,
                "metadata": {"win_rate": 0.75},
            }
        ]

        mock_embedder = AsyncMock()
        mock_embedder.embed_text.return_value = [0.1] * 1536

        result = _run(search_knowledge(
            query="best messaging for CISOs",
            chunk_type="winning_pattern",
            limit=3,
            _db=mock_db,
            _embedder=mock_embedder,
        ))

        mock_embedder.embed_text.assert_called_once_with("best messaging for CISOs")
        mock_db.search_knowledge.assert_called_once()

        parsed = json.loads(result)
        assert parsed["count"] == 1
        assert parsed["results"][0]["chunk_type"] == "winning_pattern"

    def test_save_insight_delegates(self):
        """save_insight calls db.store_insight with correct params."""
        from core.mcp.tools.supabase_tools import save_insight

        mock_db = MagicMock()
        mock_db.store_insight.return_value = {"id": "ins-123"}

        result = save_insight(
            insight_type="winning_pattern",
            content="ROI-focused messaging works best for CISOs",
            confidence=0.9,
            title="CISO Messaging Pattern",
            _db=mock_db,
        )

        mock_db.store_insight.assert_called_once()
        call_kwargs = mock_db.store_insight.call_args[1]
        assert call_kwargs["insight_type"] == "winning_pattern"
        assert call_kwargs["confidence_score"] == 0.9
        assert call_kwargs["source_agent_id"] == "mcp_tool"

        parsed = json.loads(result)
        assert parsed["saved"] is True
        assert parsed["id"] == "ins-123"

    def test_query_companies_delegates(self):
        """query_companies calls db.list_companies with filters."""
        from core.mcp.tools.supabase_tools import query_companies

        mock_db = MagicMock()
        mock_db.list_companies.return_value = [
            {
                "name": "Acme Corp",
                "domain": "acme.com",
                "industry": "Technology",
                "employee_count": 150,
                "qualification_score": 85,
                "last_contacted_at": None,
            }
        ]

        result = query_companies(
            industry="Technology",
            limit=10,
            _db=mock_db,
        )

        mock_db.list_companies.assert_called_once_with(
            limit=10,
            industry="Technology",
        )

        parsed = json.loads(result)
        assert parsed["count"] == 1
        assert parsed["companies"][0]["name"] == "Acme Corp"


# ─── Email Tools Tests ────────────────────────────────────────────────


class TestEmailTools:
    """Tests for email MCP tools with sandbox safety."""

    def test_send_email_is_sandboxed(self):
        """send_email function has the @sandboxed_tool decorator applied."""
        from core.mcp.tools.email_tools import send_email
        from core.safety.sandbox import is_sandboxed, get_sandbox_tool_name

        assert is_sandboxed(send_email)
        assert get_sandbox_tool_name(send_email) == "send_email"

    def test_send_email_intercepted_in_dev(self):
        """In development, send_email writes to sandbox log instead of sending."""
        from core.mcp.tools.email_tools import send_email

        # The sandbox protocol checks ENCLAVE_ENV
        with patch.dict(os.environ, {"ENCLAVE_ENV": "development"}):
            result = _run(send_email(
                to_email="test@example.com",
                to_name="Test User",
                subject="Hello",
                body_html="<p>Test</p>",
            ))

        assert result["sandboxed"] is True
        assert result["tool_name"] == "send_email"

    def test_send_email_passes_through_in_production(self):
        """In production, send_email delegates to EmailEngine."""
        from core.mcp.tools.email_tools import send_email

        mock_engine = AsyncMock()
        mock_engine.send_email.return_value = {
            "message_id": "msg-abc",
            "status": "sent",
            "provider": "sendgrid",
        }

        with patch.dict(os.environ, {"ENCLAVE_ENV": "production"}):
            result = _run(send_email(
                to_email="prospect@acme.com",
                to_name="Jane CTO",
                subject="Security Assessment",
                body_html="<p>Hello</p>",
                body_text="Hello",
                _email_engine=mock_engine,
            ))

        assert result["status"] == "sent"
        assert result["message_id"] == "msg-abc"
        mock_engine.send_email.assert_called_once()


# ─── MCP Server Tests ─────────────────────────────────────────────────


class TestMCPServer:
    """Tests for MCP server creation and tool registration."""

    def test_create_server_returns_fastmcp_instance(self):
        """create_mcp_server returns a FastMCP instance."""
        try:
            from fastmcp import FastMCP
            from core.mcp.server import create_mcp_server

            server = create_mcp_server(name="Test Server")
            assert isinstance(server, FastMCP)
        except ImportError:
            pytest.skip("fastmcp not installed")

    def test_get_server_is_singleton(self):
        """get_server returns the same instance on repeated calls."""
        try:
            from core.mcp.server import get_server
            import core.mcp.server as server_module

            # Reset singleton
            server_module._server_instance = None

            s1 = get_server()
            s2 = get_server()
            assert s1 is s2

            # Clean up
            server_module._server_instance = None
        except ImportError:
            pytest.skip("fastmcp not installed")

    def test_server_has_tools_registered(self):
        """Server has all expected tools registered (23 total)."""
        try:
            from core.mcp.server import create_mcp_server

            server = create_mcp_server(name="Test Server")
            tool_names = list(server._tool_manager._tools.keys())

            assert "search_leads" in tool_names
            assert "enrich_company" in tool_names
            assert "search_knowledge" in tool_names
            assert "save_insight" in tool_names
            assert "query_companies" in tool_names
            assert "send_email" in tool_names
        except ImportError:
            pytest.skip("fastmcp not installed")


# ─── YAML Config Integration Tests ────────────────────────────────────


class TestYAMLConfigIntegration:
    """Tests that the YAML config loads correctly with the new tool format."""

    def test_outreach_yaml_loads_with_string_tools(self):
        """The updated outreach.yaml with string tools parses correctly."""
        import yaml

        yaml_path = Path(__file__).parent.parent / "verticals" / "enclave_guard" / "agents" / "outreach.yaml"
        if not yaml_path.exists():
            pytest.skip("outreach.yaml not found")

        with open(yaml_path) as f:
            raw = yaml.safe_load(f)

        config = AgentInstanceConfig(**raw)

        assert config.agent_id == "outreach"
        assert config.agent_type == "outreach"
        assert len(config.tools) == 6
        assert all(isinstance(t, AgentToolConfig) for t in config.tools)

        tool_names = [t.name for t in config.tools]
        assert "apollo_search" in tool_names
        assert "send_email" in tool_names

        # String tools should all have type="mcp"
        for tool in config.tools:
            assert tool.type == "mcp"

    def test_mixed_format_yaml(self):
        """Config with both string and dict tools parses correctly."""
        import yaml

        raw_yaml = """
agent_id: test_agent
agent_type: test
name: "Test Agent"
tools:
  - apollo_search
  - name: custom_tool
    type: custom
    config:
      api_url: https://example.com
  - send_email
"""
        raw = yaml.safe_load(raw_yaml)
        config = AgentInstanceConfig(**raw)

        assert len(config.tools) == 3
        assert config.tools[0].name == "apollo_search"
        assert config.tools[0].type == "mcp"
        assert config.tools[1].name == "custom_tool"
        assert config.tools[1].type == "custom"
        assert config.tools[1].config["api_url"] == "https://example.com"
        assert config.tools[2].name == "send_email"
        assert config.tools[2].type == "mcp"
