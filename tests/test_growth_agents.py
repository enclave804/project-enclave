"""
Tests for Phase 11: Growth Agents.

Covers:
    - Proposal Builder Agent (state, contracts, graph, nodes)
    - Social Media Agent (state, contracts, graph, nodes, connectors)
    - Ads Strategy Agent (state, contracts, graph, nodes)
    - Social media MCP tools
    - Twitter/X client (mock mode)
    - LinkedIn client (mock mode)
    - Content contracts (ProposalRequest, SocialMediaPost, AdCampaign, AdCreative)
    - DB migration schema (008_growth_agents.sql)
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ══════════════════════════════════════════════════════════════════════
# State Tests
# ══════════════════════════════════════════════════════════════════════


class TestProposalBuilderState:
    """Tests for ProposalBuilderAgentState TypedDict."""

    def test_import(self):
        from core.agents.state import ProposalBuilderAgentState
        assert ProposalBuilderAgentState is not None

    def test_create_minimal(self):
        from core.agents.state import ProposalBuilderAgentState
        state: ProposalBuilderAgentState = {
            "agent_id": "proposal_builder_v1",
            "vertical_id": "enclave_guard",
            "company_name": "Acme Corp",
        }
        assert state["company_name"] == "Acme Corp"

    def test_create_full(self):
        from core.agents.state import ProposalBuilderAgentState
        state: ProposalBuilderAgentState = {
            "agent_id": "proposal_builder_v1",
            "vertical_id": "enclave_guard",
            "company_name": "Acme Corp",
            "company_domain": "acme.com",
            "contact_name": "Jane Doe",
            "contact_email": "jane@acme.com",
            "proposal_type": "full_proposal",
            "pricing_tier": "professional",
            "pricing_amount": 15000.0,
            "timeline_weeks": 4,
            "deliverables": ["Assessment", "Report"],
            "draft_proposal": "# Proposal for Acme",
            "proposal_approved": False,
            "delivered": False,
        }
        assert state["pricing_amount"] == 15000.0
        assert state["timeline_weeks"] == 4
        assert len(state["deliverables"]) == 2

    def test_deal_stage_field(self):
        from core.agents.state import ProposalBuilderAgentState
        state: ProposalBuilderAgentState = {
            "deal_stage": "proposal",
        }
        assert state["deal_stage"] == "proposal"

    def test_revision_count_field(self):
        from core.agents.state import ProposalBuilderAgentState
        state: ProposalBuilderAgentState = {
            "revision_count": 2,
        }
        assert state["revision_count"] == 2


class TestSocialMediaState:
    """Tests for SocialMediaAgentState TypedDict."""

    def test_import(self):
        from core.agents.state import SocialMediaAgentState
        assert SocialMediaAgentState is not None

    def test_create_minimal(self):
        from core.agents.state import SocialMediaAgentState
        state: SocialMediaAgentState = {
            "agent_id": "social_v1",
            "platform": "twitter",
        }
        assert state["platform"] == "twitter"

    def test_create_with_posts(self):
        from core.agents.state import SocialMediaAgentState
        state: SocialMediaAgentState = {
            "platform": "linkedin",
            "draft_posts": [
                {"content": "Hello LinkedIn!", "platform": "linkedin"},
            ],
            "post_count": 1,
            "posts_published": 0,
        }
        assert state["post_count"] == 1
        assert len(state["draft_posts"]) == 1

    def test_engagement_metrics(self):
        from core.agents.state import SocialMediaAgentState
        state: SocialMediaAgentState = {
            "engagement_metrics": {
                "impressions": 5000,
                "clicks": 200,
                "shares": 50,
            },
        }
        assert state["engagement_metrics"]["impressions"] == 5000

    def test_content_calendar(self):
        from core.agents.state import SocialMediaAgentState
        state: SocialMediaAgentState = {
            "content_calendar": [
                {"date": "2025-01-15", "platform": "twitter", "topic": "AI"},
            ],
        }
        assert len(state["content_calendar"]) == 1


class TestAdsStrategyState:
    """Tests for AdsStrategyAgentState TypedDict."""

    def test_import(self):
        from core.agents.state import AdsStrategyAgentState
        assert AdsStrategyAgentState is not None

    def test_create_minimal(self):
        from core.agents.state import AdsStrategyAgentState
        state: AdsStrategyAgentState = {
            "platform": "google",
            "campaign_objective": "lead_gen",
        }
        assert state["campaign_objective"] == "lead_gen"

    def test_budget_fields(self):
        from core.agents.state import AdsStrategyAgentState
        state: AdsStrategyAgentState = {
            "budget_daily": 100.0,
            "budget_total": 3000.0,
            "target_cpa": 25.0,
        }
        assert state["budget_daily"] == 100.0

    def test_keyword_research(self):
        from core.agents.state import AdsStrategyAgentState
        state: AdsStrategyAgentState = {
            "seed_keywords": ["cybersecurity", "penetration testing"],
            "keyword_research": [
                {"keyword": "cybersecurity consulting", "volume": 5000, "cpc": 12.50},
            ],
        }
        assert len(state["seed_keywords"]) == 2


# ══════════════════════════════════════════════════════════════════════
# Contract Tests
# ══════════════════════════════════════════════════════════════════════


class TestProposalContracts:
    """Tests for proposal-related data contracts."""

    def test_proposal_request_create(self):
        from core.agents.contracts import ProposalRequest
        req = ProposalRequest(
            company_name="Acme Corp",
            contact_name="Jane Doe",
            contact_email="jane@acme.com",
        )
        assert req.company_name == "Acme Corp"
        assert req.proposal_type == "full_proposal"
        assert req.pricing_tier == "professional"

    def test_proposal_request_custom_type(self):
        from core.agents.contracts import ProposalRequest
        req = ProposalRequest(
            company_name="BigCo",
            contact_name="Bob",
            contact_email="bob@bigco.com",
            proposal_type="sow",
            pricing_tier="enterprise",
        )
        assert req.proposal_type == "sow"
        assert req.pricing_tier == "enterprise"

    def test_generated_proposal_create(self):
        from core.agents.contracts import GeneratedProposal
        prop = GeneratedProposal(
            title="Security Assessment Proposal",
            proposal_type="full_proposal",
            full_markdown="# Proposal\n\nContent here",
            pricing_amount=15000.0,
            timeline_weeks=4,
        )
        assert prop.pricing_amount == 15000.0
        assert prop.timeline_weeks == 4

    def test_generated_proposal_with_sections(self):
        from core.agents.contracts import GeneratedProposal
        prop = GeneratedProposal(
            title="SOW",
            proposal_type="sow",
            sections=[
                {"title": "Scope", "content": "Full scope", "order": 0},
                {"title": "Timeline", "content": "4 weeks", "order": 1},
            ],
        )
        assert len(prop.sections) == 2

    def test_proposal_request_serialization(self):
        from core.agents.contracts import ProposalRequest
        req = ProposalRequest(
            company_name="Test",
            contact_name="Alice",
            contact_email="alice@test.com",
            custom_requirements=["SOC 2", "ISO 27001"],
        )
        data = req.model_dump()
        assert data["custom_requirements"] == ["SOC 2", "ISO 27001"]


class TestSocialContracts:
    """Tests for social media data contracts."""

    def test_social_media_post_create(self):
        from core.agents.contracts import SocialMediaPost
        post = SocialMediaPost(
            platform="twitter",
            content="Hello world! #CyberSecurity",
        )
        assert post.platform == "twitter"
        assert post.post_type == "thought_leadership"

    def test_social_media_post_with_hashtags(self):
        from core.agents.contracts import SocialMediaPost
        post = SocialMediaPost(
            platform="linkedin",
            content="Big announcement!",
            hashtags=["#InfoSec", "#AI"],
            post_type="product_launch",
        )
        assert len(post.hashtags) == 2
        assert post.post_type == "product_launch"

    def test_content_calendar_entry(self):
        from core.agents.contracts import ContentCalendarEntry
        entry = ContentCalendarEntry(
            date="2025-02-15",
            platform="twitter",
            topic="Zero trust architecture",
        )
        assert entry.status == "planned"

    def test_ad_campaign_create(self):
        from core.agents.contracts import AdCampaign
        campaign = AdCampaign(
            platform="google",
            campaign_name="Cybersecurity Leads Q1",
            budget_daily=100.0,
        )
        assert campaign.objective == "lead_gen"

    def test_ad_creative_create(self):
        from core.agents.contracts import AdCreative
        ad = AdCreative(
            headline="Protect Your Business Today",
            description="Expert cybersecurity consulting",
            call_to_action="Get Assessment",
        )
        assert ad.headline == "Protect Your Business Today"


# ══════════════════════════════════════════════════════════════════════
# Twitter Client Tests (Mock Mode)
# ══════════════════════════════════════════════════════════════════════


class TestTwitterClient:
    """Tests for TwitterClient in mock mode."""

    def test_auto_mock_mode(self):
        from core.integrations.social.twitter_client import TwitterClient
        client = TwitterClient()  # No API key
        assert client.mock_mode is True

    def test_explicit_mock_mode(self):
        from core.integrations.social.twitter_client import TwitterClient
        client = TwitterClient(api_key="test", mock_mode=True)
        assert client.mock_mode is True

    def test_real_mode_with_key(self):
        from core.integrations.social.twitter_client import TwitterClient
        client = TwitterClient(api_key="test_key", mock_mode=False)
        assert client.mock_mode is False

    def test_from_env_no_keys(self):
        from core.integrations.social.twitter_client import TwitterClient
        with patch.dict(os.environ, {}, clear=True):
            client = TwitterClient.from_env()
            assert client.mock_mode is True

    @pytest.mark.asyncio
    async def test_post_tweet_mock(self, tmp_path):
        from core.integrations.social.twitter_client import TwitterClient
        mock_file = tmp_path / "social_mock.json"
        client = TwitterClient(mock_mode=True, mock_storage_path=mock_file)
        result = await client.post_tweet("Hello from test! 🔒")
        assert result["mock"] is True
        assert result["action"] == "post_tweet"
        assert result["platform"] == "twitter"
        # Check file was written
        assert mock_file.exists()
        entries = json.loads(mock_file.read_text())
        assert len(entries) == 1
        assert entries[0]["data"]["text"] == "Hello from test! 🔒"

    @pytest.mark.asyncio
    async def test_post_tweet_truncates(self, tmp_path):
        from core.integrations.social.twitter_client import TwitterClient
        mock_file = tmp_path / "social_mock.json"
        client = TwitterClient(mock_mode=True, mock_storage_path=mock_file)
        long_text = "A" * 500
        result = await client.post_tweet(long_text)
        data_text = result["data"]["text"]
        assert len(data_text) <= 280

    @pytest.mark.asyncio
    async def test_reply_mock(self, tmp_path):
        from core.integrations.social.twitter_client import TwitterClient
        mock_file = tmp_path / "social_mock.json"
        client = TwitterClient(mock_mode=True, mock_storage_path=mock_file)
        result = await client.reply("12345", "Great point!")
        assert result["data"]["in_reply_to"] == "12345"

    @pytest.mark.asyncio
    async def test_get_mentions_mock(self, tmp_path):
        from core.integrations.social.twitter_client import TwitterClient
        client = TwitterClient(mock_mode=True, mock_storage_path=tmp_path / "mock.json")
        mentions = await client.get_mentions(limit=5)
        assert mentions == []

    @pytest.mark.asyncio
    async def test_get_trending_mock(self, tmp_path):
        from core.integrations.social.twitter_client import TwitterClient
        client = TwitterClient(mock_mode=True, mock_storage_path=tmp_path / "mock.json")
        topics = await client.get_trending_topics(niche="cybersecurity")
        assert len(topics) == 3
        assert any("CyberSecurity" in t["topic"] for t in topics)

    @pytest.mark.asyncio
    async def test_mock_storage_accumulates(self, tmp_path):
        from core.integrations.social.twitter_client import TwitterClient
        mock_file = tmp_path / "social_mock.json"
        client = TwitterClient(mock_mode=True, mock_storage_path=mock_file)
        await client.post_tweet("First post")
        await client.post_tweet("Second post")
        entries = json.loads(mock_file.read_text())
        assert len(entries) == 2


# ══════════════════════════════════════════════════════════════════════
# LinkedIn Client Tests (Mock Mode)
# ══════════════════════════════════════════════════════════════════════


class TestLinkedInClient:
    """Tests for LinkedInClient in mock mode."""

    def test_auto_mock_mode(self):
        from core.integrations.social.linkedin_client import LinkedInClient
        client = LinkedInClient()
        assert client.mock_mode is True

    def test_real_mode_with_token(self):
        from core.integrations.social.linkedin_client import LinkedInClient
        client = LinkedInClient(access_token="test_token", mock_mode=False)
        assert client.mock_mode is False

    def test_from_env_no_keys(self):
        from core.integrations.social.linkedin_client import LinkedInClient
        with patch.dict(os.environ, {}, clear=True):
            client = LinkedInClient.from_env()
            assert client.mock_mode is True

    @pytest.mark.asyncio
    async def test_post_share_mock(self, tmp_path):
        from core.integrations.social.linkedin_client import LinkedInClient
        mock_file = tmp_path / "social_mock.json"
        client = LinkedInClient(mock_mode=True, mock_storage_path=mock_file)
        result = await client.post_share("Big news in cybersecurity!")
        assert result["mock"] is True
        assert result["action"] == "post_share"
        assert result["platform"] == "linkedin"

    @pytest.mark.asyncio
    async def test_post_share_truncates(self, tmp_path):
        from core.integrations.social.linkedin_client import LinkedInClient
        mock_file = tmp_path / "social_mock.json"
        client = LinkedInClient(mock_mode=True, mock_storage_path=mock_file)
        long_text = "A" * 5000
        result = await client.post_share(long_text)
        data_text = result["data"]["text"]
        assert len(data_text) <= 3000

    @pytest.mark.asyncio
    async def test_get_company_updates_mock(self, tmp_path):
        from core.integrations.social.linkedin_client import LinkedInClient
        client = LinkedInClient(mock_mode=True, mock_storage_path=tmp_path / "mock.json")
        updates = await client.get_company_updates()
        assert updates == []

    @pytest.mark.asyncio
    async def test_engagement_metrics_mock(self, tmp_path):
        from core.integrations.social.linkedin_client import LinkedInClient
        client = LinkedInClient(mock_mode=True, mock_storage_path=tmp_path / "mock.json")
        metrics = await client.get_engagement_metrics("post_123")
        assert metrics["mock"] is True
        assert metrics["likes"] == 0

    def test_author_urn_org(self):
        from core.integrations.social.linkedin_client import LinkedInClient
        client = LinkedInClient(org_id="12345")
        assert client._get_author_urn() == "urn:li:organization:12345"

    def test_author_urn_person(self):
        from core.integrations.social.linkedin_client import LinkedInClient
        client = LinkedInClient(person_urn="urn:li:person:abc")
        assert client._get_author_urn() == "urn:li:person:abc"


# ══════════════════════════════════════════════════════════════════════
# Social MCP Tools Tests
# ══════════════════════════════════════════════════════════════════════


class TestSocialMCPTools:
    """Tests for social_tools.py MCP tool functions."""

    @pytest.mark.asyncio
    async def test_post_social_update_twitter(self):
        with patch("core.integrations.social.twitter_client.TwitterClient.from_env") as mock_env:
            mock_client = MagicMock()
            mock_client.post_tweet = AsyncMock(return_value={"mock": True, "action": "post_tweet"})
            mock_env.return_value = mock_client

            from core.mcp.tools.social_tools import post_social_update
            result = await post_social_update("twitter", "Hello!")
            data = json.loads(result)
            assert data["mock"] is True

    @pytest.mark.asyncio
    async def test_post_social_update_linkedin(self):
        with patch("core.integrations.social.linkedin_client.LinkedInClient.from_env") as mock_env:
            mock_client = MagicMock()
            mock_client.post_share = AsyncMock(return_value={"mock": True, "action": "post_share"})
            mock_env.return_value = mock_client

            from core.mcp.tools.social_tools import post_social_update
            result = await post_social_update("linkedin", "Hello LinkedIn!")
            data = json.loads(result)
            assert data["mock"] is True

    @pytest.mark.asyncio
    async def test_post_social_update_invalid_platform(self):
        from core.mcp.tools.social_tools import post_social_update
        result = await post_social_update("tiktok", "Hello!")
        data = json.loads(result)
        assert "error" in data

    @pytest.mark.asyncio
    async def test_get_social_mentions_twitter(self):
        with patch("core.integrations.social.twitter_client.TwitterClient.from_env") as mock_env:
            mock_client = MagicMock()
            mock_client.get_mentions = AsyncMock(return_value=[])
            mock_env.return_value = mock_client

            from core.mcp.tools.social_tools import get_social_mentions
            result = await get_social_mentions("twitter", limit=5)
            data = json.loads(result)
            assert data["platform"] == "twitter"
            assert data["count"] == 0

    @pytest.mark.asyncio
    async def test_reply_to_post_twitter(self):
        with patch("core.integrations.social.twitter_client.TwitterClient.from_env") as mock_env:
            mock_client = MagicMock()
            mock_client.reply = AsyncMock(return_value={"action": "reply"})
            mock_env.return_value = mock_client

            from core.mcp.tools.social_tools import reply_to_post
            result = await reply_to_post("twitter", "12345", "Great!")
            data = json.loads(result)
            assert data["action"] == "reply"

    @pytest.mark.asyncio
    async def test_check_trending_topics(self):
        with patch("core.integrations.social.twitter_client.TwitterClient.from_env") as mock_env:
            mock_client = MagicMock()
            mock_client.get_trending_topics = AsyncMock(return_value=[
                {"topic": "#CyberSecurity", "tweet_volume": 50000}
            ])
            mock_env.return_value = mock_client

            from core.mcp.tools.social_tools import check_trending_topics
            result = await check_trending_topics("cybersecurity")
            data = json.loads(result)
            assert data["count"] == 1


# ══════════════════════════════════════════════════════════════════════
# Proposal Builder Agent Tests
# ══════════════════════════════════════════════════════════════════════


class TestProposalBuilderAgent:
    """Tests for ProposalBuilderAgent implementation."""

    def _make_agent(self, **kwargs):
        """Create a ProposalBuilderAgent with mocked dependencies."""
        from core.agents.implementations.proposal_agent import ProposalBuilderAgent
        from core.config.agent_schema import AgentInstanceConfig

        config = AgentInstanceConfig(
            agent_id="proposal_builder_v1",
            agent_type="proposal_builder",
            name="Proposal Builder Agent",
            vertical_id="enclave_guard",
            **kwargs,
        )
        db = MagicMock()
        db.search_knowledge = MagicMock(return_value=[])
        db.search_insights = MagicMock(return_value=[])
        db.store_insight = MagicMock(return_value={"id": "test"})
        db.log_agent_run = MagicMock()
        db.reset_agent_errors = MagicMock()
        db.store_training_example = MagicMock(return_value={"id": "rlhf"})

        embedder = MagicMock()
        embedder.embed_query = MagicMock(return_value=[0.0] * 1536)

        llm = MagicMock()
        llm.messages = MagicMock()
        llm.messages.create = MagicMock(return_value=MagicMock(
            content=[MagicMock(text='["SOC 2 compliance gap", "No incident response plan"]')]
        ))

        return ProposalBuilderAgent(
            config=config, db=db, embedder=embedder, anthropic_client=llm,
        )

    def test_registration(self):
        from core.agents.implementations.proposal_agent import ProposalBuilderAgent
        assert ProposalBuilderAgent.agent_type == "proposal_builder"

    def test_build_graph(self):
        agent = self._make_agent()
        graph = agent.build_graph()
        assert graph is not None

    def test_get_state_class(self):
        from core.agents.state import ProposalBuilderAgentState
        agent = self._make_agent()
        assert agent.get_state_class() is ProposalBuilderAgentState

    def test_get_tools(self):
        agent = self._make_agent()
        assert agent.get_tools() == []

    def test_prepare_initial_state(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state(
            {"company_name": "Acme Corp", "pricing_tier": "enterprise"},
            "run-123",
        )
        assert state["company_name"] == "Acme Corp"
        assert state["pricing_tier"] == "enterprise"
        assert state["proposal_approved"] is False
        assert state["delivered"] is False

    def test_parse_sections(self):
        agent = self._make_agent()
        markdown = "# Main Title\n\nIntro text\n\n## Scope\n\nScope details\n\n## Timeline\n\n4 weeks"
        sections = agent._parse_sections(markdown)
        assert len(sections) >= 2
        assert any(s["title"] == "Scope" for s in sections)

    def test_parse_sections_empty(self):
        agent = self._make_agent()
        sections = agent._parse_sections("")
        assert sections == []

    def test_system_prompt_default(self):
        agent = self._make_agent()
        prompt = agent._get_system_prompt()
        assert "proposal" in prompt.lower()
        assert "cybersecurity" in prompt.lower()

    @pytest.mark.asyncio
    async def test_gather_context_node(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state(
            {"company_name": "Test Corp", "company_domain": "test.com"},
            "run-1",
        )
        result = await agent._node_gather_context(state)
        assert result["current_node"] == "gather_context"
        assert "rag_context" in result

    @pytest.mark.asyncio
    async def test_research_company_node(self):
        agent = self._make_agent()
        state = {"company_name": "Test", "company_domain": "", "meeting_notes": "Need security audit"}
        result = await agent._node_research_company(state)
        assert result["current_node"] == "research_company"
        assert "company_pain_points" in result

    @pytest.mark.asyncio
    async def test_human_review_node(self):
        agent = self._make_agent()
        state = {"company_name": "Test", "proposal_type": "sow", "pricing_amount": 5000}
        result = await agent._node_human_review(state)
        assert result["requires_human_approval"] is True

    @pytest.mark.asyncio
    async def test_finalize_no_edits(self):
        agent = self._make_agent()
        state = {
            "draft_proposal": "# Proposal\n\nContent",
            "human_edited_proposal": None,
            "revision_count": 0,
        }
        result = await agent._node_finalize(state)
        assert result["proposal_approved"] is True
        assert result["rlhf_captured"] is False

    @pytest.mark.asyncio
    async def test_finalize_with_edits(self):
        agent = self._make_agent()
        state = {
            "draft_proposal": "# Original",
            "human_edited_proposal": "# Edited and Improved",
            "revision_count": 0,
            "company_name": "Test",
            "proposal_type": "full_proposal",
            "pricing_tier": "professional",
            "company_pain_points": [],
        }
        result = await agent._node_finalize(state)
        assert result["rlhf_captured"] is True
        assert result["revision_count"] == 1

    def test_route_after_review_approved(self):
        from core.agents.implementations.proposal_agent import ProposalBuilderAgent
        assert ProposalBuilderAgent._route_after_review({"human_approval_status": "approved"}) == "approved"

    def test_route_after_review_rejected(self):
        from core.agents.implementations.proposal_agent import ProposalBuilderAgent
        assert ProposalBuilderAgent._route_after_review({"human_approval_status": "rejected"}) == "rejected"

    def test_repr(self):
        agent = self._make_agent()
        repr_str = repr(agent)
        assert "ProposalBuilderAgent" in repr_str
        assert "proposal_builder_v1" in repr_str


# ══════════════════════════════════════════════════════════════════════
# Social Media Agent Tests
# ══════════════════════════════════════════════════════════════════════


class TestSocialMediaAgent:
    """Tests for SocialMediaAgent implementation."""

    def _make_agent(self, **kwargs):
        """Create a SocialMediaAgent with mocked dependencies."""
        from core.agents.implementations.social_agent import SocialMediaAgent
        from core.config.agent_schema import AgentInstanceConfig

        config = AgentInstanceConfig(
            agent_id="social_v1",
            agent_type="social",
            name="Social Media Agent",
            vertical_id="enclave_guard",
            params={"niche": "cybersecurity", "company_name": "Enclave Guard", "posts_per_run": 2},
            **kwargs,
        )
        db = MagicMock()
        db.search_insights = MagicMock(return_value=[])
        db.store_insight = MagicMock(return_value={"id": "test"})
        db.log_agent_run = MagicMock()
        db.reset_agent_errors = MagicMock()
        db.store_training_example = MagicMock(return_value={"id": "rlhf"})

        embedder = MagicMock()
        embedder.embed_query = MagicMock(return_value=[0.0] * 1536)

        llm = MagicMock()
        llm.messages = MagicMock()

        return SocialMediaAgent(
            config=config, db=db, embedder=embedder, anthropic_client=llm,
        )

    def test_registration(self):
        from core.agents.implementations.social_agent import SocialMediaAgent
        assert SocialMediaAgent.agent_type == "social"

    def test_build_graph(self):
        agent = self._make_agent()
        graph = agent.build_graph()
        assert graph is not None

    def test_get_state_class(self):
        from core.agents.state import SocialMediaAgentState
        agent = self._make_agent()
        assert agent.get_state_class() is SocialMediaAgentState

    def test_get_tools(self):
        agent = self._make_agent()
        assert agent.get_tools() == []

    def test_prepare_initial_state(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state(
            {"platform": "linkedin", "topic": "zero trust"},
            "run-123",
        )
        assert state["platform"] == "linkedin"
        assert state["posts_published"] == 0
        assert state["draft_posts"] == []

    @pytest.mark.asyncio
    async def test_listen_node(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state(
            {"platform": "twitter", "topic": "cybersecurity"},
            "run-1",
        )
        with patch("core.mcp.tools.social_tools.check_trending_topics",
                    new_callable=AsyncMock,
                    return_value='{"topics": [{"topic": "#CyberSecurity"}], "count": 1}'):
            result = await agent._node_listen(state)
        assert result["current_node"] == "listen"
        assert "trending_topics" in result

    @pytest.mark.asyncio
    async def test_ideate_node(self):
        agent = self._make_agent()
        # Mock LLM response
        agent.llm.messages.create = MagicMock(return_value=MagicMock(
            content=[MagicMock(text=json.dumps([
                {
                    "post_type": "thought_leadership",
                    "topic": "cybersecurity",
                    "hook": "Here's what most people get wrong...",
                    "key_message": "Zero trust is not optional",
                    "cta": "Follow for more",
                    "needs_image": False,
                    "hashtags": ["#CyberSecurity"],
                }
            ]))]
        ))
        state = {
            "platform": "twitter",
            "trending_topics": ["#CyberSecurity"],
            "outreach_insights": [],
            "seo_keywords": [],
            "task_input": {"topic": "cybersecurity", "num_posts": 1},
        }
        result = await agent._node_ideate(state)
        assert result["current_node"] == "ideate"
        assert len(result["content_calendar"]) >= 1

    @pytest.mark.asyncio
    async def test_ideate_fallback(self):
        agent = self._make_agent()
        agent.llm.messages.create = MagicMock(side_effect=Exception("LLM error"))
        state = {
            "platform": "twitter",
            "trending_topics": [],
            "outreach_insights": [],
            "seo_keywords": [],
            "task_input": {"topic": "cybersecurity"},
        }
        result = await agent._node_ideate(state)
        assert len(result["content_calendar"]) >= 1

    @pytest.mark.asyncio
    async def test_create_node(self):
        agent = self._make_agent()
        agent.llm.messages.create = MagicMock(return_value=MagicMock(
            content=[MagicMock(text="Here's what most people miss about cybersecurity... #CyberSecurity")]
        ))
        state = {
            "platform": "twitter",
            "content_calendar": [
                {
                    "post_type": "tip",
                    "topic": "cybersecurity",
                    "hook": "Most people miss this...",
                    "key_message": "Zero trust matters",
                    "cta": "",
                    "hashtags": ["#CyberSecurity"],
                    "needs_image": False,
                },
            ],
        }
        result = await agent._node_create(state)
        assert result["current_node"] == "create"
        assert result["post_count"] >= 1
        assert len(result["draft_posts"]) >= 1

    @pytest.mark.asyncio
    async def test_human_review_node(self):
        agent = self._make_agent()
        state = {"platform": "twitter", "post_count": 2}
        result = await agent._node_human_review(state)
        assert result["requires_human_approval"] is True

    @pytest.mark.asyncio
    async def test_publish_node(self):
        agent = self._make_agent()
        state = {
            "platform": "twitter",
            "draft_posts": [
                {"content": "Test post #CyberSecurity", "post_type": "tip", "index": 0},
            ],
            "human_edits": [],
        }
        with patch("core.mcp.tools.social_tools.post_social_update",
                    new_callable=AsyncMock,
                    return_value='{"mock": true, "action": "post_tweet"}'):
            result = await agent._node_publish(state)
        assert result["posts_published"] >= 1
        assert result["knowledge_written"] is True

    @pytest.mark.asyncio
    async def test_publish_with_edits(self):
        agent = self._make_agent()
        state = {
            "platform": "twitter",
            "draft_posts": [
                {"content": "Original post", "post_type": "tip", "topic": "cyber", "index": 0},
            ],
            "human_edits": [
                {"index": 0, "edited_content": "Improved post"},
            ],
        }
        with patch("core.mcp.tools.social_tools.post_social_update",
                    new_callable=AsyncMock,
                    return_value='{"mock": true}'):
            result = await agent._node_publish(state)
        assert result["posts_published"] == 1
        # RLHF should have been captured via learn()
        agent.db.store_training_example.assert_called_once()

    @pytest.mark.asyncio
    async def test_report_node(self):
        agent = self._make_agent()
        state = {
            "platform": "twitter",
            "post_count": 3,
            "posts_published": 2,
            "posts_approved": True,
            "trending_topics": ["#AI"],
            "engagement_metrics": {},
        }
        result = await agent._node_report(state)
        assert result["current_node"] == "report"
        assert "Social Media Activity Report" in result["report_summary"]

    def test_route_after_review_approved(self):
        from core.agents.implementations.social_agent import SocialMediaAgent
        assert SocialMediaAgent._route_after_review({"human_approval_status": "approved"}) == "approved"

    def test_route_after_review_rejected(self):
        from core.agents.implementations.social_agent import SocialMediaAgent
        assert SocialMediaAgent._route_after_review({"human_approval_status": "rejected"}) == "rejected"

    def test_system_prompt_default(self):
        agent = self._make_agent()
        prompt = agent._get_system_prompt()
        assert "social media" in prompt.lower()

    def test_repr(self):
        agent = self._make_agent()
        assert "SocialMediaAgent" in repr(agent)


# ══════════════════════════════════════════════════════════════════════
# Agent Registry Integration Tests
# ══════════════════════════════════════════════════════════════════════


class TestGrowthAgentRegistry:
    """Test that Growth Agents register correctly."""

    def test_proposal_builder_in_registry(self):
        from core.agents.implementations.proposal_agent import ProposalBuilderAgent  # noqa: F401
        from core.agents.registry import AGENT_IMPLEMENTATIONS
        assert "proposal_builder" in AGENT_IMPLEMENTATIONS

    def test_social_in_registry(self):
        from core.agents.implementations.social_agent import SocialMediaAgent  # noqa: F401
        from core.agents.registry import AGENT_IMPLEMENTATIONS
        assert "social" in AGENT_IMPLEMENTATIONS


# ══════════════════════════════════════════════════════════════════════
# YAML Config Tests
# ══════════════════════════════════════════════════════════════════════


class TestGrowthAgentYAML:
    """Test that Growth Agent YAML configs parse correctly."""

    def test_proposal_builder_yaml_loads(self):
        import yaml
        from core.config.agent_schema import AgentInstanceConfig
        yaml_path = Path(__file__).parent.parent / "verticals" / "enclave_guard" / "agents" / "proposal_builder.yaml"
        if yaml_path.exists():
            with open(yaml_path) as f:
                raw = yaml.safe_load(f)
            raw["vertical_id"] = "enclave_guard"
            config = AgentInstanceConfig(**raw)
            assert config.agent_type == "proposal_builder"
            assert config.agent_id == "proposal_builder_v1"

    def test_social_yaml_loads(self):
        import yaml
        from core.config.agent_schema import AgentInstanceConfig
        yaml_path = Path(__file__).parent.parent / "verticals" / "enclave_guard" / "agents" / "social.yaml"
        if yaml_path.exists():
            with open(yaml_path) as f:
                raw = yaml.safe_load(f)
            raw["vertical_id"] = "enclave_guard"
            config = AgentInstanceConfig(**raw)
            assert config.agent_type == "social"
            assert config.agent_id == "social_v1"
            assert config.params["niche"] == "cybersecurity"


# ══════════════════════════════════════════════════════════════════════
# Migration SQL Tests
# ══════════════════════════════════════════════════════════════════════


class TestGrowthMigration:
    """Test that the migration SQL file is valid."""

    def test_migration_file_exists(self):
        path = Path(__file__).parent.parent / "infrastructure" / "migrations" / "008_growth_agents.sql"
        assert path.exists()

    def test_migration_has_proposals_table(self):
        path = Path(__file__).parent.parent / "infrastructure" / "migrations" / "008_growth_agents.sql"
        content = path.read_text()
        assert "CREATE TABLE IF NOT EXISTS proposals" in content

    def test_migration_has_social_posts_table(self):
        path = Path(__file__).parent.parent / "infrastructure" / "migrations" / "008_growth_agents.sql"
        content = path.read_text()
        assert "CREATE TABLE IF NOT EXISTS social_posts" in content

    def test_migration_has_ad_campaigns_table(self):
        path = Path(__file__).parent.parent / "infrastructure" / "migrations" / "008_growth_agents.sql"
        content = path.read_text()
        assert "CREATE TABLE IF NOT EXISTS ad_campaigns" in content

    def test_migration_has_content_calendar_table(self):
        path = Path(__file__).parent.parent / "infrastructure" / "migrations" / "008_growth_agents.sql"
        content = path.read_text()
        assert "CREATE TABLE IF NOT EXISTS content_calendar" in content

    def test_migration_has_growth_stats_rpc(self):
        path = Path(__file__).parent.parent / "infrastructure" / "migrations" / "008_growth_agents.sql"
        content = path.read_text()
        assert "get_growth_stats" in content


# ══════════════════════════════════════════════════════════════════════
# Ads Strategy Agent Tests
# ══════════════════════════════════════════════════════════════════════


class TestAdsStrategyAgent:
    """Tests for AdsStrategyAgent implementation."""

    def _make_agent(self, **kwargs):
        """Create an AdsStrategyAgent with mocked dependencies."""
        from core.agents.implementations.ads_agent import AdsStrategyAgent
        from core.config.agent_schema import AgentInstanceConfig

        config = AgentInstanceConfig(
            agent_id="ads_strategy_v1",
            agent_type="ads_strategy",
            name="Ads Strategy Agent",
            vertical_id="enclave_guard",
            **kwargs,
        )
        db = MagicMock()
        db.search_knowledge = MagicMock(return_value=[])
        db.search_insights = MagicMock(return_value=[])
        db.store_insight = MagicMock(return_value={"id": "test"})
        db.log_agent_run = MagicMock()
        db.reset_agent_errors = MagicMock()
        db.store_training_example = MagicMock(return_value={"id": "rlhf"})

        embedder = MagicMock()
        embedder.embed_query = MagicMock(return_value=[0.0] * 1536)

        llm = MagicMock()
        llm.messages = MagicMock()
        llm.messages.create = MagicMock(return_value=MagicMock(
            content=[MagicMock(text=json.dumps({
                "ad_groups": [
                    {
                        "name": "Cybersecurity Assessment",
                        "keywords": [
                            {"keyword": "cybersecurity assessment", "volume": 5000, "cpc": 12.50, "competition": "high"},
                            {"keyword": "security audit", "volume": 3000, "cpc": 10.00, "competition": "medium"},
                        ],
                    }
                ],
                "negative_keywords": ["free", "cheap", "diy"],
                "target_audience": {
                    "demographics": "IT decision makers, 35-55",
                    "interests": ["cybersecurity", "compliance"],
                    "behaviors": ["software purchasers"],
                },
            }))]
        ))

        return AdsStrategyAgent(
            config=config, db=db, embedder=embedder, anthropic_client=llm,
        )

    def test_registration(self):
        from core.agents.implementations.ads_agent import AdsStrategyAgent
        assert AdsStrategyAgent.agent_type == "ads_strategy"

    def test_build_graph(self):
        agent = self._make_agent()
        graph = agent.build_graph()
        assert graph is not None

    def test_get_state_class(self):
        from core.agents.state import AdsStrategyAgentState
        agent = self._make_agent()
        assert agent.get_state_class() is AdsStrategyAgentState

    def test_get_tools(self):
        agent = self._make_agent()
        tools = agent.get_tools()
        assert isinstance(tools, list)

    def test_prepare_initial_state(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state(
            task={"platform": "google", "objective": "lead_gen", "budget_daily": 75.0},
            run_id="test-run-1",
        )
        assert state["platform"] == "google"
        assert state["campaign_objective"] == "lead_gen"
        assert state["budget_daily"] == 75.0
        assert state["generated_ads"] == []
        assert state["deployed_campaigns"] == []
        assert state["campaigns_approved"] is False

    def test_budget_guardrails_cap(self):
        """Budget should be capped at max_daily_budget from config params."""
        agent = self._make_agent(params={"max_daily_budget": 100.0, "max_total_budget": 2000.0})
        state = agent._prepare_initial_state(
            task={"budget_daily": 999.0, "budget_total": 99999.0},
            run_id="test-run-2",
        )
        assert state["budget_daily"] == 100.0
        assert state["budget_total"] == 2000.0

    def test_budget_guardrails_min_cpa(self):
        """CPA target should have a minimum floor."""
        agent = self._make_agent()
        state = agent._prepare_initial_state(
            task={"target_cpa": 0.001},
            run_id="test-run-3",
        )
        assert state["target_cpa"] >= 1.0

    def test_analyze_performance_node(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state(
            task={"platform": "google", "seed_keywords": ["pen testing"]},
            run_id="test-run-4",
        )
        result = asyncio.get_event_loop().run_until_complete(
            agent._node_analyze_performance(state)
        )
        assert result["current_node"] == "analyze_performance"
        assert isinstance(result.get("optimization_suggestions"), list)
        assert len(result["optimization_suggestions"]) > 0

    def test_research_keywords_node(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state(
            task={"platform": "google", "seed_keywords": ["cybersecurity assessment"]},
            run_id="test-run-5",
        )
        result = asyncio.get_event_loop().run_until_complete(
            agent._node_research_keywords(state)
        )
        assert result["current_node"] == "research_keywords"
        assert isinstance(result.get("keyword_research"), list)
        assert isinstance(result.get("ad_groups"), list)
        assert len(result["ad_groups"]) > 0
        assert isinstance(result.get("negative_keywords"), list)
        assert isinstance(result.get("selected_keywords"), list)

    def test_research_keywords_fallback(self):
        """If LLM fails, should produce fallback ad group from seeds."""
        agent = self._make_agent()
        agent.llm.messages.create = MagicMock(side_effect=RuntimeError("LLM down"))
        state = agent._prepare_initial_state(
            task={"seed_keywords": ["security audit", "pen test"]},
            run_id="test-run-6",
        )
        result = asyncio.get_event_loop().run_until_complete(
            agent._node_research_keywords(state)
        )
        assert result["current_node"] == "research_keywords"
        assert len(result["ad_groups"]) == 1
        assert result["ad_groups"][0]["name"].endswith("- General")

    def test_generate_campaigns_node(self):
        agent = self._make_agent()
        # Set up LLM to return ad variants
        agent.llm.messages.create = MagicMock(return_value=MagicMock(
            content=[MagicMock(text=json.dumps([
                {"headline": "Expert Security Audit", "description": "Protect your business today.", "call_to_action": "Get Started", "variant_id": "A", "tone": "authority"},
                {"headline": "Stop Data Breaches", "description": "Free assessment available.", "call_to_action": "Learn More", "variant_id": "B", "tone": "urgency"},
            ]))]
        ))
        state = agent._prepare_initial_state(
            task={"platform": "google"},
            run_id="test-run-7",
        )
        state["ad_groups"] = [{"name": "Security", "keywords": [{"keyword": "security"}]}]
        result = asyncio.get_event_loop().run_until_complete(
            agent._node_generate_campaigns(state)
        )
        assert result["current_node"] == "generate_campaigns"
        assert isinstance(result.get("generated_ads"), list)
        assert len(result["generated_ads"]) >= 1
        assert isinstance(result.get("ad_variants"), list)

    def test_generate_campaigns_fallback(self):
        """If LLM fails, should produce fallback ad."""
        agent = self._make_agent()
        agent.llm.messages.create = MagicMock(side_effect=RuntimeError("LLM down"))
        state = agent._prepare_initial_state(
            task={"platform": "google"},
            run_id="test-run-8",
        )
        state["ad_groups"] = [{"name": "General", "keywords": []}]
        result = asyncio.get_event_loop().run_until_complete(
            agent._node_generate_campaigns(state)
        )
        assert len(result["generated_ads"]) == 1
        assert result["generated_ads"][0]["variant_id"] == "A"

    def test_human_review_node(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state(
            task={"platform": "google"},
            run_id="test-run-9",
        )
        state["generated_ads"] = [{"headline": "Test Ad"}]
        result = asyncio.get_event_loop().run_until_complete(
            agent._node_human_review(state)
        )
        assert result["current_node"] == "human_review"
        assert result["requires_human_approval"] is True

    def test_deploy_node(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state(
            task={"platform": "google", "objective": "lead_gen"},
            run_id="test-run-10",
        )
        state["generated_ads"] = [
            {"ad_group": "Security", "headline": "Expert Audit", "description": "Protect your biz", "variant_id": "A"},
            {"ad_group": "Security", "headline": "Stop Breaches", "description": "Free assessment", "variant_id": "B"},
        ]
        state["target_audience"] = {"demographics": "IT managers"}
        result = asyncio.get_event_loop().run_until_complete(
            agent._node_deploy(state)
        )
        assert result["current_node"] == "deploy"
        assert result["campaigns_active"] >= 1
        assert result["campaigns_approved"] is True
        assert len(result["deployed_campaigns"]) >= 1
        assert result["deployed_campaigns"][0]["status"] == "deployed_shadow"
        assert result["knowledge_written"] is True
        # Verify insight was stored
        agent.db.store_insight.assert_called_once()

    def test_deploy_no_ads(self):
        """Deploy with no ads should return empty deployment."""
        agent = self._make_agent()
        state = agent._prepare_initial_state(
            task={"platform": "google"},
            run_id="test-run-11",
        )
        state["generated_ads"] = []
        result = asyncio.get_event_loop().run_until_complete(
            agent._node_deploy(state)
        )
        assert result["campaigns_active"] == 0
        assert result["deployed_campaigns"] == []

    def test_report_node(self):
        agent = self._make_agent()
        state = agent._prepare_initial_state(
            task={"platform": "google", "objective": "lead_gen", "budget_daily": 50.0},
            run_id="test-run-12",
        )
        state["ad_groups"] = [{"name": "Security", "keywords": []}]
        state["generated_ads"] = [{"variant_id": "A", "headline": "Test", "description": "Desc"}]
        state["deployed_campaigns"] = [{"campaign_name": "Test Campaign"}]
        state["campaigns_approved"] = True
        result = asyncio.get_event_loop().run_until_complete(
            agent._node_report(state)
        )
        assert result["current_node"] == "report"
        assert "Ads Strategy Report" in result["report_summary"]
        assert "$50.00" in result["report_summary"]
        assert result["report_generated_at"] != ""

    def test_route_after_review_approved(self):
        from core.agents.implementations.ads_agent import AdsStrategyAgent
        result = AdsStrategyAgent._route_after_review({"human_approval_status": "approved"})
        assert result == "approved"

    def test_route_after_review_rejected(self):
        from core.agents.implementations.ads_agent import AdsStrategyAgent
        result = AdsStrategyAgent._route_after_review({"human_approval_status": "rejected"})
        assert result == "rejected"

    def test_route_after_review_default(self):
        from core.agents.implementations.ads_agent import AdsStrategyAgent
        result = AdsStrategyAgent._route_after_review({})
        assert result == "approved"

    def test_system_prompt_default(self):
        agent = self._make_agent()
        prompt = agent._get_system_prompt()
        assert "advertising" in prompt.lower()
        assert "ROI" in prompt

    def test_repr(self):
        agent = self._make_agent()
        r = repr(agent)
        assert "AdsStrategyAgent" in r
        assert "ads_strategy_v1" in r

    def test_ad_copy_limits_enforcement(self):
        """Ad copy should be truncated to platform limits."""
        agent = self._make_agent()
        agent.llm.messages.create = MagicMock(return_value=MagicMock(
            content=[MagicMock(text=json.dumps([
                {
                    "headline": "X" * 100,  # Way over 30-char limit
                    "description": "Y" * 200,  # Way over 90-char limit
                    "call_to_action": "CTA",
                    "variant_id": "A",
                    "tone": "authority",
                },
            ]))]
        ))
        state = agent._prepare_initial_state(task={"platform": "google"}, run_id="test-limits")
        state["ad_groups"] = [{"name": "Test", "keywords": [{"keyword": "test"}]}]
        result = asyncio.get_event_loop().run_until_complete(
            agent._node_generate_campaigns(state)
        )
        for ad in result["generated_ads"]:
            assert len(ad["headline"]) <= 30
            assert len(ad["description"]) <= 90


# ══════════════════════════════════════════════════════════════════════
# Growth Dashboard Helper Tests
# ══════════════════════════════════════════════════════════════════════


class TestGrowthDashboardHelpers:
    """Tests for Growth Dashboard helper functions."""

    def test_compute_proposal_stats_empty(self):
        from dashboard.pages._growth_helpers import compute_proposal_stats
        stats = compute_proposal_stats([])
        assert stats["total"] == 0
        assert stats["win_rate"] == 0.0
        assert stats["avg_value"] == 0.0

    def test_compute_proposal_stats_with_data(self):
        from dashboard.pages._growth_helpers import compute_proposal_stats
        proposals = [
            {"status": "sent", "pricing_amount": 5000},
            {"status": "accepted", "pricing_amount": 15000},
            {"status": "rejected", "pricing_amount": 10000},
            {"status": "draft", "pricing_amount": 8000},
        ]
        stats = compute_proposal_stats(proposals)
        assert stats["total"] == 4
        assert stats["sent"] == 1
        assert stats["accepted"] == 1
        assert stats["rejected"] == 1
        assert stats["pending"] == 1
        assert stats["total_value"] == 38000
        assert stats["accepted_value"] == 15000
        assert stats["win_rate"] == 50.0  # 1 accepted / 2 closed

    def test_format_proposal_status(self):
        from dashboard.pages._growth_helpers import format_proposal_status
        text, color = format_proposal_status("accepted")
        assert text == "Accepted"
        assert color == "#10B981"
        text, color = format_proposal_status("unknown_status")
        assert text == "Unknown_Status"

    def test_compute_social_stats_empty(self):
        from dashboard.pages._growth_helpers import compute_social_stats
        stats = compute_social_stats([])
        assert stats["total"] == 0
        assert stats["engagement_rate"] == 0.0
        assert stats["top_platform"] == "—"

    def test_compute_social_stats_with_data(self):
        from dashboard.pages._growth_helpers import compute_social_stats
        posts = [
            {"status": "published", "platform": "twitter", "impressions": 1000, "likes": 50, "shares": 10, "comments": 5, "clicks": 20},
            {"status": "published", "platform": "twitter", "impressions": 2000, "likes": 100, "shares": 30, "comments": 10, "clicks": 40},
            {"status": "draft", "platform": "linkedin", "impressions": 0, "likes": 0, "shares": 0, "comments": 0, "clicks": 0},
        ]
        stats = compute_social_stats(posts)
        assert stats["total"] == 3
        assert stats["published"] == 2
        assert stats["draft"] == 1
        assert stats["total_impressions"] == 3000
        assert stats["total_engagements"] == 265  # 150+40+15+60
        assert stats["top_platform"] == "twitter"
        assert stats["engagement_rate"] > 0

    def test_format_platform_icon(self):
        from dashboard.pages._growth_helpers import format_platform_icon
        assert format_platform_icon("twitter") == "🐦"
        assert format_platform_icon("linkedin") == "💼"
        assert format_platform_icon("google") == "🔍"
        assert format_platform_icon("unknown") == "📱"

    def test_compute_ads_stats_empty(self):
        from dashboard.pages._growth_helpers import compute_ads_stats
        stats = compute_ads_stats([])
        assert stats["total"] == 0
        assert stats["avg_ctr"] == 0.0

    def test_compute_ads_stats_with_data(self):
        from dashboard.pages._growth_helpers import compute_ads_stats
        campaigns = [
            {"status": "active", "total_spend": 500, "impressions": 10000, "clicks": 300, "conversions": 10},
            {"status": "paused", "total_spend": 200, "impressions": 5000, "clicks": 100, "conversions": 3},
        ]
        stats = compute_ads_stats(campaigns)
        assert stats["total"] == 2
        assert stats["active"] == 1
        assert stats["paused"] == 1
        assert stats["total_spend"] == 700
        assert stats["total_clicks"] == 400
        assert stats["total_conversions"] == 13
        assert stats["avg_ctr"] > 0

    def test_compute_campaign_health(self):
        from dashboard.pages._growth_helpers import compute_campaign_health
        # No impressions
        assert compute_campaign_health({"impressions": 0}) == "needs_attention"
        # CPA way over target
        assert compute_campaign_health({"impressions": 1000, "cpa": 50, "target_cpa": 20, "ctr": 3.0}) == "critical"
        # Low CTR
        assert compute_campaign_health({"impressions": 1000, "cpa": 10, "target_cpa": 25, "ctr": 0.5}) == "needs_attention"
        # Healthy
        assert compute_campaign_health({"impressions": 1000, "cpa": 10, "target_cpa": 25, "ctr": 3.0}) == "healthy"

    def test_group_calendar_by_date(self):
        from dashboard.pages._growth_helpers import group_calendar_by_date
        entries = [
            {"scheduled_date": "2025-02-10", "topic": "Post A"},
            {"scheduled_date": "2025-02-10", "topic": "Post B"},
            {"scheduled_date": "2025-02-11", "topic": "Post C"},
        ]
        grouped = group_calendar_by_date(entries)
        assert len(grouped) == 2
        assert len(grouped["2025-02-10"]) == 2
        assert len(grouped["2025-02-11"]) == 1

    def test_compute_growth_score_zero(self):
        from dashboard.pages._growth_helpers import compute_growth_score
        p = {"total": 0, "win_rate": 0, "published": 0, "engagement_rate": 0, "total_conversions": 0}
        s = {"total": 0, "published": 0, "engagement_rate": 0}
        a = {"total": 0, "avg_ctr": 0, "total_conversions": 0}
        score = compute_growth_score(p, s, a)
        assert score == 0

    def test_compute_growth_score_with_data(self):
        from dashboard.pages._growth_helpers import compute_growth_score
        p = {"total": 10, "win_rate": 60.0}
        s = {"total": 20, "published": 15, "engagement_rate": 5.0}
        a = {"total": 5, "avg_ctr": 3.0, "total_conversions": 20}
        score = compute_growth_score(p, s, a)
        assert 0 < score <= 100

    def test_growth_dashboard_page_exists(self):
        path = Path(__file__).parent.parent / "dashboard" / "pages" / "4_Growth.py"
        assert path.exists()


class TestAdsStrategyAgentRegistry:
    """Verify ads_strategy is in the agent registry."""

    def test_ads_strategy_in_registry(self):
        from core.agents.registry import AGENT_IMPLEMENTATIONS
        # Force import to trigger registration
        import core.agents.implementations.ads_agent  # noqa: F401
        assert "ads_strategy" in AGENT_IMPLEMENTATIONS


class TestAdsStrategyYAML:
    """Verify the ads_strategy YAML config is valid."""

    def test_ads_strategy_yaml_loads(self):
        import yaml
        from core.config.agent_schema import AgentInstanceConfig

        path = Path(__file__).parent.parent / "verticals" / "enclave_guard" / "agents" / "ads_strategy.yaml"
        assert path.exists(), f"YAML not found: {path}"

        with open(path) as f:
            raw = yaml.safe_load(f)

        raw.setdefault("vertical_id", "enclave_guard")
        config = AgentInstanceConfig(**raw)
        assert config.agent_type == "ads_strategy"
        assert config.agent_id == "ads_strategy_v1"
        assert config.human_gates.enabled is True
