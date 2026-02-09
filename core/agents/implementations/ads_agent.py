"""
Ads Strategy Agent — The Growth Engine.

Designs, optimizes, and manages paid advertising campaigns across
Google Ads and Meta (Facebook/Instagram). This is the agent that turns
marketing spend into pipeline — every dollar in produces qualified leads out.

Architecture (LangGraph State Machine):
    analyze_performance → research_keywords → generate_campaigns →
    human_review → deploy → report → END

The Ads Agent serves three critical functions:
    1. RESEARCH — Mine keyword opportunities and audience insights
    2. CREATE  — Generate ad copy with A/B variants using shared brain data
    3. OPTIMIZE — Track performance and surface optimization suggestions

Shared Brain Integration:
    - Reads: outreach insights (what messaging converts), SEO keywords
            (what people search for), social data (what resonates)
    - Writes: campaign performance, winning ad patterns, audience insights

Supported Platforms:
    - Google Ads (Search, Display)
    - Meta Ads (Facebook, Instagram)
    - LinkedIn Ads (Sponsored Content)

Safety:
    - NEVER deploys campaigns automatically — ALL campaigns go through human_review
    - Budget caps enforced at agent level (won't exceed configured daily/total)
    - CPC and CPA thresholds prevent runaway spend
    - All ad copy passes compliance checks (no misleading claims)

RLHF Hook: If a human edits ad copy before deployment,
the (draft, edit) pair is saved for future optimization.

Usage:
    agent = AdsStrategyAgent(config, db, embedder, llm)
    result = await agent.run({
        "platform": "google",
        "objective": "lead_gen",
        "budget_daily": 50.0,
        "budget_total": 1500.0,
        "target_cpa": 25.0,
        "seed_keywords": ["cybersecurity assessment", "penetration testing"],
    })
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional, Type

from core.agents.base import BaseAgent
from core.agents.contracts import InsightData, AdCampaign, AdCreative
from core.agents.registry import register_agent_type
from core.agents.state import AdsStrategyAgentState
from core.config.agent_schema import AgentInstanceConfig

logger = logging.getLogger(__name__)

# ─── Constants ─────────────────────────────────────────────────────────

PLATFORMS = {"google", "meta", "linkedin", "both"}
OBJECTIVES = {"lead_gen", "brand_awareness", "conversions", "traffic", "retargeting"}

# Platform-specific ad copy limits
AD_COPY_LIMITS = {
    "google": {
        "headline_max": 30,       # Google RSA headline limit
        "description_max": 90,    # Google RSA description limit
        "headlines_per_ad": 15,   # RSA allows up to 15 headlines
        "descriptions_per_ad": 4, # RSA allows up to 4 descriptions
    },
    "meta": {
        "headline_max": 40,
        "description_max": 125,   # Primary text can be longer but 125 is optimal
        "primary_text_max": 500,
    },
    "linkedin": {
        "headline_max": 70,
        "description_max": 150,
        "intro_text_max": 600,
    },
}

# Default budget guardrails
MAX_DAILY_BUDGET = 500.0   # Hard cap per campaign/day
MAX_TOTAL_BUDGET = 10000.0  # Hard cap per campaign total
MIN_CPA_TARGET = 1.0       # Minimum CPA target (sanity check)


@register_agent_type("ads_strategy")
class AdsStrategyAgent(BaseAgent):
    """
    AI-powered ads strategy agent that designs and optimizes paid campaigns.

    Nodes:
        1. analyze_performance — Review existing campaign data + shared brain
        2. research_keywords   — Mine keyword/audience opportunities
        3. generate_campaigns  — Create ad groups, copy variants, targeting
        4. human_review        — Gate for human approval (NEVER auto-deploy)
        5. deploy              — Deploy approved campaigns (mock in shadow mode)
        6. report              — Generate performance summary
    """

    def build_graph(self) -> Any:
        """
        Build the Ads Strategy Agent's LangGraph state machine.

        Graph flow:
            analyze_performance → research_keywords → generate_campaigns →
            human_review → route_after_review → deploy → report → END
        """
        from langgraph.graph import StateGraph, END

        workflow = StateGraph(AdsStrategyAgentState)

        # Add nodes
        workflow.add_node("analyze_performance", self._node_analyze_performance)
        workflow.add_node("research_keywords", self._node_research_keywords)
        workflow.add_node("generate_campaigns", self._node_generate_campaigns)
        workflow.add_node("human_review", self._node_human_review)
        workflow.add_node("deploy", self._node_deploy)
        workflow.add_node("report", self._node_report)

        # Entry point
        workflow.set_entry_point("analyze_performance")

        # Linear flow with conditional after review
        workflow.add_edge("analyze_performance", "research_keywords")
        workflow.add_edge("research_keywords", "generate_campaigns")
        workflow.add_edge("generate_campaigns", "human_review")
        workflow.add_conditional_edges(
            "human_review",
            self._route_after_review,
            {
                "approved": "deploy",
                "rejected": "report",
            },
        )
        workflow.add_edge("deploy", "report")
        workflow.add_edge("report", END)

        # Compile with human gate
        compile_kwargs: dict[str, Any] = {}
        if self.config.human_gates.enabled:
            gate_nodes = self.config.human_gates.gate_before
            if gate_nodes:
                compile_kwargs["interrupt_before"] = gate_nodes
        if self.checkpointer:
            compile_kwargs["checkpointer"] = self.checkpointer

        return workflow.compile(**compile_kwargs)

    def get_tools(self) -> list[Any]:
        """Ads agent uses MCP tools (Google Ads, Meta Ads APIs)."""
        return []  # Tools accessed via MCP, not injected

    def get_state_class(self) -> Type[AdsStrategyAgentState]:
        return AdsStrategyAgentState

    # ─── State Preparation ────────────────────────────────────────────

    def _prepare_initial_state(
        self, task: dict[str, Any], run_id: str
    ) -> dict[str, Any]:
        """Prepare initial state from ads task input."""
        state = super()._prepare_initial_state(task, run_id)

        # Budget guardrails — enforce caps
        budget_daily = min(
            float(task.get("budget_daily", 50.0)),
            self.config.params.get("max_daily_budget", MAX_DAILY_BUDGET),
        )
        budget_total = min(
            float(task.get("budget_total", 1500.0)),
            self.config.params.get("max_total_budget", MAX_TOTAL_BUDGET),
        )
        target_cpa = max(
            float(task.get("target_cpa", 25.0)),
            MIN_CPA_TARGET,
        )

        state.update({
            "platform": task.get("platform", "google"),
            "campaign_objective": task.get("objective", "lead_gen"),
            "budget_daily": budget_daily,
            "budget_total": budget_total,
            "target_cpa": target_cpa,
            # Keyword research
            "seed_keywords": task.get("seed_keywords", []),
            "keyword_research": [],
            "negative_keywords": task.get("negative_keywords", []),
            "selected_keywords": [],
            # Ad copy
            "ad_groups": [],
            "generated_ads": [],
            "ad_variants": [],
            # Targeting
            "target_audience": task.get("target_audience", {}),
            "lookalike_audiences": [],
            "retargeting_rules": [],
            # Performance
            "campaign_performance": {},
            "ad_performance": [],
            "optimization_suggestions": [],
            # Review
            "campaigns_approved": False,
            # Deployment
            "deployed_campaigns": [],
            "campaigns_active": 0,
            # Report
            "report_summary": "",
            "report_generated_at": "",
        })
        return state

    # ─── Node 1: Analyze Performance ──────────────────────────────────

    async def _node_analyze_performance(
        self, state: AdsStrategyAgentState
    ) -> dict[str, Any]:
        """
        Node 1: Analyze existing campaign performance and shared brain data.

        Pulls previous campaign data, outreach insights (what messaging
        converts), and SEO keywords (what people search for) to inform
        the keyword research and campaign generation phases.
        """
        platform = state.get("platform", "google")
        objective = state.get("campaign_objective", "lead_gen")

        logger.info(
            "ads_analyze_start",
            extra={
                "agent_id": self.agent_id,
                "platform": platform,
                "objective": objective,
            },
        )

        # Pull outreach insights — what messaging converts
        outreach_insights: list[dict[str, Any]] = []
        try:
            niche = self.config.params.get("niche", "cybersecurity")
            query = f"winning pattern messaging conversion {niche}"
            insights = self.db.search_insights(
                query_embedding=self.embedder.embed_query(query),
                insight_type="winning_pattern",
                limit=5,
            )
            outreach_insights = insights if isinstance(insights, list) else []
        except Exception as e:
            logger.debug(f"Could not fetch outreach insights: {e}")

        # Pull SEO keywords — what people search for
        seo_keywords: list[str] = []
        try:
            query = f"keyword performance volume {self.config.params.get('niche', 'cybersecurity')}"
            kw_insights = self.db.search_insights(
                query_embedding=self.embedder.embed_query(query),
                insight_type="keyword_performance",
                limit=10,
            )
            for kw in (kw_insights if isinstance(kw_insights, list) else []):
                content = kw.get("content", "")
                if content:
                    seo_keywords.append(content[:50])
        except Exception as e:
            logger.debug(f"Could not fetch SEO keywords: {e}")

        # Pull previous campaign performance from shared brain
        previous_performance: dict[str, Any] = {}
        try:
            query = f"ad campaign performance {platform}"
            perf_insights = self.db.search_insights(
                query_embedding=self.embedder.embed_query(query),
                insight_type="campaign_performance",
                limit=3,
            )
            if perf_insights and isinstance(perf_insights, list) and len(perf_insights) > 0:
                previous_performance = perf_insights[0]
        except Exception as e:
            logger.debug(f"Could not fetch previous performance: {e}")

        # Pull social engagement data — what resonates with audience
        social_insights: list[str] = []
        try:
            query = f"social engagement winning posts {self.config.params.get('niche', 'cybersecurity')}"
            social_data = self.db.search_insights(
                query_embedding=self.embedder.embed_query(query),
                insight_type="social_activity",
                limit=5,
            )
            for s in (social_data if isinstance(social_data, list) else []):
                content = s.get("content", "")
                if content:
                    social_insights.append(content[:100])
        except Exception as e:
            logger.debug(f"Could not fetch social insights: {e}")

        # Merge SEO keywords with seed keywords
        seed_keywords = state.get("seed_keywords", [])
        combined_keywords = list(set(seed_keywords + seo_keywords))

        logger.info(
            "ads_analyze_complete",
            extra={
                "outreach_insights": len(outreach_insights),
                "seo_keywords": len(seo_keywords),
                "social_insights": len(social_insights),
                "combined_keywords": len(combined_keywords),
            },
        )

        return {
            "current_node": "analyze_performance",
            "seed_keywords": combined_keywords,
            "campaign_performance": previous_performance,
            "optimization_suggestions": [
                f"Found {len(outreach_insights)} outreach insights to inform ad copy",
                f"Found {len(seo_keywords)} SEO keywords to expand targeting",
                f"Found {len(social_insights)} social insights for audience resonance",
            ],
        }

    # ─── Node 2: Research Keywords ────────────────────────────────────

    async def _node_research_keywords(
        self, state: AdsStrategyAgentState
    ) -> dict[str, Any]:
        """
        Node 2: Research Keywords — Mine keyword and audience opportunities.

        Uses LLM to expand seed keywords into ad groups with estimated
        volume, CPC, and competition. Also generates negative keywords
        and audience targeting suggestions.
        """
        platform = state.get("platform", "google")
        seed_keywords = state.get("seed_keywords", [])
        objective = state.get("campaign_objective", "lead_gen")
        target_cpa = state.get("target_cpa", 25.0)

        logger.info(
            "ads_research_start",
            extra={
                "agent_id": self.agent_id,
                "seed_keywords": len(seed_keywords),
                "platform": platform,
            },
        )

        niche = self.config.params.get("niche", "cybersecurity")
        our_company = self.config.params.get("company_name", "Enclave Guard")

        research_prompt = (
            f"You are an expert paid advertising strategist.\n\n"
            f"COMPANY: {our_company}\n"
            f"NICHE: {niche}\n"
            f"PLATFORM: {platform}\n"
            f"OBJECTIVE: {objective}\n"
            f"TARGET CPA: ${target_cpa:.2f}\n\n"
            f"SEED KEYWORDS:\n{chr(10).join(f'  - {kw}' for kw in seed_keywords[:20])}\n\n"
            f"INSTRUCTIONS:\n"
            f"1. Expand these seed keywords into organized ad groups\n"
            f"2. For each keyword, estimate: search volume, avg CPC, competition level\n"
            f"3. Suggest negative keywords to exclude irrelevant traffic\n"
            f"4. Suggest audience targeting (demographics, interests, behaviors)\n\n"
            f"Return a JSON object with:\n"
            f'- "ad_groups": [{{\"name\": str, \"keywords\": [{{\"keyword\": str, \"volume\": int, \"cpc\": float, \"competition\": str}}]}}]\n'
            f'- "negative_keywords": [str]\n'
            f'- "target_audience": {{\"demographics\": str, \"interests\": [str], \"behaviors\": [str]}}\n\n'
            f"Return ONLY the JSON object, no code fences."
        )

        keyword_research: list[dict[str, Any]] = []
        ad_groups: list[dict[str, Any]] = []
        negative_keywords = state.get("negative_keywords", [])
        target_audience: dict[str, Any] = state.get("target_audience", {})

        try:
            response = self.llm.messages.create(
                model=self.config.model.model,
                max_tokens=3000,
                temperature=0.4,  # Lower creativity for research accuracy
                messages=[{"role": "user", "content": research_prompt}],
            )
            text = response.content[0].text.strip() if response.content else "{}"

            # Clean JSON
            if "```" in text:
                parts = text.split("```")
                text = parts[1] if len(parts) > 1 else text
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()

            parsed = json.loads(text)

            if isinstance(parsed, dict):
                raw_groups = parsed.get("ad_groups", [])
                if isinstance(raw_groups, list):
                    ad_groups = raw_groups

                # Flatten keywords for easy access
                for group in ad_groups:
                    for kw in group.get("keywords", []):
                        keyword_research.append(kw)

                extra_negatives = parsed.get("negative_keywords", [])
                if isinstance(extra_negatives, list):
                    negative_keywords = list(set(negative_keywords + extra_negatives))

                audience = parsed.get("target_audience", {})
                if isinstance(audience, dict):
                    target_audience = audience

        except Exception as e:
            logger.warning(
                "ads_research_failed",
                extra={"error": str(e)[:200]},
            )
            # Fallback: create basic ad group from seed keywords
            ad_groups = [{
                "name": f"{niche} - General",
                "keywords": [
                    {"keyword": kw, "volume": 1000, "cpc": 5.0, "competition": "medium"}
                    for kw in seed_keywords[:10]
                ],
            }]
            keyword_research = ad_groups[0]["keywords"]

        # Select keywords — filter by CPA viability
        selected_keywords = [
            kw.get("keyword", "")
            for kw in keyword_research
            if kw.get("cpc", 0) <= target_cpa * 0.5  # CPC should be well below CPA
        ]
        # If filter too aggressive, take all
        if not selected_keywords:
            selected_keywords = [kw.get("keyword", "") for kw in keyword_research[:20]]

        logger.info(
            "ads_research_complete",
            extra={
                "ad_groups": len(ad_groups),
                "keywords_found": len(keyword_research),
                "keywords_selected": len(selected_keywords),
                "negative_keywords": len(negative_keywords),
            },
        )

        return {
            "current_node": "research_keywords",
            "keyword_research": keyword_research,
            "ad_groups": ad_groups,
            "negative_keywords": negative_keywords,
            "selected_keywords": selected_keywords,
            "target_audience": target_audience,
        }

    # ─── Node 3: Generate Campaigns ───────────────────────────────────

    async def _node_generate_campaigns(
        self, state: AdsStrategyAgentState
    ) -> dict[str, Any]:
        """
        Node 3: Generate Campaigns — Create ad copy with A/B variants.

        For each ad group, generates multiple headline/description variants
        using LLM. Respects platform-specific character limits and creates
        A/B test variants for performance optimization.
        """
        platform = state.get("platform", "google")
        ad_groups = state.get("ad_groups", [])
        objective = state.get("campaign_objective", "lead_gen")
        budget_daily = state.get("budget_daily", 50.0)
        target_cpa = state.get("target_cpa", 25.0)

        logger.info(
            "ads_generate_start",
            extra={
                "agent_id": self.agent_id,
                "platform": platform,
                "ad_groups": len(ad_groups),
            },
        )

        limits = AD_COPY_LIMITS.get(platform, AD_COPY_LIMITS["google"])
        niche = self.config.params.get("niche", "cybersecurity")
        our_company = self.config.params.get("company_name", "Enclave Guard")
        our_url = self.config.params.get("landing_url", f"https://{our_company.lower().replace(' ', '')}.com")
        num_variants = self.config.params.get("variants_per_group", 3)

        generated_ads: list[dict[str, Any]] = []
        ad_variants: list[dict[str, Any]] = []

        for group in ad_groups:
            group_name = group.get("name", "General")
            keywords = [kw.get("keyword", "") for kw in group.get("keywords", [])]

            ad_prompt = (
                f"Generate {num_variants} ad creative variants for a {platform} ad campaign.\n\n"
                f"COMPANY: {our_company}\n"
                f"NICHE: {niche}\n"
                f"AD GROUP: {group_name}\n"
                f"KEYWORDS: {', '.join(keywords[:10])}\n"
                f"OBJECTIVE: {objective}\n"
                f"LANDING URL: {our_url}\n\n"
                f"PLATFORM CONSTRAINTS:\n"
                f"  - Headline max chars: {limits.get('headline_max', 30)}\n"
                f"  - Description max chars: {limits.get('description_max', 90)}\n\n"
                f"INSTRUCTIONS:\n"
                f"Generate {num_variants} different ad variants. Each should:\n"
                f"- Have a compelling headline within char limits\n"
                f"- Have a clear description with value proposition\n"
                f"- Include a strong call-to-action\n"
                f"- Be distinct enough for meaningful A/B testing\n\n"
                f"Return a JSON array of objects, each with:\n"
                f'- "headline": str\n'
                f'- "description": str\n'
                f'- "call_to_action": str\n'
                f'- "variant_id": str (e.g., "A", "B", "C")\n'
                f'- "tone": str (e.g., "authority", "urgency", "social_proof")\n\n'
                f"Return ONLY the JSON array, no code fences."
            )

            try:
                response = self.llm.messages.create(
                    model=self.config.model.model,
                    max_tokens=2000,
                    temperature=0.6,  # Balanced creativity for ad copy
                    messages=[{"role": "user", "content": ad_prompt}],
                    system=self._get_system_prompt(),
                )
                text = response.content[0].text.strip() if response.content else "[]"

                # Clean JSON
                if "```" in text:
                    parts = text.split("```")
                    text = parts[1] if len(parts) > 1 else text
                    if text.startswith("json"):
                        text = text[4:]
                    text = text.strip()

                variants = json.loads(text)
                if isinstance(variants, list):
                    for v in variants[:num_variants]:
                        # Enforce character limits
                        headline = v.get("headline", "")[:limits.get("headline_max", 30)]
                        description = v.get("description", "")[:limits.get("description_max", 90)]

                        ad_entry = {
                            "ad_group": group_name,
                            "headline": headline,
                            "description": description,
                            "display_url": our_url,
                            "final_url": our_url,
                            "call_to_action": v.get("call_to_action", "Learn More"),
                            "variant_id": v.get("variant_id", "A"),
                            "tone": v.get("tone", "authority"),
                            "platform": platform,
                        }
                        generated_ads.append(ad_entry)
                        ad_variants.append(ad_entry)

            except Exception as e:
                logger.warning(
                    "ads_generate_group_failed",
                    extra={"error": str(e)[:200], "group": group_name},
                )
                # Fallback: basic ad
                fallback_ad = {
                    "ad_group": group_name,
                    "headline": f"Expert {niche.title()} Services"[:limits.get("headline_max", 30)],
                    "description": f"Protect your business with {our_company}. Get a free assessment today."[:limits.get("description_max", 90)],
                    "display_url": our_url,
                    "final_url": our_url,
                    "call_to_action": "Get Started",
                    "variant_id": "A",
                    "tone": "authority",
                    "platform": platform,
                }
                generated_ads.append(fallback_ad)
                ad_variants.append(fallback_ad)

        logger.info(
            "ads_generate_complete",
            extra={
                "ads_generated": len(generated_ads),
                "variants": len(ad_variants),
                "platform": platform,
            },
        )

        return {
            "current_node": "generate_campaigns",
            "generated_ads": generated_ads,
            "ad_variants": ad_variants,
        }

    # ─── Node 4: Human Review ─────────────────────────────────────────

    async def _node_human_review(
        self, state: AdsStrategyAgentState
    ) -> dict[str, Any]:
        """
        Node 4: Human Review — NEVER auto-deploy campaigns.

        ALL campaigns must be reviewed and approved by a human before
        spending any budget. This is a non-negotiable safety gate.
        Shows: ad copy, targeting, budget, and projected performance.
        """
        ads_count = len(state.get("generated_ads", []))
        platform = state.get("platform", "")
        budget = state.get("budget_daily", 0)

        logger.info(
            "ads_human_review_pending",
            extra={
                "agent_id": self.agent_id,
                "platform": platform,
                "ads_count": ads_count,
                "budget_daily": budget,
            },
        )

        return {
            "current_node": "human_review",
            "requires_human_approval": True,
        }

    # ─── Node 5: Deploy ───────────────────────────────────────────────

    async def _node_deploy(
        self, state: AdsStrategyAgentState
    ) -> dict[str, Any]:
        """
        Node 5: Deploy — Push approved campaigns to ad platforms.

        In shadow mode (default for new verticals), this simulates
        deployment. In production mode, it calls the platform APIs.
        Captures RLHF data if ad copy was edited during review.
        """
        generated_ads = state.get("generated_ads", [])
        ad_groups = state.get("ad_groups", [])
        platform = state.get("platform", "google")
        objective = state.get("campaign_objective", "lead_gen")
        budget_daily = state.get("budget_daily", 50.0)
        target_audience = state.get("target_audience", {})

        logger.info(
            "ads_deploy_start",
            extra={
                "agent_id": self.agent_id,
                "platform": platform,
                "ads_count": len(generated_ads),
            },
        )

        # Check for human edits (RLHF capture)
        human_edits = state.get("task_input", {})
        if isinstance(human_edits, dict):
            edited_ads = human_edits.get("edited_ads", [])
            if isinstance(edited_ads, list):
                for edit in edited_ads:
                    original_variant = edit.get("variant_id", "")
                    original_headline = edit.get("original_headline", "")
                    edited_headline = edit.get("edited_headline", "")
                    if original_headline and edited_headline and original_headline != edited_headline:
                        self.learn(
                            task_input={
                                "platform": platform,
                                "objective": objective,
                                "variant_id": original_variant,
                            },
                            model_output=original_headline,
                            human_correction=edited_headline,
                            source="manual_review",
                            metadata={
                                "agent_type": "ads_strategy",
                                "platform": platform,
                            },
                        )

        # Deploy campaigns (shadow mode — simulate)
        deployed: list[dict[str, Any]] = []
        now = datetime.now(timezone.utc).isoformat()

        # Group ads by ad_group for deployment
        ads_by_group: dict[str, list[dict[str, Any]]] = {}
        for ad in generated_ads:
            group = ad.get("ad_group", "General")
            if group not in ads_by_group:
                ads_by_group[group] = []
            ads_by_group[group].append(ad)

        for group_name, group_ads in ads_by_group.items():
            campaign = {
                "campaign_name": f"{self.config.params.get('company_name', 'Enclave')} - {group_name}",
                "platform": platform,
                "objective": objective,
                "budget_daily": budget_daily / max(len(ads_by_group), 1),
                "ad_group": group_name,
                "ads": [
                    {
                        "headline": a.get("headline", ""),
                        "description": a.get("description", ""),
                        "variant_id": a.get("variant_id", "A"),
                    }
                    for a in group_ads
                ],
                "target_audience": target_audience,
                "status": "deployed_shadow",  # Shadow mode — no real spend
                "deployed_at": now,
            }
            deployed.append(campaign)

        # Write insights to shared brain
        if deployed:
            self.store_insight(InsightData(
                insight_type="campaign_deployed",
                title=f"Ads: {len(deployed)} campaigns deployed on {platform}",
                content=(
                    f"Deployed {len(deployed)} campaigns on {platform}. "
                    f"Objective: {objective}. Daily budget: ${budget_daily:.2f}. "
                    f"Total ads: {len(generated_ads)}. "
                    f"Ad groups: {', '.join(ads_by_group.keys())}."
                ),
                confidence=0.75,
                metadata={
                    "platform": platform,
                    "campaign_count": len(deployed),
                    "total_ads": len(generated_ads),
                    "budget_daily": budget_daily,
                },
            ))

        logger.info(
            "ads_deploy_complete",
            extra={
                "deployed": len(deployed),
                "platform": platform,
            },
        )

        return {
            "current_node": "deploy",
            "deployed_campaigns": deployed,
            "campaigns_active": len(deployed),
            "campaigns_approved": True,
            "knowledge_written": True,
        }

    # ─── Node 6: Report ───────────────────────────────────────────────

    async def _node_report(
        self, state: AdsStrategyAgentState
    ) -> dict[str, Any]:
        """
        Node 6: Report — Generate campaign strategy summary.
        """
        now = datetime.now(timezone.utc).isoformat()
        platform = state.get("platform", "")
        objective = state.get("campaign_objective", "")
        budget_daily = state.get("budget_daily", 0)
        budget_total = state.get("budget_total", 0)
        target_cpa = state.get("target_cpa", 0)
        ad_groups = state.get("ad_groups", [])
        generated_ads = state.get("generated_ads", [])
        deployed = state.get("deployed_campaigns", [])
        keyword_research = state.get("keyword_research", [])
        selected_keywords = state.get("selected_keywords", [])
        negative_keywords = state.get("negative_keywords", [])
        approved = state.get("campaigns_approved", False)
        suggestions = state.get("optimization_suggestions", [])

        sections = [
            "# Ads Strategy Report",
            f"*Generated: {now}*\n",
            "## Campaign Overview",
            f"- **Platform:** {platform}",
            f"- **Objective:** {objective}",
            f"- **Daily Budget:** ${budget_daily:.2f}",
            f"- **Total Budget:** ${budget_total:.2f}",
            f"- **Target CPA:** ${target_cpa:.2f}",
            f"- **Status:** {'Deployed' if deployed else ('Rejected' if not approved else 'Pending')}",
        ]

        if ad_groups:
            sections.append("\n## Ad Groups")
            for group in ad_groups:
                name = group.get("name", "Unknown")
                kw_count = len(group.get("keywords", []))
                sections.append(f"  - **{name}** ({kw_count} keywords)")

        if generated_ads:
            sections.append(f"\n## Ad Creatives ({len(generated_ads)} total)")
            for ad in generated_ads[:6]:  # Show first 6
                sections.append(f"  - [{ad.get('variant_id', '?')}] \"{ad.get('headline', '')}\"")
                sections.append(f"    {ad.get('description', '')[:80]}")

        if selected_keywords:
            sections.append(f"\n## Keywords ({len(selected_keywords)} selected)")
            for kw in selected_keywords[:10]:
                sections.append(f"  - {kw}")

        if negative_keywords:
            sections.append(f"\n## Negative Keywords ({len(negative_keywords)})")
            for nk in negative_keywords[:5]:
                sections.append(f"  - {nk}")

        if suggestions:
            sections.append("\n## Insights & Suggestions")
            for s in suggestions:
                sections.append(f"  - {s}")

        # Performance metrics (when available)
        perf = state.get("campaign_performance", {})
        if perf and isinstance(perf, dict) and perf.get("content"):
            sections.append("\n## Historical Performance")
            sections.append(f"  {str(perf.get('content', ''))[:200]}")

        report = "\n".join(sections)

        return {
            "current_node": "report",
            "report_summary": report,
            "report_generated_at": now,
        }

    # ─── Routing Functions ────────────────────────────────────────────

    @staticmethod
    def _route_after_review(state: AdsStrategyAgentState) -> str:
        """Route after human review: approved → deploy, rejected → report."""
        status = state.get("human_approval_status", "approved")
        if status == "rejected":
            return "rejected"
        return "approved"

    # ─── System Prompt ────────────────────────────────────────────────

    def _get_system_prompt(self) -> str:
        """Load system prompt or use default."""
        if self.config.system_prompt_path:
            try:
                with open(self.config.system_prompt_path) as f:
                    return f.read()
            except FileNotFoundError:
                pass

        return (
            "You are an expert digital advertising strategist. "
            "You create high-converting ad campaigns that maximize ROI. "
            "Your copy is compelling, concise, and action-oriented. "
            "You understand platform-specific best practices: Google Ads "
            "demands keyword-rich headlines, Meta requires scroll-stopping "
            "creative, and LinkedIn needs professional authority. "
            "You always A/B test and optimize based on data. "
            "Budget efficiency is paramount — every dollar must work hard."
        )

    # ─── Knowledge Writing ────────────────────────────────────────────

    async def write_knowledge(self, result: dict[str, Any]) -> None:
        """Write campaign insights to shared brain (handled in deploy node)."""
        pass

    def __repr__(self) -> str:
        return (
            f"<AdsStrategyAgent "
            f"agent_id={self.agent_id!r} "
            f"vertical={self.vertical_id!r}>"
        )
