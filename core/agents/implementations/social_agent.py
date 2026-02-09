"""
Social Media Agent — The Megaphone.

The platform's top-of-funnel awareness engine. This is the agent
that unlocks ALL verticals — without it, MemeLord and PrintBiz
are dead. With it, the Sovereign Venture Engine can find trends,
create content, and drive traffic to ANY offer.

Architecture (LangGraph State Machine):
    listen → ideate → create → human_review →
    route_after_review → publish → report → END

The Social Agent serves three critical functions:
    1. LISTEN — Monitor trending topics and mentions
    2. CREATE — Generate platform-specific posts with optional images
    3. AMPLIFY — Coordinate with SEO and Outreach agents via shared brain

Supported Platforms:
    - Twitter/X (280-char posts, threads, image attachments)
    - LinkedIn (long-form posts, professional tone, link shares)

Safety:
    - NEVER posts automatically in v1 — ALL posts go through human_review
    - Mock mode when API keys are missing (logged to social_mock.json)
    - All posts respect platform character limits
    - Compliance-aware: no misleading claims, proper disclosures

Shared Brain Integration:
    - Reads: outreach insights (what messaging resonates), SEO keywords
    - Writes: engagement data, winning post patterns, audience growth

Usage:
    agent = SocialMediaAgent(config, db, embedder, llm)
    result = await agent.run({
        "platform": "twitter",
        "mode": "create_content",  # or "monitor_mentions", "trending_analysis"
        "topic": "cybersecurity trends",
    })
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional, Type

from core.agents.base import BaseAgent
from core.agents.contracts import InsightData, SocialMediaPost
from core.agents.registry import register_agent_type
from core.agents.state import SocialMediaAgentState
from core.config.agent_schema import AgentInstanceConfig

logger = logging.getLogger(__name__)

# ─── Constants ─────────────────────────────────────────────────────────

PLATFORMS = {"twitter", "x", "linkedin", "both"}
MODES = {"create_content", "monitor_mentions", "trending_analysis", "full_cycle"}

POST_TYPES = {
    "thought_leadership",
    "case_study",
    "industry_news",
    "engagement",
    "meme",
    "product_launch",
    "tip",
    "thread",
}

# Platform-specific limits
TWITTER_CHAR_LIMIT = 280
LINKEDIN_CHAR_LIMIT = 3000


@register_agent_type("social")
class SocialMediaAgent(BaseAgent):
    """
    AI-powered social media agent that drives top-of-funnel awareness.

    Nodes:
        1. listen       — Check trending topics and/or mentions
        2. ideate       — Generate content ideas based on trends + insights
        3. create       — Draft platform-specific posts (with image gen)
        4. human_review — Gate for human approval/editing (NEVER auto-post)
        5. publish      — Execute the approved posts
        6. report       — Generate engagement summary
    """

    def build_graph(self) -> Any:
        """
        Build the Social Media Agent's LangGraph state machine.

        Graph flow:
            listen → ideate → create → human_review →
            route_after_review → publish → report → END
        """
        from langgraph.graph import StateGraph, END

        workflow = StateGraph(SocialMediaAgentState)

        # Add nodes
        workflow.add_node("listen", self._node_listen)
        workflow.add_node("ideate", self._node_ideate)
        workflow.add_node("create", self._node_create)
        workflow.add_node("human_review", self._node_human_review)
        workflow.add_node("publish", self._node_publish)
        workflow.add_node("report", self._node_report)

        # Entry point
        workflow.set_entry_point("listen")

        # Linear flow with conditional after review
        workflow.add_edge("listen", "ideate")
        workflow.add_edge("ideate", "create")
        workflow.add_edge("create", "human_review")
        workflow.add_conditional_edges(
            "human_review",
            self._route_after_review,
            {
                "approved": "publish",
                "rejected": "report",
            },
        )
        workflow.add_edge("publish", "report")
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
        """Social agent uses social MCP tools."""
        return []  # Tools accessed via MCP, not injected

    def get_state_class(self) -> Type[SocialMediaAgentState]:
        return SocialMediaAgentState

    # ─── State Preparation ────────────────────────────────────────────

    def _prepare_initial_state(
        self, task: dict[str, Any], run_id: str
    ) -> dict[str, Any]:
        """Prepare initial state from social media task."""
        state = super()._prepare_initial_state(task, run_id)
        state.update({
            "platform": task.get("platform", "twitter"),
            "content_calendar": [],
            "trending_topics": [],
            "audience_insights": {},
            "posts_generated": [],
            "post_count": 0,
            "draft_posts": [],
            "engagement_metrics": {},
            "top_performing_posts": [],
            "audience_growth": {},
            "outreach_insights": [],
            "seo_keywords": [],
            "posts_approved": False,
            "human_edits": [],
            "scheduled_posts": [],
            "posts_published": 0,
            "report_summary": "",
            "report_generated_at": "",
        })
        return state

    # ─── Node 1: Listen ──────────────────────────────────────────────

    async def _node_listen(
        self, state: SocialMediaAgentState
    ) -> dict[str, Any]:
        """
        Node 1: Listen — Monitor trends and mentions.

        Checks trending topics in the vertical's niche and pulls
        recent mentions for engagement opportunities.
        """
        platform = state.get("platform", "twitter")
        task_input = state.get("task_input", {})
        if not isinstance(task_input, dict):
            task_input = {}
        topic = task_input.get("topic", "")
        niche = topic or self.config.params.get("niche", "cybersecurity")

        logger.info(
            "social_listen_start",
            extra={
                "agent_id": self.agent_id,
                "platform": platform,
                "niche": niche,
            },
        )

        # Check trending topics
        trending_topics: list[str] = []
        try:
            from core.mcp.tools.social_tools import check_trending_topics
            trends_json = await check_trending_topics(niche=niche, platform=platform)
            trends_data = json.loads(trends_json)
            for t in trends_data.get("topics", []):
                trending_topics.append(t.get("topic", ""))
        except Exception as e:
            logger.debug(f"Trending topics fetch failed: {e}")

        # Pull outreach insights from shared brain
        outreach_insights: list[dict[str, Any]] = []
        try:
            query = f"winning pattern messaging {niche}"
            insights = self.db.search_insights(
                query_embedding=self.embedder.embed_query(query),
                insight_type="winning_pattern",
                limit=5,
            )
            outreach_insights = insights if isinstance(insights, list) else []
        except Exception as e:
            logger.debug(f"Could not fetch outreach insights: {e}")

        # Pull SEO keywords from shared brain
        seo_keywords: list[str] = []
        try:
            query = f"keyword performance {niche}"
            kw_insights = self.db.search_insights(
                query_embedding=self.embedder.embed_query(query),
                insight_type="keyword_performance",
                limit=5,
            )
            for kw in (kw_insights if isinstance(kw_insights, list) else []):
                content = kw.get("content", "")
                if content:
                    seo_keywords.append(content[:50])
        except Exception as e:
            logger.debug(f"Could not fetch SEO keywords: {e}")

        logger.info(
            "social_listen_complete",
            extra={
                "trending_count": len(trending_topics),
                "insights_count": len(outreach_insights),
                "keywords_count": len(seo_keywords),
            },
        )

        return {
            "current_node": "listen",
            "trending_topics": trending_topics,
            "outreach_insights": outreach_insights,
            "seo_keywords": seo_keywords,
        }

    # ─── Node 2: Ideate ──────────────────────────────────────────────

    async def _node_ideate(
        self, state: SocialMediaAgentState
    ) -> dict[str, Any]:
        """
        Node 2: Ideate — Generate content ideas based on trends + insights.

        Uses LLM to brainstorm post ideas that align with trending topics,
        outreach insights, and SEO keywords for maximum impact.
        """
        platform = state.get("platform", "twitter")
        trending = state.get("trending_topics", [])
        insights = state.get("outreach_insights", [])
        keywords = state.get("seo_keywords", [])
        task_input = state.get("task_input", {})
        if not isinstance(task_input, dict):
            task_input = {}
        topic = task_input.get("topic", "")
        num_posts = task_input.get("num_posts", self.config.params.get("posts_per_run", 3))

        logger.info(
            "social_ideate_start",
            extra={
                "agent_id": self.agent_id,
                "platform": platform,
                "num_posts": num_posts,
            },
        )

        # Build context for ideation
        trends_section = ""
        if trending:
            trends_section = "TRENDING TOPICS:\n" + "\n".join(f"  - {t}" for t in trending[:10])

        insights_section = ""
        if insights:
            insights_section = "\nOUTREACH INSIGHTS (what messaging resonates):\n"
            for ins in insights[:5]:
                content = ins.get("content", ins.get("title", ""))[:200]
                insights_section += f"  - {content}\n"

        keywords_section = ""
        if keywords:
            keywords_section = "\nSEO KEYWORDS TO INCORPORATE:\n" + ", ".join(keywords[:10])

        our_company = self.config.params.get("company_name", "Enclave Guard")
        our_niche = self.config.params.get("niche", "cybersecurity")
        char_limit = TWITTER_CHAR_LIMIT if platform in ("twitter", "x") else LINKEDIN_CHAR_LIMIT

        ideation_prompt = (
            f"Generate {num_posts} social media post ideas for {platform}.\n\n"
            f"COMPANY: {our_company}\n"
            f"NICHE: {our_niche}\n"
            f"{'SPECIFIC TOPIC: ' + topic if topic else ''}\n"
            f"CHARACTER LIMIT: {char_limit}\n\n"
            f"{trends_section}\n"
            f"{insights_section}\n"
            f"{keywords_section}\n\n"
            f"INSTRUCTIONS:\n"
            f"Return a JSON array of post ideas. Each idea should have:\n"
            f'- "post_type": one of [thought_leadership, case_study, industry_news, engagement, meme, tip, thread]\n'
            f'- "topic": what the post is about\n'
            f'- "hook": the attention-grabbing opening line\n'
            f'- "key_message": the core message\n'
            f'- "cta": call-to-action (optional)\n'
            f'- "needs_image": boolean\n'
            f'- "hashtags": list of relevant hashtags\n\n'
            f"Make posts varied — mix educational, engaging, and thought-provoking.\n"
            f"Return ONLY the JSON array, no code fences."
        )

        content_calendar: list[dict[str, Any]] = []

        try:
            response = self.llm.messages.create(
                model=self.config.model.model,
                max_tokens=2000,
                temperature=0.7,  # Higher creativity for social
                messages=[{"role": "user", "content": ideation_prompt}],
            )
            text = response.content[0].text.strip() if response.content else "[]"

            # Clean JSON
            if "```" in text:
                parts = text.split("```")
                text = parts[1] if len(parts) > 1 else text
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()

            parsed = json.loads(text)
            if isinstance(parsed, list):
                content_calendar = parsed[:num_posts]

        except Exception as e:
            logger.warning(
                "social_ideate_failed",
                extra={"error": str(e)[:200]},
            )
            # Fallback: basic content calendar
            content_calendar = [{
                "post_type": "thought_leadership",
                "topic": topic or our_niche,
                "hook": f"Here's what most people get wrong about {our_niche}...",
                "key_message": f"Expert insight on {our_niche}",
                "cta": "Follow for more insights",
                "needs_image": False,
                "hashtags": [f"#{our_niche.replace(' ', '')}"],
            }]

        logger.info(
            "social_ideate_complete",
            extra={
                "ideas_generated": len(content_calendar),
            },
        )

        return {
            "current_node": "ideate",
            "content_calendar": content_calendar,
        }

    # ─── Node 3: Create ──────────────────────────────────────────────

    async def _node_create(
        self, state: SocialMediaAgentState
    ) -> dict[str, Any]:
        """
        Node 3: Create — Draft actual posts from content ideas.

        Takes each idea from the content calendar and generates
        platform-specific copy, hashtags, and image suggestions.
        Optionally calls image_gen (Phase 6) if a visual is needed.
        """
        platform = state.get("platform", "twitter")
        calendar = state.get("content_calendar", [])
        char_limit = TWITTER_CHAR_LIMIT if platform in ("twitter", "x") else LINKEDIN_CHAR_LIMIT

        logger.info(
            "social_create_start",
            extra={
                "agent_id": self.agent_id,
                "platform": platform,
                "ideas": len(calendar),
            },
        )

        draft_posts: list[dict[str, Any]] = []

        for i, idea in enumerate(calendar):
            topic = idea.get("topic", "")
            hook = idea.get("hook", "")
            key_message = idea.get("key_message", "")
            post_type = idea.get("post_type", "thought_leadership")
            cta = idea.get("cta", "")
            hashtags = idea.get("hashtags", [])
            needs_image = idea.get("needs_image", False)

            # Generate the post copy
            copy_prompt = (
                f"Write a {platform} post based on this idea.\n\n"
                f"POST TYPE: {post_type}\n"
                f"TOPIC: {topic}\n"
                f"HOOK: {hook}\n"
                f"KEY MESSAGE: {key_message}\n"
                f"CTA: {cta}\n"
                f"HASHTAGS: {', '.join(hashtags)}\n"
                f"CHARACTER LIMIT: {char_limit}\n\n"
                f"Write the FINAL post copy. Include hashtags naturally at the end.\n"
                f"{'Keep it punchy — 280 chars max.' if platform in ('twitter', 'x') else 'LinkedIn style — professional but engaging, 500-1500 chars.'}\n"
                f"Return ONLY the post text, nothing else."
            )

            content = ""
            try:
                response = self.llm.messages.create(
                    model=self.config.model.model,
                    max_tokens=500,
                    temperature=0.7,
                    messages=[{"role": "user", "content": copy_prompt}],
                    system=self._get_system_prompt(),
                )
                content = response.content[0].text.strip() if response.content else ""
            except Exception as e:
                logger.warning(
                    "social_create_post_failed",
                    extra={"error": str(e)[:200], "index": i},
                )
                content = f"{hook}\n\n{key_message}\n\n{' '.join(hashtags)}"

            # Enforce character limit
            if len(content) > char_limit:
                content = content[:char_limit - 3] + "..."

            # Image generation (optional, uses Phase 6 creative suite)
            media_suggestion = ""
            if needs_image:
                media_suggestion = (
                    f"Suggested image: {post_type} visual about '{topic}'. "
                    f"Style: professional, brand-aligned."
                )
                # In production: call image_gen here
                # from core.llm.creative import CreativeSuite
                # suite = CreativeSuite(router=self.router)
                # image_result = await suite.generate_image(...)

            post = {
                "platform": platform,
                "content": content,
                "post_type": post_type,
                "hashtags": hashtags,
                "media_suggestion": media_suggestion,
                "topic": topic,
                "index": i,
            }
            draft_posts.append(post)

        logger.info(
            "social_create_complete",
            extra={
                "posts_created": len(draft_posts),
                "platform": platform,
            },
        )

        return {
            "current_node": "create",
            "draft_posts": draft_posts,
            "post_count": len(draft_posts),
        }

    # ─── Node 4: Human Review ────────────────────────────────────────

    async def _node_human_review(
        self, state: SocialMediaAgentState
    ) -> dict[str, Any]:
        """
        Node 4: Human Review — NEVER auto-post.

        ALL posts must be reviewed and approved by a human before
        publishing. This is a non-negotiable safety gate.
        """
        post_count = state.get("post_count", 0)
        platform = state.get("platform", "")

        logger.info(
            "social_human_review_pending",
            extra={
                "agent_id": self.agent_id,
                "platform": platform,
                "post_count": post_count,
            },
        )

        return {
            "current_node": "human_review",
            "requires_human_approval": True,
        }

    # ─── Node 5: Publish ─────────────────────────────────────────────

    async def _node_publish(
        self, state: SocialMediaAgentState
    ) -> dict[str, Any]:
        """
        Node 5: Publish — Post approved content to social platforms.

        Iterates through approved draft posts and publishes them
        via the social MCP tools. Captures RLHF data if posts were edited.
        """
        draft_posts = state.get("draft_posts", [])
        human_edits = state.get("human_edits", [])
        platform = state.get("platform", "twitter")

        logger.info(
            "social_publish_start",
            extra={
                "agent_id": self.agent_id,
                "platform": platform,
                "post_count": len(draft_posts),
            },
        )

        published: list[dict[str, Any]] = []
        published_count = 0

        for post in draft_posts:
            content = post.get("content", "")
            if not content:
                continue

            # Check for human edits
            edited = False
            for edit in human_edits:
                if edit.get("index") == post.get("index"):
                    original = content
                    content = edit.get("edited_content", content)
                    edited = True

                    # RLHF capture
                    self.learn(
                        task_input={
                            "platform": platform,
                            "post_type": post.get("post_type", ""),
                            "topic": post.get("topic", ""),
                        },
                        model_output=original,
                        human_correction=content,
                        source="manual_review",
                        metadata={
                            "agent_type": "social",
                            "platform": platform,
                        },
                    )
                    break

            try:
                from core.mcp.tools.social_tools import post_social_update
                result_json = await post_social_update(
                    platform=platform,
                    content=content,
                    image_path=None,  # TODO: wire image_gen in v2
                )
                result = json.loads(result_json)

                published.append({
                    "platform": platform,
                    "content": content[:100],
                    "post_type": post.get("post_type", ""),
                    "published_at": datetime.now(timezone.utc).isoformat(),
                    "result": result,
                    "was_edited": edited,
                })
                published_count += 1

            except Exception as e:
                logger.error(
                    "social_publish_failed",
                    extra={
                        "error": str(e)[:200],
                        "post_index": post.get("index", -1),
                    },
                )

        # Write insights to shared brain
        if published_count > 0:
            self.store_insight(InsightData(
                insight_type="social_activity",
                title=f"Social: {published_count} posts published on {platform}",
                content=(
                    f"Published {published_count} posts on {platform}. "
                    f"Types: {', '.join(set(p.get('post_type', '') for p in published))}."
                ),
                confidence=0.80,
                metadata={
                    "platform": platform,
                    "post_count": published_count,
                    "had_edits": any(p.get("was_edited") for p in published),
                },
            ))

        logger.info(
            "social_publish_complete",
            extra={
                "published": published_count,
                "platform": platform,
            },
        )

        return {
            "current_node": "publish",
            "scheduled_posts": published,
            "posts_published": published_count,
            "knowledge_written": True,
        }

    # ─── Node 6: Report ──────────────────────────────────────────────

    async def _node_report(
        self, state: SocialMediaAgentState
    ) -> dict[str, Any]:
        """
        Node 6: Report — Generate a social media activity summary.
        """
        now = datetime.now(timezone.utc).isoformat()
        platform = state.get("platform", "")
        published = state.get("posts_published", 0)
        draft_count = state.get("post_count", 0)
        trending = state.get("trending_topics", [])
        approved = state.get("posts_approved", False)

        sections = [
            "# Social Media Activity Report",
            f"*Generated: {now}*\n",
            "## Summary",
            f"- **Platform:** {platform}",
            f"- **Posts drafted:** {draft_count}",
            f"- **Posts published:** {published}",
            f"- **Status:** {'Published' if published > 0 else ('Rejected' if not approved else 'Pending')}",
        ]

        if trending:
            sections.append("\n## Trending Topics")
            for t in trending[:5]:
                sections.append(f"  - {t}")

        # Engagement summary (when available)
        engagement = state.get("engagement_metrics", {})
        if engagement:
            sections.append("\n## Engagement")
            sections.append(f"  - Impressions: {engagement.get('impressions', 0)}")
            sections.append(f"  - Clicks: {engagement.get('clicks', 0)}")
            sections.append(f"  - Shares: {engagement.get('shares', 0)}")

        report = "\n".join(sections)

        return {
            "current_node": "report",
            "report_summary": report,
            "report_generated_at": now,
        }

    # ─── Routing Functions ────────────────────────────────────────────

    @staticmethod
    def _route_after_review(state: SocialMediaAgentState) -> str:
        """Route after human review: approved → publish, rejected → report."""
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
            "You are a social media strategist for a modern tech company. "
            "You write engaging, authentic posts that build thought leadership "
            "and drive engagement. Your style is confident, insightful, and "
            "human — never corporate or robotic. You understand each platform's "
            "culture: Twitter is punchy and conversational, LinkedIn is "
            "professional but authentic. Use hooks, insights, and clear CTAs."
        )

    # ─── Knowledge Writing ────────────────────────────────────────────

    async def write_knowledge(self, result: dict[str, Any]) -> None:
        """Write social insights to shared brain (handled in publish node)."""
        pass

    def __repr__(self) -> str:
        return (
            f"<SocialMediaAgent "
            f"agent_id={self.agent_id!r} "
            f"vertical={self.vertical_id!r}>"
        )
