"""
The Strategist — Cross-Agent Performance Analysis & Growth Recommendations.

Analyzes agent performance, budget efficiency, and market trends across
the entire vertical. Identifies growth opportunities and generates
actionable recommendations — either as A/B test proposals (via
ExperimentEngine) or new vertical suggestions (via Genesis).

The Strategist NEVER makes changes directly. It proposes:
1. Experiments (validated via ExperimentEngine's Bayesian framework)
2. Budget reallocations (executed via BudgetManager after approval)
3. New vertical ideas (submitted for human review)

Think of it as the "Chief Strategy Officer" that reads every report
from every department and identifies where to invest next.

Usage:
    strategist = Strategist(db=db, hive_mind=hive)

    # Get cross-agent performance snapshot
    perf = strategist.scan_performance("enclave_guard", days=30)

    # Find growth opportunities
    opps = strategist.detect_opportunities(perf)

    # Generate experiment proposals
    experiments = strategist.propose_experiments(opps)

    # Generate full strategy report
    report = strategist.generate_strategy_report(perf, opps, experiments)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ── Opportunity Types ─────────────────────────────────────────

OPPORTUNITY_TYPES = {
    "underperforming_agent",
    "high_performing_agent",
    "budget_inefficiency",
    "stale_experiment",
    "untapped_channel",
    "audience_mismatch",
    "content_gap",
    "timing_optimization",
    "vertical_expansion",
}

# Impact levels for prioritization
IMPACT_HIGH = "high"
IMPACT_MEDIUM = "medium"
IMPACT_LOW = "low"

# Thresholds for opportunity detection
MIN_AGENT_RUNS_FOR_ANALYSIS = 5
LOW_SUCCESS_RATE_THRESHOLD = 0.5
HIGH_SUCCESS_RATE_THRESHOLD = 0.85
STALE_EXPERIMENT_DAYS = 14
MIN_LEAD_SCORE_THRESHOLD = 40.0


class Strategist:
    """
    Cross-agent performance analysis and growth recommendations.

    Reads from:
    - agent_runs (OverseerAgent metrics)
    - shared_insights (Hive Mind)
    - experiments (ExperimentEngine)
    - companies (lead data)
    - budget_snapshots (BudgetManager)

    Outputs:
    - Performance snapshots
    - Growth opportunities
    - Experiment proposals (ExperimentEngine-compatible)
    - Vertical suggestions (Genesis-compatible)
    - Human-readable strategy reports
    """

    def __init__(
        self,
        db: Any,
        embedder: Any = None,
        hive_mind: Any = None,
    ):
        self.db = db
        self.embedder = embedder
        self.hive = hive_mind

    # ── Performance Scanning ──────────────────────────────────

    def scan_performance(
        self,
        vertical_id: str,
        days: int = 30,
    ) -> dict[str, Any]:
        """
        Cross-agent performance snapshot.

        Collects metrics from all agents, experiments, insights,
        and lead pipeline for a given vertical.

        Returns:
            {
                "vertical_id": str,
                "period_days": int,
                "agents": [{id, runs, success_rate, avg_duration, errors}],
                "experiments": [{id, name, status, observations, has_winner}],
                "insights_count": int,
                "insight_categories": {category: count},
                "lead_stats": {total, avg_score, qualified_pct},
                "budget_summary": {total_spend, total_revenue, roas},
                "scanned_at": str,
            }
        """
        agents = self._scan_agent_metrics(vertical_id, days)
        experiments = self._scan_experiments(vertical_id)
        insights = self._scan_insights(vertical_id, days)
        lead_stats = self._scan_lead_pipeline(vertical_id)
        budget = self._scan_budget(vertical_id, days)

        return {
            "vertical_id": vertical_id,
            "period_days": days,
            "agents": agents,
            "experiments": experiments,
            "insights_count": insights.get("total", 0),
            "insight_categories": insights.get("by_category", {}),
            "lead_stats": lead_stats,
            "budget_summary": budget,
            "scanned_at": datetime.now(timezone.utc).isoformat(),
        }

    def _scan_agent_metrics(
        self, vertical_id: str, days: int,
    ) -> list[dict[str, Any]]:
        """Get per-agent run stats from agent_runs table."""
        try:
            result = (
                self.db.client.table("agent_runs")
                .select("agent_id, status, duration_ms, error_message")
                .eq("vertical_id", vertical_id)
                .order("created_at", desc=True)
                .limit(500)
                .execute()
            )

            # Aggregate by agent
            agents: dict[str, dict[str, Any]] = {}
            for run in (result.data or []):
                aid = run.get("agent_id", "unknown")
                if aid not in agents:
                    agents[aid] = {
                        "agent_id": aid,
                        "total_runs": 0,
                        "successful_runs": 0,
                        "failed_runs": 0,
                        "total_duration_ms": 0,
                        "errors": [],
                    }

                agents[aid]["total_runs"] += 1
                status = run.get("status", "")
                if status in ("completed", "success"):
                    agents[aid]["successful_runs"] += 1
                elif status in ("failed", "error"):
                    agents[aid]["failed_runs"] += 1
                    error = run.get("error_message", "")
                    if error and len(agents[aid]["errors"]) < 3:
                        agents[aid]["errors"].append(error[:200])

                duration = run.get("duration_ms")
                if duration:
                    agents[aid]["total_duration_ms"] += int(duration)

            # Compute rates
            agent_list = []
            for a in agents.values():
                total = a["total_runs"]
                a["success_rate"] = (
                    round(a["successful_runs"] / total, 3) if total > 0 else 0.0
                )
                a["avg_duration_ms"] = (
                    round(a["total_duration_ms"] / total) if total > 0 else 0
                )
                agent_list.append(a)

            # Sort by total runs descending
            agent_list.sort(key=lambda x: x["total_runs"], reverse=True)
            return agent_list

        except Exception as e:
            logger.error(f"Failed to scan agent metrics: {e}")
            return []

    def _scan_experiments(self, vertical_id: str) -> list[dict[str, Any]]:
        """Get experiment status from experiments table."""
        try:
            result = (
                self.db.client.table("experiments")
                .select("*")
                .eq("vertical_id", vertical_id)
                .order("created_at", desc=True)
                .limit(20)
                .execute()
            )

            experiments = []
            for exp in (result.data or []):
                experiments.append({
                    "experiment_id": exp.get("experiment_id", ""),
                    "name": exp.get("name", ""),
                    "status": exp.get("status", "unknown"),
                    "agent_id": exp.get("agent_id", ""),
                    "metric": exp.get("metric", ""),
                    "variants": exp.get("variants", []),
                    "created_at": exp.get("created_at", ""),
                })

            return experiments

        except Exception as e:
            logger.error(f"Failed to scan experiments: {e}")
            return []

    def _scan_insights(
        self, vertical_id: str, days: int,
    ) -> dict[str, Any]:
        """Count insights by category from shared_insights."""
        try:
            result = (
                self.db.client.table("shared_insights")
                .select("insight_type")
                .eq("vertical_id", vertical_id)
                .order("created_at", desc=True)
                .limit(500)
                .execute()
            )

            by_category: dict[str, int] = {}
            total = 0
            for insight in (result.data or []):
                cat = insight.get("insight_type", "unknown")
                by_category[cat] = by_category.get(cat, 0) + 1
                total += 1

            return {"total": total, "by_category": by_category}

        except Exception as e:
            logger.error(f"Failed to scan insights: {e}")
            return {"total": 0, "by_category": {}}

    def _scan_lead_pipeline(self, vertical_id: str) -> dict[str, Any]:
        """Get lead pipeline statistics from companies table."""
        try:
            result = (
                self.db.client.table("companies")
                .select("lead_score, status")
                .eq("vertical_id", vertical_id)
                .limit(500)
                .execute()
            )

            scores = []
            statuses: dict[str, int] = {}
            for row in (result.data or []):
                score = row.get("lead_score")
                if score is not None:
                    scores.append(float(score))
                status = row.get("status", "unknown")
                statuses[status] = statuses.get(status, 0) + 1

            total = len(scores)
            avg_score = round(sum(scores) / total, 1) if total > 0 else 0.0
            qualified = sum(
                1 for s in scores if s >= MIN_LEAD_SCORE_THRESHOLD
            )
            qualified_pct = round(qualified / total, 3) if total > 0 else 0.0

            return {
                "total": total,
                "avg_score": avg_score,
                "qualified_count": qualified,
                "qualified_pct": qualified_pct,
                "by_status": statuses,
            }

        except Exception as e:
            logger.error(f"Failed to scan lead pipeline: {e}")
            return {
                "total": 0, "avg_score": 0.0,
                "qualified_count": 0, "qualified_pct": 0.0,
                "by_status": {},
            }

    def _scan_budget(
        self, vertical_id: str, days: int,
    ) -> dict[str, Any]:
        """Get budget summary from budget_snapshots or shared_insights."""
        try:
            result = (
                self.db.client.table("budget_snapshots")
                .select("total_spend, total_revenue, roas")
                .eq("vertical_id", vertical_id)
                .order("period", desc=True)
                .limit(1)
                .execute()
            )

            if result.data:
                latest = result.data[0]
                return {
                    "total_spend": float(latest.get("total_spend", 0)),
                    "total_revenue": float(latest.get("total_revenue", 0)),
                    "roas": float(latest.get("roas", 0)),
                }
        except Exception as e:
            logger.debug(f"Budget scan fallback: {e}")

        return {"total_spend": 0.0, "total_revenue": 0.0, "roas": 0.0}

    # ── Opportunity Detection ─────────────────────────────────

    def detect_opportunities(
        self,
        performance: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """
        Identify growth levers from performance data.

        Scans for:
        - Underperforming agents → optimization opportunity
        - High performers → scale-up opportunity
        - Budget inefficiency → reallocation opportunity
        - Stale experiments → action needed
        - Content gaps → new content experiments
        - Low lead quality → targeting optimization

        Returns:
            [
                {
                    "type": str,
                    "description": str,
                    "potential_impact": "high"|"medium"|"low",
                    "confidence": float,  # 0.0–1.0
                    "suggested_action": str,
                    "details": dict,
                }
            ]
        """
        opportunities: list[dict[str, Any]] = []

        # 1. Agent performance opportunities
        for agent in performance.get("agents", []):
            runs = agent.get("total_runs", 0)
            if runs < MIN_AGENT_RUNS_FOR_ANALYSIS:
                continue

            success_rate = agent.get("success_rate", 0)
            agent_id = agent.get("agent_id", "unknown")

            if success_rate < LOW_SUCCESS_RATE_THRESHOLD:
                opportunities.append({
                    "type": "underperforming_agent",
                    "description": (
                        f"Agent '{agent_id}' has {success_rate:.0%} success rate "
                        f"over {runs} runs — below {LOW_SUCCESS_RATE_THRESHOLD:.0%} threshold"
                    ),
                    "potential_impact": IMPACT_HIGH,
                    "confidence": min(0.5 + (runs / 100), 0.95),
                    "suggested_action": (
                        f"Investigate {agent_id} errors and apply config optimization"
                    ),
                    "details": {
                        "agent_id": agent_id,
                        "success_rate": success_rate,
                        "total_runs": runs,
                        "recent_errors": agent.get("errors", []),
                    },
                })
            elif success_rate >= HIGH_SUCCESS_RATE_THRESHOLD and runs >= 20:
                opportunities.append({
                    "type": "high_performing_agent",
                    "description": (
                        f"Agent '{agent_id}' has {success_rate:.0%} success rate — "
                        f"consider increasing its workload or frequency"
                    ),
                    "potential_impact": IMPACT_MEDIUM,
                    "confidence": min(0.6 + (runs / 200), 0.90),
                    "suggested_action": (
                        f"Increase {agent_id} execution frequency or expand scope"
                    ),
                    "details": {
                        "agent_id": agent_id,
                        "success_rate": success_rate,
                        "total_runs": runs,
                    },
                })

        # 2. Budget inefficiency
        budget = performance.get("budget_summary", {})
        roas = budget.get("roas", 0)
        total_spend = budget.get("total_spend", 0)

        if total_spend > 0 and roas < 1.0:
            opportunities.append({
                "type": "budget_inefficiency",
                "description": (
                    f"Overall ROAS is {roas:.1f}x — spending ${total_spend:.0f} "
                    f"but generating less revenue. Budget reallocation recommended."
                ),
                "potential_impact": IMPACT_HIGH,
                "confidence": 0.85,
                "suggested_action": (
                    "Run budget reallocation analysis to shift spend toward "
                    "higher-performing campaigns"
                ),
                "details": {
                    "roas": roas,
                    "total_spend": total_spend,
                    "total_revenue": budget.get("total_revenue", 0),
                },
            })
        elif total_spend > 0 and roas >= 3.0:
            opportunities.append({
                "type": "high_performing_agent",
                "description": (
                    f"ROAS is excellent at {roas:.1f}x — consider increasing "
                    f"total ad budget to capture more market share"
                ),
                "potential_impact": IMPACT_HIGH,
                "confidence": 0.80,
                "suggested_action": "Evaluate budget increase with stakeholders",
                "details": {"roas": roas, "total_spend": total_spend},
            })

        # 3. Stale experiments
        for exp in performance.get("experiments", []):
            if exp.get("status") == "active":
                created = exp.get("created_at", "")
                if created:
                    try:
                        created_dt = datetime.fromisoformat(
                            created.replace("Z", "+00:00")
                        )
                        age_days = (
                            datetime.now(timezone.utc) - created_dt
                        ).days
                        if age_days > STALE_EXPERIMENT_DAYS:
                            opportunities.append({
                                "type": "stale_experiment",
                                "description": (
                                    f"Experiment '{exp.get('name', '')}' has been "
                                    f"running for {age_days} days without conclusion"
                                ),
                                "potential_impact": IMPACT_LOW,
                                "confidence": 0.70,
                                "suggested_action": (
                                    "Review experiment results and conclude if "
                                    "sufficient data collected"
                                ),
                                "details": {
                                    "experiment_id": exp.get("experiment_id"),
                                    "name": exp.get("name"),
                                    "age_days": age_days,
                                },
                            })
                    except (ValueError, TypeError):
                        pass

        # 4. Low lead quality
        lead_stats = performance.get("lead_stats", {})
        qualified_pct = lead_stats.get("qualified_pct", 0)
        total_leads = lead_stats.get("total", 0)

        if total_leads >= 10 and qualified_pct < 0.3:
            opportunities.append({
                "type": "audience_mismatch",
                "description": (
                    f"Only {qualified_pct:.0%} of {total_leads} leads are qualified "
                    f"(score ≥ {MIN_LEAD_SCORE_THRESHOLD}). Targeting may need adjustment."
                ),
                "potential_impact": IMPACT_MEDIUM,
                "confidence": 0.75,
                "suggested_action": (
                    "Review ICP criteria and adjust lead sourcing parameters"
                ),
                "details": {
                    "total_leads": total_leads,
                    "qualified_pct": qualified_pct,
                    "avg_score": lead_stats.get("avg_score", 0),
                },
            })

        # 5. Content gaps — no insights in important categories
        insight_cats = performance.get("insight_categories", {})
        expected_categories = {
            "email_performance", "ad_performance",
            "content_performance", "audience_response",
        }
        missing = expected_categories - set(insight_cats.keys())
        if missing and performance.get("insights_count", 0) > 10:
            opportunities.append({
                "type": "content_gap",
                "description": (
                    f"Missing insight data in categories: {', '.join(sorted(missing))}. "
                    f"Consider activating agents that cover these areas."
                ),
                "potential_impact": IMPACT_LOW,
                "confidence": 0.60,
                "suggested_action": (
                    "Review agent coverage and enable agents for missing categories"
                ),
                "details": {
                    "missing_categories": sorted(missing),
                    "existing_categories": sorted(insight_cats.keys()),
                },
            })

        # Sort by impact and confidence
        impact_order = {IMPACT_HIGH: 3, IMPACT_MEDIUM: 2, IMPACT_LOW: 1}
        opportunities.sort(
            key=lambda x: (
                impact_order.get(x.get("potential_impact", ""), 0),
                x.get("confidence", 0),
            ),
            reverse=True,
        )

        return opportunities

    # ── Experiment Proposals ──────────────────────────────────

    def propose_experiments(
        self,
        opportunities: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Generate A/B test proposals from detected opportunities.

        Returns ExperimentEngine-compatible dicts:
            [
                {
                    "name": str,
                    "variants": [str, str],
                    "agent_id": str,
                    "metric": str,
                    "metadata": {...},
                    "source_opportunity": str,
                }
            ]
        """
        proposals: list[dict[str, Any]] = []

        for opp in opportunities:
            opp_type = opp.get("type", "")
            details = opp.get("details", {})

            if opp_type == "underperforming_agent":
                agent_id = details.get("agent_id", "unknown")
                proposals.append({
                    "name": f"Config Optimization: {agent_id}",
                    "variants": ["current_config", "optimized_config"],
                    "agent_id": agent_id,
                    "metric": "success_rate",
                    "metadata": {
                        "experiment_type": "config_optimization",
                        "baseline_success_rate": details.get("success_rate", 0),
                        "source": "strategist_auto",
                    },
                    "source_opportunity": opp_type,
                })

            elif opp_type == "budget_inefficiency":
                proposals.append({
                    "name": "Budget Reallocation Strategy Test",
                    "variants": ["current_allocation", "roas_weighted"],
                    "agent_id": "ads_strategy",
                    "metric": "roas",
                    "metadata": {
                        "experiment_type": "budget_optimization",
                        "baseline_roas": details.get("roas", 0),
                        "source": "strategist_auto",
                    },
                    "source_opportunity": opp_type,
                })

            elif opp_type == "audience_mismatch":
                proposals.append({
                    "name": "ICP Criteria Refinement Test",
                    "variants": ["current_icp", "refined_icp"],
                    "agent_id": "outreach",
                    "metric": "qualified_rate",
                    "metadata": {
                        "experiment_type": "targeting_optimization",
                        "baseline_qualified_pct": details.get("qualified_pct", 0),
                        "source": "strategist_auto",
                    },
                    "source_opportunity": opp_type,
                })

            elif opp_type == "timing_optimization":
                proposals.append({
                    "name": "Outreach Timing Test",
                    "variants": ["morning_send", "afternoon_send"],
                    "agent_id": "outreach",
                    "metric": "reply_rate",
                    "metadata": {
                        "experiment_type": "timing",
                        "source": "strategist_auto",
                    },
                    "source_opportunity": opp_type,
                })

        return proposals

    # ── Vertical Proposals ────────────────────────────────────

    def propose_vertical(
        self,
        trends: list[dict[str, Any]],
        existing_verticals: list[str],
    ) -> Optional[dict[str, Any]]:
        """
        Suggest a new vertical based on market trends.

        Analyzes trend data (from Hive Mind market_signal insights)
        and compares against existing verticals to find gaps.

        Returns:
            BusinessBlueprint-compatible dict with:
            - name: suggested vertical name
            - industry: target industry
            - icp: ideal customer profile
            - reasoning: why this vertical
            - confidence: 0.0-1.0
            Or None if no strong proposal.
        """
        if not trends:
            return None

        # Score each trend by relevance and novelty
        scored_trends: list[tuple[float, dict]] = []

        for trend in trends:
            industry = trend.get("industry", "").lower()
            signal_strength = float(trend.get("signal_strength", 0))
            growth_rate = float(trend.get("growth_rate", 0))

            # Skip if already covered
            if any(
                industry in v.lower() or v.lower() in industry
                for v in existing_verticals
            ):
                continue

            # Score: signal_strength * growth_rate * novelty_bonus
            novelty_bonus = 1.2  # Bonus for not being in existing verticals
            score = signal_strength * max(growth_rate, 0.1) * novelty_bonus

            scored_trends.append((score, trend))

        if not scored_trends:
            return None

        # Pick the highest-scoring trend
        scored_trends.sort(key=lambda x: x[0], reverse=True)
        best_score, best_trend = scored_trends[0]

        # Only propose if sufficiently strong
        if best_score < 0.3:
            return None

        confidence = min(best_score, 1.0)

        return {
            "name": best_trend.get("industry", "new_vertical"),
            "industry": best_trend.get("industry", ""),
            "icp": {
                "company_size": best_trend.get("target_company_size", "10-500"),
                "decision_maker": best_trend.get("decision_maker_title", "CTO"),
                "pain_points": best_trend.get("pain_points", []),
            },
            "reasoning": (
                f"Market signal strength: {best_trend.get('signal_strength', 0):.1f}, "
                f"growth rate: {best_trend.get('growth_rate', 0):.0%}. "
                f"Not covered by existing verticals: {existing_verticals}"
            ),
            "confidence": round(confidence, 2),
            "source_trend": best_trend,
        }

    # ── Strategy Report ───────────────────────────────────────

    def generate_strategy_report(
        self,
        performance: dict[str, Any],
        opportunities: list[dict[str, Any]],
        experiments: list[dict[str, Any]],
    ) -> str:
        """
        Generate human-readable strategy summary.

        Produces a markdown-formatted report covering:
        - Performance overview
        - Key opportunities
        - Recommended experiments
        - Actionable next steps

        Returns:
            Markdown string.
        """
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        vertical = performance.get("vertical_id", "unknown")
        period = performance.get("period_days", 30)

        lines = [
            f"# Strategy Report — {vertical}",
            f"*Generated: {now} | Period: {period} days*",
            "",
        ]

        # ── Performance Overview
        lines.append("## Performance Overview")
        lines.append("")

        agents = performance.get("agents", [])
        if agents:
            lines.append(f"**Active Agents:** {len(agents)}")
            avg_success = (
                sum(a.get("success_rate", 0) for a in agents) / len(agents)
            )
            lines.append(f"**Average Success Rate:** {avg_success:.0%}")
            total_runs = sum(a.get("total_runs", 0) for a in agents)
            lines.append(f"**Total Runs:** {total_runs}")
        else:
            lines.append("No agent metrics available.")
        lines.append("")

        # Budget
        budget = performance.get("budget_summary", {})
        spend = budget.get("total_spend", 0)
        if spend > 0:
            lines.append(
                f"**Budget:** ${spend:,.0f} spent | "
                f"${budget.get('total_revenue', 0):,.0f} revenue | "
                f"ROAS: {budget.get('roas', 0):.1f}x"
            )
            lines.append("")

        # Lead pipeline
        leads = performance.get("lead_stats", {})
        if leads.get("total", 0) > 0:
            lines.append(
                f"**Leads:** {leads['total']} total | "
                f"Avg score: {leads.get('avg_score', 0):.0f} | "
                f"Qualified: {leads.get('qualified_pct', 0):.0%}"
            )
            lines.append("")

        # ── Opportunities
        if opportunities:
            lines.append(f"## Opportunities Detected ({len(opportunities)})")
            lines.append("")

            for i, opp in enumerate(opportunities, 1):
                impact = opp.get("potential_impact", "unknown")
                confidence = opp.get("confidence", 0)
                icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(
                    impact, "⚪"
                )
                lines.append(
                    f"{i}. {icon} **[{impact.upper()}]** "
                    f"{opp.get('description', '')}"
                )
                lines.append(
                    f"   - Action: {opp.get('suggested_action', 'N/A')}"
                )
                lines.append(f"   - Confidence: {confidence:.0%}")
                lines.append("")
        else:
            lines.append("## No Opportunities Detected")
            lines.append("System is performing within expected parameters.")
            lines.append("")

        # ── Experiment Proposals
        if experiments:
            lines.append(
                f"## Proposed Experiments ({len(experiments)})"
            )
            lines.append("")

            for exp in experiments:
                lines.append(f"- **{exp.get('name', 'Unnamed')}**")
                lines.append(
                    f"  Variants: {', '.join(exp.get('variants', []))}"
                )
                lines.append(f"  Metric: {exp.get('metric', 'N/A')}")
                lines.append(
                    f"  Agent: {exp.get('agent_id', 'N/A')}"
                )
                lines.append("")

        # ── Next Steps
        lines.append("## Recommended Next Steps")
        lines.append("")

        high_impact = [
            o for o in opportunities
            if o.get("potential_impact") == IMPACT_HIGH
        ]
        if high_impact:
            lines.append(
                f"1. Address {len(high_impact)} high-impact "
                f"opportunit{'y' if len(high_impact) == 1 else 'ies'} above"
            )
        if experiments:
            lines.append(
                f"{'2' if high_impact else '1'}. Launch "
                f"{len(experiments)} proposed experiment(s)"
            )
        if not high_impact and not experiments:
            lines.append(
                "1. System performing well — continue monitoring"
            )

        lines.append("")
        lines.append("---")
        lines.append(
            "*Report generated by Strategist v1.0 — "
            "Sovereign Venture Engine*"
        )

        return "\n".join(lines)

    # ── Hive Mind Integration ─────────────────────────────────

    def get_market_trends(
        self,
        vertical_id: str,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """
        Fetch market signal insights from Hive Mind.

        Returns trend data for vertical proposal analysis.
        """
        try:
            result = (
                self.db.client.table("shared_insights")
                .select("*")
                .eq("vertical_id", vertical_id)
                .eq("insight_type", "market_signal")
                .order("created_at", desc=True)
                .limit(limit)
                .execute()
            )

            trends = []
            for row in (result.data or []):
                meta = row.get("metadata", {}) or {}
                trends.append({
                    "industry": meta.get("industry", ""),
                    "signal_strength": float(meta.get("signal_strength", 0)),
                    "growth_rate": float(meta.get("growth_rate", 0)),
                    "target_company_size": meta.get("target_company_size", ""),
                    "decision_maker_title": meta.get("decision_maker_title", ""),
                    "pain_points": meta.get("pain_points", []),
                    "content": row.get("content", ""),
                    "created_at": row.get("created_at", ""),
                })

            return trends

        except Exception as e:
            logger.error(f"Failed to get market trends: {e}")
            return []

    def __repr__(self) -> str:
        return f"Strategist(db={self.db is not None}, hive={self.hive is not None})"
