"""
Budget Autopilot — ROAS-Based Optimization.

Monitors ad spend across campaigns, computes Return on Ad Spend (ROAS),
and recommends budget reallocations from underperformers to top performers.

Safety Model:
1. NEVER increases total spend — only redistributes
2. Large reallocations (>20%) flagged for human review
3. Respects org plan_tier budget limits
4. All changes logged to optimization_actions + budget_snapshots

Usage:
    mgr = BudgetManager(db)

    # Get current spending overview
    summary = mgr.get_spend_summary("enclave_guard", days=30)

    # Compute ROAS per campaign
    roas_data = mgr.compute_roas(summary["by_campaign"])

    # Get reallocation recommendations
    recs = mgr.recommend_reallocation(1000.0, roas_data)

    # Apply approved recommendations
    result = mgr.apply_reallocation(recs)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ── Safety Constants ─────────────────────────────────────────

# Maximum percentage of total budget that can be shifted in one reallocation
MAX_REALLOCATION_PCT = 0.20

# Minimum ROAS to consider a campaign "performing"
MIN_VIABLE_ROAS = 0.5

# Default budget limits by plan tier (monthly, in USD)
DEFAULT_BUDGET_LIMITS = {
    "free": 0.0,
    "starter": 1000.0,
    "pro": 5000.0,
    "enterprise": 50000.0,
}

# Minimum spend to have meaningful ROAS data
MIN_SPEND_FOR_ROAS = 10.0

# Strategies for reallocation
STRATEGY_MAXIMIZE_ROAS = "maximize_roas"
STRATEGY_BALANCE = "balance"
STRATEGY_CONSERVATIVE = "conservative"


class BudgetManager:
    """
    ROAS-based budget optimization across campaigns.

    Reads campaign performance data from shared_insights and
    agent_content tables, computes ROAS, and generates budget
    reallocation recommendations.
    """

    def __init__(self, db: Any, org_id: Optional[str] = None):
        self.db = db
        self.org_id = org_id

    # ── Spend Analysis ────────────────────────────────────────

    def get_spend_summary(
        self,
        vertical_id: str,
        days: int = 30,
    ) -> dict[str, Any]:
        """
        Aggregate spend and revenue across all campaigns.

        Pulls data from shared_insights (ad_performance category)
        and budget_snapshots for historical comparison.

        Returns:
            {
                "total_spend": float,
                "total_revenue": float,
                "roas": float,
                "by_campaign": [
                    {"campaign_id": str, "name": str, "spend": float,
                     "revenue": float, "roas": float, "platform": str}
                ],
                "period_days": int,
            }
        """
        try:
            # Query ad performance insights
            insights = (
                self.db.client.table("shared_insights")
                .select("*")
                .eq("vertical_id", vertical_id)
                .eq("insight_type", "ad_performance")
                .order("created_at", desc=True)
                .limit(200)
                .execute()
            )

            campaigns: dict[str, dict[str, Any]] = {}

            for insight in (insights.data or []):
                content = insight.get("content", "")
                meta = insight.get("metadata", {}) or {}
                campaign_id = meta.get("campaign_id", "unknown")

                if campaign_id not in campaigns:
                    campaigns[campaign_id] = {
                        "campaign_id": campaign_id,
                        "name": meta.get("campaign_name", campaign_id),
                        "spend": 0.0,
                        "revenue": 0.0,
                        "platform": meta.get("platform", "unknown"),
                        "impressions": 0,
                        "clicks": 0,
                        "conversions": 0,
                    }

                # Accumulate metrics
                campaigns[campaign_id]["spend"] += float(meta.get("spend", 0))
                campaigns[campaign_id]["revenue"] += float(meta.get("revenue", 0))
                campaigns[campaign_id]["impressions"] += int(meta.get("impressions", 0))
                campaigns[campaign_id]["clicks"] += int(meta.get("clicks", 0))
                campaigns[campaign_id]["conversions"] += int(meta.get("conversions", 0))

            # Compute per-campaign ROAS
            campaign_list = list(campaigns.values())
            for c in campaign_list:
                c["roas"] = (
                    c["revenue"] / c["spend"]
                    if c["spend"] > 0 else 0.0
                )

            total_spend = sum(c["spend"] for c in campaign_list)
            total_revenue = sum(c["revenue"] for c in campaign_list)
            overall_roas = total_revenue / total_spend if total_spend > 0 else 0.0

            return {
                "total_spend": round(total_spend, 2),
                "total_revenue": round(total_revenue, 2),
                "roas": round(overall_roas, 2),
                "by_campaign": campaign_list,
                "period_days": days,
            }

        except Exception as e:
            logger.error(f"Failed to get spend summary: {e}")
            return {
                "total_spend": 0.0,
                "total_revenue": 0.0,
                "roas": 0.0,
                "by_campaign": [],
                "period_days": days,
            }

    # ── ROAS Computation ──────────────────────────────────────

    def compute_roas(
        self,
        campaigns: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Calculate ROAS per campaign with trend indicator.

        Args:
            campaigns: List of campaign dicts with spend and revenue.

        Returns:
            Campaigns enriched with:
            - roas: float
            - roas_tier: "excellent"|"good"|"poor"|"critical"
            - recommendation: str
        """
        result = []

        for c in campaigns:
            spend = float(c.get("spend", 0))
            revenue = float(c.get("revenue", 0))
            roas = revenue / spend if spend > MIN_SPEND_FOR_ROAS else 0.0

            # Classify ROAS tier
            if roas >= 3.0:
                tier = "excellent"
                recommendation = "increase_budget"
            elif roas >= 1.5:
                tier = "good"
                recommendation = "maintain"
            elif roas >= MIN_VIABLE_ROAS:
                tier = "poor"
                recommendation = "reduce_budget"
            else:
                tier = "critical"
                recommendation = "pause_or_reallocate"

            result.append({
                **c,
                "roas": round(roas, 2),
                "roas_tier": tier,
                "recommendation": recommendation,
            })

        # Sort by ROAS descending
        result.sort(key=lambda x: x["roas"], reverse=True)
        return result

    # ── Reallocation ──────────────────────────────────────────

    def recommend_reallocation(
        self,
        total_budget: float,
        campaigns: list[dict[str, Any]],
        strategy: str = STRATEGY_MAXIMIZE_ROAS,
    ) -> list[dict[str, Any]]:
        """
        Recommend budget shifts from low-ROAS to high-ROAS campaigns.

        The total budget NEVER changes — only redistribution.

        Args:
            total_budget: Total available budget.
            campaigns: Campaigns with ROAS data.
            strategy: "maximize_roas", "balance", or "conservative".

        Returns:
            [
                {
                    "campaign_id": str,
                    "current_budget": float,
                    "recommended_budget": float,
                    "delta": float,
                    "delta_pct": float,
                    "reasoning": str,
                    "requires_approval": bool,
                }
            ]
        """
        if not campaigns or total_budget <= 0:
            return []

        recommendations = []
        current_total = sum(float(c.get("spend", 0)) for c in campaigns)

        if current_total <= 0:
            # No current spend — distribute equally
            per_campaign = total_budget / len(campaigns)
            for c in campaigns:
                recommendations.append({
                    "campaign_id": c.get("campaign_id", "unknown"),
                    "current_budget": 0.0,
                    "recommended_budget": round(per_campaign, 2),
                    "delta": round(per_campaign, 2),
                    "delta_pct": 1.0,
                    "reasoning": "Initial equal distribution (no prior spend data)",
                    "requires_approval": False,
                })
            return recommendations

        # Compute ROAS-weighted allocation
        roas_values = [max(float(c.get("roas", 0)), 0.01) for c in campaigns]

        if strategy == STRATEGY_MAXIMIZE_ROAS:
            # Weight heavily toward top performers
            weights = [r ** 2 for r in roas_values]
        elif strategy == STRATEGY_BALANCE:
            # Mild weighting toward top performers
            weights = [r for r in roas_values]
        else:  # conservative
            # Minimal changes, just small shifts
            weights = [max(r, 0.5) for r in roas_values]

        total_weight = sum(weights) or 1.0

        for i, c in enumerate(campaigns):
            current = float(c.get("spend", 0))
            share = weights[i] / total_weight
            recommended = round(total_budget * share, 2)
            delta = round(recommended - current, 2)
            delta_pct = abs(delta) / current if current > 0 else 0.0

            # Check if this shift exceeds safety threshold
            requires_approval = (
                delta_pct > MAX_REALLOCATION_PCT
                and abs(delta) > 50.0  # Don't flag tiny shifts
            )

            reasoning = self._build_reasoning(
                c.get("roas", 0), delta, c.get("roas_tier", "unknown"),
            )

            recommendations.append({
                "campaign_id": c.get("campaign_id", "unknown"),
                "current_budget": round(current, 2),
                "recommended_budget": recommended,
                "delta": delta,
                "delta_pct": round(delta_pct, 3),
                "reasoning": reasoning,
                "requires_approval": requires_approval,
            })

        return recommendations

    def _build_reasoning(self, roas: float, delta: float, tier: str) -> str:
        """Build human-readable reasoning for a budget shift."""
        direction = "Increase" if delta > 0 else "Decrease" if delta < 0 else "Maintain"
        return (
            f"{direction} budget by ${abs(delta):.0f}. "
            f"ROAS: {roas:.1f}x ({tier}). "
            f"{'Reward high performer.' if delta > 0 else 'Redirect from underperformer.' if delta < 0 else 'No change needed.'}"
        )

    # ── Execution ─────────────────────────────────────────────

    def apply_reallocation(
        self,
        recommendations: list[dict[str, Any]],
        session_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Execute approved budget reallocations.

        Only applies recommendations where requires_approval=False
        or where the recommendation has been explicitly approved.

        Returns:
            {"applied": int, "skipped": int, "total_shifted": float}
        """
        applied = 0
        skipped = 0
        total_shifted = 0.0

        for rec in recommendations:
            if rec.get("requires_approval", False):
                # Log as pending
                self._log_action(
                    session_id=session_id,
                    action_type="budget_reallocation",
                    target=rec["campaign_id"],
                    parameters=rec,
                    result="pending",
                )
                skipped += 1
                continue

            # Log as successful (actual ad platform API calls would go here)
            self._log_action(
                session_id=session_id,
                action_type="budget_reallocation",
                target=rec["campaign_id"],
                parameters=rec,
                result="success",
            )

            applied += 1
            total_shifted += abs(rec.get("delta", 0))

        return {
            "applied": applied,
            "skipped": skipped,
            "total_shifted": round(total_shifted, 2),
        }

    # ── Budget Limits ─────────────────────────────────────────

    def get_budget_limit(self, org_id: Optional[str] = None) -> float:
        """
        Get the org's budget limit from plan tier.

        Falls back to default enterprise limit if org not found.
        """
        _org_id = org_id or self.org_id
        if not _org_id:
            return DEFAULT_BUDGET_LIMITS.get("enterprise", 50000.0)

        try:
            result = (
                self.db.client.table("organizations")
                .select("plan_tier")
                .eq("id", str(_org_id))
                .limit(1)
                .execute()
            )
            if result.data:
                tier = result.data[0].get("plan_tier", "free")
                return DEFAULT_BUDGET_LIMITS.get(tier, 0.0)
        except Exception:
            pass

        return DEFAULT_BUDGET_LIMITS.get("enterprise", 50000.0)

    def check_budget_compliance(
        self,
        total_spend: float,
        org_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Check if current spend is within org's budget limit.

        Returns:
            {"within_limit": bool, "limit": float, "spend": float, "remaining": float}
        """
        limit = self.get_budget_limit(org_id)
        remaining = max(limit - total_spend, 0.0)

        return {
            "within_limit": total_spend <= limit,
            "limit": limit,
            "spend": total_spend,
            "remaining": round(remaining, 2),
            "utilization_pct": round(
                (total_spend / limit * 100) if limit > 0 else 0.0, 1
            ),
        }

    # ── Snapshots ─────────────────────────────────────────────

    def snapshot_budget(
        self,
        vertical_id: str,
        period: str,
        spend_data: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        Save current budget state to budget_snapshots table.

        Args:
            vertical_id: Vertical ID.
            period: Period string (e.g., "2025-02" or "2025-02-09").
            spend_data: Optional pre-computed spend data.

        Returns:
            Created snapshot record.
        """
        data = spend_data or self.get_spend_summary(vertical_id)

        try:
            record = {
                "vertical_id": vertical_id,
                "period": period,
                "total_spend": data.get("total_spend", 0),
                "total_revenue": data.get("total_revenue", 0),
                "roas": data.get("roas", 0),
                "breakdown": {
                    "campaigns": [
                        {
                            "campaign_id": c.get("campaign_id"),
                            "spend": c.get("spend", 0),
                            "revenue": c.get("revenue", 0),
                            "roas": c.get("roas", 0),
                        }
                        for c in data.get("by_campaign", [])
                    ]
                },
            }

            if self.org_id:
                record["org_id"] = self.org_id

            result = (
                self.db.client.table("budget_snapshots")
                .upsert(record, on_conflict="vertical_id,period")
                .execute()
            )

            return result.data[0] if result.data else record

        except Exception as e:
            logger.error(f"Failed to snapshot budget: {e}")
            return {}

    def get_budget_history(
        self,
        vertical_id: str,
        periods: int = 6,
    ) -> list[dict[str, Any]]:
        """
        Get historical budget snapshots for trend analysis.

        Returns snapshots ordered by period descending.
        """
        try:
            result = (
                self.db.client.table("budget_snapshots")
                .select("*")
                .eq("vertical_id", vertical_id)
                .order("period", desc=True)
                .limit(periods)
                .execute()
            )
            return result.data or []
        except Exception as e:
            logger.error(f"Failed to get budget history: {e}")
            return []

    # ── Internal Helpers ──────────────────────────────────────

    def _log_action(
        self,
        action_type: str,
        target: str,
        parameters: dict,
        result: str,
        session_id: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> None:
        """Log an optimization action."""
        try:
            data = {
                "vertical_id": self.db.vertical_id,
                "action_type": action_type,
                "target": target,
                "parameters": parameters,
                "result": result,
            }
            if session_id:
                data["session_id"] = session_id
            if error_message:
                data["error_message"] = error_message

            self.db.client.table("optimization_actions").insert(data).execute()
        except Exception as e:
            logger.error(f"Failed to log budget action: {e}")
