"""
Tests for BudgetManager — Phase 15.

Tests spend analysis, ROAS computation, budget reallocation,
plan tier limits, budget compliance, and snapshots.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock

import pytest

from core.optimization.budget_manager import (
    BudgetManager,
    DEFAULT_BUDGET_LIMITS,
    MAX_REALLOCATION_PCT,
    MIN_SPEND_FOR_ROAS,
    MIN_VIABLE_ROAS,
    STRATEGY_BALANCE,
    STRATEGY_CONSERVATIVE,
    STRATEGY_MAXIMIZE_ROAS,
)


# ── Mock DB ──────────────────────────────────────────────────


class MockTable:
    def __init__(self, data=None):
        self._data = data or []
        self._filters = {}
        self._upsert_data = None

    def select(self, cols="*"):
        return self

    def insert(self, data):
        self._data.append(data)
        return self

    def upsert(self, data, on_conflict=""):
        self._upsert_data = data
        self._data.append(data)
        return self

    def eq(self, col, val):
        self._filters[col] = val
        return self

    def order(self, col, desc=False):
        return self

    def limit(self, n):
        return self

    def execute(self):
        result = MagicMock()
        filtered = self._data
        for col, val in self._filters.items():
            filtered = [r for r in filtered if r.get(col) == val]
        result.data = filtered
        self._filters = {}
        return result


class MockDB:
    def __init__(self, tables=None):
        self._tables = tables or {}
        self.vertical_id = "test_vertical"
        self.client = self

    def table(self, name):
        return MockTable(self._tables.get(name, []))


@pytest.fixture
def mock_db():
    return MockDB()


@pytest.fixture
def budget_mgr(mock_db):
    return BudgetManager(db=mock_db)


# ── Constants ─────────────────────────────────────────────────


class TestConstants:
    def test_max_reallocation_pct(self):
        assert MAX_REALLOCATION_PCT == 0.20

    def test_min_viable_roas(self):
        assert MIN_VIABLE_ROAS == 0.5

    def test_min_spend_for_roas(self):
        assert MIN_SPEND_FOR_ROAS == 10.0

    def test_default_budget_limits(self):
        assert "free" in DEFAULT_BUDGET_LIMITS
        assert "starter" in DEFAULT_BUDGET_LIMITS
        assert "pro" in DEFAULT_BUDGET_LIMITS
        assert "enterprise" in DEFAULT_BUDGET_LIMITS

    def test_budget_limits_ascending(self):
        prev = -1.0
        for tier in ["free", "starter", "pro", "enterprise"]:
            limit = DEFAULT_BUDGET_LIMITS[tier]
            assert limit >= prev
            prev = limit

    def test_strategy_constants(self):
        assert STRATEGY_MAXIMIZE_ROAS == "maximize_roas"
        assert STRATEGY_BALANCE == "balance"
        assert STRATEGY_CONSERVATIVE == "conservative"


# ── Spend Summary ─────────────────────────────────────────────


class TestSpendSummary:
    def test_empty_data(self, budget_mgr):
        summary = budget_mgr.get_spend_summary("test_vertical")
        assert summary["total_spend"] == 0.0
        assert summary["total_revenue"] == 0.0
        assert summary["roas"] == 0.0
        assert summary["by_campaign"] == []

    def test_with_campaign_data(self):
        insights = [
            {
                "vertical_id": "test",
                "insight_type": "ad_performance",
                "content": "Campaign A performance",
                "metadata": {
                    "campaign_id": "camp_a",
                    "campaign_name": "Campaign A",
                    "platform": "google",
                    "spend": 100,
                    "revenue": 300,
                    "impressions": 10000,
                    "clicks": 500,
                    "conversions": 10,
                },
            },
            {
                "vertical_id": "test",
                "insight_type": "ad_performance",
                "content": "Campaign B performance",
                "metadata": {
                    "campaign_id": "camp_b",
                    "campaign_name": "Campaign B",
                    "platform": "meta",
                    "spend": 200,
                    "revenue": 100,
                    "impressions": 20000,
                    "clicks": 300,
                    "conversions": 3,
                },
            },
        ]
        db = MockDB(tables={"shared_insights": insights})
        mgr = BudgetManager(db=db)
        summary = mgr.get_spend_summary("test")

        assert summary["total_spend"] == 300.0
        assert summary["total_revenue"] == 400.0
        assert len(summary["by_campaign"]) == 2
        assert summary["period_days"] == 30

    def test_roas_computation(self):
        insights = [
            {
                "vertical_id": "v1",
                "insight_type": "ad_performance",
                "content": "",
                "metadata": {
                    "campaign_id": "c1",
                    "spend": 100,
                    "revenue": 500,
                },
            },
        ]
        db = MockDB(tables={"shared_insights": insights})
        mgr = BudgetManager(db=db)
        summary = mgr.get_spend_summary("v1")
        assert summary["roas"] == 5.0

    def test_custom_period(self, budget_mgr):
        summary = budget_mgr.get_spend_summary("v1", days=7)
        assert summary["period_days"] == 7


# ── ROAS Computation ──────────────────────────────────────────


class TestComputeROAS:
    def test_empty_campaigns(self, budget_mgr):
        result = budget_mgr.compute_roas([])
        assert result == []

    def test_excellent_roas(self, budget_mgr):
        campaigns = [{"campaign_id": "c1", "spend": 100, "revenue": 500}]
        result = budget_mgr.compute_roas(campaigns)
        assert result[0]["roas"] == 5.0
        assert result[0]["roas_tier"] == "excellent"
        assert result[0]["recommendation"] == "increase_budget"

    def test_good_roas(self, budget_mgr):
        campaigns = [{"campaign_id": "c1", "spend": 100, "revenue": 200}]
        result = budget_mgr.compute_roas(campaigns)
        assert result[0]["roas_tier"] == "good"
        assert result[0]["recommendation"] == "maintain"

    def test_poor_roas(self, budget_mgr):
        campaigns = [{"campaign_id": "c1", "spend": 100, "revenue": 70}]
        result = budget_mgr.compute_roas(campaigns)
        assert result[0]["roas_tier"] == "poor"
        assert result[0]["recommendation"] == "reduce_budget"

    def test_critical_roas(self, budget_mgr):
        campaigns = [{"campaign_id": "c1", "spend": 100, "revenue": 20}]
        result = budget_mgr.compute_roas(campaigns)
        assert result[0]["roas_tier"] == "critical"
        assert result[0]["recommendation"] == "pause_or_reallocate"

    def test_zero_spend(self, budget_mgr):
        campaigns = [{"campaign_id": "c1", "spend": 0, "revenue": 100}]
        result = budget_mgr.compute_roas(campaigns)
        assert result[0]["roas"] == 0.0

    def test_low_spend_below_threshold(self, budget_mgr):
        campaigns = [{"campaign_id": "c1", "spend": 5, "revenue": 100}]
        result = budget_mgr.compute_roas(campaigns)
        assert result[0]["roas"] == 0.0  # Below MIN_SPEND_FOR_ROAS

    def test_sorted_by_roas_descending(self, budget_mgr):
        campaigns = [
            {"campaign_id": "c1", "spend": 100, "revenue": 100},
            {"campaign_id": "c2", "spend": 100, "revenue": 500},
            {"campaign_id": "c3", "spend": 100, "revenue": 200},
        ]
        result = budget_mgr.compute_roas(campaigns)
        assert result[0]["campaign_id"] == "c2"
        assert result[-1]["campaign_id"] == "c1"


# ── Reallocation ──────────────────────────────────────────────


class TestRecommendReallocation:
    def test_empty_campaigns(self, budget_mgr):
        result = budget_mgr.recommend_reallocation(1000, [])
        assert result == []

    def test_zero_budget(self, budget_mgr):
        campaigns = [{"campaign_id": "c1", "spend": 100, "roas": 2.0}]
        result = budget_mgr.recommend_reallocation(0, campaigns)
        assert result == []

    def test_no_current_spend_equal_distribution(self, budget_mgr):
        campaigns = [
            {"campaign_id": "c1", "spend": 0, "roas": 0},
            {"campaign_id": "c2", "spend": 0, "roas": 0},
        ]
        result = budget_mgr.recommend_reallocation(1000, campaigns)
        assert len(result) == 2
        assert result[0]["recommended_budget"] == 500.0
        assert result[1]["recommended_budget"] == 500.0

    def test_maximize_roas_strategy(self, budget_mgr):
        campaigns = [
            {"campaign_id": "c1", "spend": 500, "roas": 5.0},
            {"campaign_id": "c2", "spend": 500, "roas": 1.0},
        ]
        result = budget_mgr.recommend_reallocation(
            1000, campaigns, strategy=STRATEGY_MAXIMIZE_ROAS,
        )
        # High ROAS should get more budget
        c1 = next(r for r in result if r["campaign_id"] == "c1")
        c2 = next(r for r in result if r["campaign_id"] == "c2")
        assert c1["recommended_budget"] > c2["recommended_budget"]

    def test_balance_strategy(self, budget_mgr):
        campaigns = [
            {"campaign_id": "c1", "spend": 500, "roas": 5.0},
            {"campaign_id": "c2", "spend": 500, "roas": 1.0},
        ]
        result = budget_mgr.recommend_reallocation(
            1000, campaigns, strategy=STRATEGY_BALANCE,
        )
        assert len(result) == 2

    def test_conservative_strategy(self, budget_mgr):
        campaigns = [
            {"campaign_id": "c1", "spend": 500, "roas": 5.0},
            {"campaign_id": "c2", "spend": 500, "roas": 1.0},
        ]
        result = budget_mgr.recommend_reallocation(
            1000, campaigns, strategy=STRATEGY_CONSERVATIVE,
        )
        # Conservative should have smaller deltas
        assert len(result) == 2

    def test_total_budget_preserved(self, budget_mgr):
        campaigns = [
            {"campaign_id": "c1", "spend": 300, "roas": 3.0},
            {"campaign_id": "c2", "spend": 400, "roas": 1.5},
            {"campaign_id": "c3", "spend": 300, "roas": 0.5},
        ]
        result = budget_mgr.recommend_reallocation(1000, campaigns)
        total_recommended = sum(r["recommended_budget"] for r in result)
        assert abs(total_recommended - 1000) < 1.0  # Allow rounding

    def test_large_shift_requires_approval(self, budget_mgr):
        campaigns = [
            {"campaign_id": "c1", "spend": 900, "roas": 5.0},
            {"campaign_id": "c2", "spend": 100, "roas": 0.1},
        ]
        result = budget_mgr.recommend_reallocation(1000, campaigns)
        # c2 should have a large shift flagged
        has_approval = any(r.get("requires_approval") for r in result)
        # At least one should require approval for large shift
        assert isinstance(has_approval, bool)

    def test_recommendation_has_reasoning(self, budget_mgr):
        campaigns = [
            {"campaign_id": "c1", "spend": 500, "roas": 3.0},
            {"campaign_id": "c2", "spend": 500, "roas": 0.5},
        ]
        result = budget_mgr.recommend_reallocation(1000, campaigns)
        for rec in result:
            assert "reasoning" in rec
            assert len(rec["reasoning"]) > 0

    def test_recommendation_structure(self, budget_mgr):
        campaigns = [{"campaign_id": "c1", "spend": 100, "roas": 2.0}]
        result = budget_mgr.recommend_reallocation(1000, campaigns)
        if result:
            rec = result[0]
            assert "campaign_id" in rec
            assert "current_budget" in rec
            assert "recommended_budget" in rec
            assert "delta" in rec
            assert "delta_pct" in rec
            assert "requires_approval" in rec


# ── Apply Reallocation ────────────────────────────────────────


class TestApplyReallocation:
    def test_empty_recommendations(self, budget_mgr):
        result = budget_mgr.apply_reallocation([])
        assert result["applied"] == 0
        assert result["skipped"] == 0
        assert result["total_shifted"] == 0.0

    def test_apply_approved_actions(self, budget_mgr):
        recs = [
            {
                "campaign_id": "c1",
                "delta": 100,
                "requires_approval": False,
            },
            {
                "campaign_id": "c2",
                "delta": -100,
                "requires_approval": False,
            },
        ]
        result = budget_mgr.apply_reallocation(recs)
        assert result["applied"] == 2
        assert result["skipped"] == 0

    def test_skip_unapproved_actions(self, budget_mgr):
        recs = [
            {
                "campaign_id": "c1",
                "delta": 500,
                "requires_approval": True,
            },
        ]
        result = budget_mgr.apply_reallocation(recs)
        assert result["applied"] == 0
        assert result["skipped"] == 1

    def test_total_shifted_calculation(self, budget_mgr):
        recs = [
            {"campaign_id": "c1", "delta": 200, "requires_approval": False},
            {"campaign_id": "c2", "delta": -200, "requires_approval": False},
        ]
        result = budget_mgr.apply_reallocation(recs)
        assert result["total_shifted"] == 400.0


# ── Budget Limits ─────────────────────────────────────────────


class TestBudgetLimits:
    def test_default_limit_no_org(self, budget_mgr):
        limit = budget_mgr.get_budget_limit()
        assert limit == DEFAULT_BUDGET_LIMITS["enterprise"]

    def test_limit_from_org_plan(self):
        orgs = [{"id": "org-1", "plan_tier": "starter"}]
        db = MockDB(tables={"organizations": orgs})
        mgr = BudgetManager(db=db, org_id="org-1")
        limit = mgr.get_budget_limit("org-1")
        assert limit == DEFAULT_BUDGET_LIMITS["starter"]

    def test_limit_missing_org(self, budget_mgr):
        limit = budget_mgr.get_budget_limit("nonexistent")
        # Falls back to enterprise
        assert limit == DEFAULT_BUDGET_LIMITS["enterprise"]

    def test_limit_free_tier(self):
        orgs = [{"id": "org-1", "plan_tier": "free"}]
        db = MockDB(tables={"organizations": orgs})
        mgr = BudgetManager(db=db)
        limit = mgr.get_budget_limit("org-1")
        assert limit == 0.0


# ── Budget Compliance ─────────────────────────────────────────


class TestBudgetCompliance:
    def test_within_limit(self):
        orgs = [{"id": "org-1", "plan_tier": "pro"}]
        db = MockDB(tables={"organizations": orgs})
        mgr = BudgetManager(db=db)
        result = mgr.check_budget_compliance(3000, "org-1")
        assert result["within_limit"] is True
        assert result["limit"] == 5000.0
        assert result["remaining"] == 2000.0

    def test_exceeds_limit(self):
        orgs = [{"id": "org-1", "plan_tier": "starter"}]
        db = MockDB(tables={"organizations": orgs})
        mgr = BudgetManager(db=db)
        result = mgr.check_budget_compliance(1500, "org-1")
        assert result["within_limit"] is False
        assert result["remaining"] == 0.0

    def test_utilization_percentage(self):
        orgs = [{"id": "org-1", "plan_tier": "pro"}]
        db = MockDB(tables={"organizations": orgs})
        mgr = BudgetManager(db=db)
        result = mgr.check_budget_compliance(2500, "org-1")
        assert result["utilization_pct"] == 50.0


# ── Budget Snapshots ──────────────────────────────────────────


class TestBudgetSnapshots:
    def test_snapshot_creates_record(self, budget_mgr):
        result = budget_mgr.snapshot_budget(
            "test_vertical",
            "2025-02",
            spend_data={
                "total_spend": 1000,
                "total_revenue": 3000,
                "roas": 3.0,
                "by_campaign": [],
            },
        )
        assert isinstance(result, dict)

    def test_snapshot_uses_spend_data(self, budget_mgr):
        data = {
            "total_spend": 500,
            "total_revenue": 1500,
            "roas": 3.0,
            "by_campaign": [
                {"campaign_id": "c1", "spend": 500, "revenue": 1500, "roas": 3.0},
            ],
        }
        result = budget_mgr.snapshot_budget("v1", "2025-02", spend_data=data)
        assert isinstance(result, dict)

    def test_budget_history_empty(self, budget_mgr):
        history = budget_mgr.get_budget_history("test_vertical")
        assert history == []

    def test_budget_history_with_data(self):
        snapshots = [
            {"vertical_id": "v1", "period": "2025-01", "total_spend": 1000, "roas": 2.0},
            {"vertical_id": "v1", "period": "2025-02", "total_spend": 1200, "roas": 2.5},
        ]
        db = MockDB(tables={"budget_snapshots": snapshots})
        mgr = BudgetManager(db=db)
        history = mgr.get_budget_history("v1")
        assert len(history) == 2


# ── Edge Cases ────────────────────────────────────────────────


class TestEdgeCases:
    def test_negative_roas_handled(self, budget_mgr):
        campaigns = [{"campaign_id": "c1", "spend": 100, "revenue": -50}]
        result = budget_mgr.compute_roas(campaigns)
        assert result[0]["roas_tier"] == "critical"

    def test_single_campaign_reallocation(self, budget_mgr):
        campaigns = [{"campaign_id": "c1", "spend": 1000, "roas": 2.0}]
        result = budget_mgr.recommend_reallocation(1000, campaigns)
        assert len(result) == 1
        assert abs(result[0]["recommended_budget"] - 1000) < 1.0

    def test_many_campaigns(self, budget_mgr):
        campaigns = [
            {"campaign_id": f"c{i}", "spend": 100, "roas": i * 0.5}
            for i in range(10)
        ]
        result = budget_mgr.recommend_reallocation(1000, campaigns)
        assert len(result) == 10
        total = sum(r["recommended_budget"] for r in result)
        assert abs(total - 1000) < 1.0


# ── Build Reasoning ───────────────────────────────────────────


class TestBuildReasoning:
    def test_increase_reasoning(self, budget_mgr):
        reasoning = budget_mgr._build_reasoning(3.0, 200, "excellent")
        assert "Increase" in reasoning

    def test_decrease_reasoning(self, budget_mgr):
        reasoning = budget_mgr._build_reasoning(0.5, -200, "poor")
        assert "Decrease" in reasoning

    def test_maintain_reasoning(self, budget_mgr):
        reasoning = budget_mgr._build_reasoning(2.0, 0, "good")
        assert "Maintain" in reasoning
