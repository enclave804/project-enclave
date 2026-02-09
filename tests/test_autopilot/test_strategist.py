"""
Tests for Strategist — Phase 15.

Tests performance scanning, opportunity detection, experiment proposals,
vertical suggestions, and strategy report generation.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import MagicMock

import pytest

from core.genesis.strategist import (
    HIGH_SUCCESS_RATE_THRESHOLD,
    IMPACT_HIGH,
    IMPACT_LOW,
    IMPACT_MEDIUM,
    LOW_SUCCESS_RATE_THRESHOLD,
    MIN_AGENT_RUNS_FOR_ANALYSIS,
    MIN_LEAD_SCORE_THRESHOLD,
    OPPORTUNITY_TYPES,
    STALE_EXPERIMENT_DAYS,
    Strategist,
)


# ── Mock DB ──────────────────────────────────────────────────


class MockTable:
    def __init__(self, data=None):
        self._data = data or []
        self._filters = {}

    def select(self, cols="*"):
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
def strategist(mock_db):
    return Strategist(db=mock_db)


# ── Constants ─────────────────────────────────────────────────


class TestConstants:
    def test_opportunity_types(self):
        assert isinstance(OPPORTUNITY_TYPES, set)
        assert "underperforming_agent" in OPPORTUNITY_TYPES
        assert "budget_inefficiency" in OPPORTUNITY_TYPES
        assert "stale_experiment" in OPPORTUNITY_TYPES

    def test_thresholds(self):
        assert LOW_SUCCESS_RATE_THRESHOLD == 0.5
        assert HIGH_SUCCESS_RATE_THRESHOLD == 0.85
        assert STALE_EXPERIMENT_DAYS == 14
        assert MIN_AGENT_RUNS_FOR_ANALYSIS == 5
        assert MIN_LEAD_SCORE_THRESHOLD == 40.0

    def test_impact_levels(self):
        assert IMPACT_HIGH == "high"
        assert IMPACT_MEDIUM == "medium"
        assert IMPACT_LOW == "low"


# ── Performance Scanning ──────────────────────────────────────


class TestScanPerformance:
    def test_empty_scan(self, strategist):
        result = strategist.scan_performance("test_vertical")
        assert result["vertical_id"] == "test_vertical"
        assert result["agents"] == []
        assert result["experiments"] == []
        assert result["insights_count"] == 0
        assert "scanned_at" in result

    def test_scan_with_agent_data(self):
        runs = [
            {"agent_id": "outreach_v1", "vertical_id": "v1", "status": "completed", "duration_ms": 1000},
            {"agent_id": "outreach_v1", "vertical_id": "v1", "status": "completed", "duration_ms": 2000},
            {"agent_id": "outreach_v1", "vertical_id": "v1", "status": "failed", "duration_ms": 500, "error_message": "timeout"},
        ]
        db = MockDB(tables={"agent_runs": runs})
        s = Strategist(db=db)
        result = s.scan_performance("v1")
        assert len(result["agents"]) == 1
        agent = result["agents"][0]
        assert agent["total_runs"] == 3
        assert agent["successful_runs"] == 2

    def test_scan_period_days(self, strategist):
        result = strategist.scan_performance("v1", days=7)
        assert result["period_days"] == 7

    def test_scan_includes_budget(self):
        snapshots = [
            {"vertical_id": "v1", "total_spend": 1000, "total_revenue": 3000, "roas": 3.0},
        ]
        db = MockDB(tables={"budget_snapshots": snapshots})
        s = Strategist(db=db)
        result = s.scan_performance("v1")
        assert result["budget_summary"]["roas"] == 3.0

    def test_scan_includes_lead_stats(self):
        companies = [
            {"vertical_id": "v1", "lead_score": 80, "status": "qualified"},
            {"vertical_id": "v1", "lead_score": 30, "status": "new"},
            {"vertical_id": "v1", "lead_score": 60, "status": "qualified"},
        ]
        db = MockDB(tables={"companies": companies})
        s = Strategist(db=db)
        result = s.scan_performance("v1")
        assert result["lead_stats"]["total"] == 3

    def test_scan_includes_insights_count(self):
        insights = [
            {"vertical_id": "v1", "insight_type": "email_performance"},
            {"vertical_id": "v1", "insight_type": "ad_performance"},
            {"vertical_id": "v1", "insight_type": "email_performance"},
        ]
        db = MockDB(tables={"shared_insights": insights})
        s = Strategist(db=db)
        result = s.scan_performance("v1")
        assert result["insights_count"] == 3
        assert result["insight_categories"]["email_performance"] == 2


# ── Opportunity Detection ─────────────────────────────────────


class TestDetectOpportunities:
    def test_empty_performance(self, strategist):
        opps = strategist.detect_opportunities({})
        assert opps == []

    def test_detect_underperforming_agent(self, strategist):
        perf = {
            "agents": [
                {"agent_id": "bad_agent", "total_runs": 20, "success_rate": 0.3},
            ],
            "budget_summary": {},
            "experiments": [],
            "lead_stats": {},
            "insights_count": 0,
            "insight_categories": {},
        }
        opps = strategist.detect_opportunities(perf)
        types = [o["type"] for o in opps]
        assert "underperforming_agent" in types

    def test_detect_high_performer(self, strategist):
        perf = {
            "agents": [
                {"agent_id": "star_agent", "total_runs": 50, "success_rate": 0.95},
            ],
            "budget_summary": {},
            "experiments": [],
            "lead_stats": {},
            "insights_count": 0,
            "insight_categories": {},
        }
        opps = strategist.detect_opportunities(perf)
        types = [o["type"] for o in opps]
        assert "high_performing_agent" in types

    def test_skip_low_run_agents(self, strategist):
        perf = {
            "agents": [
                {"agent_id": "new_agent", "total_runs": 2, "success_rate": 0.1},
            ],
            "budget_summary": {},
            "experiments": [],
            "lead_stats": {},
            "insights_count": 0,
            "insight_categories": {},
        }
        opps = strategist.detect_opportunities(perf)
        assert len(opps) == 0  # Not enough data

    def test_detect_budget_inefficiency(self, strategist):
        perf = {
            "agents": [],
            "budget_summary": {"roas": 0.5, "total_spend": 5000, "total_revenue": 2500},
            "experiments": [],
            "lead_stats": {},
            "insights_count": 0,
            "insight_categories": {},
        }
        opps = strategist.detect_opportunities(perf)
        types = [o["type"] for o in opps]
        assert "budget_inefficiency" in types

    def test_detect_stale_experiment(self, strategist):
        old_date = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        perf = {
            "agents": [],
            "budget_summary": {},
            "experiments": [
                {"experiment_id": "e1", "name": "Old Test", "status": "active", "created_at": old_date},
            ],
            "lead_stats": {},
            "insights_count": 0,
            "insight_categories": {},
        }
        opps = strategist.detect_opportunities(perf)
        types = [o["type"] for o in opps]
        assert "stale_experiment" in types

    def test_detect_audience_mismatch(self, strategist):
        perf = {
            "agents": [],
            "budget_summary": {},
            "experiments": [],
            "lead_stats": {"total": 100, "qualified_pct": 0.15, "avg_score": 30},
            "insights_count": 0,
            "insight_categories": {},
        }
        opps = strategist.detect_opportunities(perf)
        types = [o["type"] for o in opps]
        assert "audience_mismatch" in types

    def test_detect_content_gap(self, strategist):
        perf = {
            "agents": [],
            "budget_summary": {},
            "experiments": [],
            "lead_stats": {},
            "insights_count": 20,
            "insight_categories": {"email_performance": 10, "deal_patterns": 10},
        }
        opps = strategist.detect_opportunities(perf)
        types = [o["type"] for o in opps]
        assert "content_gap" in types

    def test_opportunities_sorted_by_impact(self, strategist):
        perf = {
            "agents": [
                {"agent_id": "a1", "total_runs": 30, "success_rate": 0.2},
                {"agent_id": "a2", "total_runs": 50, "success_rate": 0.95},
            ],
            "budget_summary": {"roas": 0.5, "total_spend": 1000, "total_revenue": 500},
            "experiments": [],
            "lead_stats": {},
            "insights_count": 0,
            "insight_categories": {},
        }
        opps = strategist.detect_opportunities(perf)
        if len(opps) >= 2:
            # High impact should come first
            impact_order = {"high": 3, "medium": 2, "low": 1}
            for i in range(len(opps) - 1):
                curr = impact_order.get(opps[i]["potential_impact"], 0)
                next_ = impact_order.get(opps[i + 1]["potential_impact"], 0)
                assert curr >= next_

    def test_opportunity_structure(self, strategist):
        perf = {
            "agents": [{"agent_id": "a1", "total_runs": 20, "success_rate": 0.3}],
            "budget_summary": {},
            "experiments": [],
            "lead_stats": {},
            "insights_count": 0,
            "insight_categories": {},
        }
        opps = strategist.detect_opportunities(perf)
        if opps:
            opp = opps[0]
            assert "type" in opp
            assert "description" in opp
            assert "potential_impact" in opp
            assert "confidence" in opp
            assert "suggested_action" in opp
            assert "details" in opp


# ── Experiment Proposals ──────────────────────────────────────


class TestProposeExperiments:
    def test_empty_opportunities(self, strategist):
        proposals = strategist.propose_experiments([])
        assert proposals == []

    def test_propose_from_underperformer(self, strategist):
        opps = [{
            "type": "underperforming_agent",
            "description": "Agent low success",
            "potential_impact": "high",
            "confidence": 0.8,
            "suggested_action": "Optimize config",
            "details": {"agent_id": "test_agent", "success_rate": 0.3},
        }]
        proposals = strategist.propose_experiments(opps)
        assert len(proposals) == 1
        assert proposals[0]["agent_id"] == "test_agent"
        assert proposals[0]["metric"] == "success_rate"
        assert len(proposals[0]["variants"]) == 2

    def test_propose_from_budget_inefficiency(self, strategist):
        opps = [{
            "type": "budget_inefficiency",
            "description": "Low ROAS",
            "potential_impact": "high",
            "confidence": 0.85,
            "suggested_action": "Reallocate",
            "details": {"roas": 0.5},
        }]
        proposals = strategist.propose_experiments(opps)
        assert len(proposals) == 1
        assert proposals[0]["metric"] == "roas"

    def test_propose_from_audience_mismatch(self, strategist):
        opps = [{
            "type": "audience_mismatch",
            "description": "Low qualified rate",
            "potential_impact": "medium",
            "confidence": 0.75,
            "suggested_action": "Refine ICP",
            "details": {"qualified_pct": 0.15},
        }]
        proposals = strategist.propose_experiments(opps)
        assert len(proposals) == 1
        assert proposals[0]["metric"] == "qualified_rate"

    def test_experiment_has_metadata(self, strategist):
        opps = [{
            "type": "underperforming_agent",
            "description": "Low success",
            "potential_impact": "high",
            "confidence": 0.8,
            "suggested_action": "Fix",
            "details": {"agent_id": "a1", "success_rate": 0.3},
        }]
        proposals = strategist.propose_experiments(opps)
        if proposals:
            assert "metadata" in proposals[0]
            assert proposals[0]["metadata"].get("source") == "strategist_auto"

    def test_unrecognized_type_skipped(self, strategist):
        opps = [{
            "type": "something_unknown",
            "description": "Unknown",
            "potential_impact": "low",
            "confidence": 0.5,
            "suggested_action": "Nothing",
            "details": {},
        }]
        proposals = strategist.propose_experiments(opps)
        assert len(proposals) == 0


# ── Vertical Proposals ────────────────────────────────────────


class TestProposeVertical:
    def test_no_trends(self, strategist):
        result = strategist.propose_vertical([], ["enclave_guard"])
        assert result is None

    def test_propose_new_vertical(self, strategist):
        trends = [{
            "industry": "healthtech",
            "signal_strength": 0.9,
            "growth_rate": 0.3,
            "target_company_size": "50-200",
            "decision_maker_title": "CTO",
            "pain_points": ["HIPAA compliance", "data security"],
        }]
        result = strategist.propose_vertical(trends, ["enclave_guard"])
        assert result is not None
        assert result["name"] == "healthtech"
        assert result["confidence"] > 0

    def test_skip_existing_vertical(self, strategist):
        trends = [{
            "industry": "cybersecurity",
            "signal_strength": 0.9,
            "growth_rate": 0.5,
        }]
        result = strategist.propose_vertical(
            trends, ["cybersecurity"],
        )
        assert result is None  # Already covered

    def test_weak_trend_rejected(self, strategist):
        trends = [{
            "industry": "niche_market",
            "signal_strength": 0.1,
            "growth_rate": 0.01,
        }]
        result = strategist.propose_vertical(trends, [])
        # Weak signal should not produce a proposal
        assert result is None

    def test_proposal_has_icp(self, strategist):
        trends = [{
            "industry": "fintech",
            "signal_strength": 0.8,
            "growth_rate": 0.25,
            "target_company_size": "100-500",
            "decision_maker_title": "CFO",
            "pain_points": ["compliance", "reporting"],
        }]
        result = strategist.propose_vertical(trends, [])
        if result:
            assert "icp" in result
            assert "confidence" in result

    def test_multiple_trends_picks_best(self, strategist):
        trends = [
            {"industry": "weak", "signal_strength": 0.2, "growth_rate": 0.05},
            {"industry": "strong", "signal_strength": 0.95, "growth_rate": 0.4},
            {"industry": "medium", "signal_strength": 0.5, "growth_rate": 0.15},
        ]
        result = strategist.propose_vertical(trends, [])
        if result:
            assert result["name"] == "strong"


# ── Strategy Report ───────────────────────────────────────────


class TestGenerateStrategyReport:
    def test_empty_report(self, strategist):
        report = strategist.generate_strategy_report({}, [], [])
        assert "Strategy Report" in report
        assert "unknown" in report

    def test_report_with_agents(self, strategist):
        perf = {
            "vertical_id": "enclave_guard",
            "period_days": 30,
            "agents": [
                {"agent_id": "a1", "total_runs": 50, "success_rate": 0.9},
                {"agent_id": "a2", "total_runs": 30, "success_rate": 0.6},
            ],
            "budget_summary": {"total_spend": 1000, "total_revenue": 3000, "roas": 3.0},
            "lead_stats": {"total": 100, "avg_score": 65, "qualified_pct": 0.7},
        }
        report = strategist.generate_strategy_report(perf, [], [])
        assert "enclave_guard" in report
        assert "Active Agents" in report

    def test_report_with_opportunities(self, strategist):
        opps = [{
            "type": "underperforming_agent",
            "description": "Agent X is failing",
            "potential_impact": "high",
            "confidence": 0.8,
            "suggested_action": "Fix it",
        }]
        report = strategist.generate_strategy_report({}, opps, [])
        assert "Opportunities" in report

    def test_report_with_experiments(self, strategist):
        exps = [{
            "name": "Subject Line Test",
            "variants": ["Emoji", "No Emoji"],
            "metric": "reply_rate",
            "agent_id": "outreach",
        }]
        report = strategist.generate_strategy_report({}, [], exps)
        assert "Proposed Experiments" in report

    def test_report_has_next_steps(self, strategist):
        report = strategist.generate_strategy_report({}, [], [])
        assert "Next Steps" in report

    def test_report_is_markdown(self, strategist):
        report = strategist.generate_strategy_report({}, [], [])
        assert report.startswith("#")
        assert "---" in report


# ── Market Trends ─────────────────────────────────────────────


class TestMarketTrends:
    def test_empty_trends(self, strategist):
        trends = strategist.get_market_trends("v1")
        assert trends == []

    def test_trends_from_insights(self):
        insights = [
            {
                "vertical_id": "v1",
                "insight_type": "market_signal",
                "content": "Rising demand in healthtech",
                "metadata": {
                    "industry": "healthtech",
                    "signal_strength": 0.8,
                    "growth_rate": 0.25,
                },
                "created_at": "2025-02-01",
            },
        ]
        db = MockDB(tables={"shared_insights": insights})
        s = Strategist(db=db)
        trends = s.get_market_trends("v1")
        assert len(trends) == 1
        assert trends[0]["industry"] == "healthtech"


# ── Repr ──────────────────────────────────────────────────────


class TestRepr:
    def test_repr(self, strategist):
        r = repr(strategist)
        assert "Strategist" in r
        assert "db=True" in r
