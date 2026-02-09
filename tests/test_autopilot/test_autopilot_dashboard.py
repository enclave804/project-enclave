"""
Tests for Autopilot Dashboard helpers and migration schema — Phase 15.

Tests compute functions, formatting helpers, and migration SQL structure.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from dashboard.pages._autopilot_helpers import (
    compute_autopilot_stats,
    compute_budget_overview,
    compute_healing_log,
    compute_strategy_pipeline,
    format_action_type,
    format_health_score,
    format_roas,
    format_session_status,
    format_session_type,
)


# ── Autopilot Stats ──────────────────────────────────────────


class TestComputeAutopilotStats:
    def test_empty_data(self):
        stats = compute_autopilot_stats([], [])
        assert stats["total_sessions"] == 0
        assert stats["total_actions"] == 0
        assert stats["last_run"] is None
        assert stats["avg_actions_per_session"] == 0.0

    def test_with_sessions(self):
        sessions = [
            {"status": "completed", "created_at": "2025-02-01T00:00:00", "completed_at": "2025-02-01T01:00:00"},
            {"status": "completed", "created_at": "2025-02-02T00:00:00", "completed_at": "2025-02-02T01:00:00"},
            {"status": "failed", "created_at": "2025-02-03T00:00:00", "completed_at": None},
        ]
        stats = compute_autopilot_stats(sessions, [])
        assert stats["total_sessions"] == 3
        assert stats["completed_sessions"] == 2
        assert stats["failed_sessions"] == 1

    def test_with_actions(self):
        actions = [
            {"result": "success"},
            {"result": "success"},
            {"result": "pending"},
            {"result": "failed"},
        ]
        stats = compute_autopilot_stats([], actions)
        assert stats["total_actions"] == 4
        assert stats["successful_actions"] == 2
        assert stats["pending_actions"] == 1

    def test_last_run(self):
        sessions = [
            {"status": "completed", "created_at": "2025-02-01", "completed_at": "2025-02-01T12:00:00"},
            {"status": "completed", "created_at": "2025-02-05", "completed_at": "2025-02-05T12:00:00"},
        ]
        stats = compute_autopilot_stats(sessions, [])
        assert stats["last_run"] == "2025-02-05T12:00:00"

    def test_avg_actions_per_session(self):
        sessions = [{"status": "completed", "created_at": "2025-02-01"}] * 4
        actions = [{"result": "success"}] * 12
        stats = compute_autopilot_stats(sessions, actions)
        assert stats["avg_actions_per_session"] == 3.0


# ── Healing Log ───────────────────────────────────────────────


class TestComputeHealingLog:
    def test_empty_actions(self):
        log = compute_healing_log([])
        assert log == []

    def test_filters_healing_types(self):
        actions = [
            {"action_type": "config_fix", "target": "a1", "result": "success", "created_at": "2025-02-01", "parameters": {}},
            {"action_type": "budget_reallocation", "target": "c1", "result": "success", "created_at": "2025-02-01", "parameters": {}},
            {"action_type": "agent_restart", "target": "a2", "result": "failed", "created_at": "2025-02-02", "parameters": {}},
        ]
        log = compute_healing_log(actions)
        assert len(log) == 2  # Only config_fix and agent_restart

    def test_log_entry_structure(self):
        actions = [
            {
                "action_type": "config_fix",
                "target": "outreach_v1",
                "result": "success",
                "created_at": "2025-02-01T10:00:00",
                "parameters": {"reasoning": "Fixed rate limit config"},
            },
        ]
        log = compute_healing_log(actions)
        assert len(log) == 1
        entry = log[0]
        assert entry["agent_id"] == "outreach_v1"
        assert entry["action_type"] == "config_fix"
        assert entry["result"] == "success"
        assert entry["result_color"] == "#10B981"

    def test_result_colors(self):
        actions = [
            {"action_type": "config_fix", "target": "a1", "result": "success", "created_at": "", "parameters": {}},
            {"action_type": "config_fix", "target": "a2", "result": "failed", "created_at": "", "parameters": {}},
            {"action_type": "config_fix", "target": "a3", "result": "pending", "created_at": "", "parameters": {}},
            {"action_type": "config_fix", "target": "a4", "result": "rejected", "created_at": "", "parameters": {}},
        ]
        log = compute_healing_log(actions)
        colors = {entry["result"]: entry["result_color"] for entry in log}
        assert colors["success"] == "#10B981"
        assert colors["failed"] == "#EF4444"
        assert colors["pending"] == "#F59E0B"
        assert colors["rejected"] == "#6B7280"


# ── Budget Overview ───────────────────────────────────────────


class TestComputeBudgetOverview:
    def test_empty_data(self):
        overview = compute_budget_overview([], [])
        assert overview["latest_spend"] == 0.0
        assert overview["latest_revenue"] == 0.0
        assert overview["latest_roas"] == 0.0
        assert overview["spend_trend"] == []
        assert overview["reallocation_count"] == 0

    def test_with_snapshots(self):
        snapshots = [
            {"period": "2025-02", "total_spend": 2000, "total_revenue": 6000, "roas": 3.0},
            {"period": "2025-01", "total_spend": 1500, "total_revenue": 3000, "roas": 2.0},
        ]
        overview = compute_budget_overview(snapshots, [])
        assert overview["latest_spend"] == 2000.0
        assert overview["latest_roas"] == 3.0
        assert len(overview["spend_trend"]) == 2

    def test_trend_oldest_first(self):
        snapshots = [
            {"period": "2025-02", "total_spend": 2000, "total_revenue": 6000, "roas": 3.0},
            {"period": "2025-01", "total_spend": 1500, "total_revenue": 3000, "roas": 2.0},
        ]
        overview = compute_budget_overview(snapshots, [])
        # Trend should be oldest first for charting
        assert overview["spend_trend"][0]["period"] == "2025-01"
        assert overview["spend_trend"][1]["period"] == "2025-02"

    def test_reallocation_stats(self):
        actions = [
            {"action_type": "budget_reallocation", "parameters": {"delta": 200}},
            {"action_type": "budget_reallocation", "parameters": {"delta": -100}},
            {"action_type": "config_fix", "parameters": {}},
        ]
        overview = compute_budget_overview([], actions)
        assert overview["reallocation_count"] == 2
        assert overview["total_shifted"] == 300.0


# ── Strategy Pipeline ─────────────────────────────────────────


class TestComputeStrategyPipeline:
    def test_empty_data(self):
        pipeline = compute_strategy_pipeline([], [])
        assert pipeline["pending_proposals"] == []
        assert pipeline["completed_strategies"] == 0
        assert pipeline["actions_by_type"] == {}

    def test_pending_proposals(self):
        sessions = [
            {
                "id": "s1",
                "status": "running",
                "session_type": "full_analysis",
                "started_at": "2025-02-09T06:00:00",
                "strategy_output": {"summary": "Analyzing..."},
            },
        ]
        pipeline = compute_strategy_pipeline(sessions, [])
        assert len(pipeline["pending_proposals"]) == 1

    def test_completed_strategies(self):
        sessions = [
            {"id": "s1", "status": "completed", "session_type": "full_analysis", "strategy_output": {}},
            {"id": "s2", "status": "completed", "session_type": "healing", "strategy_output": {}},
            {"id": "s3", "status": "completed", "session_type": "full_analysis", "strategy_output": {}},
        ]
        pipeline = compute_strategy_pipeline(sessions, [])
        assert pipeline["completed_strategies"] == 2  # Only full_analysis

    def test_actions_by_type(self):
        actions = [
            {"action_type": "config_fix"},
            {"action_type": "config_fix"},
            {"action_type": "budget_reallocation"},
            {"action_type": "experiment_launched"},
        ]
        pipeline = compute_strategy_pipeline([], actions)
        assert pipeline["actions_by_type"]["config_fix"] == 2
        assert pipeline["actions_by_type"]["budget_reallocation"] == 1


# ── Format Helpers ────────────────────────────────────────────


class TestFormatHealthScore:
    def test_healthy(self):
        label, color = format_health_score(0.9)
        assert label == "Healthy"
        assert color == "#10B981"

    def test_good(self):
        label, color = format_health_score(0.7)
        assert label == "Good"

    def test_degraded(self):
        label, color = format_health_score(0.5)
        assert label == "Degraded"

    def test_critical(self):
        label, color = format_health_score(0.3)
        assert label == "Critical"

    def test_down(self):
        label, color = format_health_score(0.1)
        assert label == "Down"

    def test_zero(self):
        label, color = format_health_score(0.0)
        assert label == "Down"

    def test_perfect(self):
        label, color = format_health_score(1.0)
        assert label == "Healthy"


class TestFormatROAS:
    def test_excellent(self):
        text, color = format_roas(5.0)
        assert text == "5.0x"
        assert color == "#10B981"

    def test_good(self):
        text, color = format_roas(2.0)
        assert text == "2.0x"
        assert color == "#3B82F6"

    def test_break_even(self):
        text, color = format_roas(1.2)
        assert text == "1.2x"
        assert color == "#F59E0B"

    def test_losing(self):
        text, color = format_roas(0.5)
        assert text == "0.5x"
        assert color == "#EF4444"

    def test_zero(self):
        text, color = format_roas(0.0)
        assert text == "0.0x"
        assert color == "#EF4444"


class TestFormatActionType:
    def test_config_fix(self):
        label, color = format_action_type("config_fix")
        assert label == "Config Fix"

    def test_budget_reallocation(self):
        label, color = format_action_type("budget_reallocation")
        assert label == "Budget Shift"

    def test_unknown_type(self):
        label, color = format_action_type("something_new")
        assert isinstance(label, str)
        assert isinstance(color, str)


class TestFormatSessionType:
    def test_full_analysis(self):
        label, color = format_session_type("full_analysis")
        assert label == "Full Analysis"

    def test_healing(self):
        label, color = format_session_type("healing")
        assert label == "Healing"

    def test_unknown(self):
        label, color = format_session_type("weird")
        assert isinstance(label, str)


class TestFormatSessionStatus:
    def test_running(self):
        label, color = format_session_status("running")
        assert label == "Running"

    def test_completed(self):
        label, color = format_session_status("completed")
        assert label == "Completed"
        assert color == "#10B981"

    def test_failed(self):
        label, color = format_session_status("failed")
        assert label == "Failed"
        assert color == "#EF4444"


# ── Migration Schema ─────────────────────────────────────────


class TestMigrationSchema:
    @pytest.fixture
    def migration_sql(self):
        migration_path = Path(__file__).parent.parent.parent / "infrastructure" / "migrations" / "012_autopilot.sql"
        return migration_path.read_text()

    def test_autopilot_sessions_table(self, migration_sql):
        assert "CREATE TABLE" in migration_sql
        assert "autopilot_sessions" in migration_sql

    def test_sessions_columns(self, migration_sql):
        for col in ["vertical_id", "session_type", "status", "metrics_snapshot",
                     "detected_issues", "strategy_output", "actions_taken"]:
            assert col in migration_sql

    def test_sessions_type_constraint(self, migration_sql):
        assert "full_analysis" in migration_sql
        assert "healing" in migration_sql
        assert "budget" in migration_sql
        assert "strategy" in migration_sql

    def test_sessions_status_constraint(self, migration_sql):
        assert "running" in migration_sql
        assert "completed" in migration_sql
        assert "failed" in migration_sql
        assert "cancelled" in migration_sql

    def test_optimization_actions_table(self, migration_sql):
        assert "optimization_actions" in migration_sql

    def test_actions_columns(self, migration_sql):
        for col in ["session_id", "vertical_id", "action_type", "target",
                     "parameters", "result", "approved_by"]:
            assert col in migration_sql

    def test_actions_type_constraint(self, migration_sql):
        for action in ["config_fix", "agent_restart", "agent_disable",
                       "budget_reallocation", "experiment_launched"]:
            assert action in migration_sql

    def test_budget_snapshots_table(self, migration_sql):
        assert "budget_snapshots" in migration_sql

    def test_budget_snapshots_columns(self, migration_sql):
        for col in ["vertical_id", "org_id", "period", "total_spend",
                     "total_revenue", "roas", "breakdown"]:
            assert col in migration_sql

    def test_indexes_exist(self, migration_sql):
        assert "CREATE INDEX" in migration_sql
        assert "idx_autopilot_sessions" in migration_sql or "idx_" in migration_sql

    def test_rls_policies(self, migration_sql):
        assert "ROW LEVEL SECURITY" in migration_sql or "ENABLE" in migration_sql

    def test_rpc_functions(self, migration_sql):
        assert "get_autopilot_stats" in migration_sql
        assert "get_action_breakdown" in migration_sql


# ── Import Tests ──────────────────────────────────────────────


class TestImports:
    def test_healer_importable(self):
        from core.optimization.healer import SelfHealer
        assert SelfHealer is not None

    def test_budget_manager_importable(self):
        from core.optimization.budget_manager import BudgetManager
        assert BudgetManager is not None

    def test_strategist_importable(self):
        from core.genesis.strategist import Strategist
        assert Strategist is not None

    def test_autopilot_agent_importable(self):
        from core.agents.implementations.autopilot_agent import AutopilotAgent
        assert AutopilotAgent is not None

    def test_autopilot_state_importable(self):
        from core.agents.state import AutopilotAgentState
        assert AutopilotAgentState is not None

    def test_helpers_importable(self):
        from dashboard.pages._autopilot_helpers import (
            compute_autopilot_stats,
            compute_healing_log,
            compute_budget_overview,
            compute_strategy_pipeline,
            format_health_score,
            format_roas,
        )
        assert callable(compute_autopilot_stats)
        assert callable(format_health_score)
        assert callable(format_roas)


# ── YAML Config ───────────────────────────────────────────────


class TestYAMLConfig:
    @pytest.fixture
    def config_yaml(self):
        config_path = Path(__file__).parent.parent.parent / "verticals" / "enclave_guard" / "agents" / "autopilot.yaml"
        return config_path.read_text()

    def test_agent_id(self, config_yaml):
        assert "autopilot_v1" in config_yaml

    def test_agent_type(self, config_yaml):
        assert "agent_type: autopilot" in config_yaml

    def test_human_gates_enabled(self, config_yaml):
        assert "enabled: true" in config_yaml
        assert "human_review" in config_yaml

    def test_auto_approve_zero(self, config_yaml):
        assert "auto_approve_threshold: 0.0" in config_yaml

    def test_schedule(self, config_yaml):
        assert "cron:" in config_yaml

    def test_params(self, config_yaml):
        assert "analysis_period_days" in config_yaml
        assert "max_budget_reallocation_pct" in config_yaml
        assert "healing_auto_apply: false" in config_yaml
