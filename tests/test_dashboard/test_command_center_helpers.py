"""
Tests for dashboard/pages/_command_center_helpers.py

Pure-function tests — no streamlit dependency.
"""

from __future__ import annotations

import os

import pytest

from dashboard.pages._command_center_helpers import (
    compute_agent_status,
    compute_fleet_summary,
    compute_health_status,
    compute_pipeline_value,
    compute_success_rate,
    format_credential_status,
    format_run_description,
    group_content_by_status,
)


# ---------------------------------------------------------------------------
# compute_agent_status
# ---------------------------------------------------------------------------


class TestComputeAgentStatus:
    """Determine status string from an agent record dict."""

    def test_active_agent(self):
        agent = {"enabled": True, "shadow_mode": False, "config": {}}
        assert compute_agent_status(agent) == "active"

    def test_paused_agent(self):
        agent = {"enabled": False}
        assert compute_agent_status(agent) == "paused"

    def test_shadow_agent(self):
        agent = {"enabled": True, "shadow_mode": True}
        assert compute_agent_status(agent) == "shadow"

    def test_circuit_breaker_tripped(self):
        agent = {
            "enabled": True,
            "shadow_mode": False,
            "config": {"consecutive_errors": 5, "max_consecutive_errors": 5},
        }
        assert compute_agent_status(agent) == "circuit_breaker"

    def test_circuit_breaker_exceeded(self):
        agent = {
            "enabled": True,
            "config": {"consecutive_errors": 10, "max_consecutive_errors": 3},
        }
        assert compute_agent_status(agent) == "circuit_breaker"

    def test_errors_below_threshold(self):
        agent = {
            "enabled": True,
            "config": {"consecutive_errors": 2, "max_consecutive_errors": 5},
        }
        assert compute_agent_status(agent) == "active"

    def test_paused_takes_priority_over_shadow(self):
        agent = {"enabled": False, "shadow_mode": True}
        assert compute_agent_status(agent) == "paused"

    def test_shadow_takes_priority_over_circuit_breaker(self):
        agent = {
            "enabled": True,
            "shadow_mode": True,
            "config": {"consecutive_errors": 10, "max_consecutive_errors": 3},
        }
        assert compute_agent_status(agent) == "shadow"

    def test_defaults_when_fields_missing(self):
        assert compute_agent_status({}) == "active"

    def test_none_config_handled(self):
        agent = {"enabled": True, "config": None}
        assert compute_agent_status(agent) == "active"


# ---------------------------------------------------------------------------
# compute_fleet_summary
# ---------------------------------------------------------------------------


class TestComputeFleetSummary:
    """Aggregate agent statuses into fleet counts."""

    def test_empty_fleet(self):
        result = compute_fleet_summary([])
        assert result == {
            "active": 0, "paused": 0, "shadow": 0, "tripped": 0, "total": 0,
        }

    def test_all_active(self):
        agents = [
            {"enabled": True, "config": {}},
            {"enabled": True, "config": {}},
            {"enabled": True, "config": {}},
        ]
        result = compute_fleet_summary(agents)
        assert result["active"] == 3
        assert result["total"] == 3

    def test_mixed_fleet(self):
        agents = [
            {"enabled": True, "config": {}},                         # active
            {"enabled": False},                                       # paused
            {"enabled": True, "shadow_mode": True},                   # shadow
            {"enabled": True, "config": {
                "consecutive_errors": 5, "max_consecutive_errors": 5,
            }},                                                       # tripped
        ]
        result = compute_fleet_summary(agents)
        assert result["active"] == 1
        assert result["paused"] == 1
        assert result["shadow"] == 1
        assert result["tripped"] == 1
        assert result["total"] == 4

    def test_total_matches_input(self):
        agents = [{"enabled": True}] * 7
        result = compute_fleet_summary(agents)
        assert result["total"] == 7


# ---------------------------------------------------------------------------
# compute_health_status
# ---------------------------------------------------------------------------


class TestComputeHealthStatus:
    """System health classification."""

    def test_operational_all_active(self):
        agents = [{"enabled": True, "config": {}}]
        assert compute_health_status(agents) == "operational"

    def test_degraded_when_paused(self):
        agents = [
            {"enabled": True, "config": {}},
            {"enabled": False},
        ]
        assert compute_health_status(agents) == "degraded"

    def test_degraded_when_tripped(self):
        agents = [
            {"enabled": True, "config": {
                "consecutive_errors": 5, "max_consecutive_errors": 5,
            }},
        ]
        assert compute_health_status(agents) == "degraded"

    def test_degraded_when_many_failed_tasks(self):
        agents = [{"enabled": True, "config": {}}]
        assert compute_health_status(agents, failed_tasks=6) == "degraded"

    def test_operational_with_few_failed_tasks(self):
        agents = [{"enabled": True, "config": {}}]
        assert compute_health_status(agents, failed_tasks=5) == "operational"

    def test_operational_on_empty_fleet(self):
        assert compute_health_status([]) == "operational"

    def test_operational_with_shadow_only(self):
        """Shadow agents alone don't degrade health (they're intentional)."""
        agents = [{"enabled": True, "shadow_mode": True}]
        assert compute_health_status(agents) == "operational"


# ---------------------------------------------------------------------------
# compute_success_rate
# ---------------------------------------------------------------------------


class TestComputeSuccessRate:
    """Overall success rate from agent stats."""

    def test_all_successful(self):
        stats = [{"total_runs": 10, "success_runs": 10}]
        assert compute_success_rate(stats) == 100.0

    def test_partial_success(self):
        stats = [{"total_runs": 10, "success_runs": 7}]
        assert compute_success_rate(stats) == pytest.approx(70.0)

    def test_zero_runs(self):
        assert compute_success_rate([]) == 0.0
        assert compute_success_rate([{"total_runs": 0, "success_runs": 0}]) == 0.0

    def test_multiple_agents(self):
        stats = [
            {"total_runs": 10, "success_runs": 8},
            {"total_runs": 10, "success_runs": 6},
        ]
        # 14/20 = 70%
        assert compute_success_rate(stats) == pytest.approx(70.0)

    def test_missing_fields_default_zero(self):
        stats = [{}]
        assert compute_success_rate(stats) == 0.0


# ---------------------------------------------------------------------------
# format_run_description
# ---------------------------------------------------------------------------


class TestFormatRunDescription:
    """Human-readable run descriptions."""

    def test_completed_with_duration(self):
        run = {"status": "completed", "duration_ms": 450}
        assert format_run_description(run) == "Completed in 450ms"

    def test_failed_with_error(self):
        run = {"status": "failed", "error_message": "Connection timeout"}
        assert format_run_description(run) == "Failed: Connection timeout"

    def test_failed_no_error(self):
        run = {"status": "failed"}
        assert format_run_description(run) == "Failed"

    def test_failed_empty_error(self):
        run = {"status": "failed", "error_message": ""}
        assert format_run_description(run) == "Failed"

    def test_started(self):
        run = {"status": "started", "agent_type": "outreach"}
        assert format_run_description(run) == "Started (outreach)"

    def test_unknown_status(self):
        run = {"status": "queued"}
        assert format_run_description(run) == "queued"

    def test_long_error_truncated(self):
        run = {"status": "failed", "error_message": "x" * 200}
        desc = format_run_description(run)
        # Should truncate to 80 chars of error + "Failed: " prefix
        assert len(desc) <= 88

    def test_none_duration_handled(self):
        run = {"status": "completed", "duration_ms": None}
        assert format_run_description(run) == "Completed in 0ms"


# ---------------------------------------------------------------------------
# compute_pipeline_value
# ---------------------------------------------------------------------------


class TestComputePipelineValue:
    """Pipeline value = opportunities * avg_ticket."""

    def test_basic(self):
        assert compute_pipeline_value(10, 5000) == 50_000

    def test_default_ticket(self):
        assert compute_pipeline_value(10) == 60_000

    def test_zero(self):
        assert compute_pipeline_value(0) == 0

    def test_large_values(self):
        assert compute_pipeline_value(1000, 10000) == 10_000_000


# ---------------------------------------------------------------------------
# group_content_by_status
# ---------------------------------------------------------------------------


class TestGroupContentByStatus:
    """Group content items for Kanban display."""

    def test_empty_list(self):
        result = group_content_by_status([])
        assert result == {
            "draft": [], "review": [], "approved": [], "published": [],
        }

    def test_single_status(self):
        items = [
            {"id": 1, "status": "draft"},
            {"id": 2, "status": "draft"},
        ]
        result = group_content_by_status(items)
        assert len(result["draft"]) == 2
        assert len(result["review"]) == 0

    def test_mixed_statuses(self):
        items = [
            {"id": 1, "status": "draft"},
            {"id": 2, "status": "review"},
            {"id": 3, "status": "approved"},
            {"id": 4, "status": "published"},
        ]
        result = group_content_by_status(items)
        assert len(result["draft"]) == 1
        assert len(result["review"]) == 1
        assert len(result["approved"]) == 1
        assert len(result["published"]) == 1

    def test_unknown_status_ignored(self):
        items = [
            {"id": 1, "status": "draft"},
            {"id": 2, "status": "unknown_status"},
        ]
        result = group_content_by_status(items)
        assert len(result["draft"]) == 1
        total = sum(len(v) for v in result.values())
        assert total == 1  # unknown_status not in any bucket

    def test_default_status_is_draft(self):
        items = [{"id": 1}]  # No status field
        result = group_content_by_status(items)
        assert len(result["draft"]) == 1

    def test_preserves_item_data(self):
        items = [{"id": 42, "status": "approved", "title": "My Post"}]
        result = group_content_by_status(items)
        assert result["approved"][0]["title"] == "My Post"
        assert result["approved"][0]["id"] == 42


# ---------------------------------------------------------------------------
# format_credential_status
# ---------------------------------------------------------------------------


class TestFormatCredentialStatus:
    """Credential environment variable checks."""

    def test_missing_credential(self):
        # Use a var name unlikely to be set
        result = format_credential_status(
            "__TEST_ENCLAVE_MISSING_VAR__", "Test Cred",
        )
        assert result["is_set"] is False
        assert result["status_text"] == "missing"
        assert result["env_var"] == "__TEST_ENCLAVE_MISSING_VAR__"
        assert result["label"] == "Test Cred"

    def test_set_credential(self, monkeypatch):
        monkeypatch.setenv("__TEST_ENCLAVE_SET_VAR__", "some_value")
        result = format_credential_status(
            "__TEST_ENCLAVE_SET_VAR__", "Test Cred",
        )
        assert result["is_set"] is True
        assert result["status_text"] == "configured"

    def test_empty_string_is_missing(self, monkeypatch):
        monkeypatch.setenv("__TEST_ENCLAVE_EMPTY_VAR__", "")
        result = format_credential_status(
            "__TEST_ENCLAVE_EMPTY_VAR__", "Test",
        )
        assert result["is_set"] is False
