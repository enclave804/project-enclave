"""
Tests for Pipeline Dashboard — Phase 16: Sales Pipeline.

Covers:
    - Pipeline helper functions (format_deal_stage, format_pipeline_value,
      compute_pipeline_metrics, compute_sequence_stats, compute_meeting_stats)
    - STAGE_CONFIG constant
    - Migration schema validation (013_sales_pipeline.sql)
    - Module imports
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest


# ══════════════════════════════════════════════════════════════════════
# Helper Function Tests
# ══════════════════════════════════════════════════════════════════════


class TestFormatDealStage:
    """Tests for format_deal_stage helper."""

    def test_qualified(self):
        from dashboard.pages._pipeline_helpers import format_deal_stage
        label, color = format_deal_stage("qualified")
        assert label == "Qualified"
        assert color == "#3b82f6"

    def test_prospect(self):
        from dashboard.pages._pipeline_helpers import format_deal_stage
        label, color = format_deal_stage("prospect")
        assert label == "Prospect"

    def test_proposal(self):
        from dashboard.pages._pipeline_helpers import format_deal_stage
        label, color = format_deal_stage("proposal")
        assert label == "Proposal"

    def test_negotiation(self):
        from dashboard.pages._pipeline_helpers import format_deal_stage
        label, color = format_deal_stage("negotiation")
        assert label == "Negotiation"

    def test_closed_won(self):
        from dashboard.pages._pipeline_helpers import format_deal_stage
        label, color = format_deal_stage("closed_won")
        assert label == "Closed Won"
        assert color == "#10b981"

    def test_closed_lost(self):
        from dashboard.pages._pipeline_helpers import format_deal_stage
        label, color = format_deal_stage("closed_lost")
        assert label == "Closed Lost"
        assert color == "#ef4444"

    def test_unknown_stage(self):
        from dashboard.pages._pipeline_helpers import format_deal_stage
        label, color = format_deal_stage("unknown_stage")
        assert label == "Unknown Stage"
        assert color == "#6b7280"


class TestFormatPipelineValue:
    """Tests for format_pipeline_value helper."""

    def test_zero(self):
        from dashboard.pages._pipeline_helpers import format_pipeline_value
        assert format_pipeline_value(0) == "$0.00"

    def test_standard_amount(self):
        from dashboard.pages._pipeline_helpers import format_pipeline_value
        assert format_pipeline_value(150000) == "$1,500.00"

    def test_large_amount(self):
        from dashboard.pages._pipeline_helpers import format_pipeline_value
        assert format_pipeline_value(10000000) == "$100,000.00"

    def test_small_amount(self):
        from dashboard.pages._pipeline_helpers import format_pipeline_value
        assert format_pipeline_value(99) == "$0.99"

    def test_negative_amount(self):
        from dashboard.pages._pipeline_helpers import format_pipeline_value
        assert format_pipeline_value(-50000) == "-$500.00"


class TestComputePipelineMetrics:
    """Tests for compute_pipeline_metrics helper."""

    def test_empty(self):
        from dashboard.pages._pipeline_helpers import compute_pipeline_metrics
        metrics = compute_pipeline_metrics([])
        assert metrics["total_deals"] == 0
        assert metrics["active_deals"] == 0
        assert metrics["won_deals"] == 0
        assert metrics["lost_deals"] == 0
        assert metrics["win_rate"] == 0.0
        assert metrics["avg_deal_value_cents"] == 0.0

    def test_single_active_deal(self):
        from dashboard.pages._pipeline_helpers import compute_pipeline_metrics
        opps = [{"stage": "qualified", "value_cents": 150000}]
        metrics = compute_pipeline_metrics(opps)
        assert metrics["total_deals"] == 1
        assert metrics["active_deals"] == 1
        assert metrics["active_value_cents"] == 150000
        assert metrics["won_deals"] == 0

    def test_won_deals(self):
        from dashboard.pages._pipeline_helpers import compute_pipeline_metrics
        opps = [
            {"stage": "closed_won", "value_cents": 100000},
            {"stage": "closed_won", "value_cents": 200000},
        ]
        metrics = compute_pipeline_metrics(opps)
        assert metrics["won_deals"] == 2
        assert metrics["won_value_cents"] == 300000

    def test_lost_deals(self):
        from dashboard.pages._pipeline_helpers import compute_pipeline_metrics
        opps = [{"stage": "closed_lost", "value_cents": 50000}]
        metrics = compute_pipeline_metrics(opps)
        assert metrics["lost_deals"] == 1

    def test_win_rate(self):
        from dashboard.pages._pipeline_helpers import compute_pipeline_metrics
        opps = [
            {"stage": "closed_won", "value_cents": 100000},
            {"stage": "closed_lost", "value_cents": 50000},
            {"stage": "closed_won", "value_cents": 200000},
        ]
        metrics = compute_pipeline_metrics(opps)
        # 2 won / (2 won + 1 lost) = 0.6667
        assert metrics["win_rate"] == pytest.approx(0.6667, abs=0.001)

    def test_stage_breakdown(self):
        from dashboard.pages._pipeline_helpers import compute_pipeline_metrics
        opps = [
            {"stage": "prospect", "value_cents": 50000},
            {"stage": "qualified", "value_cents": 100000},
            {"stage": "qualified", "value_cents": 150000},
            {"stage": "proposal", "value_cents": 200000},
        ]
        metrics = compute_pipeline_metrics(opps)
        breakdown = metrics["stage_breakdown"]
        assert breakdown["prospect"]["count"] == 1
        assert breakdown["qualified"]["count"] == 2
        assert breakdown["qualified"]["value_cents"] == 250000
        assert breakdown["proposal"]["count"] == 1

    def test_avg_deal_value(self):
        from dashboard.pages._pipeline_helpers import compute_pipeline_metrics
        opps = [
            {"stage": "qualified", "value_cents": 100000},
            {"stage": "qualified", "value_cents": 200000},
        ]
        metrics = compute_pipeline_metrics(opps)
        assert metrics["avg_deal_value_cents"] == 150000.0

    def test_missing_value_cents(self):
        from dashboard.pages._pipeline_helpers import compute_pipeline_metrics
        opps = [{"stage": "prospect"}]  # no value_cents field
        metrics = compute_pipeline_metrics(opps)
        assert metrics["total_deals"] == 1
        assert metrics["total_value_cents"] == 0

    def test_none_value_cents(self):
        from dashboard.pages._pipeline_helpers import compute_pipeline_metrics
        opps = [{"stage": "prospect", "value_cents": None}]
        metrics = compute_pipeline_metrics(opps)
        assert metrics["total_value_cents"] == 0


class TestComputeSequenceStats:
    """Tests for compute_sequence_stats helper."""

    def test_empty(self):
        from dashboard.pages._pipeline_helpers import compute_sequence_stats
        stats = compute_sequence_stats([])
        assert stats["total_sequences"] == 0
        assert stats["active"] == 0
        assert stats["completed"] == 0
        assert stats["replied"] == 0
        assert stats["paused"] == 0
        assert stats["cancelled"] == 0
        assert stats["avg_steps_completed"] == 0.0
        assert stats["reply_rate"] == 0.0

    def test_active_sequences(self):
        from dashboard.pages._pipeline_helpers import compute_sequence_stats
        seqs = [
            {"status": "active", "current_step": 2},
            {"status": "active", "current_step": 3},
        ]
        stats = compute_sequence_stats(seqs)
        assert stats["total_sequences"] == 2
        assert stats["active"] == 2
        assert stats["avg_steps_completed"] == 2.5

    def test_mixed_statuses(self):
        from dashboard.pages._pipeline_helpers import compute_sequence_stats
        seqs = [
            {"status": "active", "current_step": 1},
            {"status": "completed", "current_step": 5},
            {"status": "replied", "current_step": 3},
            {"status": "paused", "current_step": 2},
            {"status": "cancelled", "current_step": 0},
        ]
        stats = compute_sequence_stats(seqs)
        assert stats["total_sequences"] == 5
        assert stats["active"] == 1
        assert stats["completed"] == 1
        assert stats["replied"] == 1
        assert stats["paused"] == 1
        assert stats["cancelled"] == 1

    def test_reply_rate(self):
        from dashboard.pages._pipeline_helpers import compute_sequence_stats
        seqs = [
            {"status": "completed", "current_step": 5},
            {"status": "replied", "current_step": 2},
            {"status": "completed", "current_step": 5},
            {"status": "replied", "current_step": 3},
        ]
        stats = compute_sequence_stats(seqs)
        # 2 replied / (2 replied + 2 completed) = 0.5
        assert stats["reply_rate"] == 0.5


class TestComputeMeetingStats:
    """Tests for compute_meeting_stats helper."""

    def test_empty(self):
        from dashboard.pages._pipeline_helpers import compute_meeting_stats
        stats = compute_meeting_stats([])
        assert stats["total_meetings"] == 0
        assert stats["proposed"] == 0
        assert stats["confirmed"] == 0
        assert stats["completed"] == 0
        assert stats["cancelled"] == 0
        assert stats["no_show"] == 0
        assert stats["confirmation_rate"] == 0.0
        assert stats["completion_rate"] == 0.0
        assert stats["by_type"] == {}

    def test_proposed_meetings(self):
        from dashboard.pages._pipeline_helpers import compute_meeting_stats
        mtgs = [
            {"status": "proposed", "meeting_type": "discovery"},
            {"status": "proposed", "meeting_type": "demo"},
        ]
        stats = compute_meeting_stats(mtgs)
        assert stats["total_meetings"] == 2
        assert stats["proposed"] == 2
        assert stats["by_type"]["discovery"] == 1
        assert stats["by_type"]["demo"] == 1

    def test_mixed_statuses(self):
        from dashboard.pages._pipeline_helpers import compute_meeting_stats
        mtgs = [
            {"status": "proposed", "meeting_type": "discovery"},
            {"status": "confirmed", "meeting_type": "discovery"},
            {"status": "completed", "meeting_type": "demo"},
            {"status": "cancelled", "meeting_type": "follow_up"},
            {"status": "no_show", "meeting_type": "discovery"},
        ]
        stats = compute_meeting_stats(mtgs)
        assert stats["total_meetings"] == 5
        assert stats["proposed"] == 1
        assert stats["confirmed"] == 1
        assert stats["completed"] == 1
        assert stats["cancelled"] == 1
        assert stats["no_show"] == 1

    def test_confirmation_rate(self):
        from dashboard.pages._pipeline_helpers import compute_meeting_stats
        mtgs = [
            {"status": "confirmed", "meeting_type": "discovery"},
            {"status": "completed", "meeting_type": "discovery"},
            {"status": "proposed", "meeting_type": "discovery"},
            {"status": "cancelled", "meeting_type": "discovery"},
        ]
        stats = compute_meeting_stats(mtgs)
        # confirmed(1) + completed(1) / total(4) = 0.5
        assert stats["confirmation_rate"] == 0.5

    def test_completion_rate(self):
        from dashboard.pages._pipeline_helpers import compute_meeting_stats
        mtgs = [
            {"status": "confirmed", "meeting_type": "discovery"},
            {"status": "completed", "meeting_type": "demo"},
            {"status": "completed", "meeting_type": "discovery"},
        ]
        stats = compute_meeting_stats(mtgs)
        # completed(2) / (confirmed(1) + completed(2)) = 0.6667
        assert stats["completion_rate"] == pytest.approx(0.6667, abs=0.001)


class TestStageConfig:
    """Tests for STAGE_CONFIG constant."""

    def test_all_stages_present(self):
        from dashboard.pages._pipeline_helpers import STAGE_CONFIG
        expected = {"prospect", "qualified", "proposal", "negotiation", "closed_won", "closed_lost"}
        assert set(STAGE_CONFIG.keys()) == expected

    def test_each_stage_has_label(self):
        from dashboard.pages._pipeline_helpers import STAGE_CONFIG
        for stage, config in STAGE_CONFIG.items():
            assert "label" in config
            assert isinstance(config["label"], str)

    def test_each_stage_has_color(self):
        from dashboard.pages._pipeline_helpers import STAGE_CONFIG
        for stage, config in STAGE_CONFIG.items():
            assert "color" in config
            assert config["color"].startswith("#")

    def test_each_stage_has_emoji(self):
        from dashboard.pages._pipeline_helpers import STAGE_CONFIG
        for stage, config in STAGE_CONFIG.items():
            assert "emoji" in config
            assert len(config["emoji"]) > 0


# ══════════════════════════════════════════════════════════════════════
# Migration Schema Tests
# ══════════════════════════════════════════════════════════════════════


class TestMigrationSchema:
    """Tests for 013_sales_pipeline.sql migration."""

    def _read_migration(self) -> str:
        migration_path = (
            Path(__file__).parent.parent.parent
            / "infrastructure"
            / "migrations"
            / "013_sales_pipeline.sql"
        )
        with open(migration_path) as f:
            return f.read()

    def test_migration_exists(self):
        sql = self._read_migration()
        assert len(sql) > 0

    def test_creates_follow_up_sequences(self):
        sql = self._read_migration()
        assert "follow_up_sequences" in sql
        assert "CREATE TABLE" in sql

    def test_creates_scheduled_meetings(self):
        sql = self._read_migration()
        assert "scheduled_meetings" in sql

    def test_follow_up_sequences_columns(self):
        sql = self._read_migration()
        for col in [
            "vertical_id",
            "contact_email",
            "current_step",
            "max_steps",
            "status",
            "interval_days",
            "next_send_at",
            "last_sent_at",
            "steps",
        ]:
            assert col in sql

    def test_scheduled_meetings_columns(self):
        sql = self._read_migration()
        for col in [
            "vertical_id",
            "contact_email",
            "meeting_type",
            "scheduled_at",
            "duration_minutes",
            "status",
            "agenda",
            "outcome",
        ]:
            assert col in sql

    def test_follow_up_sequence_status_check(self):
        sql = self._read_migration()
        for status in ["active", "paused", "completed", "cancelled", "replied"]:
            assert status in sql

    def test_meeting_status_check(self):
        sql = self._read_migration()
        for status in ["proposed", "confirmed", "completed", "cancelled", "no_show", "rescheduled"]:
            assert status in sql

    def test_meeting_type_check(self):
        sql = self._read_migration()
        for mtype in ["discovery", "demo", "follow_up", "negotiation", "kickoff"]:
            assert mtype in sql

    def test_indexes_created(self):
        sql = self._read_migration()
        assert "CREATE INDEX" in sql or "idx_" in sql

    def test_rpc_function(self):
        sql = self._read_migration()
        assert "get_pipeline_stats" in sql


# ══════════════════════════════════════════════════════════════════════
# Import Tests
# ══════════════════════════════════════════════════════════════════════


class TestImports:
    """Tests that all Phase 16 modules are importable."""

    def test_import_followup_agent(self):
        from core.agents.implementations.followup_agent import FollowUpAgent
        assert FollowUpAgent is not None

    def test_import_meeting_agent(self):
        from core.agents.implementations.meeting_agent import MeetingSchedulerAgent
        assert MeetingSchedulerAgent is not None

    def test_import_pipeline_agent(self):
        from core.agents.implementations.pipeline_agent import SalesPipelineAgent
        assert SalesPipelineAgent is not None

    def test_import_followup_state(self):
        from core.agents.state import FollowUpAgentState
        assert FollowUpAgentState is not None

    def test_import_meeting_state(self):
        from core.agents.state import MeetingSchedulerAgentState
        assert MeetingSchedulerAgentState is not None

    def test_import_pipeline_state(self):
        from core.agents.state import SalesPipelineAgentState
        assert SalesPipelineAgentState is not None

    def test_import_pipeline_helpers(self):
        from dashboard.pages._pipeline_helpers import (
            compute_pipeline_metrics,
            compute_sequence_stats,
            compute_meeting_stats,
            format_deal_stage,
            format_pipeline_value,
            STAGE_CONFIG,
        )
        assert compute_pipeline_metrics is not None
        assert compute_sequence_stats is not None
        assert compute_meeting_stats is not None
        assert format_deal_stage is not None
        assert format_pipeline_value is not None
        assert STAGE_CONFIG is not None

    def test_all_agents_registered(self):
        # Force imports to trigger registration
        from core.agents.implementations.followup_agent import FollowUpAgent  # noqa: F401
        from core.agents.implementations.meeting_agent import MeetingSchedulerAgent  # noqa: F401
        from core.agents.implementations.pipeline_agent import SalesPipelineAgent  # noqa: F401
        from core.agents.registry import AGENT_IMPLEMENTATIONS

        assert "followup" in AGENT_IMPLEMENTATIONS
        assert "meeting_scheduler" in AGENT_IMPLEMENTATIONS
        assert "sales_pipeline" in AGENT_IMPLEMENTATIONS
