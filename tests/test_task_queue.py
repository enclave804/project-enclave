"""
Unit tests for TaskQueueManager and EventBus.

Tests enqueue/claim/complete/fail cycle and event routing.
Uses mocked database to test pure business logic.
"""

import pytest
from unittest.mock import MagicMock, patch

from core.agents.task_queue import TaskQueueManager
from core.agents.event_bus import EventBus, EventRoute


# ─── TaskQueueManager Tests ───────────────────────────────────────────

class TestTaskQueueManager:
    """Tests for the task queue lifecycle."""

    @pytest.fixture
    def mock_db(self):
        db = MagicMock()
        db.enqueue_task.return_value = {
            "task_id": "task-001",
            "target_agent_id": "appointment_setter",
            "task_type": "handle_reply",
            "status": "pending",
        }
        db.claim_next_task.return_value = {
            "task_id": "task-001",
            "target_agent_id": "appointment_setter",
            "task_type": "handle_reply",
            "input_data": {"email": "test@example.com"},
        }
        db.complete_task.return_value = {
            "task_id": "task-001",
            "status": "completed",
        }
        db.fail_task.return_value = {
            "task_id": "task-001",
            "status": "failed",
        }
        db.count_pending_tasks.return_value = 3
        db.list_tasks.return_value = [
            {"task_id": "task-001", "status": "pending"},
            {"task_id": "task-002", "status": "completed"},
        ]
        db.recover_zombie_tasks.return_value = 2
        return db

    @pytest.fixture
    def queue(self, mock_db):
        return TaskQueueManager(mock_db)

    def test_enqueue_task(self, queue, mock_db):
        """Should create a task and return the record."""
        result = queue.enqueue(
            target_agent_id="appointment_setter",
            task_type="handle_reply",
            input_data={"email": "test@example.com"},
            source_agent_id="outreach",
            priority=3,
        )
        assert result["task_id"] == "task-001"
        mock_db.enqueue_task.assert_called_once()
        call_args = mock_db.enqueue_task.call_args[0][0]
        assert call_args["target_agent_id"] == "appointment_setter"
        assert call_args["task_type"] == "handle_reply"
        assert call_args["priority"] == 3
        assert call_args["source_agent_id"] == "outreach"

    def test_enqueue_default_priority(self, queue, mock_db):
        """Default priority should be 5."""
        queue.enqueue("agent_x", "test_task", {})
        call_args = mock_db.enqueue_task.call_args[0][0]
        assert call_args["priority"] == 5

    def test_claim_task(self, queue, mock_db):
        """Should atomically claim the next pending task."""
        task = queue.claim("appointment_setter")
        assert task is not None
        assert task["task_id"] == "task-001"
        mock_db.claim_next_task.assert_called_once_with("appointment_setter")

    def test_claim_empty_queue(self, queue, mock_db):
        """Should return None when no tasks available."""
        mock_db.claim_next_task.return_value = None
        task = queue.claim("some_agent")
        assert task is None

    def test_complete_task(self, queue, mock_db):
        """Should mark task as completed with output data."""
        result = queue.complete("task-001", output_data={"result": "success"})
        assert result["status"] == "completed"
        mock_db.complete_task.assert_called_once_with(
            "task-001", output_data={"result": "success"}
        )

    def test_fail_task_with_retry(self, queue, mock_db):
        """Should fail task and request retry."""
        result = queue.fail("task-001", "Connection timeout", retry=True)
        mock_db.fail_task.assert_called_once_with(
            "task-001", error_message="Connection timeout", retry=True
        )

    def test_fail_task_permanent(self, queue, mock_db):
        """Should fail task without retry."""
        queue.fail("task-001", "Invalid data", retry=False)
        mock_db.fail_task.assert_called_once_with(
            "task-001", error_message="Invalid data", retry=False
        )

    def test_get_pending_count(self, queue, mock_db):
        """Should return pending task count."""
        count = queue.get_pending_count("appointment_setter")
        assert count == 3

    def test_list_tasks(self, queue, mock_db):
        """Should list tasks with filters."""
        tasks = queue.list_tasks(agent_id="agent_x", status="pending", limit=10)
        assert len(tasks) == 2
        mock_db.list_tasks.assert_called_once_with(
            agent_id="agent_x", status="pending", limit=10
        )

    def test_heartbeat(self, queue, mock_db):
        """Should update heartbeat without raising on failure."""
        queue.heartbeat("task-001")
        mock_db.heartbeat_task.assert_called_once_with("task-001")

    def test_heartbeat_failure_silent(self, queue, mock_db):
        """Heartbeat failure should log warning, not raise."""
        mock_db.heartbeat_task.side_effect = Exception("DB down")
        queue.heartbeat("task-001")  # Should not raise

    def test_recover_zombies(self, queue, mock_db):
        """Should recover stale tasks."""
        count = queue.recover_zombies(stale_minutes=15)
        assert count == 2
        mock_db.recover_zombie_tasks.assert_called_once_with(15)


# ─── EventBus Tests ───────────────────────────────────────────────────

class TestEventBus:
    """Tests for event routing and dispatch."""

    @pytest.fixture
    def mock_task_queue(self):
        queue = MagicMock()
        queue.enqueue.return_value = {"task_id": "task-from-event"}
        return queue

    @pytest.fixture
    def bus(self, mock_task_queue):
        return EventBus(mock_task_queue)

    def test_register_route(self, bus):
        """Should register a route for an event type."""
        bus.register("email_reply_received", "appointment_setter", "handle_reply")
        routes = bus.get_routes_for_event("email_reply_received")
        assert len(routes) == 1
        assert routes[0].target_agent_id == "appointment_setter"
        assert routes[0].task_type == "handle_reply"

    def test_register_multiple_routes(self, bus):
        """Should support fan-out (multiple routes per event)."""
        bus.register("new_lead_added", "outreach", "process_lead")
        bus.register("new_lead_added", "seo_content", "update_topics")
        routes = bus.get_routes_for_event("new_lead_added")
        assert len(routes) == 2

    def test_dispatch_creates_tasks(self, bus, mock_task_queue):
        """Should create a task for each registered route."""
        bus.register("email_reply", "agent_a", "handle")
        bus.register("email_reply", "agent_b", "process")

        results = bus.dispatch(
            "email_reply",
            {"email": "test@test.com"},
            source_agent_id="outreach",
        )

        assert len(results) == 2
        assert mock_task_queue.enqueue.call_count == 2

    def test_dispatch_passes_priority(self, bus, mock_task_queue):
        """Should use the route's priority when creating tasks."""
        bus.register("urgent_event", "agent_a", "handle", priority=1)
        bus.dispatch("urgent_event", {"data": "test"})

        call_kwargs = mock_task_queue.enqueue.call_args
        assert call_kwargs.kwargs.get("priority") == 1 or \
               call_kwargs[1].get("priority") == 1

    def test_dispatch_no_routes(self, bus, mock_task_queue):
        """Should return empty list for unregistered events."""
        results = bus.dispatch("unknown_event", {})
        assert results == []
        mock_task_queue.enqueue.assert_not_called()

    def test_dispatch_handles_enqueue_failure(self, bus, mock_task_queue):
        """Should continue dispatching even if one route fails."""
        bus.register("test_event", "agent_a", "task_a")
        bus.register("test_event", "agent_b", "task_b")

        # First enqueue fails, second succeeds
        mock_task_queue.enqueue.side_effect = [
            Exception("DB error"),
            {"task_id": "task-002"},
        ]

        results = bus.dispatch("test_event", {})
        assert len(results) == 1  # Only the successful one

    def test_list_routes(self, bus):
        """Should return all registered routes."""
        bus.register("event_a", "agent_1", "task_1")
        bus.register("event_b", "agent_2", "task_2")
        all_routes = bus.list_routes()
        assert "event_a" in all_routes
        assert "event_b" in all_routes

    def test_route_with_description(self, bus):
        """Should store description on route."""
        bus.register(
            "email_reply",
            "appointment_setter",
            "handle_reply",
            description="Route replies to appointment setter",
        )
        routes = bus.get_routes_for_event("email_reply")
        assert routes[0].description == "Route replies to appointment setter"


# ─── Contract Tests ───────────────────────────────────────────────────

class TestContracts:
    """Tests for inter-agent data contracts."""

    def test_lead_data_contract(self):
        from core.agents.contracts import LeadData
        lead = LeadData(
            contact_email="jane@acme.com",
            contact_name="Jane Doe",
            company_domain="acme.com",
        )
        assert lead.contact_email == "jane@acme.com"
        data = lead.model_dump()
        assert "contact_email" in data
        assert "company_domain" in data

    def test_email_payload_contract(self):
        from core.agents.contracts import EmailPayload
        email = EmailPayload(
            from_email="us@enclave.com",
            to_email="jane@acme.com",
            subject="Security Assessment",
            body="Hi Jane...",
        )
        assert email.direction == "outbound"

    def test_insight_confidence_validation(self):
        from core.agents.contracts import InsightData
        # Valid
        insight = InsightData(
            insight_type="winning_pattern",
            content="CISO personas respond well to compliance angles",
            confidence=0.9,
        )
        assert insight.confidence == 0.9

        # Invalid: confidence out of range
        with pytest.raises(Exception):
            InsightData(
                insight_type="winning_pattern",
                content="test",
                confidence=1.5,
            )

    def test_meeting_request_contract(self):
        from core.agents.contracts import MeetingRequest
        req = MeetingRequest(
            contact_email="cto@acme.com",
            contact_name="Alex Chen",
            proposed_times=["2025-02-01T10:00Z", "2025-02-01T14:00Z"],
        )
        assert len(req.proposed_times) == 2
        assert req.duration_minutes == 30

    def test_content_brief_contract(self):
        from core.agents.contracts import ContentBrief
        brief = ContentBrief(
            content_type="blog_post",
            target_keywords=["penetration testing"],
            target_word_count=2000,
        )
        assert brief.content_type == "blog_post"
        data = brief.model_dump()
        assert data["target_word_count"] == 2000
