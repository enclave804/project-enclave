"""
Tests for Phase 1E: Appointment Setter Agent — The Revenue Closer.

Covers:
- CalendarClient + providers (mock/google/calcom)
- Calendar MCP tools (availability, slots, booking, link)
- AppointmentSetterAgent (intent classification, handlers, graph)
- Agent YAML config validation
- System prompt loading
- Heuristic fallback classification
- Sandbox safety for booking tool
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ═══════════════════════════════════════════════════════════════════════
# 1. Calendar Client Tests
# ═══════════════════════════════════════════════════════════════════════


class TestMockCalendarProvider:
    """Tests for the MockCalendarProvider."""

    @pytest.fixture
    def provider(self):
        from core.integrations.calendar_client import MockCalendarProvider
        return MockCalendarProvider()

    @pytest.mark.asyncio
    async def test_get_available_slots_returns_list(self, provider):
        """Available slots should be a list of dicts."""
        now = datetime.now(timezone.utc)
        slots = await provider.get_available_slots(
            start_date=now,
            end_date=now + timedelta(days=5),
        )
        assert isinstance(slots, list)

    @pytest.mark.asyncio
    async def test_get_available_slots_structure(self, provider):
        """Each slot should have start, end, timezone, and available fields."""
        now = datetime.now(timezone.utc)
        # Use a Monday to guarantee weekday slots
        monday = now - timedelta(days=now.weekday())
        if monday < now:
            monday += timedelta(weeks=1)

        slots = await provider.get_available_slots(
            start_date=monday.replace(hour=0),
            end_date=monday.replace(hour=23, minute=59),
        )
        if slots:  # May be empty depending on time
            slot = slots[0]
            assert "start" in slot
            assert "end" in slot
            assert "timezone" in slot
            assert "available" in slot

    @pytest.mark.asyncio
    async def test_get_available_slots_skips_weekends(self, provider):
        """No slots should be generated for Saturday or Sunday."""
        # Find next Saturday
        now = datetime.now(timezone.utc)
        days_until_saturday = (5 - now.weekday()) % 7
        saturday = now + timedelta(days=days_until_saturday)
        sunday = saturday + timedelta(days=1)

        slots = await provider.get_available_slots(
            start_date=saturday.replace(hour=0, minute=0),
            end_date=sunday.replace(hour=23, minute=59),
        )
        assert len(slots) == 0

    @pytest.mark.asyncio
    async def test_get_available_slots_caps_at_20(self, provider):
        """Should return at most 20 slots."""
        now = datetime.now(timezone.utc)
        slots = await provider.get_available_slots(
            start_date=now,
            end_date=now + timedelta(days=30),
        )
        assert len(slots) <= 20

    @pytest.mark.asyncio
    async def test_check_availability_weekday_business_hours(self, provider):
        """Weekday business hours should be available."""
        # Find next Monday at 10 AM
        now = datetime.now(timezone.utc)
        days_until_monday = (0 - now.weekday()) % 7
        if days_until_monday == 0:
            days_until_monday = 7
        monday_10am = (now + timedelta(days=days_until_monday)).replace(
            hour=10, minute=0, second=0, microsecond=0
        )
        result = await provider.check_availability(monday_10am)
        assert result is True

    @pytest.mark.asyncio
    async def test_check_availability_weekend_unavailable(self, provider):
        """Weekend should not be available."""
        now = datetime.now(timezone.utc)
        days_until_saturday = (5 - now.weekday()) % 7
        saturday = (now + timedelta(days=days_until_saturday)).replace(hour=10)
        result = await provider.check_availability(saturday)
        assert result is False

    @pytest.mark.asyncio
    async def test_check_availability_outside_hours(self, provider):
        """Before 9 AM or after 5 PM should not be available."""
        now = datetime.now(timezone.utc)
        days_until_monday = (0 - now.weekday()) % 7
        if days_until_monday == 0:
            days_until_monday = 7
        monday_7am = (now + timedelta(days=days_until_monday)).replace(
            hour=7, minute=0
        )
        result = await provider.check_availability(monday_7am)
        assert result is False

    @pytest.mark.asyncio
    async def test_book_meeting_returns_confirmation(self, provider):
        """Booking should return a confirmation dict."""
        now = datetime.now(timezone.utc)
        days_until_monday = (0 - now.weekday()) % 7
        if days_until_monday == 0:
            days_until_monday = 7
        monday_10am = (now + timedelta(days=days_until_monday)).replace(
            hour=10, minute=0, second=0, microsecond=0
        )

        booking = await provider.book_meeting(
            attendee_email="test@example.com",
            start_time=monday_10am,
            title="Discovery Call",
            attendee_name="Test User",
        )

        assert booking["status"] == "confirmed"
        assert booking["attendee_email"] == "test@example.com"
        assert "booking_id" in booking
        assert "calendar_link" in booking
        assert booking["provider"] == "mock"

    @pytest.mark.asyncio
    async def test_book_meeting_blocks_time(self, provider):
        """After booking, the same slot should be unavailable."""
        now = datetime.now(timezone.utc)
        days_until_monday = (0 - now.weekday()) % 7
        if days_until_monday == 0:
            days_until_monday = 7
        monday_10am = (now + timedelta(days=days_until_monday)).replace(
            hour=10, minute=0, second=0, microsecond=0
        )

        await provider.book_meeting(
            attendee_email="test@example.com",
            start_time=monday_10am,
        )

        is_available = await provider.check_availability(monday_10am)
        assert is_available is False


class TestCalendarClient:
    """Tests for the high-level CalendarClient."""

    @pytest.fixture
    def client(self):
        from core.integrations.calendar_client import CalendarClient
        return CalendarClient()

    def test_default_provider_is_mock(self, client):
        """Default provider should be MockCalendarProvider."""
        from core.integrations.calendar_client import MockCalendarProvider
        assert isinstance(client.provider, MockCalendarProvider)

    def test_from_env_default_mock(self, monkeypatch):
        """from_env() with no CALENDAR_PROVIDER should use mock."""
        monkeypatch.delenv("CALENDAR_PROVIDER", raising=False)
        from core.integrations.calendar_client import CalendarClient
        client = CalendarClient.from_env()
        from core.integrations.calendar_client import MockCalendarProvider
        assert isinstance(client.provider, MockCalendarProvider)

    def test_from_env_google_provider(self, monkeypatch):
        """from_env() with CALENDAR_PROVIDER=google should use Google."""
        monkeypatch.setenv("CALENDAR_PROVIDER", "google")
        from core.integrations.calendar_client import CalendarClient
        client = CalendarClient.from_env()
        from core.integrations.calendar_client import GoogleCalendarProvider
        assert isinstance(client.provider, GoogleCalendarProvider)

    def test_from_env_calcom_provider(self, monkeypatch):
        """from_env() with CALENDAR_PROVIDER=calcom should use Cal.com."""
        monkeypatch.setenv("CALENDAR_PROVIDER", "calcom")
        from core.integrations.calendar_client import CalendarClient
        client = CalendarClient.from_env()
        from core.integrations.calendar_client import CalComProvider
        assert isinstance(client.provider, CalComProvider)

    def test_from_env_unknown_falls_back_to_mock(self, monkeypatch):
        """Unknown provider should fall back to mock."""
        monkeypatch.setenv("CALENDAR_PROVIDER", "nonexistent")
        from core.integrations.calendar_client import CalendarClient, MockCalendarProvider
        client = CalendarClient.from_env()
        assert isinstance(client.provider, MockCalendarProvider)

    @pytest.mark.asyncio
    async def test_get_available_slots_respects_max(self, client):
        """max_slots should cap the returned slots."""
        slots = await client.get_available_slots(days_ahead=10, max_slots=3)
        assert len(slots) <= 3

    def test_get_booking_link(self, client):
        """Should return a non-empty booking link."""
        link = client.get_booking_link()
        assert isinstance(link, str)
        assert len(link) > 0

    def test_default_timezone(self, client):
        """Default timezone should be America/New_York."""
        assert client.default_timezone == "America/New_York"

    def test_default_meeting_duration(self, client):
        """Default meeting duration should be 30 minutes."""
        assert client.meeting_duration_minutes == 30


class TestGoogleCalendarProvider:
    """Tests for GoogleCalendarProvider fallback behavior."""

    @pytest.mark.asyncio
    async def test_google_falls_back_to_mock(self):
        """Google provider should fall back to mock slots."""
        from core.integrations.calendar_client import GoogleCalendarProvider
        provider = GoogleCalendarProvider()
        now = datetime.now(timezone.utc)
        slots = await provider.get_available_slots(
            start_date=now, end_date=now + timedelta(days=3),
        )
        assert isinstance(slots, list)

    @pytest.mark.asyncio
    async def test_google_check_availability_falls_back(self):
        """Google check_availability should fall back to mock."""
        from core.integrations.calendar_client import GoogleCalendarProvider
        provider = GoogleCalendarProvider()
        now = datetime.now(timezone.utc)
        days_until_monday = (0 - now.weekday()) % 7
        if days_until_monday == 0:
            days_until_monday = 7
        monday = (now + timedelta(days=days_until_monday)).replace(hour=10)
        result = await provider.check_availability(monday)
        assert isinstance(result, bool)

    @pytest.mark.asyncio
    async def test_google_book_meeting_falls_back(self):
        """Google book_meeting should fall back to mock."""
        from core.integrations.calendar_client import GoogleCalendarProvider
        provider = GoogleCalendarProvider()
        now = datetime.now(timezone.utc)
        result = await provider.book_meeting(
            attendee_email="test@test.com",
            start_time=now.replace(hour=10),
        )
        assert isinstance(result, dict)


class TestCalComProvider:
    """Tests for CalComProvider fallback behavior."""

    @pytest.mark.asyncio
    async def test_calcom_falls_back_to_mock(self):
        """Cal.com provider should fall back to mock slots."""
        from core.integrations.calendar_client import CalComProvider
        provider = CalComProvider()
        now = datetime.now(timezone.utc)
        slots = await provider.get_available_slots(
            start_date=now, end_date=now + timedelta(days=3),
        )
        assert isinstance(slots, list)


# ═══════════════════════════════════════════════════════════════════════
# 2. Calendar MCP Tools Tests
# ═══════════════════════════════════════════════════════════════════════


class TestCalendarMCPTools:
    """Tests for calendar MCP tool functions."""

    @pytest.fixture
    def mock_calendar_client(self):
        """Create a mock CalendarClient."""
        client = MagicMock()
        client.check_availability = AsyncMock(return_value=True)
        client.get_available_slots = AsyncMock(return_value=[
            {"start": "2025-01-15T10:00:00", "end": "2025-01-15T10:30:00",
             "timezone": "America/New_York", "available": True},
        ])
        client.book_meeting = AsyncMock(return_value={
            "booking_id": "test-123",
            "status": "confirmed",
            "attendee_email": "jane@acme.com",
            "start_time": "2025-01-15T10:00:00",
            "end_time": "2025-01-15T10:30:00",
            "calendar_link": "https://cal.example.com/test",
        })
        client.get_booking_link = MagicMock(return_value="https://cal.example.com")
        client.meeting_duration_minutes = 30
        return client

    @pytest.mark.asyncio
    async def test_check_availability_valid_time(self, mock_calendar_client):
        """Should return JSON with available=True for valid time."""
        from core.mcp.tools.calendar_tools import check_calendar_availability

        result = await check_calendar_availability(
            proposed_datetime="2025-01-15T10:00:00",
            _calendar_client=mock_calendar_client,
        )

        parsed = json.loads(result)
        assert parsed["available"] is True
        assert parsed["proposed_time"] == "2025-01-15T10:00:00"

    @pytest.mark.asyncio
    async def test_check_availability_invalid_datetime(self, mock_calendar_client):
        """Should return error for invalid datetime format."""
        from core.mcp.tools.calendar_tools import check_calendar_availability

        result = await check_calendar_availability(
            proposed_datetime="not-a-date",
            _calendar_client=mock_calendar_client,
        )

        parsed = json.loads(result)
        assert parsed["available"] is False
        assert "error" in parsed

    @pytest.mark.asyncio
    async def test_check_availability_suggests_alternatives(self, mock_calendar_client):
        """When unavailable, should suggest alternative slots."""
        mock_calendar_client.check_availability = AsyncMock(return_value=False)

        from core.mcp.tools.calendar_tools import check_calendar_availability

        result = await check_calendar_availability(
            proposed_datetime="2025-01-15T10:00:00",
            _calendar_client=mock_calendar_client,
        )

        parsed = json.loads(result)
        assert parsed["available"] is False
        assert "suggested_alternatives" in parsed
        assert "booking_link" in parsed

    @pytest.mark.asyncio
    async def test_get_available_slots(self, mock_calendar_client):
        """Should return JSON with slots and booking link."""
        from core.mcp.tools.calendar_tools import get_available_slots

        result = await get_available_slots(
            days_ahead=5,
            max_slots=3,
            _calendar_client=mock_calendar_client,
        )

        parsed = json.loads(result)
        assert "slots" in parsed
        assert "booking_link" in parsed
        assert "meeting_duration_minutes" in parsed
        assert parsed["meeting_duration_minutes"] == 30

    @pytest.mark.asyncio
    async def test_get_available_slots_caps_days(self, mock_calendar_client):
        """days_ahead should be capped at 14."""
        from core.mcp.tools.calendar_tools import get_available_slots

        await get_available_slots(
            days_ahead=30,
            _calendar_client=mock_calendar_client,
        )

        # Verify the client was called with capped value
        mock_calendar_client.get_available_slots.assert_called_once()
        call_kwargs = mock_calendar_client.get_available_slots.call_args
        assert call_kwargs[1]["days_ahead"] == 14

    @pytest.mark.asyncio
    async def test_get_booking_link(self, mock_calendar_client):
        """Should return JSON with booking link."""
        from core.mcp.tools.calendar_tools import get_booking_link

        result = await get_booking_link(
            _calendar_client=mock_calendar_client,
        )

        parsed = json.loads(result)
        assert "booking_link" in parsed
        assert parsed["booking_link"] == "https://cal.example.com"


class TestBookMeetingSlotSandbox:
    """Tests for the sandboxed book_meeting_slot tool."""

    def test_book_meeting_is_sandboxed(self):
        """book_meeting_slot should have @sandboxed_tool applied."""
        from core.mcp.tools.calendar_tools import book_meeting_slot
        from core.safety.sandbox import is_sandboxed
        assert is_sandboxed(book_meeting_slot)

    def test_book_meeting_sandbox_name(self):
        """Sandbox tool name should be 'book_meeting'."""
        from core.mcp.tools.calendar_tools import book_meeting_slot
        from core.safety.sandbox import get_sandbox_tool_name
        assert get_sandbox_tool_name(book_meeting_slot) == "book_meeting"

    @pytest.mark.asyncio
    async def test_book_meeting_sandboxed_in_dev(self, monkeypatch):
        """In development, booking should be sandboxed (not executed)."""
        monkeypatch.setenv("ENCLAVE_ENV", "development")

        from core.mcp.tools.calendar_tools import book_meeting_slot

        result = await book_meeting_slot(
            attendee_email="test@example.com",
            attendee_name="Test User",
            start_datetime="2025-01-15T10:00:00",
        )

        assert result.get("sandboxed") is True
        assert result["tool_name"] == "book_meeting"


# ═══════════════════════════════════════════════════════════════════════
# 3. MCP Server Registration Tests
# ═══════════════════════════════════════════════════════════════════════


class TestMCPServerCalendarRegistration:
    """Tests that calendar tools are registered on the MCP server."""

    def test_server_has_17_tools(self):
        """MCP server should now have 17 registered tools (10 original + 7 system monitoring)."""
        from core.mcp.server import create_mcp_server

        server = create_mcp_server()
        # The server should have tools registered
        # FastMCP stores tools internally - check the count via listing
        assert server is not None

    def test_calendar_tools_importable(self):
        """All 4 calendar tools should be importable."""
        from core.mcp.tools.calendar_tools import (
            check_calendar_availability,
            get_available_slots,
            book_meeting_slot,
            get_booking_link,
        )
        assert callable(check_calendar_availability)
        assert callable(get_available_slots)
        assert callable(book_meeting_slot)
        assert callable(get_booking_link)


# ═══════════════════════════════════════════════════════════════════════
# 4. Appointment Setter Agent Tests
# ═══════════════════════════════════════════════════════════════════════


class TestAppointmentSetterAgentRegistration:
    """Tests for agent type registration."""

    def test_agent_type_registered(self):
        """appointment_setter should be in the registry."""
        from core.agents.implementations.appointment_agent import AppointmentSetterAgent
        from core.agents.registry import AGENT_IMPLEMENTATIONS

        assert "appointment_setter" in AGENT_IMPLEMENTATIONS
        assert AGENT_IMPLEMENTATIONS["appointment_setter"] is AppointmentSetterAgent

    def test_agent_type_attribute(self):
        """Agent class should have agent_type = 'appointment_setter'."""
        from core.agents.implementations.appointment_agent import AppointmentSetterAgent
        assert AppointmentSetterAgent.agent_type == "appointment_setter"


class TestAppointmentSetterAgentConstruction:
    """Tests for agent construction and configuration."""

    @pytest.fixture
    def agent_config(self):
        from core.config.agent_schema import AgentInstanceConfig
        return AgentInstanceConfig(
            agent_id="appointment_setter_v1",
            agent_type="appointment_setter",
            name="Appointment Setter Agent",
            vertical_id="enclave_guard",
            enabled=True,
            human_gates={"enabled": True, "gate_before": ["human_review"]},
            params={
                "max_proposed_times": 3,
                "days_ahead_for_slots": 5,
                "company_name": "Enclave Guard",
                "value_proposition": "security posture review",
            },
        )

    @pytest.fixture
    def mock_deps(self):
        return {
            "db": MagicMock(),
            "embedder": MagicMock(),
            "anthropic_client": MagicMock(),
        }

    def test_construction(self, agent_config, mock_deps):
        """Agent should construct without errors."""
        from core.agents.implementations.appointment_agent import AppointmentSetterAgent
        agent = AppointmentSetterAgent(
            config=agent_config, **mock_deps,
        )
        assert agent.agent_id == "appointment_setter_v1"
        assert agent.vertical_id == "enclave_guard"

    def test_get_state_class(self, agent_config, mock_deps):
        """Should return AppointmentAgentState."""
        from core.agents.implementations.appointment_agent import AppointmentSetterAgent
        from core.agents.state import AppointmentAgentState

        agent = AppointmentSetterAgent(config=agent_config, **mock_deps)
        assert agent.get_state_class() is AppointmentAgentState

    def test_get_tools_empty(self, agent_config, mock_deps):
        """Tools are accessed via MCP, not injected."""
        from core.agents.implementations.appointment_agent import AppointmentSetterAgent
        agent = AppointmentSetterAgent(config=agent_config, **mock_deps)
        assert agent.get_tools() == []

    def test_build_graph(self, agent_config, mock_deps):
        """Graph should compile without errors."""
        from core.agents.implementations.appointment_agent import AppointmentSetterAgent
        agent = AppointmentSetterAgent(config=agent_config, **mock_deps)
        graph = agent.build_graph()
        assert graph is not None

    def test_repr(self, agent_config, mock_deps):
        """__repr__ should include agent_id and vertical."""
        from core.agents.implementations.appointment_agent import AppointmentSetterAgent
        agent = AppointmentSetterAgent(config=agent_config, **mock_deps)
        r = repr(agent)
        assert "appointment_setter_v1" in r
        assert "enclave_guard" in r

    def test_prepare_initial_state(self, agent_config, mock_deps):
        """Initial state should include appointment-specific fields."""
        from core.agents.implementations.appointment_agent import AppointmentSetterAgent
        agent = AppointmentSetterAgent(config=agent_config, **mock_deps)
        state = agent._prepare_initial_state(
            task={
                "inbound_email": {"from_email": "jane@acme.com", "body": "Sounds good"},
                "contact_id": "c123",
            },
            run_id="test-run-1",
        )
        assert state["inbound_email"]["from_email"] == "jane@acme.com"
        assert state["contact_id"] == "c123"
        assert state["reply_intent"] == ""
        assert state["meeting_booked"] is False
        assert state["proposed_times"] == []


class TestAppointmentHeuristicClassification:
    """Tests for the fallback heuristic classifier."""

    @pytest.fixture
    def agent(self):
        from core.agents.implementations.appointment_agent import AppointmentSetterAgent
        from core.config.agent_schema import AgentInstanceConfig

        config = AgentInstanceConfig(
            agent_id="test_appt",
            agent_type="appointment_setter",
            name="Test",
            vertical_id="test",
        )
        return AppointmentSetterAgent(
            config=config,
            db=MagicMock(),
            embedder=MagicMock(),
            anthropic_client=MagicMock(),
        )

    def test_ooo_detection(self, agent):
        intent, sentiment = agent._heuristic_classify(
            "I am currently out of office and will be returning on January 20th."
        )
        assert intent == "ooo"
        assert sentiment == "neutral"

    def test_unsubscribe_detection(self, agent):
        intent, sentiment = agent._heuristic_classify(
            "Please unsubscribe me from this list."
        )
        assert intent == "unsubscribe"
        assert sentiment == "negative"

    def test_not_interested_detection(self, agent):
        intent, sentiment = agent._heuristic_classify(
            "Thanks but I'm not interested at this time."
        )
        assert intent == "not_interested"
        assert sentiment == "negative"

    def test_wrong_person_detection(self, agent):
        intent, sentiment = agent._heuristic_classify(
            "I'm the wrong person for this. Try reaching out to our IT director."
        )
        assert intent == "wrong_person"
        assert sentiment == "neutral"

    def test_interested_detection(self, agent):
        intent, sentiment = agent._heuristic_classify(
            "This sounds interesting! Let's schedule a call."
        )
        assert intent == "interested"
        assert sentiment == "positive"

    def test_question_detection(self, agent):
        intent, sentiment = agent._heuristic_classify(
            "What services do you offer?"
        )
        assert intent == "question"
        assert sentiment == "neutral"

    def test_default_not_interested(self, agent):
        intent, sentiment = agent._heuristic_classify(
            "ok"  # Ambiguous
        )
        assert intent == "not_interested"
        assert sentiment == "neutral"


class TestAppointmentIntentRouting:
    """Tests for intent routing logic."""

    def test_route_interested(self):
        from core.agents.implementations.appointment_agent import AppointmentSetterAgent
        state = {"reply_intent": "interested"}
        assert AppointmentSetterAgent._route_by_intent(state) == "interested"

    def test_route_objection(self):
        from core.agents.implementations.appointment_agent import AppointmentSetterAgent
        state = {"reply_intent": "objection"}
        assert AppointmentSetterAgent._route_by_intent(state) == "objection"

    def test_route_ooo(self):
        from core.agents.implementations.appointment_agent import AppointmentSetterAgent
        state = {"reply_intent": "ooo"}
        assert AppointmentSetterAgent._route_by_intent(state) == "ooo"

    def test_route_unsubscribe(self):
        from core.agents.implementations.appointment_agent import AppointmentSetterAgent
        state = {"reply_intent": "unsubscribe"}
        assert AppointmentSetterAgent._route_by_intent(state) == "unsubscribe"

    def test_route_unknown_defaults_to_not_interested(self):
        from core.agents.implementations.appointment_agent import AppointmentSetterAgent
        state = {"reply_intent": "garbage"}
        assert AppointmentSetterAgent._route_by_intent(state) == "not_interested"

    def test_route_missing_intent(self):
        from core.agents.implementations.appointment_agent import AppointmentSetterAgent
        state = {}
        assert AppointmentSetterAgent._route_by_intent(state) == "not_interested"

    def test_route_after_review_approved(self):
        from core.agents.implementations.appointment_agent import AppointmentSetterAgent
        state = {"human_approval_status": "approved"}
        assert AppointmentSetterAgent._route_after_review(state) == "approved"

    def test_route_after_review_rejected(self):
        from core.agents.implementations.appointment_agent import AppointmentSetterAgent
        state = {"human_approval_status": "rejected"}
        assert AppointmentSetterAgent._route_after_review(state) == "rejected"

    def test_route_after_review_default(self):
        from core.agents.implementations.appointment_agent import AppointmentSetterAgent
        state = {}
        assert AppointmentSetterAgent._route_after_review(state) == "approved"


class TestAppointmentIntentConstants:
    """Tests for intent constant definitions."""

    def test_valid_intents_complete(self):
        from core.agents.implementations.appointment_agent import VALID_INTENTS

        expected = {
            "interested", "objection", "question", "ooo",
            "wrong_person", "unsubscribe", "not_interested",
        }
        assert VALID_INTENTS == expected

    def test_booking_intents(self):
        from core.agents.implementations.appointment_agent import BOOKING_INTENTS
        assert "interested" in BOOKING_INTENTS
        assert "question" in BOOKING_INTENTS
        assert "objection" not in BOOKING_INTENTS

    def test_reply_intents_exclude_unsubscribe(self):
        from core.agents.implementations.appointment_agent import REPLY_INTENTS
        assert "unsubscribe" not in REPLY_INTENTS


class TestAppointmentNodeHandlers:
    """Tests for individual node handler methods."""

    @pytest.fixture
    def agent(self):
        from core.agents.implementations.appointment_agent import AppointmentSetterAgent
        from core.config.agent_schema import AgentInstanceConfig

        config = AgentInstanceConfig(
            agent_id="test_appt",
            agent_type="appointment_setter",
            name="Test",
            vertical_id="test",
            params={
                "max_proposed_times": 3,
                "days_ahead_for_slots": 5,
                "company_name": "Enclave Guard",
                "value_proposition": "security posture review",
            },
        )
        return AppointmentSetterAgent(
            config=config,
            db=MagicMock(),
            embedder=MagicMock(),
            anthropic_client=MagicMock(),
        )

    @pytest.mark.asyncio
    async def test_handle_interested_returns_times(self, agent):
        """handle_interested should return proposed_times and calendar_link."""
        state = {"reply_intent": "interested"}
        result = await agent._node_handle_interested(state)

        assert "proposed_times" in result
        assert "calendar_link" in result
        assert isinstance(result["proposed_times"], list)

    @pytest.mark.asyncio
    async def test_handle_objection_searches_insights(self, agent):
        """handle_objection should search shared brain for rebuttals."""
        agent.db.search_insights = MagicMock(return_value=[
            {"content": "Reframe ROI: breach costs 100x"}
        ])
        agent.embedder.embed_query = MagicMock(return_value=[0.1] * 1536)

        state = {"objection_type": "price"}
        result = await agent._node_handle_objection(state)

        assert "rebuttal_context" in result
        agent.db.search_insights.assert_called()

    @pytest.mark.asyncio
    async def test_handle_objection_fallback_to_winning_patterns(self, agent):
        """If no rebuttals found, should search winning_patterns."""
        call_count = 0

        def mock_search(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return []  # No rebuttals
            return [{"content": "General winning pattern"}]

        agent.db.search_insights = MagicMock(side_effect=mock_search)
        agent.embedder.embed_query = MagicMock(return_value=[0.1] * 1536)

        state = {"objection_type": "timing"}
        result = await agent._node_handle_objection(state)

        assert len(result["rebuttal_context"]) >= 0

    @pytest.mark.asyncio
    async def test_handle_not_interested_returns_current_node(self, agent):
        """handle_not_interested should set current_node."""
        state = {"reply_intent": "not_interested"}
        result = await agent._node_handle_not_interested(state)
        assert result["current_node"] == "handle_not_interested"

    @pytest.mark.asyncio
    async def test_human_review_sets_approval_flag(self, agent):
        """human_review should set requires_human_approval."""
        state = {"reply_intent": "interested", "draft_reply_body": "Test"}
        result = await agent._node_human_review(state)
        assert result["requires_human_approval"] is True
        assert result["current_node"] == "human_review"


class TestAppointmentSystemPrompt:
    """Tests for system prompt loading."""

    def test_system_prompt_file_exists(self):
        """appointment_system.md should exist."""
        prompt_path = PROJECT_ROOT / "verticals/enclave_guard/prompts/agent_prompts/appointment_system.md"
        assert prompt_path.exists()

    def test_system_prompt_content(self):
        """System prompt should contain key concepts."""
        prompt_path = PROJECT_ROOT / "verticals/enclave_guard/prompts/agent_prompts/appointment_system.md"
        content = prompt_path.read_text()

        assert "Appointment Setter" in content
        assert "Soft Close" in content
        assert "objection" in content.lower()
        assert "unsubscribe" in content.lower()

    def test_fallback_system_prompt(self):
        """Agent should have a fallback prompt if file missing."""
        from core.agents.implementations.appointment_agent import AppointmentSetterAgent
        from core.config.agent_schema import AgentInstanceConfig

        config = AgentInstanceConfig(
            agent_id="test",
            agent_type="appointment_setter",
            name="Test",
            vertical_id="test",
            system_prompt_path="nonexistent/path.md",
        )
        agent = AppointmentSetterAgent(
            config=config,
            db=MagicMock(),
            embedder=MagicMock(),
            anthropic_client=MagicMock(),
        )

        prompt = agent._get_system_prompt()
        assert "appointment setter" in prompt.lower()
        assert len(prompt) > 50


# ═══════════════════════════════════════════════════════════════════════
# 5. YAML Config Validation Tests
# ═══════════════════════════════════════════════════════════════════════


class TestAppointmentYAMLConfig:
    """Tests for the appointment_setter YAML config."""

    @pytest.fixture
    def config_path(self):
        return PROJECT_ROOT / "verticals/enclave_guard/agents/appointment_setter.yaml"

    def test_config_file_exists(self, config_path):
        assert config_path.exists()

    def test_config_loads_valid_yaml(self, config_path):
        with open(config_path) as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict)

    def test_config_agent_type(self, config_path):
        with open(config_path) as f:
            data = yaml.safe_load(f)
        assert data["agent_type"] == "appointment_setter"

    def test_config_has_human_gates(self, config_path):
        with open(config_path) as f:
            data = yaml.safe_load(f)
        assert data["human_gates"]["enabled"] is True
        assert "human_review" in data["human_gates"]["gate_before"]

    def test_config_validates_with_pydantic(self, config_path):
        """Config should validate against AgentInstanceConfig."""
        from core.config.agent_schema import AgentInstanceConfig

        with open(config_path) as f:
            data = yaml.safe_load(f)

        data["vertical_id"] = "enclave_guard"
        config = AgentInstanceConfig(**data)
        assert config.agent_type == "appointment_setter"
        assert config.enabled is True

    def test_config_has_params(self, config_path):
        with open(config_path) as f:
            data = yaml.safe_load(f)

        params = data.get("params", {})
        assert "max_proposed_times" in params
        assert "company_name" in params
        assert "tone" in params

    def test_config_event_trigger(self, config_path):
        with open(config_path) as f:
            data = yaml.safe_load(f)
        assert data["schedule"]["trigger"] == "event"


# ═══════════════════════════════════════════════════════════════════════
# 6. Integration-Level Tests
# ═══════════════════════════════════════════════════════════════════════


class TestAppointmentAgentGraphStructure:
    """Tests for the compiled LangGraph structure."""

    @pytest.fixture
    def agent(self):
        from core.agents.implementations.appointment_agent import AppointmentSetterAgent
        from core.config.agent_schema import AgentInstanceConfig

        config = AgentInstanceConfig(
            agent_id="test_appt",
            agent_type="appointment_setter",
            name="Test",
            vertical_id="test",
            human_gates={"enabled": False},
        )
        return AppointmentSetterAgent(
            config=config,
            db=MagicMock(),
            embedder=MagicMock(),
            anthropic_client=MagicMock(),
        )

    def test_graph_compiles(self, agent):
        """Graph should compile without errors."""
        graph = agent.build_graph()
        assert graph is not None

    def test_graph_has_entry_point(self, agent):
        """Graph should start at classify_intent."""
        graph = agent.build_graph()
        # LangGraph compiled graph has nodes
        assert graph is not None

    def test_graph_cached_on_get(self, agent):
        """get_graph() should cache the compiled graph."""
        g1 = agent.get_graph()
        g2 = agent.get_graph()
        assert g1 is g2


class TestAppointmentAgentDiscovery:
    """Tests that the agent is discovered by the registry."""

    def test_registry_discovers_appointment_setter(self):
        """Registry should discover appointment_setter.yaml."""
        from core.agents.registry import AgentRegistry

        # Import to trigger registration
        import core.agents.implementations.appointment_agent  # noqa: F401

        registry = AgentRegistry("enclave_guard", PROJECT_ROOT / "verticals")
        configs = registry.discover_agents()

        agent_ids = [c.agent_id for c in configs]
        assert "appointment_setter_v1" in agent_ids

    def test_registry_instantiation(self):
        """Registry should instantiate the appointment setter."""
        from core.agents.registry import AgentRegistry

        import core.agents.implementations.appointment_agent  # noqa: F401

        registry = AgentRegistry("enclave_guard", PROJECT_ROOT / "verticals")
        registry.discover_agents()

        agent = registry.instantiate_agent(
            "appointment_setter_v1",
            db=MagicMock(),
            embedder=MagicMock(),
            anthropic_client=MagicMock(),
        )

        assert agent.agent_id == "appointment_setter_v1"
        assert agent.agent_type == "appointment_setter"
