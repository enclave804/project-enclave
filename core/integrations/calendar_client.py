"""
Calendar integration for the Sovereign Venture Engine.

Provider-agnostic calendar client that supports:
- Google Calendar (via service account or OAuth)
- Cal.com (via API key)
- Mock mode (for development/testing)

The active provider is controlled via CALENDAR_PROVIDER env var.
Default: "mock" (safe for development — never touches real calendars).

Usage:
    client = CalendarClient.from_env()
    slots = await client.get_available_slots(days_ahead=5)
    available = await client.check_availability(proposed_time)
    booking = await client.book_meeting(email, time, context)
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)


class CalendarProvider(ABC):
    """Abstract calendar provider interface."""

    @abstractmethod
    async def get_available_slots(
        self,
        start_date: datetime,
        end_date: datetime,
        duration_minutes: int = 30,
        timezone_str: str = "America/New_York",
    ) -> list[dict[str, Any]]:
        """Return available time slots in the given range."""
        ...

    @abstractmethod
    async def check_availability(
        self,
        proposed_time: datetime,
        duration_minutes: int = 30,
    ) -> bool:
        """Check if a specific time slot is available."""
        ...

    @abstractmethod
    async def book_meeting(
        self,
        attendee_email: str,
        start_time: datetime,
        duration_minutes: int = 30,
        title: str = "",
        description: str = "",
        attendee_name: str = "",
        timezone_str: str = "America/New_York",
    ) -> dict[str, Any]:
        """Book a meeting and send an invite."""
        ...


class MockCalendarProvider(CalendarProvider):
    """
    Mock calendar provider for development and testing.

    Generates realistic-looking availability slots without touching
    any real calendar. Bookings are logged but not persisted externally.
    """

    def __init__(self):
        self._bookings: list[dict[str, Any]] = []
        self._blocked_times: set[str] = set()

    async def get_available_slots(
        self,
        start_date: datetime,
        end_date: datetime,
        duration_minutes: int = 30,
        timezone_str: str = "America/New_York",
    ) -> list[dict[str, Any]]:
        """Generate mock available slots (9 AM - 5 PM, weekdays only)."""
        slots = []
        current = start_date.replace(hour=9, minute=0, second=0, microsecond=0)

        while current < end_date:
            # Skip weekends
            if current.weekday() < 5:  # Mon-Fri
                for hour in [9, 10, 11, 13, 14, 15, 16]:
                    slot_time = current.replace(hour=hour, minute=0)
                    if slot_time >= start_date and slot_time <= end_date:
                        iso = slot_time.isoformat()
                        if iso not in self._blocked_times:
                            slots.append({
                                "start": iso,
                                "end": (slot_time + timedelta(minutes=duration_minutes)).isoformat(),
                                "timezone": timezone_str,
                                "available": True,
                            })
            current += timedelta(days=1)

        return slots[:20]  # Cap at 20 slots

    async def check_availability(
        self,
        proposed_time: datetime,
        duration_minutes: int = 30,
    ) -> bool:
        """Check mock availability (all weekday business hours are free)."""
        if proposed_time.weekday() >= 5:  # Weekend
            return False
        if proposed_time.hour < 9 or proposed_time.hour >= 17:  # Outside hours
            return False
        return proposed_time.isoformat() not in self._blocked_times

    async def book_meeting(
        self,
        attendee_email: str,
        start_time: datetime,
        duration_minutes: int = 30,
        title: str = "",
        description: str = "",
        attendee_name: str = "",
        timezone_str: str = "America/New_York",
    ) -> dict[str, Any]:
        """Create a mock booking."""
        import uuid

        booking_id = str(uuid.uuid4())
        end_time = start_time + timedelta(minutes=duration_minutes)

        booking = {
            "booking_id": booking_id,
            "status": "confirmed",
            "attendee_email": attendee_email,
            "attendee_name": attendee_name,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_minutes": duration_minutes,
            "title": title or "Discovery Call — Enclave Guard",
            "description": description,
            "timezone": timezone_str,
            "calendar_link": f"https://cal.enclaveguard.com/booking/{booking_id[:8]}",
            "provider": "mock",
        }

        self._bookings.append(booking)
        self._blocked_times.add(start_time.isoformat())

        logger.info(
            "mock_meeting_booked",
            extra={
                "booking_id": booking_id[:8],
                "attendee": attendee_email,
                "time": start_time.isoformat(),
            },
        )

        return booking


class GoogleCalendarProvider(CalendarProvider):
    """
    Google Calendar provider via service account.

    Requires:
        GOOGLE_CALENDAR_CREDENTIALS_PATH: Path to service account JSON
        GOOGLE_CALENDAR_ID: Calendar ID to manage

    Placeholder for production implementation. The interface is stable;
    swap MockCalendarProvider for this when ready.
    """

    def __init__(
        self,
        credentials_path: Optional[str] = None,
        calendar_id: Optional[str] = None,
    ):
        self.credentials_path = credentials_path or os.environ.get(
            "GOOGLE_CALENDAR_CREDENTIALS_PATH", ""
        )
        self.calendar_id = calendar_id or os.environ.get(
            "GOOGLE_CALENDAR_ID", "primary"
        )

    async def get_available_slots(
        self,
        start_date: datetime,
        end_date: datetime,
        duration_minutes: int = 30,
        timezone_str: str = "America/New_York",
    ) -> list[dict[str, Any]]:
        """Query Google Calendar freebusy API for available slots."""
        # TODO: Implement with google-api-python-client
        # For now, fall back to mock behavior with a warning
        logger.warning(
            "google_calendar_not_implemented: falling back to mock slots"
        )
        mock = MockCalendarProvider()
        return await mock.get_available_slots(
            start_date, end_date, duration_minutes, timezone_str
        )

    async def check_availability(
        self,
        proposed_time: datetime,
        duration_minutes: int = 30,
    ) -> bool:
        logger.warning("google_calendar_not_implemented: falling back to mock")
        mock = MockCalendarProvider()
        return await mock.check_availability(proposed_time, duration_minutes)

    async def book_meeting(
        self,
        attendee_email: str,
        start_time: datetime,
        duration_minutes: int = 30,
        title: str = "",
        description: str = "",
        attendee_name: str = "",
        timezone_str: str = "America/New_York",
    ) -> dict[str, Any]:
        logger.warning("google_calendar_not_implemented: falling back to mock")
        mock = MockCalendarProvider()
        return await mock.book_meeting(
            attendee_email, start_time, duration_minutes,
            title, description, attendee_name, timezone_str,
        )


class CalComProvider(CalendarProvider):
    """
    Cal.com provider via REST API.

    Requires:
        CALCOM_API_KEY: Cal.com API key
        CALCOM_EVENT_TYPE_ID: Event type for discovery calls

    Placeholder for production implementation.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        event_type_id: Optional[int] = None,
    ):
        self.api_key = api_key or os.environ.get("CALCOM_API_KEY", "")
        self.event_type_id = event_type_id or int(
            os.environ.get("CALCOM_EVENT_TYPE_ID", "0")
        )

    async def get_available_slots(
        self,
        start_date: datetime,
        end_date: datetime,
        duration_minutes: int = 30,
        timezone_str: str = "America/New_York",
    ) -> list[dict[str, Any]]:
        logger.warning("calcom_not_implemented: falling back to mock")
        mock = MockCalendarProvider()
        return await mock.get_available_slots(
            start_date, end_date, duration_minutes, timezone_str
        )

    async def check_availability(
        self,
        proposed_time: datetime,
        duration_minutes: int = 30,
    ) -> bool:
        logger.warning("calcom_not_implemented: falling back to mock")
        mock = MockCalendarProvider()
        return await mock.check_availability(proposed_time, duration_minutes)

    async def book_meeting(
        self,
        attendee_email: str,
        start_time: datetime,
        duration_minutes: int = 30,
        title: str = "",
        description: str = "",
        attendee_name: str = "",
        timezone_str: str = "America/New_York",
    ) -> dict[str, Any]:
        logger.warning("calcom_not_implemented: falling back to mock")
        mock = MockCalendarProvider()
        return await mock.book_meeting(
            attendee_email, start_time, duration_minutes,
            title, description, attendee_name, timezone_str,
        )


# ─── Factory ────────────────────────────────────────────────────────────

PROVIDERS = {
    "mock": MockCalendarProvider,
    "google": GoogleCalendarProvider,
    "calcom": CalComProvider,
}


class CalendarClient:
    """
    High-level calendar client with provider selection.

    The provider is determined by CALENDAR_PROVIDER env var:
    - "mock" (default): Safe for development
    - "google": Google Calendar via service account
    - "calcom": Cal.com via API key

    Usage:
        client = CalendarClient.from_env()
        slots = await client.get_available_slots(days_ahead=5)
    """

    def __init__(
        self,
        provider: Optional[CalendarProvider] = None,
        default_timezone: str = "America/New_York",
        meeting_duration_minutes: int = 30,
        booking_link: str = "",
    ):
        self.provider = provider or MockCalendarProvider()
        self.default_timezone = default_timezone
        self.meeting_duration_minutes = meeting_duration_minutes
        self.booking_link = booking_link or os.environ.get(
            "BOOKING_LINK", "https://cal.enclaveguard.com"
        )

    @classmethod
    def from_env(cls) -> CalendarClient:
        """Create a CalendarClient from environment variables."""
        provider_name = os.environ.get("CALENDAR_PROVIDER", "mock").lower()
        provider_cls = PROVIDERS.get(provider_name, MockCalendarProvider)

        if provider_name not in PROVIDERS:
            logger.warning(
                f"Unknown calendar provider '{provider_name}', using mock"
            )

        return cls(
            provider=provider_cls(),
            default_timezone=os.environ.get(
                "DEFAULT_TIMEZONE", "America/New_York"
            ),
            meeting_duration_minutes=int(
                os.environ.get("MEETING_DURATION_MINUTES", "30")
            ),
        )

    async def get_available_slots(
        self,
        days_ahead: int = 5,
        max_slots: int = 6,
    ) -> list[dict[str, Any]]:
        """Get available slots for the next N business days."""
        now = datetime.now(timezone.utc)
        # Start from next business hour
        start = now + timedelta(hours=1)
        end = now + timedelta(days=days_ahead)

        slots = await self.provider.get_available_slots(
            start_date=start,
            end_date=end,
            duration_minutes=self.meeting_duration_minutes,
            timezone_str=self.default_timezone,
        )
        return slots[:max_slots]

    async def check_availability(
        self, proposed_time: datetime
    ) -> bool:
        """Check if a proposed time is available."""
        return await self.provider.check_availability(
            proposed_time=proposed_time,
            duration_minutes=self.meeting_duration_minutes,
        )

    async def book_meeting(
        self,
        attendee_email: str,
        start_time: datetime,
        title: str = "",
        description: str = "",
        attendee_name: str = "",
    ) -> dict[str, Any]:
        """Book a meeting with a prospect."""
        return await self.provider.book_meeting(
            attendee_email=attendee_email,
            start_time=start_time,
            duration_minutes=self.meeting_duration_minutes,
            title=title or "Discovery Call — Enclave Guard",
            description=description,
            attendee_name=attendee_name,
            timezone_str=self.default_timezone,
        )

    def get_booking_link(self) -> str:
        """Return the self-service booking link (Calendly/Cal.com fallback)."""
        return self.booking_link
