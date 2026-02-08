"""
Calendar MCP tools for the Sovereign Venture Engine.

Wraps CalendarClient as MCP-compatible tool functions.
book_meeting_slot is decorated with @sandboxed_tool — in non-production
environments, the booking is intercepted and logged instead of touching
any real calendar.

Tools:
    check_calendar_availability — Check if a proposed time slot is available
    get_available_slots — List available meeting slots for the next N days
    book_meeting_slot — Book a meeting (sandboxed in non-production)
    get_booking_link — Return the self-service booking URL (Calendly/Cal.com)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional

from core.safety.sandbox import sandboxed_tool

logger = logging.getLogger(__name__)


def _get_calendar_client(_calendar_client: Any = None) -> Any:
    """Lazily construct a CalendarClient from env if not injected."""
    if _calendar_client is not None:
        return _calendar_client
    from core.integrations.calendar_client import CalendarClient

    return CalendarClient.from_env()


async def check_calendar_availability(
    proposed_datetime: str,
    *,
    _calendar_client: Any = None,
) -> str:
    """
    Check if a proposed date/time is available for a meeting.

    Args:
        proposed_datetime: ISO 8601 datetime string (e.g. "2025-01-15T10:00:00").
        _calendar_client: Injected CalendarClient instance (for testing/DI).

    Returns:
        JSON string with availability status and suggested alternatives.
    """
    client = _get_calendar_client(_calendar_client)

    logger.info(
        "mcp_tool_called",
        extra={
            "tool_name": "check_calendar_availability",
            "proposed_datetime": proposed_datetime,
        },
    )

    try:
        proposed_time = datetime.fromisoformat(proposed_datetime)
    except (ValueError, TypeError) as e:
        return json.dumps({
            "available": False,
            "error": f"Invalid datetime format: {e}. Use ISO 8601 (e.g. 2025-01-15T10:00:00).",
        })

    is_available = await client.check_availability(proposed_time)

    result: dict[str, Any] = {
        "proposed_time": proposed_datetime,
        "available": is_available,
    }

    # If not available, suggest alternatives
    if not is_available:
        alternatives = await client.get_available_slots(days_ahead=3, max_slots=3)
        result["suggested_alternatives"] = alternatives
        result["booking_link"] = client.get_booking_link()

    logger.info(
        "calendar_availability_checked",
        extra={
            "tool_name": "check_calendar_availability",
            "available": is_available,
        },
    )

    return json.dumps(result, indent=2)


async def get_available_slots(
    days_ahead: int = 5,
    max_slots: int = 6,
    *,
    _calendar_client: Any = None,
) -> str:
    """
    Get available meeting time slots for the next N business days.

    Args:
        days_ahead: Number of days to look ahead (default 5).
        max_slots: Maximum number of slots to return (default 6).
        _calendar_client: Injected CalendarClient instance (for testing/DI).

    Returns:
        JSON string with available time slots and booking link.
    """
    client = _get_calendar_client(_calendar_client)

    logger.info(
        "mcp_tool_called",
        extra={
            "tool_name": "get_available_slots",
            "days_ahead": days_ahead,
            "max_slots": max_slots,
        },
    )

    slots = await client.get_available_slots(
        days_ahead=min(days_ahead, 14),  # Cap at 2 weeks
        max_slots=min(max_slots, 20),  # Cap at 20 slots
    )

    result = {
        "slots": slots,
        "slot_count": len(slots),
        "booking_link": client.get_booking_link(),
        "meeting_duration_minutes": client.meeting_duration_minutes,
    }

    logger.info(
        "calendar_slots_retrieved",
        extra={
            "tool_name": "get_available_slots",
            "slot_count": len(slots),
        },
    )

    return json.dumps(result, indent=2)


@sandboxed_tool("book_meeting")
async def book_meeting_slot(
    attendee_email: str,
    attendee_name: str,
    start_datetime: str,
    title: str = "",
    description: str = "",
    *,
    _calendar_client: Any = None,
) -> dict[str, Any]:
    """
    Book a meeting slot with a prospect.

    SAFETY: This tool is wrapped with @sandboxed_tool("book_meeting").
    In development/staging/test environments, the call is intercepted
    and logged to sandbox_logs/book_meeting.jsonl instead of booking.

    Args:
        attendee_email: The prospect's email address.
        attendee_name: The prospect's full name.
        start_datetime: ISO 8601 datetime for the meeting start.
        title: Meeting title (default: "Discovery Call — Enclave Guard").
        description: Meeting description/agenda.
        _calendar_client: Injected CalendarClient instance (for testing/DI).

    Returns:
        Dict with booking confirmation, calendar link, and meeting details.
    """
    client = _get_calendar_client(_calendar_client)

    logger.info(
        "mcp_tool_called",
        extra={
            "tool_name": "book_meeting_slot",
            "attendee_email": attendee_email,
            "start_datetime": start_datetime,
        },
    )

    try:
        start_time = datetime.fromisoformat(start_datetime)
    except (ValueError, TypeError) as e:
        return {
            "booked": False,
            "error": f"Invalid datetime format: {e}. Use ISO 8601.",
        }

    # Check availability first
    is_available = await client.check_availability(start_time)
    if not is_available:
        alternatives = await client.get_available_slots(days_ahead=3, max_slots=3)
        return {
            "booked": False,
            "error": "Requested time slot is not available.",
            "suggested_alternatives": alternatives,
            "booking_link": client.get_booking_link(),
        }

    booking = await client.book_meeting(
        attendee_email=attendee_email,
        start_time=start_time,
        title=title,
        description=description,
        attendee_name=attendee_name,
    )

    logger.info(
        "meeting_booked",
        extra={
            "tool_name": "book_meeting_slot",
            "booking_id": booking.get("booking_id", "")[:8],
            "attendee_email": attendee_email,
        },
    )

    return {
        "booked": True,
        **booking,
    }


async def get_booking_link(
    *,
    _calendar_client: Any = None,
) -> str:
    """
    Get the self-service booking link (Calendly/Cal.com fallback).

    Returns the URL that prospects can use to self-schedule a meeting.

    Args:
        _calendar_client: Injected CalendarClient instance (for testing/DI).

    Returns:
        JSON string with the booking URL.
    """
    client = _get_calendar_client(_calendar_client)

    return json.dumps({
        "booking_link": client.get_booking_link(),
        "message": "Share this link with the prospect for self-service booking.",
    })
