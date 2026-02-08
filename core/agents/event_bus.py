"""
Event routing for the Sovereign Venture Engine.

Maps triggers (new_lead, email_reply, schedule_tick) to target agents
via the task queue. Event-driven coordination — not polling.

Usage:
    bus = EventBus(task_queue, registry)
    bus.register("email_reply_received", "appointment_setter", "handle_reply")
    bus.register("new_lead_added", "outreach", "process_lead")

    # When an event occurs:
    bus.dispatch("email_reply_received", {"email": "...", "body": "..."})
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class EventRoute:
    """Maps an event type to a target agent and task type."""

    event_type: str
    target_agent_id: str
    task_type: str
    priority: int = 5
    description: str = ""


class EventBus:
    """
    Routes events to agents via the task queue.

    Each event type can have multiple routes (fan-out).
    """

    def __init__(self, task_queue: Any, registry: Any = None):
        self.task_queue = task_queue
        self.registry = registry
        self._routes: dict[str, list[EventRoute]] = {}

    def register(
        self,
        event_type: str,
        target_agent_id: str,
        task_type: str,
        priority: int = 5,
        description: str = "",
    ) -> None:
        """Register a route from an event type to a target agent."""
        route = EventRoute(
            event_type=event_type,
            target_agent_id=target_agent_id,
            task_type=task_type,
            priority=priority,
            description=description,
        )
        if event_type not in self._routes:
            self._routes[event_type] = []
        self._routes[event_type].append(route)
        logger.info(
            f"Event route registered: {event_type} → "
            f"{target_agent_id}.{task_type}"
        )

    def dispatch(
        self,
        event_type: str,
        payload: dict[str, Any],
        source_agent_id: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """
        Dispatch an event to all registered routes.

        Creates a task in the queue for each matching route (fan-out).

        Returns:
            List of created task records.
        """
        routes = self._routes.get(event_type, [])
        if not routes:
            logger.debug(f"No routes registered for event: {event_type}")
            return []

        results = []
        for route in routes:
            try:
                task = self.task_queue.enqueue(
                    target_agent_id=route.target_agent_id,
                    task_type=route.task_type,
                    input_data=payload,
                    source_agent_id=source_agent_id,
                    priority=route.priority,
                )
                results.append(task)
            except Exception as e:
                logger.error(
                    f"Failed to dispatch {event_type} → "
                    f"{route.target_agent_id}: {e}"
                )

        logger.info(
            f"Event dispatched: {event_type} → "
            f"{len(results)}/{len(routes)} tasks created"
        )
        return results

    def list_routes(self) -> dict[str, list[EventRoute]]:
        """Return all registered routes."""
        return dict(self._routes)

    def get_routes_for_event(self, event_type: str) -> list[EventRoute]:
        """Return routes for a specific event type."""
        return self._routes.get(event_type, [])
