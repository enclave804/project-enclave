"""
Cross-agent task coordination for the Sovereign Venture Engine.

Manages a Supabase-backed task queue that allows agents to create work
for other agents. Uses atomic claiming (FOR UPDATE SKIP LOCKED) to
prevent double-processing.

Usage:
    queue = TaskQueueManager(db)
    queue.enqueue("appointment_setter", "handle_reply", {...}, source="outreach")
    task = queue.claim("appointment_setter")
    if task:
        # process...
        queue.complete(task["task_id"], output_data={...})
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)


class TaskQueueManager:
    """
    Manages cross-agent task coordination via Supabase.

    Tasks flow: enqueue → claim → running → complete/fail

    Includes heartbeat support to prevent zombie tasks: agents call
    heartbeat(task_id) periodically, and a cleanup function recovers
    tasks whose heartbeat went stale (agent crashed).
    """

    def __init__(self, db: Any):
        self.db = db

    def enqueue(
        self,
        target_agent_id: str,
        task_type: str,
        input_data: dict[str, Any],
        source_agent_id: Optional[str] = None,
        priority: int = 5,
        scheduled_at: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        Add a task to the queue for a target agent.

        Also dispatches shadow copies: if any enabled agent has
        shadow_of=target_agent_id, a duplicate task is enqueued
        for the shadow with shadow_mode=True in metadata.

        Args:
            target_agent_id: Agent that should execute this task.
            task_type: Agent-specific task category.
            input_data: Task payload.
            source_agent_id: Agent that created this task (None = manual).
            priority: 1 = highest, 10 = lowest.
            scheduled_at: ISO timestamp for delayed execution.
            metadata: Optional task metadata dict.

        Returns:
            Created task record (for the primary task).
        """
        task_id = str(uuid.uuid4())
        data: dict[str, Any] = {
            "task_id": task_id,
            "source_agent_id": source_agent_id,
            "target_agent_id": target_agent_id,
            "task_type": task_type,
            "priority": priority,
            "status": "pending",
            "input_data": input_data,
            "scheduled_at": scheduled_at,
        }
        if metadata:
            data["metadata"] = metadata

        result = self.db.enqueue_task(data)
        logger.info(
            f"Task enqueued: {task_id[:8]}... "
            f"({source_agent_id or 'manual'} → {target_agent_id}, type={task_type})"
        )

        # ── Shadow Dispatch (God Mode Lite) ──────────────────────
        # Don't create shadow copies of tasks that are already shadow tasks
        is_shadow_task = (metadata or {}).get("shadow_mode", False)
        if not is_shadow_task:
            self._dispatch_to_shadows(
                target_agent_id=target_agent_id,
                task_type=task_type,
                input_data=input_data,
                source_agent_id=source_agent_id,
                priority=priority,
                scheduled_at=scheduled_at,
            )

        return result

    def _dispatch_to_shadows(
        self,
        target_agent_id: str,
        task_type: str,
        input_data: dict[str, Any],
        source_agent_id: Optional[str] = None,
        priority: int = 5,
        scheduled_at: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """
        Find shadow agents for a champion and duplicate the task for each.

        Shadow tasks are marked with metadata.shadow_mode=True so the
        sandbox protocol can intercept external tool calls.

        Returns:
            List of created shadow task records.
        """
        try:
            shadows = self.db.get_shadow_agents(target_agent_id)
        except Exception:
            # If the DB method doesn't exist yet or fails, silently skip
            return []

        results = []
        for shadow in shadows:
            shadow_agent_id = shadow.get("agent_id")
            if not shadow_agent_id:
                continue

            shadow_task_id = str(uuid.uuid4())
            shadow_data: dict[str, Any] = {
                "task_id": shadow_task_id,
                "source_agent_id": source_agent_id,
                "target_agent_id": shadow_agent_id,
                "task_type": task_type,
                "priority": priority,
                "status": "pending",
                "input_data": input_data,
                "scheduled_at": scheduled_at,
                "metadata": {
                    "shadow_mode": True,
                    "champion_agent_id": target_agent_id,
                },
            }

            try:
                shadow_result = self.db.enqueue_task(shadow_data)
                results.append(shadow_result)
                logger.info(
                    "shadow_task_enqueued",
                    extra={
                        "champion_agent": target_agent_id,
                        "shadow_agent": shadow_agent_id,
                        "task_id": shadow_task_id[:8],
                    },
                )
            except Exception as e:
                logger.warning(
                    "shadow_dispatch_failed",
                    extra={
                        "shadow_agent": shadow_agent_id,
                        "error": str(e)[:200],
                    },
                )

        return results

    def claim(self, agent_id: str) -> Optional[dict[str, Any]]:
        """
        Atomically claim the next pending task for an agent.

        Uses database-level locking (FOR UPDATE SKIP LOCKED) to prevent
        double-processing in concurrent environments.

        Returns:
            Task record if available, None if queue is empty.
        """
        task = self.db.claim_next_task(agent_id)
        if task:
            logger.info(
                f"Task claimed by {agent_id}: {task.get('task_id', '?')[:8]}... "
                f"(type={task.get('task_type', '?')})"
            )
        return task

    def complete(
        self,
        task_id: str,
        output_data: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Mark a task as completed with optional output data."""
        result = self.db.complete_task(task_id, output_data=output_data)
        logger.info(f"Task completed: {task_id[:8]}...")
        return result

    def fail(
        self,
        task_id: str,
        error_message: str,
        retry: bool = True,
    ) -> dict[str, Any]:
        """Mark a task as failed, optionally re-enqueue for retry."""
        result = self.db.fail_task(task_id, error_message=error_message, retry=retry)
        action = "retrying" if retry else "permanent failure"
        logger.warning(f"Task failed ({action}): {task_id[:8]}... — {error_message[:100]}")
        return result

    def get_pending_count(self, agent_id: str) -> int:
        """Get the number of pending tasks for an agent."""
        return self.db.count_pending_tasks(agent_id)

    def heartbeat(self, task_id: str) -> None:
        """
        Update the heartbeat timestamp for a running task.

        Agents should call this periodically during long-running tasks.
        The supervisor uses stale heartbeats to detect zombie tasks.
        """
        try:
            self.db.heartbeat_task(task_id)
        except Exception as e:
            logger.warning(f"Failed to update heartbeat for {task_id[:8]}...: {e}")

    def recover_zombies(self, stale_minutes: int = 10) -> int:
        """
        Recover zombie tasks (claimed/running but no heartbeat).

        Returns the number of tasks recovered back to pending.
        """
        count = self.db.recover_zombie_tasks(stale_minutes)
        if count > 0:
            logger.warning(f"Recovered {count} zombie tasks (stale > {stale_minutes}min)")
        return count

    def list_tasks(
        self,
        agent_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """List tasks with optional filters."""
        return self.db.list_tasks(
            agent_id=agent_id, status=status, limit=limit
        )
