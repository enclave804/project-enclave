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
    ) -> dict[str, Any]:
        """
        Add a task to the queue for a target agent.

        Args:
            target_agent_id: Agent that should execute this task.
            task_type: Agent-specific task category.
            input_data: Task payload.
            source_agent_id: Agent that created this task (None = manual).
            priority: 1 = highest, 10 = lowest.
            scheduled_at: ISO timestamp for delayed execution.

        Returns:
            Created task record.
        """
        task_id = str(uuid.uuid4())
        data = {
            "task_id": task_id,
            "source_agent_id": source_agent_id,
            "target_agent_id": target_agent_id,
            "task_type": task_type,
            "priority": priority,
            "status": "pending",
            "input_data": input_data,
            "scheduled_at": scheduled_at,
        }

        result = self.db.enqueue_task(data)
        logger.info(
            f"Task enqueued: {task_id[:8]}... "
            f"({source_agent_id or 'manual'} → {target_agent_id}, type={task_type})"
        )
        return result

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
