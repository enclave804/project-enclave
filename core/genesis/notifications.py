"""
Genesis Engine — Notification Layer.

Sends Telegram alerts at key points in the Genesis flow:
    - Interview complete
    - Blueprint generated / approved / rejected
    - Configs generated
    - Credentials collected
    - Launch complete / failed

This module is best-effort: notification failures never block the flow.

Usage:
    from core.genesis.notifications import GenesisNotifier

    notifier = GenesisNotifier(telegram_bot)
    await notifier.on_blueprint_ready(vertical_id, blueprint_name)
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class GenesisNotifier:
    """
    Push notifications for Genesis Engine lifecycle events.

    Wraps the TelegramBot.send_alert() method with Genesis-specific
    message formatting. All methods are best-effort and never raise.
    """

    def __init__(self, telegram_bot: Optional[Any] = None):
        self._bot = telegram_bot

    async def _send(self, message: str) -> bool:
        """Send a notification, silently failing on errors."""
        if self._bot is None:
            logger.debug("genesis_notify_skip: No Telegram bot configured")
            return False

        try:
            return await self._bot.send_alert(message)
        except Exception as e:
            logger.warning(f"genesis_notify_failed: {e}")
            return False

    # ------------------------------------------------------------------
    # Lifecycle Events
    # ------------------------------------------------------------------

    async def on_interview_complete(
        self,
        vertical_id: str,
        question_count: int,
        session_id: Optional[str] = None,
    ) -> bool:
        """Notify when an interview is complete."""
        return await self._send(
            f"📋 *Genesis Interview Complete*\n\n"
            f"Vertical: `{vertical_id}`\n"
            f"Questions answered: {question_count}\n"
            f"Moving to blueprint generation..."
            + (f"\nSession: `{session_id[:8]}...`" if session_id else "")
        )

    async def on_blueprint_ready(
        self,
        vertical_id: str,
        blueprint_name: str,
    ) -> bool:
        """Notify when a blueprint is ready for review."""
        return await self._send(
            f"📊 *Blueprint Ready for Review*\n\n"
            f"Vertical: `{vertical_id}`\n"
            f"Blueprint: _{blueprint_name}_\n\n"
            f"Review it in the Sovereign Cockpit to approve or request changes."
        )

    async def on_blueprint_approved(
        self,
        vertical_id: str,
        blueprint_name: str,
    ) -> bool:
        """Notify when a blueprint is approved."""
        return await self._send(
            f"✅ *Blueprint Approved*\n\n"
            f"Vertical: `{vertical_id}`\n"
            f"Blueprint: _{blueprint_name}_\n\n"
            f"Generating agent configurations..."
        )

    async def on_configs_generated(
        self,
        vertical_id: str,
        agent_count: int,
    ) -> bool:
        """Notify when configs are generated."""
        return await self._send(
            f"⚙️ *Configs Generated*\n\n"
            f"Vertical: `{vertical_id}`\n"
            f"Agents configured: {agent_count}\n\n"
            f"Waiting for credential collection."
        )

    async def on_launch_success(
        self,
        vertical_id: str,
        agent_count: int,
        agent_ids: list[str],
    ) -> bool:
        """Notify when a vertical is successfully launched."""
        agents_list = "\n".join(f"  • `{aid}`" for aid in agent_ids[:5])
        return await self._send(
            f"🚀 *Vertical Launched!*\n\n"
            f"Vertical: `{vertical_id}`\n"
            f"Status: *Shadow Mode* 👻\n"
            f"Active agents ({agent_count}):\n{agents_list}\n\n"
            f"All actions are sandboxed until you promote to live."
        )

    async def on_launch_failed(
        self,
        vertical_id: str,
        errors: list[str],
    ) -> bool:
        """Notify when a launch fails."""
        errors_list = "\n".join(f"  ❌ {e}" for e in errors[:3])
        return await self._send(
            f"⚠️ *Launch Failed*\n\n"
            f"Vertical: `{vertical_id}`\n"
            f"Errors:\n{errors_list}\n\n"
            f"Check the dashboard for details."
        )

    async def on_promotion(
        self,
        vertical_id: str,
    ) -> bool:
        """Notify when a vertical is promoted from shadow to live."""
        return await self._send(
            f"🟢 *Vertical Promoted to LIVE*\n\n"
            f"Vertical: `{vertical_id}`\n\n"
            f"Agents are now sending real emails and making real API calls. "
            f"Monitor closely for the first 24 hours."
        )
