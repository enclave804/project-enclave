"""
Sovereign Venture Engine — Telegram ChatOps Interface

Provides operational control of the agent fleet via Telegram commands.
Designed for the founder/operator to monitor and control agents from mobile.

Security:
- ALLOWED_USER_IDS whitelist (load from .env)
- All commands from non-whitelisted users are silently ignored
- Rate limiting on command execution

Commands:
    /status              — System health overview
    /agents              — List all agents with status
    /pause <agent_id>    — Pause a specific agent
    /resume <agent_id>   — Resume a specific agent
    /pause_all           — Emergency: pause all agents
    /resume_all          — Resume all agents
    /queue               — Show task queue depth
    /approvals           — Show items awaiting review
    /errors              — Show recent failures
    /help                — Show available commands

Event push:
    Agents can call send_alert(message) via the EventBus to push
    notifications to the operator's Telegram.

Usage:
    # In .env:
    TELEGRAM_BOT_TOKEN=your_bot_token
    TELEGRAM_ALLOWED_USER_IDS=123456789,987654321

    # Start:
    from core.interfaces.telegram_bot import TelegramBot
    bot = TelegramBot(db, vertical_id="enclave_guard")
    await bot.start()
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)


class TelegramBot:
    """
    Telegram ChatOps interface for the Sovereign Venture Engine.

    Wraps python-telegram-bot with security controls and DB integration.
    Can run standalone or be started alongside the agent supervisor.
    """

    def __init__(
        self,
        db: Any,
        vertical_id: str = "enclave_guard",
        token: Optional[str] = None,
        allowed_user_ids: Optional[list[int]] = None,
    ):
        self.db = db
        self.vertical_id = vertical_id
        self.token = token or os.environ.get("TELEGRAM_BOT_TOKEN", "")
        self._app = None  # telegram.ext.Application (lazy init)

        # Security: whitelist of allowed Telegram user IDs
        if allowed_user_ids is not None:
            self.allowed_user_ids = set(allowed_user_ids)
        else:
            raw = os.environ.get("TELEGRAM_ALLOWED_USER_IDS", "")
            self.allowed_user_ids = set()
            if raw:
                for uid in raw.split(","):
                    uid = uid.strip()
                    if uid.isdigit():
                        self.allowed_user_ids.add(int(uid))

        # Chat ID for push alerts (set on first /start or /status)
        self._alert_chat_id: Optional[int] = None

    # ------------------------------------------------------------------
    # Security
    # ------------------------------------------------------------------

    def _is_authorized(self, user_id: int) -> bool:
        """Check if a Telegram user is on the whitelist."""
        if not self.allowed_user_ids:
            # If no whitelist configured, deny all (fail-closed)
            logger.warning(
                "telegram_no_whitelist: TELEGRAM_ALLOWED_USER_IDS not configured. "
                "All commands denied."
            )
            return False
        return user_id in self.allowed_user_ids

    # ------------------------------------------------------------------
    # Bot Setup
    # ------------------------------------------------------------------

    def _build_app(self):
        """Build the telegram Application with command handlers."""
        try:
            from telegram.ext import (
                Application,
                CommandHandler,
                MessageHandler,
                filters,
            )
        except ImportError:
            raise ImportError(
                "python-telegram-bot is required. Install with: "
                "pip install python-telegram-bot"
            )

        if not self.token:
            raise ValueError(
                "TELEGRAM_BOT_TOKEN not set. Add it to .env or pass token= parameter."
            )

        app = Application.builder().token(self.token).build()

        # Register command handlers
        app.add_handler(CommandHandler("start", self._cmd_start))
        app.add_handler(CommandHandler("help", self._cmd_help))
        app.add_handler(CommandHandler("status", self._cmd_status))
        app.add_handler(CommandHandler("agents", self._cmd_agents))
        app.add_handler(CommandHandler("pause", self._cmd_pause))
        app.add_handler(CommandHandler("resume", self._cmd_resume))
        app.add_handler(CommandHandler("pause_all", self._cmd_pause_all))
        app.add_handler(CommandHandler("resume_all", self._cmd_resume_all))
        app.add_handler(CommandHandler("queue", self._cmd_queue))
        app.add_handler(CommandHandler("approvals", self._cmd_approvals))
        app.add_handler(CommandHandler("errors", self._cmd_errors))

        # Catch-all for unknown commands
        app.add_handler(
            MessageHandler(filters.COMMAND, self._cmd_unknown)
        )

        self._app = app
        return app

    async def start(self) -> None:
        """Start the bot (blocking, runs until stopped)."""
        app = self._build_app()
        logger.info("telegram_bot_starting", extra={"vertical_id": self.vertical_id})
        await app.initialize()
        await app.start()
        await app.updater.start_polling()

        # Keep running until stopped
        try:
            stop_event = asyncio.Event()
            await stop_event.wait()
        except (asyncio.CancelledError, KeyboardInterrupt):
            pass
        finally:
            await self.stop()

    async def stop(self) -> None:
        """Gracefully stop the bot."""
        if self._app:
            try:
                await self._app.updater.stop()
                await self._app.stop()
                await self._app.shutdown()
            except Exception:
                pass
        logger.info("telegram_bot_stopped")

    # ------------------------------------------------------------------
    # Push Alerts (agents call this)
    # ------------------------------------------------------------------

    async def send_alert(self, message: str) -> bool:
        """
        Send a push notification to the operator.

        Agents can call this via the EventBus to alert on important events
        (failed tasks, completed content, circuit breaker tripped).

        Returns True if sent successfully, False otherwise.
        """
        if not self._alert_chat_id:
            logger.warning("telegram_no_chat_id: Cannot send alert, no chat_id set.")
            return False

        if not self._app or not self._app.bot:
            logger.warning("telegram_bot_not_running: Cannot send alert.")
            return False

        try:
            await self._app.bot.send_message(
                chat_id=self._alert_chat_id,
                text=f"🔔 *Alert*\n\n{message}",
                parse_mode="Markdown",
            )
            return True
        except Exception as e:
            logger.error(f"telegram_alert_failed: {e}")
            return False

    # ------------------------------------------------------------------
    # Command Handlers
    # ------------------------------------------------------------------

    async def _cmd_start(self, update, context) -> None:
        """Handle /start — welcome message + register chat for alerts."""
        if not self._is_authorized(update.effective_user.id):
            return

        self._alert_chat_id = update.effective_chat.id
        await update.message.reply_text(
            "🛡️ *Sovereign Venture Engine — ChatOps*\n\n"
            "You are now connected to Mission Control.\n"
            "Use /help to see available commands.\n\n"
            "Push alerts will be sent to this chat.",
            parse_mode="Markdown",
        )

    async def _cmd_help(self, update, context) -> None:
        """Handle /help — list commands."""
        if not self._is_authorized(update.effective_user.id):
            return

        help_text = (
            "🛡️ *Sovereign Cockpit Commands*\n\n"
            "📊 *Monitoring*\n"
            "/status — System health overview\n"
            "/agents — List all agents\n"
            "/queue — Task queue depth\n"
            "/approvals — Items awaiting review\n"
            "/errors — Recent failures\n\n"
            "🎮 *Control*\n"
            "/pause <agent\\_id> — Pause an agent\n"
            "/resume <agent\\_id> — Resume an agent\n"
            "/pause\\_all — Emergency stop all\n"
            "/resume\\_all — Resume all agents\n"
        )
        await update.message.reply_text(help_text, parse_mode="Markdown")

    async def _cmd_status(self, update, context) -> None:
        """Handle /status — system health overview."""
        if not self._is_authorized(update.effective_user.id):
            return

        self._alert_chat_id = update.effective_chat.id

        try:
            agents = self.db.list_agent_records()
            enabled = [a for a in agents if a.get("enabled", True)]
            disabled = [a for a in agents if not a.get("enabled", True)]

            pending = len(self.db.list_tasks(status="pending", limit=1000))
            failed = len(self.db.list_tasks(status="failed", limit=100))
            leads = self.db.count_contacts()

            status_emoji = "🟢" if not disabled and failed <= 5 else "🔴"

            msg = (
                f"{status_emoji} *System Status*\n\n"
                f"🤖 Active Agents: {len(enabled)}/{len(agents)}\n"
                f"👤 Total Leads: {leads}\n"
                f"📋 Pending Tasks: {pending}\n"
                f"❌ Failed Tasks: {failed}\n"
            )

            if disabled:
                names = ", ".join(
                    a.get("agent_id", "?") for a in disabled
                )
                msg += f"\n⚠️ Disabled: {names}"

            await update.message.reply_text(msg, parse_mode="Markdown")

        except Exception as e:
            await update.message.reply_text(f"❌ Error fetching status: {e}")

    async def _cmd_agents(self, update, context) -> None:
        """Handle /agents — list all agents with status."""
        if not self._is_authorized(update.effective_user.id):
            return

        try:
            agents = self.db.list_agent_records()
            if not agents:
                await update.message.reply_text("No agents registered.")
                return

            lines = ["🤖 *Agent Fleet*\n"]
            for a in agents:
                enabled = a.get("enabled", True)
                shadow = a.get("shadow_mode", False)
                icon = "🟢" if enabled else "🔴"
                if shadow:
                    icon = "👻"

                pending = self.db.count_pending_tasks(a.get("agent_id", ""))
                lines.append(
                    f"{icon} `{a.get('agent_id', '?')}` — "
                    f"{a.get('name', '?')} "
                    f"(tasks: {pending})"
                )

            await update.message.reply_text(
                "\n".join(lines), parse_mode="Markdown"
            )
        except Exception as e:
            await update.message.reply_text(f"❌ Error: {e}")

    async def _cmd_pause(self, update, context) -> None:
        """Handle /pause <agent_id> — pause a specific agent."""
        if not self._is_authorized(update.effective_user.id):
            return

        if not context.args:
            await update.message.reply_text(
                "Usage: /pause <agent\\_id>\n"
                "Example: /pause outreach",
                parse_mode="Markdown",
            )
            return

        agent_id = context.args[0]
        try:
            self.db.client.table("agents").update(
                {"enabled": False}
            ).eq("agent_id", agent_id).eq(
                "vertical_id", self.vertical_id
            ).execute()
            await update.message.reply_text(f"⏸️ Paused agent: `{agent_id}`", parse_mode="Markdown")
        except Exception as e:
            await update.message.reply_text(f"❌ Failed to pause {agent_id}: {e}")

    async def _cmd_resume(self, update, context) -> None:
        """Handle /resume <agent_id> — resume a specific agent."""
        if not self._is_authorized(update.effective_user.id):
            return

        if not context.args:
            await update.message.reply_text(
                "Usage: /resume <agent\\_id>\n"
                "Example: /resume outreach",
                parse_mode="Markdown",
            )
            return

        agent_id = context.args[0]
        try:
            self.db.client.table("agents").update(
                {"enabled": True}
            ).eq("agent_id", agent_id).eq(
                "vertical_id", self.vertical_id
            ).execute()
            await update.message.reply_text(f"▶️ Resumed agent: `{agent_id}`", parse_mode="Markdown")
        except Exception as e:
            await update.message.reply_text(f"❌ Failed to resume {agent_id}: {e}")

    async def _cmd_pause_all(self, update, context) -> None:
        """Handle /pause_all — emergency stop all agents."""
        if not self._is_authorized(update.effective_user.id):
            return

        try:
            agents = self.db.list_agent_records()
            count = 0
            for a in agents:
                if a.get("enabled", True):
                    self.db.client.table("agents").update(
                        {"enabled": False}
                    ).eq("agent_id", a["agent_id"]).eq(
                        "vertical_id", self.vertical_id
                    ).execute()
                    count += 1

            await update.message.reply_text(
                f"🚨 *EMERGENCY STOP*\n\n{count} agents paused.",
                parse_mode="Markdown",
            )
        except Exception as e:
            await update.message.reply_text(f"❌ Failed: {e}")

    async def _cmd_resume_all(self, update, context) -> None:
        """Handle /resume_all — resume all agents."""
        if not self._is_authorized(update.effective_user.id):
            return

        try:
            agents = self.db.list_agent_records()
            count = 0
            for a in agents:
                if not a.get("enabled", True):
                    self.db.client.table("agents").update(
                        {"enabled": True}
                    ).eq("agent_id", a["agent_id"]).eq(
                        "vertical_id", self.vertical_id
                    ).execute()
                    count += 1

            await update.message.reply_text(
                f"▶️ {count} agents resumed.", parse_mode="Markdown"
            )
        except Exception as e:
            await update.message.reply_text(f"❌ Failed: {e}")

    async def _cmd_queue(self, update, context) -> None:
        """Handle /queue — show task queue depth by status."""
        if not self._is_authorized(update.effective_user.id):
            return

        try:
            pending = len(self.db.list_tasks(status="pending", limit=1000))
            running = len(self.db.list_tasks(status="running", limit=1000))
            failed = len(self.db.list_tasks(status="failed", limit=1000))
            completed = len(self.db.list_tasks(status="completed", limit=1000))

            msg = (
                "📋 *Task Queue*\n\n"
                f"⏳ Pending: {pending}\n"
                f"🔄 Running: {running}\n"
                f"✅ Completed: {completed}\n"
                f"❌ Failed: {failed}\n"
            )
            await update.message.reply_text(msg, parse_mode="Markdown")
        except Exception as e:
            await update.message.reply_text(f"❌ Error: {e}")

    async def _cmd_approvals(self, update, context) -> None:
        """Handle /approvals — show items awaiting human review."""
        if not self._is_authorized(update.effective_user.id):
            return

        try:
            content = self.db.list_content(status="review", limit=10)

            if not content:
                await update.message.reply_text("✅ No items awaiting approval.")
                return

            lines = [f"📝 *{len(content)} Items Awaiting Review*\n"]
            for item in content:
                title = item.get("title", "Untitled")
                ctype = item.get("content_type", "?")
                agent = item.get("agent_id", "?")
                created = (item.get("created_at") or "")[:10]
                lines.append(f"• [{ctype}] _{title}_ by `{agent}` ({created})")

            lines.append("\n→ Review in the Sovereign Cockpit dashboard")
            await update.message.reply_text(
                "\n".join(lines), parse_mode="Markdown"
            )
        except Exception as e:
            await update.message.reply_text(f"❌ Error: {e}")

    async def _cmd_errors(self, update, context) -> None:
        """Handle /errors — show recent agent failures."""
        if not self._is_authorized(update.effective_user.id):
            return

        try:
            failed_runs = self.db.get_agent_runs(status="failed", limit=5)

            if not failed_runs:
                await update.message.reply_text("✅ No recent failures.")
                return

            lines = ["❌ *Recent Failures*\n"]
            for r in failed_runs:
                agent = r.get("agent_id", "?")
                error = (r.get("error_message") or "Unknown error")[:80]
                time = (r.get("created_at") or "")[:19]
                lines.append(f"• `{agent}` — {error}\n  _{time}_")

            await update.message.reply_text(
                "\n".join(lines), parse_mode="Markdown"
            )
        except Exception as e:
            await update.message.reply_text(f"❌ Error: {e}")

    async def _cmd_unknown(self, update, context) -> None:
        """Handle unknown commands."""
        if not self._is_authorized(update.effective_user.id):
            return

        await update.message.reply_text(
            "Unknown command. Use /help to see available commands."
        )


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

def start_bot(
    db: Any = None,
    vertical_id: str = "enclave_guard",
) -> None:
    """
    Start the Telegram bot (blocking).

    Can be called from CLI:
        python -m core.interfaces.telegram_bot
    """
    if db is None:
        from core.integrations.supabase_client import EnclaveDB
        db = EnclaveDB(vertical_id)

    bot = TelegramBot(db=db, vertical_id=vertical_id)
    asyncio.run(bot.start())


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    start_bot()
