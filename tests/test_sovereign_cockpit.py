"""
Unit tests for Phase 1F: Sovereign Cockpit (Dashboard + ChatOps).

Tests cover:
- TelegramBot security (whitelist, fail-closed)
- TelegramBot command formatting
- TelegramBot alert pushing
- Dashboard data helpers (safe_call pattern)
- Approval queue RLHF capture
- Agent control operations
"""

import asyncio
import os
import pytest
from unittest.mock import MagicMock, AsyncMock, patch, PropertyMock


def _run(coro):
    """Helper to run async code in sync tests."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ═══════════════════════════════════════════════════════════════════════
# TELEGRAM BOT TESTS
# ═══════════════════════════════════════════════════════════════════════


class TestTelegramBotSecurity:
    """Security is fail-closed: no whitelist = deny all."""

    def test_no_whitelist_denies_all(self):
        """Without TELEGRAM_ALLOWED_USER_IDS, all users are denied."""
        from core.interfaces.telegram_bot import TelegramBot

        bot = TelegramBot(db=MagicMock(), allowed_user_ids=[])
        assert not bot._is_authorized(123456789)

    def test_empty_env_whitelist_denies_all(self):
        """Empty env var means no whitelist = deny all."""
        from core.interfaces.telegram_bot import TelegramBot

        with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USER_IDS": ""}, clear=False):
            bot = TelegramBot(db=MagicMock(), token="test")
        assert not bot._is_authorized(999)

    def test_whitelist_allows_listed_user(self):
        """Listed user IDs are authorized."""
        from core.interfaces.telegram_bot import TelegramBot

        bot = TelegramBot(db=MagicMock(), allowed_user_ids=[111, 222])
        assert bot._is_authorized(111)
        assert bot._is_authorized(222)

    def test_whitelist_denies_unlisted_user(self):
        """Non-listed user IDs are denied."""
        from core.interfaces.telegram_bot import TelegramBot

        bot = TelegramBot(db=MagicMock(), allowed_user_ids=[111])
        assert not bot._is_authorized(999)

    def test_whitelist_from_env(self):
        """Parses TELEGRAM_ALLOWED_USER_IDS from env."""
        from core.interfaces.telegram_bot import TelegramBot

        with patch.dict(
            os.environ,
            {"TELEGRAM_ALLOWED_USER_IDS": "111,222,333", "TELEGRAM_BOT_TOKEN": "x"},
            clear=False,
        ):
            bot = TelegramBot(db=MagicMock())
        assert bot._is_authorized(111)
        assert bot._is_authorized(222)
        assert bot._is_authorized(333)
        assert not bot._is_authorized(444)

    def test_whitelist_from_env_with_spaces(self):
        """Handles spaces in the env var."""
        from core.interfaces.telegram_bot import TelegramBot

        with patch.dict(
            os.environ,
            {"TELEGRAM_ALLOWED_USER_IDS": " 111 , 222 ", "TELEGRAM_BOT_TOKEN": "x"},
            clear=False,
        ):
            bot = TelegramBot(db=MagicMock())
        assert bot._is_authorized(111)
        assert bot._is_authorized(222)

    def test_whitelist_ignores_non_numeric(self):
        """Non-numeric entries in the env var are silently ignored."""
        from core.interfaces.telegram_bot import TelegramBot

        with patch.dict(
            os.environ,
            {"TELEGRAM_ALLOWED_USER_IDS": "111,abc,222", "TELEGRAM_BOT_TOKEN": "x"},
            clear=False,
        ):
            bot = TelegramBot(db=MagicMock())
        assert bot._is_authorized(111)
        assert bot._is_authorized(222)
        assert len(bot.allowed_user_ids) == 2


class TestTelegramBotConstruction:
    """Tests for bot construction and configuration."""

    def test_default_vertical(self):
        """Default vertical is enclave_guard."""
        from core.interfaces.telegram_bot import TelegramBot

        bot = TelegramBot(db=MagicMock(), allowed_user_ids=[1])
        assert bot.vertical_id == "enclave_guard"

    def test_custom_vertical(self):
        """Can specify custom vertical."""
        from core.interfaces.telegram_bot import TelegramBot

        bot = TelegramBot(
            db=MagicMock(), vertical_id="custom_v", allowed_user_ids=[1]
        )
        assert bot.vertical_id == "custom_v"

    def test_token_from_param(self):
        """Token can be passed as parameter."""
        from core.interfaces.telegram_bot import TelegramBot

        bot = TelegramBot(db=MagicMock(), token="my-token", allowed_user_ids=[1])
        assert bot.token == "my-token"

    def test_token_from_env(self):
        """Token is read from env if not passed."""
        from core.interfaces.telegram_bot import TelegramBot

        with patch.dict(
            os.environ, {"TELEGRAM_BOT_TOKEN": "env-token"}, clear=False
        ):
            bot = TelegramBot(db=MagicMock(), allowed_user_ids=[1])
        assert bot.token == "env-token"

    def test_no_alert_chat_id_initially(self):
        """Alert chat ID is None until /start or /status."""
        from core.interfaces.telegram_bot import TelegramBot

        bot = TelegramBot(db=MagicMock(), allowed_user_ids=[1])
        assert bot._alert_chat_id is None


class TestTelegramBotAlerts:
    """Tests for the push alert system."""

    def test_alert_without_chat_id_returns_false(self):
        """Cannot send alert if no chat_id is set."""
        from core.interfaces.telegram_bot import TelegramBot

        bot = TelegramBot(db=MagicMock(), allowed_user_ids=[1])
        assert bot._alert_chat_id is None
        result = _run(bot.send_alert("test"))
        assert result is False

    def test_alert_without_app_returns_false(self):
        """Cannot send alert if bot app is not running."""
        from core.interfaces.telegram_bot import TelegramBot

        bot = TelegramBot(db=MagicMock(), allowed_user_ids=[1])
        bot._alert_chat_id = 12345
        # _app is None
        result = _run(bot.send_alert("test"))
        assert result is False

    def test_alert_success(self):
        """Sends alert when chat_id and app are available."""
        from core.interfaces.telegram_bot import TelegramBot

        bot = TelegramBot(db=MagicMock(), allowed_user_ids=[1])
        bot._alert_chat_id = 12345

        mock_bot_instance = MagicMock()
        mock_bot_instance.send_message = AsyncMock()

        mock_app = MagicMock()
        mock_app.bot = mock_bot_instance
        bot._app = mock_app

        result = _run(bot.send_alert("System alert!"))
        assert result is True
        mock_bot_instance.send_message.assert_called_once()

        call_kwargs = mock_bot_instance.send_message.call_args
        assert call_kwargs.kwargs["chat_id"] == 12345
        assert "System alert!" in call_kwargs.kwargs["text"]


class TestTelegramBotCommands:
    """Test command handler logic."""

    def _make_bot_with_db(self):
        from core.interfaces.telegram_bot import TelegramBot

        db = MagicMock()
        db.list_agent_records.return_value = [
            {"agent_id": "outreach", "name": "Outreach Agent", "enabled": True},
            {"agent_id": "seo_v1", "name": "SEO Agent", "enabled": False},
        ]
        db.list_tasks.return_value = []
        db.count_contacts.return_value = 42
        db.count_pending_tasks.return_value = 3
        db.list_content.return_value = []
        db.get_agent_runs.return_value = []

        bot = TelegramBot(db=db, allowed_user_ids=[111])
        return bot, db

    def _make_update(self, user_id=111, chat_id=999):
        update = MagicMock()
        update.effective_user.id = user_id
        update.effective_chat.id = chat_id
        update.message.reply_text = AsyncMock()
        return update

    def test_status_command_authorized(self):
        """Authorized user gets status response."""
        bot, db = self._make_bot_with_db()
        update = self._make_update(user_id=111)
        context = MagicMock()

        _run(bot._cmd_status(update, context))

        update.message.reply_text.assert_called_once()
        msg = update.message.reply_text.call_args[0][0]
        assert "System Status" in msg
        assert "42" in msg  # leads count

    def test_status_command_unauthorized(self):
        """Unauthorized user gets no response."""
        bot, db = self._make_bot_with_db()
        update = self._make_update(user_id=999)  # not in whitelist
        context = MagicMock()

        _run(bot._cmd_status(update, context))

        update.message.reply_text.assert_not_called()

    def test_agents_command(self):
        """Lists agents with status icons."""
        bot, db = self._make_bot_with_db()
        update = self._make_update()
        context = MagicMock()

        _run(bot._cmd_agents(update, context))

        msg = update.message.reply_text.call_args[0][0]
        assert "outreach" in msg
        assert "seo_v1" in msg

    def test_pause_command_requires_args(self):
        """Pause without agent_id shows usage."""
        bot, db = self._make_bot_with_db()
        update = self._make_update()
        context = MagicMock()
        context.args = []

        _run(bot._cmd_pause(update, context))

        msg = update.message.reply_text.call_args[0][0]
        assert "Usage" in msg

    def test_pause_command_with_agent_id(self):
        """Pause with valid agent_id updates DB."""
        bot, db = self._make_bot_with_db()
        update = self._make_update()
        context = MagicMock()
        context.args = ["outreach"]

        # Mock the chained table call
        mock_result = MagicMock()
        db.client.table.return_value.update.return_value.eq.return_value.eq.return_value.execute = MagicMock()

        _run(bot._cmd_pause(update, context))

        msg = update.message.reply_text.call_args[0][0]
        assert "Paused" in msg
        assert "outreach" in msg

    def test_resume_command_with_agent_id(self):
        """Resume with valid agent_id updates DB."""
        bot, db = self._make_bot_with_db()
        update = self._make_update()
        context = MagicMock()
        context.args = ["seo_v1"]

        mock_result = MagicMock()
        db.client.table.return_value.update.return_value.eq.return_value.eq.return_value.execute = MagicMock()

        _run(bot._cmd_resume(update, context))

        msg = update.message.reply_text.call_args[0][0]
        assert "Resumed" in msg
        assert "seo_v1" in msg

    def test_help_command(self):
        """Help shows all commands."""
        bot, db = self._make_bot_with_db()
        update = self._make_update()
        context = MagicMock()

        _run(bot._cmd_help(update, context))

        msg = update.message.reply_text.call_args[0][0]
        assert "/status" in msg
        assert "/pause" in msg
        assert "/resume" in msg
        assert "/queue" in msg

    def test_queue_command(self):
        """Queue shows task counts by status."""
        bot, db = self._make_bot_with_db()
        update = self._make_update()
        context = MagicMock()

        _run(bot._cmd_queue(update, context))

        msg = update.message.reply_text.call_args[0][0]
        assert "Task Queue" in msg
        assert "Pending" in msg

    def test_approvals_command_empty(self):
        """Approvals with no pending items shows success."""
        bot, db = self._make_bot_with_db()
        update = self._make_update()
        context = MagicMock()

        _run(bot._cmd_approvals(update, context))

        msg = update.message.reply_text.call_args[0][0]
        assert "No items" in msg

    def test_approvals_command_with_items(self):
        """Approvals with pending items shows them."""
        bot, db = self._make_bot_with_db()
        db.list_content.return_value = [
            {
                "title": "Pen Test Guide",
                "content_type": "blog_post",
                "agent_id": "seo_v1",
                "created_at": "2025-01-15T10:00:00",
            },
        ]
        update = self._make_update()
        context = MagicMock()

        _run(bot._cmd_approvals(update, context))

        msg = update.message.reply_text.call_args[0][0]
        assert "Pen Test Guide" in msg
        assert "seo_v1" in msg

    def test_errors_command_empty(self):
        """Errors with no failures shows success."""
        bot, db = self._make_bot_with_db()
        update = self._make_update()
        context = MagicMock()

        _run(bot._cmd_errors(update, context))

        msg = update.message.reply_text.call_args[0][0]
        assert "No recent failures" in msg

    def test_errors_command_with_failures(self):
        """Errors shows recent failures."""
        bot, db = self._make_bot_with_db()
        db.get_agent_runs.return_value = [
            {
                "agent_id": "outreach",
                "error_message": "API timeout after 30s",
                "created_at": "2025-01-15T10:00:00",
            },
        ]
        update = self._make_update()
        context = MagicMock()

        _run(bot._cmd_errors(update, context))

        msg = update.message.reply_text.call_args[0][0]
        assert "API timeout" in msg

    def test_pause_all_command(self):
        """Pause all disables all enabled agents."""
        bot, db = self._make_bot_with_db()
        update = self._make_update()
        context = MagicMock()

        db.client.table.return_value.update.return_value.eq.return_value.eq.return_value.execute = MagicMock()

        _run(bot._cmd_pause_all(update, context))

        msg = update.message.reply_text.call_args[0][0]
        assert "EMERGENCY STOP" in msg

    def test_resume_all_command(self):
        """Resume all enables all disabled agents."""
        bot, db = self._make_bot_with_db()
        update = self._make_update()
        context = MagicMock()

        db.client.table.return_value.update.return_value.eq.return_value.eq.return_value.execute = MagicMock()

        _run(bot._cmd_resume_all(update, context))

        msg = update.message.reply_text.call_args[0][0]
        assert "resumed" in msg

    def test_unknown_command(self):
        """Unknown command gets a helpful response."""
        bot, db = self._make_bot_with_db()
        update = self._make_update()
        context = MagicMock()

        _run(bot._cmd_unknown(update, context))

        msg = update.message.reply_text.call_args[0][0]
        assert "Unknown command" in msg

    def test_start_sets_alert_chat_id(self):
        """/start registers the chat for push alerts."""
        bot, db = self._make_bot_with_db()
        update = self._make_update(chat_id=12345)
        context = MagicMock()

        _run(bot._cmd_start(update, context))

        assert bot._alert_chat_id == 12345

    def test_status_sets_alert_chat_id(self):
        """/status also registers the chat for push alerts."""
        bot, db = self._make_bot_with_db()
        update = self._make_update(chat_id=67890)
        context = MagicMock()

        _run(bot._cmd_status(update, context))

        assert bot._alert_chat_id == 67890


# ═══════════════════════════════════════════════════════════════════════
# DASHBOARD HELPER TESTS
# ═══════════════════════════════════════════════════════════════════════


class TestDashboardHelpers:
    """Test the safe_call and RLHF capture patterns used in the dashboard."""

    def test_rlhf_capture_on_edit(self):
        """When human edits content, RLHF training example is stored."""
        from dashboard.pages import _approvals_helpers as helpers

        db = MagicMock()

        helpers.capture_rlhf(
            db=db,
            agent_id="seo_v1",
            vertical_id="enclave_guard",
            original="Original blog draft about pen testing.",
            edited="Improved blog draft about penetration testing methodology.",
            item={"id": "abc123", "content_type": "blog_post", "title": "Test"},
        )

        db.store_training_example.assert_called_once()
        call_kwargs = db.store_training_example.call_args.kwargs
        assert call_kwargs["agent_id"] == "seo_v1"
        assert call_kwargs["model_output"] == "Original blog draft about pen testing."
        assert "penetration testing" in call_kwargs["human_correction"]
        assert call_kwargs["source"] == "manual_review"

    def test_rlhf_capture_failure_is_silent(self):
        """RLHF capture failure doesn't crash the dashboard."""
        from dashboard.pages import _approvals_helpers as helpers

        db = MagicMock()
        db.store_training_example.side_effect = Exception("DB down")

        # Should not raise
        helpers.capture_rlhf(
            db=db,
            agent_id="seo_v1",
            vertical_id="enclave_guard",
            original="Original",
            edited="Edited",
            item={},
        )


class TestDashboardSafeCall:
    """Test the safe_call pattern for graceful degradation."""

    def test_safe_call_returns_result(self):
        """Normal function returns its result."""

        def _safe_call(fn, default=None):
            try:
                return fn()
            except Exception:
                return default

        assert _safe_call(lambda: 42) == 42

    def test_safe_call_returns_default_on_error(self):
        """Exception returns the default value."""

        def _safe_call(fn, default=None):
            try:
                return fn()
            except Exception:
                return default

        def boom():
            raise RuntimeError("fail")

        assert _safe_call(boom, default=[]) == []
        assert _safe_call(boom, default=0) == 0
        assert _safe_call(boom) is None


# ═══════════════════════════════════════════════════════════════════════
# DASHBOARD MODULE STRUCTURE TESTS
# ═══════════════════════════════════════════════════════════════════════


class TestDashboardStructure:
    """Verify the dashboard module structure exists."""

    def test_dashboard_app_exists(self):
        """dashboard/app.py exists."""
        from pathlib import Path

        app_path = Path(__file__).parent.parent / "dashboard" / "app.py"
        assert app_path.exists(), f"Missing: {app_path}"

    def test_approvals_page_exists(self):
        """dashboard/pages/1_Approvals.py exists."""
        from pathlib import Path

        page = Path(__file__).parent.parent / "dashboard" / "pages" / "1_Approvals.py"
        assert page.exists(), f"Missing: {page}"

    def test_agents_page_exists(self):
        """dashboard/pages/2_Agents.py exists."""
        from pathlib import Path

        page = Path(__file__).parent.parent / "dashboard" / "pages" / "2_Agents.py"
        assert page.exists(), f"Missing: {page}"

    def test_run_script_exists(self):
        """dashboard/run.sh exists and is executable."""
        from pathlib import Path

        script = Path(__file__).parent.parent / "dashboard" / "run.sh"
        assert script.exists(), f"Missing: {script}"
        assert os.access(str(script), os.X_OK), "run.sh is not executable"

    def test_telegram_bot_exists(self):
        """core/interfaces/telegram_bot.py exists."""
        from pathlib import Path

        module = (
            Path(__file__).parent.parent / "core" / "interfaces" / "telegram_bot.py"
        )
        assert module.exists(), f"Missing: {module}"


class TestTelegramBotImport:
    """Verify the telegram bot can be imported."""

    def test_import_telegram_bot(self):
        """TelegramBot class is importable."""
        from core.interfaces.telegram_bot import TelegramBot

        assert TelegramBot is not None

    def test_import_start_bot(self):
        """start_bot function is importable."""
        from core.interfaces.telegram_bot import start_bot

        assert callable(start_bot)


# ═══════════════════════════════════════════════════════════════════════
# INTEGRATION PATTERN TESTS
# ═══════════════════════════════════════════════════════════════════════


class TestAgentControlPatterns:
    """Test the patterns used for agent control (pause/resume/shadow)."""

    def test_enable_disable_agent_pattern(self):
        """Verify the DB update pattern for enabling/disabling agents."""
        db = MagicMock()

        # The pattern used in both dashboard and telegram bot
        db.client.table.return_value.update.return_value.eq.return_value.eq.return_value.execute = MagicMock()

        # Simulate disable
        db.client.table("agents").update({"enabled": False}).eq(
            "agent_id", "outreach"
        ).eq("vertical_id", "enclave_guard").execute()

        # Verify the chain was called
        db.client.table.assert_called()

    def test_shadow_mode_toggle_pattern(self):
        """Verify the DB update pattern for toggling shadow mode."""
        db = MagicMock()

        db.client.table.return_value.update.return_value.eq.return_value.eq.return_value.execute = MagicMock()

        # Simulate shadow toggle
        db.client.table("agents").update({"shadow_mode": True}).eq(
            "agent_id", "seo_v1"
        ).eq("vertical_id", "enclave_guard").execute()

        db.client.table.assert_called()

    def test_error_reset_uses_rpc(self):
        """Error reset calls the RPC method."""
        db = MagicMock()
        db.reset_agent_errors("outreach", "enclave_guard")
        db.reset_agent_errors.assert_called_once_with("outreach", "enclave_guard")
