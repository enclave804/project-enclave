"""
Dashboard Authentication — The Fortress Gate.

Provides password-based authentication for the Sovereign Cockpit.
Uses Streamlit's session_state for persistence across reruns.

Security model:
- Password verified against DASHBOARD_PASSWORD env var or st.secrets
- Session state tracks authentication status
- Failed attempts are logged with timestamps for audit trail
- Rate limiting: 5 failed attempts triggers a 30-second cooldown

Usage:
    # At the top of any dashboard page:
    from dashboard.auth import require_auth
    require_auth()  # Blocks page render until authenticated

Configuration:
    Option A: st.secrets["DASHBOARD_PASSWORD"]
    Option B: DASHBOARD_PASSWORD environment variable
    If neither is set, authentication is DISABLED with a warning.
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# Rate limiting constants
MAX_FAILED_ATTEMPTS = 5
COOLDOWN_SECONDS = 30


def _get_dashboard_password() -> Optional[str]:
    """
    Retrieve the dashboard password from secrets or environment.

    Priority:
    1. Streamlit secrets (st.secrets["DASHBOARD_PASSWORD"])
    2. Environment variable (DASHBOARD_PASSWORD)
    3. None (authentication disabled)
    """
    # Try Streamlit secrets first
    try:
        import streamlit as st
        password = st.secrets.get("DASHBOARD_PASSWORD")
        if password:
            return str(password)
    except Exception:
        pass

    # Fall back to environment variable
    env_password = os.environ.get("DASHBOARD_PASSWORD")
    if env_password:
        return env_password

    return None


def _verify_password(input_password: str, stored_password: str) -> bool:
    """
    Constant-time password comparison to prevent timing attacks.

    Uses hmac.compare_digest via hashlib for safe comparison.
    """
    import hmac
    return hmac.compare_digest(
        hashlib.sha256(input_password.encode()).hexdigest(),
        hashlib.sha256(stored_password.encode()).hexdigest(),
    )


def _is_rate_limited(st) -> bool:
    """Check if login attempts are rate-limited."""
    failed_count = st.session_state.get("auth_failed_count", 0)
    last_failed = st.session_state.get("auth_last_failed_at", 0)

    if failed_count >= MAX_FAILED_ATTEMPTS:
        elapsed = time.time() - last_failed
        if elapsed < COOLDOWN_SECONDS:
            return True
        # Cooldown expired — reset counter
        st.session_state["auth_failed_count"] = 0
    return False


def _record_failed_attempt(st) -> None:
    """Record a failed login attempt."""
    st.session_state["auth_failed_count"] = (
        st.session_state.get("auth_failed_count", 0) + 1
    )
    st.session_state["auth_last_failed_at"] = time.time()

    logger.warning(
        "dashboard_auth_failed",
        extra={
            "attempt_count": st.session_state["auth_failed_count"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )


def require_auth() -> None:
    """
    Require authentication before rendering the dashboard page.

    Call this at the top of every dashboard page. If the user is not
    authenticated, it renders a login form and calls st.stop() to
    prevent the rest of the page from rendering.

    If no password is configured, authentication is bypassed with a
    warning banner (dev-friendly default).
    """
    import streamlit as st

    # Check if already authenticated
    if st.session_state.get("authenticated", False):
        return

    # Check if authentication is configured
    password = _get_dashboard_password()
    if password is None:
        # No password configured — bypass auth with warning
        st.session_state["authenticated"] = True
        logger.warning(
            "dashboard_auth_disabled: No DASHBOARD_PASSWORD configured. "
            "Set DASHBOARD_PASSWORD env var or st.secrets for production."
        )
        return

    # Render login form
    st.markdown("## 🔐 Sovereign Cockpit — Authentication Required")
    st.markdown("Enter the dashboard password to continue.")

    # Rate limiting check
    if _is_rate_limited(st):
        remaining = COOLDOWN_SECONDS - (
            time.time() - st.session_state.get("auth_last_failed_at", 0)
        )
        st.error(
            f"⏳ Too many failed attempts. Please wait {int(remaining)} seconds."
        )
        st.stop()
        return

    with st.form("login_form"):
        entered_password = st.text_input(
            "Password",
            type="password",
            placeholder="Enter dashboard password",
        )
        submitted = st.form_submit_button("🔓 Unlock Dashboard")

        if submitted:
            if entered_password and _verify_password(entered_password, password):
                st.session_state["authenticated"] = True
                st.session_state["auth_failed_count"] = 0
                st.session_state["auth_timestamp"] = datetime.now(
                    timezone.utc
                ).isoformat()

                logger.info(
                    "dashboard_auth_success",
                    extra={
                        "timestamp": st.session_state["auth_timestamp"],
                    },
                )
                st.rerun()
            else:
                _record_failed_attempt(st)
                failed_count = st.session_state.get("auth_failed_count", 0)
                remaining = MAX_FAILED_ATTEMPTS - failed_count
                if remaining > 0:
                    st.error(
                        f"❌ Invalid password. {remaining} attempts remaining."
                    )
                else:
                    st.error(
                        f"🔒 Too many failed attempts. "
                        f"Locked for {COOLDOWN_SECONDS} seconds."
                    )

    st.stop()
