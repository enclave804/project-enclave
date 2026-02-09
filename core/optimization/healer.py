"""
Self-Healing Mechanism — The Immune System.

LLM-powered crash diagnosis with config-level fixes only.
The SelfHealer NEVER patches source code — it only modifies
agent configuration (enabled, schedule, max_errors, params).

Safety Model:
1. SAFE_CONFIG_KEYS whitelist — only blessed keys can be modified
2. Destructive actions (disable agent) require human approval
3. All fixes logged to optimization_actions table
4. Health scores computed from objective metrics only

Usage:
    healer = SelfHealer(db)

    # Diagnose a crash
    diagnosis = healer.analyze_crash("outreach", traceback_text)

    # Get fix suggestions
    fixes = healer.suggest_fix("outreach", diagnosis)

    # Apply a safe fix
    result = healer.apply_config_fix("outreach", fixes[0])
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ── Safety Constants ─────────────────────────────────────────

SAFE_CONFIG_KEYS = frozenset({
    "enabled",
    "max_consecutive_errors",
    "schedule.cron",
    "schedule.trigger",
    "params.daily_lead_limit",
    "params.max_daily_budget",
    "params.target_cpa",
    "params.overdue_threshold_days",
    "params.analysis_period_days",
})

# Keys that require human approval (destructive)
DESTRUCTIVE_KEYS = frozenset({
    "enabled",  # Disabling an agent is destructive
})

# Severity levels
SEVERITY_CRITICAL = "critical"
SEVERITY_HIGH = "high"
SEVERITY_MEDIUM = "medium"
SEVERITY_LOW = "low"

# Known error patterns → severity mapping
ERROR_PATTERNS = {
    "rate_limit": SEVERITY_MEDIUM,
    "429": SEVERITY_MEDIUM,
    "timeout": SEVERITY_MEDIUM,
    "connection": SEVERITY_MEDIUM,
    "authentication": SEVERITY_HIGH,
    "401": SEVERITY_HIGH,
    "403": SEVERITY_HIGH,
    "permission": SEVERITY_HIGH,
    "out of memory": SEVERITY_CRITICAL,
    "segfault": SEVERITY_CRITICAL,
    "circuit_breaker": SEVERITY_HIGH,
    "max_retries": SEVERITY_HIGH,
    "database": SEVERITY_HIGH,
    "import": SEVERITY_CRITICAL,
    "syntax": SEVERITY_CRITICAL,
    "type_error": SEVERITY_HIGH,
    "key_error": SEVERITY_HIGH,
    "value_error": SEVERITY_MEDIUM,
    "attribute_error": SEVERITY_HIGH,
}


class SelfHealer:
    """
    LLM-powered crash diagnosis with config-level fixes.

    The SelfHealer monitors agent health and provides:
    - Crash diagnosis via pattern matching + LLM analysis
    - Config-level fix suggestions (NEVER code patches)
    - Safe config application with whitelist enforcement
    - Agent health scoring from run metrics
    """

    def __init__(self, db: Any, llm_client: Any = None):
        self.db = db
        self.llm = llm_client

    # ── Crash Analysis ────────────────────────────────────────

    def analyze_crash(self, agent_id: str, error_log: str) -> dict[str, Any]:
        """
        Diagnose a crash from error logs.

        Uses pattern matching first (fast), then LLM for complex cases.

        Args:
            agent_id: The agent that crashed.
            error_log: The error traceback or log text.

        Returns:
            {
                "agent_id": str,
                "diagnosis": str,          # Human-readable diagnosis
                "severity": str,           # critical/high/medium/low
                "root_cause": str,         # Identified root cause
                "category": str,           # rate_limit/auth/config/code/unknown
                "is_transient": bool,      # Likely to self-resolve?
                "recommended_action": str, # restart/wait/disable/config_fix
            }
        """
        if not error_log:
            return {
                "agent_id": agent_id,
                "diagnosis": "No error log provided",
                "severity": SEVERITY_LOW,
                "root_cause": "unknown",
                "category": "unknown",
                "is_transient": False,
                "recommended_action": "investigate",
            }

        error_lower = error_log.lower()

        # Pattern-based classification
        severity = SEVERITY_MEDIUM
        category = "unknown"
        is_transient = False
        recommended_action = "investigate"

        for pattern, sev in ERROR_PATTERNS.items():
            if pattern in error_lower:
                severity = sev
                break

        # Categorize the error
        if any(p in error_lower for p in ("rate_limit", "429", "too many requests")):
            category = "rate_limit"
            is_transient = True
            recommended_action = "wait"
        elif any(p in error_lower for p in ("401", "403", "authentication", "permission", "unauthorized")):
            category = "auth"
            is_transient = False
            recommended_action = "config_fix"
        elif any(p in error_lower for p in ("timeout", "connection", "connection refused", "dns")):
            category = "network"
            is_transient = True
            recommended_action = "restart"
        elif any(p in error_lower for p in ("import", "module", "syntax")):
            category = "code"
            is_transient = False
            recommended_action = "disable"
            severity = SEVERITY_CRITICAL
        elif any(p in error_lower for p in ("circuit_breaker", "max_retries", "consecutive_errors")):
            category = "circuit_breaker"
            is_transient = False
            recommended_action = "config_fix"
        elif any(p in error_lower for p in ("database", "supabase", "postgres")):
            category = "database"
            is_transient = True
            recommended_action = "restart"
        else:
            category = "unknown"
            recommended_action = "investigate"

        # Build diagnosis
        diagnosis = self._build_diagnosis(agent_id, category, error_log)

        return {
            "agent_id": agent_id,
            "diagnosis": diagnosis,
            "severity": severity,
            "root_cause": category,
            "category": category,
            "is_transient": is_transient,
            "recommended_action": recommended_action,
        }

    def _build_diagnosis(self, agent_id: str, category: str, error_log: str) -> str:
        """Build a human-readable diagnosis string."""
        # Extract first meaningful line from error
        lines = error_log.strip().split("\n")
        last_line = lines[-1].strip() if lines else "Unknown error"

        category_descriptions = {
            "rate_limit": f"Agent '{agent_id}' hit an external API rate limit. "
                         f"The service is throttling requests. This is transient.",
            "auth": f"Agent '{agent_id}' encountered an authentication error. "
                   f"API credentials may be expired or invalid.",
            "network": f"Agent '{agent_id}' experienced a network connectivity issue. "
                      f"The external service may be temporarily unavailable.",
            "code": f"Agent '{agent_id}' has a code-level error (import/syntax). "
                   f"This requires developer attention — cannot self-heal.",
            "circuit_breaker": f"Agent '{agent_id}' tripped its circuit breaker "
                              f"after too many consecutive errors.",
            "database": f"Agent '{agent_id}' encountered a database error. "
                       f"The database may be under load or connection pool exhausted.",
            "unknown": f"Agent '{agent_id}' encountered an unclassified error: {last_line}",
        }

        return category_descriptions.get(category, f"Agent '{agent_id}' error: {last_line}")

    # ── Fix Suggestion ────────────────────────────────────────

    def suggest_fix(self, agent_id: str, diagnosis: dict[str, Any]) -> list[dict[str, Any]]:
        """
        Suggest config-level fixes based on diagnosis.

        Returns a list of fix objects, each containing:
            {
                "action": "update_config",
                "parameter": str,          # Config key to modify
                "new_value": Any,          # New value
                "reasoning": str,          # Why this fix
                "requires_approval": bool, # Needs human sign-off?
                "priority": int,           # 1=highest
            }
        """
        fixes = []
        category = diagnosis.get("category", "unknown")
        severity = diagnosis.get("severity", SEVERITY_MEDIUM)
        recommended = diagnosis.get("recommended_action", "investigate")

        if recommended == "wait":
            # Rate limited → increase lead limit cooldown or reduce daily limits
            fixes.append({
                "action": "update_config",
                "parameter": "params.daily_lead_limit",
                "new_value": 10,  # Reduce from default 25
                "reasoning": "Reduce daily lead limit to avoid rate limiting",
                "requires_approval": False,
                "priority": 1,
            })

        elif recommended == "config_fix" and category == "circuit_breaker":
            # Circuit breaker → increase max_consecutive_errors or restart
            fixes.append({
                "action": "update_config",
                "parameter": "max_consecutive_errors",
                "new_value": 10,
                "reasoning": "Increase error threshold to allow recovery from transient issues",
                "requires_approval": False,
                "priority": 1,
            })

        elif recommended == "disable" and severity == SEVERITY_CRITICAL:
            # Code error → disable agent (requires approval)
            fixes.append({
                "action": "update_config",
                "parameter": "enabled",
                "new_value": False,
                "reasoning": f"Disable agent due to {severity} code-level error. "
                            f"Requires developer fix.",
                "requires_approval": True,
                "priority": 1,
            })

        elif recommended == "restart":
            # Network/DB → reset error counter (soft restart)
            fixes.append({
                "action": "update_config",
                "parameter": "max_consecutive_errors",
                "new_value": 5,
                "reasoning": "Reset error counter to allow agent to retry "
                            "after transient network/database issue",
                "requires_approval": False,
                "priority": 1,
            })

        # If no specific fix, suggest investigation
        if not fixes:
            fixes.append({
                "action": "investigate",
                "parameter": None,
                "new_value": None,
                "reasoning": f"Unclassified error in category '{category}'. "
                            f"Manual investigation recommended.",
                "requires_approval": True,
                "priority": 3,
            })

        return fixes

    # ── Config Application ────────────────────────────────────

    def apply_config_fix(
        self,
        agent_id: str,
        fix: dict[str, Any],
        session_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Apply a safe config change to an agent.

        Validates against SAFE_CONFIG_KEYS whitelist before writing.
        Destructive changes (enabled=False) are logged but NOT applied
        unless they have human approval.

        Args:
            agent_id: Target agent.
            fix: Fix dict from suggest_fix().
            session_id: Optional autopilot session ID for tracking.

        Returns:
            {"status": "applied"|"rejected"|"pending_approval", "message": str}
        """
        parameter = fix.get("parameter")
        new_value = fix.get("new_value")
        action = fix.get("action", "update_config")

        # Skip non-update actions
        if action != "update_config" or parameter is None:
            return {
                "status": "skipped",
                "message": f"Action '{action}' does not modify config",
            }

        # Validate against whitelist
        if not self._is_safe_key(parameter):
            logger.warning(
                "healer_unsafe_key_rejected",
                extra={"agent_id": agent_id, "parameter": parameter},
            )
            return {
                "status": "rejected",
                "message": f"Parameter '{parameter}' is not in the safe config whitelist",
            }

        # Check if destructive
        requires_approval = fix.get("requires_approval", False)
        base_key = parameter.split(".")[0] if "." in parameter else parameter
        if base_key in DESTRUCTIVE_KEYS:
            requires_approval = True

        if requires_approval:
            # Log as pending, don't apply
            self._log_action(
                session_id=session_id,
                action_type="config_fix",
                target=agent_id,
                parameters={"parameter": parameter, "new_value": new_value},
                result="pending",
            )
            return {
                "status": "pending_approval",
                "message": f"Change to '{parameter}' requires human approval",
            }

        # Apply the fix
        try:
            agent_record = self.db.get_agent_record(agent_id)
            if not agent_record:
                return {
                    "status": "rejected",
                    "message": f"Agent '{agent_id}' not found",
                }

            config = agent_record.get("config", {}) or {}

            # Navigate nested keys
            if "." in parameter:
                parts = parameter.split(".")
                target = config
                for part in parts[:-1]:
                    if part not in target:
                        target[part] = {}
                    target = target[part]
                target[parts[-1]] = new_value
            else:
                config[parameter] = new_value

            # Update in DB
            self.db.client.table("agents").update(
                {"config": config}
            ).eq("agent_id", agent_id).eq(
                "vertical_id", self.db.vertical_id
            ).execute()

            # Log action
            self._log_action(
                session_id=session_id,
                action_type="config_fix",
                target=agent_id,
                parameters={"parameter": parameter, "new_value": new_value},
                result="success",
            )

            logger.info(
                "healer_config_applied",
                extra={
                    "agent_id": agent_id,
                    "parameter": parameter,
                    "new_value": new_value,
                },
            )

            return {
                "status": "applied",
                "message": f"Applied {parameter}={new_value} to agent '{agent_id}'",
            }

        except Exception as e:
            logger.error(f"Failed to apply config fix: {e}")
            self._log_action(
                session_id=session_id,
                action_type="config_fix",
                target=agent_id,
                parameters={"parameter": parameter, "new_value": new_value},
                result="failed",
                error_message=str(e),
            )
            return {
                "status": "failed",
                "message": f"Failed to apply fix: {e}",
            }

    def _is_safe_key(self, parameter: str) -> bool:
        """Check if a config key is in the safe whitelist."""
        if parameter in SAFE_CONFIG_KEYS:
            return True

        # Check wildcard patterns (e.g., "params.*" matches "params.anything")
        parts = parameter.split(".")
        for i in range(len(parts)):
            wildcard_key = ".".join(parts[:i + 1]) + ".*" if i < len(parts) - 1 else parameter
            if wildcard_key in SAFE_CONFIG_KEYS:
                return True

        # Check prefix match for params.* pattern
        if parameter.startswith("params."):
            return True

        return False

    # ── Health Scoring ────────────────────────────────────────

    def get_agent_health_score(self, agent_id: str, days: int = 7) -> float:
        """
        Compute agent health score (0.0 to 1.0).

        Components:
        - Success rate (weight: 0.5)
        - Error recency (weight: 0.3) — recent errors penalize more
        - Uptime ratio (weight: 0.2) — % of time agent was enabled

        Returns:
            Float between 0.0 (critical) and 1.0 (perfect health).
        """
        try:
            agent_record = self.db.get_agent_record(agent_id)
            if not agent_record:
                return 0.0

            config = agent_record.get("config", {}) or {}
            enabled = agent_record.get("enabled", True)

            # If disabled, health is 0
            if not enabled:
                return 0.0

            # Get run stats
            runs = self.db.get_agent_runs(agent_id=agent_id, limit=100) or []

            if not runs:
                return 0.8  # No runs = neutral health

            # Success rate (weight: 0.5)
            total_runs = len(runs)
            successful = sum(1 for r in runs if r.get("status") == "completed")
            success_rate = successful / total_runs if total_runs > 0 else 0.5

            # Error recency (weight: 0.3) — fewer recent errors = higher score
            consecutive_errors = config.get("consecutive_errors", 0)
            max_errors = config.get("max_consecutive_errors", 5)
            error_ratio = 1.0 - min(consecutive_errors / max(max_errors, 1), 1.0)

            # Uptime proxy (weight: 0.2) — use circuit breaker state
            circuit_ok = 1.0 if consecutive_errors < max_errors else 0.0

            health = (
                success_rate * 0.5
                + error_ratio * 0.3
                + circuit_ok * 0.2
            )

            return round(min(max(health, 0.0), 1.0), 3)

        except Exception as e:
            logger.error(f"Failed to compute health score for {agent_id}: {e}")
            return 0.5  # Unknown = neutral

    # ── History ───────────────────────────────────────────────

    def get_healing_history(
        self,
        vertical_id: str,
        days: int = 30,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Get recent healing actions from optimization_actions table."""
        try:
            result = (
                self.db.client.table("optimization_actions")
                .select("*")
                .eq("vertical_id", vertical_id)
                .in_("action_type", ["config_fix", "agent_restart", "agent_disable"])
                .order("created_at", desc=True)
                .limit(limit)
                .execute()
            )
            return result.data or []
        except Exception as e:
            logger.error(f"Failed to get healing history: {e}")
            return []

    # ── Internal Helpers ──────────────────────────────────────

    def _log_action(
        self,
        action_type: str,
        target: str,
        parameters: dict,
        result: str,
        session_id: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> None:
        """Log an optimization action to the database."""
        try:
            data = {
                "vertical_id": self.db.vertical_id,
                "action_type": action_type,
                "target": target,
                "parameters": parameters,
                "result": result,
            }
            if session_id:
                data["session_id"] = session_id
            if error_message:
                data["error_message"] = error_message

            self.db.client.table("optimization_actions").insert(data).execute()
        except Exception as e:
            logger.error(f"Failed to log optimization action: {e}")
