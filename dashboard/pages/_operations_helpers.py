"""
Extracted helpers for the Operations Dashboard page.

Separated from the Streamlit page so they can be unit-tested
without importing streamlit (which requires a running server).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


# ─── Invoice Helpers ─────────────────────────────────────────

def compute_invoice_stats(invoices: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Compute aggregate stats from a list of invoices.

    Returns:
        Dict with: total, open, paid, overdue, void,
        total_amount, paid_amount, overdue_amount,
        collection_rate, avg_invoice
    """
    total = len(invoices)
    open_count = sum(1 for i in invoices if i.get("status") == "open")
    paid = sum(1 for i in invoices if i.get("status") == "paid")
    overdue = sum(1 for i in invoices if i.get("status") == "overdue")
    void = sum(1 for i in invoices if i.get("status") == "void")
    draft = sum(1 for i in invoices if i.get("status") == "draft")

    total_amount = sum(int(i.get("amount_cents", 0) or 0) for i in invoices)
    paid_amount = sum(
        int(i.get("amount_cents", 0) or 0)
        for i in invoices
        if i.get("status") == "paid"
    )
    overdue_amount = sum(
        int(i.get("amount_cents", 0) or 0)
        for i in invoices
        if i.get("status") == "overdue"
    )
    open_amount = sum(
        int(i.get("amount_cents", 0) or 0)
        for i in invoices
        if i.get("status") == "open"
    )

    closed = paid + void
    collection_rate = (paid / closed * 100) if closed > 0 else 0.0
    avg_invoice = total_amount / total if total > 0 else 0.0

    return {
        "total": total,
        "open": open_count,
        "paid": paid,
        "overdue": overdue,
        "void": void,
        "draft": draft,
        "total_amount": total_amount,
        "paid_amount": paid_amount,
        "overdue_amount": overdue_amount,
        "open_amount": open_amount,
        "collection_rate": collection_rate,
        "avg_invoice": avg_invoice,
    }


def format_invoice_status(status: str) -> tuple[str, str]:
    """
    Return (display_text, color) for an invoice status.
    """
    status_map = {
        "draft": ("Draft", "#8B8B8B"),
        "open": ("Open", "#3B82F6"),
        "paid": ("Paid", "#10B981"),
        "overdue": ("Overdue", "#EF4444"),
        "void": ("Void", "#6B7280"),
        "uncollectible": ("Uncollectible", "#DC2626"),
    }
    text, color = status_map.get(status, (status.title(), "#8B8B8B"))
    return text, color


def format_amount_cents(amount_cents: int, currency: str = "usd") -> str:
    """
    Format amount in cents to display string.

    >>> format_amount_cents(150000)
    '$1,500.00'
    >>> format_amount_cents(0)
    '$0.00'
    """
    symbols = {"usd": "$", "eur": "\u20ac", "gbp": "\u00a3"}
    symbol = symbols.get(currency.lower(), "$")
    return f"{symbol}{amount_cents / 100:,.2f}"


# ─── Reminder Helpers ────────────────────────────────────────

def compute_reminder_stats(reminders: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Compute aggregate stats from a list of payment reminders.

    Returns:
        Dict with: total, sent, pending, draft, approved,
        by_tone (polite/firm/final counts)
    """
    total = len(reminders)
    sent = sum(1 for r in reminders if r.get("status") == "sent")
    draft = sum(1 for r in reminders if r.get("status") == "draft")
    approved = sum(1 for r in reminders if r.get("status") == "approved")
    rejected = sum(1 for r in reminders if r.get("status") == "rejected")
    pending = draft + approved

    by_tone = {
        "polite": sum(1 for r in reminders if r.get("tone") == "polite"),
        "firm": sum(1 for r in reminders if r.get("tone") == "firm"),
        "final": sum(1 for r in reminders if r.get("tone") == "final"),
    }

    return {
        "total": total,
        "sent": sent,
        "draft": draft,
        "approved": approved,
        "rejected": rejected,
        "pending": pending,
        "by_tone": by_tone,
    }


def format_reminder_tone(tone: str) -> tuple[str, str]:
    """
    Return (display_text, color) for a reminder tone.
    """
    tone_map = {
        "polite": ("Polite", "#3B82F6"),
        "firm": ("Firm", "#E8A838"),
        "final": ("Final Notice", "#EF4444"),
    }
    text, color = tone_map.get(tone, (tone.title(), "#8B8B8B"))
    return text, color


# ─── Client Helpers ──────────────────────────────────────────

def compute_client_stats(clients: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Compute aggregate stats from a list of client records.

    Returns:
        Dict with: total, active, onboarding, at_risk, churned,
        avg_sentiment, avg_churn_risk, onboarding_rate
    """
    total = len(clients)
    active = sum(1 for c in clients if c.get("status") == "active")
    onboarding = sum(1 for c in clients if c.get("status") == "onboarding")
    at_risk = sum(1 for c in clients if c.get("status") == "at_risk")
    churned = sum(1 for c in clients if c.get("status") == "churned")
    paused = sum(1 for c in clients if c.get("status") == "paused")

    sentiments = [
        float(c.get("sentiment_score", 0) or 0)
        for c in clients
        if c.get("status") in ("active", "at_risk")
    ]
    avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0.0

    risks = [
        float(c.get("churn_risk", 0) or 0)
        for c in clients
        if c.get("status") in ("active", "at_risk")
    ]
    avg_churn_risk = sum(risks) / len(risks) if risks else 0.0

    completed_onboarding = sum(
        1 for c in clients if c.get("onboarding_complete", False)
    )
    onboarding_rate = (
        (completed_onboarding / total * 100) if total > 0 else 0.0
    )

    return {
        "total": total,
        "active": active,
        "onboarding": onboarding,
        "at_risk": at_risk,
        "churned": churned,
        "paused": paused,
        "avg_sentiment": avg_sentiment,
        "avg_churn_risk": avg_churn_risk,
        "onboarding_rate": onboarding_rate,
    }


def format_client_status(status: str) -> tuple[str, str]:
    """
    Return (display_text, color) for a client status.
    """
    status_map = {
        "onboarding": ("Onboarding", "#8B5CF6"),
        "active": ("Active", "#10B981"),
        "at_risk": ("At Risk", "#EF4444"),
        "churned": ("Churned", "#6B7280"),
        "paused": ("Paused", "#E8A838"),
    }
    text, color = status_map.get(status, (status.title(), "#8B8B8B"))
    return text, color


def format_churn_risk(risk: float) -> tuple[str, str]:
    """
    Return (display_text, color) for a churn risk score.

    >>> format_churn_risk(0.2)
    ('Low', '#10B981')
    >>> format_churn_risk(0.5)
    ('Moderate', '#E8A838')
    >>> format_churn_risk(0.8)
    ('Critical', '#EF4444')
    """
    if risk < 0.3:
        return "Low", "#10B981"
    elif risk < 0.5:
        return "Moderate", "#E8A838"
    elif risk < 0.7:
        return "High", "#F59E0B"
    else:
        return "Critical", "#EF4444"


# ─── CS Interaction Helpers ──────────────────────────────────

def compute_interaction_stats(interactions: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Compute aggregate stats from a list of CS interactions.

    Returns:
        Dict with: total, sent, pending, by_type counts
    """
    total = len(interactions)
    sent = sum(1 for i in interactions if i.get("status") == "sent")
    draft = sum(1 for i in interactions if i.get("status") == "draft")
    approved = sum(1 for i in interactions if i.get("status") == "approved")
    pending = draft + approved

    by_type = {
        "onboarding": sum(1 for i in interactions if i.get("checkin_type") == "onboarding"),
        "30_day": sum(1 for i in interactions if i.get("checkin_type") == "30_day"),
        "60_day": sum(1 for i in interactions if i.get("checkin_type") == "60_day"),
        "quarterly": sum(1 for i in interactions if i.get("checkin_type") == "quarterly"),
        "health_check": sum(1 for i in interactions if i.get("checkin_type") == "health_check"),
    }

    return {
        "total": total,
        "sent": sent,
        "draft": draft,
        "approved": approved,
        "pending": pending,
        "by_type": by_type,
    }


def format_interaction_type(itype: str) -> tuple[str, str]:
    """
    Return (display_text, color) for an interaction type.
    """
    type_map = {
        "onboarding": ("Onboarding", "#8B5CF6"),
        "30_day": ("30-Day Review", "#3B82F6"),
        "60_day": ("60-Day Review", "#3B82F6"),
        "quarterly": ("Quarterly Review", "#10B981"),
        "health_check": ("Health Check", "#EF4444"),
        "checkin": ("Check-in", "#3B82F6"),
        "escalation": ("Escalation", "#DC2626"),
        "renewal": ("Renewal", "#10B981"),
        "feedback": ("Feedback", "#E8A838"),
    }
    text, color = type_map.get(itype, (itype.replace("_", " ").title(), "#8B8B8B"))
    return text, color


# ─── Operations Score ────────────────────────────────────────

def compute_operations_score(
    invoice_stats: dict[str, Any],
    client_stats: dict[str, Any],
    reminder_stats: dict[str, Any],
) -> dict[str, Any]:
    """
    Compute a composite operations health score (0-100).

    Weights:
    - Collection rate: 40%  (paid/total invoices)
    - Client health: 30%    (1 - avg churn risk)
    - Responsiveness: 30%   (reminders sent / total reminders)

    Returns:
        Dict with: score, grade, collection_score, health_score,
        responsiveness_score
    """
    # Collection score (0-100): % of invoices collected
    collection_score = min(invoice_stats.get("collection_rate", 0), 100)

    # Client health score (0-100): inverse of avg churn risk
    avg_risk = client_stats.get("avg_churn_risk", 0)
    health_score = (1 - avg_risk) * 100 if client_stats.get("total", 0) > 0 else 50

    # Responsiveness score (0-100): % of reminders actually sent
    total_reminders = reminder_stats.get("total", 0)
    sent_reminders = reminder_stats.get("sent", 0)
    responsiveness_score = (
        (sent_reminders / total_reminders * 100)
        if total_reminders > 0
        else 50  # No reminders needed = neutral
    )

    # Weighted composite
    score = (
        collection_score * 0.4
        + health_score * 0.3
        + responsiveness_score * 0.3
    )

    # Grade
    if score >= 90:
        grade = "A"
    elif score >= 80:
        grade = "B"
    elif score >= 70:
        grade = "C"
    elif score >= 60:
        grade = "D"
    else:
        grade = "F"

    return {
        "score": round(score, 1),
        "grade": grade,
        "collection_score": round(collection_score, 1),
        "health_score": round(health_score, 1),
        "responsiveness_score": round(responsiveness_score, 1),
    }
