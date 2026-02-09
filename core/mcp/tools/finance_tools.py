"""
Finance MCP tools for the Sovereign Venture Engine.

Provides tools for B2B invoicing, payment tracking, and financial
reporting. Wraps StripeInvoiceClient operations.

Tools:
    - generate_invoice_from_proposal: Creates invoice from accepted proposal
    - check_overdue_invoices: Lists unpaid invoices past due date
    - send_payment_reminder: Drafts/sends payment reminder email
    - get_monthly_pnl: Aggregates revenue vs costs
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Lazy-initialized client
_stripe_client = None


def _get_client():
    """Lazily initialize the Stripe invoice client."""
    global _stripe_client
    if _stripe_client is None:
        from core.integrations.finance.stripe_invoice_client import StripeInvoiceClient
        _stripe_client = StripeInvoiceClient.from_env()
    return _stripe_client


async def generate_invoice_from_proposal(
    proposal_id: str,
    company_name: str,
    contact_email: str,
    contact_name: str = "",
    amount_cents: int = 0,
    description: str = "Professional Services",
    due_days: int = 30,
    currency: str = "usd",
) -> str:
    """
    Generate a Stripe invoice from an accepted proposal.

    Creates a customer (if needed) and generates a draft invoice
    with line items derived from the proposal.

    Returns JSON with invoice details.
    """
    client = _get_client()

    try:
        # Create customer
        customer = await client.create_customer(
            email=contact_email,
            name=contact_name,
            company=company_name,
            metadata={"proposal_id": proposal_id},
        )

        # Create invoice with line items
        line_items = [
            {
                "description": description,
                "amount": amount_cents,  # In cents
                "quantity": 1,
            },
        ]

        invoice = await client.create_invoice(
            customer_id=customer["id"],
            line_items=line_items,
            due_days=due_days,
            currency=currency,
            memo=f"Invoice for proposal {proposal_id}",
            metadata={"proposal_id": proposal_id},
        )

        # Finalize (make payable)
        finalized = await client.finalize_invoice(invoice["id"])

        result = {
            "status": "success",
            "invoice_id": invoice["id"],
            "customer_id": customer["id"],
            "total_amount_cents": amount_cents,
            "total_amount_display": f"${amount_cents / 100:,.2f}",
            "currency": currency,
            "due_days": due_days,
            "invoice_status": finalized.get("status", "open"),
            "hosted_url": finalized.get("hosted_invoice_url", invoice.get("hosted_invoice_url", "")),
            "proposal_id": proposal_id,
        }

        logger.info(
            "invoice_generated",
            extra={
                "invoice_id": invoice["id"],
                "amount": amount_cents,
                "proposal_id": proposal_id,
            },
        )

        return json.dumps(result)

    except Exception as e:
        logger.error(f"Invoice generation failed: {e}")
        return json.dumps({
            "status": "error",
            "error": str(e)[:200],
            "proposal_id": proposal_id,
        })


async def check_overdue_invoices(
    days_overdue: int = 7,
) -> str:
    """
    Check for overdue invoices (unpaid past due date).

    Returns JSON with list of overdue invoices and summary.
    """
    client = _get_client()

    try:
        # Get all open invoices
        open_invoices = await client.list_invoices(status="open")

        now = datetime.now(timezone.utc)
        overdue: list[dict[str, Any]] = []

        for inv in open_invoices:
            due_date_str = inv.get("due_date", "")
            if not due_date_str:
                continue

            try:
                due_date = datetime.fromisoformat(due_date_str.replace("Z", "+00:00"))
                if due_date.tzinfo is None:
                    due_date = due_date.replace(tzinfo=timezone.utc)
                days_past = (now - due_date).days
                if days_past >= days_overdue:
                    inv["days_overdue"] = days_past
                    overdue.append(inv)
            except (ValueError, TypeError):
                continue

        total_overdue = sum(inv.get("amount_due", 0) for inv in overdue)

        result = {
            "status": "success",
            "overdue_count": len(overdue),
            "total_overdue_cents": total_overdue,
            "total_overdue_display": f"${total_overdue / 100:,.2f}",
            "threshold_days": days_overdue,
            "invoices": overdue,
        }

        logger.info(
            "overdue_check_complete",
            extra={
                "overdue_count": len(overdue),
                "total_overdue": total_overdue,
            },
        )

        return json.dumps(result, default=str)

    except Exception as e:
        logger.error(f"Overdue check failed: {e}")
        return json.dumps({"status": "error", "error": str(e)[:200]})


async def send_payment_reminder(
    invoice_id: str,
    contact_email: str,
    company_name: str = "",
    amount_display: str = "",
    days_overdue: int = 0,
    tone: str = "polite",
) -> str:
    """
    Generate a payment reminder draft for an overdue invoice.

    Tone options: "polite" (7-14 days), "firm" (15-30 days), "final" (30+ days).
    Returns the draft text — human review required before sending.
    """
    tone_templates = {
        "polite": (
            f"Hi,\n\n"
            f"This is a friendly reminder that invoice {invoice_id} for {amount_display} "
            f"is {days_overdue} days past due. We understand things can get busy!\n\n"
            f"Could you let us know when we can expect payment? "
            f"If you've already sent it, please disregard this message.\n\n"
            f"Best regards"
        ),
        "firm": (
            f"Dear {company_name} team,\n\n"
            f"We're writing regarding invoice {invoice_id} for {amount_display}, "
            f"which is now {days_overdue} days overdue.\n\n"
            f"To avoid any service interruption, please arrange payment "
            f"at your earliest convenience. If there are any issues with the invoice, "
            f"please let us know immediately so we can resolve them.\n\n"
            f"Thank you for your prompt attention."
        ),
        "final": (
            f"Dear {company_name} team,\n\n"
            f"FINAL NOTICE: Invoice {invoice_id} for {amount_display} "
            f"is now {days_overdue} days overdue.\n\n"
            f"Please remit payment within 5 business days to avoid "
            f"escalation of this matter. If payment has already been sent, "
            f"please forward confirmation.\n\n"
            f"If you're experiencing financial difficulties, we're open to "
            f"discussing a payment plan. Please contact us directly."
        ),
    }

    draft = tone_templates.get(tone, tone_templates["polite"])

    result = {
        "status": "draft_ready",
        "invoice_id": invoice_id,
        "contact_email": contact_email,
        "company_name": company_name,
        "tone": tone,
        "days_overdue": days_overdue,
        "draft_text": draft,
        "requires_human_review": True,
    }

    logger.info(
        "payment_reminder_drafted",
        extra={
            "invoice_id": invoice_id,
            "tone": tone,
            "days_overdue": days_overdue,
        },
    )

    return json.dumps(result)


async def get_monthly_pnl(
    vertical_id: str = "",
) -> str:
    """
    Calculate monthly P&L from all revenue streams.

    Aggregates:
    - Service revenue (invoices paid)
    - Commerce revenue (Stripe payments from e-commerce)
    - Costs (API usage, infrastructure estimates)

    Returns JSON with P&L breakdown.
    """
    client = _get_client()

    try:
        # Get paid invoices (service revenue)
        paid_invoices = await client.list_invoices(status="paid")
        service_revenue = sum(inv.get("total_amount", 0) for inv in paid_invoices)

        # Open invoices (accounts receivable)
        open_invoices = await client.list_invoices(status="open")
        accounts_receivable = sum(inv.get("amount_due", 0) for inv in open_invoices)

        # Estimate costs (API calls, infrastructure)
        # In production, pull from actual usage metrics
        estimated_api_costs = 0  # Placeholder — connect to actual billing
        estimated_infra_costs = 0

        total_costs = estimated_api_costs + estimated_infra_costs
        net_profit = service_revenue - total_costs

        result = {
            "status": "success",
            "period": datetime.now(timezone.utc).strftime("%Y-%m"),
            "service_revenue_cents": service_revenue,
            "service_revenue_display": f"${service_revenue / 100:,.2f}",
            "commerce_revenue_cents": 0,  # From CommerceClient in production
            "commerce_revenue_display": "$0.00",
            "total_revenue_cents": service_revenue,
            "total_revenue_display": f"${service_revenue / 100:,.2f}",
            "accounts_receivable_cents": accounts_receivable,
            "accounts_receivable_display": f"${accounts_receivable / 100:,.2f}",
            "total_costs_cents": total_costs,
            "total_costs_display": f"${total_costs / 100:,.2f}",
            "net_profit_cents": net_profit,
            "net_profit_display": f"${net_profit / 100:,.2f}",
            "invoices_paid": len(paid_invoices),
            "invoices_open": len(open_invoices),
            "vertical_id": vertical_id,
        }

        logger.info(
            "pnl_calculated",
            extra={
                "service_revenue": service_revenue,
                "net_profit": net_profit,
            },
        )

        return json.dumps(result)

    except Exception as e:
        logger.error(f"P&L calculation failed: {e}")
        return json.dumps({"status": "error", "error": str(e)[:200]})
