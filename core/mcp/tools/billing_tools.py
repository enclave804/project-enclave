"""
Billing MCP tools for the Sovereign Venture Engine.

Provides tools for invoice creation, payment tracking, reminder
generation, and line-item calculation. Used by the Invoice Agent
and Finance Agent for B2B billing workflows.

Tools:
    - create_invoice: Create a new invoice with line items
    - check_payment_status: Check payment status for a specific invoice
    - send_payment_reminder: Generate and queue a payment reminder
    - calculate_line_items: Calculate totals from line items
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)


async def create_invoice(
    company_name: str,
    contact_email: str,
    line_items: list,
    due_days: int = 30,
    currency: str = "usd",
) -> str:
    """
    Create a new invoice with line items.

    Each line item should be a dict with keys:
        description (str), quantity (int), unit_price_cents (int)

    Returns JSON with invoice_id, total, status, and due_date.
    """
    try:
        now = datetime.now(timezone.utc)

        # Calculate totals from line items
        total_cents = 0
        processed_items: list[dict[str, Any]] = []
        for item in line_items:
            qty = item.get("quantity", 1)
            unit_price = item.get("unit_price_cents", 0)
            item_total = qty * unit_price
            total_cents += item_total
            processed_items.append({
                "description": item.get("description", ""),
                "quantity": qty,
                "unit_price_cents": unit_price,
                "total_cents": item_total,
            })

        # Stub: generate a deterministic invoice ID
        invoice_id = f"inv_{now.strftime('%Y%m%d%H%M%S')}_{abs(hash(contact_email)) % 100000:05d}"

        result = {
            "status": "success",
            "invoice_id": invoice_id,
            "company_name": company_name,
            "contact_email": contact_email,
            "currency": currency,
            "line_items": processed_items,
            "total_cents": total_cents,
            "total_display": f"${total_cents / 100:,.2f}",
            "due_days": due_days,
            "created_at": now.isoformat(),
            "invoice_status": "draft",
        }

        logger.info(
            "invoice_created",
            extra={
                "invoice_id": invoice_id,
                "total_cents": total_cents,
                "company": company_name,
            },
        )

        return json.dumps(result)

    except Exception as e:
        logger.error(f"Invoice creation failed: {e}")
        return json.dumps({
            "status": "error",
            "error": str(e)[:200],
            "company_name": company_name,
        })


async def check_payment_status(invoice_id: str) -> str:
    """
    Check payment status for a specific invoice.

    Returns JSON with invoice_id, status, amount_due, amount_paid,
    and payment history.
    """
    try:
        # Stub: return mock payment status
        result = {
            "status": "success",
            "invoice_id": invoice_id,
            "invoice_status": "open",
            "amount_due_cents": 0,
            "amount_paid_cents": 0,
            "currency": "usd",
            "created_at": None,
            "due_date": None,
            "days_overdue": 0,
            "payment_history": [],
        }

        logger.info(
            "payment_status_checked",
            extra={"invoice_id": invoice_id},
        )

        return json.dumps(result, default=str)

    except Exception as e:
        logger.error(f"Payment status check failed: {e}")
        return json.dumps({
            "status": "error",
            "error": str(e)[:200],
            "invoice_id": invoice_id,
        })


async def send_payment_reminder(
    invoice_id: str,
    contact_email: str,
    tone: str = "polite",
    days_overdue: int = 0,
) -> str:
    """
    Generate and queue a payment reminder for an overdue invoice.

    Tone options: "polite" (1-14 days), "firm" (15-30 days), "final" (30+ days).
    Returns the draft reminder text for human review before sending.
    """
    try:
        tone_labels = {
            "polite": "Friendly Reminder",
            "firm": "Payment Required",
            "final": "Final Notice",
        }

        subject = f"{tone_labels.get(tone, 'Payment Reminder')} — Invoice {invoice_id}"

        result = {
            "status": "draft_ready",
            "invoice_id": invoice_id,
            "contact_email": contact_email,
            "tone": tone,
            "days_overdue": days_overdue,
            "subject": subject,
            "draft_text": f"[{tone.upper()} REMINDER] Invoice {invoice_id} is {days_overdue} days overdue.",
            "requires_human_review": True,
            "queued_at": datetime.now(timezone.utc).isoformat(),
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

    except Exception as e:
        logger.error(f"Payment reminder generation failed: {e}")
        return json.dumps({
            "status": "error",
            "error": str(e)[:200],
            "invoice_id": invoice_id,
        })


async def calculate_line_items(items: list) -> str:
    """
    Calculate totals from line items.

    Each item should be a dict with:
        description (str), quantity (int), unit_price_cents (int)

    Returns JSON with itemized breakdown, subtotal, tax, and total.
    """
    try:
        processed: list[dict[str, Any]] = []
        subtotal_cents = 0

        for item in items:
            description = item.get("description", "")
            quantity = item.get("quantity", 1)
            unit_price_cents = item.get("unit_price_cents", 0)
            line_total_cents = quantity * unit_price_cents

            subtotal_cents += line_total_cents
            processed.append({
                "description": description,
                "quantity": quantity,
                "unit_price_cents": unit_price_cents,
                "unit_price_display": f"${unit_price_cents / 100:,.2f}",
                "line_total_cents": line_total_cents,
                "line_total_display": f"${line_total_cents / 100:,.2f}",
            })

        result = {
            "status": "success",
            "line_items": processed,
            "item_count": len(processed),
            "subtotal_cents": subtotal_cents,
            "subtotal_display": f"${subtotal_cents / 100:,.2f}",
            "tax_cents": 0,
            "tax_display": "$0.00",
            "total_cents": subtotal_cents,
            "total_display": f"${subtotal_cents / 100:,.2f}",
        }

        logger.info(
            "line_items_calculated",
            extra={
                "item_count": len(processed),
                "subtotal_cents": subtotal_cents,
            },
        )

        return json.dumps(result)

    except Exception as e:
        logger.error(f"Line item calculation failed: {e}")
        return json.dumps({
            "status": "error",
            "error": str(e)[:200],
        })
