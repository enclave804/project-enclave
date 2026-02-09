"""
Stripe Invoice Client for B2B service revenue.

Distinct from CommerceClient (Phase 8) which handles e-commerce payments.
This client handles professional service invoicing: create customers,
generate invoices from proposals, track payment status, chase overdue.

Mock mode: When STRIPE_SECRET_KEY is not configured, all operations are
logged to storage/finance_mock.json for development and testing.

Usage:
    client = StripeInvoiceClient.from_env()
    # client auto-detects mock mode if no API key

    # Create a customer
    customer = await client.create_customer(
        email="jane@acme.com",
        name="Jane Doe",
        company="Acme Corp",
    )

    # Generate an invoice
    invoice = await client.create_invoice(
        customer_id=customer["id"],
        line_items=[
            {"description": "Security Assessment", "amount": 15000_00, "quantity": 1},
        ],
        due_days=30,
    )

    # Finalize and send
    await client.finalize_invoice(invoice["id"])

    # Check status
    status = await client.get_invoice(invoice["id"])
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Mock storage path
MOCK_STORAGE_DIR = Path("storage")
MOCK_STORAGE_FILE = MOCK_STORAGE_DIR / "finance_mock.json"
MAX_MOCK_ENTRIES = 500


class StripeInvoiceClient:
    """
    Stripe Invoicing client with automatic mock mode.

    When STRIPE_SECRET_KEY is missing, operations are simulated
    and logged to storage/finance_mock.json.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        mock_mode: bool = False,
    ):
        self.api_key = api_key
        self.mock_mode = mock_mode or not api_key
        self._stripe = None
        self._mock_customers: dict[str, dict[str, Any]] = {}
        self._mock_invoices: dict[str, dict[str, Any]] = {}

        if self.mock_mode:
            logger.info(
                "stripe_invoice_mock_mode",
                extra={"reason": "No STRIPE_SECRET_KEY configured"},
            )
        else:
            try:
                import stripe
                stripe.api_key = self.api_key
                self._stripe = stripe
                logger.info("stripe_invoice_live_mode")
            except ImportError:
                logger.warning("stripe package not installed, falling back to mock mode")
                self.mock_mode = True

    @classmethod
    def from_env(cls) -> StripeInvoiceClient:
        """Create client from environment variables."""
        api_key = os.environ.get("STRIPE_SECRET_KEY", "")
        return cls(api_key=api_key if api_key else None)

    # ─── Customer Operations ──────────────────────────────────────────

    async def create_customer(
        self,
        email: str,
        name: str = "",
        company: str = "",
        metadata: Optional[dict[str, str]] = None,
    ) -> dict[str, Any]:
        """Create a Stripe customer for invoicing."""
        if self.mock_mode:
            customer_id = f"cus_mock_{uuid.uuid4().hex[:12]}"
            customer = {
                "id": customer_id,
                "email": email,
                "name": name or company,
                "company": company,
                "metadata": metadata or {},
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            self._mock_customers[customer_id] = customer
            self._log_mock("create_customer", customer)
            logger.info(
                "stripe_customer_created_mock",
                extra={"customer_id": customer_id, "email": email},
            )
            return customer

        # Real Stripe
        stripe_meta = metadata or {}
        if company:
            stripe_meta["company"] = company

        result = self._stripe.Customer.create(
            email=email,
            name=name or company,
            metadata=stripe_meta,
        )
        return {
            "id": result["id"],
            "email": result["email"],
            "name": result.get("name", ""),
            "company": company,
            "metadata": dict(result.get("metadata", {})),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

    async def get_customer(self, customer_id: str) -> Optional[dict[str, Any]]:
        """Retrieve a customer by ID."""
        if self.mock_mode:
            return self._mock_customers.get(customer_id)

        try:
            result = self._stripe.Customer.retrieve(customer_id)
            return {
                "id": result["id"],
                "email": result.get("email", ""),
                "name": result.get("name", ""),
                "metadata": dict(result.get("metadata", {})),
            }
        except Exception as e:
            logger.error(f"Failed to get customer {customer_id}: {e}")
            return None

    # ─── Invoice Operations ───────────────────────────────────────────

    async def create_invoice(
        self,
        customer_id: str,
        line_items: list[dict[str, Any]],
        due_days: int = 30,
        currency: str = "usd",
        memo: str = "",
        metadata: Optional[dict[str, str]] = None,
    ) -> dict[str, Any]:
        """
        Create a draft invoice with line items.

        Args:
            customer_id: Stripe customer ID
            line_items: List of {description, amount (cents), quantity}
            due_days: Days until due (Net 30, Net 60, etc.)
            currency: ISO currency code
            memo: Invoice memo/notes
            metadata: Additional key-value pairs
        """
        due_date = (datetime.now(timezone.utc) + timedelta(days=due_days)).isoformat()
        total_amount = sum(
            item.get("amount", 0) * item.get("quantity", 1)
            for item in line_items
        )

        if self.mock_mode:
            invoice_id = f"inv_mock_{uuid.uuid4().hex[:12]}"
            invoice = {
                "id": invoice_id,
                "customer_id": customer_id,
                "status": "draft",
                "currency": currency,
                "line_items": line_items,
                "total_amount": total_amount,
                "amount_due": total_amount,
                "due_date": due_date,
                "memo": memo,
                "metadata": metadata or {},
                "created_at": datetime.now(timezone.utc).isoformat(),
                "finalized_at": "",
                "paid_at": "",
                "hosted_invoice_url": f"https://invoice.stripe.com/i/mock/{invoice_id}",
            }
            self._mock_invoices[invoice_id] = invoice
            self._log_mock("create_invoice", invoice)
            logger.info(
                "stripe_invoice_created_mock",
                extra={
                    "invoice_id": invoice_id,
                    "total": total_amount,
                    "items": len(line_items),
                },
            )
            return invoice

        # Real Stripe
        inv = self._stripe.Invoice.create(
            customer=customer_id,
            currency=currency,
            collection_method="send_invoice",
            days_until_due=due_days,
            metadata=metadata or {},
        )

        # Add line items
        for item in line_items:
            self._stripe.InvoiceItem.create(
                customer=customer_id,
                invoice=inv["id"],
                description=item.get("description", "Service"),
                amount=item.get("amount", 0),
                currency=currency,
                quantity=item.get("quantity", 1),
            )

        # Refresh invoice to get updated totals
        inv = self._stripe.Invoice.retrieve(inv["id"])

        return {
            "id": inv["id"],
            "customer_id": customer_id,
            "status": inv["status"],
            "currency": inv["currency"],
            "line_items": line_items,
            "total_amount": inv.get("total", 0),
            "amount_due": inv.get("amount_due", 0),
            "due_date": due_date,
            "memo": memo,
            "metadata": dict(inv.get("metadata", {})),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "hosted_invoice_url": inv.get("hosted_invoice_url", ""),
        }

    async def finalize_invoice(self, invoice_id: str) -> dict[str, Any]:
        """Finalize a draft invoice (makes it payable and sends to customer)."""
        if self.mock_mode:
            inv = self._mock_invoices.get(invoice_id, {})
            inv["status"] = "open"
            inv["finalized_at"] = datetime.now(timezone.utc).isoformat()
            self._log_mock("finalize_invoice", {"invoice_id": invoice_id})
            logger.info(
                "stripe_invoice_finalized_mock",
                extra={"invoice_id": invoice_id},
            )
            return inv

        result = self._stripe.Invoice.finalize_invoice(invoice_id)
        return {
            "id": result["id"],
            "status": result["status"],
            "hosted_invoice_url": result.get("hosted_invoice_url", ""),
        }

    async def get_invoice(self, invoice_id: str) -> Optional[dict[str, Any]]:
        """Retrieve an invoice by ID."""
        if self.mock_mode:
            return self._mock_invoices.get(invoice_id)

        try:
            result = self._stripe.Invoice.retrieve(invoice_id)
            return {
                "id": result["id"],
                "customer_id": result.get("customer", ""),
                "status": result["status"],
                "total_amount": result.get("total", 0),
                "amount_due": result.get("amount_due", 0),
                "currency": result.get("currency", "usd"),
                "hosted_invoice_url": result.get("hosted_invoice_url", ""),
                "due_date": result.get("due_date", ""),
                "paid_at": "",
            }
        except Exception as e:
            logger.error(f"Failed to get invoice {invoice_id}: {e}")
            return None

    async def list_invoices(
        self,
        status: Optional[str] = None,
        customer_id: Optional[str] = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """List invoices, optionally filtered by status or customer."""
        if self.mock_mode:
            invoices = list(self._mock_invoices.values())
            if status:
                invoices = [i for i in invoices if i.get("status") == status]
            if customer_id:
                invoices = [i for i in invoices if i.get("customer_id") == customer_id]
            return invoices[:limit]

        params: dict[str, Any] = {"limit": limit}
        if status:
            params["status"] = status
        if customer_id:
            params["customer"] = customer_id

        result = self._stripe.Invoice.list(**params)
        return [
            {
                "id": inv["id"],
                "customer_id": inv.get("customer", ""),
                "status": inv["status"],
                "total_amount": inv.get("total", 0),
                "amount_due": inv.get("amount_due", 0),
                "currency": inv.get("currency", "usd"),
                "due_date": inv.get("due_date", ""),
                "hosted_invoice_url": inv.get("hosted_invoice_url", ""),
            }
            for inv in result.get("data", [])
        ]

    async def void_invoice(self, invoice_id: str) -> dict[str, Any]:
        """Void an open invoice (cancel without deleting)."""
        if self.mock_mode:
            inv = self._mock_invoices.get(invoice_id, {})
            inv["status"] = "void"
            self._log_mock("void_invoice", {"invoice_id": invoice_id})
            return inv

        result = self._stripe.Invoice.void_invoice(invoice_id)
        return {"id": result["id"], "status": result["status"]}

    # ─── Mock Storage ─────────────────────────────────────────────────

    def _log_mock(self, action: str, data: dict[str, Any]) -> None:
        """Log mock operation to storage/finance_mock.json."""
        try:
            MOCK_STORAGE_DIR.mkdir(parents=True, exist_ok=True)

            entries: list[dict[str, Any]] = []
            if MOCK_STORAGE_FILE.exists():
                try:
                    entries = json.loads(MOCK_STORAGE_FILE.read_text())
                except (json.JSONDecodeError, OSError):
                    entries = []

            entries.append({
                "action": action,
                "data": data,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

            # Keep only last N entries
            if len(entries) > MAX_MOCK_ENTRIES:
                entries = entries[-MAX_MOCK_ENTRIES:]

            MOCK_STORAGE_FILE.write_text(json.dumps(entries, indent=2, default=str))
        except Exception as e:
            logger.debug(f"Mock storage write failed: {e}")
