"""
Commerce integration for the Sovereign Venture Engine.

Provider-agnostic commerce client that supports:
- Shopify (product catalog, inventory, orders)
- Stripe (payment links, payment status)
- Mock mode (for development/testing — returns realistic dummy data)

The active providers are controlled via env vars:
- SHOPIFY_API_KEY / SHOPIFY_STORE_URL → Shopify provider
- STRIPE_API_KEY → Stripe provider
- If missing → MockCommerceProvider (safe for development)

Usage:
    client = CommerceClient.from_env()
    products = await client.get_products(limit=10)
    orders = await client.get_recent_orders(days=1)
    link = await client.create_payment_link("Widget", 2999)
    status = await client.check_payment("pi_xxx")
"""

from __future__ import annotations

import logging
import os
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract Interfaces
# ---------------------------------------------------------------------------


class StorefrontProvider(ABC):
    """Abstract storefront provider (Shopify, WooCommerce, etc.)."""

    @abstractmethod
    async def get_products(
        self,
        limit: int = 10,
        status: str = "active",
    ) -> list[dict[str, Any]]:
        """Return product catalog with variants."""
        ...

    @abstractmethod
    async def update_inventory(
        self,
        variant_id: str,
        quantity: int,
    ) -> dict[str, Any]:
        """Update inventory for a product variant."""
        ...

    @abstractmethod
    async def get_recent_orders(
        self,
        days: int = 1,
    ) -> list[dict[str, Any]]:
        """Return orders from the last N days."""
        ...

    @abstractmethod
    async def get_order(self, order_id: str) -> dict[str, Any]:
        """Get details for a single order."""
        ...


class PaymentProvider(ABC):
    """Abstract payment provider (Stripe, Square, etc.)."""

    @abstractmethod
    async def create_payment_link(
        self,
        product_name: str,
        amount_cents: int,
        currency: str = "usd",
    ) -> dict[str, Any]:
        """Create a hosted payment link for a product."""
        ...

    @abstractmethod
    async def check_payment(
        self,
        payment_intent_id: str,
    ) -> dict[str, Any]:
        """Check the status of a payment intent."""
        ...

    @abstractmethod
    async def process_refund(
        self,
        payment_intent_id: str,
        amount_cents: Optional[int] = None,
        reason: str = "",
    ) -> dict[str, Any]:
        """Process a refund (full or partial)."""
        ...


# ---------------------------------------------------------------------------
# Mock Providers (Development / Testing)
# ---------------------------------------------------------------------------

MOCK_PRODUCTS = [
    {
        "product_id": "prod_001",
        "title": "Custom 3D Printed Figurine",
        "description": "High-detail resin figurine, made to order.",
        "status": "active",
        "vendor": "PrintShop Pro",
        "product_type": "Figurine",
        "tags": ["3d-print", "custom", "resin"],
        "variants": [
            {
                "variant_id": "var_001a",
                "title": "Small (4 inch)",
                "price": "29.99",
                "inventory_quantity": 50,
                "sku": "FIG-SM-001",
            },
            {
                "variant_id": "var_001b",
                "title": "Large (8 inch)",
                "price": "59.99",
                "inventory_quantity": 25,
                "sku": "FIG-LG-001",
            },
        ],
        "created_at": "2024-06-01T10:00:00Z",
    },
    {
        "product_id": "prod_002",
        "title": "Cybersecurity Assessment Report",
        "description": "Comprehensive security audit with actionable recommendations.",
        "status": "active",
        "vendor": "Enclave Guard",
        "product_type": "Service",
        "tags": ["cybersecurity", "assessment", "report"],
        "variants": [
            {
                "variant_id": "var_002a",
                "title": "Standard (up to 50 endpoints)",
                "price": "2999.00",
                "inventory_quantity": 999,
                "sku": "SEC-STD-001",
            },
            {
                "variant_id": "var_002b",
                "title": "Enterprise (unlimited endpoints)",
                "price": "9999.00",
                "inventory_quantity": 999,
                "sku": "SEC-ENT-001",
            },
        ],
        "created_at": "2024-07-15T14:30:00Z",
    },
    {
        "product_id": "prod_003",
        "title": "Branded T-Shirt",
        "description": "Premium cotton tee with company logo.",
        "status": "active",
        "vendor": "Merch Co",
        "product_type": "Apparel",
        "tags": ["merch", "t-shirt", "branded"],
        "variants": [
            {
                "variant_id": "var_003a",
                "title": "S",
                "price": "24.99",
                "inventory_quantity": 100,
                "sku": "TEE-S-001",
            },
            {
                "variant_id": "var_003b",
                "title": "M",
                "price": "24.99",
                "inventory_quantity": 3,
                "sku": "TEE-M-001",
            },
            {
                "variant_id": "var_003c",
                "title": "L",
                "price": "24.99",
                "inventory_quantity": 75,
                "sku": "TEE-L-001",
            },
        ],
        "created_at": "2024-08-01T09:00:00Z",
    },
]

MOCK_ORDERS = [
    {
        "order_id": "ord_1001",
        "order_number": "#1001",
        "customer_email": "alice@techcorp.com",
        "customer_name": "Alice Johnson",
        "total_price": "59.99",
        "currency": "USD",
        "financial_status": "paid",
        "fulfillment_status": "unfulfilled",
        "line_items": [
            {
                "product_id": "prod_001",
                "variant_id": "var_001b",
                "title": "Custom 3D Printed Figurine - Large",
                "quantity": 1,
                "price": "59.99",
            }
        ],
        "created_at": "2024-10-15T14:22:00Z",
        "tags": [],
    },
    {
        "order_id": "ord_1002",
        "order_number": "#1002",
        "customer_email": "bob.whale@megacorp.com",
        "customer_name": "Bob Whaleton",
        "total_price": "9999.00",
        "currency": "USD",
        "financial_status": "paid",
        "fulfillment_status": "unfulfilled",
        "line_items": [
            {
                "product_id": "prod_002",
                "variant_id": "var_002b",
                "title": "Cybersecurity Assessment Report - Enterprise",
                "quantity": 1,
                "price": "9999.00",
            }
        ],
        "created_at": "2024-10-15T16:45:00Z",
        "tags": ["vip"],
    },
    {
        "order_id": "ord_1003",
        "order_number": "#1003",
        "customer_email": "carol@startup.io",
        "customer_name": "Carol Smith",
        "total_price": "54.98",
        "currency": "USD",
        "financial_status": "paid",
        "fulfillment_status": "fulfilled",
        "line_items": [
            {
                "product_id": "prod_003",
                "variant_id": "var_003a",
                "title": "Branded T-Shirt - S",
                "quantity": 1,
                "price": "24.99",
            },
            {
                "product_id": "prod_003",
                "variant_id": "var_003b",
                "title": "Branded T-Shirt - M",
                "quantity": 1,
                "price": "24.99",
            },
        ],
        "created_at": "2024-10-14T11:30:00Z",
        "tags": [],
    },
]


class MockStorefrontProvider(StorefrontProvider):
    """
    Mock storefront for development and testing.

    Returns realistic product/order data without touching Shopify.
    Inventory updates are tracked in-memory.
    """

    def __init__(self):
        import copy
        self._products = copy.deepcopy(MOCK_PRODUCTS)
        self._orders = copy.deepcopy(MOCK_ORDERS)
        self._inventory_changes: list[dict[str, Any]] = []

    async def get_products(
        self,
        limit: int = 10,
        status: str = "active",
    ) -> list[dict[str, Any]]:
        filtered = [p for p in self._products if p["status"] == status]
        return filtered[:limit]

    async def update_inventory(
        self,
        variant_id: str,
        quantity: int,
    ) -> dict[str, Any]:
        for product in self._products:
            for variant in product.get("variants", []):
                if variant["variant_id"] == variant_id:
                    old_qty = variant["inventory_quantity"]
                    variant["inventory_quantity"] = quantity

                    change = {
                        "variant_id": variant_id,
                        "product_id": product["product_id"],
                        "old_quantity": old_qty,
                        "new_quantity": quantity,
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                    }
                    self._inventory_changes.append(change)

                    logger.info(
                        "mock_inventory_updated",
                        extra={
                            "variant_id": variant_id,
                            "old_qty": old_qty,
                            "new_qty": quantity,
                        },
                    )
                    return {
                        "success": True,
                        "variant_id": variant_id,
                        "old_quantity": old_qty,
                        "new_quantity": quantity,
                        "provider": "mock",
                    }

        return {
            "success": False,
            "error": f"Variant {variant_id} not found",
            "provider": "mock",
        }

    async def get_recent_orders(self, days: int = 1) -> list[dict[str, Any]]:
        # In mock mode, return all orders regardless of date
        return list(self._orders)

    async def get_order(self, order_id: str) -> dict[str, Any]:
        for order in self._orders:
            if order["order_id"] == order_id:
                return order
        return {"error": f"Order {order_id} not found"}


class MockPaymentProvider(PaymentProvider):
    """
    Mock payment provider for development and testing.

    Generates realistic-looking Stripe-style responses.
    """

    def __init__(self):
        self._payments: dict[str, dict[str, Any]] = {}
        self._refunds: list[dict[str, Any]] = []

    async def create_payment_link(
        self,
        product_name: str,
        amount_cents: int,
        currency: str = "usd",
    ) -> dict[str, Any]:
        link_id = f"plink_{uuid.uuid4().hex[:12]}"
        pi_id = f"pi_{uuid.uuid4().hex[:16]}"

        payment = {
            "payment_link_id": link_id,
            "payment_intent_id": pi_id,
            "url": f"https://checkout.stripe.com/c/pay/{link_id}",
            "product_name": product_name,
            "amount_cents": amount_cents,
            "currency": currency,
            "status": "pending",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "provider": "mock",
        }
        self._payments[pi_id] = payment

        logger.info(
            "mock_payment_link_created",
            extra={
                "link_id": link_id,
                "product": product_name,
                "amount_cents": amount_cents,
            },
        )
        return payment

    async def check_payment(
        self,
        payment_intent_id: str,
    ) -> dict[str, Any]:
        if payment_intent_id in self._payments:
            return self._payments[payment_intent_id]

        # Return a realistic mock for unknown IDs
        return {
            "payment_intent_id": payment_intent_id,
            "status": "succeeded",
            "amount_cents": 5999,
            "currency": "usd",
            "customer_email": "customer@example.com",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "provider": "mock",
        }

    async def process_refund(
        self,
        payment_intent_id: str,
        amount_cents: Optional[int] = None,
        reason: str = "",
    ) -> dict[str, Any]:
        refund_id = f"re_{uuid.uuid4().hex[:12]}"
        payment = self._payments.get(payment_intent_id, {})
        original_amount = payment.get("amount_cents", 0)
        refund_amount = amount_cents or original_amount

        refund = {
            "refund_id": refund_id,
            "payment_intent_id": payment_intent_id,
            "amount_cents": refund_amount,
            "currency": payment.get("currency", "usd"),
            "status": "succeeded",
            "reason": reason,
            "refund_type": "partial" if amount_cents and amount_cents < original_amount else "full",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "provider": "mock",
        }
        self._refunds.append(refund)

        logger.info(
            "mock_refund_processed",
            extra={
                "refund_id": refund_id,
                "payment_intent_id": payment_intent_id,
                "amount_cents": refund_amount,
            },
        )
        return refund


# ---------------------------------------------------------------------------
# Shopify Provider (Placeholder — interface stable)
# ---------------------------------------------------------------------------


class ShopifyProvider(StorefrontProvider):
    """
    Shopify Admin API provider.

    Requires:
        SHOPIFY_STORE_URL: e.g. "https://my-store.myshopify.com"
        SHOPIFY_API_KEY: Admin API access token

    Falls back to mock when not configured.
    """

    def __init__(
        self,
        store_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.store_url = store_url or os.environ.get("SHOPIFY_STORE_URL", "")
        self.api_key = api_key or os.environ.get("SHOPIFY_API_KEY", "")

    async def get_products(self, limit: int = 10, status: str = "active") -> list[dict[str, Any]]:
        logger.warning("shopify_not_implemented: falling back to mock products")
        mock = MockStorefrontProvider()
        return await mock.get_products(limit, status)

    async def update_inventory(self, variant_id: str, quantity: int) -> dict[str, Any]:
        logger.warning("shopify_not_implemented: falling back to mock inventory")
        mock = MockStorefrontProvider()
        return await mock.update_inventory(variant_id, quantity)

    async def get_recent_orders(self, days: int = 1) -> list[dict[str, Any]]:
        logger.warning("shopify_not_implemented: falling back to mock orders")
        mock = MockStorefrontProvider()
        return await mock.get_recent_orders(days)

    async def get_order(self, order_id: str) -> dict[str, Any]:
        logger.warning("shopify_not_implemented: falling back to mock")
        mock = MockStorefrontProvider()
        return await mock.get_order(order_id)


# ---------------------------------------------------------------------------
# Stripe Provider (Placeholder — interface stable)
# ---------------------------------------------------------------------------


class StripeProvider(PaymentProvider):
    """
    Stripe payment provider.

    Requires:
        STRIPE_API_KEY: Stripe secret key (sk_...)

    Falls back to mock when not configured.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("STRIPE_API_KEY", "")

    async def create_payment_link(
        self, product_name: str, amount_cents: int, currency: str = "usd",
    ) -> dict[str, Any]:
        logger.warning("stripe_not_implemented: falling back to mock")
        mock = MockPaymentProvider()
        return await mock.create_payment_link(product_name, amount_cents, currency)

    async def check_payment(self, payment_intent_id: str) -> dict[str, Any]:
        logger.warning("stripe_not_implemented: falling back to mock")
        mock = MockPaymentProvider()
        return await mock.check_payment(payment_intent_id)

    async def process_refund(
        self, payment_intent_id: str, amount_cents: Optional[int] = None, reason: str = "",
    ) -> dict[str, Any]:
        logger.warning("stripe_not_implemented: falling back to mock")
        mock = MockPaymentProvider()
        return await mock.process_refund(payment_intent_id, amount_cents, reason)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

STOREFRONT_PROVIDERS = {
    "mock": MockStorefrontProvider,
    "shopify": ShopifyProvider,
}

PAYMENT_PROVIDERS = {
    "mock": MockPaymentProvider,
    "stripe": StripeProvider,
}


class CommerceClient:
    """
    High-level commerce client with provider selection.

    Combines storefront (products/orders) and payment (Stripe) providers
    behind a single interface.

    Usage:
        client = CommerceClient.from_env()
        products = await client.get_products()
        link = await client.create_payment_link("Widget", 2999)
    """

    def __init__(
        self,
        storefront: Optional[StorefrontProvider] = None,
        payment: Optional[PaymentProvider] = None,
    ):
        self.storefront = storefront or MockStorefrontProvider()
        self.payment = payment or MockPaymentProvider()

    @classmethod
    def from_env(cls) -> CommerceClient:
        """Create a CommerceClient from environment variables."""
        # Storefront
        sf_name = os.environ.get("STOREFRONT_PROVIDER", "mock").lower()
        sf_cls = STOREFRONT_PROVIDERS.get(sf_name, MockStorefrontProvider)
        if sf_name not in STOREFRONT_PROVIDERS:
            logger.warning(f"Unknown storefront provider '{sf_name}', using mock")

        # Payment
        pay_name = os.environ.get("PAYMENT_PROVIDER", "mock").lower()
        pay_cls = PAYMENT_PROVIDERS.get(pay_name, MockPaymentProvider)
        if pay_name not in PAYMENT_PROVIDERS:
            logger.warning(f"Unknown payment provider '{pay_name}', using mock")

        return cls(storefront=sf_cls(), payment=pay_cls())

    # ─── Storefront delegation ──────────────────────────────────────

    async def get_products(
        self, limit: int = 10, status: str = "active",
    ) -> list[dict[str, Any]]:
        return await self.storefront.get_products(limit, status)

    async def update_inventory(
        self, variant_id: str, quantity: int,
    ) -> dict[str, Any]:
        return await self.storefront.update_inventory(variant_id, quantity)

    async def get_recent_orders(self, days: int = 1) -> list[dict[str, Any]]:
        return await self.storefront.get_recent_orders(days)

    async def get_order(self, order_id: str) -> dict[str, Any]:
        return await self.storefront.get_order(order_id)

    # ─── Payment delegation ─────────────────────────────────────────

    async def create_payment_link(
        self, product_name: str, amount_cents: int, currency: str = "usd",
    ) -> dict[str, Any]:
        return await self.payment.create_payment_link(
            product_name, amount_cents, currency,
        )

    async def check_payment(self, payment_intent_id: str) -> dict[str, Any]:
        return await self.payment.check_payment(payment_intent_id)

    async def process_refund(
        self,
        payment_intent_id: str,
        amount_cents: Optional[int] = None,
        reason: str = "",
    ) -> dict[str, Any]:
        return await self.payment.process_refund(
            payment_intent_id, amount_cents, reason,
        )
