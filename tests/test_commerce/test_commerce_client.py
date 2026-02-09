"""
Tests for the Commerce Client — Shopify + Stripe integration layer.

Covers:
- MockStorefrontProvider (products, inventory, orders)
- MockPaymentProvider (payment links, status, refunds)
- CommerceClient factory (env-based provider selection)
- ShopifyProvider / StripeProvider fallback behavior
"""

from __future__ import annotations

import asyncio
import os
from unittest.mock import patch

import pytest

from core.integrations.commerce_client import (
    CommerceClient,
    MockPaymentProvider,
    MockStorefrontProvider,
    ShopifyProvider,
    StripeProvider,
)


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ═══════════════════════════════════════════════════════════════════════
# 1. Mock Storefront Provider Tests
# ═══════════════════════════════════════════════════════════════════════


class TestMockStorefrontProvider:
    """Tests for the MockStorefrontProvider."""

    @pytest.fixture
    def provider(self):
        return MockStorefrontProvider()

    def test_get_products_returns_list(self, provider):
        products = _run(provider.get_products())
        assert isinstance(products, list)
        assert len(products) > 0

    def test_get_products_respects_limit(self, provider):
        products = _run(provider.get_products(limit=1))
        assert len(products) == 1

    def test_get_products_filters_by_status(self, provider):
        products = _run(provider.get_products(status="active"))
        assert all(p["status"] == "active" for p in products)

    def test_get_products_has_variants(self, provider):
        products = _run(provider.get_products())
        for p in products:
            assert "variants" in p
            assert len(p["variants"]) > 0
            for v in p["variants"]:
                assert "variant_id" in v
                assert "price" in v
                assert "inventory_quantity" in v

    def test_update_inventory_success(self, provider):
        result = _run(provider.update_inventory("var_001a", 100))
        assert result["success"] is True
        assert result["new_quantity"] == 100
        assert result["old_quantity"] == 50

    def test_update_inventory_persists(self, provider):
        _run(provider.update_inventory("var_001a", 42))
        products = _run(provider.get_products())
        for p in products:
            for v in p.get("variants", []):
                if v["variant_id"] == "var_001a":
                    assert v["inventory_quantity"] == 42

    def test_update_inventory_not_found(self, provider):
        result = _run(provider.update_inventory("var_nonexistent", 10))
        assert result["success"] is False
        assert "not found" in result["error"]

    def test_get_recent_orders_returns_list(self, provider):
        orders = _run(provider.get_recent_orders(days=1))
        assert isinstance(orders, list)
        assert len(orders) > 0

    def test_get_recent_orders_have_required_fields(self, provider):
        orders = _run(provider.get_recent_orders())
        for o in orders:
            assert "order_id" in o
            assert "customer_email" in o
            assert "total_price" in o
            assert "line_items" in o

    def test_get_order_by_id(self, provider):
        order = _run(provider.get_order("ord_1001"))
        assert order["order_id"] == "ord_1001"
        assert order["customer_email"] == "alice@techcorp.com"

    def test_get_order_not_found(self, provider):
        order = _run(provider.get_order("ord_nonexistent"))
        assert "error" in order

    def test_mock_has_vip_order(self, provider):
        """Mock data includes a high-value VIP order (>$500)."""
        orders = _run(provider.get_recent_orders())
        vip_orders = [o for o in orders if float(o["total_price"]) >= 500]
        assert len(vip_orders) >= 1, "Mock data should include at least one VIP order"

    def test_mock_has_low_stock_variant(self, provider):
        """Mock data includes a low-stock variant (<= 5 units)."""
        products = _run(provider.get_products())
        low_stock = []
        for p in products:
            for v in p.get("variants", []):
                if v["inventory_quantity"] <= 5:
                    low_stock.append(v)
        assert len(low_stock) >= 1, "Mock data should include at least one low-stock variant"


# ═══════════════════════════════════════════════════════════════════════
# 2. Mock Payment Provider Tests
# ═══════════════════════════════════════════════════════════════════════


class TestMockPaymentProvider:
    """Tests for the MockPaymentProvider."""

    @pytest.fixture
    def provider(self):
        return MockPaymentProvider()

    def test_create_payment_link(self, provider):
        result = _run(provider.create_payment_link("Widget", 2999))
        assert "url" in result
        assert result["amount_cents"] == 2999
        assert result["product_name"] == "Widget"
        assert result["status"] == "pending"
        assert result["payment_intent_id"].startswith("pi_")

    def test_create_payment_link_custom_currency(self, provider):
        result = _run(provider.create_payment_link("EU Widget", 4999, "eur"))
        assert result["currency"] == "eur"

    def test_check_payment_known(self, provider):
        link = _run(provider.create_payment_link("Test", 1000))
        pi_id = link["payment_intent_id"]
        status = _run(provider.check_payment(pi_id))
        assert status["payment_intent_id"] == pi_id
        assert status["amount_cents"] == 1000

    def test_check_payment_unknown_returns_mock(self, provider):
        status = _run(provider.check_payment("pi_unknown_12345"))
        assert status["status"] == "succeeded"
        assert status["provider"] == "mock"

    def test_process_full_refund(self, provider):
        link = _run(provider.create_payment_link("Refundable", 5000))
        pi_id = link["payment_intent_id"]
        refund = _run(provider.process_refund(pi_id))
        assert refund["status"] == "succeeded"
        assert refund["amount_cents"] == 5000
        assert refund["refund_type"] == "full"
        assert refund["refund_id"].startswith("re_")

    def test_process_partial_refund(self, provider):
        link = _run(provider.create_payment_link("Partial", 10000))
        pi_id = link["payment_intent_id"]
        refund = _run(provider.process_refund(pi_id, amount_cents=3000, reason="Damaged"))
        assert refund["amount_cents"] == 3000
        assert refund["refund_type"] == "partial"
        assert refund["reason"] == "Damaged"


# ═══════════════════════════════════════════════════════════════════════
# 3. CommerceClient Factory Tests
# ═══════════════════════════════════════════════════════════════════════


class TestCommerceClient:
    """Tests for the CommerceClient factory and delegation."""

    def test_default_creates_mock_providers(self):
        client = CommerceClient()
        assert isinstance(client.storefront, MockStorefrontProvider)
        assert isinstance(client.payment, MockPaymentProvider)

    def test_from_env_default_mock(self):
        with patch.dict(os.environ, {}, clear=True):
            client = CommerceClient.from_env()
            assert isinstance(client.storefront, MockStorefrontProvider)
            assert isinstance(client.payment, MockPaymentProvider)

    def test_from_env_shopify(self):
        with patch.dict(os.environ, {"STOREFRONT_PROVIDER": "shopify"}):
            client = CommerceClient.from_env()
            assert isinstance(client.storefront, ShopifyProvider)

    def test_from_env_stripe(self):
        with patch.dict(os.environ, {"PAYMENT_PROVIDER": "stripe"}):
            client = CommerceClient.from_env()
            assert isinstance(client.payment, StripeProvider)

    def test_from_env_unknown_falls_back(self):
        with patch.dict(os.environ, {"STOREFRONT_PROVIDER": "woocommerce"}):
            client = CommerceClient.from_env()
            assert isinstance(client.storefront, MockStorefrontProvider)

    def test_delegates_get_products(self):
        client = CommerceClient()
        products = _run(client.get_products())
        assert isinstance(products, list)
        assert len(products) > 0

    def test_delegates_create_payment_link(self):
        client = CommerceClient()
        link = _run(client.create_payment_link("Test", 999))
        assert "url" in link

    def test_delegates_process_refund(self):
        client = CommerceClient()
        link = _run(client.create_payment_link("Refundable", 2000))
        refund = _run(client.process_refund(link["payment_intent_id"]))
        assert refund["status"] == "succeeded"


# ═══════════════════════════════════════════════════════════════════════
# 4. Placeholder Provider Fallback Tests
# ═══════════════════════════════════════════════════════════════════════


class TestPlaceholderProviders:
    """Tests that Shopify/Stripe providers fall back to mock."""

    def test_shopify_provider_falls_back(self):
        provider = ShopifyProvider()
        products = _run(provider.get_products())
        assert isinstance(products, list)

    def test_stripe_provider_falls_back(self):
        provider = StripeProvider()
        link = _run(provider.create_payment_link("Test", 1000))
        assert "url" in link

    def test_stripe_provider_refund_falls_back(self):
        provider = StripeProvider()
        refund = _run(provider.process_refund("pi_test"))
        assert refund["status"] == "succeeded"
