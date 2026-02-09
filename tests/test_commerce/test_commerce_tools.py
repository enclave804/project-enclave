"""
Tests for Commerce MCP tools.

Covers:
- shopify_get_products (catalog + low stock detection)
- shopify_update_inventory (sandboxed)
- shopify_get_recent_orders (VIP detection)
- stripe_create_payment_link
- stripe_check_payment
- stripe_process_refund (sandboxed)
"""

from __future__ import annotations

import asyncio
import json
import os
from unittest.mock import patch

import pytest

from core.integrations.commerce_client import (
    CommerceClient,
    MockPaymentProvider,
    MockStorefrontProvider,
)


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def _mock_client():
    """Create a CommerceClient with mock providers."""
    return CommerceClient(
        storefront=MockStorefrontProvider(),
        payment=MockPaymentProvider(),
    )


# ═══════════════════════════════════════════════════════════════════════
# 1. Shopify Product Tools
# ═══════════════════════════════════════════════════════════════════════


class TestShopifyGetProducts:
    """Tests for the shopify_get_products MCP tool."""

    def test_returns_valid_json(self):
        from core.mcp.tools.commerce_tools import shopify_get_products
        result = _run(shopify_get_products(_client=_mock_client()))
        data = json.loads(result)
        assert "products" in data
        assert "product_count" in data

    def test_includes_low_stock_alerts(self):
        from core.mcp.tools.commerce_tools import shopify_get_products
        result = _run(shopify_get_products(_client=_mock_client()))
        data = json.loads(result)
        assert "low_stock_alerts" in data
        assert "low_stock_count" in data
        # Mock data has at least one low-stock variant (TEE-M at 3 units)
        assert data["low_stock_count"] >= 1

    def test_product_summaries_have_fields(self):
        from core.mcp.tools.commerce_tools import shopify_get_products
        result = _run(shopify_get_products(_client=_mock_client()))
        data = json.loads(result)
        for p in data["products"]:
            assert "product_id" in p
            assert "title" in p
            assert "total_inventory" in p
            assert "price_range" in p

    def test_respects_limit(self):
        from core.mcp.tools.commerce_tools import shopify_get_products
        result = _run(shopify_get_products(limit=1, _client=_mock_client()))
        data = json.loads(result)
        assert data["product_count"] == 1

    def test_handles_error_gracefully(self):
        from core.mcp.tools.commerce_tools import shopify_get_products
        from unittest.mock import AsyncMock, MagicMock

        bad_client = MagicMock()
        bad_client.get_products = AsyncMock(side_effect=Exception("Connection failed"))
        result = _run(shopify_get_products(_client=bad_client))
        data = json.loads(result)
        assert "error" in data


# ═══════════════════════════════════════════════════════════════════════
# 2. Shopify Inventory Tools
# ═══════════════════════════════════════════════════════════════════════


class TestShopifyUpdateInventory:
    """Tests for the shopify_update_inventory MCP tool (sandboxed)."""

    def test_is_sandboxed(self):
        from core.mcp.tools.commerce_tools import shopify_update_inventory
        from core.safety.sandbox import is_sandboxed
        assert is_sandboxed(shopify_update_inventory)

    def test_intercepted_in_dev(self):
        from core.mcp.tools.commerce_tools import shopify_update_inventory
        with patch.dict(os.environ, {"ENCLAVE_ENV": "development"}):
            result = _run(shopify_update_inventory(
                variant_id="var_001a",
                quantity=100,
                _client=_mock_client(),
            ))
        assert result["sandboxed"] is True
        assert result["tool_name"] == "shopify_update_inventory"

    def test_passes_through_in_production(self):
        from core.mcp.tools.commerce_tools import shopify_update_inventory
        with patch.dict(os.environ, {"ENCLAVE_ENV": "production"}):
            result = _run(shopify_update_inventory(
                variant_id="var_001a",
                quantity=100,
                _client=_mock_client(),
            ))
        assert result["success"] is True
        assert result["new_quantity"] == 100


# ═══════════════════════════════════════════════════════════════════════
# 3. Shopify Orders Tools
# ═══════════════════════════════════════════════════════════════════════


class TestShopifyGetRecentOrders:
    """Tests for the shopify_get_recent_orders MCP tool."""

    def test_returns_valid_json(self):
        from core.mcp.tools.commerce_tools import shopify_get_recent_orders
        result = _run(shopify_get_recent_orders(_client=_mock_client()))
        data = json.loads(result)
        assert "orders" in data
        assert "summary" in data

    def test_detects_vip_orders(self):
        from core.mcp.tools.commerce_tools import shopify_get_recent_orders
        result = _run(shopify_get_recent_orders(_client=_mock_client()))
        data = json.loads(result)
        assert data["summary"]["vip_order_count"] >= 1

    def test_calculates_revenue(self):
        from core.mcp.tools.commerce_tools import shopify_get_recent_orders
        result = _run(shopify_get_recent_orders(_client=_mock_client()))
        data = json.loads(result)
        assert data["summary"]["total_revenue"] > 0

    def test_computes_avg_order_value(self):
        from core.mcp.tools.commerce_tools import shopify_get_recent_orders
        result = _run(shopify_get_recent_orders(_client=_mock_client()))
        data = json.loads(result)
        assert data["summary"]["avg_order_value"] > 0

    def test_order_has_vip_flag(self):
        from core.mcp.tools.commerce_tools import shopify_get_recent_orders
        result = _run(shopify_get_recent_orders(_client=_mock_client()))
        data = json.loads(result)
        vip_found = any(o["is_vip"] for o in data["orders"])
        assert vip_found, "Should have at least one VIP-flagged order"


# ═══════════════════════════════════════════════════════════════════════
# 4. Stripe Payment Tools
# ═══════════════════════════════════════════════════════════════════════


class TestStripeCreatePaymentLink:
    """Tests for the stripe_create_payment_link MCP tool."""

    def test_creates_link(self):
        from core.mcp.tools.commerce_tools import stripe_create_payment_link
        result = _run(stripe_create_payment_link(
            product_name="Widget",
            amount_cents=2999,
            _client=_mock_client(),
        ))
        data = json.loads(result)
        assert "url" in data
        assert data["amount_cents"] == 2999

    def test_custom_currency(self):
        from core.mcp.tools.commerce_tools import stripe_create_payment_link
        result = _run(stripe_create_payment_link(
            product_name="EU Widget",
            amount_cents=4999,
            currency="eur",
            _client=_mock_client(),
        ))
        data = json.loads(result)
        assert data["currency"] == "eur"


class TestStripeCheckPayment:
    """Tests for the stripe_check_payment MCP tool."""

    def test_check_unknown_payment(self):
        from core.mcp.tools.commerce_tools import stripe_check_payment
        result = _run(stripe_check_payment(
            payment_intent_id="pi_test123",
            _client=_mock_client(),
        ))
        data = json.loads(result)
        assert "status" in data

    def test_handles_error(self):
        from core.mcp.tools.commerce_tools import stripe_check_payment
        from unittest.mock import AsyncMock, MagicMock

        bad_client = MagicMock()
        bad_client.check_payment = AsyncMock(side_effect=Exception("API error"))
        result = _run(stripe_check_payment(
            payment_intent_id="pi_fail",
            _client=bad_client,
        ))
        data = json.loads(result)
        assert "error" in data


class TestStripeProcessRefund:
    """Tests for the stripe_process_refund MCP tool (sandboxed)."""

    def test_is_sandboxed(self):
        from core.mcp.tools.commerce_tools import stripe_process_refund
        from core.safety.sandbox import is_sandboxed
        assert is_sandboxed(stripe_process_refund)

    def test_intercepted_in_dev(self):
        from core.mcp.tools.commerce_tools import stripe_process_refund
        with patch.dict(os.environ, {"ENCLAVE_ENV": "development"}):
            result = _run(stripe_process_refund(
                payment_intent_id="pi_test",
                _client=_mock_client(),
            ))
        assert result["sandboxed"] is True
        assert result["tool_name"] == "stripe_process_refund"

    def test_passes_through_in_production(self):
        from core.mcp.tools.commerce_tools import stripe_process_refund
        with patch.dict(os.environ, {"ENCLAVE_ENV": "production"}):
            result = _run(stripe_process_refund(
                payment_intent_id="pi_test",
                reason="Customer request",
                _client=_mock_client(),
            ))
        assert result["status"] == "succeeded"
