"""
Commerce MCP tools for the Sovereign Venture Engine.

Exposes Shopify storefront and Stripe payment operations as MCP tools
for the CommerceAgent. Tools auto-detect mock vs production mode based
on environment variables (SHOPIFY_API_KEY, STRIPE_API_KEY).

Architecture:
    shopify_get_products()       -> CommerceClient.get_products()
    shopify_update_inventory()   -> CommerceClient.update_inventory()  [@sandboxed_tool]
    shopify_get_recent_orders()  -> CommerceClient.get_recent_orders()
    stripe_create_payment_link() -> CommerceClient.create_payment_link()
    stripe_check_payment()       -> CommerceClient.check_payment()
    stripe_process_refund()      -> CommerceClient.process_refund()    [@sandboxed_tool]

Safety:
    - Inventory updates are sandboxed in non-production environments
    - Refunds are sandboxed in non-production environments
    - Read-only tools (get_products, get_orders, check_payment) are always live
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from core.safety.sandbox import sandboxed_tool

logger = logging.getLogger(__name__)


# ─── Lazy client construction ────────────────────────────────────────────

def _get_commerce_client(_client: Any = None) -> Any:
    """Lazily construct a CommerceClient from env if not injected."""
    if _client is not None:
        return _client
    from core.integrations.commerce_client import CommerceClient
    return CommerceClient.from_env()


# ─── Shopify / Storefront Tools ──────────────────────────────────────────


async def shopify_get_products(
    limit: int = 10,
    status: str = "active",
    *,
    _client: Any = None,
) -> str:
    """
    Retrieve products from the storefront catalog.

    Returns product names, prices, inventory levels, and variants.
    Use this to monitor stock levels and find products to promote.

    Args:
        limit: Maximum number of products to return (default: 10).
        status: Product status filter — "active", "draft", "archived".
        _client: Injected CommerceClient for testing.

    Returns:
        JSON string with product catalog.
    """
    client = _get_commerce_client(_client)

    logger.info(
        "mcp_tool_called",
        extra={"tool_name": "shopify_get_products", "limit": limit},
    )

    try:
        products = await client.get_products(limit=limit, status=status)

        # Summarize for the agent
        summaries = []
        low_stock_alerts = []
        for p in products:
            total_inventory = sum(
                v.get("inventory_quantity", 0) for v in p.get("variants", [])
            )
            summary = {
                "product_id": p.get("product_id"),
                "title": p.get("title"),
                "vendor": p.get("vendor"),
                "product_type": p.get("product_type"),
                "total_inventory": total_inventory,
                "variant_count": len(p.get("variants", [])),
                "price_range": _get_price_range(p.get("variants", [])),
                "tags": p.get("tags", []),
            }
            summaries.append(summary)

            # Flag low stock
            for v in p.get("variants", []):
                qty = v.get("inventory_quantity", 0)
                if qty <= 5:
                    low_stock_alerts.append({
                        "product": p.get("title"),
                        "variant": v.get("title"),
                        "variant_id": v.get("variant_id"),
                        "quantity": qty,
                        "sku": v.get("sku"),
                    })

        return json.dumps({
            "products": summaries,
            "product_count": len(summaries),
            "low_stock_alerts": low_stock_alerts,
            "low_stock_count": len(low_stock_alerts),
        }, indent=2)

    except Exception as e:
        logger.error("shopify_get_products_failed", extra={"error": str(e)})
        return json.dumps({"error": str(e), "products": []})


def _get_price_range(variants: list[dict]) -> str:
    """Extract price range from variants."""
    if not variants:
        return "N/A"
    prices = [float(v.get("price", 0)) for v in variants]
    lo, hi = min(prices), max(prices)
    if lo == hi:
        return f"${lo:.2f}"
    return f"${lo:.2f} - ${hi:.2f}"


@sandboxed_tool("shopify_update_inventory")
async def shopify_update_inventory(
    variant_id: str,
    quantity: int,
    *,
    _client: Any = None,
) -> dict[str, Any]:
    """
    Update inventory quantity for a product variant.

    SAFETY: This tool is wrapped with @sandboxed_tool("shopify_update_inventory").
    In development/staging/test environments, the call is intercepted
    and logged instead of modifying real inventory.

    Args:
        variant_id: The storefront variant ID to update.
        quantity: New inventory quantity (absolute, not delta).
        _client: Injected CommerceClient for testing.

    Returns:
        Dict with update result including old and new quantities.
    """
    client = _get_commerce_client(_client)

    logger.info(
        "mcp_tool_called",
        extra={
            "tool_name": "shopify_update_inventory",
            "variant_id": variant_id,
            "quantity": quantity,
        },
    )

    result = await client.update_inventory(variant_id, quantity)
    return result


async def shopify_get_recent_orders(
    days: int = 1,
    *,
    _client: Any = None,
) -> str:
    """
    Retrieve recent orders from the storefront.

    Returns order details including customer info, line items, and
    fulfillment status. The CommerceAgent uses this to detect
    VIP customers and monitor order flow.

    Args:
        days: Look-back window in days (default: 1).
        _client: Injected CommerceClient for testing.

    Returns:
        JSON string with recent orders and summary stats.
    """
    client = _get_commerce_client(_client)

    logger.info(
        "mcp_tool_called",
        extra={"tool_name": "shopify_get_recent_orders", "days": days},
    )

    try:
        orders = await client.get_recent_orders(days=days)

        # Compute summary stats
        total_revenue = sum(float(o.get("total_price", 0)) for o in orders)
        vip_threshold = 500.0  # Orders above this are VIP
        vip_orders = [
            o for o in orders if float(o.get("total_price", 0)) >= vip_threshold
        ]

        # Format orders for the agent
        formatted = []
        for o in orders:
            order_total = float(o.get("total_price", 0))
            formatted.append({
                "order_id": o.get("order_id"),
                "order_number": o.get("order_number"),
                "customer_email": o.get("customer_email"),
                "customer_name": o.get("customer_name"),
                "total_price": o.get("total_price"),
                "currency": o.get("currency", "USD"),
                "financial_status": o.get("financial_status"),
                "fulfillment_status": o.get("fulfillment_status"),
                "item_count": sum(
                    item.get("quantity", 0) for item in o.get("line_items", [])
                ),
                "is_vip": order_total >= vip_threshold,
                "created_at": o.get("created_at"),
            })

        return json.dumps({
            "orders": formatted,
            "summary": {
                "order_count": len(orders),
                "total_revenue": round(total_revenue, 2),
                "currency": "USD",
                "vip_order_count": len(vip_orders),
                "avg_order_value": round(
                    total_revenue / len(orders), 2
                ) if orders else 0,
                "days_window": days,
            },
        }, indent=2)

    except Exception as e:
        logger.error("shopify_get_recent_orders_failed", extra={"error": str(e)})
        return json.dumps({"error": str(e), "orders": []})


# ─── Stripe / Payment Tools ─────────────────────────────────────────────


async def stripe_create_payment_link(
    product_name: str,
    amount_cents: int,
    currency: str = "usd",
    *,
    _client: Any = None,
) -> str:
    """
    Create a Stripe payment link for a product or service.

    Generates a hosted checkout URL that can be sent to customers.
    The CommerceAgent uses this to create on-the-fly payment links
    for custom orders or quotes.

    Args:
        product_name: Display name for the product on the checkout page.
        amount_cents: Price in cents (e.g., 2999 = $29.99).
        currency: Three-letter currency code (default: "usd").
        _client: Injected CommerceClient for testing.

    Returns:
        JSON string with payment link URL and payment intent ID.
    """
    client = _get_commerce_client(_client)

    logger.info(
        "mcp_tool_called",
        extra={
            "tool_name": "stripe_create_payment_link",
            "product": product_name,
            "amount_cents": amount_cents,
        },
    )

    try:
        result = await client.create_payment_link(
            product_name=product_name,
            amount_cents=amount_cents,
            currency=currency,
        )
        return json.dumps(result, indent=2)

    except Exception as e:
        logger.error("stripe_create_payment_link_failed", extra={"error": str(e)})
        return json.dumps({"error": str(e)})


async def stripe_check_payment(
    payment_intent_id: str,
    *,
    _client: Any = None,
) -> str:
    """
    Check the status of a Stripe payment intent.

    Returns payment status (pending, succeeded, failed), amount, and
    customer details. Used by the CommerceAgent to verify order payments.

    Args:
        payment_intent_id: The Stripe payment intent ID (pi_...).
        _client: Injected CommerceClient for testing.

    Returns:
        JSON string with payment status and details.
    """
    client = _get_commerce_client(_client)

    logger.info(
        "mcp_tool_called",
        extra={
            "tool_name": "stripe_check_payment",
            "payment_intent_id": payment_intent_id[:16],
        },
    )

    try:
        result = await client.check_payment(payment_intent_id)
        return json.dumps(result, indent=2)

    except Exception as e:
        logger.error("stripe_check_payment_failed", extra={"error": str(e)})
        return json.dumps({"error": str(e)})


@sandboxed_tool("stripe_process_refund")
async def stripe_process_refund(
    payment_intent_id: str,
    amount_cents: Optional[int] = None,
    reason: str = "",
    *,
    _client: Any = None,
) -> dict[str, Any]:
    """
    Process a refund for a Stripe payment.

    SAFETY: This tool is wrapped with @sandboxed_tool("stripe_process_refund").
    In development/staging/test environments, the call is intercepted
    and logged instead of processing a real refund.

    Args:
        payment_intent_id: The Stripe payment intent ID to refund.
        amount_cents: Amount to refund in cents. None = full refund.
        reason: Reason for the refund (shown to customer).
        _client: Injected CommerceClient for testing.

    Returns:
        Dict with refund result including refund ID and status.
    """
    client = _get_commerce_client(_client)

    logger.info(
        "mcp_tool_called",
        extra={
            "tool_name": "stripe_process_refund",
            "payment_intent_id": payment_intent_id[:16],
            "amount_cents": amount_cents,
        },
    )

    result = await client.process_refund(
        payment_intent_id=payment_intent_id,
        amount_cents=amount_cents,
        reason=reason,
    )
    return result
