"""
FastMCP server for the Sovereign Venture Engine.

Exposes Apollo, Supabase/RAG, Email, Calendar, System, Commerce, and Voice
tools to agents via the Model Context Protocol (MCP). Each tool is a thin
wrapper around the existing integration clients.

Entry points:
    python -m core.mcp           # stdio transport (for agent integration)
    create_mcp_server()          # programmatic use (for testing)

Architecture:
    FastMCP server (29 tools)
    +-- search_leads()                -> ApolloClient.search_people()
    +-- enrich_company()              -> ApolloClient.enrich_company()
    +-- search_knowledge()            -> EnclaveDB.search_knowledge() via embedder
    +-- save_insight()                -> EnclaveDB.store_insight()
    +-- query_companies()             -> EnclaveDB.list_companies()
    +-- send_email()                  -> EmailEngine.send_email() [@sandboxed_tool]
    +-- check_calendar_availability() -> CalendarClient.check_availability()
    +-- get_available_slots()         -> CalendarClient.get_available_slots()
    +-- book_meeting_slot()           -> CalendarClient.book_meeting() [@sandboxed_tool]
    +-- get_booking_link()            -> CalendarClient.get_booking_link()
    +-- get_recent_logs()             -> LogBuffer (in-memory structured logs)
    +-- query_run_history()           -> EnclaveDB.get_agent_runs()
    +-- get_system_health()           -> Aggregate health check
    +-- get_task_queue_status()       -> EnclaveDB task queue inspection
    +-- get_agent_error_rates()       -> Per-agent failure analysis
    +-- get_knowledge_stats()         -> Shared brain utilization
    +-- get_cache_performance()       -> ResponseCache stats
    +-- shopify_get_products()        -> CommerceClient.get_products()
    +-- shopify_update_inventory()    -> CommerceClient.update_inventory() [@sandboxed_tool]
    +-- shopify_get_recent_orders()   -> CommerceClient.get_recent_orders()
    +-- stripe_create_payment_link()  -> CommerceClient.create_payment_link()
    +-- stripe_check_payment()        -> CommerceClient.check_payment()
    +-- stripe_process_refund()       -> CommerceClient.process_refund() [@sandboxed_tool]
    +-- send_sms()                    -> Twilio SMS API                  [@sandboxed_tool]
    +-- make_call()                   -> Synthesize audio → Twilio call  [@sandboxed_tool]
    +-- get_call_logs()               -> Twilio call history
    +-- get_sms_logs()                -> Twilio SMS history
    +-- buy_phone_number()            -> Twilio phone number purchase    [@sandboxed_tool]
    +-- transcribe_audio()            -> Whisper API (via Transcriber)
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Lazy import to avoid import errors when fastmcp isn't installed
_server_instance = None


def create_mcp_server(
    name: str = "Enclave Tools",
    vertical_id: str = "enclave_guard",
) -> "FastMCP":
    """
    Create and configure the FastMCP server with all tools registered.

    Uses Tool.from_function() to wrap existing tool functions,
    which accepts raw async/sync functions and converts them to
    FastMCP Tool objects.

    Args:
        name: Server name for MCP protocol handshake.
        vertical_id: Default vertical for Supabase queries.

    Returns:
        Configured FastMCP server instance.
    """
    from fastmcp import FastMCP
    from fastmcp.tools import Tool

    mcp = FastMCP(name)

    # ─── Apollo Tools ─────────────────────────────────────────────
    from core.mcp.tools.apollo_tools import search_leads, enrich_company

    mcp.add_tool(Tool.from_function(search_leads))
    mcp.add_tool(Tool.from_function(enrich_company))

    # ─── Supabase / RAG Tools ────────────────────────────────────
    from core.mcp.tools.supabase_tools import (
        search_knowledge,
        save_insight,
        query_companies,
    )

    mcp.add_tool(Tool.from_function(search_knowledge))
    mcp.add_tool(Tool.from_function(save_insight))
    mcp.add_tool(Tool.from_function(query_companies))

    # ─── Email Tools (sandboxed) ─────────────────────────────────
    from core.mcp.tools.email_tools import send_email

    mcp.add_tool(Tool.from_function(send_email))

    # ─── Calendar Tools (book_meeting sandboxed) ──────────────
    from core.mcp.tools.calendar_tools import (
        check_calendar_availability,
        get_available_slots,
        book_meeting_slot,
        get_booking_link,
    )

    mcp.add_tool(Tool.from_function(check_calendar_availability))
    mcp.add_tool(Tool.from_function(get_available_slots))
    mcp.add_tool(Tool.from_function(book_meeting_slot))
    mcp.add_tool(Tool.from_function(get_booking_link))

    # ─── System Monitoring Tools (Overseer) ─────────────────────
    from core.mcp.tools.system_tools import (
        get_recent_logs,
        query_run_history,
        get_system_health,
        get_task_queue_status,
        get_agent_error_rates,
        get_knowledge_stats,
        get_cache_performance,
    )

    mcp.add_tool(Tool.from_function(get_recent_logs))
    mcp.add_tool(Tool.from_function(query_run_history))
    mcp.add_tool(Tool.from_function(get_system_health))
    mcp.add_tool(Tool.from_function(get_task_queue_status))
    mcp.add_tool(Tool.from_function(get_agent_error_rates))
    mcp.add_tool(Tool.from_function(get_knowledge_stats))
    mcp.add_tool(Tool.from_function(get_cache_performance))

    # ─── Commerce Tools (inventory/refund sandboxed) ───────────
    from core.mcp.tools.commerce_tools import (
        shopify_get_products,
        shopify_update_inventory,
        shopify_get_recent_orders,
        stripe_create_payment_link,
        stripe_check_payment,
        stripe_process_refund,
    )

    mcp.add_tool(Tool.from_function(shopify_get_products))
    mcp.add_tool(Tool.from_function(shopify_update_inventory))
    mcp.add_tool(Tool.from_function(shopify_get_recent_orders))
    mcp.add_tool(Tool.from_function(stripe_create_payment_link))
    mcp.add_tool(Tool.from_function(stripe_check_payment))
    mcp.add_tool(Tool.from_function(stripe_process_refund))

    # ─── Voice & SMS Tools (send/buy sandboxed) ──────────────────
    from core.mcp.tools.voice_tools import (
        send_sms,
        make_call,
        get_call_logs,
        get_sms_logs,
        buy_phone_number,
        transcribe_audio,
    )

    mcp.add_tool(Tool.from_function(send_sms))
    mcp.add_tool(Tool.from_function(make_call))
    mcp.add_tool(Tool.from_function(get_call_logs))
    mcp.add_tool(Tool.from_function(get_sms_logs))
    mcp.add_tool(Tool.from_function(buy_phone_number))
    mcp.add_tool(Tool.from_function(transcribe_audio))

    logger.info(
        "mcp_server_created",
        extra={
            "server_name": name,
            "vertical_id": vertical_id,
            "tool_count": 29,
        },
    )

    return mcp


def get_server() -> "FastMCP":
    """Get or create the singleton MCP server instance."""
    global _server_instance
    if _server_instance is None:
        _server_instance = create_mcp_server()
    return _server_instance
