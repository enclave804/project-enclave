"""
MCP tool modules for the Sovereign Venture Engine.

Each module wraps an existing integration client (Apollo, Supabase, Email)
as MCP-compatible tool functions with simple string/int parameters.

Security scanning tools (Phase 18 — Domain Expert Agents):
    - ssl_scan_tools: SSL certificate and TLS protocol analysis
    - http_scanner_tools: HTTP security headers, CORS, cookies, CSP
    - dns_tools: DNS enumeration, subdomain discovery, DNSSEC, SPF/DMARC
    - compliance_framework_tools: Framework requirements, control mapping, scoring

3D printing tools (Phase 19 — PrintBiz Domain Expert Agents):
    - mesh_tools: Mesh analysis, repair, manifold check, volume computation
    - printer_tools: Printer farm management, job scheduling, progress monitoring
    - shipping_tools: Carrier rates, label creation, shipment tracking
    - measurement_tools: Dimensional accuracy, surface quality, geometry comparison

Phase 21 — Business Operations tools:
    - billing_tools: Invoice creation, payment tracking, reminders, line-item calculation
    - survey_tools: NPS survey dispatch, response collection, score calculation, sentiment
    - data_quality_tools: Email/phone validation, duplicate detection, data freshness
"""
