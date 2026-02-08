"""
Observability module for the Sovereign Venture Engine.

Provides distributed tracing via LangFuse and structured logging.
All tracing is opt-in: if LANGFUSE_PUBLIC_KEY is not set, the
tracer operates as a no-op so local dev never breaks.
"""
