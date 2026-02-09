"""
LLM Abstraction Layer — Multi-model routing and provider adapters.

Provides a unified interface for calling different LLM providers
(Anthropic, OpenAI, Ollama) with intent-based routing and
automatic fallback.

Modules:
- llm_config: Intent definitions, model profiles, routing tables
- router: ModelRouter — intent-based dispatch with fallback
- streaming: StreamingRouter — async iterator token streaming
- tools: ToolRouter — function calling / tool use across providers
- cache: ResponseCache — TTL-based LLM response caching
- vision: VisionClient — image analysis via multimodal models
"""
