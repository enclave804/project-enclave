"""
Genesis Engine — The business launcher for the Sovereign Venture Engine.

Transforms a business idea into a fully operational agent fleet through
four stages:

1. Interview (Discovery) — Gather business context through adaptive Q&A
2. Blueprint (Strategy)  — AI generates a strategic business plan
3. Build (Instantiation) — Auto-generate validated YAML configs
4. Launch (Deployment)   — Deploy agent fleet in shadow mode

Usage:
    from core.genesis.blueprint import BusinessBlueprint
    from core.genesis.config_generator import ConfigGenerator

    blueprint = BusinessBlueprint(...)
    generator = ConfigGenerator()
    generator.generate_vertical(blueprint, output_dir="verticals")
"""

from core.genesis.blueprint import BusinessBlueprint, BusinessContext
from core.genesis.interview import InterviewEngine, InterviewPhase

__all__ = [
    "BusinessBlueprint",
    "BusinessContext",
    "InterviewEngine",
    "InterviewPhase",
]
