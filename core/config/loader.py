"""
Configuration loader for Project Enclave verticals.

Loads a vertical's config.yaml, validates it against the Pydantic schema,
and provides access to the configuration throughout the application.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import yaml
from pydantic import ValidationError

from core.config.schema import VerticalConfig

# Module-level cache: vertical_id -> VerticalConfig
_loaded_configs: dict[str, VerticalConfig] = {}


def find_verticals_dir() -> Path:
    """Locate the verticals/ directory relative to the project root."""
    # Walk up from this file to find the project root
    current = Path(__file__).resolve()
    for parent in current.parents:
        candidate = parent / "verticals"
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        "Could not find 'verticals/' directory. "
        "Ensure you're running from the project root."
    )


def load_vertical_config(
    vertical_id: str,
    config_path: Optional[str | Path] = None,
) -> VerticalConfig:
    """
    Load and validate a vertical's configuration.

    Args:
        vertical_id: The vertical identifier (e.g. 'enclave_guard').
        config_path: Optional explicit path to config.yaml.
                     If not provided, looks in verticals/{vertical_id}/config.yaml.

    Returns:
        Validated VerticalConfig instance.

    Raises:
        FileNotFoundError: If the config file doesn't exist.
        ValueError: If the config is invalid.
    """
    if vertical_id in _loaded_configs:
        return _loaded_configs[vertical_id]

    if config_path is None:
        verticals_dir = find_verticals_dir()
        config_path = verticals_dir / vertical_id / "config.yaml"

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config not found: {config_path}\n"
            f"Create verticals/{vertical_id}/config.yaml to define this vertical."
        )

    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    if raw is None:
        raise ValueError(f"Config file is empty: {config_path}")

    # Inject vertical_id if not present in the file
    if "vertical_id" not in raw:
        raw["vertical_id"] = vertical_id

    try:
        config = VerticalConfig(**raw)
    except ValidationError as e:
        raise ValueError(
            f"Invalid config for vertical '{vertical_id}':\n{e}"
        ) from e

    _loaded_configs[vertical_id] = config
    return config


def get_vertical_config(vertical_id: str) -> VerticalConfig:
    """
    Retrieve a previously loaded config. Raises if not loaded yet.
    """
    if vertical_id not in _loaded_configs:
        raise RuntimeError(
            f"Vertical '{vertical_id}' not loaded. "
            f"Call load_vertical_config('{vertical_id}') first."
        )
    return _loaded_configs[vertical_id]


def list_available_verticals() -> list[str]:
    """List all verticals that have a config.yaml file."""
    try:
        verticals_dir = find_verticals_dir()
    except FileNotFoundError:
        return []

    verticals = []
    for entry in verticals_dir.iterdir():
        if entry.is_dir() and (entry / "config.yaml").exists():
            verticals.append(entry.name)
    return sorted(verticals)


def resolve_env_vars(config: VerticalConfig) -> dict[str, str]:
    """
    Collect all environment variable references from the config and
    check that they are set. Returns a dict of var_name -> value.

    Raises EnvironmentError if any required vars are missing.
    """
    required_vars: set[str] = set()

    # Apollo API key
    required_vars.add(config.apollo.api_key_env)

    # Enrichment source API keys
    for source in config.enrichment.sources:
        if source.api_key_env:
            required_vars.add(source.api_key_env)

    # Always need the Anthropic key
    required_vars.add("ANTHROPIC_API_KEY")

    # Supabase
    required_vars.add("SUPABASE_URL")
    required_vars.add("SUPABASE_SERVICE_KEY")

    resolved: dict[str, str] = {}
    missing: list[str] = []

    for var in sorted(required_vars):
        value = os.environ.get(var)
        if value:
            resolved[var] = value
        else:
            missing.append(var)

    if missing:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing)}\n"
            f"Set them in your .env file or shell environment."
        )

    return resolved


def clear_cache() -> None:
    """Clear the config cache. Useful for testing."""
    _loaded_configs.clear()
