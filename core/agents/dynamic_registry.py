"""
Dynamic Agent Registry — Hot-loading for generated agents.

Extends the static registry (core/agents/registry.py) with runtime
module loading capabilities. When the Agent Factory generates a new
agent class, the DynamicRegistry can load it into memory WITHOUT
restarting the server.

Safety guarantees:
1. Only loads from whitelisted directories (verticals/ and core/agents/)
2. Validates the loaded module contains a BaseAgent subclass
3. Verifies @register_agent_type decorator was applied
4. Catches import errors and reports them cleanly
5. Supports unloading (for testing and hot-swap)

Usage:
    from core.agents.dynamic_registry import DynamicRegistry

    registry = DynamicRegistry()

    # Load a generated agent
    result = registry.load_agent_module(
        "verticals/epic_bearz/agents/implementations/inventory_agent.py"
    )
    if result.success:
        print(f"Loaded: {result.agent_type}")

    # Now the agent type is available in AGENT_IMPLEMENTATIONS
    from core.agents.registry import AGENT_IMPLEMENTATIONS
    assert "inventory_manager" in AGENT_IMPLEMENTATIONS
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Optional, Type

from core.agents.base import BaseAgent
from core.agents.registry import AGENT_IMPLEMENTATIONS, get_registered_types

logger = logging.getLogger(__name__)

# Directories we allow loading from (security whitelist)
ALLOWED_LOAD_DIRS = {
    "verticals",
    "core/agents/implementations",
    "core/agents",
}


# ---------------------------------------------------------------------------
# Result Type
# ---------------------------------------------------------------------------

@dataclass
class LoadResult:
    """Result of loading a dynamic agent module."""
    success: bool
    file_path: str
    agent_type: str = ""
    class_name: str = ""
    module_name: str = ""
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Dynamic Registry
# ---------------------------------------------------------------------------

class DynamicRegistry:
    """
    Manages hot-loading of dynamically generated agent modules.

    Complements the static AgentRegistry by adding runtime loading.
    Loaded agents are registered into the global AGENT_IMPLEMENTATIONS
    dict, making them available to AgentRegistry.discover_agents().
    """

    def __init__(self, allowed_dirs: Optional[set[str]] = None):
        self._allowed_dirs = allowed_dirs or ALLOWED_LOAD_DIRS
        self._loaded_modules: dict[str, ModuleType] = {}
        self._loaded_types: dict[str, str] = {}  # module_name → agent_type

    def load_agent_module(
        self,
        file_path: str | Path,
        *,
        force_reload: bool = False,
    ) -> LoadResult:
        """
        Load a Python module containing a BaseAgent subclass.

        The module must:
        1. Be in an allowed directory
        2. Parse as valid Python
        3. Contain a class decorated with @register_agent_type
        4. That class must inherit from BaseAgent

        Args:
            file_path: Path to the .py file
            force_reload: If True, reload even if already loaded

        Returns:
            LoadResult with success status and details
        """
        file_path = Path(file_path)
        result = LoadResult(success=False, file_path=str(file_path))

        # --- Validation ---

        # Check file exists
        if not file_path.exists():
            result.errors.append(f"File not found: {file_path}")
            return result

        if not file_path.suffix == ".py":
            result.errors.append(f"Not a Python file: {file_path}")
            return result

        # Security: check file is in allowed directory
        if not self._is_allowed_path(file_path):
            result.errors.append(
                f"File not in allowed directory. Allowed: {self._allowed_dirs}"
            )
            return result

        # Read and validate syntax
        try:
            source_code = file_path.read_text()
        except OSError as e:
            result.errors.append(f"Cannot read file: {e}")
            return result

        try:
            ast.parse(source_code)
        except SyntaxError as e:
            result.errors.append(
                f"Syntax error at line {e.lineno}: {e.msg}"
            )
            return result

        # Pre-check: does it contain register_agent_type?
        if "register_agent_type" not in source_code:
            result.errors.append(
                "Module does not use @register_agent_type decorator. "
                "Every dynamic agent must be registered."
            )
            return result

        # Pre-check: does it reference BaseAgent?
        if "BaseAgent" not in source_code:
            result.errors.append(
                "Module does not reference BaseAgent. "
                "All agents must inherit from BaseAgent."
            )
            return result

        # --- Loading ---

        # Generate a unique module name
        module_name = self._path_to_module_name(file_path)

        # Check if already loaded
        if module_name in self._loaded_modules and not force_reload:
            existing_type = self._loaded_types.get(module_name, "unknown")
            result.success = True
            result.agent_type = existing_type
            result.module_name = module_name
            result.warnings.append(
                f"Module already loaded (agent_type={existing_type}). "
                f"Use force_reload=True to reload."
            )
            return result

        # Record registered types BEFORE loading (to detect what was added)
        types_before = set(AGENT_IMPLEMENTATIONS.keys())

        try:
            # Use importlib to load the module
            spec = importlib.util.spec_from_file_location(
                module_name, str(file_path)
            )
            if spec is None or spec.loader is None:
                result.errors.append(f"Cannot create import spec for: {file_path}")
                return result

            module = importlib.util.module_from_spec(spec)

            # Add to sys.modules before exec (allows relative imports)
            sys.modules[module_name] = module

            # Execute the module (this triggers @register_agent_type)
            spec.loader.exec_module(module)

            # Check what new types were registered
            types_after = set(AGENT_IMPLEMENTATIONS.keys())
            new_types = types_after - types_before

            if not new_types:
                result.errors.append(
                    "Module loaded but no new agent type was registered. "
                    "Ensure @register_agent_type decorator is applied to a class."
                )
                # Clean up
                sys.modules.pop(module_name, None)
                return result

            # Verify the registered class is a proper BaseAgent subclass
            for agent_type in new_types:
                cls = AGENT_IMPLEMENTATIONS[agent_type]
                if not issubclass(cls, BaseAgent):
                    result.errors.append(
                        f"Registered class {cls.__name__} for type "
                        f"'{agent_type}' does not inherit from BaseAgent"
                    )
                    # Remove invalid registration
                    AGENT_IMPLEMENTATIONS.pop(agent_type, None)
                    sys.modules.pop(module_name, None)
                    return result

            # Success!
            agent_type = next(iter(new_types))  # Primary type
            self._loaded_modules[module_name] = module
            self._loaded_types[module_name] = agent_type

            result.success = True
            result.agent_type = agent_type
            result.class_name = AGENT_IMPLEMENTATIONS[agent_type].__name__
            result.module_name = module_name

            if len(new_types) > 1:
                result.warnings.append(
                    f"Module registered {len(new_types)} types: {new_types}. "
                    f"Primary: {agent_type}"
                )

            logger.info(
                "dynamic_agent_loaded",
                extra={
                    "agent_type": agent_type,
                    "class_name": result.class_name,
                    "file_path": str(file_path),
                    "module_name": module_name,
                },
            )

            return result

        except Exception as e:
            result.errors.append(f"Import error: {str(e)}")
            # Clean up on failure
            sys.modules.pop(module_name, None)
            logger.error(
                "dynamic_agent_load_failed",
                extra={
                    "file_path": str(file_path),
                    "error": str(e)[:200],
                },
            )
            return result

    def unload_agent_type(self, agent_type: str) -> bool:
        """
        Unload a dynamically loaded agent type.

        Removes it from AGENT_IMPLEMENTATIONS and sys.modules.
        Useful for testing and hot-swapping.

        Returns True if the type was found and removed.
        """
        # Find the module that registered this type
        module_name = None
        for mod_name, a_type in self._loaded_types.items():
            if a_type == agent_type:
                module_name = mod_name
                break

        if module_name is None:
            # Check if it exists in AGENT_IMPLEMENTATIONS at all
            if agent_type in AGENT_IMPLEMENTATIONS:
                logger.warning(
                    f"Agent type '{agent_type}' exists but was not dynamically loaded. "
                    f"Cannot unload static registrations."
                )
            return False

        # Remove from registries
        AGENT_IMPLEMENTATIONS.pop(agent_type, None)
        sys.modules.pop(module_name, None)
        self._loaded_modules.pop(module_name, None)
        self._loaded_types.pop(module_name, None)

        logger.info(
            "dynamic_agent_unloaded",
            extra={"agent_type": agent_type, "module_name": module_name},
        )
        return True

    def list_loaded(self) -> list[dict[str, str]]:
        """Return info about all dynamically loaded agents."""
        return [
            {
                "module_name": mod_name,
                "agent_type": self._loaded_types.get(mod_name, "unknown"),
                "class_name": AGENT_IMPLEMENTATIONS.get(
                    self._loaded_types.get(mod_name, ""), type(None)
                ).__name__,
            }
            for mod_name in self._loaded_modules
        ]

    def is_loaded(self, agent_type: str) -> bool:
        """Check if an agent type was dynamically loaded."""
        return agent_type in self._loaded_types.values()

    # --- Private ---

    def _is_allowed_path(self, file_path: Path) -> bool:
        """Check if the file is in an allowed directory."""
        resolved = file_path.resolve()
        str_path = str(resolved)

        for allowed_dir in self._allowed_dirs:
            if f"/{allowed_dir}/" in str_path or str_path.endswith(f"/{allowed_dir}"):
                return True

        return False

    @staticmethod
    def _path_to_module_name(file_path: Path) -> str:
        """
        Convert a file path to a Python module name.

        Example:
            verticals/epic_bearz/agents/inventory_agent.py
            → dynamic_agents.epic_bearz.inventory_agent
        """
        stem = file_path.stem  # filename without .py
        # Use a unique namespace to avoid collisions
        parts = list(file_path.parts)

        # Try to extract meaningful path components
        meaningful = []
        for part in reversed(parts[:-1]):  # skip the filename
            if part in (".", "..", ""):
                continue
            meaningful.insert(0, part)
            if part == "verticals" or part == "implementations":
                break

        # Limit depth
        meaningful = meaningful[-3:]
        meaningful.append(stem)

        return "dynamic_agents." + ".".join(meaningful)


# ---------------------------------------------------------------------------
# Convenience: Load all custom agents for a vertical
# ---------------------------------------------------------------------------

def load_vertical_agents(
    vertical_id: str,
    verticals_dir: Optional[Path] = None,
) -> list[LoadResult]:
    """
    Discover and load all custom agent implementations for a vertical.

    Looks in: verticals/{vertical_id}/agents/implementations/*.py

    Returns list of LoadResults.
    """
    if verticals_dir is None:
        verticals_dir = Path(__file__).parent.parent.parent / "verticals"

    impl_dir = verticals_dir / vertical_id / "agents" / "implementations"
    if not impl_dir.exists():
        return []

    registry = DynamicRegistry()
    results = []

    for py_file in sorted(impl_dir.glob("*.py")):
        if py_file.name.startswith("_"):
            continue  # Skip __init__.py etc.

        result = registry.load_agent_module(py_file)
        results.append(result)

        if result.success:
            logger.info(
                f"Loaded custom agent for {vertical_id}: "
                f"{result.agent_type} ({result.class_name})"
            )
        else:
            logger.warning(
                f"Failed to load {py_file.name} for {vertical_id}: "
                f"{result.errors}"
            )

    return results
