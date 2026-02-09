"""
Genesis Tool Generator — MCP Tool Stub Generator.

Generates MCP tool files for new integrations. Since we can't implement
real API calls for every possible integration upfront, this generates
**mock tools** (stubs) that return realistic dummy data.

Each stub:
1. Has the correct function signature and docstring
2. Returns realistic mock data matching the expected schema
3. Is decorated with @mcp.tool() for FastMCP registration
4. Can be progressively replaced with real implementations

The stubs follow the same pattern as our existing tools
(core/mcp/tools/apollo_tools.py, supabase_tools.py, etc.)

Usage:
    from core.genesis.tool_generator import ToolGenerator

    generator = ToolGenerator()
    result = generator.generate_tools(
        integration_name="shopify",
        tools=[
            {"name": "get_products", "description": "List products from Shopify",
             "params": [{"name": "limit", "type": "int", "default": "25"}],
             "returns": "list of product dicts"},
            {"name": "update_stock", "description": "Update product stock level",
             "params": [{"name": "product_id", "type": "str"}, {"name": "quantity", "type": "int"}],
             "returns": "updated product dict"},
        ],
    )
    if result.success:
        result.write_to(Path("verticals/epic_bearz/tools/shopify_tools.py"))
"""

from __future__ import annotations

import ast
import logging
import textwrap
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class ToolParamSpec:
    """A parameter for a generated tool function."""
    name: str
    type: str = "str"  # Python type annotation
    description: str = ""
    default: Optional[str] = None  # Python literal for default, None = required
    required: bool = True


@dataclass
class ToolSpec:
    """Specification for a single MCP tool to generate."""
    name: str
    description: str
    params: list[ToolParamSpec] = field(default_factory=list)
    returns: str = "dict"  # Description of return value
    mock_return: str = "{}"  # Python literal for mock return value
    is_dangerous: bool = False  # If True, gets @sandboxed_tool decorator


@dataclass
class ToolGenerationResult:
    """Result of tool generation."""
    success: bool
    integration_name: str
    code: str = ""
    tool_count: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def write_to(self, path: Path) -> None:
        """Write generated tools to a file."""
        if not self.success:
            raise ValueError(f"Cannot write failed generation: {self.errors}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.code)
        logger.info(
            "tools_code_written",
            extra={
                "integration": self.integration_name,
                "path": str(path),
                "tools": self.tool_count,
            },
        )


# ---------------------------------------------------------------------------
# Mock Data Templates (realistic per integration type)
# ---------------------------------------------------------------------------

MOCK_DATA_TEMPLATES: dict[str, dict[str, str]] = {
    "shopify": {
        "product": textwrap.dedent('''\
            {
                "id": "gid://shopify/Product/7654321",
                "title": "Epic Bearz Active Tee",
                "handle": "epic-bearz-active-tee",
                "status": "active",
                "vendor": "Epic Bearz",
                "product_type": "T-Shirt",
                "variants": [
                    {"id": "v1", "title": "S / Black", "price": "39.99", "inventory_quantity": 42},
                    {"id": "v2", "title": "M / Black", "price": "39.99", "inventory_quantity": 67},
                ],
                "images": [{"src": "https://cdn.shopify.com/mock/tee-black.jpg"}],
                "created_at": "2024-01-15T10:30:00Z",
            }'''),
        "order": textwrap.dedent('''\
            {
                "id": "gid://shopify/Order/1234567",
                "order_number": 1042,
                "total_price": "79.98",
                "currency": "USD",
                "customer": {"first_name": "Jane", "last_name": "Doe", "email": "jane@example.com"},
                "line_items": [
                    {"title": "Epic Bearz Active Tee", "quantity": 2, "price": "39.99"},
                ],
                "fulfillment_status": "fulfilled",
                "created_at": "2024-02-20T14:22:00Z",
            }'''),
        "inventory": textwrap.dedent('''\
            {
                "product_id": "gid://shopify/Product/7654321",
                "title": "Epic Bearz Active Tee",
                "total_inventory": 109,
                "variants": [
                    {"variant_id": "v1", "title": "S / Black", "available": 42},
                    {"variant_id": "v2", "title": "M / Black", "available": 67},
                ],
                "low_stock_threshold": 10,
                "reorder_needed": False,
            }'''),
    },
    "stripe": {
        "payment": textwrap.dedent('''\
            {
                "id": "pi_3MtwBwLkdIwHu7ix28a3tqPa",
                "amount": 3999,
                "currency": "usd",
                "status": "succeeded",
                "customer": "cus_NeZwdNtLkQT28a",
                "description": "Payment for Epic Bearz Active Tee",
                "created": 1706115738,
            }'''),
    },
    "twilio": {
        "call": textwrap.dedent('''\
            {
                "sid": "CA1234567890abcdef1234567890abcdef",
                "from": "+15551234567",
                "to": "+15559876543",
                "status": "completed",
                "duration": "127",
                "direction": "outbound-api",
                "date_created": "2024-02-15T10:30:00Z",
            }'''),
    },
    "default": {
        "item": textwrap.dedent('''\
            {
                "id": "item_001",
                "name": "Sample Item",
                "status": "active",
                "created_at": "2024-01-01T00:00:00Z",
                "metadata": {},
            }'''),
    },
}


# ---------------------------------------------------------------------------
# Tool Generator
# ---------------------------------------------------------------------------

class ToolGenerator:
    """
    Generates MCP tool stub files with realistic mock data.

    Each tool follows the FastMCP pattern used throughout the codebase.
    Tools are generated as standalone Python files that can be:
    1. Used immediately with mock data for testing
    2. Progressively replaced with real API implementations
    """

    def __init__(self, anthropic_client: Any = None):
        self.llm = anthropic_client

    def generate_tools(
        self,
        integration_name: str,
        tools: list[dict[str, Any]],
        vertical_name: str = "Default",
    ) -> ToolGenerationResult:
        """
        Generate a Python file with MCP tool stubs.

        Args:
            integration_name: Name of the integration (e.g., "shopify", "stripe")
            tools: List of tool specifications (dicts with name, description, params)
            vertical_name: Business vertical name

        Returns:
            ToolGenerationResult with generated code
        """
        result = ToolGenerationResult(
            success=False,
            integration_name=integration_name,
        )

        try:
            # Parse tool specs
            tool_specs = self._parse_tool_specs(tools, integration_name)

            # Generate code
            code = self._render_tools(integration_name, tool_specs, vertical_name)

            # Validate syntax
            try:
                ast.parse(code)
            except SyntaxError as e:
                result.errors.append(
                    f"Generated code has syntax error at line {e.lineno}: {e.msg}"
                )
                result.code = code
                return result

            result.success = True
            result.code = code
            result.tool_count = len(tool_specs)

            logger.info(
                "tools_generated",
                extra={
                    "integration": integration_name,
                    "tool_count": len(tool_specs),
                },
            )

            return result

        except Exception as e:
            result.errors.append(f"Tool generation failed: {str(e)}")
            return result

    def generate_tools_from_specs(
        self,
        integration_name: str,
        tool_specs: list[ToolSpec],
        vertical_name: str = "Default",
    ) -> ToolGenerationResult:
        """Generate from pre-built ToolSpec objects."""
        result = ToolGenerationResult(
            success=False,
            integration_name=integration_name,
        )

        try:
            code = self._render_tools(integration_name, tool_specs, vertical_name)

            try:
                ast.parse(code)
            except SyntaxError as e:
                result.errors.append(f"Syntax error at line {e.lineno}: {e.msg}")
                result.code = code
                return result

            result.success = True
            result.code = code
            result.tool_count = len(tool_specs)
            return result

        except Exception as e:
            result.errors.append(f"Tool generation failed: {str(e)}")
            return result

    # --- Internal ---

    def _parse_tool_specs(
        self, tools: list[dict[str, Any]], integration_name: str
    ) -> list[ToolSpec]:
        """Parse raw tool dicts into ToolSpec objects."""
        specs = []
        for tool in tools:
            params = [
                ToolParamSpec(
                    name=p.get("name", "param"),
                    type=p.get("type", "str"),
                    description=p.get("description", ""),
                    default=p.get("default"),
                    required=p.get("required", p.get("default") is None),
                )
                for p in tool.get("params", [])
            ]

            # Get mock return value based on integration type
            mock_data = MOCK_DATA_TEMPLATES.get(
                integration_name, MOCK_DATA_TEMPLATES["default"]
            )
            mock_key = self._guess_mock_key(tool.get("name", ""), mock_data)
            mock_return = mock_data.get(mock_key, mock_data.get("item", "{}"))

            specs.append(ToolSpec(
                name=tool["name"],
                description=tool.get("description", f"Tool: {tool['name']}"),
                params=params,
                returns=tool.get("returns", "dict"),
                mock_return=mock_return,
                is_dangerous=tool.get("is_dangerous", False),
            ))

        return specs

    def _render_tools(
        self,
        integration_name: str,
        tool_specs: list[ToolSpec],
        vertical_name: str,
    ) -> str:
        """Render the tools into a Python file."""
        now = datetime.now(timezone.utc).isoformat()
        class_name = "".join(w.capitalize() for w in integration_name.split("_"))

        lines = [
            f'"""',
            f'{class_name} Tools — MCP tool stubs for {vertical_name}.',
            f'',
            f'Generated by: Genesis Engine Tool Factory',
            f'Generated at: {now}',
            f'',
            f'These are MOCK implementations that return realistic dummy data.',
            f'Replace with real API calls when integrating with the actual service.',
            f'"""',
            f'',
            f'from __future__ import annotations',
            f'',
            f'import logging',
            f'from typing import Any, Optional',
            f'',
            f'logger = logging.getLogger(__name__)',
            f'',
            f'',
            f'# ---------------------------------------------------------------------------',
            f'# {class_name} Tools (Mock Implementations)',
            f'# ---------------------------------------------------------------------------',
            f'',
        ]

        for spec in tool_specs:
            lines.extend(self._render_tool_function(spec, integration_name))
            lines.append("")
            lines.append("")

        # Add a registration function for MCP server
        tool_names = [s.name for s in tool_specs]
        lines.extend([
            f'def register_{integration_name}_tools(mcp_server: Any) -> None:',
            f'    """Register all {integration_name} tools with the MCP server."""',
        ])
        for name in tool_names:
            lines.append(f'    mcp_server.tool()({name})')
        lines.append("")
        lines.append("")

        # Add a get_tools helper
        lines.extend([
            f'def get_{integration_name}_tool_names() -> list[str]:',
            f'    """Return all tool names for this integration."""',
            f'    return {tool_names!r}',
            f'',
        ])

        return "\n".join(lines)

    def _render_tool_function(
        self, spec: ToolSpec, integration_name: str
    ) -> list[str]:
        """Render a single tool function."""
        lines = []

        # Build parameter list
        param_parts = []
        for p in spec.params:
            type_hint = p.type
            if p.default is not None:
                param_parts.append(f"    {p.name}: {type_hint} = {p.default},")
            elif not p.required:
                param_parts.append(f"    {p.name}: Optional[{type_hint}] = None,")
            else:
                param_parts.append(f"    {p.name}: {type_hint},")

        # Function signature
        if param_parts:
            lines.append(f"def {spec.name}(")
            lines.extend(param_parts)
            lines.append(f") -> dict[str, Any]:")
        else:
            lines.append(f"def {spec.name}() -> dict[str, Any]:")

        # Docstring
        lines.append(f'    """')
        lines.append(f'    {spec.description}')
        lines.append(f'')
        if spec.params:
            lines.append(f'    Args:')
            for p in spec.params:
                desc = p.description or p.name
                lines.append(f'        {p.name}: {desc}')
            lines.append(f'')
        lines.append(f'    Returns:')
        lines.append(f'        {spec.returns}')
        lines.append(f'')
        lines.append(f'    Note: This is a MOCK implementation.')
        lines.append(f'          Replace with real {integration_name} API call.')
        lines.append(f'    """')

        # Mock implementation
        lines.append(f'    logger.info(')
        lines.append(f'        "{integration_name}_{spec.name}_called",')
        if spec.params:
            param_names = [p.name for p in spec.params[:3]]
            extra_parts = ", ".join(f'"{p}": {p}' for p in param_names)
            lines.append(f'        extra={{{extra_parts}}},')
        lines.append(f'    )')

        # Return mock data
        lines.append(f'    # TODO: Replace with real {integration_name} API call')
        lines.append(f'    return {spec.mock_return}')

        return lines

    def _guess_mock_key(self, tool_name: str, mock_data: dict) -> str:
        """Guess the best mock data key based on tool name."""
        tool_lower = tool_name.lower()
        for key in mock_data:
            if key in tool_lower:
                return key
        # Return first key as fallback
        return next(iter(mock_data), "item")
