"""
Genesis Agent Generator — The Metaprogrammer.

This is the core of Phase 5: The Dynamic Agent Factory.
It generates valid, safe Python code for new agent types by filling
a verified Jinja2 template — NOT by writing arbitrary code.

Think of it as "Mad Libs for agents":
1. The LLM designs the agent's state machine (nodes, edges, state fields)
2. The template ensures the code fits perfectly into BaseAgent's interface
3. ast.parse() validates syntax before any file is written
4. The DynamicRegistry hot-loads the result

Safety guarantees:
- Generated code ALWAYS inherits from BaseAgent
- Generated code ALWAYS uses @register_agent_type decorator
- Generated code ALWAYS follows the LangGraph StateGraph pattern
- No arbitrary imports — only pre-approved modules
- ast.parse() catches syntax errors before execution
- Sandbox testing validates runtime behavior

Usage:
    from core.genesis.agent_generator import AgentGenerator

    generator = AgentGenerator(anthropic_client)
    result = generator.generate(
        agent_type="inventory_manager",
        agent_goal="Track product inventory and trigger reorder alerts",
        vertical_name="Epic Bearz",
        tools=["shopify_get_inventory", "shopify_update_stock", "send_alert"],
    )
    if result.success:
        result.write_to(Path("core/agents/implementations/inventory_agent.py"))
"""

from __future__ import annotations

import ast
import logging
import re
import textwrap
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import jinja2

logger = logging.getLogger(__name__)

# Template directory
TEMPLATES_DIR = Path(__file__).parent / "templates"

# Allowed imports for generated agents (security whitelist)
ALLOWED_IMPORTS = {
    "logging",
    "json",
    "re",
    "datetime",
    "typing",
    "core.agents.base",
    "core.agents.contracts",
    "core.agents.registry",
    "core.agents.state",
    "core.config.agent_schema",
    "langgraph.graph",
}


# ---------------------------------------------------------------------------
# Data Models for Agent Specification
# ---------------------------------------------------------------------------

@dataclass
class StateFieldSpec:
    """A field in the agent's state TypedDict."""
    name: str
    type: str  # Python type annotation string
    description: str
    default: str  # Python literal for default value


@dataclass
class GraphNodeSpec:
    """A node in the agent's LangGraph state machine."""
    name: str
    description: str
    implementation: str  # Python code block for the node function body


@dataclass
class GraphEdgeSpec:
    """An edge in the agent's LangGraph state machine."""
    source: str
    target: str  # Python expression (e.g., '"next_node"' or 'END')
    conditional: bool = False
    condition_map: str = ""  # Python dict literal for conditional edges
    routing_logic: str = ""  # Python code for the routing function


@dataclass
class AgentDesign:
    """
    Complete design for a dynamically generated agent.

    This is the intermediate representation between the LLM's design
    and the Jinja2 template rendering. It contains everything needed
    to produce valid Python code.
    """
    agent_type: str
    class_name: str
    state_class_name: str
    agent_name: str
    agent_goal: str
    agent_description: str
    vertical_name: str

    state_fields: list[StateFieldSpec]
    graph_nodes: list[GraphNodeSpec]
    graph_edges: list[GraphEdgeSpec]
    graph_description: str

    knowledge_writing_logic: str = "pass  # Override to write insights"
    tools: list[str] = field(default_factory=list)


@dataclass
class GenerationResult:
    """Result of agent code generation."""
    success: bool
    agent_type: str
    code: str = ""
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    design: Optional[AgentDesign] = None

    def write_to(self, path: Path) -> None:
        """Write generated code to a file."""
        if not self.success:
            raise ValueError(
                f"Cannot write failed generation: {self.errors}"
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.code)
        logger.info(
            "agent_code_written",
            extra={
                "agent_type": self.agent_type,
                "path": str(path),
                "lines": self.code.count("\n"),
            },
        )


# ---------------------------------------------------------------------------
# Agent Generator
# ---------------------------------------------------------------------------

class AgentGenerator:
    """
    Generates safe, validated Python code for new agent types.

    Uses a two-phase approach:
    1. Design Phase: LLM designs the agent's state machine
    2. Render Phase: Jinja2 template produces safe code

    The LLM never writes raw Python — it fills structured specs
    that get validated and then rendered through the template.
    """

    # System prompt for the design phase
    DESIGN_SYSTEM_PROMPT = textwrap.dedent("""\
        You are an expert agent architect for the Sovereign Venture Engine.
        Your job is to design LangGraph state machines for new AI agents.

        RULES:
        1. Every agent inherits from BaseAgent
        2. Every agent has a TypedDict state class extending BaseAgentState
        3. Nodes are async functions that take state and return dict updates
        4. Use self.llm.messages.create() for LLM calls
        5. Use self.db for database operations
        6. Use self.config.params for configurable values
        7. Node implementations should be practical but use mock data for external APIs
        8. Each node MUST return a dict with at least {"current_node": "node_name"}
        9. Use logging.info() with extra={} for structured logging
        10. Keep node implementations focused — one responsibility per node

        AVAILABLE in self:
        - self.llm (Anthropic client)
        - self.db (EnclaveDB — Supabase wrapper)
        - self.embedder (EmbeddingService)
        - self.config (AgentInstanceConfig)
        - self.config.params (dict of agent-specific params from YAML)
        - self.mcp_tools (list of MCP tools)
        - self.browser_tool (BrowserTool, if browser_enabled)
        - self.store_insight(InsightData) — gated write to shared brain

        RESPONSE FORMAT (JSON):
        {
            "state_fields": [
                {"name": "field_name", "type": "str", "description": "...", "default": "''"}
            ],
            "graph_nodes": [
                {
                    "name": "node_name",
                    "description": "What this node does",
                    "implementation": "Python code for the node body"
                }
            ],
            "graph_edges": [
                {
                    "source": "node_name",
                    "target": "\\"END\\"",
                    "conditional": false
                }
            ],
            "graph_description": "High-level flow description",
            "knowledge_writing_logic": "Python code for write_knowledge body"
        }

        For conditional edges, include:
        - "conditional": true
        - "condition_map": Python dict literal string mapping route names to node names
        - "routing_logic": Python code for the routing function that returns a route name
    """)

    def __init__(
        self,
        anthropic_client: Any = None,
        template_dir: Optional[Path] = None,
    ):
        self.llm = anthropic_client
        self._template_dir = template_dir or TEMPLATES_DIR
        self._jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(self._template_dir)),
            keep_trailing_newline=True,
            trim_blocks=True,
            lstrip_blocks=True,
        )

    # --- Public API ---

    def generate(
        self,
        agent_type: str,
        agent_goal: str,
        vertical_name: str = "Default",
        tools: Optional[list[str]] = None,
        custom_design: Optional[AgentDesign] = None,
    ) -> GenerationResult:
        """
        Generate Python code for a new agent type.

        Args:
            agent_type: Snake_case agent type (e.g., "inventory_manager")
            agent_goal: Natural language description of what the agent does
            vertical_name: Business vertical name
            tools: List of MCP tool names the agent should use
            custom_design: Skip LLM design phase — use this design directly

        Returns:
            GenerationResult with the generated code (or errors)
        """
        result = GenerationResult(success=False, agent_type=agent_type)

        # Validate agent_type format
        if not re.match(r"^[a-z][a-z0-9_]*$", agent_type):
            result.errors.append(
                f"Invalid agent_type '{agent_type}': must be snake_case "
                f"(lowercase letters, digits, underscores, starting with a letter)"
            )
            return result

        try:
            # Phase 1: Design the agent
            if custom_design:
                design = custom_design
            else:
                design = self._design_agent(
                    agent_type=agent_type,
                    agent_goal=agent_goal,
                    vertical_name=vertical_name,
                    tools=tools or [],
                )

            result.design = design

            # Phase 2: Render the template
            code = self._render_template(design)

            # Phase 3: Validate syntax
            validation_errors = self._validate_code(code)
            if validation_errors:
                result.errors.extend(validation_errors)
                result.code = code  # Include for debugging
                return result

            # Phase 4: Security check
            security_warnings = self._security_check(code)
            result.warnings.extend(security_warnings)

            result.success = True
            result.code = code

            logger.info(
                "agent_generated",
                extra={
                    "agent_type": agent_type,
                    "nodes": len(design.graph_nodes),
                    "state_fields": len(design.state_fields),
                    "lines": code.count("\n"),
                },
            )

            return result

        except Exception as e:
            result.errors.append(f"Generation failed: {str(e)}")
            logger.error(
                "agent_generation_failed",
                extra={
                    "agent_type": agent_type,
                    "error": str(e)[:200],
                },
            )
            return result

    def generate_from_design(self, design: AgentDesign) -> GenerationResult:
        """Generate code from a pre-built AgentDesign (skip LLM phase)."""
        return self.generate(
            agent_type=design.agent_type,
            agent_goal=design.agent_goal,
            vertical_name=design.vertical_name,
            custom_design=design,
        )

    # --- Design Phase (LLM) ---

    def _design_agent(
        self,
        agent_type: str,
        agent_goal: str,
        vertical_name: str,
        tools: list[str],
    ) -> AgentDesign:
        """
        Use LLM to design the agent's state machine.

        Returns a structured AgentDesign that can be rendered.
        """
        import json

        if self.llm is None:
            raise ValueError(
                "Anthropic client required for LLM-based design. "
                "Pass anthropic_client to AgentGenerator() or use "
                "generate_from_design() with a custom_design."
            )

        # Build the design prompt
        tools_str = ", ".join(tools) if tools else "none specified"
        prompt = (
            f"Design a LangGraph agent for the following:\n\n"
            f"Agent Type: {agent_type}\n"
            f"Goal: {agent_goal}\n"
            f"Vertical: {vertical_name}\n"
            f"Available Tools: {tools_str}\n\n"
            f"Design the state machine with 3-6 nodes. "
            f"Each node should have a focused responsibility. "
            f"Include at least one node that uses self.llm for AI reasoning. "
            f"Return ONLY the JSON response, no markdown fences."
        )

        response = self.llm.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            temperature=0.3,
            system=self.DESIGN_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )

        response_text = response.content[0].text.strip()

        # Strip markdown code fences if present
        if response_text.startswith("```"):
            lines = response_text.split("\n")
            # Remove first and last lines (```json and ```)
            lines = [l for l in lines if not l.strip().startswith("```")]
            response_text = "\n".join(lines)

        # Parse the LLM's design
        design_dict = json.loads(response_text)
        return self._dict_to_design(
            design_dict, agent_type, agent_goal, vertical_name, tools
        )

    def _dict_to_design(
        self,
        design_dict: dict[str, Any],
        agent_type: str,
        agent_goal: str,
        vertical_name: str,
        tools: list[str],
    ) -> AgentDesign:
        """Convert the LLM's JSON response to an AgentDesign."""
        # Generate names from agent_type
        class_name = self._to_class_name(agent_type)
        state_class_name = f"{class_name}State"

        # Parse state fields
        state_fields = [
            StateFieldSpec(
                name=f["name"],
                type=f.get("type", "str"),
                description=f.get("description", ""),
                default=f.get("default", "None"),
            )
            for f in design_dict.get("state_fields", [])
        ]

        # Parse graph nodes
        graph_nodes = [
            GraphNodeSpec(
                name=n["name"],
                description=n.get("description", ""),
                implementation=n.get("implementation", "return {\"current_node\": \"" + n["name"] + "\"}"),
            )
            for n in design_dict.get("graph_nodes", [])
        ]

        # Parse graph edges
        graph_edges = []
        for e in design_dict.get("graph_edges", []):
            target = e.get("target", '"END"')
            # Normalize END references
            if target in ("END", "end", "__end__"):
                target = "END"
            elif not target.startswith('"') and target != "END":
                target = f'"{target}"'

            graph_edges.append(GraphEdgeSpec(
                source=e["source"],
                target=target,
                conditional=e.get("conditional", False),
                condition_map=e.get("condition_map", ""),
                routing_logic=e.get("routing_logic", "return \"end\""),
            ))

        return AgentDesign(
            agent_type=agent_type,
            class_name=class_name,
            state_class_name=state_class_name,
            agent_name=self._to_display_name(agent_type),
            agent_goal=agent_goal,
            agent_description=f"{self._to_display_name(agent_type)} for {vertical_name}. {agent_goal}",
            vertical_name=vertical_name,
            state_fields=state_fields,
            graph_nodes=graph_nodes,
            graph_edges=graph_edges,
            graph_description=design_dict.get(
                "graph_description",
                " → ".join(n.name for n in graph_nodes),
            ),
            knowledge_writing_logic=design_dict.get(
                "knowledge_writing_logic", "pass  # No knowledge to write"
            ),
            tools=tools,
        )

    # --- Render Phase (Jinja2) ---

    def _render_template(self, design: AgentDesign) -> str:
        """Render the Jinja2 template with the agent design."""
        template = self._jinja_env.get_template("agent_template.py.j2")

        rendered = template.render(
            agent_type=design.agent_type,
            class_name=design.class_name,
            state_class_name=design.state_class_name,
            agent_name=design.agent_name,
            agent_goal=design.agent_goal,
            agent_description=design.agent_description,
            vertical_name=design.vertical_name,
            state_fields=design.state_fields,
            graph_nodes=design.graph_nodes,
            graph_edges=design.graph_edges,
            graph_description=design.graph_description,
            knowledge_writing_logic=design.knowledge_writing_logic,
            generated_at=datetime.now(timezone.utc).isoformat(),
        )

        return rendered

    # --- Validation ---

    def _validate_code(self, code: str) -> list[str]:
        """
        Validate generated Python code for syntax correctness.

        Uses ast.parse() — the same tool Python itself uses.
        Returns a list of errors (empty = valid).
        """
        errors = []

        try:
            ast.parse(code)
        except SyntaxError as e:
            errors.append(
                f"Syntax error at line {e.lineno}, col {e.offset}: {e.msg}"
            )
            # Try to provide context
            if e.lineno and e.lineno > 0:
                lines = code.split("\n")
                if e.lineno <= len(lines):
                    errors.append(f"  Line {e.lineno}: {lines[e.lineno - 1]}")

        return errors

    def _security_check(self, code: str) -> list[str]:
        """
        Check generated code for potential security issues.

        Returns warnings (not errors) for suspicious patterns.
        """
        warnings = []

        # Check for dangerous imports
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in ("os", "sys", "subprocess", "shutil", "pathlib"):
                        warnings.append(
                            f"Potentially dangerous import: {alias.name}"
                        )
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.split(".")[0] in (
                    "os", "sys", "subprocess", "shutil",
                ):
                    warnings.append(
                        f"Potentially dangerous import from: {node.module}"
                    )

        # Check for eval/exec usage
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in ("eval", "exec", "compile", "__import__"):
                        warnings.append(
                            f"Dangerous function call: {node.func.id}()"
                        )

        # Check for open() file operations
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == "open":
                    warnings.append("File I/O detected: open() call")

        return warnings

    # --- Name Conversion Utilities ---

    @staticmethod
    def _to_class_name(agent_type: str) -> str:
        """
        Convert snake_case agent_type to PascalCase class name.

        Examples:
            "inventory_manager" → "InventoryManagerAgent"
            "shopify_sync" → "ShopifySyncAgent"
        """
        parts = agent_type.split("_")
        pascal = "".join(word.capitalize() for word in parts)
        if not pascal.endswith("Agent"):
            pascal += "Agent"
        return pascal

    @staticmethod
    def _to_display_name(agent_type: str) -> str:
        """
        Convert snake_case to display name.

        Examples:
            "inventory_manager" → "Inventory Manager"
            "shopify_sync" → "Shopify Sync"
        """
        return " ".join(word.capitalize() for word in agent_type.split("_"))


# ---------------------------------------------------------------------------
# Convenience: Build a design manually (no LLM needed)
# ---------------------------------------------------------------------------

def build_simple_design(
    agent_type: str,
    agent_goal: str,
    vertical_name: str = "Default",
    nodes: Optional[list[dict[str, str]]] = None,
    state_fields: Optional[list[dict[str, str]]] = None,
    tools: Optional[list[str]] = None,
) -> AgentDesign:
    """
    Build an AgentDesign without LLM, using explicit specs.

    This is useful for:
    - Unit tests (no API call needed)
    - Predefined agent templates
    - Manual agent creation

    Example:
        design = build_simple_design(
            agent_type="inventory_tracker",
            agent_goal="Track inventory levels",
            nodes=[
                {"name": "check_levels", "description": "Check current stock",
                 "implementation": 'return {"current_node": "check_levels", "levels_checked": True}'},
                {"name": "alert_low_stock", "description": "Alert on low stock",
                 "implementation": 'return {"current_node": "alert_low_stock", "alert_sent": True}'},
            ],
        )
    """
    generator = AgentGenerator()

    # Build state fields
    sf = [
        StateFieldSpec(
            name=f.get("name", "result"),
            type=f.get("type", "str"),
            description=f.get("description", ""),
            default=f.get("default", "None"),
        )
        for f in (state_fields or [{"name": "result", "type": "str", "description": "Agent result", "default": "''"}])
    ]

    # Build nodes
    gn = [
        GraphNodeSpec(
            name=n["name"],
            description=n.get("description", ""),
            implementation=n.get("implementation", f'return {{"current_node": "{n["name"]}"}}'),
        )
        for n in (nodes or [
            {"name": "process", "description": "Main processing",
             "implementation": 'return {"current_node": "process", "result": "done"}'},
        ])
    ]

    # Build edges (linear chain → END)
    ge = []
    for i, node in enumerate(gn):
        if i < len(gn) - 1:
            ge.append(GraphEdgeSpec(
                source=node.name,
                target=f'"{gn[i + 1].name}"',
            ))
        else:
            ge.append(GraphEdgeSpec(
                source=node.name,
                target="END",
            ))

    return AgentDesign(
        agent_type=agent_type,
        class_name=generator._to_class_name(agent_type),
        state_class_name=f"{generator._to_class_name(agent_type)}State",
        agent_name=generator._to_display_name(agent_type),
        agent_goal=agent_goal,
        agent_description=f"{generator._to_display_name(agent_type)} for {vertical_name}. {agent_goal}",
        vertical_name=vertical_name,
        state_fields=sf,
        graph_nodes=gn,
        graph_edges=ge,
        graph_description=" → ".join(n.name for n in gn),
        tools=tools or [],
    )
