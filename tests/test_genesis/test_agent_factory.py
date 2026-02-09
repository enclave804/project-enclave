"""
Tests for Phase 5: The Dynamic Agent Factory.

Covers:
1. AgentGenerator — template rendering, syntax validation, security checks
2. ToolGenerator — MCP tool stub generation
3. DynamicRegistry — hot-loading of generated agents
4. End-to-end: generate → validate → load → verify

These tests do NOT require an LLM API call — they use build_simple_design()
to create agent designs programmatically.
"""

from __future__ import annotations

import ast
import sys
import textwrap
from pathlib import Path

import pytest

from core.genesis.agent_generator import (
    AgentDesign,
    AgentGenerator,
    GenerationResult,
    GraphEdgeSpec,
    GraphNodeSpec,
    StateFieldSpec,
    build_simple_design,
)
from core.genesis.tool_generator import (
    ToolGenerator,
    ToolGenerationResult,
    ToolParamSpec,
    ToolSpec,
)
from core.agents.dynamic_registry import (
    DynamicRegistry,
    LoadResult,
    load_vertical_agents,
)
from core.agents.registry import AGENT_IMPLEMENTATIONS


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def generator():
    """AgentGenerator without LLM (for template-based tests)."""
    return AgentGenerator()


@pytest.fixture
def tool_generator():
    """ToolGenerator without LLM."""
    return ToolGenerator()


@pytest.fixture
def simple_design():
    """A minimal valid agent design for testing."""
    return build_simple_design(
        agent_type="test_factory_agent",
        agent_goal="A test agent for the factory tests",
        vertical_name="TestVertical",
        nodes=[
            {
                "name": "analyze",
                "description": "Analyze the input data",
                "implementation": textwrap.dedent('''\
                    task = state.get("task_input", {})
                    logger.info("analyze_started", extra={"agent_id": self.agent_id})
                    return {"current_node": "analyze", "analysis_done": True}'''),
            },
            {
                "name": "report",
                "description": "Generate a report",
                "implementation": textwrap.dedent('''\
                    logger.info("report_generated", extra={"agent_id": self.agent_id})
                    return {"current_node": "report", "report_text": "Analysis complete."}'''),
            },
        ],
        state_fields=[
            {"name": "analysis_done", "type": "bool", "description": "Whether analysis is complete", "default": "False"},
            {"name": "report_text", "type": "str", "description": "Generated report text", "default": "''"},
        ],
    )


@pytest.fixture
def complex_design():
    """A more complex agent design with conditional edges."""
    gen = AgentGenerator()
    return AgentDesign(
        agent_type="inventory_tracker",
        class_name="InventoryTrackerAgent",
        state_class_name="InventoryTrackerAgentState",
        agent_name="Inventory Tracker",
        agent_goal="Track product inventory and trigger reorder alerts",
        agent_description="Tracks inventory levels for e-commerce.",
        vertical_name="Epic Bearz",
        state_fields=[
            StateFieldSpec("products_checked", "int", "Number of products checked", "0"),
            StateFieldSpec("low_stock_items", "list[dict[str, Any]]", "Items with low stock", "[]"),
            StateFieldSpec("alert_sent", "bool", "Whether low stock alert was sent", "False"),
            StateFieldSpec("reorder_triggered", "bool", "Whether reorder was triggered", "False"),
        ],
        graph_nodes=[
            GraphNodeSpec(
                "check_inventory",
                "Check current stock levels for all products",
                textwrap.dedent('''\
                    logger.info("checking_inventory", extra={"agent_id": self.agent_id})
                    threshold = self.config.params.get("low_stock_threshold", 10)
                    return {
                        "current_node": "check_inventory",
                        "products_checked": 42,
                        "low_stock_items": [{"name": "Tee", "stock": 3}],
                    }'''),
            ),
            GraphNodeSpec(
                "evaluate",
                "Decide if alerts or reorders are needed",
                textwrap.dedent('''\
                    low_stock = state.get("low_stock_items", [])
                    return {
                        "current_node": "evaluate",
                        "alert_sent": len(low_stock) > 0,
                    }'''),
            ),
            GraphNodeSpec(
                "send_alert",
                "Send low stock alerts",
                textwrap.dedent('''\
                    logger.info("alert_sent", extra={"items": len(state.get("low_stock_items", []))})
                    return {"current_node": "send_alert", "alert_sent": True}'''),
            ),
            GraphNodeSpec(
                "finish",
                "Complete the inventory check",
                textwrap.dedent('''\
                    return {"current_node": "finish", "knowledge_written": True}'''),
            ),
        ],
        graph_edges=[
            GraphEdgeSpec("check_inventory", '"evaluate"'),
            GraphEdgeSpec(
                "evaluate",
                "",
                conditional=True,
                condition_map='{"needs_alert": "send_alert", "ok": "finish"}',
                routing_logic=textwrap.dedent('''\
                    if state.get("low_stock_items"):
                        return "needs_alert"
                    return "ok"'''),
            ),
            GraphEdgeSpec("send_alert", '"finish"'),
            GraphEdgeSpec("finish", "END"),
        ],
        graph_description="check_inventory → evaluate → [send_alert | finish]",
        tools=["shopify_get_inventory", "send_alert"],
    )


@pytest.fixture
def dynamic_registry(tmp_path):
    """DynamicRegistry that allows loading from tmp_path."""
    return DynamicRegistry(allowed_dirs={"verticals", "core/agents", str(tmp_path)})


# ===========================================================================
# Test: AgentGenerator — Name Conversion
# ===========================================================================

class TestNameConversion:
    """Test snake_case → PascalCase conversion."""

    def test_simple_name(self, generator):
        assert generator._to_class_name("outreach") == "OutreachAgent"

    def test_multi_word_name(self, generator):
        assert generator._to_class_name("inventory_manager") == "InventoryManagerAgent"

    def test_already_has_agent_suffix(self, generator):
        # _to_class_name detects trailing "Agent" and doesn't double-add it
        assert generator._to_class_name("seo_agent") == "SeoAgent"

    def test_single_word(self, generator):
        assert generator._to_class_name("janitor") == "JanitorAgent"

    def test_display_name(self, generator):
        assert generator._to_display_name("inventory_manager") == "Inventory Manager"

    def test_display_name_single(self, generator):
        assert generator._to_display_name("outreach") == "Outreach"


# ===========================================================================
# Test: AgentGenerator — Template Rendering
# ===========================================================================

class TestTemplateRendering:
    """Test Jinja2 template rendering produces valid Python."""

    def test_simple_design_renders(self, generator, simple_design):
        """Simple design renders without errors."""
        result = generator.generate_from_design(simple_design)
        assert result.success, f"Errors: {result.errors}"
        assert result.code
        assert len(result.code) > 100

    def test_complex_design_renders(self, generator, complex_design):
        """Complex design with conditional edges renders."""
        result = generator.generate_from_design(complex_design)
        assert result.success, f"Errors: {result.errors}"
        assert "InventoryTrackerAgent" in result.code
        assert "InventoryTrackerAgentState" in result.code

    def test_rendered_code_has_correct_class(self, generator, simple_design):
        """Generated code has the correct class name."""
        result = generator.generate_from_design(simple_design)
        # test_factory_agent → TestFactoryAgent (no double suffix)
        assert "TestFactoryAgent" in result.code

    def test_rendered_code_has_register_decorator(self, generator, simple_design):
        """Generated code has @register_agent_type decorator."""
        result = generator.generate_from_design(simple_design)
        assert '@register_agent_type("test_factory_agent")' in result.code

    def test_rendered_code_inherits_base_agent(self, generator, simple_design):
        """Generated code inherits from BaseAgent."""
        result = generator.generate_from_design(simple_design)
        assert "BaseAgent" in result.code
        assert "(BaseAgent)" in result.code

    def test_rendered_code_has_build_graph(self, generator, simple_design):
        """Generated code implements build_graph()."""
        result = generator.generate_from_design(simple_design)
        assert "def build_graph(self)" in result.code

    def test_rendered_code_has_get_tools(self, generator, simple_design):
        """Generated code implements get_tools()."""
        result = generator.generate_from_design(simple_design)
        assert "def get_tools(self)" in result.code

    def test_rendered_code_has_get_state_class(self, generator, simple_design):
        """Generated code implements get_state_class()."""
        result = generator.generate_from_design(simple_design)
        assert "def get_state_class(self)" in result.code

    def test_rendered_code_has_state_class(self, generator, simple_design):
        """Generated code has the state TypedDict."""
        result = generator.generate_from_design(simple_design)
        assert "class TestFactoryAgentState(BaseAgentState" in result.code

    def test_rendered_code_has_nodes(self, generator, simple_design):
        """Generated code has node methods."""
        result = generator.generate_from_design(simple_design)
        assert "_node_analyze" in result.code
        assert "_node_report" in result.code

    def test_conditional_edges_render(self, generator, complex_design):
        """Conditional edges generate routing functions."""
        result = generator.generate_from_design(complex_design)
        assert "add_conditional_edges" in result.code
        assert "_route_evaluate" in result.code

    def test_state_fields_render(self, generator, simple_design):
        """State fields appear in the state class."""
        result = generator.generate_from_design(simple_design)
        assert "analysis_done" in result.code
        assert "report_text" in result.code


# ===========================================================================
# Test: AgentGenerator — Syntax Validation
# ===========================================================================

class TestSyntaxValidation:
    """Test that ast.parse catches syntax errors."""

    def test_valid_code_passes(self, generator, simple_design):
        """Valid design produces code that passes ast.parse."""
        result = generator.generate_from_design(simple_design)
        assert result.success
        # Double check with our own ast.parse
        ast.parse(result.code)

    def test_complex_code_passes_ast(self, generator, complex_design):
        """Complex design with conditionals passes ast.parse."""
        result = generator.generate_from_design(complex_design)
        assert result.success
        ast.parse(result.code)

    def test_invalid_agent_type_rejected(self, generator):
        """Invalid agent_type format is rejected."""
        result = generator.generate(
            agent_type="InvalidType",  # Not snake_case
            agent_goal="Test",
        )
        assert not result.success
        assert any("snake_case" in e for e in result.errors)

    def test_empty_agent_type_rejected(self, generator):
        """Empty agent_type is rejected."""
        result = generator.generate(
            agent_type="",
            agent_goal="Test",
        )
        assert not result.success

    def test_agent_type_starting_with_number_rejected(self, generator):
        """agent_type starting with a number is rejected."""
        result = generator.generate(
            agent_type="3d_printer",
            agent_goal="Test",
        )
        assert not result.success


# ===========================================================================
# Test: AgentGenerator — Security Checks
# ===========================================================================

class TestSecurityChecks:
    """Test security validation of generated code."""

    def test_clean_code_no_warnings(self, generator, simple_design):
        """Clean generated code has no security warnings."""
        result = generator.generate_from_design(simple_design)
        assert result.success
        # Template-generated code shouldn't trigger warnings
        assert len(result.warnings) == 0

    def test_detects_dangerous_imports(self, generator):
        """Security check catches os/subprocess imports."""
        code = textwrap.dedent("""\
            import os
            import subprocess
            class Foo:
                pass
        """)
        warnings = generator._security_check(code)
        assert len(warnings) >= 2
        assert any("os" in w for w in warnings)
        assert any("subprocess" in w for w in warnings)

    def test_detects_eval_exec(self, generator):
        """Security check catches eval/exec calls."""
        code = textwrap.dedent("""\
            result = eval("1+1")
        """)
        warnings = generator._security_check(code)
        assert any("eval" in w for w in warnings)

    def test_detects_open_calls(self, generator):
        """Security check catches file I/O."""
        code = textwrap.dedent("""\
            f = open("/etc/passwd", "r")
        """)
        warnings = generator._security_check(code)
        assert any("open" in w for w in warnings)


# ===========================================================================
# Test: AgentGenerator — Design Building
# ===========================================================================

class TestDesignBuilding:
    """Test the build_simple_design convenience function."""

    def test_builds_minimal_design(self):
        """Minimal design builds successfully."""
        design = build_simple_design(
            agent_type="test_minimal",
            agent_goal="Do nothing",
        )
        assert design.agent_type == "test_minimal"
        assert design.class_name == "TestMinimalAgent"
        assert design.state_class_name == "TestMinimalAgentState"
        assert len(design.graph_nodes) >= 1
        assert len(design.graph_edges) >= 1

    def test_builds_with_custom_nodes(self):
        """Design with custom nodes."""
        design = build_simple_design(
            agent_type="multi_step",
            agent_goal="Process in steps",
            nodes=[
                {"name": "step_one", "description": "First step"},
                {"name": "step_two", "description": "Second step"},
                {"name": "step_three", "description": "Third step"},
            ],
        )
        assert len(design.graph_nodes) == 3
        assert design.graph_nodes[0].name == "step_one"
        assert design.graph_nodes[2].name == "step_three"

    def test_builds_linear_edges(self):
        """Design creates linear edge chain."""
        design = build_simple_design(
            agent_type="linear_agent",
            agent_goal="Linear flow",
            nodes=[
                {"name": "a", "description": "Node A"},
                {"name": "b", "description": "Node B"},
                {"name": "c", "description": "Node C"},
            ],
        )
        # Edges: a→b, b→c, c→END
        assert len(design.graph_edges) == 3
        assert design.graph_edges[0].source == "a"
        assert design.graph_edges[0].target == '"b"'
        assert design.graph_edges[1].source == "b"
        assert design.graph_edges[1].target == '"c"'
        assert design.graph_edges[2].source == "c"
        assert design.graph_edges[2].target == "END"

    def test_builds_with_state_fields(self):
        """Design with custom state fields."""
        design = build_simple_design(
            agent_type="stateful_agent",
            agent_goal="Track state",
            state_fields=[
                {"name": "counter", "type": "int", "description": "A counter", "default": "0"},
                {"name": "items", "type": "list[str]", "description": "Item list", "default": "[]"},
            ],
        )
        assert len(design.state_fields) == 2
        assert design.state_fields[0].name == "counter"
        assert design.state_fields[1].type == "list[str]"


# ===========================================================================
# Test: AgentGenerator — End-to-End (no LLM)
# ===========================================================================

class TestEndToEndGeneration:
    """Test full generation pipeline without LLM."""

    def test_generate_and_validate(self, generator, simple_design):
        """Generate code, validate syntax, and check structure."""
        result = generator.generate_from_design(simple_design)
        assert result.success

        # Parse the AST to verify structure
        tree = ast.parse(result.code)

        # Find classes
        classes = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef)
        ]
        class_names = {c.name for c in classes}
        # test_factory_agent → TestFactoryAgent + TestFactoryAgentState
        assert "TestFactoryAgentState" in class_names
        assert "TestFactoryAgent" in class_names

    def test_generate_complex_agent(self, generator, complex_design):
        """Generate a complex agent with conditional edges."""
        result = generator.generate_from_design(complex_design)
        assert result.success

        # Verify structural elements
        assert "check_inventory" in result.code
        assert "evaluate" in result.code
        assert "send_alert" in result.code
        assert "finish" in result.code
        assert "add_conditional_edges" in result.code
        assert "low_stock_items" in result.code

    def test_generate_writes_file(self, generator, simple_design, tmp_path):
        """Generated code can be written to a file."""
        result = generator.generate_from_design(simple_design)
        assert result.success

        out_path = tmp_path / "test_agent.py"
        result.write_to(out_path)

        assert out_path.exists()
        content = out_path.read_text()
        assert "TestFactoryAgent" in content

    def test_generate_result_has_design(self, generator, simple_design):
        """GenerationResult includes the design object."""
        result = generator.generate_from_design(simple_design)
        assert result.design is not None
        assert result.design.agent_type == "test_factory_agent"

    def test_cannot_write_failed_result(self, generator, tmp_path):
        """write_to raises on failed result."""
        result = GenerationResult(
            success=False,
            agent_type="bad",
            errors=["Something went wrong"],
        )
        with pytest.raises(ValueError, match="Cannot write failed"):
            result.write_to(tmp_path / "bad.py")


# ===========================================================================
# Test: ToolGenerator
# ===========================================================================

class TestToolGenerator:
    """Test MCP tool stub generation."""

    def test_generate_simple_tool(self, tool_generator):
        """Generate a single simple tool."""
        result = tool_generator.generate_tools(
            integration_name="test_api",
            tools=[
                {
                    "name": "get_items",
                    "description": "Retrieve all items",
                    "params": [],
                    "returns": "list of item dicts",
                },
            ],
        )
        assert result.success
        assert result.tool_count == 1
        assert "get_items" in result.code
        assert "def get_items()" in result.code

    def test_generate_tool_with_params(self, tool_generator):
        """Generate a tool with parameters."""
        result = tool_generator.generate_tools(
            integration_name="test_api",
            tools=[
                {
                    "name": "search_products",
                    "description": "Search for products",
                    "params": [
                        {"name": "query", "type": "str", "description": "Search query"},
                        {"name": "limit", "type": "int", "default": "25"},
                    ],
                },
            ],
        )
        assert result.success
        assert "query: str" in result.code
        assert "limit: int = 25" in result.code

    def test_generate_multiple_tools(self, tool_generator):
        """Generate multiple tools in one file."""
        result = tool_generator.generate_tools(
            integration_name="shopify",
            tools=[
                {"name": "get_products", "description": "List products"},
                {"name": "get_orders", "description": "List orders"},
                {"name": "update_stock", "description": "Update stock level",
                 "params": [{"name": "product_id", "type": "str"}]},
            ],
        )
        assert result.success
        assert result.tool_count == 3
        assert "get_products" in result.code
        assert "get_orders" in result.code
        assert "update_stock" in result.code

    def test_generated_tools_have_valid_syntax(self, tool_generator):
        """All generated tool code passes ast.parse."""
        result = tool_generator.generate_tools(
            integration_name="complex_api",
            tools=[
                {"name": "get_data", "description": "Get data",
                 "params": [
                     {"name": "id", "type": "str"},
                     {"name": "include_metadata", "type": "bool", "default": "False"},
                 ]},
                {"name": "post_data", "description": "Post data",
                 "params": [{"name": "payload", "type": "dict"}]},
            ],
        )
        assert result.success
        ast.parse(result.code)

    def test_generated_tools_have_registration(self, tool_generator):
        """Generated file includes tool registration function."""
        result = tool_generator.generate_tools(
            integration_name="my_api",
            tools=[{"name": "ping", "description": "Health check"}],
        )
        assert result.success
        assert "register_my_api_tools" in result.code
        assert "get_my_api_tool_names" in result.code

    def test_shopify_mock_data(self, tool_generator):
        """Shopify tools get realistic mock data."""
        result = tool_generator.generate_tools(
            integration_name="shopify",
            tools=[{"name": "get_products", "description": "List products"}],
        )
        assert result.success
        assert "Epic Bearz" in result.code or "shopify" in result.code.lower()

    def test_write_tools_to_file(self, tool_generator, tmp_path):
        """Generated tools can be written to a file."""
        result = tool_generator.generate_tools(
            integration_name="test_write",
            tools=[{"name": "test_func", "description": "Test function"}],
        )
        assert result.success

        out_path = tmp_path / "test_tools.py"
        result.write_to(out_path)

        assert out_path.exists()
        content = out_path.read_text()
        assert "test_func" in content

    def test_generate_from_specs(self, tool_generator):
        """Generate from pre-built ToolSpec objects."""
        specs = [
            ToolSpec(
                name="custom_tool",
                description="A custom tool",
                params=[
                    ToolParamSpec("item_id", "str", "Item ID"),
                    ToolParamSpec("quantity", "int", "Quantity", default="1"),
                ],
                returns="Updated item dict",
                mock_return='{"id": "item_001", "quantity": 1}',
            ),
        ]
        result = tool_generator.generate_tools_from_specs(
            integration_name="custom",
            tool_specs=specs,
        )
        assert result.success
        assert "custom_tool" in result.code
        assert "item_id" in result.code


# ===========================================================================
# Test: DynamicRegistry
# ===========================================================================

def _generate_unique_agent(generator, agent_type, tmp_path, subdir="agents"):
    """Helper: generate agent code with a unique type and write to disk."""
    design = build_simple_design(
        agent_type=agent_type,
        agent_goal=f"Test agent for {agent_type}",
        vertical_name="TestVertical",
        nodes=[
            {
                "name": "process",
                "description": "Process data",
                "implementation": 'return {"current_node": "process"}',
            },
        ],
    )
    result = generator.generate_from_design(design)
    assert result.success, f"Generation failed: {result.errors}"

    agent_file = tmp_path / "verticals" / "test" / subdir / f"{agent_type}.py"
    agent_file.parent.mkdir(parents=True, exist_ok=True)
    agent_file.write_text(result.code)
    return result, agent_file


class TestDynamicRegistry:
    """Test hot-loading of generated agent modules."""

    def test_load_valid_module(self, generator, tmp_path, dynamic_registry):
        """Load a valid generated agent module."""
        agent_type = "dyn_load_valid"
        _, agent_file = _generate_unique_agent(generator, agent_type, tmp_path)

        # Ensure type not already registered
        AGENT_IMPLEMENTATIONS.pop(agent_type, None)

        load_result = dynamic_registry.load_agent_module(agent_file)
        assert load_result.success, f"Errors: {load_result.errors}"
        assert load_result.agent_type == agent_type
        assert "DynLoadValid" in load_result.class_name

        # Verify it's in the global registry
        assert agent_type in AGENT_IMPLEMENTATIONS

        # Clean up
        dynamic_registry.unload_agent_type(agent_type)

    def test_load_nonexistent_file(self, dynamic_registry):
        """Loading a nonexistent file fails gracefully."""
        result = dynamic_registry.load_agent_module("/no/such/file.py")
        assert not result.success
        assert any("not found" in e.lower() for e in result.errors)

    def test_load_non_python_file(self, dynamic_registry, tmp_path):
        """Loading a non-Python file fails."""
        txt_file = tmp_path / "verticals" / "test.txt"
        txt_file.parent.mkdir(parents=True, exist_ok=True)
        txt_file.write_text("not python")

        result = dynamic_registry.load_agent_module(txt_file)
        assert not result.success
        assert any("not a python" in e.lower() for e in result.errors)

    def test_load_syntax_error(self, dynamic_registry, tmp_path):
        """Loading a file with syntax errors fails."""
        bad_file = tmp_path / "verticals" / "bad.py"
        bad_file.parent.mkdir(parents=True, exist_ok=True)
        bad_file.write_text("def broken(:\n    pass")

        result = dynamic_registry.load_agent_module(bad_file)
        assert not result.success
        assert any("syntax" in e.lower() for e in result.errors)

    def test_load_no_register_decorator(self, dynamic_registry, tmp_path):
        """Loading a module without @register_agent_type fails."""
        no_register = tmp_path / "verticals" / "no_register.py"
        no_register.parent.mkdir(parents=True, exist_ok=True)
        no_register.write_text(textwrap.dedent("""\
            from core.agents.base import BaseAgent
            class PlainAgent(BaseAgent):
                def build_graph(self):
                    pass
                def get_tools(self):
                    return []
                def get_state_class(self):
                    return dict
        """))

        result = dynamic_registry.load_agent_module(no_register)
        assert not result.success
        assert any("register_agent_type" in e for e in result.errors)

    def test_load_no_base_agent(self, dynamic_registry, tmp_path):
        """Loading a module without BaseAgent reference fails."""
        no_base = tmp_path / "verticals" / "no_base.py"
        no_base.parent.mkdir(parents=True, exist_ok=True)
        no_base.write_text(textwrap.dedent("""\
            from core.agents.registry import register_agent_type
            @register_agent_type("rogue")
            class RogueClass:
                pass
        """))

        result = dynamic_registry.load_agent_module(no_base)
        assert not result.success
        assert any("BaseAgent" in e for e in result.errors)

    def test_unload_agent(self, generator, tmp_path, dynamic_registry):
        """Unload a dynamically loaded agent."""
        agent_type = "dyn_unload_test"
        _, agent_file = _generate_unique_agent(generator, agent_type, tmp_path)

        # Ensure clean state
        AGENT_IMPLEMENTATIONS.pop(agent_type, None)

        # Load
        load_result = dynamic_registry.load_agent_module(agent_file)
        assert load_result.success, f"Errors: {load_result.errors}"
        assert agent_type in AGENT_IMPLEMENTATIONS

        # Unload
        unloaded = dynamic_registry.unload_agent_type(agent_type)
        assert unloaded
        assert agent_type not in AGENT_IMPLEMENTATIONS

    def test_list_loaded(self, generator, tmp_path, dynamic_registry):
        """List loaded modules."""
        agent_type = "dyn_list_test"
        _, agent_file = _generate_unique_agent(generator, agent_type, tmp_path)

        # Ensure clean state
        AGENT_IMPLEMENTATIONS.pop(agent_type, None)

        dynamic_registry.load_agent_module(agent_file)

        loaded = dynamic_registry.list_loaded()
        assert len(loaded) >= 1
        assert any(l["agent_type"] == agent_type for l in loaded)

        # Clean up
        dynamic_registry.unload_agent_type(agent_type)

    def test_is_loaded(self, generator, tmp_path, dynamic_registry):
        """Check if a type is dynamically loaded."""
        agent_type = "dyn_is_loaded_test"
        _, agent_file = _generate_unique_agent(generator, agent_type, tmp_path)

        # Ensure clean state
        AGENT_IMPLEMENTATIONS.pop(agent_type, None)

        assert not dynamic_registry.is_loaded(agent_type)

        dynamic_registry.load_agent_module(agent_file)
        assert dynamic_registry.is_loaded(agent_type)

        dynamic_registry.unload_agent_type(agent_type)
        assert not dynamic_registry.is_loaded(agent_type)

    def test_skip_reload_without_force(self, generator, tmp_path, dynamic_registry):
        """Second load without force_reload returns cached result."""
        agent_type = "dyn_reload_test"
        _, agent_file = _generate_unique_agent(generator, agent_type, tmp_path)

        # Ensure clean state
        AGENT_IMPLEMENTATIONS.pop(agent_type, None)

        # First load
        load_1 = dynamic_registry.load_agent_module(agent_file)
        assert load_1.success, f"Errors: {load_1.errors}"

        # Second load (should be cached)
        load_2 = dynamic_registry.load_agent_module(agent_file)
        assert load_2.success
        assert any("already loaded" in w.lower() for w in load_2.warnings)

        # Clean up
        dynamic_registry.unload_agent_type(agent_type)


# ===========================================================================
# Test: Full Pipeline — Generate → Load → Verify
# ===========================================================================

class TestFullPipeline:
    """End-to-end integration tests."""

    def test_generate_load_verify(self, generator, tmp_path, dynamic_registry):
        """Full pipeline: generate code → write → load → verify registration."""
        agent_type = "pipeline_verify_agent"

        # Ensure clean state
        AGENT_IMPLEMENTATIONS.pop(agent_type, None)

        # Step 1: Generate
        design = build_simple_design(
            agent_type=agent_type,
            agent_goal="Pipeline verification test",
            vertical_name="TestVertical",
        )
        gen_result = generator.generate_from_design(design)
        assert gen_result.success

        # Step 2: Write
        agent_file = tmp_path / "verticals" / "test_vert" / "agents" / "implementations" / f"{agent_type}.py"
        gen_result.write_to(agent_file)
        assert agent_file.exists()

        # Step 3: Load
        load_result = dynamic_registry.load_agent_module(agent_file)
        assert load_result.success, f"Load errors: {load_result.errors}"

        # Step 4: Verify
        assert agent_type in AGENT_IMPLEMENTATIONS
        cls = AGENT_IMPLEMENTATIONS[agent_type]
        assert issubclass(cls, __import__("core.agents.base", fromlist=["BaseAgent"]).BaseAgent)

        # Clean up
        dynamic_registry.unload_agent_type(agent_type)

    def test_generate_tools_and_agent(self, generator, tool_generator, simple_design, tmp_path):
        """Generate both tools and agent code for a vertical."""
        # Generate tools
        tool_result = tool_generator.generate_tools(
            integration_name="test_service",
            tools=[
                {"name": "get_data", "description": "Get data"},
                {"name": "post_data", "description": "Post data",
                 "params": [{"name": "payload", "type": "dict"}]},
            ],
            vertical_name="TestVert",
        )
        assert tool_result.success

        # Generate agent
        agent_result = generator.generate_from_design(simple_design)
        assert agent_result.success

        # Write both
        tools_file = tmp_path / "verticals" / "test_vert" / "tools" / "test_service_tools.py"
        tool_result.write_to(tools_file)

        agent_file = tmp_path / "verticals" / "test_vert" / "agents" / "test_agent.py"
        agent_result.write_to(agent_file)

        assert tools_file.exists()
        assert agent_file.exists()

    def test_complex_inventory_agent(self, generator, complex_design, tmp_path, dynamic_registry):
        """Generate and load the complex inventory tracker agent."""
        # Generate
        result = generator.generate_from_design(complex_design)
        assert result.success, f"Generation errors: {result.errors}"

        # Write
        agent_file = tmp_path / "verticals" / "epic_bearz" / "agents" / "inventory_agent.py"
        result.write_to(agent_file)

        # Load
        load_result = dynamic_registry.load_agent_module(agent_file)
        assert load_result.success, f"Load errors: {load_result.errors}"
        assert load_result.agent_type == "inventory_tracker"

        # Verify class structure
        cls = AGENT_IMPLEMENTATIONS["inventory_tracker"]
        assert hasattr(cls, "build_graph")
        assert hasattr(cls, "get_tools")
        assert hasattr(cls, "get_state_class")

        # Clean up
        dynamic_registry.unload_agent_type("inventory_tracker")


# ===========================================================================
# Test: Edge Cases
# ===========================================================================

class TestEdgeCases:
    """Test boundary conditions and edge cases."""

    def test_agent_with_no_state_fields(self, generator):
        """Agent with empty state fields still generates valid code."""
        design = build_simple_design(
            agent_type="stateless_agent",
            agent_goal="Stateless processing",
            state_fields=[],
        )
        result = generator.generate_from_design(design)
        assert result.success
        ast.parse(result.code)

    def test_agent_with_single_node(self, generator):
        """Agent with just one node generates valid code."""
        design = build_simple_design(
            agent_type="single_node_agent",
            agent_goal="One-shot processing",
            nodes=[{"name": "execute", "description": "Execute the task"}],
        )
        result = generator.generate_from_design(design)
        assert result.success
        ast.parse(result.code)

    def test_agent_with_many_nodes(self, generator):
        """Agent with many nodes generates valid code."""
        nodes = [
            {"name": f"step_{i}", "description": f"Step {i}"}
            for i in range(10)
        ]
        design = build_simple_design(
            agent_type="many_steps_agent",
            agent_goal="Multi-step processing",
            nodes=nodes,
        )
        result = generator.generate_from_design(design)
        assert result.success
        ast.parse(result.code)

    def test_design_with_special_characters_in_goal(self, generator):
        """Goal with special characters doesn't break template."""
        design = build_simple_design(
            agent_type="special_agent",
            agent_goal='Handle "quoted" text & <html> with \'apostrophes\'',
        )
        result = generator.generate_from_design(design)
        assert result.success
        ast.parse(result.code)

    def test_no_llm_raises_clear_error(self, generator):
        """Trying to use LLM design without a client gives clear error."""
        result = generator.generate(
            agent_type="needs_llm",
            agent_goal="This needs LLM",
        )
        assert not result.success
        assert any("anthropic" in e.lower() or "client" in e.lower() for e in result.errors)

    def test_unload_nonexistent_type(self, dynamic_registry):
        """Unloading a non-existent type returns False."""
        assert not dynamic_registry.unload_agent_type("nonexistent_agent_type_xyz")
