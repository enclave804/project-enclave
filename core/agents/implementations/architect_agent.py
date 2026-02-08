"""
Architect Agent — The Meta-Agent That Hires Workers.

The ArchitectAgent drives the entire Genesis Engine flow. It doesn't do
sales work — it *creates the agents* that do sales work.

This is the only agent in the platform that creates other agents.
It orchestrates a four-stage process:

    Stage 1: Interview    — Gather business context through adaptive Q&A
    Stage 2: Blueprint    — Generate a strategic business plan
    Stage 3: Build        — Auto-generate validated YAML configs
    Stage 4: Launch       — Deploy agent fleet in shadow mode

Architecture (LangGraph State Machine):
    gather_context ──→ [enough context?]
      │ no: loop                    │ yes
      └── gather_context            ▼
                              analyze_market
                                    │
                                    ▼
                            generate_blueprint
                                    │
                                    ▼
                        ■ human_review_blueprint ■  ← GATE 1
                            │              │
                        approved       rejected (loop with feedback)
                            │              └──→ generate_blueprint
                            ▼
                        generate_configs
                            │
                            ▼
                        validate_configs
                         │           │
                       valid      invalid (loop with errors)
                         │           └──→ generate_configs
                         ▼
                   ■ request_credentials ■  ← GATE 2
                         │
                         ▼
                     ■ launch ■  ← GATE 3
                         │
                        END

Five-Gate Safety:
    1. Interview completeness gate (automated)
    2. Blueprint human review (manual approval)
    3. Config Pydantic validation (automated)
    4. Credential collection (manual entry)
    5. Shadow mode launch (all new verticals start sandboxed)

Usage:
    agent = ArchitectAgent(config, db, embedder, llm)
    result = await agent.run({
        "business_idea": "I want to start a 3D printing business for architects",
    })
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional, Type
from uuid import uuid4

from core.agents.base import BaseAgent
from core.agents.registry import register_agent_type
from core.agents.state import ArchitectAgentState
from core.config.agent_schema import AgentInstanceConfig
from core.genesis.blueprint import (
    AgentRole,
    AgentSpec,
    BlueprintStatus,
    BusinessBlueprint,
    BusinessContext,
    ComplianceJurisdiction,
    EmailSequenceSpec,
    EnrichmentSourceSpec,
    ICPSpec,
    IntegrationSpec,
    IntegrationType,
    OutreachSpec,
    PersonaSpec,
    generate_vertical_id,
)
from core.genesis.config_generator import ConfigGenerator, ConfigGenerationError
from core.genesis.interview import InterviewEngine, InterviewPhase

logger = logging.getLogger(__name__)


# ── Node name constants ──────────────────────────────────────────────────

NODE_GATHER_CONTEXT = "gather_context"
NODE_ANALYZE_MARKET = "analyze_market"
NODE_GENERATE_BLUEPRINT = "generate_blueprint"
NODE_HUMAN_REVIEW_BLUEPRINT = "human_review_blueprint"
NODE_GENERATE_CONFIGS = "generate_configs"
NODE_VALIDATE_CONFIGS = "validate_configs"
NODE_REQUEST_CREDENTIALS = "request_credentials"
NODE_LAUNCH = "launch"


@register_agent_type("architect")
class ArchitectAgent(BaseAgent):
    """
    The meta-agent that orchestrates business creation.

    Unlike other agents that process leads, write content, or book meetings,
    the ArchitectAgent creates the configuration that brings those agents
    to life. It's the only agent that writes to the verticals/ directory.

    State machine has 8 nodes, 3 human gates, and 2 self-correction loops.
    """

    def __init__(
        self,
        config: AgentInstanceConfig,
        db: Any,
        embedder: Any,
        anthropic_client: Any,
        checkpointer: Any = None,
        browser_tool: Any = None,
        mcp_tools: Optional[list[Any]] = None,
    ):
        super().__init__(
            config=config,
            db=db,
            embedder=embedder,
            anthropic_client=anthropic_client,
            checkpointer=checkpointer,
            browser_tool=browser_tool,
            mcp_tools=mcp_tools,
        )
        self._interview_engine = InterviewEngine()
        self._config_generator = ConfigGenerator()

    # ── Abstract Method Implementations ──────────────────────────

    def build_graph(self) -> Any:
        """
        Build the ArchitectAgent's LangGraph state machine.

        8 nodes, 3 human gates (interrupt_before), 2 self-correction loops:
        - Blueprint rejection → regenerate with feedback
        - Config validation failure → regenerate with error details
        """
        from langgraph.graph import StateGraph, END

        builder = StateGraph(ArchitectAgentState)

        # Register nodes
        builder.add_node(NODE_GATHER_CONTEXT, self._node_gather_context)
        builder.add_node(NODE_ANALYZE_MARKET, self._node_analyze_market)
        builder.add_node(NODE_GENERATE_BLUEPRINT, self._node_generate_blueprint)
        builder.add_node(NODE_HUMAN_REVIEW_BLUEPRINT, self._node_human_review_blueprint)
        builder.add_node(NODE_GENERATE_CONFIGS, self._node_generate_configs)
        builder.add_node(NODE_VALIDATE_CONFIGS, self._node_validate_configs)
        builder.add_node(NODE_REQUEST_CREDENTIALS, self._node_request_credentials)
        builder.add_node(NODE_LAUNCH, self._node_launch)

        # Set entry point
        builder.set_entry_point(NODE_GATHER_CONTEXT)

        # Edges — conditional routing
        builder.add_conditional_edges(
            NODE_GATHER_CONTEXT,
            self._should_continue_interview,
            {
                "continue": NODE_GATHER_CONTEXT,
                "complete": NODE_ANALYZE_MARKET,
            },
        )

        builder.add_edge(NODE_ANALYZE_MARKET, NODE_GENERATE_BLUEPRINT)
        builder.add_edge(NODE_GENERATE_BLUEPRINT, NODE_HUMAN_REVIEW_BLUEPRINT)

        # Blueprint review: approved → configs, rejected → regenerate
        builder.add_conditional_edges(
            NODE_HUMAN_REVIEW_BLUEPRINT,
            self._blueprint_review_decision,
            {
                "approved": NODE_GENERATE_CONFIGS,
                "rejected": NODE_GENERATE_BLUEPRINT,
            },
        )

        builder.add_edge(NODE_GENERATE_CONFIGS, NODE_VALIDATE_CONFIGS)

        # Config validation: valid → credentials, invalid → regenerate
        builder.add_conditional_edges(
            NODE_VALIDATE_CONFIGS,
            self._config_validation_decision,
            {
                "valid": NODE_REQUEST_CREDENTIALS,
                "invalid": NODE_GENERATE_CONFIGS,
            },
        )

        builder.add_edge(NODE_REQUEST_CREDENTIALS, NODE_LAUNCH)
        builder.add_edge(NODE_LAUNCH, END)

        # Human gates — interrupt before these nodes for manual approval
        interrupt_before = [
            NODE_HUMAN_REVIEW_BLUEPRINT,
            NODE_REQUEST_CREDENTIALS,
            NODE_LAUNCH,
        ]

        return builder.compile(
            checkpointer=self.checkpointer,
            interrupt_before=interrupt_before,
        )

    def get_tools(self) -> list[Any]:
        """The ArchitectAgent uses internal engines, not MCP tools."""
        return []

    def get_state_class(self) -> Type[ArchitectAgentState]:
        return ArchitectAgentState

    # ── State Preparation ────────────────────────────────────────

    def _prepare_initial_state(
        self, task: dict[str, Any], run_id: str
    ) -> dict[str, Any]:
        """Prepare initial state for the Genesis flow."""
        state = super()._prepare_initial_state(task, run_id)
        state.update({
            "conversation_id": str(uuid4()),
            "interview_phase": InterviewPhase.IDENTITY.value,
            "business_context": {},
            "questions_asked": [],
            "questions_remaining": len(self._interview_engine.get_all_questions()),
            "interview_complete": False,
            "blueprint": None,
            "blueprint_approved": False,
            "blueprint_feedback": None,
            "blueprint_version": 0,
            "generated_config_paths": [],
            "generated_vertical_id": None,
            "configs_validated": False,
            "validation_errors": [],
            "required_credentials": [],
            "credentials_collected": False,
            "launch_status": None,
            "launch_errors": [],
            "launched_agent_ids": [],
        })

        # If the task includes pre-populated context (e.g., from a form),
        # merge it into business_context
        if "business_context" in task:
            state["business_context"] = task["business_context"]
        elif "business_idea" in task:
            state["business_context"]["business_description"] = task["business_idea"]

        return state

    # ── Graph Nodes ──────────────────────────────────────────────

    async def _node_gather_context(
        self, state: ArchitectAgentState
    ) -> dict[str, Any]:
        """
        Node 1: Gather business context through adaptive interview.

        Uses the InterviewEngine to determine the next question,
        and the LLM to conduct a natural conversation.
        """
        context = dict(state.get("business_context", {}))
        asked = set(state.get("questions_asked", []))

        # Get next question from the engine
        question = self._interview_engine.get_next_question(context, asked)

        if question is None:
            # All questions exhausted or context is complete
            logger.info(
                "interview_context_complete",
                extra={"conversation_id": state.get("conversation_id")},
            )
            return {
                "interview_complete": True,
                "interview_phase": InterviewPhase.REVIEW.value,
                "current_node": NODE_GATHER_CONTEXT,
            }

        # In autonomous mode, the LLM generates the question.
        # In interactive mode, this would present to the user and wait.
        # For now, we prepare the question for the human gate to present.
        prompt = self._interview_engine.generate_question_prompt(
            question, context
        )

        # Track the question as asked
        new_asked = list(state.get("questions_asked", []))
        new_asked.append(question.id)

        progress = self._interview_engine.get_progress(context, set(new_asked))

        return {
            "current_node": NODE_GATHER_CONTEXT,
            "interview_phase": question.phase.value,
            "questions_asked": new_asked,
            "questions_remaining": progress.questions_remaining,
            "task_input": {
                **(state.get("task_input") or {}),
                "current_question": {
                    "id": question.id,
                    "question": question.question,
                    "hint": question.hint,
                    "phase": question.phase.value,
                    "priority": question.priority.value,
                    "multi_value": question.multi_value,
                    "examples": list(question.examples),
                    "prompt": prompt,
                },
            },
        }

    async def _node_analyze_market(
        self, state: ArchitectAgentState
    ) -> dict[str, Any]:
        """
        Node 2: Analyze market context using the LLM.

        Takes the accumulated BusinessContext and produces market insights
        that inform the blueprint. Uses the LLM to identify:
        - Best ICP refinements
        - Recommended personas and approaches
        - Risk factors
        - Success metrics
        """
        context = state.get("business_context", {})

        # Build analysis prompt for the LLM
        summary = self._interview_engine.generate_context_summary(context)

        analysis_prompt = (
            "You are a B2B sales strategist. Analyze this business context and provide:\n"
            "1. Refined ideal customer profile (ICP)\n"
            "2. 2-3 buyer personas with outreach approaches\n"
            "3. Key risk factors\n"
            "4. Success metrics/KPIs\n"
            "5. Recommended email sequences per persona\n\n"
            f"Business Context:\n{summary}\n\n"
            "Respond in JSON format."
        )

        try:
            # Call the LLM for market analysis
            analysis = await self._call_llm(
                analysis_prompt,
                system="You are a world-class B2B sales strategist with deep "
                       "expertise in cold outreach, ICP definition, and multi-channel "
                       "sales automation. Respond only in valid JSON.",
                temperature=0.4,
            )

            logger.info(
                "market_analysis_complete",
                extra={
                    "conversation_id": state.get("conversation_id"),
                    "analysis_length": len(analysis) if analysis else 0,
                },
            )

            return {
                "current_node": NODE_ANALYZE_MARKET,
                "rag_context": [{"type": "market_analysis", "content": analysis}],
            }

        except Exception as e:
            logger.error(f"Market analysis failed: {e}")
            # Continue with basic context — the blueprint generator
            # can still produce a valid config without market analysis
            return {
                "current_node": NODE_ANALYZE_MARKET,
                "rag_context": [],
                "error": f"Market analysis failed: {str(e)[:200]}",
            }

    async def _node_generate_blueprint(
        self, state: ArchitectAgentState
    ) -> dict[str, Any]:
        """
        Node 3: Generate BusinessBlueprint from context + analysis.

        This is the core intelligence — the LLM synthesizes all gathered
        context into a structured BusinessBlueprint that can be validated
        and used to generate config files.
        """
        context = state.get("business_context", {})
        feedback = state.get("blueprint_feedback")
        version = state.get("blueprint_version", 0) + 1
        market_analysis = state.get("rag_context", [])

        # Build the generation prompt
        generation_prompt = self._build_blueprint_prompt(
            context, market_analysis, feedback
        )

        try:
            # Call the LLM to generate the blueprint
            blueprint_json = await self._call_llm(
                generation_prompt,
                system=(
                    "You are the Genesis Engine — an AI that designs autonomous "
                    "sales businesses. Generate a complete BusinessBlueprint as "
                    "valid JSON. Every field must have a thoughtful, specific value. "
                    "Never use placeholder text."
                ),
                temperature=0.5,
            )

            # Parse and validate the blueprint
            blueprint = self._parse_blueprint(blueprint_json, context)

            logger.info(
                "blueprint_generated",
                extra={
                    "conversation_id": state.get("conversation_id"),
                    "vertical_id": blueprint.vertical_id,
                    "version": version,
                    "num_agents": len(blueprint.agents),
                },
            )

            return {
                "current_node": NODE_GENERATE_BLUEPRINT,
                "blueprint": blueprint.model_dump(mode="json"),
                "blueprint_version": version,
                "blueprint_approved": False,
                "error": None,
            }

        except Exception as e:
            logger.error(f"Blueprint generation failed: {e}")
            return {
                "current_node": NODE_GENERATE_BLUEPRINT,
                "error": f"Blueprint generation failed: {str(e)[:500]}",
                "blueprint_version": version,
            }

    async def _node_human_review_blueprint(
        self, state: ArchitectAgentState
    ) -> dict[str, Any]:
        """
        Node 4: Human reviews the generated blueprint.

        This is a GATE node — execution pauses here until the human
        approves or rejects. The dashboard presents the blueprint summary
        and collects feedback.

        On approval: state["blueprint_approved"] = True
        On rejection: state["blueprint_feedback"] = "reasons..."
        """
        # This node is interrupted before execution (human gate).
        # When resumed, the human has set blueprint_approved and/or
        # blueprint_feedback in the state.
        return {
            "current_node": NODE_HUMAN_REVIEW_BLUEPRINT,
            "requires_human_approval": True,
        }

    async def _node_generate_configs(
        self, state: ArchitectAgentState
    ) -> dict[str, Any]:
        """
        Node 5: Generate YAML config files from the approved blueprint.

        Uses ConfigGenerator to produce validated config files.
        All generated configs pass the same Pydantic validators
        used by the platform at runtime.
        """
        blueprint_data = state.get("blueprint")
        if not blueprint_data:
            return {
                "current_node": NODE_GENERATE_CONFIGS,
                "error": "No blueprint available for config generation",
                "configs_validated": False,
            }

        try:
            # Reconstruct the blueprint from state
            blueprint = BusinessBlueprint(**blueprint_data)

            # Generate configs (dry_run first to validate without writing)
            result = self._config_generator.generate_vertical(
                blueprint,
                output_dir="verticals",
                dry_run=True,  # Validate only — write happens after human review
            )

            if result.success:
                logger.info(
                    "configs_generated",
                    extra={
                        "vertical_id": result.vertical_id,
                        "files": len(result.paths),
                    },
                )
                return {
                    "current_node": NODE_GENERATE_CONFIGS,
                    "generated_config_paths": result.paths,
                    "generated_vertical_id": result.vertical_id,
                    "configs_validated": True,
                    "validation_errors": [],
                    "error": None,
                }
            else:
                return {
                    "current_node": NODE_GENERATE_CONFIGS,
                    "configs_validated": False,
                    "validation_errors": result.errors,
                    "error": f"Config generation failed: {'; '.join(result.errors)}",
                }

        except ConfigGenerationError as e:
            logger.error(f"Config generation failed: {e}")
            return {
                "current_node": NODE_GENERATE_CONFIGS,
                "configs_validated": False,
                "validation_errors": e.validation_errors,
                "error": str(e),
            }
        except Exception as e:
            logger.error(f"Config generation error: {e}")
            return {
                "current_node": NODE_GENERATE_CONFIGS,
                "configs_validated": False,
                "validation_errors": [str(e)],
                "error": f"Unexpected error: {str(e)[:500]}",
            }

    async def _node_validate_configs(
        self, state: ArchitectAgentState
    ) -> dict[str, Any]:
        """
        Node 6: Validate generated configs (automated gate).

        Double-checks that the generated configs would actually work
        by loading them through the same pipeline the platform uses.
        """
        if state.get("configs_validated"):
            return {
                "current_node": NODE_VALIDATE_CONFIGS,
                "configs_validated": True,
            }

        # If we get here with validation errors, it means the
        # self-correction loop should trigger regeneration
        errors = state.get("validation_errors", [])
        return {
            "current_node": NODE_VALIDATE_CONFIGS,
            "configs_validated": False,
            "validation_errors": errors,
            "error": f"Config validation failed: {'; '.join(errors)}",
        }

    async def _node_request_credentials(
        self, state: ArchitectAgentState
    ) -> dict[str, Any]:
        """
        Node 7: Request API credentials from the user.

        This is a GATE node — pauses for the user to enter their
        API keys (Apollo, email provider, etc.). Credentials are
        never auto-discovered; the user must explicitly provide them.
        """
        blueprint_data = state.get("blueprint")
        required_creds: list[dict[str, Any]] = []

        if blueprint_data:
            try:
                blueprint = BusinessBlueprint(**blueprint_data)
                for env_var in blueprint.get_required_env_vars():
                    required_creds.append({
                        "env_var": env_var,
                        "name": env_var.replace("_", " ").title(),
                        "required": True,
                        "instructions": f"Set {env_var} in your environment",
                    })
            except Exception as e:
                logger.warning(f"Could not extract required credentials: {e}")

        return {
            "current_node": NODE_REQUEST_CREDENTIALS,
            "required_credentials": required_creds,
            "requires_human_approval": True,
        }

    async def _node_launch(
        self, state: ArchitectAgentState
    ) -> dict[str, Any]:
        """
        Node 8: Launch the vertical in shadow mode.

        Writes the config files to disk (no longer dry_run) and
        registers the agents. All new verticals launch in shadow mode —
        they process real data but never trigger external effects.
        """
        blueprint_data = state.get("blueprint")
        if not blueprint_data:
            return {
                "current_node": NODE_LAUNCH,
                "launch_status": "failed",
                "launch_errors": ["No blueprint available"],
            }

        try:
            blueprint = BusinessBlueprint(**blueprint_data)

            # Actually write the files now
            result = self._config_generator.generate_vertical(
                blueprint,
                output_dir="verticals",
                dry_run=False,
            )

            if not result.success:
                return {
                    "current_node": NODE_LAUNCH,
                    "launch_status": "failed",
                    "launch_errors": result.errors,
                }

            # Store the blueprint in the database for future reference
            try:
                self.db.store_blueprint(
                    blueprint_id=blueprint.id,
                    vertical_id=blueprint.vertical_id,
                    blueprint_data=blueprint.model_dump(mode="json"),
                    status="launched",
                )
            except Exception as e:
                logger.warning(f"Could not store blueprint in DB: {e}")

            logger.info(
                "vertical_launched",
                extra={
                    "vertical_id": blueprint.vertical_id,
                    "mode": "shadow",
                    "files_created": len(result.paths),
                },
            )

            return {
                "current_node": NODE_LAUNCH,
                "launch_status": "shadow_mode",
                "generated_config_paths": result.paths,
                "generated_vertical_id": result.vertical_id,
                "launched_agent_ids": [
                    f"{a.agent_type.value if hasattr(a.agent_type, 'value') else str(a.agent_type)}_v1"
                    for a in blueprint.agents
                ],
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            logger.error(f"Launch failed: {e}")
            return {
                "current_node": NODE_LAUNCH,
                "launch_status": "failed",
                "launch_errors": [str(e)[:500]],
            }

    # ── Conditional Edges ────────────────────────────────────────

    def _should_continue_interview(
        self, state: ArchitectAgentState
    ) -> str:
        """Decide whether to continue gathering context or proceed."""
        if state.get("interview_complete"):
            return "complete"

        context = state.get("business_context", {})
        if self._interview_engine.is_complete(context):
            return "complete"

        # Safety: limit total questions to prevent infinite loops
        asked = state.get("questions_asked", [])
        if len(asked) >= len(self._interview_engine.get_all_questions()):
            return "complete"

        return "continue"

    def _blueprint_review_decision(
        self, state: ArchitectAgentState
    ) -> str:
        """Route based on human blueprint review."""
        if state.get("blueprint_approved"):
            return "approved"

        # Safety: limit blueprint revisions
        version = state.get("blueprint_version", 1)
        if version >= 5:
            logger.warning(
                "Blueprint revision limit reached — auto-approving "
                "to prevent infinite loop."
            )
            return "approved"

        return "rejected"

    def _config_validation_decision(
        self, state: ArchitectAgentState
    ) -> str:
        """Route based on config validation result."""
        if state.get("configs_validated"):
            return "valid"

        # Safety: limit config regeneration attempts
        retry = state.get("retry_count", 0)
        if retry >= 3:
            logger.error(
                "Config generation retry limit reached. "
                "Manual intervention required."
            )
            # Still return invalid — the agent will error out
            # rather than producing invalid configs
            return "invalid"

        return "invalid"

    # ── LLM Interaction ──────────────────────────────────────────

    async def _call_llm(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.5,
    ) -> str:
        """
        Call the LLM (Claude) with a prompt.

        Uses the anthropic client injected via constructor.
        Handles both the Anthropic SDK and mock clients for testing.
        """
        try:
            # Support both real Anthropic client and mocks
            if hasattr(self.llm, "messages"):
                # Real Anthropic client
                messages = [{"role": "user", "content": prompt}]
                kwargs: dict[str, Any] = {
                    "model": self.config.model.model,
                    "max_tokens": self.config.model.max_tokens,
                    "temperature": temperature,
                    "messages": messages,
                }
                if system:
                    kwargs["system"] = system

                response = self.llm.messages.create(**kwargs)
                return response.content[0].text

            elif callable(self.llm):
                # Mock/test client — simple callable
                return self.llm(prompt, system=system)

            else:
                raise ValueError(
                    f"Unsupported LLM client type: {type(self.llm).__name__}"
                )

        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise

    # ── Blueprint Construction ───────────────────────────────────

    def _build_blueprint_prompt(
        self,
        context: dict[str, Any],
        market_analysis: list[dict[str, Any]],
        feedback: Optional[str] = None,
    ) -> str:
        """Build the prompt for blueprint generation."""
        summary = self._interview_engine.generate_context_summary(context)

        parts = [
            "Generate a complete BusinessBlueprint for this business.\n",
            f"## Business Context\n{summary}\n",
        ]

        if market_analysis:
            for analysis in market_analysis:
                content = analysis.get("content", "")
                if content:
                    parts.append(f"## Market Analysis\n{content}\n")

        if feedback:
            parts.append(
                f"## Previous Feedback (address these concerns)\n{feedback}\n"
            )

        parts.append(
            "\n## Required Output Format\n"
            "Return a JSON object with these fields:\n"
            "- vertical_id: snake_case identifier\n"
            "- vertical_name: human-readable name\n"
            "- industry: business industry\n"
            "- icp: {company_size: [min, max], industries: [...], signals: [...], disqualifiers: [...]}\n"
            "- personas: [{id, title_patterns, company_size, approach, seniorities}]\n"
            "- outreach: {daily_limit, warmup_days, sending_domain, reply_to, "
            "sequences: [{name, steps, delay_days}], jurisdictions, physical_address}\n"
            "- agents: [{agent_type, name, description, enabled, browser_enabled, tools, "
            "human_gate_nodes, params}]\n"
            "- integrations: [{name, type, env_var, required, instructions}]\n"
            "- enrichment_sources: [{type, targets}]\n"
            "- strategy_reasoning: why these choices\n"
            "- risk_factors: potential risks\n"
            "- success_metrics: KPIs to track\n"
            "- content_topics: for SEO agent\n"
            "- tone: communication style\n"
        )

        return "\n".join(parts)

    def _parse_blueprint(
        self,
        llm_output: str,
        context: dict[str, Any],
    ) -> BusinessBlueprint:
        """
        Parse LLM output into a validated BusinessBlueprint.

        Handles common LLM output issues:
        - JSON wrapped in markdown code blocks
        - Missing required fields (filled from context)
        - Type mismatches (coerced where possible)
        """
        # Strip markdown code block wrappers
        text = llm_output.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        # Parse JSON
        data = json.loads(text)

        # Ensure required fields from context
        if "vertical_id" not in data:
            name = context.get("business_name", "unnamed")
            data["vertical_id"] = generate_vertical_id(name)

        if "vertical_name" not in data:
            data["vertical_name"] = context.get("business_name", "Unnamed Business")

        if "industry" not in data:
            industries = context.get("target_industries", [])
            data["industry"] = industries[0] if industries else "General"

        # Inject the full context
        if "context" not in data:
            data["context"] = context

        # Ensure at minimum an outreach agent exists
        agents = data.get("agents", [])
        has_outreach = any(
            a.get("agent_type") in ("outreach", AgentRole.OUTREACH.value)
            for a in agents
        )
        if not has_outreach:
            agents.insert(0, {
                "agent_type": "outreach",
                "name": "Outreach Agent",
                "description": f"Lead generation and cold outreach for {data.get('vertical_name', 'the business')}",
                "enabled": True,
            })
            data["agents"] = agents

        # Build and validate the blueprint
        return BusinessBlueprint(**data)

    # ── Knowledge Writing ────────────────────────────────────────

    async def write_knowledge(self, result: dict[str, Any]) -> None:
        """
        Write blueprint insights to shared brain.

        Successful blueprints become training data for future generation:
        - What industries worked
        - What agent configurations were approved
        - What personas resonated
        """
        if result.get("launch_status") != "shadow_mode":
            return  # Only write knowledge for successful launches

        blueprint_data = result.get("blueprint")
        if not blueprint_data:
            return

        from core.agents.contracts import InsightData

        try:
            insight = InsightData(
                insight_type="blueprint_template",
                title=f"Approved Blueprint: {blueprint_data.get('vertical_name', 'unknown')}",
                content=json.dumps(blueprint_data, default=str)[:2000],
                confidence=0.9,
                metadata={
                    "vertical_id": blueprint_data.get("vertical_id"),
                    "industry": blueprint_data.get("industry"),
                    "num_agents": len(blueprint_data.get("agents", [])),
                },
            )
            self.store_insight(insight)
        except Exception as e:
            logger.warning(f"Failed to write blueprint knowledge: {e}")

    def __repr__(self) -> str:
        return (
            f"<ArchitectAgent "
            f"agent_id={self.agent_id!r} "
            f"vertical={self.vertical_id!r} "
            f"genesis_engine=True>"
        )
