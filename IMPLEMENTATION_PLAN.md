# Sovereign Venture Engine — Master Implementation Plan

## From "B2B Sales Tool" to "AI Company Builder"

**Team:** Jose (CEO) + Claude (Lead Architect) + Gemini (Research/Creative)
**Current State:** Phase 4 complete — 1,102 tests, 5 agent types, Genesis Engine, Dashboard
**Goal:** Platform that takes ANY business idea and deploys a custom AI workforce to run it
**Timeline:** 9 phases, ~2-3 months at our pace (1 phase per 1-3 day session)

---

## Architecture Principle: BUILD ONCE, USE EVERYWHERE

Every component we build must be:
- **Config-driven** — behavior comes from YAML, not code
- **Registry-based** — new types added at runtime, not import time
- **Composable** — mix and match tools, models, workflows
- **Self-documenting** — agents describe their own capabilities

---

## PHASE 5: THE AGENT FACTORY
### "Genesis Creates ANY Agent, Not Just 5"

**Priority:** CRITICAL — This is the multiplier. Everything after depends on this.
**Estimated:** 1-2 sessions

### Why First?
Currently: Genesis knows 5 agent types (outreach, seo_content, appointment_setter, architect, janitor).
After: Genesis can DESIGN and DEPLOY any agent type — "I need an Inventory Agent" → built and running.

### Sprint 5.1: Universal Interview Engine

**Problem:** Interview asks sales-only questions (email sequences, sending domain, CAN-SPAM).
**Solution:** Vertical-type-aware interview with conditional question banks.

**Files to create/modify:**
```
core/genesis/interview.py          — MODIFY: Add vertical_type routing
core/genesis/question_banks/       — NEW directory
  ├── __init__.py
  ├── base_questions.py            — Questions ALL verticals need
  ├── sales_outreach.py            — Current questions (extracted)
  ├── ecommerce.py                 — Shopify, fulfillment, PPC questions
  ├── content_marketing.py         — SEO, publishing, audience questions
  ├── service_business.py          — Booking, pricing, service area
  └── custom.py                    — AI-generated questions for unknown types
```

**Key changes:**
- BusinessContext gets `vertical_type: str` field (free-form, not enum)
- InterviewEngine Phase 1 (IDENTITY) determines vertical_type
- Subsequent phases load appropriate question bank
- For UNKNOWN vertical types → Claude generates custom questions on the fly
- All existing sales questions preserved in `sales_outreach.py`

**Tests:** ~40 new tests
- Test each question bank loads correctly
- Test vertical_type detection from business description
- Test custom question generation for unknown types
- Test backward compatibility (enclave_guard still works)

### Sprint 5.2: Flexible Blueprint Schema

**Problem:** Blueprint requires outreach agent, email sequences, Apollo config.
**Solution:** Make everything optional, validate by vertical_type.

**Files to modify:**
```
core/genesis/blueprint.py          — MODIFY: Conditional validation
core/config/schema.py              — MODIFY: Optional sections
```

**Key changes to blueprint.py:**
- Remove hard requirement for OUTREACH agent
- Add `vertical_type` to BusinessBlueprint
- Validator checks required agents BY vertical_type:
  - sales_outreach → needs outreach agent
  - ecommerce → needs storefront agent + marketing agent
  - content_marketing → needs content agent
  - custom → needs at least 1 agent (any type)
- AgentRole enum → extensible (allow custom string values)
- IntegrationType enum → extensible

**Key changes to schema.py:**
- `outreach: Optional[OutreachConfig] = None`
- `apollo: Optional[ApolloConfig] = None`
- New optional sections: `ecommerce: Optional[EcommerceConfig]`
- New optional sections: `advertising: Optional[AdvertisingConfig]`
- `vertical_type` field on VerticalConfig
- Validator ensures required sections present for vertical_type
- ChunkType enum → allow custom string values for RAG

**Tests:** ~50 new tests
- Blueprint validation per vertical_type
- Config generation for non-sales verticals
- Schema validation with optional sections
- Backward compatibility (all existing verticals still valid)

### Sprint 5.3: Dynamic Agent Template Engine

**Problem:** Each agent type is a Python class. Can't create new types without code.
**Solution:** Agent Template Engine — Claude designs agent behavior from description.

**Files to create:**
```
core/agents/template_engine.py     — NEW: Generates agent configs from description
core/agents/generic_agent.py       — NEW: Universal agent that runs any graph topology
core/agents/graph_builder.py       — NEW: Builds LangGraph from YAML topology
core/agents/handler_registry.py    — NEW: Registry of reusable node handlers
```

**template_engine.py** — The brain:
- Input: Agent description + available tools + vertical context
- Output: Complete agent YAML config including graph topology
- Uses Claude to design: system prompt, tool selection, graph flow, human gates
- Validates output against AgentInstanceConfig schema
- Caches templates for reuse (same agent type = same template)

**generic_agent.py** — The universal soldier:
- Extends BaseAgent
- Instead of hardcoded build_graph(), reads topology from config
- graph_builder.py compiles YAML topology → LangGraph
- Can run ANY workflow: linear, branching, looping, parallel
- Falls back to simple linear flow if no topology specified

**graph_builder.py** — The compiler:
- Input: YAML graph spec (nodes, edges, conditions)
- Output: Compiled LangGraph StateGraph
- Supports: sequential, conditional, parallel, human-interrupt nodes
- Validates: no cycles (unless explicitly allowed), all nodes reachable
- Handler resolution: looks up node handlers from handler_registry

**handler_registry.py** — Reusable building blocks:
- Pre-built handlers that any agent can use:
  - `handler_llm_call` — Call LLM with prompt + context
  - `handler_tool_call` — Execute an MCP tool
  - `handler_rag_search` — Search knowledge base
  - `handler_rag_write` — Write to knowledge base
  - `handler_human_review` — Interrupt for human approval
  - `handler_api_call` — Generic HTTP API call
  - `handler_transform` — Transform data between nodes
  - `handler_conditional` — Route based on state
- Custom handlers loaded from vertical modules

**Tests:** ~80 new tests
- Template generation for various agent descriptions
- GenericAgent runs simple linear graph
- GenericAgent runs conditional branching
- GenericAgent runs with human interrupt
- Graph builder validates topology
- Handler registry resolves handlers
- End-to-end: description → template → config → running agent

### Sprint 5.4: Dynamic Tool & Integration Registry

**Problem:** Tools registered at import time. Can't add tools per vertical.
**Solution:** Runtime tool registry + integration framework.

**Files to create/modify:**
```
core/mcp/tool_registry.py          — NEW: Dynamic tool management
core/integrations/base.py          — NEW: BaseIntegration abstract class
core/integrations/registry.py      — NEW: Integration registry
core/mcp/server.py                 — MODIFY: Use tool registry
```

**tool_registry.py:**
- `register_tool(name, handler, schema, category)` — Add tool at runtime
- `get_tools_for_agent(agent_config)` — Return tools matching agent's config
- `list_available_tools()` — Discovery endpoint
- `get_tool_schema(name)` — Return parameter schema
- Tools organized by category: lead_gen, email, content, ecommerce, analytics, etc.

**base.py (BaseIntegration):**
```python
class BaseIntegration(ABC):
    @abstractmethod
    async def test_connection(self) -> bool
    @abstractmethod
    def get_tools(self) -> list[Tool]
    @abstractmethod
    def get_required_credentials(self) -> list[str]
```

**registry.py (IntegrationRegistry):**
- Register integration classes by type
- Instantiate from config (vertical YAML)
- Auto-register tools from integration into tool_registry
- Support custom integrations loaded from vertical modules

**Tests:** ~50 new tests
- Runtime tool registration
- Tool discovery by category
- Integration registration and instantiation
- Tool wiring from integration to agent
- Backward compatibility (existing tools still work)

### Sprint 5.5: Config Generator Upgrade

**Problem:** Config generator has hardcoded defaults for 5 agent types.
**Solution:** Generator queries template engine for unknown types.

**Files to modify:**
```
core/genesis/config_generator.py   — MODIFY: Support dynamic agent types
```

**Key changes:**
- When agent_type not in DEFAULT_AGENT_TOOLS:
  - Query template_engine for agent schema
  - Use returned tools, human gates, graph topology
- Generate agent YAML with graph topology section
- Generate integration configs from blueprint
- Validate everything against schema before writing

**Tests:** ~30 new tests
- Generate config for custom agent type
- Generate config for ecommerce vertical
- Validate generated configs
- End-to-end: blueprint with custom agents → valid configs

### Phase 5 Milestone Test
> `genesis start` → "I want an e-commerce clothing brand"
> → Interview asks about products, fulfillment, marketing
> → Blueprint includes: Storefront Agent, Marketing Agent, Content Agent
> → Configs generated for all agents (including custom types)
> → All agents deploy in shadow mode
> → 1,250+ tests passing

---

## PHASE 6: MULTI-MODEL & CREATIVE ENGINE
### "Right Model for the Right Job"

**Estimated:** 1-2 sessions

### Sprint 6.1: Model Router

**Files to create:**
```
core/models/                        — NEW directory
  ├── __init__.py
  ├── router.py                     — Model selection logic
  ├── registry.py                   — Available models database
  ├── providers/
  │   ├── __init__.py
  │   ├── anthropic_provider.py     — Claude models
  │   ├── openai_provider.py        — GPT models
  │   ├── google_provider.py        — Gemini models
  │   ├── ollama_provider.py        — Local models
  │   └── base.py                   — BaseProvider interface
  └── cost_tracker.py               — Per-task cost tracking
```

**router.py** — Smart model selection:
- Route by task type: reasoning → Claude, fast classification → Haiku, creative → GPT-4
- Route by cost budget: cheap tasks → small models, important → large
- Route by latency: real-time → fast models, batch → powerful
- Fallback chains: if primary model fails, try secondary
- A/B testing: split traffic between models, measure quality

**Tests:** ~60 new tests

### Sprint 6.2: Image Generation

**Files to create:**
```
core/creative/                      — NEW directory
  ├── __init__.py
  ├── image_generator.py            — Multi-provider image gen
  ├── brand_memory.py               — Remember brand style/colors/logo
  ├── providers/
  │   ├── dalle_provider.py
  │   ├── flux_provider.py
  │   └── base.py
  └── asset_manager.py              — Store/organize generated media
```

**Tests:** ~40 new tests

### Sprint 6.3: Voice Synthesis (Foundation)

**Files to create:**
```
core/voice/                         — NEW directory
  ├── __init__.py
  ├── tts_engine.py                 — Text-to-speech
  ├── stt_engine.py                 — Speech-to-text
  ├── providers/
  │   ├── elevenlabs_provider.py
  │   ├── deepgram_provider.py
  │   └── base.py
  └── voice_profile.py              — Per-vertical voice settings
```

**Tests:** ~30 new tests

### Phase 6 Milestone
> Model router selects optimal model per task
> Agent generates product images for Epic Bearz
> Voice profile created for brand
> Cost tracking shows per-task spend
> 1,400+ tests passing

---

## PHASE 7: E-COMMERCE STACK
### "Agents Run Your Online Store"

**Estimated:** 1-2 sessions

### Sprint 7.1: Shopify Integration

**Files to create:**
```
core/integrations/shopify_client.py   — Shopify API wrapper
core/integrations/stripe_client.py    — Stripe payments
core/integrations/printful_client.py  — Print-on-demand fulfillment
```

Each integration follows BaseIntegration pattern from Phase 5.
Auto-registers tools in tool_registry.

### Sprint 7.2: E-commerce Agent Types

Using the Agent Factory from Phase 5, create template configs for:
- **Storefront Agent** — Product listings, pricing, inventory
- **Order Manager Agent** — Track orders, handle fulfillment
- **Customer Support Agent** — Handle inquiries, returns, sizing

These are YAML configs, not new Python classes (thanks to GenericAgent).

### Sprint 7.3: E-commerce Question Bank

```
core/genesis/question_banks/ecommerce.py
```
- What platform? (Shopify/WooCommerce/BigCommerce)
- Product type? (physical/digital/print-on-demand)
- Fulfillment? (self/dropship/POD provider)
- Payment processing?
- Target market/demographics?
- Competitors?

### Phase 7 Milestone
> `genesis start` → "I want a print-on-demand clothing brand called Epic Bearz"
> → Creates Shopify integration, Printful connection, Stripe payments
> → Deploys Storefront Agent + Marketing Agent + Customer Support Agent
> → Agents can list products, track orders, respond to customers
> 1,550+ tests passing

---

## PHASE 8: VOICE & PHONE
### "Agents Make and Receive Calls"

**Estimated:** 1-2 sessions

### Sprint 8.1: Twilio Integration

```
core/integrations/twilio_client.py    — Voice + SMS
core/voice/call_engine.py             — Manage live calls
core/voice/conversation_engine.py     — Real-time dialogue
```

### Sprint 8.2: Phone Agent Template

- Outbound calls (sales, follow-up, appointment confirmation)
- Inbound calls (customer service, order status, booking)
- Voicemail detection + handling
- Call recording + transcription → RAG

### Sprint 8.3: SMS/WhatsApp

```
core/integrations/whatsapp_client.py
```
- Conversational messaging
- Media sharing (product images)
- Quick reply buttons

### Phase 8 Milestone
> Appointment Setter can call leads by phone
> Customer Support Agent answers incoming calls
> All conversations transcribed and fed to RAG
> 1,700+ tests passing

---

## PHASE 9: ADVERTISING ENGINE
### "Agents Manage Your Ad Spend"

**Estimated:** 1-2 sessions

### Sprint 9.1: Ad Platform Integrations

```
core/integrations/google_ads_client.py
core/integrations/meta_ads_client.py
core/integrations/tiktok_ads_client.py
```

### Sprint 9.2: PPC Agent Template

- Campaign creation from vertical context
- Keyword research + bidding
- Audience targeting from ICP
- Budget management + optimization
- Creative generation (using Phase 6 image gen)
- Performance tracking → RAG (learn what converts)

### Sprint 9.3: Retargeting & Attribution

- Pixel/conversion tracking integration
- Cross-channel attribution
- Retargeting audience sync from CRM data

### Phase 9 Milestone
> PPC Agent creates and manages Google Ads campaign for Epic Bearz
> Auto-generates ad creatives with brand style
> Optimizes spend based on conversion data
> 1,850+ tests passing

---

## PHASE 10: SOCIAL MEDIA ENGINE
### "Agents Build Your Online Presence"

**Estimated:** 1-2 sessions

### Sprint 10.1: Social Platform Integrations

```
core/integrations/instagram_client.py
core/integrations/tiktok_client.py
core/integrations/pinterest_client.py
core/integrations/linkedin_client.py
core/integrations/twitter_client.py
```

### Sprint 10.2: Social Media Agent Template

- Content calendar generation
- Multi-platform posting (adapted per platform)
- Image/video generation per post (Phase 6)
- Hashtag research + trending topics
- Community management (reply to comments/DMs)
- Analytics aggregation

### Sprint 10.3: Influencer Outreach

- Find influencers by niche/audience size
- Generate personalized collab proposals
- Track partnerships and ROI

### Phase 10 Milestone
> Social Media Agent posts daily to Instagram, TikTok, Pinterest
> Each post has AI-generated image in Epic Bearz brand style
> Agent responds to comments and DMs
> 2,000+ tests passing

---

## PHASE 11: OPERATIONS & ADVANCED N8N
### "Back Office Runs Itself"

**Estimated:** 1-2 sessions

### Sprint 11.1: Finance Agent

```
core/integrations/quickbooks_client.py  (or Xero)
core/integrations/stripe_billing.py     (invoicing)
```

- Invoice generation
- Expense tracking + categorization
- P&L reporting
- Cash flow projections
- Tax obligation alerts

### Sprint 11.2: Advanced N8N Workflows

```
core/integrations/n8n_client.py    — MAJOR UPGRADE
core/workflows/                     — NEW directory
  ├── __init__.py
  ├── workflow_engine.py            — Multi-step automation
  ├── triggers.py                   — Event → workflow mapping
  └── templates/                    — Pre-built workflow templates
      ├── ecommerce_order_flow.py
      ├── lead_nurture_flow.py
      └── content_publish_flow.py
```

- Complex event-triggered automations
- Cross-platform orchestration
- Workflow templates per industry
- Error handling + retry logic

### Sprint 11.3: Operations Agent Template

- Inventory monitoring + reorder alerts
- Shipping/logistics tracking
- Supplier communication
- Daily/weekly operations digest

### Phase 11 Milestone
> Finance Agent generates monthly P&L for Epic Bearz
> N8N automates: new order → fulfillment → shipping notification → review request
> Operations Agent sends daily digest to Telegram
> 2,150+ tests passing

---

## PHASE 12: THE LEARNING SYSTEM
### "The Platform Gets Smarter Every Day"

**Estimated:** 2-3 sessions (most complex phase)

### Sprint 12.1: RLHF Pipeline

```
core/learning/                      — NEW directory
  ├── __init__.py
  ├── feedback_collector.py         — Collect human ratings
  ├── training_pipeline.py          — Prepare training data
  ├── evaluator.py                  — Model quality scoring
  └── ab_testing.py                 — Compare model variants
```

- Every approve/reject in human review = training signal
- Every email reply = outcome signal
- Every ad conversion = performance signal
- Aggregate signals → training datasets
- Fine-tune per-vertical models

### Sprint 12.2: Fine-Tuning Infrastructure

```
core/learning/fine_tuning/
  ├── __init__.py
  ├── data_preparation.py           — Clean + format training data
  ├── lora_trainer.py               — LoRA fine-tuning for open models
  ├── api_fine_tuning.py            — OpenAI/Anthropic fine-tuning APIs
  └── model_registry.py             — Track fine-tuned models
```

- Support multiple fine-tuning approaches:
  - API-based (OpenAI fine-tuning)
  - Local (LoRA/QLoRA on open models)
  - Prompt tuning (optimal prompts from data)
- Per-vertical model specialization
- Automatic retraining on schedule

### Sprint 12.3: Predictive Analytics

```
core/learning/analytics/
  ├── __init__.py
  ├── lead_scoring.py               — Predict which leads will convert
  ├── content_scoring.py            — Predict which content will perform
  ├── churn_prediction.py           — Predict customer churn risk
  └── recommendation_engine.py      — Suggest next best action
```

### Sprint 12.4: Cross-Vertical Intelligence

- When a new vertical launches, seed it with general knowledge
- Transfer patterns that work across industries
- "Companies in the 50-200 employee range respond best on Tuesdays" — applies everywhere

### Phase 12 Milestone
> Fine-tuned Epic Bearz model outperforms base model on brand-specific tasks
> Lead scoring predicts conversion with 80%+ accuracy
> Content scoring predicts engagement before publishing
> New verticals launch pre-loaded with cross-vertical intelligence
> 2,400+ tests passing

---

## PHASE 13: SCALE & ENTERPRISE
### "100 Verticals Running Simultaneously"

**Estimated:** 1-2 sessions

### Sprint 13.1: Kubernetes Deployment

```
infrastructure/k8s/
  ├── deployment.yaml
  ├── service.yaml
  ├── ingress.yaml
  ├── hpa.yaml                      — Horizontal Pod Autoscaler
  ├── configmap.yaml
  └── secrets.yaml
```

### Sprint 13.2: Multi-Region

- Database read replicas
- CDN for media assets
- Region-aware routing

### Sprint 13.3: Enterprise Compliance

- Full GDPR implementation
- PECR, CASL, LGPD
- SOC2 audit preparation
- Data residency controls

### Sprint 13.4: Monitoring & Alerting

- Prometheus metrics
- Grafana dashboards
- PagerDuty/OpsGenie alerting
- SLA tracking

### Phase 13 Milestone
> Platform runs 100 concurrent verticals
> Auto-scales based on load
> Full international compliance
> 99.9% uptime architecture
> 2,600+ tests passing

---

## DEPENDENCY GRAPH

```
Phase 5 (Agent Factory) ─────────┬──→ Phase 7 (E-commerce)
         │                        ├──→ Phase 8 (Voice/Phone)
         │                        ├──→ Phase 9 (Advertising)
         │                        ├──→ Phase 10 (Social Media)
         │                        └──→ Phase 11 (Operations/N8N)
         │
         └─→ Phase 6 (Multi-Model) ──→ Phase 9 (needs image gen for ads)
                    │                 └─→ Phase 10 (needs image gen for social)
                    │
                    └─→ Phase 12 (Learning System)
                              │
                              └─→ Phase 13 (Scale)
```

**Critical path:** Phase 5 → Phase 6 → Phase 12 → Phase 13
**Can be parallelized:** Phases 7, 8, 9, 10, 11 (all depend on 5+6, not each other)

---

## ACCELERATION STRATEGIES

### 1. Phase 5 Unlocks Everything
Once GenericAgent + Agent Factory works, phases 7-11 become mostly:
- Write integration client (API wrapper)
- Write question bank (interview questions)
- Write agent template YAML (not code)
- Write tests
Genesis Engine does the rest.

### 2. Gemini Pre-Research
While Claude builds Phase N, Jose has Gemini research Phase N+1:
- API documentation
- SDK examples
- Best practices
- Competitor analysis

### 3. Template Pattern
Every integration follows the same pattern:
```python
class XxxClient(BaseIntegration):
    async def test_connection(self) -> bool
    def get_tools(self) -> list[Tool]
    def get_required_credentials(self) -> list[str]
```
Write one, stamp out the rest.

### 4. Test-Driven Speed
Write tests first → implementation follows fast.
Tests also serve as documentation and prevent regressions.

### 5. Session Cadence
- **Session prep:** Jose + Gemini research APIs, gather docs
- **Build session:** Jose + Claude implement + test
- **Review session:** Jose reviews, tests with real APIs
- **Repeat**

---

## DOCUMENTATION REQUIREMENTS

Every phase must produce:
1. **Code** — Implementation with docstrings
2. **Tests** — Comprehensive test coverage
3. **CHANGELOG.md** — What changed and why
4. **USER_MANUAL.md** — Updated user-facing docs
5. **API_REFERENCE.md** — Technical API documentation
6. **Architecture diagrams** — Updated when structure changes

---

## PRE-FLIGHT CHECKLIST (Before Starting Phase 5)

- [ ] Commit USER_MANUAL.md and ROADMAP_AND_TEAM.md
- [ ] Commit IMPLEMENTATION_PLAN.md (this file)
- [ ] Run full test suite — confirm 1,102 passing
- [ ] Review current Genesis Engine flow end-to-end
- [ ] Ensure all current tests still pass after any file modifications

---

## SUCCESS METRICS

| Milestone | Metric |
|-----------|--------|
| Phase 5 complete | Genesis deploys custom agent types |
| Phase 7 complete | E-commerce vertical runs end-to-end |
| Phase 10 complete | Epic Bearz has full marketing operation |
| Phase 12 complete | Platform demonstrably improves over time |
| Phase 13 complete | 100 verticals, 99.9% uptime |

---

## THE BET

One person. Two AIs. Zero payroll.
Building what funded teams spend millions on.
Not because we're reckless — because the tools are that good.

Let's go. 🚀
