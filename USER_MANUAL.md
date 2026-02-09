# Sovereign Venture Engine — User Manual

> **Version 1.0** | Phase 4 Complete (Genesis Engine Foundation)
> 1,102 tests passing | Built by Jose + Claude + Gemini
> Next: Phase 5 — The Agent Factory (Dynamic Agent Creation)

---

## Table of Contents

1. [What Is the Sovereign Venture Engine?](#1-what-is-the-sovereign-venture-engine)
2. [Quick Start](#2-quick-start)
3. [Architecture Overview](#3-architecture-overview)
4. [The Genesis Engine — Launch a New Business](#4-the-genesis-engine--launch-a-new-business)
5. [CLI Reference](#5-cli-reference)
6. [The Sovereign Cockpit (Dashboard)](#6-the-sovereign-cockpit-dashboard)
7. [The Sales Pipeline](#7-the-sales-pipeline)
8. [Agent System](#8-agent-system)
9. [Configuration](#9-configuration)
10. [Credentials and Security](#10-credentials-and-security)
11. [Compliance](#11-compliance)
12. [RAG Knowledge Base](#12-rag-knowledge-base)
13. [Adding a New Vertical Manually](#13-adding-a-new-vertical-manually)
14. [Troubleshooting](#14-troubleshooting)
15. [Glossary](#15-glossary)

---

## 1. What Is the Sovereign Venture Engine?

The Sovereign Venture Engine (SVE) is an AI-powered platform that turns a business idea into a fully operational AI workforce. You describe your business, the platform interviews you, designs the exact team of AI agents your business needs, and deploys them — all under your control.

**The vision:** Tell it "I want to start a clothing brand called Epic Bearz" and it builds and deploys 10-15 AI agents to handle everything — storefront, marketing, customer support, advertising, social media, finance, and more. Each agent learns from every interaction, getting smarter over time.

**Current state (Phase 4):** The foundation is complete — Genesis Engine, 5 agent types, 10-node sales pipeline, RAG knowledge base, dashboard, and ChatOps. Phase 5+ will expand to unlimited dynamic agent types, multi-model AI, e-commerce, voice/phone, advertising, and the full self-improving learning system.

**What it does:**

- **Generates a business vertical** from a conversational interview (the Genesis Engine)
- **Finds leads** via Apollo.io based on your Ideal Customer Profile
- **Enriches leads** with company intel, tech stacks, and security posture data
- **Qualifies leads** against configurable scoring criteria
- **Drafts personalized emails** using Claude AI with RAG-powered context
- **Checks compliance** across CAN-SPAM, GDPR, PECR, and CASL
- **Requires human approval** before sending anything
- **Learns from outcomes** via a self-improving knowledge base

**Key principles:**

- **Config-driven** — New business verticals are just YAML files
- **Human-in-the-loop** — No email is sent without your approval
- **Shadow mode** — New verticals launch sandboxed; nothing goes out for real
- **Privacy-first** — Only uses publicly available data for enrichment
- **Compliance-built-in** — Jurisdiction-aware outreach with suppression lists

---

## 2. Quick Start

### Prerequisites

- Python 3.11+
- A Supabase project (with pgvector enabled)
- API keys: Anthropic, Apollo.io, OpenAI (for embeddings)
- Optional: SendGrid or Mailgun for email delivery

### Installation

```bash
git clone <repo-url> && cd "Project Enclave"
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Set Environment Variables

Create a `.env` file in the project root:

```env
ANTHROPIC_API_KEY=sk-ant-...
SUPABASE_URL=https://xxx.supabase.co
SUPABASE_SERVICE_KEY=eyJ...
OPENAI_API_KEY=sk-...
APOLLO_API_KEY=...
ENCLAVE_MASTER_KEY=your-secret-master-key-here
```

### Run the Pipeline (Test Mode)

```bash
python main.py run enclave_guard --test --leads 3 --verbose
```

### Launch the Dashboard

```bash
python main.py dashboard
```

Then open http://localhost:8501 in your browser.

---

## 3. Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│                    SOVEREIGN VENTURE ENGINE                    │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────────┐    │
│  │ GENESIS  │───→│ CONFIG GEN   │───→│ LAUNCHER         │    │
│  │ ENGINE   │    │ (YAML files) │    │ (Shadow Mode)    │    │
│  └──────────┘    └──────────────┘    └──────────────────┘    │
│       │                                       │               │
│       │ Interview + Blueprint                 │ Agent Fleet   │
│       ▼                                       ▼               │
│  ┌──────────────────────────────────────────────────────┐    │
│  │                   AGENT REGISTRY                      │    │
│  │   outreach │ seo_content │ appointment │ maintenance  │    │
│  └──────────────────────────────────────────────────────┘    │
│                          │                                    │
│                          ▼                                    │
│  ┌──────────────────────────────────────────────────────┐    │
│  │              LANGGRAPH PIPELINE                       │    │
│  │  duplicate → enrich → qualify → strategy → draft      │    │
│  │  → compliance → HUMAN REVIEW → send → write_to_rag   │    │
│  └──────────────────────────────────────────────────────┘    │
│                          │                                    │
│              ┌───────────┼────────────┐                      │
│              ▼           ▼            ▼                       │
│         ┌────────┐ ┌──────────┐ ┌──────────┐                │
│         │ Apollo │ │ Supabase │ │ SendGrid │                │
│         │  (CRM) │ │(DB+RAG)  │ │ (Email)  │                │
│         └────────┘ └──────────┘ └──────────┘                │
│                                                               │
│  ┌──────────────────────────────────────────────────────┐    │
│  │              SOVEREIGN COCKPIT (Streamlit)            │    │
│  │   New Business │ Approvals │ Agent Command Center     │    │
│  └──────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
```

**Data flows top-down:** A business idea enters through the Genesis Engine, gets turned into configs, agents launch in shadow mode, and the pipeline processes leads through a 10-node LangGraph workflow.

---

## 4. The Genesis Engine — Launch a New Business

The Genesis Engine is the meta-layer that turns a business idea into a fully configured, running agent fleet. It replaces weeks of manual setup with a guided conversation.

### The Five-Gate Safety Model

Every new business goes through five gates before anything goes live:

| Gate | What Happens | Safety Check |
|------|-------------|--------------|
| **1. Interview** | AI asks adaptive questions about your business | Validates understanding before proceeding |
| **2. Blueprint Review** | AI generates a strategic blueprint | You approve the strategy before configs |
| **3. Config Validation** | Pydantic validates all generated YAML | No invalid configs can exist |
| **4. Credential Collection** | You provide API keys (encrypted at rest) | All required credentials verified |
| **5. Shadow Mode** | Agents launch but nothing goes out for real | Mandatory — no opt-out for new verticals |

### Using Genesis via the Dashboard

1. Navigate to **New Business** in the sidebar
2. Describe your business idea (minimum 10 characters)
3. Answer the AI interviewer's questions (typically 5-8 questions)
4. Review the generated blueprint — approve or request changes
5. Review the generated agent configurations
6. Provide required API credentials
7. Launch! Your agents start in shadow mode

### Using Genesis via CLI

```bash
# Run pre-flight checks (config + agents + credentials)
python main.py genesis preflight my_vertical

# Launch a vertical in shadow mode
python main.py genesis start my_vertical

# Check launch status
python main.py genesis status my_vertical

# View credential status
python main.py genesis credentials my_vertical
```

### The Blueprint

A `BusinessBlueprint` captures everything about your business:

- **Business Context** — Name, model, price range, target market
- **ICP (Ideal Customer Profile)** — Company size, industries, signals, disqualifiers
- **Personas** — Buyer personas with title patterns and outreach approaches
- **Outreach Strategy** — Email sequences, daily limits, compliance jurisdictions
- **Agent Fleet** — Which agents to deploy and their configurations
- **Integrations** — External services needed (Apollo, CRM, etc.)

### Shadow Mode

When a vertical launches, all agents run in **shadow mode**:

- Emails are drafted but **never sent**
- API calls to external services are logged but **responses are simulated**
- All decisions are recorded for your review
- You can see exactly what the system *would have done*

Promotion from shadow to live requires manual action via the dashboard. This cannot be automated — it's a deliberate safety choice.

---

## 5. CLI Reference

The CLI uses [Typer](https://typer.tiangolo.com/) with Rich console output.

### Pipeline Commands

```bash
# Validate configuration
python main.py validate <vertical_id>

# Run the full pipeline
python main.py run <vertical_id> [OPTIONS]
  --leads, -n    Number of leads to process (default: 10)
  --verbose, -v  Show detailed output including draft emails
  --test         Use mock Apollo data, simulate sending

# Test run with mock data
python main.py test-run <vertical_id> [OPTIONS]
  --leads, -n    Number of mock leads (default: 3)
  --verbose, -v  Show draft emails
```

### Dashboard

```bash
python main.py dashboard [vertical_id] [OPTIONS]
  --port    Streamlit server port (default: 8501)
```

### Agent Commands

```bash
# List all agents for a vertical
python main.py agent list <vertical_id>

# Run a specific agent
python main.py agent run <vertical_id> <agent_id> [OPTIONS]
  --leads, -n    Number of leads for outreach agents (default: 10)
  --dry-run      Simulate without side effects

# Show agent run statistics (last 7 days)
python main.py agent status <vertical_id>
```

### Genesis Commands

```bash
# Launch a vertical in shadow mode
python main.py genesis start <vertical_id> [OPTIONS]
  --output-dir          Base directory (default: "verticals")
  --skip-credentials    Skip credential validation (testing only)

# Pre-flight checks only (no launch)
python main.py genesis preflight <vertical_id> [OPTIONS]
  --output-dir          Base directory (default: "verticals")

# Check launch status
python main.py genesis status <vertical_id>

# Show credential status
python main.py genesis credentials <vertical_id>
```

---

## 6. The Sovereign Cockpit (Dashboard)

The dashboard is a Streamlit multi-page application for monitoring and controlling your agent fleet.

### Accessing the Dashboard

```bash
python main.py dashboard
# Opens at http://localhost:8501
```

Authentication is enforced — you need a valid `DASHBOARD_PASSWORD` environment variable or Supabase auth configured.

### Pages

#### New Business (Genesis Onboarding)

A five-stage wizard for launching a new business vertical:

1. **Welcome** — Describe your business idea
2. **Interview** — Chat-based Q&A with the ArchitectAgent
3. **Blueprint Review** — Approve or revise the strategic blueprint
4. **Config Review** — Inspect generated YAML configs
5. **Credentials** — Enter API keys (encrypted with Fernet)
6. **Launched** — Monitor shadow mode activity

#### Approval Queue

Human-in-the-loop review for all agent-generated content:

- **Pending Review** — Items waiting for your approval
- **All Content** — Full content history with status filters
- **Task Queue** — Background task monitoring

For each item, you can:
- **Approve** — Send it as-is
- **Edit** — Modify the content (captured as RLHF training data)
- **Reject** — Send it back for re-drafting
- **Skip** — Drop it from the pipeline

Every edit you make becomes a training example for improving future drafts.

#### Agent Command Center

Operational control panel for all agents:

- **Status cards** — Green (active), Red (paused), Ghost (shadow), Orange (circuit breaker)
- **Per-agent controls** — Pause, Resume, Toggle Shadow Mode, Reset Errors
- **Performance metrics** — Total runs, success rate, avg duration, failed runs
- **Run history** — Last 10 runs with status, duration, and error details
- **Emergency controls** — Pause ALL / Resume ALL agents

### Sidebar

The sidebar provides:
- **Vertical selector** — Switch between business verticals (auto-discovered from filesystem)
- **Emergency controls** — On the Agent Command page
- **Version info** — Platform version display

---

## 7. The Sales Pipeline

The core pipeline is a 10-node LangGraph state machine that processes leads from discovery to outreach.

### Pipeline Flow

```
                    ┌─────────────────┐
                    │  Apollo Search   │  (Lead sourcing)
                    └────────┬────────┘
                             ▼
                    ┌─────────────────┐
              ┌─ NO │ Check Duplicate │ (90-day cooldown)
              │     └────────┬────────┘
              │              ▼ NOT DUPLICATE
              │     ┌─────────────────┐
              │     │ Enrich Company  │ (Tech stack, funding, size)
              │     └────────┬────────┘
              │              ▼
              │     ┌─────────────────┐
              │     │ Enrich Contact  │ (Upsert to database)
              │     └────────┬────────┘
              │              ▼
              │     ┌─────────────────┐
              ├─ NO │  Qualify Lead   │ (Score 0-100%, threshold 30%)
              │     └────────┬────────┘
              │              ▼ QUALIFIED
              │     ┌─────────────────┐
              │     │ Select Strategy │ (Persona + RAG patterns)
              │     └────────┬────────┘
              │              ▼
              │     ┌─────────────────┐
              │     │ Draft Outreach  │ (Claude AI personalization)
              │     └────────┬────────┘
              │              ▼
              │     ┌─────────────────┐
              ├─ NO │Compliance Check │ (CAN-SPAM, GDPR, suppression)
              │     └────────┬────────┘
              │              ▼ PASSED
              │     ┌─────────────────┐
              │     │  Human Review   │ ◄── INTERRUPT (requires approval)
              │     └────────┬────────┘
              │         ▼         ▼
              │     APPROVED   REJECTED ──→ Re-draft (max 3x)
              │         ▼
              │     ┌─────────────────┐
              │     │ Send Outreach   │ (Email dispatch)
              │     └────────┬────────┘
              │              ▼
              └─────►┌─────────────────┐
                     │ Write to RAG    │ (Knowledge loop)
                     └─────────────────┘
```

### Lead Qualification Scoring

Leads are scored on a 0-100% scale across four dimensions:

| Signal | Weight | Criteria |
|--------|--------|----------|
| Company Size | 30% | Falls within ICP range |
| Industry Match | 30% | Matches target industries |
| Buying Signals | 30% | Recent funding, job postings, tech adoption |
| Title Match | 10% | Contact title matches persona patterns |

Leads scoring below 30% or hitting any disqualifier are automatically skipped.

### Strategy Selection

The strategy engine uses:
1. **Persona matching** — Match the contact's title to configured personas
2. **RAG winning patterns** — Query the knowledge base for successful templates
3. **Override logic** — If a RAG pattern has >15% win rate, it overrides the default

### Human Review

Every outreach email requires human approval before sending:

- Pipeline pauses (LangGraph `interrupt_before`)
- State is persisted to a SQLite checkpoint
- You review via the Dashboard or CLI
- Your edits become RLHF training data

You can configure `auto_approve_threshold` to skip review for high-confidence drafts, but this is disabled by default.

---

## 8. Agent System

### Agent Types

| Agent Type | Purpose | Status |
|-----------|---------|--------|
| `outreach` | Lead generation and email outreach | Active |
| `seo_content` | Blog/content generation with SEO | Active |
| `appointment_setter` | Reply handling and meeting scheduling | Active |
| `janitor` | Database cleanup and maintenance | Active |
| `architect` | Genesis Engine meta-agent (interview + blueprint) | Internal |

### Agent Lifecycle

```
Registration (@register_agent_type decorator)
    ▼
Discovery (AgentRegistry scans verticals/{id}/agents/*.yaml)
    ▼
Instantiation (Config → Agent instance with DB, LLM, tools)
    ▼
Execution (Neural Router → Security Airlock → Graph → Refinement)
    ▼
Learning (RLHF data collection, RAG knowledge writing)
```

### Safety Features

- **Neural Router** — Cheap classifier before expensive LLM (intent-based routing)
- **Security Airlock** — Scans tasks for prompt injection before processing
- **Circuit Breaker** — Auto-disables agent after 5 consecutive errors
- **Refinement Loop** — Agent self-critiques output quality
- **Confidence Gating** — Only high-confidence insights written to RAG
- **Shadow Mode** — A/B testing without real side effects
- **Feature Flags** — Dynamic runtime control with percentage rollouts

### Observability

Every agent run is tracked with:
- Run ID, status, duration, error messages
- LangFuse distributed tracing (spans for each pipeline stage)
- Agent-level metrics (success rate, avg duration, failed runs)

---

## 9. Configuration

### Vertical Config (`verticals/{id}/config.yaml`)

This is the master configuration for a business vertical. Key sections:

```yaml
vertical_id: enclave_guard       # snake_case identifier
vertical_name: Enclave Guard     # Human-readable name
industry: Cybersecurity Consulting

business:
  ticket_range: [5000, 15000]    # Deal size in USD
  currency: USD
  sales_cycle_days: 30

targeting:
  ideal_customer_profile:
    company_size: [51, 500]       # Employee count range
    industries: [technology, finance, healthcare]
    signals: [recent_funding, hiring_security, compliance_mention]
    disqualifiers: [government, defense_contractor]
  personas:
    - id: cto
      title_patterns: [CTO, Chief Technology Officer]
      company_size: [51, 500]
      approach: initial_outreach
      seniorities: [c_suite, vp]

outreach:
  email:
    daily_limit: 25
    warmup_days: 14
    sending_domain: mail.example.com
    reply_to: hello@example.com
    sequences:
      - name: initial_outreach
        steps: 3
        delay_days: [0, 3, 7]
  compliance:
    jurisdictions: [US_CAN_SPAM]
    physical_address: "123 Main St, Austin TX"
    unsubscribe_mechanism: one_click

apollo:
  filters:
    person_titles: [CTO, VP Engineering, CISO]
    person_seniorities: [c_suite, vp, director]
    organization_num_employees_ranges: ["51,200", "201,500"]
    q_organization_keyword_tags: [cybersecurity, infosec]

pipeline:
  duplicate_cooldown_days: 90
  max_retries_per_node: 3
  human_review_required: true

rag:
  learning_threshold: 100        # Events before learning loop activates
```

### Agent Config (`verticals/{id}/agents/{agent_id}.yaml`)

Individual agent configuration:

```yaml
agent_id: outreach_v1
agent_type: outreach
name: Lead Generation Agent
description: Outbound prospecting for cybersecurity consulting
enabled: true

model:
  provider: anthropic
  model: claude-sonnet-4-20250514
  temperature: 0.7
  max_tokens: 4096

human_gates:
  enabled: true
  gate_before: [send_outreach]
  auto_approve_threshold: null    # null = always require review

schedule:
  trigger: manual                 # manual | scheduled | event
  cron: null

shadow_mode: true                 # Start in shadow mode
max_consecutive_errors: 5         # Circuit breaker threshold
```

---

## 10. Credentials and Security

### Master Key

All credentials are encrypted using Fernet symmetric encryption. The encryption key is derived from `ENCLAVE_MASTER_KEY`:

```
ENCLAVE_MASTER_KEY → SHA-256 → Base64 → Fernet Key
```

**Set your master key** (use a strong, random string):
```bash
export ENCLAVE_MASTER_KEY="your-very-long-random-secret-key-here"
```

### Credential Storage

- Credentials are **never stored in plaintext** — not in memory, not on disk, not in the database
- Each credential is independently encrypted before storage
- The encrypted value is stored in Supabase with vertical scoping
- Falls back to in-memory storage if the database is unavailable

### Required Credentials

| Credential | Env Variable | Purpose |
|-----------|-------------|---------|
| Anthropic API Key | `ANTHROPIC_API_KEY` | AI agent decisions (Claude) |
| Supabase URL | `SUPABASE_URL` | Database connection |
| Supabase Service Key | `SUPABASE_SERVICE_KEY` | Database authentication |
| OpenAI API Key | `OPENAI_API_KEY` | Text embeddings for RAG |
| Apollo API Key | `APOLLO_API_KEY` | Lead sourcing and enrichment |

### Optional Credentials

| Credential | Env Variable | Purpose |
|-----------|-------------|---------|
| SendGrid API Key | `SENDGRID_API_KEY` | Email delivery |
| Mailgun API Key | `MAILGUN_API_KEY` | Alternative email delivery |
| Telegram Bot Token | `TELEGRAM_BOT_TOKEN` | Genesis lifecycle notifications |
| Dashboard Password | `DASHBOARD_PASSWORD` | Dashboard authentication |

### Checking Credential Status

```bash
python main.py genesis credentials <vertical_id>
```

Output:
```
┌──────────────────────────────────────────────────────┐
│               Credentials — enclave_guard            │
├────────────────────┬──────────────┬──────────┬───────┤
│ Env Variable       │ Name         │ Required │ Status│
├────────────────────┼──────────────┼──────────┼───────┤
│ ANTHROPIC_API_KEY  │ Anthropic    │ required │ ✓ set │
│ SUPABASE_URL       │ Supabase URL │ required │ ✓ set │
│ APOLLO_API_KEY     │ Apollo       │ required │ ✗ miss│
└────────────────────┴──────────────┴──────────┴───────┘
```

---

## 11. Compliance

The platform enforces outreach compliance across multiple jurisdictions.

### Supported Jurisdictions

| Jurisdiction | Law | Key Requirements |
|-------------|-----|-----------------|
| US | CAN-SPAM | Unsubscribe mechanism, physical address, honest subject lines |
| EU | GDPR | Prior consent required, data processing records |
| UK | PECR | Similar to GDPR, explicit opt-in |
| Canada | CASL | Express or implied consent required |

### How It Works

Before any email is sent, the compliance checker validates:

1. **Suppression list** — Is the recipient on the do-not-contact list?
2. **Jurisdiction detection** — What laws apply? (Based on TLD and config)
3. **Required elements** — Unsubscribe link, physical address, valid sender
4. **Country exclusions** — Is the recipient in an excluded country?

If any check fails, the email is blocked and the reason is logged.

### Configuration

```yaml
outreach:
  compliance:
    jurisdictions: [US_CAN_SPAM]
    unsubscribe_mechanism: one_click    # RFC 8058
    physical_address: "123 Main St, Austin TX 78701"
    suppress_list_path: suppressions.csv
    exclude_countries: [CN, RU]
```

---

## 12. RAG Knowledge Base

The RAG (Retrieval-Augmented Generation) system is the platform's learning engine. It improves over time as you process more leads.

### How It Works

1. **Vector storage** — Documents are embedded and stored in Supabase pgvector
2. **Hybrid search** — Combines vector similarity with metadata filters
3. **Learning loop** — Activates after 100 outreach events
4. **Confidence gating** — Only insights above 0.7 confidence are stored

### Knowledge Types

| Chunk Type | What It Stores |
|-----------|---------------|
| `company_intel` | Company research summaries |
| `outreach_result` | Email outcomes (opened, replied, converted) |
| `winning_pattern` | High-performing email templates |
| `vulnerability_knowledge` | Security findings for enrichment |
| `industry_insight` | Market trends and data |
| `objection_handling` | Rebuttals for common objections |

### The Learning Flywheel

```
Send Email → Track Response → Store Outcome
    ▲                              │
    │                              ▼
    │                    Identify Winning Patterns
    │                              │
    │                              ▼
    └──── Use Patterns in Future Drafts
```

Every human edit in the approval queue also becomes an RLHF training example:
- The original AI draft (what the model produced)
- Your edited version (what you wanted)
- This pair is stored for future fine-tuning

---

## 13. Adding a New Vertical Manually

While the Genesis Engine automates this, you can also create verticals by hand.

### Directory Structure

```
verticals/
└── my_business/
    ├── config.yaml              # Vertical configuration
    ├── agents/
    │   ├── outreach_v1.yaml     # Outreach agent config
    │   └── seo_v1.yaml          # SEO content agent config
    └── prompts/
        └── agent_prompts/
            └── outreach_v1.txt  # System prompt for the agent
```

### Steps

1. Create the directory: `mkdir -p verticals/my_business/agents`
2. Copy and modify `verticals/enclave_guard/config.yaml`
3. Create agent YAML files in the `agents/` subdirectory
4. Validate: `python main.py validate my_business`
5. Pre-flight: `python main.py genesis preflight my_business`
6. Launch: `python main.py genesis start my_business`

### Validation Rules

- `vertical_id` must be snake_case: `^[a-z][a-z0-9_]*$`
- At least one outreach agent is required
- All persona approaches must match defined email sequences
- Company size ranges must be valid tuples (min <= max)
- Price ranges must be non-negative

---

## 14. Troubleshooting

### Common Issues

**"No agents found for vertical"**
- Ensure YAML files exist in `verticals/{id}/agents/`
- Check YAML syntax: `python -c "import yaml; yaml.safe_load(open('path/to/file.yaml'))"`
- Verify `agent_type` matches a registered implementation

**"Unregistered agent types"**
- Available types: `outreach`, `seo_content`, `appointment_setter`, `janitor`
- Ensure the implementation module is imported (check `main.py` imports)

**"Config validation failed"**
- Run `python main.py validate <vertical_id>` for detailed error messages
- Common issues: missing required fields, invalid ranges, wrong field types

**"Missing required credentials"**
- Run `python main.py genesis credentials <vertical_id>`
- Set missing env vars in your `.env` file
- Ensure `ENCLAVE_MASTER_KEY` is set for encrypted credential storage

**Pipeline stuck at human review**
- This is by design — emails require approval before sending
- Open the dashboard (Approval Queue) or check the CLI output
- The state is persisted; you can resume at any time

**Circuit breaker tripped**
- Agent auto-disabled after 5 consecutive errors
- Check the Agent Command Center for error details
- Fix the underlying issue, then click "Reset Errors"

### Logs

The platform uses structured logging via Python's `logging` module:

```bash
# Run with debug logging
LOG_LEVEL=DEBUG python main.py run enclave_guard --test
```

### Database

Run migrations against your Supabase instance:

```sql
-- Located in infrastructure/migrations/
-- Apply in order: 001_core.sql, 002_rag.sql, etc.
```

---

## 15. Glossary

| Term | Definition |
|------|-----------|
| **Vertical** | A business configuration (e.g., "Enclave Guard" for cybersecurity consulting) |
| **Blueprint** | The strategic specification generated by the ArchitectAgent |
| **ICP** | Ideal Customer Profile — defines who you're selling to |
| **Persona** | A buyer archetype (e.g., "CTO at a 50-200 person tech company") |
| **Shadow Mode** | Sandboxed execution — all actions logged but no real effects |
| **Genesis Engine** | The meta-system that generates new verticals from interviews |
| **Circuit Breaker** | Auto-disables an agent after too many consecutive errors |
| **RLHF** | Reinforcement Learning from Human Feedback — learns from your edits |
| **RAG** | Retrieval-Augmented Generation — the shared knowledge base |
| **Winning Pattern** | A high-performing email template identified from past results |
| **Suppression List** | Contacts who should never be emailed |
| **Human Gate** | A pipeline node that requires human approval before proceeding |
| **Preflight** | Pre-launch validation (config, agents, credentials) |
| **LangGraph** | The state machine framework powering the pipeline |
| **pgvector** | PostgreSQL extension for vector similarity search (RAG storage) |

---

## Notifications

The Genesis Engine sends Telegram notifications at key lifecycle events:

| Event | When It Fires |
|-------|--------------|
| Interview Complete | After the adaptive Q&A finishes |
| Blueprint Ready | When a blueprint is generated for review |
| Blueprint Approved | When you approve the strategic plan |
| Configs Generated | When YAML files are created |
| Launch Success | When agents come online in shadow mode |
| Launch Failed | When pre-flight or launch fails |
| Promoted to Live | When shadow mode is turned off |

To enable notifications, set:
```env
TELEGRAM_BOT_TOKEN=your-bot-token
TELEGRAM_CHAT_ID=your-chat-id
```

---

*Built with Claude AI, LangGraph, Supabase, and Streamlit.*
*1,102 tests. Zero emails sent without your approval.*
