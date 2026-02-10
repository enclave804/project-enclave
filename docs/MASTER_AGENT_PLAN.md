# Master Agent Ecosystem Plan
## Sovereign Venture Engine — Complete Automation Blueprint

**Generated:** 2026-02-09
**Status:** Planning — Awaiting Approval
**Current Stats:** 17 agent types | 2,823 tests | 2 verticals

---

## 1. CURRENT STATE

### Registered Agent Types (17 — Universal Business Layer)

| # | Agent Type | Code File | Purpose |
|---|-----------|-----------|---------|
| 1 | `outreach` | outreach_agent.py | Lead gen + cold email sequences |
| 2 | `seo_content` | seo_content_agent.py | Blog/article generation for SEO |
| 3 | `appointment_setter` | appointment_agent.py | Books discovery calls |
| 4 | `maintenance` | maintenance_agent.py | System health + data janitor |
| 5 | `architect` | architect_agent.py | Genesis Engine — spawns new verticals |
| 6 | `overseer` | overseer_agent.py | Cross-agent orchestrator |
| 7 | `commerce` | commerce_agent.py | Shopify/Stripe storefront ops |
| 8 | `voice` | voice_agent.py | Twilio SMS/calls |
| 9 | `proposal_builder` | proposal_agent.py | SOW/proposal generation |
| 10 | `social` | social_agent.py | Social media posting/monitoring |
| 11 | `ads_strategy` | ads_agent.py | Ad campaign management |
| 12 | `finance` | finance_agent.py | Invoicing + A/R management |
| 13 | `cs` | cs_agent.py | Customer success + onboarding |
| 14 | `autopilot` | autopilot_agent.py | Autonomous execution scheduler |
| 15 | `followup` | followup_agent.py | Multi-touch follow-up sequences |
| 16 | `meeting_scheduler` | meeting_agent.py | Calendar booking + invites |
| 17 | `sales_pipeline` | pipeline_agent.py | Deal stage tracking + forecasting |

### Vertical Deployment Status

| Vertical | YAML Configs | Agent Coverage | Domain Experts |
|----------|-------------|----------------|----------------|
| **Enclave Guard** (Cybersecurity) | 13 of 17 | 76% | 0 |
| **PrintBiz** (3D Printing) | 1 of 17 | 6% | 0 |

**Gap:** Both verticals have ZERO domain expert agents. The platform automates business operations but doesn't deliver the actual service expertise.

---

## 2. THE THREE AGENT LAYERS

```
┌─────────────────────────────────────────────────────────────┐
│ LAYER 3: DOMAIN EXPERT AGENTS (service delivery)            │
│ Cybersecurity analysts, 3D print engineers, CAD advisors    │
│ These are the PRODUCT — what clients pay for                │
├─────────────────────────────────────────────────────────────┤
│ LAYER 2: UNIVERSAL BUSINESS AGENTS (ops + revenue)          │
│ Outreach, Pipeline, Proposals, Invoicing, Follow-up, etc.   │
│ These are the ENGINE — how the business runs                │
├─────────────────────────────────────────────────────────────┤
│ LAYER 1: PLATFORM AGENTS (infrastructure)                   │
│ Overseer, Autopilot, Maintenance, Architect                 │
│ These are the FOUNDATION — keeps everything running         │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. ENCLAVE GUARD — CYBERSECURITY VERTICAL

### 3A. Business Agents Deployed (13/17)

Already deployed:
- outreach, seo_content, appointment_setter, autopilot, cs, finance,
  followup, maintenance, meeting_scheduler, pipeline, proposal_builder,
  social, ads_strategy (via janitor.yaml for maintenance)

Missing YAML configs (need to add):
| Agent Type | Priority | Why |
|-----------|----------|-----|
| `commerce` | Low | B2B consulting doesn't use storefront |
| `voice` | Medium | Could use for appointment reminders |
| `overseer` | High | Needed for multi-agent orchestration |
| `architect` | Low | Already launched, not needed daily |

### 3B. Domain Expert Agents (NEW — 10 agents)

These agents deliver the actual cybersecurity consulting service. They scan, analyze, assess, and produce the deliverables that clients pay for.

| # | Agent Type | Name | Purpose | Trigger |
|---|-----------|------|---------|---------|
| 1 | `vuln_scanner` | **Vulnerability Assessment Agent** | Runs Shodan/SSL Labs/header scans, produces vulnerability reports, scores risk severity | Event: `new_client_onboarded` or Scheduled: weekly |
| 2 | `network_analyst` | **Network Architecture Analyst** | Maps exposed services, analyzes network topology from public data, identifies attack surface | Event: `vuln_scan_complete` |
| 3 | `iam_analyst` | **IAM & Access Control Analyst** | Evaluates identity/access policies from questionnaire data, flags over-permissioned accounts, MFA gaps | Event: `questionnaire_received` |
| 4 | `compliance_mapper` | **Policy & Compliance Mapper** | Maps current posture to SOC2/HIPAA/PCI/ISO27001 frameworks, identifies gaps, generates compliance roadmap | Scheduled: per engagement |
| 5 | `cloud_security` | **Cloud Security Posture Agent** | Analyzes cloud configs (from client-shared data), checks S3 buckets, IAM roles, security groups | Event: `cloud_access_granted` |
| 6 | `appsec_reviewer` | **Application Security Reviewer** | Reviews web app headers, CSP policies, cookie flags, API exposure, OWASP Top 10 checks | Event: `new_client_onboarded` |
| 7 | `incident_readiness` | **Incident Response Readiness Agent** | Evaluates IR plans, backup strategies, recovery procedures, produces readiness score | Event: `assessment_started` |
| 8 | `security_trainer` | **Security Awareness & Human Risk Agent** | Generates phishing simulation scenarios, security training content, risk scoring for human factors | Scheduled: monthly per client |
| 9 | `risk_reporter` | **Risk Quantification & Executive Reporting** | Aggregates all assessment data into executive-level reports, quantifies risk in dollar terms, produces board-ready decks | Event: `all_assessments_complete` |
| 10 | `remediation_guide` | **Remediation Guidance & Verification Agent** | Produces step-by-step fix instructions for each finding, tracks remediation progress, runs re-verification scans | Event: `report_delivered` |

**Data flow:**
```
New Client → vuln_scanner → network_analyst → appsec_reviewer
                                                    ↓
          cloud_security ← (if cloud) ← iam_analyst ← questionnaire
                                                    ↓
          incident_readiness → compliance_mapper → risk_reporter
                                                    ↓
          security_trainer (ongoing) ← remediation_guide (tracks fixes)
```

**Key tools each agent needs:**
- `vuln_scanner`: shodan_tools, ssl_tools, http_scanner, supabase_tools
- `network_analyst`: shodan_tools, dns_tools, supabase_tools
- `iam_analyst`: questionnaire_tools, supabase_tools, llm
- `compliance_mapper`: framework_db (SOC2/HIPAA/PCI templates), supabase_tools, llm
- `cloud_security`: cloud_scanner_tools (read-only AWS/GCP/Azure checks), supabase_tools
- `appsec_reviewer`: http_scanner, header_analyzer, supabase_tools
- `incident_readiness`: questionnaire_tools, scoring_engine, supabase_tools
- `security_trainer`: content_generator, phishing_sim_tools, supabase_tools
- `risk_reporter`: report_generator, chart_tools, supabase_tools, email_tools
- `remediation_guide`: ticket_tracker, verification_scanner, supabase_tools

---

## 4. PRINTBIZ — 3D PRINTING VERTICAL

### 4A. Business Agents to Deploy (12 YAML configs needed)

PrintBiz currently has only `outreach`. It needs YAML configs for:

| # | Agent Type | Priority | PrintBiz Customization |
|---|-----------|----------|----------------------|
| 1 | `seo_content` | High | Architecture 3D printing blog posts, case studies |
| 2 | `appointment_setter` | High | Book design review calls |
| 3 | `followup` | High | Follow up after quote requests |
| 4 | `meeting_scheduler` | High | Schedule design consultations |
| 5 | `sales_pipeline` | High | Track from inquiry → quote → production → delivery |
| 6 | `proposal_builder` | High | Generate custom quotes with material/size/timeline |
| 7 | `finance` | High | Invoice management for print jobs |
| 8 | `cs` | High | Post-delivery check-in, reorder prompts |
| 9 | `social` | Medium | Showcase finished prints, before/after |
| 10 | `ads_strategy` | Medium | Target architects on LinkedIn/Instagram |
| 11 | `commerce` | Medium | Online quoting/ordering portal |
| 12 | `autopilot` | Medium | Autonomous scheduling |
| 13 | `maintenance` | Low | System health |
| 14 | `voice` | Low | Delivery notifications |

### 4B. Domain Expert Agents (NEW — 10 agents)

These agents deliver the actual 3D printing service expertise. They analyze files, optimize prints, select materials, manage production, and ensure quality.

| # | Agent Type | Name | Purpose | Trigger |
|---|-----------|------|---------|---------|
| 1 | `file_analyst` | **File Intake & Geometry Analyst** | Validates STL/OBJ/STEP files, checks manifold integrity, wall thickness, overhangs, reports printability issues | Event: `file_uploaded` |
| 2 | `mesh_repair` | **Mesh Repair & Print Prep Agent** | Auto-repairs common mesh issues, orients parts for optimal printing, adds supports, generates print-ready files | Event: `geometry_analysis_complete` |
| 3 | `scale_optimizer` | **Scale & Detail Optimizer** | Ensures architectural models are correctly scaled, verifies detail resolution at target size, suggests scale adjustments | Event: `file_validated` |
| 4 | `material_advisor` | **Material & Technology Selector** | Recommends print technology (FDM/SLA/SLS/MJF) and material based on use case, budget, detail requirements, finish quality | Event: `project_requirements_received` |
| 5 | `print_manager` | **Print Farm & Slicing Manager** | Manages print queue across multiple printers, optimizes bed packing, estimates print times, monitors job status | Scheduled: every 30 min |
| 6 | `post_process` | **Post-Processing & Finishing Guide** | Recommends finishing steps (sanding, painting, coating, assembly), generates work orders for post-processing team | Event: `print_complete` |
| 7 | `qc_inspector` | **Quality Control & Inspection Agent** | Compares finished prints against specs, checks dimensional accuracy, surface quality, flags defects | Event: `post_processing_complete` |
| 8 | `quote_engine` | **Quoting & Project Estimator** | Calculates cost from material volume, print time, post-processing, generates detailed quotes with timeline | Event: `file_analyzed` OR `manual_quote_request` |
| 9 | `cad_advisor` | **CAD & Architectural Design Advisor** | Advises architects on design-for-3D-printing, suggests modifications for better printability, reviews designs before submission | Event: `design_consultation_booked` |
| 10 | `logistics` | **Packaging, Logistics & Delivery Agent** | Plans packaging for fragile models, generates shipping labels, tracks deliveries, manages returns | Event: `qc_passed` |

**Data flow:**
```
File Upload → file_analyst → mesh_repair → scale_optimizer
                                                ↓
                quote_engine ← material_advisor ← requirements
                    ↓
                (client approves quote)
                    ↓
                print_manager → (printing) → post_process → qc_inspector
                                                                ↓
                cad_advisor (parallel — consultations)       logistics
```

**Key tools each agent needs:**
- `file_analyst`: mesh_analysis_tools (trimesh/numpy), supabase_tools
- `mesh_repair`: mesh_repair_tools (trimesh auto-repair), file_tools, supabase_tools
- `scale_optimizer`: geometry_tools, measurement_tools, supabase_tools
- `material_advisor`: material_db, pricing_engine, supabase_tools, llm
- `print_manager`: printer_api_tools (OctoPrint/Bambu), queue_tools, supabase_tools
- `post_process`: workflow_tools, work_order_tools, supabase_tools
- `qc_inspector`: measurement_tools, image_analysis_tools, supabase_tools
- `quote_engine`: pricing_engine, material_db, supabase_tools, email_tools
- `cad_advisor`: cad_review_tools, printability_checker, supabase_tools, llm
- `logistics`: shipping_api_tools (EasyPost/ShipStation), supabase_tools

---

## 5. NEW UNIVERSAL BUSINESS AGENTS (4 new types)

These apply to ALL verticals:

| # | Agent Type | Name | Purpose |
|---|-----------|------|---------|
| 1 | `contract_manager` | **Contract & Agreement Agent** | Generates service agreements, MSAs, NDAs from templates; tracks signature status; manages renewals |
| 2 | `support_agent` | **Client Support Agent** | Handles inbound support questions, routes to domain experts or human, maintains FAQ knowledge base |
| 3 | `competitive_intel` | **Competitive Intelligence Agent** | Monitors competitor pricing, offerings, reviews; alerts on market changes; feeds insights to strategy |
| 4 | `reporting` | **Analytics & Reporting Agent** | Generates weekly/monthly business reports, KPI dashboards, revenue forecasting, churn analysis |

---

## 6. COMPLETE AGENT COUNT

### Summary by Layer

| Layer | Existing | New | Total |
|-------|----------|-----|-------|
| Platform (infra) | 4 | 0 | 4 |
| Universal Business | 13 | 4 | 17 |
| Enclave Guard Domain | 0 | 10 | 10 |
| PrintBiz Domain | 0 | 10 | 10 |
| **TOTAL** | **17** | **24** | **41** |

### Files to Create

**New agent implementations (24 files):**
```
core/agents/implementations/
├── vuln_scanner_agent.py          # Enclave Guard domain
├── network_analyst_agent.py
├── iam_analyst_agent.py
├── compliance_mapper_agent.py
├── cloud_security_agent.py
├── appsec_reviewer_agent.py
├── incident_readiness_agent.py
├── security_trainer_agent.py
├── risk_reporter_agent.py
├── remediation_guide_agent.py
├── file_analyst_agent.py          # PrintBiz domain
├── mesh_repair_agent.py
├── scale_optimizer_agent.py
├── material_advisor_agent.py
├── print_manager_agent.py
├── post_process_agent.py
├── qc_inspector_agent.py
├── quote_engine_agent.py
├── cad_advisor_agent.py
├── logistics_agent.py
├── contract_manager_agent.py      # Universal business
├── support_agent.py
├── competitive_intel_agent.py
└── reporting_agent.py
```

**New YAML configs (26 files):**
```
verticals/enclave_guard/agents/
├── vuln_scanner.yaml              # 10 domain expert configs
├── network_analyst.yaml
├── iam_analyst.yaml
├── compliance_mapper.yaml
├── cloud_security.yaml
├── appsec_reviewer.yaml
├── incident_readiness.yaml
├── security_trainer.yaml
├── risk_reporter.yaml
├── remediation_guide.yaml
├── overseer.yaml                  # Missing business agent
├── contract_manager.yaml          # 4 new universal
├── support.yaml
├── competitive_intel.yaml
└── reporting.yaml

verticals/print_biz/agents/
├── seo_content.yaml               # 12 business agent configs
├── appointment_setter.yaml
├── followup.yaml
├── meeting_scheduler.yaml
├── pipeline.yaml
├── proposal_builder.yaml
├── finance.yaml
├── cs.yaml
├── social.yaml
├── ads_strategy.yaml
├── commerce.yaml
├── autopilot.yaml
├── file_analyst.yaml              # 10 domain expert configs
├── mesh_repair.yaml
├── scale_optimizer.yaml
├── material_advisor.yaml
├── print_manager.yaml
├── post_process.yaml
├── qc_inspector.yaml
├── quote_engine.yaml
├── cad_advisor.yaml
├── logistics.yaml
├── contract_manager.yaml          # 4 new universal
├── support.yaml
├── competitive_intel.yaml
└── reporting.yaml
```

**New state schemas:** 24 new TypedDicts in `core/agents/state.py`

**New MCP tool modules:**
```
core/mcp/tools/
├── mesh_analysis_tools.py         # trimesh geometry analysis
├── mesh_repair_tools.py           # trimesh auto-repair
├── material_db_tools.py           # material/pricing database
├── printer_api_tools.py           # OctoPrint/Bambu API
├── shipping_tools.py              # EasyPost/ShipStation
├── ssl_scan_tools.py              # SSL Labs integration
├── dns_tools.py                   # DNS enumeration
├── http_scanner_tools.py          # Header/CSP analysis
├── compliance_framework_tools.py  # SOC2/HIPAA/PCI templates
├── report_generator_tools.py      # PDF/deck generation
├── questionnaire_tools.py         # Client assessment forms
└── contract_tools.py              # Agreement templates
```

**New tests:** ~600 tests across 24 test files (~25 each)

**New migration:**
- `014_domain_agents.sql` — Tables for assessments, print_jobs, qc_reports, contracts, support_tickets

---

## 7. BUILD ORDER (Prioritized by Revenue Impact)

### Phase 17: PrintBiz Business Stack (YAML configs only — no new code)
**Effort:** 1 session | **Impact:** PrintBiz goes from 6% → 80% coverage
- Create 12 YAML configs for PrintBiz using existing agent types
- Add PrintBiz prompts, email templates, knowledge seed data
- **Result:** PrintBiz can run full sales pipeline autonomously

### Phase 18: Enclave Guard Domain Experts (Batch 1 — Scan & Assess)
**Effort:** 2-3 sessions | **Impact:** Automated service delivery begins
- `vuln_scanner` — The foundation. Everything starts here.
- `network_analyst` — Feeds into vulnerability context
- `appsec_reviewer` — Web app security checks
- `compliance_mapper` — Maps findings to frameworks
- Migration: `014_domain_agents.sql` (assessments table, findings table)
- New tools: ssl_scan_tools, http_scanner_tools, compliance_framework_tools
- ~100 tests

### Phase 19: Enclave Guard Domain Experts (Batch 2 — Report & Remediate)
**Effort:** 2-3 sessions | **Impact:** Full assessment → report → fix cycle
- `risk_reporter` — Executive reports from all assessment data
- `remediation_guide` — Step-by-step fix instructions
- `incident_readiness` — IR plan evaluation
- `iam_analyst` — Access control review
- New tools: report_generator_tools, questionnaire_tools
- ~100 tests

### Phase 20: Enclave Guard Domain Experts (Batch 3 — Advanced)
**Effort:** 2 sessions | **Impact:** Premium service tiers
- `cloud_security` — Cloud posture assessment
- `security_trainer` — Phishing simulations + training
- ~50 tests

### Phase 21: PrintBiz Domain Experts (Batch 1 — File → Quote)
**Effort:** 2-3 sessions | **Impact:** Automated quoting from file upload
- `file_analyst` — STL/OBJ validation and analysis
- `mesh_repair` — Auto-fix common issues
- `scale_optimizer` — Architectural model scaling
- `material_advisor` — Tech + material recommendation
- `quote_engine` — Automated pricing
- New tools: mesh_analysis_tools, mesh_repair_tools, material_db_tools
- Migration: extend 014 with print_jobs, materials tables
- ~125 tests

### Phase 22: PrintBiz Domain Experts (Batch 2 — Production → Delivery)
**Effort:** 2-3 sessions | **Impact:** Full production pipeline
- `print_manager` — Print farm queue management
- `post_process` — Finishing workflow
- `qc_inspector` — Quality checks
- `logistics` — Packaging + shipping
- New tools: printer_api_tools, shipping_tools
- ~100 tests

### Phase 23: PrintBiz Domain Expert (Batch 3 — Advisory)
**Effort:** 1 session | **Impact:** Premium design consulting
- `cad_advisor` — Design-for-manufacturing guidance
- ~25 tests

### Phase 24: Universal Business Agents
**Effort:** 2 sessions | **Impact:** Applies to ALL verticals
- `contract_manager` — Agreements + renewals
- `support_agent` — Client support routing
- `competitive_intel` — Market monitoring
- `reporting` — Business analytics
- New tools: contract_tools
- ~100 tests

---

## 8. PROJECTED FINAL STATE

| Metric | Current | After All Phases |
|--------|---------|-----------------|
| Agent Types | 17 | 41 |
| Enclave Guard Agents | 13 | 27 |
| PrintBiz Agents | 1 | 26 |
| Tests | 2,823 | ~3,423 |
| MCP Tool Modules | 10 | 22 |
| Dashboard Pages | 9 | 11+ |
| Migrations | 13 | 14+ |

**Fully automated lifecycle:**
```
ENCLAVE GUARD:
Lead → Enrich → Qualify → Outreach → Follow-Up → Meeting →
  Discovery → Vuln Scan → Network Scan → App Scan →
    Compliance Map → Risk Report → Proposal → Contract →
      Invoice → Remediation Guidance → Ongoing Monitoring →
        Customer Success → Renewal/Upsell

PRINTBIZ:
Lead → Enrich → Qualify → Outreach → Follow-Up → Meeting →
  Design Consultation → File Upload → Geometry Analysis →
    Mesh Repair → Scale Check → Material Selection → Quote →
      Proposal → Contract → Invoice → Print Queue →
        Production → Post-Process → QC → Ship → CS → Reorder
```

---

## 9. MODEL ROUTING STRATEGY

| Task Complexity | Model | Use Cases |
|----------------|-------|-----------|
| **Critical/Creative** | Claude Opus 4.6 | Risk reports, executive proposals, compliance mapping, CAD design advice |
| **Standard/Analytical** | Claude Sonnet 4.5 | Email drafting, deal analysis, vulnerability assessment, quoting |
| **Routine/Classification** | Claude Haiku 4.5 | Lead scoring, status checks, file validation, routing decisions |

---

*This plan is a living document. Each phase will be planned in detail before implementation begins.*
