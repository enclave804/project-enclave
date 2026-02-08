"""
Project Enclave - Main Entry Point

CLI for running the Sovereign Venture Engine pipeline.
Supports daily batch runs, single lead processing, and human review.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.config.loader import load_vertical_config, list_available_verticals
from core.config.schema import VerticalConfig

# Load environment (override=True to ensure .env values take precedence)
root_env = Path(__file__).parent / ".env"
infra_env = Path(__file__).parent / "infrastructure" / ".env"

if root_env.exists():
    load_dotenv(root_env, override=True)
elif infra_env.exists():
    load_dotenv(infra_env, override=True)
else:
    load_dotenv(override=True)

app = typer.Typer(
    name="enclave",
    help="Project Enclave - Sovereign Venture Engine",
)
console = Console()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("enclave")


def _get_config(vertical: str) -> VerticalConfig:
    """Load and return vertical config, with friendly error on failure."""
    try:
        return load_vertical_config(vertical)
    except FileNotFoundError:
        available = list_available_verticals()
        available_list = ", ".join(available) if available else "none found"
        console.print(Panel(
            f"[red]Vertical not found:[/] [bold]{vertical}[/]\n\n"
            f"Available verticals: [cyan]{available_list}[/]\n\n"
            f"To create a new vertical:\n"
            f"  [dim]mkdir -p verticals/{vertical}[/]\n"
            f"  [dim]# Add config.yaml based on an existing vertical[/]",
            title="⚠ Configuration Error",
            border_style="red",
        ))
        raise typer.Exit(code=1)


def _check_env_key(var_name: str, label: str, required: bool = True) -> str | None:
    """Check if an environment variable is set. Shows a friendly error if missing."""
    value = os.environ.get(var_name, "").strip()
    if not value:
        if required:
            console.print(Panel(
                f"[red]Missing required API key:[/] [bold]{var_name}[/]\n\n"
                f"This key is needed for: [cyan]{label}[/]\n\n"
                f"Set it in your .env file:\n"
                f"  [dim]{var_name}=your_key_here[/]",
                title="⚠ Configuration Error",
                border_style="red",
            ))
            raise typer.Exit(code=1)
        return None
    return value


def _init_components(config: VerticalConfig):
    """Initialize all pipeline components with friendly error messages."""
    from anthropic import Anthropic

    from core.integrations.apollo_client import ApolloClient
    from core.integrations.supabase_client import EnclaveDB
    from core.rag.embeddings import EmbeddingEngine

    # Pre-check all required keys before initializing anything
    _check_env_key("SUPABASE_URL", "Database connection")
    _check_env_key("SUPABASE_SERVICE_KEY", "Database authentication")
    _check_env_key(config.apollo.api_key_env, "Lead sourcing (Apollo.io)")
    _check_env_key("OPENAI_API_KEY", "Text embeddings (RAG)")
    _check_env_key("ANTHROPIC_API_KEY", "AI email drafting (Claude)")

    db = EnclaveDB(vertical_id=config.vertical_id)
    apollo = ApolloClient(api_key_env=config.apollo.api_key_env)
    embedder = EmbeddingEngine()
    anthropic_client = Anthropic()

    return db, apollo, embedder, anthropic_client


# =========================================================================
# Commands
# =========================================================================


@app.command()
def info():
    """Show available verticals and their status."""
    verticals = list_available_verticals()

    if not verticals:
        console.print("[yellow]No verticals found. Create one in verticals/[/]")
        return

    table = Table(title="Project Enclave - Available Verticals")
    table.add_column("Vertical ID", style="cyan")
    table.add_column("Name", style="white")
    table.add_column("Industry", style="green")
    table.add_column("Ticket Range", style="yellow")

    for v in verticals:
        try:
            cfg = load_vertical_config(v)
            table.add_row(
                cfg.vertical_id,
                cfg.vertical_name,
                cfg.industry,
                f"${cfg.business.ticket_range[0]:,}-${cfg.business.ticket_range[1]:,}",
            )
        except Exception as e:
            table.add_row(v, f"[red]Error: {e}[/]", "", "")

    console.print(table)


@app.command()
def validate(
    vertical: str = typer.Argument(
        ..., help="Vertical ID (e.g., 'enclave_guard')"
    ),
):
    """Validate a vertical's configuration."""
    try:
        config = _get_config(vertical)
        console.print(Panel(
            f"[green]Configuration valid![/]\n\n"
            f"Vertical: {config.vertical_name}\n"
            f"Industry: {config.industry}\n"
            f"Personas: {len(config.targeting.personas)}\n"
            f"Email sequences: {len(config.outreach.email.sequences)}\n"
            f"Daily limit: {config.outreach.email.daily_limit}\n"
            f"Enrichment sources: {len(config.enrichment.sources)}\n"
            f"RAG chunk types: {len(config.rag.chunk_types)}\n"
            f"Pipeline: human_review={'required' if config.pipeline.human_review_required else 'selective'}",
            title=f"Config: {vertical}",
        ))
    except Exception as e:
        console.print(f"[red]Validation failed:[/] {e}")
        raise typer.Exit(1)


@app.command()
def pull_leads(
    vertical: str = typer.Argument(..., help="Vertical ID"),
    count: int = typer.Option(25, help="Number of leads to pull"),
    page: int = typer.Option(1, help="Apollo search page"),
):
    """Pull leads from Apollo.io based on vertical config."""

    async def _run():
        config = _get_config(vertical)
        db, apollo, _, _ = _init_components(config)

        console.print(f"[cyan]Pulling {count} leads from Apollo...[/]")

        filters = config.apollo.filters.model_dump()
        filters["per_page"] = count

        leads = await apollo.search_and_parse(filters, page=page)

        table = Table(title=f"Leads Found: {len(leads)}")
        table.add_column("#", style="dim")
        table.add_column("Name", style="cyan")
        table.add_column("Title", style="white")
        table.add_column("Company", style="green")
        table.add_column("Email", style="yellow")
        table.add_column("Industry", style="blue")

        for i, lead in enumerate(leads, 1):
            c = lead["contact"]
            co = lead["company"]
            table.add_row(
                str(i),
                c.get("name", ""),
                c.get("title", ""),
                co.get("name", ""),
                c.get("email", ""),
                co.get("industry", ""),
            )

        console.print(table)
        return leads

    asyncio.run(_run())


@app.command()
def run(
    vertical: str = typer.Argument(..., help="Vertical ID"),
    count: int = typer.Option(10, help="Number of leads to process"),
    dry_run: bool = typer.Option(False, help="Don't send emails, just draft"),
):
    """Run the full sales pipeline for a vertical."""

    async def _run():
        config = _get_config(vertical)
        db, apollo, embedder, anthropic_client = _init_components(config)

        from core.graph.workflow_engine import build_pipeline_graph, process_batch

        console.print(Panel(
            f"[cyan]Starting pipeline for {config.vertical_name}[/]\n"
            f"Leads to process: {count}\n"
            f"Dry run: {dry_run}\n"
            f"Human review: {'required' if config.pipeline.human_review_required else 'selective'}",
            title="Pipeline Run",
        ))

        # Build the graph
        graph = build_pipeline_graph(
            config=config,
            db=db,
            apollo=apollo,
            embedder=embedder,
            anthropic_client=anthropic_client,
            test_mode=dry_run,
        )

        if dry_run:
            console.print("[yellow]⚠ DRY RUN: Emails will be drafted but NOT sent.[/]")

        # Pull leads
        console.print("[cyan]Pulling leads from Apollo...[/]")
        filters = config.apollo.filters.model_dump()
        filters["per_page"] = count
        leads = await apollo.search_and_parse(filters)

        if not leads:
            console.print("[yellow]No leads found matching criteria.[/]")
            return

        console.print(f"[green]Found {len(leads)} leads. Processing...[/]")

        # Process batch
        results = await process_batch(graph, leads, config.vertical_id)

        # Display results
        table = Table(title="Pipeline Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Count", style="white")

        table.add_row("Total Leads", str(results["total"]))
        table.add_row("Processed", str(results["processed"]))
        sent_label = "[green]Simulated Sends[/]" if dry_run else "[green]Emails Sent[/]"
        table.add_row(sent_label, str(results["sent"]))
        table.add_row("[yellow]Skipped[/]", str(results["skipped"]))
        table.add_row("[red]Errors[/]", str(results["errors"]))
        review_label = "[blue]Auto-Approved[/]" if dry_run else "[blue]Awaiting Review[/]"
        table.add_row(review_label, str(results["interrupted"]))

        console.print(table)

        # Show details
        if results.get("details"):
            detail_table = Table(title="Lead Details")
            detail_table.add_column("Email", style="cyan")
            detail_table.add_column("Company", style="green")
            detail_table.add_column("Status", style="white")

            for d in results["details"]:
                detail_table.add_row(
                    d.get("email", ""),
                    d.get("company", ""),
                    d.get("status", ""),
                )
            console.print(detail_table)

    asyncio.run(_run())


@app.command()
def review(
    vertical: str = typer.Argument(..., help="Vertical ID"),
):
    """Review pending outreach emails (human-in-the-loop)."""
    console.print(Panel(
        "[cyan]Human Review Interface[/]\n\n"
        "This will show pending email drafts for approval.\n"
        "You can: approve, edit, reject, or skip each email.",
        title="Review Mode",
    ))

    # TODO: Implement review interface that:
    # 1. Queries LangGraph checkpointer for interrupted runs
    # 2. Displays each pending email
    # 3. Captures human decision
    # 4. Resumes the graph with the decision

    console.print(
        "[yellow]Review interface coming in Phase 2. "
        "For now, the pipeline pauses at the human_review node "
        "and requires manual intervention.[/]"
    )


@app.command()
def stats(
    vertical: str = typer.Argument(..., help="Vertical ID"),
    days: int = typer.Option(30, help="Number of days to report on"),
):
    """Show outreach statistics for a vertical."""
    from core.integrations.supabase_client import EnclaveDB

    config = _get_config(vertical)
    db = EnclaveDB(vertical_id=config.vertical_id)

    stats = db.get_outreach_stats(days=days)

    if not stats:
        console.print("[yellow]No outreach data yet.[/]")
        return

    table = Table(title=f"Outreach Stats - Last {days} Days")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("Total Sent", str(stats.get("total_sent", 0)))
    table.add_row("Opened", str(stats.get("total_opened", 0)))
    table.add_row("Replied", str(stats.get("total_replied", 0)))
    table.add_row("Bounced", str(stats.get("total_bounced", 0)))
    table.add_row("Meetings Booked", str(stats.get("total_meetings", 0)))
    table.add_row("Open Rate", f"{stats.get('open_rate', 0):.1%}")
    table.add_row("Reply Rate", f"{stats.get('reply_rate', 0):.1%}")
    table.add_row("Bounce Rate", f"{stats.get('bounce_rate', 0):.1%}")
    table.add_row("Meeting Rate", f"{stats.get('meeting_rate', 0):.1%}")

    console.print(table)

    # Check if learning threshold is met
    total_events = db.count_outreach_events()
    threshold = config.rag.learning_threshold
    if total_events < threshold:
        console.print(
            f"\n[yellow]RAG learning loop inactive. "
            f"Need {threshold - total_events} more outreach events "
            f"({total_events}/{threshold}).[/]"
        )
    else:
        console.print(
            f"\n[green]RAG learning loop active! "
            f"({total_events}/{threshold} events)[/]"
        )


@app.command()
def seed_knowledge(
    vertical: str = typer.Argument(..., help="Vertical ID"),
):
    """Load seed knowledge data into the RAG database."""

    async def _run():
        from core.integrations.supabase_client import EnclaveDB
        from core.rag.embeddings import EmbeddingEngine
        from core.rag.ingestion import KnowledgeIngester

        config = _get_config(vertical)
        db = EnclaveDB(vertical_id=config.vertical_id)
        embedder = EmbeddingEngine()

        ingester = KnowledgeIngester(db=db, embedder=embedder)

        seed_path = config.rag.seed_data_path
        if not seed_path:
            console.print("[yellow]No seed data path configured.[/]")
            return

        # Resolve path relative to vertical directory
        full_path = (
            Path(__file__).parent
            / "verticals"
            / config.vertical_id
            / seed_path
        )

        console.print(f"[cyan]Loading seed data from {full_path}...[/]")
        count = await ingester.load_seed_data(full_path)
        console.print(f"[green]Loaded {count} knowledge chunks.[/]")

    asyncio.run(_run())


@app.command()
def scan(
    vertical: str = typer.Argument(..., help="Vertical ID"),
    domain: str = typer.Argument(..., help="Domain to scan"),
):
    """Scan a domain for security findings (Enclave Guard)."""

    async def _run():
        from verticals.enclave_guard.enrichment.tech_stack_scanner import (
            TechStackScanner,
        )

        scanner = TechStackScanner()
        console.print(f"[cyan]Scanning {domain}...[/]")

        results = await scanner.scan_domain(domain)

        # Tech stack
        if results["tech_stack"]:
            table = Table(title="Tech Stack")
            table.add_column("Technology", style="cyan")
            table.add_column("Details", style="white")
            for tech, detail in results["tech_stack"].items():
                table.add_row(tech, str(detail))
            console.print(table)

        # Vulnerabilities
        if results["vulnerabilities"]:
            table = Table(title="Security Findings")
            table.add_column("Severity", style="red")
            table.add_column("Type", style="cyan")
            table.add_column("Description", style="white")
            for vuln in results["vulnerabilities"]:
                severity = vuln.get("severity", "unknown")
                style = {
                    "critical": "bold red",
                    "high": "red",
                    "medium": "yellow",
                    "low": "green",
                }.get(severity, "white")
                table.add_row(
                    f"[{style}]{severity.upper()}[/]",
                    vuln.get("type", ""),
                    vuln.get("description", ""),
                )
            console.print(table)
        else:
            console.print("[green]No security findings detected.[/]")

        # Headers
        if results["headers_info"]:
            hi = results["headers_info"]
            if hi.get("missing_headers"):
                console.print(
                    f"\n[yellow]Missing security headers: "
                    f"{', '.join(hi['missing_headers'])}[/]"
                )
            if hi.get("present_headers"):
                console.print(
                    f"[green]Present security headers: "
                    f"{', '.join(hi['present_headers'])}[/]"
                )

        console.print(
            f"\n[dim]Scan sources: {', '.join(results['scan_sources'])}[/]"
        )

    asyncio.run(_run())


def _init_test_components(config: VerticalConfig):
    """Initialize components for test mode (no Apollo key required)."""
    from anthropic import Anthropic

    from core.integrations.supabase_client import EnclaveDB
    from core.rag.embeddings import EmbeddingEngine
    from core.testing.mock_apollo import MockApolloClient

    # Pre-check keys needed for test mode (Apollo is mocked, so not required)
    _check_env_key("SUPABASE_URL", "Database connection")
    _check_env_key("SUPABASE_SERVICE_KEY", "Database authentication")
    _check_env_key("OPENAI_API_KEY", "Text embeddings (RAG)")
    _check_env_key("ANTHROPIC_API_KEY", "AI email drafting (Claude)")

    db = EnclaveDB(vertical_id=config.vertical_id)
    mock_apollo = MockApolloClient()
    embedder = EmbeddingEngine()
    anthropic_client = Anthropic()

    return db, mock_apollo, embedder, anthropic_client


@app.command(name="test-run")
def test_run(
    vertical: str = typer.Argument(
        "enclave_guard", help="Vertical ID (default: enclave_guard)"
    ),
    lead_count: int = typer.Option(
        3, "--leads", "-n", help="Number of mock leads to process (1-3)"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show full draft emails"
    ),
):
    """Run the pipeline with mock leads to validate the setup end-to-end."""

    async def _run():
        config = _get_config(vertical)
        db, mock_apollo, embedder, anthropic_client = _init_test_components(config)

        from core.graph.workflow_engine import build_pipeline_graph, process_lead
        from core.testing.mock_leads import MOCK_LEADS

        # Banner
        console.print(Panel(
            "[bold cyan]TEST RUN MODE[/bold cyan]\n\n"
            f"Vertical: {config.vertical_name}\n"
            f"Mock leads: {min(lead_count, len(MOCK_LEADS))}\n\n"
            f"Apollo:        [yellow]MOCKED[/yellow]  (no API key needed)\n"
            f"Email sending: [yellow]SIMULATED[/yellow]  (no emails sent)\n"
            f"Human review:  [yellow]AUTO-APPROVED[/yellow]\n"
            f"Supabase:      [green]LIVE[/green]  (test data will be written)\n"
            f"Anthropic:     [green]LIVE[/green]  (real Claude drafts)\n"
            f"OpenAI:        [green]LIVE[/green]  (real embeddings)\n",
            title="Pipeline Test Run",
            border_style="yellow",
        ))

        # Build graph in test mode
        graph = build_pipeline_graph(
            config=config,
            db=db,
            apollo=mock_apollo,
            embedder=embedder,
            anthropic_client=anthropic_client,
            test_mode=True,
        )

        leads = MOCK_LEADS[:min(lead_count, len(MOCK_LEADS))]
        total_success = 0
        total_errors = 0

        for i, lead in enumerate(leads, 1):
            contact = lead["contact"]
            company = lead["company"]
            console.print(
                f"\n[bold cyan]{'=' * 60}[/bold cyan]"
                f"\n[bold cyan]Lead {i}/{len(leads)}: "
                f"{contact['name']} ({contact['title']} @ {company['name']})"
                f"[/bold cyan]"
                f"\n[bold cyan]{'=' * 60}[/bold cyan]"
            )

            try:
                result = await process_lead(graph, lead, config.vertical_id)

                # Build per-node status table
                table = Table(show_header=True, header_style="bold", show_lines=False)
                table.add_column("Node", style="cyan", width=20)
                table.add_column("Status", width=14)
                table.add_column("Details")

                # check_duplicate
                if result.get("is_duplicate"):
                    table.add_row(
                        "check_duplicate",
                        "[red]DUPLICATE[/]",
                        result.get("skip_reason", "Previously contacted"),
                    )
                else:
                    table.add_row("check_duplicate", "[green]PASS[/]", "Not a duplicate")

                # enrich_company
                size = result.get("company_size", company.get("employee_count", "?"))
                ind = result.get("company_industry", company.get("industry", "?"))
                table.add_row("enrich_company", "[green]PASS[/]", f"{size} employees, {ind}")

                # enrich_contact
                cid = result.get("contact_id", "")
                cid_short = str(cid)[:8] + "..." if cid else "N/A"
                table.add_row("enrich_contact", "[green]PASS[/]", f"Contact ID: {cid_short}")

                # qualify_lead
                score = result.get("qualification_score", 0)
                signals = result.get("matching_signals", [])
                if result.get("qualified"):
                    signal_str = ", ".join(signals[:3]) if signals else "ICP match"
                    table.add_row(
                        "qualify_lead",
                        "[green]QUALIFIED[/]",
                        f"Score: {score:.0%}, Signals: {signal_str}",
                    )
                else:
                    reason = result.get("disqualification_reason", "Did not meet ICP")
                    table.add_row("qualify_lead", "[red]DISQUALIFIED[/]", reason)

                # select_strategy
                persona = result.get("selected_persona", "")
                approach = result.get("selected_approach", "")
                if persona:
                    table.add_row(
                        "select_strategy",
                        "[green]PASS[/]",
                        f"Persona: {persona}, Approach: {approach}",
                    )

                # draft_outreach
                subject = result.get("draft_email_subject", "")
                if subject:
                    subj_preview = subject[:50] + ("..." if len(subject) > 50 else "")
                    table.add_row(
                        "draft_outreach",
                        "[green]PASS[/]",
                        f'Subject: "{subj_preview}"',
                    )

                # compliance_check
                if result.get("compliance_passed"):
                    table.add_row("compliance_check", "[green]PASS[/]", "No issues")
                elif result.get("compliance_issues"):
                    issues = "; ".join(result.get("compliance_issues", []))
                    table.add_row("compliance_check", "[red]FAIL[/]", issues)

                # human_review
                if result.get("human_review_status"):
                    table.add_row(
                        "human_review",
                        "[yellow]AUTO[/]",
                        "Auto-approved (test mode)",
                    )

                # send_outreach
                if result.get("email_sent"):
                    table.add_row(
                        "send_outreach",
                        "[yellow]SIMULATED[/]",
                        f"Would send to {result.get('contact_email', contact['email'])}",
                    )

                # write_to_rag
                if result.get("knowledge_written"):
                    table.add_row("write_to_rag", "[green]PASS[/]", "Knowledge stored")
                else:
                    table.add_row("write_to_rag", "[green]PASS[/]", "Completed")

                console.print(table)

                # Show draft email in verbose mode
                if verbose and result.get("draft_email_body"):
                    console.print(Panel(
                        f"[bold]To:[/bold] {result.get('contact_email', contact['email'])}\n"
                        f"[bold]Subject:[/bold] {result.get('draft_email_subject', '')}\n\n"
                        f"{result.get('draft_email_body', '')}",
                        title="Draft Email",
                        border_style="blue",
                    ))

                total_success += 1

            except Exception as e:
                console.print(f"  [red]ERROR:[/red] {e}")
                logger.exception(f"Test run error for lead {i}")
                total_errors += 1

        # Summary
        console.print(f"\n[bold]{'=' * 60}[/bold]")
        style = "green" if total_errors == 0 else "yellow"
        console.print(Panel(
            f"[{style}]Test Run Complete[/{style}]\n\n"
            f"Total leads:  {len(leads)}\n"
            f"Successful:   [green]{total_success}[/green]\n"
            f"Errors:       [red]{total_errors}[/red]",
            title="Summary",
            border_style=style,
        ))

    asyncio.run(_run())


if __name__ == "__main__":
    app()
