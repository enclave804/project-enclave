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
agent_app = typer.Typer(
    name="agent",
    help="Agent framework commands — list, run, and inspect agents.",
)
app.add_typer(agent_app, name="agent")

genesis_app = typer.Typer(
    name="genesis",
    help="Genesis Engine — launch new business verticals from scratch.",
)
app.add_typer(genesis_app, name="genesis")

api_app = typer.Typer(
    name="api",
    help="Enterprise API Gateway — start and manage the REST API.",
)
app.add_typer(api_app, name="api")

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
def status(
    vertical: str = typer.Argument(
        "enclave_guard", help="Vertical ID (default: enclave_guard)"
    ),
):
    """Check all API connections and service status."""
    import httpx

    config = _get_config(vertical)

    table = Table(title=f"Service Status — {config.vertical_name}")
    table.add_column("Service", style="cyan", width=20)
    table.add_column("Status", width=14)
    table.add_column("Details")

    # 1. Supabase
    url = os.environ.get("SUPABASE_URL", "").strip()
    key = os.environ.get("SUPABASE_SERVICE_KEY", "").strip()
    if url and key:
        try:
            from core.integrations.supabase_client import EnclaveDB
            db = EnclaveDB(vertical_id=config.vertical_id)
            # Quick health check: count companies
            result = db.client.table("companies").select("id", count="exact").limit(0).execute()
            count_val = result.count if hasattr(result, "count") else "?"
            table.add_row("Supabase", "[green]CONNECTED[/]", f"{url.split('//')[1].split('.')[0]} ({count_val} companies)")
        except Exception as e:
            table.add_row("Supabase", "[red]ERROR[/]", str(e)[:60])
    else:
        table.add_row("Supabase", "[red]MISSING[/]", "Set SUPABASE_URL + SUPABASE_SERVICE_KEY")

    # 2. Anthropic
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if anthropic_key:
        try:
            from anthropic import Anthropic
            client = Anthropic()
            # Light-weight check: create a tiny completion
            resp = client.messages.create(
                model="claude-sonnet-4-5-20250514",
                max_tokens=5,
                messages=[{"role": "user", "content": "Say OK"}],
            )
            table.add_row("Anthropic", "[green]CONNECTED[/]", f"Key: ...{anthropic_key[-6:]}")
        except Exception as e:
            err = str(e)
            if "credit" in err.lower() or "billing" in err.lower() or "insufficient" in err.lower():
                table.add_row("Anthropic", "[yellow]NO CREDITS[/]", "Key valid, add credits at console.anthropic.com")
            else:
                table.add_row("Anthropic", "[red]ERROR[/]", err[:60])
    else:
        table.add_row("Anthropic", "[red]MISSING[/]", "Set ANTHROPIC_API_KEY")

    # 3. OpenAI (embeddings)
    openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if openai_key:
        try:
            from core.rag.embeddings import EmbeddingEngine
            embedder = EmbeddingEngine()
            # Quick test: embed a short string
            import asyncio
            vec = asyncio.run(embedder.embed_text("test"))
            dim = len(vec) if vec else 0
            table.add_row("OpenAI", "[green]CONNECTED[/]", f"Embeddings OK ({dim}D vectors)")
        except Exception as e:
            table.add_row("OpenAI", "[red]ERROR[/]", str(e)[:60])
    else:
        table.add_row("OpenAI", "[red]MISSING[/]", "Set OPENAI_API_KEY")

    # 4. Apollo
    apollo_key = os.environ.get(config.apollo.api_key_env, "").strip()
    if apollo_key:
        table.add_row("Apollo.io", "[green]CONFIGURED[/]", f"Key: ...{apollo_key[-6:]}")
    else:
        table.add_row("Apollo.io", "[yellow]NOT SET[/]", f"Set {config.apollo.api_key_env} for lead sourcing")

    # 5. SendGrid
    sg_key = os.environ.get("SENDGRID_API_KEY", "").strip()
    if sg_key:
        table.add_row("SendGrid", "[green]CONFIGURED[/]", f"Domain: {config.outreach.email.sending_domain}")
    else:
        table.add_row("SendGrid", "[dim]NOT SET[/]", "Optional — set SENDGRID_API_KEY for email sending")

    # 6. Mailgun
    mg_key = os.environ.get("MAILGUN_API_KEY", "").strip()
    if mg_key:
        table.add_row("Mailgun", "[green]CONFIGURED[/]", f"Domain: {config.outreach.email.sending_domain}")
    else:
        table.add_row("Mailgun", "[dim]NOT SET[/]", "Optional — set MAILGUN_API_KEY for email sending")

    # 7. Shodan
    shodan_key = os.environ.get("SHODAN_API_KEY", "").strip()
    if shodan_key:
        table.add_row("Shodan", "[green]CONFIGURED[/]", "Enhanced port scanning enabled")
    else:
        table.add_row("Shodan", "[dim]NOT SET[/]", "Optional — HTTP header scanning still works")

    console.print(table)


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

    async def _run():
        from core.graph.workflow_engine import (
            build_pipeline_graph,
            get_persistent_checkpointer,
        )

        config = _get_config(vertical)

        # Use persistent checkpointer to find interrupted runs
        checkpointer = get_persistent_checkpointer()

        # Review doesn't need Apollo (we're resuming, not sourcing leads)
        db, mock_apollo, embedder, anthropic_client = _init_test_components(config)

        graph = build_pipeline_graph(
            config=config,
            db=db,
            apollo=mock_apollo,
            embedder=embedder,
            anthropic_client=anthropic_client,
            checkpointer=checkpointer,
        )

        # Find all interrupted threads by scanning checkpoint storage
        console.print("[cyan]Scanning for pending reviews...[/]")

        pending = []
        try:
            # SqliteSaver stores checkpoints with thread_id configs
            # We need to list threads that are interrupted at human_review
            import sqlite3
            from core.graph.workflow_engine import CHECKPOINT_DB_PATH

            if not CHECKPOINT_DB_PATH.exists():
                console.print(Panel(
                    "[yellow]No checkpoint database found.[/]\n\n"
                    "Run the pipeline first to generate email drafts:\n"
                    f"  [dim]python main.py run {vertical}[/]",
                    title="No Pending Reviews",
                ))
                return

            conn = sqlite3.connect(str(CHECKPOINT_DB_PATH))
            cursor = conn.cursor()

            # Check if checkpoints table exists
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='checkpoints'"
            )
            if not cursor.fetchone():
                conn.close()
                console.print(Panel(
                    "[yellow]No pipeline runs recorded yet.[/]\n\n"
                    "Run the pipeline first to generate email drafts:\n"
                    f"  [dim]python main.py run {vertical}[/]",
                    title="No Pending Reviews",
                ))
                return

            # Get all distinct thread IDs from checkpoints
            cursor.execute("""
                SELECT DISTINCT thread_id FROM checkpoints
                ORDER BY thread_ts DESC
            """)
            thread_ids = [row[0] for row in cursor.fetchall()]
            conn.close()

            for thread_id in thread_ids:
                try:
                    state = graph.get_state({"configurable": {"thread_id": thread_id}})
                    if state and state.next and "human_review" in state.next:
                        vals = state.values
                        pending.append({
                            "thread_id": thread_id,
                            "state": vals,
                            "config": {"configurable": {"thread_id": thread_id}},
                        })
                except Exception:
                    continue  # Skip corrupted/invalid checkpoints

        except Exception as e:
            console.print(f"[red]Error scanning checkpoints: {e}[/]")
            return

        if not pending:
            console.print(Panel(
                "[green]No pending reviews![/]\n\n"
                "All email drafts have been reviewed.\n"
                "Run the pipeline to generate new drafts.",
                title="Review Queue Empty",
            ))
            return

        console.print(Panel(
            f"[cyan]Found {len(pending)} email(s) awaiting review.[/]\n\n"
            "For each email you can:\n"
            "  [green]a[/]pprove  — send the email as-is\n"
            "  [yellow]e[/]dit     — modify subject/body, then approve\n"
            "  [red]r[/]eject   — send back for re-drafting with feedback\n"
            "  [dim]s[/]kip     — skip this lead entirely",
            title="Human Review",
            border_style="cyan",
        ))

        approved = 0
        rejected = 0
        skipped = 0

        for i, item in enumerate(pending, 1):
            state = item["state"]
            thread_config = item["config"]

            console.print(f"\n[bold cyan]{'─' * 60}[/bold cyan]")
            console.print(
                f"[bold]Review {i}/{len(pending)}:[/bold] "
                f"{state.get('contact_name', '?')} "
                f"({state.get('contact_title', '?')} @ {state.get('company_name', '?')})"
            )

            # Show the draft
            console.print(Panel(
                f"[bold]To:[/bold] {state.get('contact_email', '?')}\n"
                f"[bold]Subject:[/bold] {state.get('draft_email_subject', '(no subject)')}\n"
                f"[bold]Approach:[/bold] {state.get('selected_approach', '?')} "
                f"| [bold]Persona:[/bold] {state.get('selected_persona', '?')}\n"
                f"[bold]Score:[/bold] {state.get('qualification_score', 0):.0%}\n\n"
                f"{state.get('draft_email_body', '(no body)')}",
                title="Draft Email",
                border_style="blue",
            ))

            # Get user decision
            while True:
                choice = Prompt.ask(
                    "[bold]Decision[/bold]",
                    choices=["a", "e", "r", "s"],
                    default="a",
                )

                if choice == "a":
                    # Approve — update state and resume
                    graph.update_state(
                        thread_config,
                        {
                            "human_review_status": "approved",
                            "review_attempts": state.get("review_attempts", 0) + 1,
                        },
                        as_node="human_review",
                    )
                    console.print("[green]✓ Approved — sending...[/]")
                    try:
                        result = await graph.ainvoke(None, config=thread_config)
                        if result.get("email_sent"):
                            console.print(
                                f"[green]✓ Email sent to {state.get('contact_email')}[/]"
                            )
                        else:
                            console.print(
                                f"[yellow]⚠ Email recorded (no provider configured)[/]"
                            )
                    except Exception as e:
                        console.print(f"[red]Error resuming pipeline: {e}[/]")
                    approved += 1
                    break

                elif choice == "e":
                    # Edit — let user modify subject and body
                    new_subject = Prompt.ask(
                        "New subject",
                        default=state.get("draft_email_subject", ""),
                    )
                    console.print(
                        "[dim]Enter new body (press Enter twice to finish):[/]"
                    )
                    body_lines = []
                    while True:
                        line = input()
                        if line == "" and body_lines and body_lines[-1] == "":
                            body_lines.pop()  # remove trailing blank
                            break
                        body_lines.append(line)
                    new_body = "\n".join(body_lines) if body_lines else state.get("draft_email_body", "")

                    graph.update_state(
                        thread_config,
                        {
                            "human_review_status": "edited",
                            "edited_subject": new_subject,
                            "edited_body": new_body,
                            "review_attempts": state.get("review_attempts", 0) + 1,
                        },
                        as_node="human_review",
                    )
                    console.print("[green]✓ Edited & approved — sending...[/]")
                    try:
                        result = await graph.ainvoke(None, config=thread_config)
                        if result.get("email_sent"):
                            console.print(
                                f"[green]✓ Email sent to {state.get('contact_email')}[/]"
                            )
                        else:
                            console.print(
                                f"[yellow]⚠ Email recorded (no provider configured)[/]"
                            )
                    except Exception as e:
                        console.print(f"[red]Error resuming pipeline: {e}[/]")
                    approved += 1
                    break

                elif choice == "r":
                    # Reject — ask for feedback and loop back to draft
                    feedback = Prompt.ask("Feedback for re-drafting")
                    graph.update_state(
                        thread_config,
                        {
                            "human_review_status": "rejected",
                            "human_feedback": feedback,
                            "review_attempts": state.get("review_attempts", 0) + 1,
                        },
                        as_node="human_review",
                    )
                    console.print(
                        "[yellow]↻ Rejected — will re-draft with your feedback.[/]"
                    )
                    try:
                        # Resume — graph will loop back to draft_outreach
                        result = await graph.ainvoke(None, config=thread_config)
                        # After re-draft, it'll hit human_review again
                        # and interrupt — so it'll show up in next review
                        console.print(
                            "[yellow]New draft generated — it will appear in next review.[/]"
                        )
                    except Exception as e:
                        console.print(f"[red]Error resuming pipeline: {e}[/]")
                    rejected += 1
                    break

                elif choice == "s":
                    # Skip — end this lead's pipeline
                    graph.update_state(
                        thread_config,
                        {
                            "human_review_status": "skipped",
                            "skip_reason": "Skipped during human review",
                            "review_attempts": state.get("review_attempts", 0) + 1,
                        },
                        as_node="human_review",
                    )
                    console.print("[dim]⊘ Skipped[/]")
                    try:
                        await graph.ainvoke(None, config=thread_config)
                    except Exception:
                        pass  # Best-effort resume to write_to_rag
                    skipped += 1
                    break

        # Summary
        console.print(f"\n[bold]{'─' * 60}[/bold]")
        console.print(Panel(
            f"[green]Approved:[/green] {approved}\n"
            f"[yellow]Rejected (re-drafting):[/yellow] {rejected}\n"
            f"[dim]Skipped:[/dim] {skipped}",
            title="Review Summary",
        ))

    asyncio.run(_run())


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


@app.command()
def dashboard(
    vertical: str = typer.Argument(
        "enclave_guard", help="Vertical ID (default: enclave_guard)"
    ),
    port: int = typer.Option(8501, help="Streamlit server port"),
):
    """Launch the Streamlit dashboard for monitoring pipeline health."""
    import subprocess
    import shutil

    dashboard_path = Path(__file__).parent / "dashboard.py"
    if not dashboard_path.exists():
        console.print("[red]dashboard.py not found in project root.[/]")
        raise typer.Exit(1)

    if not shutil.which("streamlit"):
        console.print(Panel(
            "[red]Streamlit is not installed.[/]\n\n"
            "Install it with:\n"
            "  [dim]pip install streamlit[/]",
            title="⚠ Missing Dependency",
            border_style="red",
        ))
        raise typer.Exit(1)

    console.print(Panel(
        f"[cyan]Launching dashboard for {vertical}[/]\n\n"
        f"URL: [bold]http://localhost:{port}[/bold]\n"
        f"Press Ctrl+C to stop.",
        title="🛡️ Enclave Dashboard",
        border_style="cyan",
    ))

    subprocess.run(
        [
            "streamlit", "run", str(dashboard_path),
            "--server.port", str(port),
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false",
        ],
    )


# =========================================================================
# Agent Framework Commands
# =========================================================================


def _init_agent_registry(vertical_id: str):
    """
    Discover agents for a vertical using the AgentRegistry.

    Imports the outreach agent implementation so @register_agent_type fires.
    """
    # Import implementations so decorators register agent types
    import core.agents.implementations.outreach_agent  # noqa: F401

    from core.agents.registry import AgentRegistry

    registry = AgentRegistry(vertical_id)
    registry.discover_agents()
    return registry


@agent_app.command(name="list")
def agent_list(
    vertical: str = typer.Argument(
        "enclave_guard", help="Vertical ID (default: enclave_guard)"
    ),
):
    """List all registered agents for a vertical."""
    from core.agents.registry import get_registered_types

    registry = _init_agent_registry(vertical)
    configs = registry.list_configs()

    if not configs:
        console.print(
            f"[yellow]No agents found for vertical '{vertical}'.[/]\n"
            f"Add YAML configs in verticals/{vertical}/agents/"
        )
        return

    # Show registered implementation types
    reg_types = get_registered_types()
    console.print(
        f"[dim]Registered agent types: {', '.join(reg_types)}[/]\n"
    )

    table = Table(title=f"Agents — {vertical}")
    table.add_column("Agent ID", style="cyan")
    table.add_column("Type", style="green")
    table.add_column("Name", style="white")
    table.add_column("Schedule", style="yellow")
    table.add_column("Human Gates", style="blue")
    table.add_column("Status", style="white")

    for cfg in configs:
        gates = ", ".join(cfg.human_gates.gate_before) if cfg.human_gates.gate_before else "none"
        schedule = cfg.schedule.cron if cfg.schedule.cron else cfg.schedule.trigger
        status = "[green]enabled[/]" if cfg.enabled else "[red]disabled[/]"
        table.add_row(
            cfg.agent_id,
            cfg.agent_type,
            cfg.name,
            schedule,
            gates,
            status,
        )

    console.print(table)


@agent_app.command(name="run")
def agent_run(
    vertical: str = typer.Argument(..., help="Vertical ID"),
    agent_id: str = typer.Argument(..., help="Agent ID to run"),
    count: int = typer.Option(10, "--leads", "-n", help="Number of leads (outreach)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Simulate without side effects"),
):
    """Run a specific agent within a vertical."""

    async def _run():
        config = _get_config(vertical)
        registry = _init_agent_registry(vertical)

        # Check agent exists
        agent_config = registry.get_config(agent_id)
        if not agent_config:
            console.print(
                f"[red]Agent not found:[/] [bold]{agent_id}[/]\n"
                f"Available agents: {registry.list_agent_ids()}"
            )
            raise typer.Exit(1)

        console.print(Panel(
            f"[cyan]Running agent:[/] [bold]{agent_config.name}[/]\n"
            f"Type: {agent_config.agent_type}\n"
            f"Vertical: {vertical}\n"
            f"Dry run: {dry_run}",
            title="Agent Run",
            border_style="cyan",
        ))

        # --- Outreach agent (legacy adapter) ---
        if agent_config.agent_type == "outreach":
            await _run_outreach_agent(
                config, registry, agent_id, count, dry_run
            )
        else:
            # Future agents will be dispatched here
            console.print(
                f"[yellow]Agent type '{agent_config.agent_type}' "
                f"does not have a runner yet.[/]"
            )

    asyncio.run(_run())


async def _run_outreach_agent(
    vertical_config: VerticalConfig,
    registry,
    agent_id: str,
    count: int,
    dry_run: bool,
):
    """Run the outreach agent via the strangler-fig adapter."""
    from core.agents.implementations.outreach_agent import OutreachAgent

    db, apollo, embedder, anthropic_client = _init_components(vertical_config)

    agent: OutreachAgent = registry.instantiate_agent(
        agent_id, db, embedder, anthropic_client,
    )
    agent.set_legacy_deps(apollo=apollo, vertical_config=vertical_config)

    if dry_run:
        console.print("[yellow]DRY RUN: Emails will be drafted but NOT sent.[/]")
        # In dry-run mode, use test mode graph
        from core.graph.workflow_engine import build_pipeline_graph, process_batch

        graph = build_pipeline_graph(
            config=vertical_config,
            db=db,
            apollo=apollo,
            embedder=embedder,
            anthropic_client=anthropic_client,
            test_mode=True,
        )

        console.print(f"[cyan]Pulling {count} leads from Apollo...[/]")
        filters = vertical_config.apollo.filters.model_dump()
        filters["per_page"] = count
        leads = await apollo.search_and_parse(filters)

        if not leads:
            console.print("[yellow]No leads found.[/]")
            return

        results = await process_batch(graph, leads, vertical_config.vertical_id)
    else:
        # Full run via agent framework
        console.print(f"[cyan]Pulling {count} leads from Apollo...[/]")
        filters = vertical_config.apollo.filters.model_dump()
        filters["per_page"] = count
        leads = await apollo.search_and_parse(filters)

        if not leads:
            console.print("[yellow]No leads found.[/]")
            return

        console.print(f"[green]Found {len(leads)} leads. Processing via agent framework...[/]")
        results = await agent.run({"leads": leads})

    # Display results
    table = Table(title=f"Agent Results — {agent_id}")
    table.add_column("Metric", style="cyan")
    table.add_column("Count", style="white")

    table.add_row("Total Leads", str(results.get("total", len(leads))))
    table.add_row("Processed", str(results.get("processed", 0)))
    sent_label = "[green]Simulated[/]" if dry_run else "[green]Sent[/]"
    table.add_row(sent_label, str(results.get("sent", 0)))
    table.add_row("[yellow]Skipped[/]", str(results.get("skipped", 0)))
    table.add_row("[red]Errors[/]", str(results.get("errors", 0)))

    console.print(table)


@agent_app.command(name="status")
def agent_status(
    vertical: str = typer.Argument(
        "enclave_guard", help="Vertical ID (default: enclave_guard)"
    ),
):
    """Show agent run statistics for a vertical."""
    from core.integrations.supabase_client import EnclaveDB

    _get_config(vertical)  # Validate vertical exists
    registry = _init_agent_registry(vertical)

    _check_env_key("SUPABASE_URL", "Database connection")
    _check_env_key("SUPABASE_SERVICE_KEY", "Database authentication")

    db = EnclaveDB(vertical_id=vertical)

    configs = registry.list_configs()
    if not configs:
        console.print(f"[yellow]No agents found for vertical '{vertical}'.[/]")
        return

    table = Table(title=f"Agent Status — {vertical}")
    table.add_column("Agent ID", style="cyan")
    table.add_column("Type", style="green")
    table.add_column("Runs (7d)", style="white")
    table.add_column("Success", style="green")
    table.add_column("Failed", style="red")
    table.add_column("Avg Duration", style="yellow")

    for cfg in configs:
        try:
            stats = db.get_agent_stats(
                vertical_id=vertical,
                agent_id=cfg.agent_id,
                days=7,
            )
            if stats:
                total = stats.get("total_runs", 0)
                success = stats.get("completed_runs", 0)
                failed = stats.get("failed_runs", 0)
                avg_ms = stats.get("avg_duration_ms", 0)
                avg_str = f"{avg_ms / 1000:.1f}s" if avg_ms else "—"
                table.add_row(
                    cfg.agent_id, cfg.agent_type,
                    str(total), str(success), str(failed), avg_str,
                )
            else:
                table.add_row(
                    cfg.agent_id, cfg.agent_type,
                    "0", "0", "0", "—",
                )
        except Exception:
            table.add_row(
                cfg.agent_id, cfg.agent_type,
                "[dim]?[/]", "[dim]?[/]", "[dim]?[/]", "[dim]?[/]",
            )

    console.print(table)


# ── Genesis Engine Commands ─────────────────────────────────────────


@genesis_app.command(name="start")
def genesis_start(
    vertical: str = typer.Argument(
        ..., help="Vertical ID to launch (e.g. 'print_biz')"
    ),
    output_dir: str = typer.Option(
        "verticals", help="Base directory for vertical outputs"
    ),
    skip_credentials: bool = typer.Option(
        False, "--skip-credentials", help="Skip credential validation (testing only)"
    ),
):
    """
    Launch a new vertical's agent fleet in shadow mode.

    Runs pre-flight checks (config validation, agent registration,
    credentials) and brings agents online in shadow mode.
    """
    from core.genesis import GenesisLauncher, CredentialManager

    console.print(Panel(
        f"[bold cyan]Genesis Engine[/] — Launching vertical [bold]{vertical}[/]",
        border_style="cyan",
    ))

    # Run pre-flight first
    console.print("\n[bold]Running pre-flight checks...[/]\n")

    cm = CredentialManager()
    launcher = GenesisLauncher(credential_manager=cm)

    preflight = launcher.preflight_check(vertical, output_dir)

    # Display check results
    for check in preflight.checks:
        icon = "[green]✓[/]" if check["passed"] else "[red]✗[/]"
        detail = f" — {check['detail']}" if check.get("detail") else ""
        console.print(f"  {icon} {check['name']}{detail}")

    if not preflight.passed:
        failed = preflight.failed_checks
        # Allow skipping credentials
        critical = [
            c for c in failed
            if not (skip_credentials and c["name"] == "credentials_available")
        ]
        if critical:
            console.print(Panel(
                f"[red]Pre-flight failed with {len(critical)} error(s).[/]\n"
                "Fix the issues above and try again.",
                title="⚠ Launch Aborted",
                border_style="red",
            ))
            raise typer.Exit(code=1)
        else:
            console.print(
                "\n[yellow]Credential check skipped (--skip-credentials).[/]"
            )

    # Launch
    console.print("\n[bold]Launching in shadow mode...[/]\n")

    result = launcher.launch_vertical(
        vertical,
        output_dir,
        skip_credential_check=skip_credentials,
    )

    if result.success:
        console.print(Panel(
            f"[bold green]Vertical launched successfully![/]\n\n"
            f"  Status:  [cyan]{result.status}[/]\n"
            f"  Agents:  [white]{len(result.agent_ids)}[/] "
            f"({', '.join(result.agent_ids) or 'none'})\n"
            + (f"\n  ⚠ Warnings:\n" + "\n".join(
                f"    • {w}" for w in result.warnings
            ) if result.warnings else ""),
            title="🚀 Genesis Complete",
            border_style="green",
        ))
    else:
        console.print(Panel(
            f"[red]Launch failed.[/]\n\n"
            + "\n".join(f"  • {e}" for e in result.errors),
            title="⚠ Launch Failed",
            border_style="red",
        ))
        raise typer.Exit(code=1)


@genesis_app.command(name="preflight")
def genesis_preflight(
    vertical: str = typer.Argument(
        ..., help="Vertical ID to check"
    ),
    output_dir: str = typer.Option(
        "verticals", help="Base directory for vertical outputs"
    ),
):
    """
    Run pre-flight checks without launching.

    Validates the vertical's config, agents, types, and credentials.
    """
    from core.genesis import GenesisLauncher, CredentialManager

    console.print(Panel(
        f"[bold cyan]Genesis Pre-flight[/] — Checking vertical [bold]{vertical}[/]",
        border_style="cyan",
    ))

    cm = CredentialManager()
    launcher = GenesisLauncher(credential_manager=cm)
    preflight = launcher.preflight_check(vertical, output_dir)

    console.print()
    for check in preflight.checks:
        icon = "[green]✓[/]" if check["passed"] else "[red]✗[/]"
        detail = f" — {check['detail']}" if check.get("detail") else ""
        console.print(f"  {icon} {check['name']}{detail}")

    console.print()
    if preflight.passed:
        console.print("[bold green]All checks passed.[/] Ready to launch.")
    else:
        failed = len(preflight.failed_checks)
        console.print(f"[bold red]{failed} check(s) failed.[/]")
        raise typer.Exit(code=1)


@genesis_app.command(name="status")
def genesis_status(
    vertical: str = typer.Argument(
        ..., help="Vertical ID to check status for"
    ),
):
    """
    Check the launch status of a vertical.

    Shows the latest genesis session record from the database.
    """
    from core.genesis import GenesisLauncher

    launcher = GenesisLauncher()
    status = launcher.get_launch_status(vertical)

    if status is None:
        console.print(
            f"[yellow]No launch record found for vertical '{vertical}'.[/]\n"
            f"Use [bold]genesis start {vertical}[/] to launch."
        )
        return

    console.print(Panel(
        f"  Vertical:    [bold]{vertical}[/]\n"
        f"  Status:      [cyan]{status.get('status', 'unknown')}[/]\n"
        f"  Session:     [dim]{status.get('id', 'unknown')}[/]\n"
        f"  Created:     {status.get('created_at', 'unknown')}\n"
        f"  Updated:     {status.get('updated_at', 'unknown')}\n"
        + (f"  Completed:   {status['completed_at']}\n"
           if status.get('completed_at') else "")
        + (f"  Error:       [red]{status['error_message']}[/]\n"
           if status.get('error_message') else ""),
        title=f"🔍 Genesis Status — {vertical}",
        border_style="cyan",
    ))


@genesis_app.command(name="credentials")
def genesis_credentials(
    vertical: str = typer.Argument(
        ..., help="Vertical ID to check credentials for"
    ),
):
    """
    Show credential status for a vertical.

    Lists all required and optional credentials, and whether they're set.
    """
    from core.genesis import CredentialManager

    cm = CredentialManager()
    report = cm.get_credential_report(vertical)

    table = Table(title=f"Credentials — {vertical}")
    table.add_column("Env Variable", style="cyan")
    table.add_column("Name", style="white")
    table.add_column("Required", style="yellow")
    table.add_column("Status", style="white")

    for cred in report.credentials:
        req = "[bold]required[/]" if cred.required else "[dim]optional[/]"
        if cred.is_set:
            status = "[green]✓ set[/]"
        elif cred.required:
            status = "[red]✗ missing[/]"
        else:
            status = "[dim]— not set[/]"

        table.add_row(
            cred.env_var_name,
            cred.credential_name,
            req,
            status,
        )

    console.print(table)
    console.print()

    if report.all_required_set:
        console.print(
            f"[green]All required credentials set "
            f"({report.total_set}/{len(report.credentials)}).[/]"
        )
    else:
        missing = [c.env_var_name for c in report.missing_required]
        console.print(
            f"[red]Missing {len(missing)} required credential(s):[/] "
            f"{', '.join(missing)}"
        )


# ── API Gateway Commands ──────────────────────────────────────────


@api_app.command("start")
def api_start(
    port: int = typer.Option(8000, help="Port to run the API on"),
    host: str = typer.Option("0.0.0.0", help="Host to bind to"),
    vertical: str = typer.Option("enclave_guard", help="Default vertical ID"),
    reload: bool = typer.Option(False, help="Enable auto-reload for development"),
):
    """Start the Enterprise API Gateway."""
    console.print(
        Panel(
            f"[bold]Starting Sovereign Venture Engine API[/]\n\n"
            f"  Host:     {host}:{port}\n"
            f"  Vertical: {vertical}\n"
            f"  Docs:     http://{host}:{port}/api/docs\n"
            f"  Reload:   {'enabled' if reload else 'disabled'}",
            title="◆ API Gateway",
            border_style="blue",
        )
    )

    try:
        import uvicorn
        from core.integrations.supabase_client import EnclaveDB
        from core.enterprise.api_server import create_api_app

        db = EnclaveDB(vertical)

        # Try to create embedder for RAG search
        embedder = None
        try:
            from core.rag.embeddings import get_embedder
            embedder = get_embedder()
        except Exception:
            console.print("[yellow]⚠ Embedder not available — RAG search disabled[/]")

        api_application = create_api_app(db=db, embedder=embedder)

        uvicorn.run(
            api_application,
            host=host,
            port=port,
            log_level="info",
        )

    except ImportError as e:
        console.print(f"[red]Missing dependency: {e}[/]")
        console.print("Install with: pip install 'fastapi' 'uvicorn[standard]'")
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"[red]API start failed: {e}[/]")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
