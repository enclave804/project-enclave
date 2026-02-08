"""
Email template engine for Project Enclave.

Renders Jinja2 templates with lead-specific data for outreach emails.
Templates are stored as markdown files in vertical prompt directories.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from jinja2 import Environment, FileSystemLoader, TemplateNotFound

logger = logging.getLogger(__name__)


class TemplateEngine:
    """
    Renders email templates with context data.

    Templates use Jinja2 syntax and are stored as .md files in
    verticals/{vertical_id}/prompts/email_templates/
    """

    def __init__(self, template_dir: str | Path):
        self.template_dir = Path(template_dir)
        if self.template_dir.exists():
            self.env = Environment(
                loader=FileSystemLoader(str(self.template_dir)),
                autoescape=False,  # email templates, not HTML pages
                trim_blocks=True,
                lstrip_blocks=True,
            )
        else:
            self.env = None
            logger.warning(f"Template directory not found: {self.template_dir}")

    def render(
        self,
        template_name: str,
        context: dict[str, Any],
    ) -> dict[str, str]:
        """
        Render a template with context data.

        Args:
            template_name: Name of the template file (e.g., 'vulnerability_alert.md').
            context: Dict with keys like 'contact_name', 'company_name',
                     'tech_stack', 'vulnerabilities', etc.

        Returns:
            Dict with 'subject' and 'body' keys.
        """
        if not self.env:
            raise FileNotFoundError(
                f"Template directory not found: {self.template_dir}"
            )

        try:
            template = self.env.get_template(template_name)
        except TemplateNotFound:
            raise FileNotFoundError(
                f"Template not found: {template_name} "
                f"(looked in {self.template_dir})"
            )

        rendered = template.render(**context)

        # Parse subject from template (first line starting with "Subject:")
        lines = rendered.strip().split("\n")
        subject = ""
        body_start = 0

        for i, line in enumerate(lines):
            if line.strip().lower().startswith("subject:"):
                subject = line.split(":", 1)[1].strip()
                body_start = i + 1
                break

        # Skip separator line (---) if present
        if body_start < len(lines) and lines[body_start].strip() == "---":
            body_start += 1

        body = "\n".join(lines[body_start:]).strip()

        return {"subject": subject, "body": body}

    def list_templates(self) -> list[str]:
        """List all available template files."""
        if not self.template_dir.exists():
            return []
        return [
            f.name for f in self.template_dir.glob("*.md")
        ]

    def get_template_content(self, template_name: str) -> str:
        """Read raw template content for preview."""
        path = self.template_dir / template_name
        if not path.exists():
            raise FileNotFoundError(f"Template not found: {path}")
        return path.read_text()


def build_template_context(
    state: dict[str, Any],
    config_extras: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Build the template context from pipeline state.

    This standardizes the variables available to all templates:
    - contact_name, contact_title, contact_email
    - company_name, company_domain, company_industry, company_size
    - tech_stack (dict), vulnerabilities (list)
    - physical_address, unsubscribe_url
    """
    context = {
        "contact_name": state.get("contact_name", ""),
        "contact_first_name": (state.get("contact_name", "") or "").split()[0]
        if state.get("contact_name")
        else "",
        "contact_title": state.get("contact_title", ""),
        "contact_email": state.get("contact_email", ""),
        "company_name": state.get("company_name", ""),
        "company_domain": state.get("company_domain", ""),
        "company_industry": state.get("company_industry", ""),
        "company_size": state.get("company_size", 0),
        "tech_stack": state.get("tech_stack", {}),
        "tech_stack_list": list(state.get("tech_stack", {}).keys()),
        "vulnerabilities": state.get("vulnerabilities", []),
        "vulnerability_count": len(state.get("vulnerabilities", [])),
        "selected_approach": state.get("selected_approach", ""),
        "selected_persona": state.get("selected_persona", ""),
    }

    if config_extras:
        context.update(config_extras)

    return context
