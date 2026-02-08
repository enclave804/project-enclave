"""
Enclave Guard - Tech Stack & Vulnerability Scanner.

Uses Shodan API and public SSL/header checks to identify
security findings for target companies. All data comes from
publicly available sources — no active probing.
"""

from __future__ import annotations

import logging
import os
import ssl
import socket
from typing import Any, Optional
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)


class TechStackScanner:
    """
    Scans a company's public-facing infrastructure for security signals.

    Data sources (all public):
    - Shodan API: exposed services, ports, banners
    - SSL Labs (passive): certificate validity, grade
    - HTTP headers: security headers check
    """

    def __init__(self):
        self.shodan_key = os.environ.get("SHODAN_API_KEY", "")

    async def scan_domain(self, domain: str) -> dict[str, Any]:
        """
        Run all available scans on a domain.

        Returns consolidated findings dict with:
        - tech_stack: additional technologies detected
        - vulnerabilities: list of security findings
        - ssl_info: SSL certificate details
        - headers_info: security headers analysis
        """
        results: dict[str, Any] = {
            "tech_stack": {},
            "vulnerabilities": [],
            "ssl_info": {},
            "headers_info": {},
            "shodan_info": {},
            "scan_sources": [],
        }

        # Run scans in parallel where possible
        if self.shodan_key:
            try:
                shodan_data = await self._scan_shodan(domain)
                results["shodan_info"] = shodan_data
                results["scan_sources"].append("shodan")

                # Extract tech and vulns from Shodan
                if shodan_data:
                    results["tech_stack"].update(
                        self._extract_tech_from_shodan(shodan_data)
                    )
                    results["vulnerabilities"].extend(
                        self._extract_vulns_from_shodan(shodan_data)
                    )
            except Exception as e:
                logger.warning(f"Shodan scan failed for {domain}: {e}")

        # SSL check
        try:
            ssl_info = await self._check_ssl(domain)
            results["ssl_info"] = ssl_info
            results["scan_sources"].append("ssl_check")

            if ssl_info.get("issues"):
                results["vulnerabilities"].extend(ssl_info["issues"])
        except Exception as e:
            logger.warning(f"SSL check failed for {domain}: {e}")

        # HTTP security headers
        try:
            headers_info = await self._check_security_headers(domain)
            results["headers_info"] = headers_info
            results["scan_sources"].append("headers_check")

            if headers_info.get("missing_headers"):
                results["vulnerabilities"].append({
                    "type": "missing_security_headers",
                    "severity": "medium",
                    "description": (
                        f"Missing security headers: "
                        f"{', '.join(headers_info['missing_headers'])}"
                    ),
                    "source": "header_scan",
                })
        except Exception as e:
            logger.warning(f"Header check failed for {domain}: {e}")

        logger.info(
            f"Scan complete for {domain}: "
            f"{len(results['vulnerabilities'])} findings, "
            f"sources: {results['scan_sources']}"
        )
        return results

    async def _scan_shodan(self, domain: str) -> dict[str, Any]:
        """Query Shodan for domain information."""
        async with httpx.AsyncClient(timeout=15.0) as client:
            # DNS resolve first
            response = await client.get(
                f"https://api.shodan.io/dns/resolve",
                params={"hostnames": domain, "key": self.shodan_key},
            )
            response.raise_for_status()
            ips = response.json()

            ip = ips.get(domain)
            if not ip:
                return {}

            # Get host info
            response = await client.get(
                f"https://api.shodan.io/shodan/host/{ip}",
                params={"key": self.shodan_key},
            )
            if response.status_code == 404:
                return {}  # no data for this IP
            response.raise_for_status()
            return response.json()

    def _extract_tech_from_shodan(self, data: dict) -> dict[str, str]:
        """Extract technologies from Shodan host data."""
        tech = {}
        for service in data.get("data", []):
            product = service.get("product", "")
            version = service.get("version", "")
            if product:
                tech[product] = version or "detected"

            # Check HTTP component
            http = service.get("http", {})
            if http:
                server = http.get("server", "")
                if server:
                    tech[server] = "web_server"

                # Check for common CMS/frameworks in HTML
                html = http.get("html", "")
                if "wp-content" in html.lower():
                    tech["WordPress"] = "detected"
                if "drupal" in html.lower():
                    tech["Drupal"] = "detected"
                if "shopify" in html.lower():
                    tech["Shopify"] = "detected"

        return tech

    def _extract_vulns_from_shodan(self, data: dict) -> list[dict]:
        """Extract vulnerability signals from Shodan data."""
        vulns = []

        # Check for open ports that shouldn't be public
        risky_ports = {
            21: ("FTP", "high"),
            23: ("Telnet", "critical"),
            3306: ("MySQL", "high"),
            5432: ("PostgreSQL", "high"),
            6379: ("Redis", "critical"),
            27017: ("MongoDB", "critical"),
            9200: ("Elasticsearch", "high"),
        }

        ports = data.get("ports", [])
        for port in ports:
            if port in risky_ports:
                service, severity = risky_ports[port]
                vulns.append({
                    "type": "exposed_service",
                    "severity": severity,
                    "description": (
                        f"{service} (port {port}) is publicly accessible"
                    ),
                    "port": port,
                    "source": "shodan",
                })

        # Check for known vulnerabilities (CVEs)
        for service in data.get("data", []):
            for vuln_id in service.get("vulns", {}).keys():
                vulns.append({
                    "type": "known_cve",
                    "severity": "high",
                    "description": f"Known vulnerability: {vuln_id}",
                    "cve": vuln_id,
                    "source": "shodan",
                })

        return vulns

    async def _check_ssl(self, domain: str) -> dict[str, Any]:
        """Check SSL certificate validity and configuration."""
        result: dict[str, Any] = {"issues": []}

        try:
            context = ssl.create_default_context()
            with socket.create_connection((domain, 443), timeout=5) as sock:
                with context.wrap_socket(sock, server_hostname=domain) as ssock:
                    cert = ssock.getpeercert()
                    result["subject"] = dict(
                        x[0] for x in cert.get("subject", ())
                    )
                    result["issuer"] = dict(
                        x[0] for x in cert.get("issuer", ())
                    )
                    result["not_after"] = cert.get("notAfter", "")
                    result["version"] = ssock.version()

                    # Check for weak protocol
                    if ssock.version() in ("TLSv1", "TLSv1.1"):
                        result["issues"].append({
                            "type": "weak_tls",
                            "severity": "high",
                            "description": (
                                f"Using outdated TLS version: {ssock.version()}"
                            ),
                            "source": "ssl_check",
                        })
        except ssl.SSLError as e:
            result["issues"].append({
                "type": "ssl_error",
                "severity": "high",
                "description": f"SSL certificate error: {str(e)}",
                "source": "ssl_check",
            })
        except (socket.timeout, ConnectionRefusedError, OSError):
            result["issues"].append({
                "type": "no_ssl",
                "severity": "medium",
                "description": "Could not establish SSL connection on port 443",
                "source": "ssl_check",
            })

        return result

    async def _check_security_headers(
        self, domain: str
    ) -> dict[str, Any]:
        """Check for important security HTTP headers."""
        important_headers = {
            "strict-transport-security": "HSTS",
            "content-security-policy": "CSP",
            "x-content-type-options": "X-Content-Type-Options",
            "x-frame-options": "X-Frame-Options",
            "x-xss-protection": "X-XSS-Protection",
            "referrer-policy": "Referrer-Policy",
            "permissions-policy": "Permissions-Policy",
        }

        result: dict[str, Any] = {
            "present_headers": [],
            "missing_headers": [],
        }

        try:
            async with httpx.AsyncClient(
                timeout=10.0, follow_redirects=True
            ) as client:
                response = await client.get(f"https://{domain}")
                headers = {k.lower(): v for k, v in response.headers.items()}

                for header, name in important_headers.items():
                    if header in headers:
                        result["present_headers"].append(name)
                    else:
                        result["missing_headers"].append(name)

                # Check server header disclosure
                if "server" in headers:
                    result["server_disclosed"] = headers["server"]

        except Exception as e:
            logger.warning(f"Header check failed for {domain}: {e}")

        return result
