# ═══════════════════════════════════════════════════════════════════════
# Sovereign Venture Engine — Production Dockerfile
#
# Multi-stage build for minimal image size and security.
# Includes Playwright/Chromium for browser automation (SEO Agent).
#
# Build:  docker build -t enclave .
# Run:    docker run --env-file .env.prod -p 8501:8501 enclave
# ═══════════════════════════════════════════════════════════════════════

FROM python:3.11-slim AS base

# Metadata
LABEL maintainer="Sovereign Venture Engine"
LABEL description="Multi-agent AI platform for autonomous business operations"

# Prevent Python from writing .pyc and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# ─── System Dependencies ──────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    build-essential \
    # Playwright system dependencies
    libnss3 \
    libnspr4 \
    libdbus-1-3 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libxkbcommon0 \
    libatspi2.0-0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libpango-1.0-0 \
    libcairo2 \
    libasound2 \
    && rm -rf /var/lib/apt/lists/*

# ─── Python Dependencies ──────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ─── Playwright Browser ───────────────────────────────────────────────
RUN playwright install chromium

# ─── Application Code ─────────────────────────────────────────────────
COPY core/ ./core/
COPY verticals/ ./verticals/
COPY dashboard/ ./dashboard/
COPY infrastructure/migrations/ ./infrastructure/migrations/
COPY main.py .

# ─── Create non-root user ─────────────────────────────────────────────
RUN groupadd -r enclave && useradd -r -g enclave -d /app enclave \
    && chown -R enclave:enclave /app

# Create directories for runtime artifacts
RUN mkdir -p /app/sandbox_logs /app/browser_sessions \
    && chown -R enclave:enclave /app/sandbox_logs /app/browser_sessions

USER enclave

# ─── Health Check ──────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# ─── Expose Streamlit Port ────────────────────────────────────────────
EXPOSE 8501

# ─── Entrypoint ───────────────────────────────────────────────────────
CMD ["streamlit", "run", "dashboard/app.py", \
     "--server.port", "8501", \
     "--server.headless", "true", \
     "--browser.gatherUsageStats", "false", \
     "--theme.primaryColor", "#4F46E5", \
     "--theme.backgroundColor", "#0E1117", \
     "--theme.secondaryBackgroundColor", "#1A1D29", \
     "--theme.textColor", "#FAFAFA"]
