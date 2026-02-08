#!/usr/bin/env bash
# Sovereign Cockpit — Dashboard Launcher
#
# Usage: ./dashboard/run.sh [--port PORT]
#
# Launches the Streamlit dashboard with optimal settings.
# The old single-file dashboard.py is still available at the project root.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default port
PORT="${1:-8501}"
if [[ "$1" == "--port" ]] && [[ -n "${2:-}" ]]; then
    PORT="$2"
fi

# Activate venv if it exists
if [[ -f "$PROJECT_ROOT/.venv/bin/activate" ]]; then
    source "$PROJECT_ROOT/.venv/bin/activate"
fi

echo "🛡️  Sovereign Cockpit starting on http://localhost:$PORT"
echo "   Dashboard: $SCRIPT_DIR/app.py"
echo ""

exec streamlit run "$SCRIPT_DIR/app.py" \
    --server.port "$PORT" \
    --server.headless true \
    --browser.gatherUsageStats false \
    --theme.primaryColor "#4F46E5" \
    --theme.backgroundColor "#0E1117" \
    --theme.secondaryBackgroundColor "#1A1D29" \
    --theme.textColor "#FAFAFA"
