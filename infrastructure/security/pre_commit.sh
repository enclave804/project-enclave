#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
# Secret Scanner — Pre-Commit Hook
#
# Scans staged files for accidentally committed secrets:
#   - Anthropic API keys (sk-ant-*)
#   - Supabase project keys (sbp_*)
#   - OpenAI/Project keys (sk-proj-*)
#   - Generic API keys and tokens
#   - .env files with sensitive content
#
# Installation:
#   cp infrastructure/security/pre_commit.sh .git/hooks/pre-commit
#   chmod +x .git/hooks/pre-commit
#
# Or use with pre-commit framework:
#   - repo: local
#     hooks:
#       - id: secret-scanner
#         name: Secret Scanner
#         entry: infrastructure/security/pre_commit.sh
#         language: script
#         stages: [commit]
# ═══════════════════════════════════════════════════════════════════════

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}🔍 Scanning staged files for secrets...${NC}"

# Get list of staged files (excluding deleted files)
STAGED_FILES=$(git diff --cached --name-only --diff-filter=ACMR 2>/dev/null || true)

if [ -z "$STAGED_FILES" ]; then
    echo -e "${GREEN}✅ No staged files to scan.${NC}"
    exit 0
fi

FOUND_SECRETS=0

# ─── Pattern Definitions ───────────────────────────────────────────────

declare -a PATTERNS=(
    "sk-ant-[a-zA-Z0-9_-]{20,}"           # Anthropic API keys
    "sk-proj-[a-zA-Z0-9_-]{20,}"          # OpenAI project keys
    "sbp_[a-zA-Z0-9]{20,}"                # Supabase project keys
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"  # JWT tokens (Supabase anon/service keys)
    "SUPABASE_SERVICE_ROLE_KEY=['\"]?[a-zA-Z0-9._-]{30,}"
    "ANTHROPIC_API_KEY=['\"]?sk-ant-"
    "password['\"]?\s*[:=]\s*['\"][^'\"]{8,}" # Hardcoded passwords
)

declare -a PATTERN_NAMES=(
    "Anthropic API key"
    "OpenAI project key"
    "Supabase project key"
    "JWT token"
    "Supabase service role key"
    "Anthropic API key assignment"
    "Hardcoded password"
)

# ─── Scan Each Staged File ─────────────────────────────────────────────

for file in $STAGED_FILES; do
    # Skip binary files, lock files, and this script itself
    if [[ "$file" == *.lock ]] || [[ "$file" == *.png ]] || \
       [[ "$file" == *.jpg ]] || [[ "$file" == *.ico ]] || \
       [[ "$file" == *.woff* ]] || [[ "$file" == *.ttf ]] || \
       [[ "$file" == "infrastructure/security/pre_commit.sh" ]]; then
        continue
    fi

    # Get staged content (not working directory)
    CONTENT=$(git show ":$file" 2>/dev/null || true)
    if [ -z "$CONTENT" ]; then
        continue
    fi

    for i in "${!PATTERNS[@]}"; do
        MATCH=$(echo "$CONTENT" | grep -nE "${PATTERNS[$i]}" 2>/dev/null || true)
        if [ -n "$MATCH" ]; then
            echo -e "${RED}🚨 POTENTIAL SECRET FOUND!${NC}"
            echo -e "   File: ${YELLOW}$file${NC}"
            echo -e "   Type: ${RED}${PATTERN_NAMES[$i]}${NC}"
            echo -e "   Match: $(echo "$MATCH" | head -1 | cut -c1-80)..."
            echo ""
            FOUND_SECRETS=1
        fi
    done
done

# ─── Block .env files ──────────────────────────────────────────────────

for file in $STAGED_FILES; do
    if [[ "$file" == ".env" ]] || [[ "$file" == ".env.prod" ]] || \
       [[ "$file" == ".env.local" ]] || [[ "$file" == ".env.production" ]]; then
        echo -e "${RED}🚨 ENVIRONMENT FILE STAGED: ${YELLOW}$file${NC}"
        echo -e "   .env files should NEVER be committed."
        echo -e "   Add them to .gitignore."
        echo ""
        FOUND_SECRETS=1
    fi
done

# ─── Result ────────────────────────────────────────────────────────────

if [ "$FOUND_SECRETS" -eq 1 ]; then
    echo -e "${RED}═══════════════════════════════════════════════════════${NC}"
    echo -e "${RED}  ❌ COMMIT BLOCKED — Secrets detected in staged files  ${NC}"
    echo -e "${RED}═══════════════════════════════════════════════════════${NC}"
    echo ""
    echo "  To fix:"
    echo "    1. Remove the secret from the file"
    echo "    2. Use environment variables instead"
    echo "    3. Add sensitive files to .gitignore"
    echo ""
    echo "  To bypass (NOT recommended):"
    echo "    git commit --no-verify"
    echo ""
    exit 1
else
    echo -e "${GREEN}✅ No secrets detected. Commit safe.${NC}"
    exit 0
fi
