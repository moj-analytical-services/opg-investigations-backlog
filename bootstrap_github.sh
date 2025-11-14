#!/usr/bin/env bash
set -euo pipefail

REPO="moj-analytical-services/opg-investigations-backlog"

echo "Creating labels..."
gh label create "epic" --repo "$REPO" --color 5319e7 --description "Parent issue grouping related work" || true
gh label create "task" --repo "$REPO" --color 0e8a16 --description "Actionable task" || true
gh label create "decision" --repo "$REPO" --color d93f0b --description "Decision point/governance" || true
gh label create "forecasting" --repo "$REPO" --color 1d76db || true
gh label create "analytics" --repo "$REPO" --color 0052cc || true
gh label create "data" --repo "$REPO" --color 1d76db || true
gh label create "pipeline" --repo "$REPO" --color 5319e7 || true
gh label create "ops" --repo "$REPO" --color 0e8a16 || true
gh label create "intervals" --repo "$REPO" --color 5319e7 || true
gh label create "workforce" --repo "$REPO" --color 5319e7 || true
gh label create "simulation" --repo "$REPO" --color 5319e7 || true
gh label create "simul8" --repo "$REPO" --color 0366d6 || true
gh label create "python" --repo "$REPO" --color 0366d6 || true
gh label create "policy" --repo "$REPO" --color c2e0c6 || true
gh label create "governance" --repo "$REPO" --color fef2c0 || true
gh label create "priority-P0" --repo "$REPO" --color b60205 --description "Highest priority" || true
gh label create "priority-P1" --repo "$REPO" --color d93f0b || true
gh label create "priority-P2" --repo "$REPO" --color fbca04 || true
gh label create "priority-P3" --repo "$REPO" --color c2e0c6 || true
gh label create "stage-1" --repo "$REPO" --color 0e8a16 || true
gh label create "stage-2" --repo "$REPO" --color 0e8a16 || true
gh label create "stage-3A" --repo "$REPO" --color 0e8a16 || true
gh label create "stage-3B" --repo "$REPO" --color 0e8a16 || true

echo "Creating milestones..."
gh api repos/$REPO/milestones -f title="Stage 1 — Forecasting by Case Type" -f state="open" -f due_on="2026-01-09T17:00:00Z" -f description="10 Nov 2025 → 9 Jan 2026" >/dev/null || true
gh api repos/$REPO/milestones -f title="Stage 2 — Operational Dynamics" -f state="open" -f due_on="2026-04-03T17:00:00Z" -f description="12 Jan 2026 → 3 Apr 2026" >/dev/null || true
gh api repos/$REPO/milestones -f title="Decision — Simul8 vs Python" -f state="open" -f due_on="2026-04-10T17:00:00Z" -f description="6–10 Apr 2026" >/dev/null || true
gh api repos/$REPO/milestones -f title="Stage 3A — End-to-end with Simul8" -f state="open" -f due_on="2026-07-31T17:00:00Z" -f description="13 Apr 2026 → 31 Jul 2026" >/dev/null || true
gh api repos/$REPO/milestones -f title="Stage 3B — End-to-end with Python" -f state="open" -f due_on="2026-09-25T17:00:00Z" -f description="13 Apr 2026 → 25 Sep 2026" >/dev/null || true

echo "Done. Now import issues (Section 2) and add them to the Project board (Section 3)."
