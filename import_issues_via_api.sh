#!/usr/bin/env bash
set -euo pipefail
REPO="moj-analytical-services/opg-investigations-backlog"
gh api repos/$REPO/import/issues --input "/mnt/data/issues.json" -H "Accept: application/vnd.github+json"
echo "Submitted import. Poll status:"
echo 'gh api repos/$REPO/import/issues --method GET'
