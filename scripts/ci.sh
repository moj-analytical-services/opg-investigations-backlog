#!/usr/bin/env bash
set -euo pipefail

python -m pip install -U pip
pip install -r requirements.txt

ruff check .
black --check .

pytest -q -m "smoke" --maxfail=1
