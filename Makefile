# --- Makefile for opg-investigations-backlog ---
# Usage:
#   make setup
#   make run-all RAW=data/raw/raw.csv
#   make intervals ENG=data/processed/engineered.csv

PY   ?= python
RAW  ?= data/raw/raw.csv
ENG  ?= data/processed/engineered.csv
OUT  ?= .
VENV ?= .venv

.PHONY: setup lint test precommit run-all prep intervals trend-demo dist microsim docs

setup:
	$(PY) -m venv $(VENV)
	. $(VENV)/bin/activate && pip install -U pip && pip install -r requirements.txt
	. $(VENV)/bin/activate && pre-commit install || true

lint:
	. $(VENV)/bin/activate && ruff check .
	. $(VENV)/bin/activate && black --check .

test:
	. $(VENV)/bin/activate && pytest -q

precommit:
	. $(VENV)/bin/activate && pre-commit run --all-files || true

# --- Orchestrated pipeline (uses your existing notebook logic via cli_nbwrap) ---
run-all:
	. $(VENV)/bin/activate && $(PY) -m cli_nbwrap run-all --raw $(RAW) --outbase $(OUT)

# Individual steps (optional helpers)
prep:
	. $(VENV)/bin/activate && $(PY) -m cli_nbwrap prep --raw $(RAW) --outdir data/processed

intervals:
	. $(VENV)/bin/activate && $(PY) -m cli_nbwrap intervals --eng $(ENG) --outdir data/processed

trend-demo:
	. $(VENV)/bin/activate && $(PY) -m cli_nbwrap trend-demo --eng $(ENG) --backlog data/processed/backlog_series.csv --out reports/last_year_by_team.csv

dist:
	. $(VENV)/bin/activate && $(PY) -m cli_nbwrap interval-distribution --eng $(ENG) --interval-col days_to_alloc --group case_type --outdir reports

# Docs (if you enabled MkDocs previously)
docs:
	. $(VENV)/bin/activate && mkdocs build --strict
