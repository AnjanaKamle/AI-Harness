# AI Coding Harness — required targets: setup, run, test, clean.
# Credentials come from the environment (AI_API_KEY); nothing secret lives here.

PYTHON ?= python3
VENV   ?= .venv
BIN    := $(VENV)/bin
STAMP  := $(VENV)/.installed

# Passed through the environment (not the command line) so quotes in the task are safe.
export TASK
export REPO

.DEFAULT_GOAL := help
.PHONY: help setup run demo check test test-live clean

help: ## List targets
	@grep -E '^[a-z]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  %-8s %s\n", $$1, $$2}'

$(STAMP): pyproject.toml
	@$(PYTHON) -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else "Python 3.11+ is required (set PYTHON=/path/to/python3.11)")'
	$(PYTHON) -m venv $(VENV)
	$(BIN)/python3 -m pip install --quiet --upgrade pip
	$(BIN)/python3 -m pip install --quiet -e ".[dev]"
	@touch $(STAMP)

setup: $(STAMP) ## Create .venv and install dependencies
	@echo "Setup complete. Export AI_API_KEY before 'make run'."

run: $(STAMP) ## Run the harness: make run TASK="Fix the bug" [REPO=/path/to/repo]
	$(BIN)/python3 -m harness.main

demo: $(STAMP) ## Scripted demo on a sample repo (no model needed): make demo [TASK="..."]
	AI_PROVIDER=scripted $(BIN)/python3 -m harness.main

check: $(STAMP) ## Check the configured LLM provider (one minimal request)
	$(BIN)/python3 -m harness.main --check

test: $(STAMP) ## Run the test suite (deterministic, no network or credentials)
	$(BIN)/python3 -m pytest

test-live: $(STAMP) ## Optional live provider tests (needs AI_PROVIDER/AI_MODEL; AI_LIVE_TESTS=1)
	AI_LIVE_TESTS=1 $(BIN)/python3 -m pytest tests/live -m live -rs

clean: ## Remove generated artifacts (keeps source and .venv)
	rm -rf build dist .pytest_cache .coverage .coverage.* htmlcov coverage.xml .mypy_cache .ruff_cache
	find . -path ./$(VENV) -prune -o -name '__pycache__' -type d -prune -exec rm -rf {} +
	find . -path ./$(VENV) -prune -o -name '*.egg-info' -type d -prune -exec rm -rf {} +
