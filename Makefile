PYTHON ?= python3
VENV   ?= .venv
BIN    := $(VENV)/bin

.DEFAULT_GOAL := help
.PHONY: help venv install test test-cov lint format typecheck check run config ping clean

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

$(BIN)/python:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip

venv: $(BIN)/python ## Create the virtualenv

install: $(BIN)/python ## Install the package + dev dependencies (editable)
	$(BIN)/pip install -e ".[dev]"
	@test -f .env || (cp .env.example .env && echo "Created .env from .env.example")

test: ## Run unit tests
	$(BIN)/pytest -m "not integration"

test-cov: ## Run unit tests with coverage
	$(BIN)/pytest -m "not integration" --cov=harness --cov-report=term-missing

lint: ## Lint with ruff
	$(BIN)/ruff check src tests

format: ## Auto-format and fix lint issues
	$(BIN)/ruff format src tests
	$(BIN)/ruff check --fix src tests

typecheck: ## Static type check with mypy
	$(BIN)/mypy

check: lint typecheck test ## Lint + typecheck + test

config: ## Print resolved configuration
	$(BIN)/harness config

ping: ## Smoke-test the configured LLM
	$(BIN)/harness ping

run: ## Run a task: make run TASK="fix the failing test"
	$(BIN)/harness run "$(TASK)"

clean: ## Remove caches and build artifacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov build dist src/*.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
