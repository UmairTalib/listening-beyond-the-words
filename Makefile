# Common development commands. Run `make` to list them.
# Works with GNU Make 3.81 (macOS) and newer.

# PyTorch build: cpu on the Mac and in CI, cu126 on OMNI (export LBW_TORCH=cu126 there).
LBW_TORCH ?= cpu
# Optional dependency sets, separated by spaces (on OMNI: export LBW_EXTRAS=pipeline).
LBW_EXTRAS ?=

EXTRAS := --extra $(LBW_TORCH) $(addprefix --extra ,$(LBW_EXTRAS))
RUN := uv run --frozen $(EXTRAS)

.DEFAULT_GOAL := help
.PHONY: help env lock fmt lint typecheck test test-all test-gpu check hooks hooks-run clean

help: ## List the commands
	@awk 'BEGIN { FS = ":.*## " } /^[a-z-]+:.*## / { printf "  %-10s %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

env: ## Create or update the environment from uv.lock
	uv sync --locked $(EXTRAS)

lock: ## Update uv.lock after editing dependencies in pyproject.toml
	uv lock

fmt: ## Format the code and apply safe lint fixes
	$(RUN) ruff format .
	$(RUN) ruff check --fix .

lint: ## Check format and lint rules without changing files
	$(RUN) ruff format --check .
	$(RUN) ruff check .

typecheck: ## Type-check the package
	$(RUN) mypy

test: ## Run the fast tests (synthetic data, CPU)
	$(RUN) pytest

test-all: ## Run every CPU test, including slow ones
	$(RUN) pytest -m "not gpu"

test-gpu: ## Run the GPU tests (on OMNI, inside a GPU job)
	$(RUN) pytest -m gpu

check: lint typecheck test ## Run lint, type check and tests (CI also runs make hooks-run)

hooks: ## Install the git hooks (once per clone)
	$(RUN) pre-commit install

hooks-run: ## Run every git hook on every file
	$(RUN) pre-commit run --all-files

clean: ## Delete tool caches and build output (never data)
	rm -rf .mypy_cache .pytest_cache .ruff_cache build dist
	find src tests scripts -type d -name __pycache__ -prune -exec rm -rf {} +
