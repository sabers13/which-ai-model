# Which AI Model — one command per job.
#
# `make all` rebuilds results, page data and the page in replay mode (no key, no network).
# `make release-check` is what CI runs. All verification commands run locally.

SHELL := /bin/bash
.DEFAULT_GOAL := help

# Replay is the default for everything run through make; `bench-live` overrides it.
WAM_MODE ?= replay
export WAM_MODE

PYTHON := uv run python
PYTEST := uv run pytest

.PHONY: help setup fetch-data samples baseline test lint format typecheck scan-forbidden scan-secrets \
        facts doctor inventory bench-replay bench-live gate-demo rag-index results page-data \
        defect-baseline \
        page all release-check clean task-check cards prompt-library recheck-demo bench-live-v11 manifest

help: ## Show this help
	@grep -hE '^[a-zA-Z0-9_-]+:.*## ' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

setup: ## Create the venv and install exact locked versions
	uv sync

fetch-data: ## Download Banking77 and CLINC150 into data/raw/
	$(PYTHON) scripts/fetch_data.py

samples: ## Rebuild data/samples/ id lists from data/raw
	$(PYTHON) -m which_ai_model.bench.sample

baseline: ## Train and score the TF-IDF baseline into results/baseline
	$(PYTHON) -m which_ai_model.bench.baseline

defect-baseline: ## Train and score the T-DEFECT baselines into results/baseline
	$(PYTHON) -m which_ai_model.bench.defect_baseline

test: ## Run the test suite with the network blocked
	$(PYTEST) -q

lint: ## Ruff lint and format check
	uv run ruff check .
	uv run ruff format --check .

format: ## Apply Ruff formatting
	uv run ruff check --fix .
	uv run ruff format .

typecheck: ## Strict mypy on src/
	uv run mypy

scan-forbidden: ## Fail on a forbidden name in tracked files or recent commits
	$(PYTHON) scripts/scan_forbidden.py

scan-secrets: ## Fail on credential-looking strings
	$(PYTHON) scripts/scan_secrets.py

facts: ## Check every published number against the results file that proves it
	$(PYTHON) scripts/check_facts.py

doctor: ## Probe configured connections
	uv run wam doctor

inventory: ## Scan the five tools for LLM call sites
	$(PYTHON) -m which_ai_model.cli inventory src/which_ai_model/legacy \
		--json results/inventory/before.json --md results/inventory/before.md
	$(PYTHON) -m which_ai_model.cli inventory src/which_ai_model/tools \
		--json results/inventory/after.json --md results/inventory/after.md \
		--note '**Known blind spot.** The scanner counts an `invoke` it can see in the scanned module. Four of the five tools reach the model through a shared helper — `_template.ask` → `core.structured` (router, scope, drafter) and `rag.answer_question_with_metadata` (docs) — and those calls live in `core/` and `rag/`, outside this root, so only the two graph-level invokes above (agent, scope) are listed. The after picture is five tools on one shared structured call path, with `LangChain structured` output for all five: the row count is a lower bound, not a call total.'

bench-replay: ## Score recorded responses for every model x task
	$(PYTHON) -m which_ai_model.cli bench --mode replay \
		--config tests/fixtures/bench/config \
		--models tests/fixtures/bench/models.lock.toml \
		--recordings tests/fixtures/bench/recordings \
		--samples tests/fixtures/bench/samples \
		--items tests/fixtures/bench/items \
		--baseline tests/fixtures/bench/baseline-metrics.json \
		--recommendations results/runs/replay-demo/recommendations.json \
		--run-id replay-demo

bench-live: ## The one paid run; requires .env and the budget
	@if [ ! -f "$${WAM_ENV_FILE:-.env}" ]; then \
		echo "bench-live: no .env (or WAM_ENV_FILE) — refusing to start a paid run"; exit 1; \
	fi
	$(PYTHON) -m which_ai_model.cli bench --mode record --run-id live --rag-docs

gate-demo: ## Regression gate demonstration
	$(PYTHON) scripts/build_results.py --gate-only

# Task extension targets. `make all` rebuilds the cards and the prompt library from the
# committed evidence as well, and writes the manifest last so a deleted results/ comes
# back byte-identically.
task-check: ## Validate every task spec and its data (v1.1)
	@for spec in tasks/*.toml; do $(PYTHON) -m which_ai_model.cli task check "$$spec" || exit 1; done

cards: ## Build results/cards/<task>.{md,json} for the four v1 tasks and any spec run (v1.1)
	$(PYTHON) -m which_ai_model.report.cards --cards

prompt-library: ## Build results/prompt-library/ entries (v1.1)
	$(PYTHON) -m which_ai_model.report.cards --prompt-library

recheck-demo: ## Re-check demo: fresh answers gated against an earlier run
	@if [ "$(WAM_MODE)" = "record" ] && [ ! -f "$${WAM_ENV_FILE:-.env}" ]; then \
		echo "recheck-demo: no .env (or WAM_ENV_FILE) — refusing a paid run"; exit 1; \
	fi
	$(PYTHON) -m which_ai_model.recheck.demo
	@if [ "$(WAM_MODE)" = "record" ]; then \
		mkdir -p recordings/_run; \
		rm -rf recordings/_run/recheck-demo recordings/_run/recheck-demo-accepted; \
		cp -a results/runs/recheck-demo recordings/_run/recheck-demo; \
		cp -a results/runs/recheck-demo-accepted recordings/_run/recheck-demo-accepted; \
	fi

bench-live-v11: ## The v1.1 paid run
	@if [ ! -f "$${WAM_ENV_FILE:-.env}" ]; then \
		echo "bench-live-v11: no .env (or WAM_ENV_FILE) — refusing a paid run"; exit 1; \
	fi
	$(PYTHON) -m which_ai_model.cli bench --mode record --tasks defect \
		--run-id live-v1.1 --no-peak-wait --cap-usd "$${WAM_CAP_USD:-5}" \
		--recommendations results/runs/live-v1.1/recommendations.json
	@mkdir -p recordings/_run
	@rm -rf recordings/_run/live-v1.1
	cp -a results/runs/live-v1.1 recordings/_run/live-v1.1

rag-index: ## Build the docs corpus and index it in pgvector
	uv run wam rag index

results: baseline defect-baseline inventory ## Export results/ from recordings (replay mode)
	$(PYTHON) scripts/build_results.py

page-data: ## Build results/page/page-data.json
	$(PYTHON) -m which_ai_model.report.page_data

page: ## Build dist/which-ai-model/index.html
	$(PYTHON) -m which_ai_model.report.page

manifest: ## Rewrite results/MANIFEST.sha256 over every committed result
	$(PYTHON) scripts/build_results.py --manifest-only

all: results cards prompt-library page-data page manifest ## Rebuild results, cards and the page in replay mode

release-check: lint typecheck test scan-forbidden scan-secrets facts ## CI gate
	@test -f LICENSE && grep -q "MIT License" LICENSE || { echo "LICENSE (MIT) missing"; exit 1; }
	@test -f README.md || { echo "README.md missing"; exit 1; }
	@echo "release-check: green"

clean: ## Remove caches (never touches results/ or recordings/)
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -not -path "./.venv/*" -prune -exec rm -rf {} +
