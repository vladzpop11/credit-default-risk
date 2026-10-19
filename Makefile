IMAGE ?= credit-default-risk:dev
NAME  ?= credit-default-risk
PORT  ?= 8000

.PHONY: help setup format lint test check data split eda cv train analysis evaluate \
        uncertainty verify reproduce-check test-model serve docker-build docker-run docker-stop \
        docker-smoke latency demo

help:  ## List the targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-16s %s\n", $$1, $$2}'

setup:  ## Install dependencies from uv.lock
	uv sync --locked

format:  ## Format and auto-fix lint
	uv run ruff format .
	uv run ruff check --fix .

lint:  ## Ruff lint and format check
	uv run ruff check .
	uv run ruff format --check .

test:  ## Run the test suite
	uv run pytest

check: lint test  ## Lint + tests

data:  ## Download the UCI data (SHA-256 verified)
	uv run python -m credit_default_risk.data

split:  ## Stratified 80/20 train/test split
	uv run python -m credit_default_risk.split

eda:  ## Exploratory plots (training split only)
	uv run python -m credit_default_risk.eda

cv:  ## Cross-validate all candidate models
	uv run python -m credit_default_risk.cv

train:  ## Train the chosen model (refuses to replace the evaluated one unless FORCE=1)
	uv run python -m credit_default_risk.train

analysis:  ## Threshold, error and fairness analysis (training split, out-of-fold)
	uv run python -m credit_default_risk.analysis

evaluate:  ## One-time test evaluation (refuses to run again unless FORCE=1)
	uv run python -m credit_default_risk.evaluate

uncertainty:  ## Bootstrap intervals for the recorded test result (no new decisions)
	uv run python -m credit_default_risk.uncertainty

verify:  ## Is models/model.joblib the evaluated file? (exit 1 if not)
	uv run python -m credit_default_risk.verify model

reproduce-check:  ## Is the local retrain within tolerance of the evaluated model? (exit 1 if not)
	uv run python -m credit_default_risk.verify reproduce

test-model:  ## Behavioural tests of models/model.joblib (fail, not skip, if it is missing)
	REQUIRE_MODEL=1 uv run pytest tests/test_model_behaviour.py

serve:  ## Run the API locally without Docker
	uv run uvicorn credit_default_risk.api:app --host 127.0.0.1 --port $(PORT)

docker-build: models/model.joblib  ## Build the API image
	docker build --platform=linux/amd64 -t $(IMAGE) .

docker-run:  ## Start the API container on localhost:$(PORT) (replaces a running one)
	-@docker rm -f $(NAME) >/dev/null 2>&1
	docker run --rm -d --name $(NAME) -p 127.0.0.1:$(PORT):8000 $(IMAGE)

docker-stop:  ## Stop the API container
	docker stop $(NAME)

docker-smoke: docker-run  ## Start the container, check /health and /predict, stop it
	@for i in $$(seq 1 30); do curl -sf localhost:$(PORT)/health >/dev/null && break; sleep 1; done
	curl -sf localhost:$(PORT)/health
	@echo
	curl -sf -X POST localhost:$(PORT)/predict -H 'Content-Type: application/json' \
	  -d @tests/example_client.json
	@echo
	docker stop $(NAME)

latency:  ## Measure /predict latency against the running API (reports/ only for the evaluated model)
	uv run python -m credit_default_risk.latency http://localhost:$(PORT)

# Trains only when no model exists, so make demo never replaces the evaluated model.
models/model.joblib:
	uv run python -m credit_default_risk.train

demo: setup data split models/model.joblib docker-build docker-run  ## Everything from scratch, then the API
	@echo "API running: http://localhost:$(PORT)/docs  (stop with: make docker-stop NAME=$(NAME))"
