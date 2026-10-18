IMAGE := credit-default-risk:dev
PORT  := 8000

.PHONY: setup format lint test check data split eda cv train analysis evaluate serve docker-build docker-run docker-stop demo

setup:
	uv sync

format:
	uv run ruff format .
	uv run ruff check --fix .

lint:
	uv run ruff check .
	uv run ruff format --check .

test:
	uv run pytest

check: lint test

data:
	uv run python -m credit_default_risk.data

split:
	uv run python -m credit_default_risk.split

eda:
	uv run python -m credit_default_risk.eda

cv:
	uv run python -m credit_default_risk.cv

train:
	uv run python -m credit_default_risk.train

analysis:
	uv run python -m credit_default_risk.analysis

# Runs once: refuses to overwrite reports/test_metrics.json unless FORCE=1.
evaluate:
	uv run python -m credit_default_risk.evaluate

serve:
	uv run uvicorn credit_default_risk.api:app --host 127.0.0.1 --port $(PORT)

docker-build: models/model.joblib
	docker build --platform=linux/amd64 -t $(IMAGE) .

docker-run:
	docker run --rm -d --name credit-default-risk -p $(PORT):8000 $(IMAGE)

docker-stop:
	docker stop credit-default-risk

models/model.joblib:
	@echo "No trained model: run make data split train first." && exit 1

# Everything from scratch: download, split, train, build and start the API.
demo: setup data split train docker-build docker-run
	@echo "API running: http://localhost:$(PORT)/docs  (stop with: make docker-stop)"
