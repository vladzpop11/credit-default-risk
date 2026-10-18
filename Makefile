IMAGE := credit-default-risk:dev
PORT  := 8000

.PHONY: setup format lint test check data split eda cv train serve docker-build docker-run

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
	@echo "train: not implemented yet (Project 1, step 6)"

serve:
	@echo "serve: not implemented yet (Project 1, step 8)"

docker-build:
	docker build --platform=linux/amd64 -t $(IMAGE) .

docker-run:
	@echo "docker-run: not implemented yet (Project 1, step 8)"
