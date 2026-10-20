# syntax=docker/dockerfile:1
# Base images are pinned by digest; bump them deliberately (Dependabot opens PRs).
ARG PYTHON_IMAGE=python:3.12-slim@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.13.1@sha256:4a1f3666816dafadd20ceb71eac9a7d56d7ef30279f7d64048519ab901ada336

FROM ${UV_IMAGE} AS uv

# --- build stage: resolve and install the runtime dependencies only -----------------
FROM ${PYTHON_IMAGE} AS build
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app

# Dependencies first, from the lock file only, so code edits don't reinstall them.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-default-groups --no-install-project

COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-default-groups

# --- runtime stage: no uv, no build cache --------------------------------------------
FROM ${PYTHON_IMAGE}
# xgboost needs the OpenMP runtime on Linux
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=/app/models/model.joblib
WORKDIR /app

COPY --from=build /app /app
COPY configs ./configs
# The model is trained outside the image (make train) and copied in, with the record of
# the evaluated model so /health can say whether this is that model.
COPY models/model.joblib ./models/model.joblib
COPY reports/test_metrics.json ./reports/test_metrics.json

# Files stay owned by root; the app only needs to read them.
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=2)"]

# --limit-concurrency sheds load with 503s instead of piling up connections.
CMD ["uvicorn", "credit_default_risk.api:app", "--host", "0.0.0.0", "--port", "8000", \
     "--limit-concurrency", "64", "--timeout-keep-alive", "5"]
