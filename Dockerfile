FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /usr/local/bin/uv

# xgboost needs the OpenMP runtime on Linux
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
COPY configs ./configs
RUN uv sync --frozen --no-dev

# The model is trained outside the image (make train) and copied in.
COPY models/model.joblib ./models/model.joblib
ENV MODEL_PATH=/app/models/model.joblib

# Files stay owned by root; the app only needs to read them.
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=2)"]

CMD ["uvicorn", "credit_default_risk.api:app", "--host", "0.0.0.0", "--port", "8000"]
