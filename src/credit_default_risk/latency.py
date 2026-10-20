"""Measure end-to-end /predict latency against a running API (e.g. after make docker-run).

python -m credit_default_risk.latency [URL]   default http://localhost:8000

Sends the example client sequentially after a warm-up. The tracked record,
reports/latency.json, is only updated when the API serves the evaluated model; any other
model's measurement goes to models/latency.json (git-ignored).
"""

import json
import math
import platform
import statistics
import sys
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

from credit_default_risk import artifacts
from credit_default_risk.config import load_config, project_path

N_REQUESTS = 50
N_WARMUP = 5
EXAMPLE = project_path("tests/example_client.json")


def timed_request(url: str, body: bytes) -> float:
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=10) as resp:
        resp.read()
    return (time.perf_counter() - start) * 1000


def output_path(cfg: dict, evaluated_model: bool) -> Path:
    if evaluated_model:
        return artifacts.test_metrics_path(cfg).with_name("latency.json")
    return artifacts.model_path(cfg).with_name("latency.json")


def main() -> None:
    base = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    body = Path(EXAMPLE).read_bytes()
    for _ in range(N_WARMUP):
        timed_request(f"{base}/predict", body)
    ms = sorted(timed_request(f"{base}/predict", body) for _ in range(N_REQUESTS))
    with urllib.request.urlopen(f"{base}/health", timeout=10) as resp:
        health = json.load(resp)
    result = {
        "measured_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "url": base,
        "client_machine": f"{platform.system()} {platform.machine()}",
        "requests": N_REQUESTS,
        "mean_ms": statistics.mean(ms),
        "p50_ms": statistics.median(ms),
        "p95_ms": ms[math.ceil(0.95 * len(ms)) - 1],  # nearest rank
        "max_ms": ms[-1],
        "model_sha256": health["model_sha256"],
        "evaluated_model": health["evaluated_model"],
    }
    out = output_path(load_config(), health["evaluated_model"])
    artifacts.write_json(out, result)
    print(
        f"mean {result['mean_ms']:.1f} ms, p50 {result['p50_ms']:.1f} ms, "
        f"p95 {result['p95_ms']:.1f} ms over {N_REQUESTS} requests -> {out}"
    )


if __name__ == "__main__":
    main()
