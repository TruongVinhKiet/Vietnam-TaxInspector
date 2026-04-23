"""
Simple /api/graph p95 benchmark utility.

Usage:
    python -m app.scripts.benchmark_graph_p95 --base-url http://127.0.0.1:8000 --runs 20
"""

from __future__ import annotations

import argparse
import statistics
import time
from urllib.parse import urlencode
from urllib.request import urlopen


def run_benchmark(base_url: str, runs: int, tax_code: str | None, depth: int) -> dict:
    latencies_ms = []
    for _ in range(max(1, runs)):
        params = {"depth": depth}
        if tax_code:
            params["tax_code"] = tax_code
        url = f"{base_url.rstrip('/')}/api/graph?{urlencode(params)}"
        started = time.perf_counter()
        with urlopen(url, timeout=30) as response:
            response.read()
            if response.status != 200:
                raise RuntimeError(f"Unexpected status: {response.status}")
        elapsed_ms = (time.perf_counter() - started) * 1000
        latencies_ms.append(elapsed_ms)

    p95 = statistics.quantiles(latencies_ms, n=100)[94] if len(latencies_ms) >= 2 else latencies_ms[0]
    return {
        "runs": len(latencies_ms),
        "min_ms": round(min(latencies_ms), 2),
        "avg_ms": round(sum(latencies_ms) / len(latencies_ms), 2),
        "p95_ms": round(p95, 2),
        "max_ms": round(max(latencies_ms), 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark /api/graph latency (p95).")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--tax-code", default=None)
    parser.add_argument("--depth", type=int, default=2)
    args = parser.parse_args()

    metrics = run_benchmark(args.base_url, args.runs, args.tax_code, args.depth)
    print("Graph API benchmark")
    print(f"runs={metrics['runs']} min={metrics['min_ms']}ms avg={metrics['avg_ms']}ms p95={metrics['p95_ms']}ms max={metrics['max_ms']}ms")


if __name__ == "__main__":
    main()
