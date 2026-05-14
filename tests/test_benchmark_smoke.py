"""Smoke test for the throughput benchmark script.

Runs bench/throughput.py in --smoke mode (1 warmup, 5 measured runs) and
asserts the results JSON is non-empty.

Skip with: BENCH_SMOKE=0 pytest tests/test_benchmark_smoke.py
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

BENCH_SCRIPT = Path(__file__).parent.parent / "bench" / "throughput.py"
RESULTS_JSON = Path(__file__).parent.parent / "bench" / "results" / "throughput.json"


# The smoke test launches a subprocess that JIT-compiles the env (~30-90 s on
# CPU).  That's too slow for inclusion in the default regression suite — the
# JIT cache state can also occasionally deadlock under XLA, defeating
# pytest-timeout (which can't reach into XLA's C++ thread).  Opt-in only:
#
#     BENCH_SMOKE=1 pytest tests/test_benchmark_smoke.py
#
@pytest.mark.skipif(
    os.environ.get("BENCH_SMOKE") != "1",
    reason="set BENCH_SMOKE=1 to run the benchmark smoke test (~80 s subprocess)",
)
@pytest.mark.timeout(600)
def test_benchmark_smoke(tmp_path):
    """Run benchmark in smoke mode; assert non-empty JSON output."""
    # JIT compile of env._step_jit happens on first call inside the subprocess
    # (single-env path only in smoke mode).  On CPU this is ~3-5 min, so the
    # subprocess timeout has to cover one full cold-start compile.
    result = subprocess.run(
        [sys.executable, str(BENCH_SCRIPT), "--smoke"],
        capture_output=True,
        text=True,
        timeout=600,
        env={**os.environ, "JAX_PLATFORMS": "cpu"},
    )
    assert result.returncode == 0, (
        f"Benchmark exited with code {result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    assert result.stdout.strip(), "Benchmark produced no stdout output"

    # Verify JSON was written and is non-empty
    assert RESULTS_JSON.exists(), f"Results JSON not found at {RESULTS_JSON}"
    with open(RESULTS_JSON) as f:
        data = json.load(f)

    assert "cpu" in data, "Results JSON missing 'cpu' key"
    assert "system" in data, "Results JSON missing 'system' key"
    assert len(data["cpu"]) > 0, "CPU results dict is empty"
