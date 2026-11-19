"""The core stays dependency free: importing it loads no SDK (ADR 0002).

Runs in a fresh interpreter, because this test session has already imported
the SDKs for the adapter tests.
"""

import json
import subprocess
import sys

CORE_MODULES = [
    "aire",
    "aire.providers",
    "aire.providers.client",
    "aire.providers.local",
    "aire.providers.sdk_errors",
    "aire.runner",
    "aire.runner.async_runner",
    "aire.systems.rag",
    "aire.testing.fake_server",
    "aire.evaluator.engine",
    "aire.regression.gate",
    "aire.report",
]
THIRD_PARTY = ["openai", "anthropic", "google", "opentelemetry", "httpx2", "httpx", "pydantic"]


def test_importing_the_core_loads_no_third_party_package() -> None:
    code = (
        "import importlib, json, sys\n"
        f"for name in {CORE_MODULES!r}:\n"
        "    importlib.import_module(name)\n"
        f"roots = {THIRD_PARTY!r}\n"
        "print(json.dumps(sorted(m for m in sys.modules if m.split('.')[0] in roots)))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout
    assert json.loads(out) == []
