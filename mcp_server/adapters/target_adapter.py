"""
mcp_server/adapters/target_adapter.py

Adapter for Person B's hardware target outputs.

When CODE2EDGE_*_MODE=real, the MCP tools call functions here.
These stubs raise NotImplementedError with a clear message about what is needed.

Person B: implement the real functions here once tools/target/ is ready.
DO NOT write any files into src/firmware/, tools/target/, or reference/.
"""
from __future__ import annotations

from typing import Any


def run_check_target(model_file: str, arena_kb: float,
                     target_id: str = "STM32U585") -> dict[str, Any]:
    """Return check_target Output payload for the given model and arena size."""
    raise NotImplementedError(
        "waiting on Person B: implement check_target in target_adapter.py. "
        "Needs: tools/target/check_target.py to be importable and return a dict "
        "matching contracts/check_target.schema.json Output shape. "
        "Must report sram_budget_kb=786, flash_budget_kb=2048, fits bool, headroom_kb."
    )


def run_benchmark_target(model_file: str, n_inferences: int,
                          target_id: str = "STM32U585",
                          corpus_dir: str = "") -> dict[str, Any]:
    """Return benchmark_target Output payload."""
    raise NotImplementedError(
        "waiting on Person B: implement benchmark_target in target_adapter.py. "
        "Needs: connected STM32U585 over UART/USB, model flashed, and "
        "benchmark_target() function returning latency_ms, sram_peak_kb, "
        "flash_used_kb, predictions[], predicted_vs_measured[]."
    )
