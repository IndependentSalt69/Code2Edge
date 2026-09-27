# evidence/mcp

Real MCP tool response captures from live Bob → MCP → hardware executions.

| File | Tool | Source | Date | Notes |
|------|------|--------|------|-------|
| `check_target_real_response.json` | `check_target_tool` | real | 2026-09-26 | Captured during the final integration run — real STM32U585 target profile via `tools/target/check_target.py` |

These files are **read-only evidence**. Never overwrite them with mock data.
New captures from future runs should be named `<tool>_real_response_<run_id>.json`.
