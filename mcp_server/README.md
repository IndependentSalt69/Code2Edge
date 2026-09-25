# Code2Edge MCP Server

FastMCP server exposing the Code2Edge hardware-aware deployment tools over **stdio transport**.

---

## Quick start

```bash
# Install dependencies (Python 3.10+)
pip install -r mcp_server/requirements.txt

# Run the server (stdio — connect via Bob's MCP config)
python -m mcp_server.server
```

---

## Tools

| Tool | Description |
|------|-------------|
| `profile_model_tool` | Analyse model architecture, layers, MACs, quantization |
| `inspect_pipeline_tool` | Trace Python preprocessing pipeline; return runtime constants |
| `check_target_tool` | Memory budget check for STM32U585 (786 KB SRAM / 2 MB flash) |
| `run_parity_test_tool` | Stage-wise differential parity: Python ref vs generated C++ |
| `benchmark_target_tool` | On-device latency + memory measurement |
| `quantize_model_tool` | Post-training int8 quantization (optional) |

---

## Environment variables

### Per-tool mode switch

```
CODE2EDGE_PROFILE_MODEL_MODE     = mock | real   (default: mock)
CODE2EDGE_INSPECT_PIPELINE_MODE  = mock | real
CODE2EDGE_CHECK_TARGET_MODE      = mock | real
CODE2EDGE_RUN_PARITY_TEST_MODE   = mock | real
CODE2EDGE_BENCHMARK_TARGET_MODE  = mock | real
CODE2EDGE_QUANTIZE_MODEL_MODE    = mock | real
```

**`mock`** — loads a fixture from `mcp_server/mocks/<tool>.json`, stamps a fresh `run_id` and `timestamp`, and sets `source="mock"`.

**`real`** — delegates to the matching adapter:
- `pipeline_adapter.py` (Person A's pipeline and parity outputs)
- `target_adapter.py` (Person B's hardware target outputs)

> **Never falls back silently from real to mock.** If the adapter raises `NotImplementedError`, the tool returns `{"status": "ERROR", "error_message": "<what Person A/B needs to implement>"}`.

### Parity scenario (mock mode only)

```
CODE2EDGE_MOCK_PARITY_SCENARIO = pass | fail_then_pass | always_fail | fail_mel_scale | fail_framing_shape
```

| Scenario | Behaviour |
|----------|-----------|
| `pass` | All 7 stages PASS; prediction_agreement ≥ 0.95 |
| `always_fail` | Every call: FAIL at `framing` stage (shape 49 vs 50 frames); diagnosis hints present |
| `fail_framing_shape` | Same as `always_fail` — explicit framing off-by-one |
| `fail_mel_scale` | FAIL at `mel` stage (constant scale error ~1.84, HTK vs Slaney filterbank); pre-mel stages PASS |
| `fail_then_pass` | First call per `run_id` → FAIL (framing shape); second call → PASS. Simulates successful repair loop. |

---

## Running tests

```bash
# From repo root
pip install pytest pytest-asyncio
pytest mcp_server/tests/ -v
```

Expected: **all tests pass**.

---

## Adding real adapters

When Person A delivers `dump_reference.py` output and the C++ binary:

1. Open `mcp_server/adapters/pipeline_adapter.py`
2. Replace the `NotImplementedError` body with real logic
3. Set `CODE2EDGE_RUN_PARITY_TEST_MODE=real` (and other tools as ready)

When Person B delivers `tools/target/check_target.py`:

1. Open `mcp_server/adapters/target_adapter.py`
2. Import and call `tools.target.check_target` from there
3. Set `CODE2EDGE_CHECK_TARGET_MODE=real`

**Never write into `src/pipeline/`, `src/firmware/`, `reference/`, or `tools/`.**
All adapter logic stays inside `mcp_server/adapters/`.

---

## Schema validation

Every tool validates its output against the matching schema in `contracts/` before returning.
A validation failure surfaces as a tool error, not a silent bad payload.

Run the full contracts validator independently:

```bash
python contracts/validate.py
```
