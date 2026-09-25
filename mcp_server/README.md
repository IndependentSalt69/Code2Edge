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

---

## Using the server from Bob

### Registration

The server is registered as **`code2edge`** in `.bob/mcp.json` at the project root.
Bob reads this file automatically — no global config changes needed.

> **`.bob/mcp.json` is gitignored** (the `.bob/` directory is in `.gitignore`).
> It is force-added to git so teammates can share it.
> If you clone the repo fresh, run `git checkout .bob/mcp.json` or copy it from the
> `person-c/work` branch — it will not appear in `git status` automatically.

### First-time setup for a new machine

1. Open `.bob/mcp.json` and update two fields to match your local environment:

   ```json
   "command": "/path/to/your/python3",
   "cwd": "/absolute/path/to/Code2Edge"
   ```

   On Windows with Miniconda this looks like:
   ```json
   "command": "C:\\Users\\YOU\\miniconda3\\python.exe",
   "cwd": "C:\\Users\\YOU\\...\\Code2Edge"
   ```

2. Make sure the Python interpreter has the dependencies:
   ```bash
   pip install -r mcp_server/requirements.txt
   ```

3. In Bob → Settings → MCP, click the **restart** icon next to `code2edge`.
   The status indicator should turn green.

### Reload after config changes

Any change to `.bob/mcp.json` (e.g. switching parity scenario) requires a server
restart. In Bob → Settings → MCP → restart icon next to `code2edge`, **or** close
and reopen the Bob window.

### Smoke-testing all tools from Bob

Once the server is connected (green), ask Bob (in Agent mode):

```
Call profile_model_tool with repo_path="." and model_file="reference/tiny-kws/requirements.txt"
```

Or ask it to run all tools in sequence:

```
Run the full Code2Edge analysis pipeline in mock mode:
1. profile_model_tool(repo_path=".", model_file="model.tflite")
2. inspect_pipeline_tool(repo_path=".", manifest_path="reference/tiny-kws/assets/metrics.json")
3. check_target_tool(model_file="model.tflite", arena_kb=128)
4. run_parity_test_tool(gate="host", attempt=1, corpus_dir="reference/corpus",
   ref_pipeline_path="src/pipeline", impl_pipeline_path="src/parity",
   run_id="demo-001")
5. benchmark_target_tool(model_file="model.tflite", n_inferences=100)
```

### Live smoke-test results (2026-09-25, mock mode)

All 6 tools were called through MCP and returned schema-valid responses:

| Tool | Status | Key result |
|------|--------|------------|
| `profile_model_tool` | ✅ PASS | DS-CNN, 24 922 params, 5.7 M MACs, int8 |
| `inspect_pipeline_tool` | ✅ PASS | 7 stages, KWS domain, all constants extracted |
| `check_target_tool` | ✅ PASS | fits=true, headroom=650 KB SRAM (arena=128 KB) |
| `run_parity_test_tool` (scenario=pass) | ✅ PASS | 7/7 stages PASS, pred_agreement=0.978 |
| `run_parity_test_tool` (scenario=fail_mel_scale) | ✅ FAIL as expected | first_divergent=mel, max_abs_diff=1.843, HTK/Slaney hint |
| `benchmark_target_tool` | ✅ PASS | 28.4 ms mean latency, 143 KB SRAM peak |

### Switching parity scenarios

Edit `CODE2EDGE_MOCK_PARITY_SCENARIO` in `.bob/mcp.json` and restart the server:

| Value | What it simulates |
|-------|-------------------|
| `pass` | All 7 stages within tolerance — green run |
| `fail_framing_shape` | Off-by-one frame count (49 vs 50); cascade FAIL from framing |
| `fail_mel_scale` | HTK vs Slaney filterbank mismatch; FAIL only at mel+log+normalize |
| `always_fail` | Framing shape bug that never resolves — tests escalation path |
| `fail_then_pass` | Attempt 1 FAIL, attempt 2 PASS (same run_id) — simulates repair loop |

