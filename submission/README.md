# Submission Bundle

This directory contains the final project submission artifacts for **Code2Edge**.

---

## What Goes Here

| File / Directory | Description |
|-----------------|-------------|
| `report.pdf` | Final written project report (architecture, results, analysis) |
| `evidence_snapshot/` | Zipped copy of the `evidence/` directory at submission time |
| `demo_video.mp4` | Screen recording of live Bob → MCP → STM32U585 execution |
| `README.md` | This file — submission manifest |

---

## How to Build the Submission Bundle

```bash
# 1. Run the full test suite and confirm all pass
pytest tests/ mcp_server/tests/ workflow/tests/ \
  --ignore=tests/mcp --ignore=tests/target -q

# 2. Validate all contracts
python contracts/mcp/validate.py

# 3. Snapshot the evidence directory
zip -r submission/evidence_snapshot.zip evidence/

# 4. Copy the final audit document
cp docs/AUDIT.md submission/
```

---

## Key Evidence Locations (in the main repo)

| Artifact | Path |
|----------|------|
| Host parity report (500 samples, 2500/2500 PASS) | [`evidence/parity/host_parity_report.json`](../evidence/parity/host_parity_report.json) |
| Device parity UART dump | [`evidence/parity/device_parity_yes_uart.txt`](../evidence/parity/device_parity_yes_uart.txt) |
| Authoritative 50-run benchmark | [`evidence/benchmarks/stm32u585_benchmark_report.json`](../evidence/benchmarks/stm32u585_benchmark_report.json) |
| Model artifact validation | [`evidence/model/model_artifact_validation.json`](../evidence/model/model_artifact_validation.json) |
| Real deployment run record | [`evidence/runs/real-run-7d942761/`](../evidence/runs/real-run-7d942761/) |
| Repository audit | [`docs/AUDIT.md`](../docs/AUDIT.md) |
