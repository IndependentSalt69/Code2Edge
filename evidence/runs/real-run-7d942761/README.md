Data: real (host parity, edge readiness, analysis) / mock (device parity, benchmark)

Run `run-7d942761`, the first real Code2Edge deployment run: end-to-end
`workflow/WORKFLOW.md` executed with real `profile_model`, `inspect_pipeline`,
`check_target`, and a real `run_parity_test(gate="host")` — 500 samples,
4 stages, 2500/2500 PASS, passed on the first attempt (0 repairs needed).

Device parity and benchmark are mocked (`mock_scenario="pass"`) — no
physical board was connected in this session. Both are clearly
`"source": "mock"` in their JSON, never blended with the real host data.
See `deploy/HARDWARE_VALIDATION_GUIDE.md` on the deploy branch for how to
replace them with real hardware data.

Deploy PR: https://github.com/IndependentSalt69/Code2Edge/pull/10
Deploy branch: `deploy/kws-stm32u585-run-7d942761`

## Files

| File | What it is |
|---|---|
| `edge_readiness.md` | Edge Readiness Report shown at the approval checkpoint (APPROVEd) |
| `parity_host_attempt1.json` | Host parity gate, attempt 1 — real, PASS |
| `parity_device_attempt1.json` | Device parity gate, attempt 1 — mock, PASS |
| `benchmark.json` | Benchmark — mock |
| `predicted_vs_measured.md` | Predicted-vs-measured table — mock |
| `state.json` | Full run state: every stage transition, gate decision, approval |
