# Person C session transcript — Prompt 11 (Claude Code)

**Date:** 2026-09-26
**Tool:** Claude Code (no confirmed Bob custom-mode mechanism; see `workflow/WORKFLOW.md`'s "Custom mode" note).
**Covers:** the real deployment run (Prompt 11), preceded by wiring `run_parity_test`'s host gate to real data (a prerequisite blocker found while checking Prompt 11's own "every tool must be real" gate).

---

## Pre-run: wiring `run_parity_test(gate="host")` to real data

Prompt 11 opens with "check that every tool is set to real; if any are
missing, stop." Checked: `profile_model`, `inspect_pipeline`, `check_target`,
`benchmark_target` were real (prior prompts); `run_parity_test` and
`quantize_model` were still `NotImplementedError` stubs.

`quantize_model` was left alone (optional tool, model already frozen int8).
`run_parity_test(gate="host")` was the real blocker — but by this point
Person A had independently produced a full 500-sample, 5-stage corpus-wide
host parity report (`evidence/parity/host_parity_report.json`, via
`tools/run_host_parity.py`): 2500/2500 stage-evaluations PASS.

Wired `pipeline_adapter.run_run_parity_test(gate="host")` to aggregate that
report into the contract's `StageResult` shape (worst-case max_abs_diff,
mean of per-sample mean_abs_diff, worst cosine similarity, which sample/index
was worst) for the 4 real stages (`fft`, `mel`, `log`, `normalize` — same
mapping as `inspect_pipeline`). Also:

- Made `contracts/run_parity_test.schema.json`'s `end_to_end` fields
  nullable (additive) — the real harness only measures preprocessing-stage
  parity, not corpus-wide model prediction agreement, so those fields are
  honestly `null` rather than fabricated.
- Generated a real per-sample CSV artifact from the same report, with
  labels cross-referenced from `reference/corpus_manifest.json`.
- `gate="device"` still raises `NotImplementedError`: no corpus-wide
  on-device sweep exists, only a single-fixture (yes.wav) smoke test.
- Logged in `docs/interface-requests.md`: `tools/run_host_parity.py`
  hardcodes `src/pipeline/feature_extraction.c` as the file under test —
  not parameterized by path, so it can't independently re-verify a copy of
  the pipeline sitting in a deploy worktree.

3 commits (contracts, adapter, tests), all pushed and merged to `main`
before starting the real run.

## The real run — `run-7d942761`

| Step | Result |
|---|---|
| `start_run_tool` | `run_id = run-7d942761` |
| Analysis | `profile_model`, `inspect_pipeline`, `check_target` — all real |
| Edge Readiness | fits=true, 598 KB SRAM headroom, model 168.2 KB (real, from `model_data.h`'s documented size — first attempt used the wrong file (`model_data.c`, the 1MB *source* text) and got a nonsense 1061.9 KB; caught and corrected before finalizing the report) |
| **Human checkpoint** | shown to the user, replied **APPROVE** |
| Generation | Copied Person A's already-verified `feature_extraction.c/.h` + `model_data.c/.h` into `../code2edge-deploy/deploy/` — no independent code-generation step in this workflow, Bob orchestrates and proves, doesn't author new C++ |
| Host parity | attempt 1 — **PASS**, real (500 samples, 4 stages, 2500/2500) → **continue**, 0 repairs |
| MCU build | Not independently re-compiled (no board connected); referenced Person B's existing evidence (compiles clean, 68.75 KB flash / 35.66 KB SRAM) |
| Device parity | attempt 1 — **mock** (`mock_scenario="pass"`), clearly `"source": "mock"` → continue |
| Benchmark | **mock**, clearly `"source": "mock"` |
| Commit + push + PR | 4 commits on `deploy/kws-stm32u585-run-7d942761`, pushed, PR opened: [#10](https://github.com/IndependentSalt69/Code2Edge/pull/10) |

Included a `deploy/HARDWARE_VALIDATION_GUIDE.md` with exact steps for
whoever has the physical Arduino UNO Q to replace the two mocked sections
(device parity, benchmark) with real data on the same PR.

Run finished `status: completed`. Deploy worktree left in place per
instructions (not a mock rehearsal this time — real artifacts, real PR).

### What's still open

- `run_parity_test(gate="host")`'s real harness verifies identity with
  Person A's committed pipeline, not an independent re-verification of the
  deploy worktree's copy (harness isn't parameterized by source path yet).
- Device parity and benchmark remain mocked until someone runs them on the
  physical board.
- `evidence/runs/real-run-7d942761/` holds the same artifacts as the deploy
  branch, for the static evidence page (Prompt 12) to read from later.
