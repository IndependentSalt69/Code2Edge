MOCK DATA — rehearsal run

Run `run-30027352`, executed 2026-09-26 to rehearse the end-to-end
`workflow/WORKFLOW.md` deployment loop before real Person A/B adapters were
wired in. `mock_scenario="fail_then_pass"` on the host gate,
`"pass"` on the device gate. No hardware, no real generated code — see
`bob_sessions/person-c/claude-code-session-prompts-7-9-2026-09-26.md` for
the full transcript.

## Files

| File | What it is |
|---|---|
| `edge_readiness.md` | Edge Readiness Report shown to the human at the approval checkpoint (APPROVEd) |
| `parity_host_attempt1.json` | Host parity gate, attempt 1 — FAIL at `framing` |
| `parity_host_attempt2.json` | Host parity gate, attempt 2 — PASS (repair loop succeeded) |
| `parity_device_attempt1.json` | Device parity gate, attempt 1 — PASS |
| `gate_decisions.json` | `gate_step` decisions for both gates: repair -> continue (host), continue (device) |
| `predicted_vs_measured.md` | Benchmark table: predicted (from `check_target`/`profile_model`) vs measured (from `benchmark_target`) |

Run finished `status: completed`, all 10 workflow stages `passed`, no
escalation triggered. Deploy worktree and its local branch were deleted
after the rehearsal — nothing from it was pushed or opened as a PR.
