# Person C session transcript — Prompts 7–9 (Claude Code)

**Date:** 2026-09-26
**Tool:** Claude Code (this Bob build has no confirmed project-level custom-mode
mechanism — see "Custom mode" note in `workflow/WORKFLOW.md`; steps 7–9 were
executed here instead of inside Bob's own UI, using the same MCP server and
`workflow/` package Bob would call through).
**Covers:** the "Next few hours" block of the Person C prompt plan — Prompt 7
(parity gate + approval checkpoint), Prompt 8 (Bob workflow spec), Prompt 9
(full mock rehearsal). Continues from Prompts 1–6, exported separately by Bob
to `bob_sessions/person-c/bob-task-519fe82157902d9ecbb01fb6fb19c531-2026-09-25.md`.

---

## Prompt 7 — generic parity gate, retry cap, approval checkpoint

Read `.bob/rules/person-c-guardrails.md` (write scope: `mcp_server/`,
`workflow/`, `contracts/`, `evidence/`, `submission/`, `bob_sessions/`,
`.github/`, `CODEOWNERS`, `README.md`, plus new `docs/context.md` /
`docs/interface-requests.md`).

Built:
- `workflow/state.py` — run state machine. `start_run()` / `get_run_status()`,
  10 fixed stages (`analysis` → ... → `pr`), state persisted at
  `workflow/runs/<run_id>/state.json` (gitignored).
- `workflow/gates.py` — `gate_step(run_id, gate, parity_result)`. Retry cap
  (3 attempts) enforced in code: PASS → `continue`; FAIL with attempt < 3 →
  `repair`; FAIL at attempt 3, or ERROR at any attempt → `escalate` and write
  `workflow/runs/<run_id>/escalation_<gate>.md` with every recorded attempt's
  stage table and diagnosis hints.
- `workflow/approval.py` — `record_approval()` / `assert_generation_approved()`.
  Generation is blocked until the `edge_readiness` checkpoint has an APPROVE;
  the most recent decision wins, so a later APPROVE overrides an earlier
  REJECT.
- Added `contracts/start_run.schema.json` and `contracts/get_run_status.schema.json`
  (additive — the two tools gate_step/record_approval's own schemas already
  referenced but that didn't exist yet).
- Exposed all four as MCP tools (`start_run_tool`, `get_run_status_tool`,
  `gate_step_tool`, `record_approval_tool`) on the FastMCP server, each
  schema-validated.
- 16 new tests in `workflow/tests/`, 4 in `mcp_server/tests/test_workflow_tools.py`.

Result: 36/36 tests pass, 10/10 contract examples validate. 6 commits pushed
to `person-c/work` (contracts, state, gates, approval, mcp exposure, tests).

## Prompt 8 — WORKFLOW.md and custom mode

Wrote `workflow/WORKFLOW.md`: the canonical step a–j spec (start_run →
analysis → Edge Readiness Report → human APPROVE/REJECT checkpoint →
isolated `../code2edge-deploy` worktree → host parity gate loop → MCU build
→ device parity gate loop → benchmark → commit/push/PR), grounded in the
real repo paths from `docs/workflow.md` (`src/pipeline/`, `src/firmware/`,
`tools/target/`, `reference/tiny-kws/`).

Searched the repo for a project-level custom-mode config
(`.bob/custom_modes.yaml`, `.bob/modes.json`, etc.) — found none, only the
three built-in `.bob/rules-{agent,ask,plan}/AGENTS.md` mode files. Could not
confirm from the filesystem whether this Bob build even supports defining a
new named mode; documented that in `WORKFLOW.md` and added a Fallback prompt
for Orchestrator/Agent mode so the workflow is runnable regardless of the
answer.

1 commit pushed.

## Prompt 9 — full mock rehearsal

Before starting: `git fetch origin` + merge (Person B's firmware commit had
landed on `main`; clean merge, no conflicts, pushed).

Ran the workflow end to end as a rehearsal, `mock_scenario="fail_then_pass"`
on the host gate, `"pass"` on the device gate:

| Step | Result |
|---|---|
| `start_run_tool` | `run_id = run-30027352` |
| Analysis | `profile_model_tool`, `inspect_pipeline_tool`, `check_target_tool` called sequentially (no confirmed parallel-subtask support) |
| Edge Readiness Report | fits=true, 650 KB SRAM headroom, no unsupported ops |
| **Human checkpoint** | shown to the user in chat, user replied **APPROVE**; `record_approval_tool` logged it |
| Generation | placeholder `preprocess.c/.h` + README written to `../code2edge-deploy/deploy/` (isolated worktree, branch `deploy/kws-stm32u585-run-30027352`) |
| Host parity gate | attempt 1 **FAIL** (framing) → `repair` → attempt 2 **PASS** → `continue` |
| MCU build | mock, no real `arduino-cli` invoked |
| Device parity gate | attempt 1 **PASS** → `continue` |
| Benchmark | 28.4 ms mean latency, 143.2 KB SRAM peak; predicted-vs-measured table written |
| Commit | local commit `be0d755` in the worktree — **not pushed, no PR** (mock rule) |

**Fix made mid-run:** the pre-commit hook's scope allowlist (from Prompt 1)
didn't include `deploy/`, which blocked the worktree commit — that folder is
the documented generation exception in the guardrails but predates it in the
hook. Asked the user, who chose to add `deploy/` to the hook's `ALLOWED`
regex (`.git/hooks/pre-commit`, local-only, not tracked) rather than bypass
with `--no-verify`.

Run finished with `status: completed`, all 10 stages `passed`, no escalation
triggered. Data: **mock** throughout.

---

### What's next (not done in this session)

- Remove the `../code2edge-deploy` worktree and its local `deploy/kws-stm32u585-run-30027352`
  branch (mock run — must not be pushed).
- Copy `workflow/runs/run-30027352/` artifacts into
  `evidence/runs/mock-rehearsal-2026-09-26/`.
- Add both this file and the Bob export to `bob_sessions/INDEX.md`.
- Open the PR bundling Prompts 7–9 work.
