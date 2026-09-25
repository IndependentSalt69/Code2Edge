# Interface Requests

Person C uses this file to ask Person A, Person B, or the repo owner for changes
in territory that is read-only for C. Do NOT make direct edits to those paths —
add a row here and flag it in the relevant chat/PR instead.

| Date | From | To | Request | Status |
|------|------|----|---------|--------|
| 2026-09-26 | C | Team | **Folder layout:** `AGENTS.md` says application code goes in `src/`, but the team context document assumes `pipeline/` (A) and `firmware/` (B) as top-level dirs. Proposed resolution: A owns `reference/` + `src/pipeline/`; B owns `src/firmware/` + `tools/target/`; `tests/` and `deploy/` are shared (deploy branch only). Needs explicit confirmation from A and B before anyone writes code. | Open |
| 2026-09-26 | C | Owner of `.gitignore` | **`.bob` is gitignored:** The current `.gitignore` has a bare `.bob` rule, so any new file created under `.bob/` will be silently untracked. C needs to commit two shared Bob config files: `.bob/mcp.json` and `.bob/custom_modes.yaml`. C will use `git add -f` for those two paths only. Preferred fix: change the rule to `.bob/*` with explicit exceptions (e.g. `!.bob/mcp.json`, `!.bob/custom_modes.yaml`), or remove the `.bob` ignore entirely. Please advise. | Open |
