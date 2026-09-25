# Project Architecture Rules

This file provides guidance to agents when working with code in this repository.

## Status

No architecture has been implemented. The project is a blank scaffold.

## Known Constraints

- Stack is undecided — `.gitignore` supports Node.js and Python but neither is committed to yet.
- Directory structure is pre-defined (`src/`, `tests/`, `deploy/`, `tools/`, `reference/`) — respect this layout in any architectural plan.
- `bob_sessions/` and `.bob/` are reserved for Bob AI tooling; do not repurpose them for application concerns.

## Planning Priority

Before any implementation planning, the following must be decided and documented in `AGENTS.md`:
1. Language and framework choice
2. Build/test/lint toolchain
3. Whether the stack is Node.js, Python, or polyglot
