# AGENTS.md

This file provides guidance to agents when working with code in this repository.

## Project Status

**Code2Edge is an empty scaffold.** All source directories (`src/`, `tests/`, `deploy/`, `tools/`, `reference/`, `bob_sessions/`) contain only `.gitkeep` placeholders. No build system, test framework, or application code exists yet.

## Directory Layout (Intended)

| Path | Purpose |
|------|---------|
| `src/` | Application source code |
| `tests/` | Test files |
| `deploy/` | Deployment configuration/scripts |
| `tools/` | Developer tooling/scripts |
| `reference/` | Reference materials |
| `docs/` | Architecture, problem statement, feasibility docs (stub templates) |
| `bob_sessions/` | Bob AI workspace session data |
| `.bob/` | Bob configuration and metadata |

## Stack Hints (from `.gitignore`)

The `.gitignore` covers both **Node.js** (`node_modules/`, `dist/`, `build/`) and **Python** (`venv/`, `__pycache__/`, `*.py[cod]`). The actual language/framework has not yet been decided or implemented.

## Commands

No build, lint, test, or run commands exist yet. Once a stack is chosen, document them here.

## Code Style

No linter, formatter, or style config exists yet. Once established, document rules here.

## Key Gotchas

- The `bob_sessions/` directory is for Bob AI session metadata — do not store application data there.
- Docs in `docs/` are stub templates (`# Problem Statement`, `# Architecture`, `# Feasibility Study`) with empty bullet points — treat them as planning documents to fill in, not authoritative references.
- `.env` and `.env.local` are gitignored; create them locally for secrets.
