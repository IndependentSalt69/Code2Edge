# Project Coding Rules

This file provides guidance to agents when working with code in this repository.

## Status

The project has no source code yet. When implementing:

- Place application code in `src/`
- Place tests in `tests/` (co-locate or mirror `src/` structure — decide when first tests are written)
- Place deployment scripts in `deploy/`
- Place dev tooling in `tools/`

## Stack Decision Pending

`.gitignore` covers both Node.js and Python. Before writing any code, confirm the chosen language/framework and update `AGENTS.md` with build/lint/test commands.

## Secrets

Use `.env` (gitignored) for secrets. Never commit credentials.
