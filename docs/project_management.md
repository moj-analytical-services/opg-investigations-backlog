# Project Management & Collaboration

## Branching
- `main` is protected. Use feature branches: `feat/`, `fix/`, `refactor/`.
- Open PRs with a clear summary, tests, and screenshots if relevant.

## Reviews
- 1 reviewer minimum. CODEOWNERS auto-requests leads.
- Use the PR template; link issues; keep changeset focused.

## Issues & Boards
- Use templates for bugs and features.
- Create a GitHub Project board with columns _Backlog → In Progress → Review → Done_.
- Tag issues with `data`, `model`, `infra`, `docs`, `good-first-issue`.

## Releases
- Semantic versioning via annotated tags.
- Changelog in release notes.

## Quality Gates
- CI green (lint + tests) before merge.
- Datasets must pass the schema checks.
