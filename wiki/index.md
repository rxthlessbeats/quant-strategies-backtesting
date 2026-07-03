# Codebase Wiki Index

This index is the entrypoint for codebase knowledge queries. Read this file first, then open linked pages.

## Architecture
- [architecture-overview](pages/architecture-overview.md) - System-level map of backend/frontend layers and runtime boundaries.
- [backend-api-surface](pages/backend-api-surface.md) - FastAPI routers, endpoints, and response surface.
- [backend-data-and-services](pages/backend-data-and-services.md) - Data models, providers, sync behavior, and service layer.
- [frontend-routing-and-composition](pages/frontend-routing-and-composition.md) - Next.js route tree and major page composition.
- [frontend-data-flow-and-state](pages/frontend-data-flow-and-state.md) - API client usage, client state, and chart workflow behavior.
- [runbook-local-dev](pages/runbook-local-dev.md) - Current code-backed local run/debug workflow.
- [glossary](pages/glossary.md) - Domain and codebase vocabulary.

## Operations
- [manual-ops-prompts](pages/manual-ops-prompts.md) - Wiki ops model plus `/wiki-ingest`, `/wiki-query`, and `/wiki-lint` prompt templates.
- [lint-report-2026-07-01](pages/lint-report-2026-07-01.md) - Wiki health-check baseline and semantic lint findings.
- [pilot-2026-07-01](pages/pilot-2026-07-01.md) - Initial pilot and calibration notes.

## Conventions
- Source of truth is code, diffs, and runtime evidence.
- README claims are non-authoritative unless validated against code.
- `wiki/log.md` is append-only chronological history.
- Wiki operation policy in `.cursor/rules/wiki-ops-enforcement.mdc` applies during explicit `/wiki-ingest`, `/wiki-query`, or `/wiki-lint` skill runs.
