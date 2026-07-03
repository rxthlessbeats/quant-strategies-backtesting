# Lint Report 2026-07-01

## Claims
- Structural lint remains clean: all wiki pages include required sections and are indexed.
- Semantic lint found stale policy wording after the skills-only migration and skill-triggered rule scope change.
- Fixes were applied during this lint pass for high/medium findings.

## Evidence
- Required sections present on all 10 pages under `wiki/pages/`.
- `wiki/index.md` lists all architecture and operations pages.
- Rule scope verified in `.cursor/rules/wiki-ops-enforcement.mdc` (`alwaysApply: false`, skill-triggered only).
- Scheduler behavior verified in `backend/app/main.py` and `backend/app/services/market_data_scheduler.py` with default `refresh_scheduler_enabled=False` in `backend/app/schemas/settings.py`.
- Analysis indicator endpoints verified in `backend/app/api/analysis_routes.py` and `frontend/src/lib/api.ts`.

## Open Questions
- Should lint policy enforce minimum evidence count per page?
- Should a dedicated scheduler page be added instead of architecture-overview notes?
- Should query-generated pages require explicit canonical owner page links?

## Links
- [../index.md](../index.md)
- [manual-ops-prompts](manual-ops-prompts.md)
- [pilot-2026-07-01](pilot-2026-07-01.md)

## Findings (High)

### H1 — Policy docs contradicted skill-triggered rule scope
- **Issue:** `wiki/index.md` and `AGENTS.md` implied always-on rule enforcement.
- **Fix applied:** Updated both to state enforcement applies only during explicit `/wiki-ingest`, `/wiki-query`, or `/wiki-lint` runs.

### H2 — Pilot page stale after skills migration
- **Issue:** Pilot still described manual prompts as the primary operating model.
- **Fix applied:** Updated pilot claims/evidence to reference project skills and skill-triggered rule behavior.

## Findings (Medium)

### M1 — Scheduler semantics were underspecified
- **Issue:** Architecture page said scheduler starts "optionally" without explaining startup guard conditions.
- **Fix applied:** Clarified lifespan always calls scheduler startup, but jobs run only when `refresh_scheduler_enabled` and `refresh_symbols` are set (disabled by default).

### M2 — Indicators API mapping was thin
- **Issue:** Glossary mentioned indicator catalog endpoint but API surface page lacked client mapping/examples.
- **Fix applied:** Extended `backend-api-surface.md` with frontend client functions and query examples.

### M3 — Weak cross-links on operations pages
- **Issue:** Lint/pilot/glossary pages had limited inbound links beyond index.
- **Fix applied:** Added cross-links from `architecture-overview.md`, `manual-ops-prompts.md`, and `backend-api-surface.md`.

## Findings (Low)

### L1 — Baseline lint report needed semantic refresh
- **Issue:** Original report captured structural baseline only.
- **Fix applied:** Replaced report content with semantic findings and remediation status.

### L2 — Open coverage gaps remain tracked
- Scheduler dedicated page, performance hotspot page, and API contract examples remain open questions (not blockers).

## Residual Uncertainty
- Full claim-by-claim semantic verification was not run for every wiki sentence; spot checks focused on policy drift and high-traffic API/architecture paths.

## Pages Touched (this lint pass)
- `AGENTS.md`
- `wiki/index.md`
- `wiki/log.md`
- `wiki/pages/lint-report-2026-07-01.md`
- `wiki/pages/pilot-2026-07-01.md`
- `wiki/pages/architecture-overview.md`
- `wiki/pages/backend-api-surface.md`
- `wiki/pages/manual-ops-prompts.md`
