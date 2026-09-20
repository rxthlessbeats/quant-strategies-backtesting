# Wiki Operation Log

Use one heading per operation with this format:

`## [YYYY-MM-DD] <operation> | <title>`

Where `<operation>` is one of: `ingest`, `query`, `lint`, `seed`, `pilot`.

---

## [2026-07-01] seed | Initialize code-as-source wiki

### Summary
- Created wiki structure and seeded core architecture pages.
- Seeded from code/runtime evidence only.

### Pages touched
- `wiki/index.md`
- `wiki/log.md`
- `wiki/pages/architecture-overview.md`
- `wiki/pages/backend-api-surface.md`
- `wiki/pages/backend-data-and-services.md`
- `wiki/pages/frontend-routing-and-composition.md`
- `wiki/pages/frontend-data-flow-and-state.md`
- `wiki/pages/runbook-local-dev.md`
- `wiki/pages/glossary.md`
- `wiki/pages/manual-ops-prompts.md`
- `wiki/pages/lint-report-2026-07-01.md`
- `wiki/pages/pilot-2026-07-01.md`

### Open questions
- Confirm whether scheduler should remain disabled by default in local workflow.
- Confirm preferred data provider for day-to-day development (`yahoo` vs `alpha_vantage`).

## [2026-07-01] lint | Baseline structural lint

### Summary
- Executed a manual lint pass using the wiki lint workflow.
- Structural status: no missing required sections, no orphan pages.

### Pages touched
- `wiki/pages/lint-report-2026-07-01.md`

### Contradictions / uncertainty
- Structural lint does not detect semantic contradictions; semantic drift must be checked during manual lint passes.

## [2026-07-01] pilot | Initial codebase wiki calibration

### Summary
- Ran pilot against three high-value areas:
  - backend API surface
  - backend provider/sync data flow
  - frontend chart workspace data orchestration
- Calibrated taxonomy/rules to keep `Architecture` and `Operations` as top-level index groups.

### Pages touched
- `wiki/pages/backend-api-surface.md`
- `wiki/pages/backend-data-and-services.md`
- `wiki/pages/frontend-data-flow-and-state.md`
- `wiki/pages/pilot-2026-07-01.md`
- `AGENTS.md`

### Contradictions / uncertainty
- README command guidance may differ from code over time; continue to treat README as non-authoritative unless validated.

## [2026-07-01] pilot | Skills + rule migration validation

### Summary
- Dry-ran `/wiki-ingest` workflow on wiki policy files by checking wiki-first order, evidence rules, and required bookkeeping.
- Dry-ran `/wiki-query` workflow by verifying query templates and citation requirements in wiki docs.
- Dry-ran `/wiki-lint` workflow by validating required section checks and lint-report expectations.
- Confirmed deleted script references were removed.

### Pages touched
- `AGENTS.md`
- `wiki/index.md`
- `wiki/pages/manual-ops-prompts.md`
- `wiki/pages/pilot-2026-07-01.md`
- `wiki/log.md`

### Contradictions / uncertainty
- Skill execution is validated through workflow dry-run and file-level policy checks; future live wiki operations should continue appending operation entries here.

## [2026-07-01] lint | Semantic lint and policy drift fixes

### Summary
- Ran semantic lint after skills-only migration and skill-triggered rule scope change.
- Fixed stale policy wording in `AGENTS.md` and `wiki/index.md`.
- Refreshed pilot, architecture scheduler semantics, and analysis API client mapping.
- Updated lint report with prioritized findings and remediation status.

### Pages touched
- `AGENTS.md`
- `wiki/index.md`
- `wiki/pages/lint-report-2026-07-01.md`
- `wiki/pages/pilot-2026-07-01.md`
- `wiki/pages/architecture-overview.md`
- `wiki/pages/backend-api-surface.md`
- `wiki/pages/manual-ops-prompts.md`
- `wiki/log.md`

### Contradictions / uncertainty
- Full sentence-level semantic verification was not run for every wiki page; remaining open coverage gaps (scheduler page, API examples page) are tracked as open questions.

## [2026-07-01] ingest | Wiki ops model and indicator pipeline

### Summary
- Ingested canonical wiki ops model (skills + skill-triggered rule, no helper scripts).
- Ingested indicator end-to-end flow from catalog fetch through chart API to workspace orchestration.
- Extended glossary with wiki skill terms and indicator query-string format.

### Pages touched
- `wiki/pages/manual-ops-prompts.md`
- `wiki/pages/frontend-data-flow-and-state.md`
- `wiki/pages/glossary.md`
- `wiki/index.md`
- `wiki/log.md`

### Contradictions / uncertainty
- Ingest scope inferred from recent wiki migration work; no explicit file/diff scope was provided in the user request.

## [2026-09-19] ingest | uv start command and Vercel branches

### Summary
- Canonical local start is now `uv run start` (FastAPI + Next.js).
- Python deps live in root `pyproject.toml` / `uv.lock`.
- Vercel is frontend-only: `main` production, `dev` preview.

### Pages touched
- `wiki/pages/runbook-local-dev.md`
- `wiki/log.md`

## [2026-09-19] ingest | Vercel FastAPI backend project

### Summary
- FastAPI can deploy as a second Vercel project with Root Directory `backend/`.
- Vercel uses `/tmp` SQLite and skips APScheduler.

### Pages touched
- `wiki/pages/runbook-local-dev.md`
- `wiki/log.md`
