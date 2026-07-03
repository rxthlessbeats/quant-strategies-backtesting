# trading_rookie Codebase Wiki Rules

This repository uses an LLM-maintained wiki in `wiki/` as a persistent codebase knowledge layer.

## Source of Truth
1. Code files in this repository (`backend/`, `frontend/`, `scripts/`).
2. Diffs/commits/PR context.
3. Runtime evidence (terminal output, API behavior, diagnostics).
4. README text only if verified against code.

## Wiki Structure
- `wiki/pages/` - Synthesized knowledge pages.
- `wiki/index.md` - Content-oriented catalog.
- `wiki/log.md` - Append-only operation timeline.

No separate `raw_sources` folder is required for codebase knowledge mode.

## Required Page Sections
All `wiki/pages/*.md` files should contain:
- `## Claims`
- `## Evidence`
- `## Open Questions`
- `## Links`

## Operation Workflows

### Ingest (manual trigger)
Use when source input is code changes/modules/commits. Invoke `/wiki-ingest`.
1. Read `wiki/index.md`.
2. Inspect changed or target code paths.
3. Update existing pages first; create new pages only if needed.
4. Update `wiki/index.md`.
5. Append operation entry to `wiki/log.md`.

### Query (manual trigger)
Invoke `/wiki-query`.
1. Read `wiki/index.md` first.
2. Read relevant pages.
3. Verify claims against current code when needed.
4. Answer with citations to wiki pages and code paths.
5. If durable, save answer back into `wiki/pages/`, then update index/log.

### Lint (manual trigger)
Invoke `/wiki-lint`.
1. Check stale claims vs code.
2. Detect contradiction, weak links, orphan pages, missing canonical pages.
3. Create/update lint report page.
4. Update index/log.

## Guardrails
- Follow `.cursor/rules/wiki-ops-enforcement.mdc` during explicit wiki skill runs (`/wiki-ingest`, `/wiki-query`, `/wiki-lint`).
- Prefer canonical page updates over duplicates.
- Use explicit uncertainty when evidence is incomplete.
- Every wiki-changing ingest/query/lint operation must append to `wiki/log.md`.
- Claims about behavior must cite code paths and/or runtime output.

## Taxonomy Calibration
- Keep `wiki/index.md` grouped at minimum into:
  - `Architecture`
  - `Operations`
- Name recurring operational pages with stable prefixes:
  - `lint-report-YYYY-MM-DD.md`
  - `pilot-YYYY-MM-DD.md`
- Prefer extending existing architecture pages before introducing new top-level concepts.
