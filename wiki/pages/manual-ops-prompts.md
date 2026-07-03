# Manual Ops Prompts

## Claims
- Reusable prompts reduce wiki maintenance drift and ensure index/log updates are not skipped.
- Wiki maintenance runs through three explicit project skills: ingest, query, and lint.
- Policy enforcement is skill-triggered via `.cursor/rules/wiki-ops-enforcement.mdc` (`alwaysApply: false`).
- Helper PowerShell scripts were removed; skills are the sole operation entrypoints.

## Evidence
- Project skills: `.cursor/skills/wiki-ingest/SKILL.md`, `.cursor/skills/wiki-query/SKILL.md`, `.cursor/skills/wiki-lint/SKILL.md`.
- Skill-triggered rule: `.cursor/rules/wiki-ops-enforcement.mdc`.
- Policy companion: `AGENTS.md` (page structure, guardrails, taxonomy).
- Bookkeeping targets: `wiki/index.md`, `wiki/log.md`.
- Prior script removal validated in `wiki/log.md` (skills + rule migration entries).

## Open Questions
- Should we add dedicated prompts for release-note generation and refactor impact summaries?
- Should `AGENTS.md` be reduced to a pointer doc now that skills + rule carry most policy?

## Links
- [runbook-local-dev](runbook-local-dev.md)
- [architecture-overview](architecture-overview.md)
- [lint-report-2026-07-01](lint-report-2026-07-01.md)
- [pilot-2026-07-01](pilot-2026-07-01.md)

## Wiki Ops Model

| Skill | When to use | Required bookkeeping |
|---|---|---|
| `/wiki-ingest` | Persist code/diff knowledge into `wiki/pages/` | Update pages, `wiki/index.md` (if structure changes), append `wiki/log.md` |
| `/wiki-query` | Answer questions wiki-first with code verification | Append `wiki/log.md` only if wiki content is edited |
| `/wiki-lint` | Audit drift, contradictions, links, required sections | Update/create `lint-report-YYYY-MM-DD.md`, `wiki/index.md` (if needed), append `wiki/log.md` |

All three skills require evidence-backed claims (code paths and/or runtime output). README-only claims are insufficient unless verified.

## Ingest Prompt (code/diff context)
```text
/wiki-ingest
Ingest this codebase context into the wiki:
<files / diff / commit range / module scope>

Requirements:
1) Read wiki/index.md first.
2) Update existing relevant pages in wiki/pages before creating new pages.
3) Claims must cite code paths and/or runtime output.
4) Update wiki/index.md.
5) Append a log entry to wiki/log.md with:
   - operation date/title
   - pages touched
   - contradictions/uncertainty

Return:
- changed pages
- top claims added/updated
- unresolved questions
```

## Query Prompt (wiki-first)
```text
/wiki-query
Answer this from the codebase wiki first:
<question>

Requirements:
1) Start with wiki/index.md.
2) Read relevant wiki pages.
3) Verify key claims against current code paths.
4) Cite wiki pages and code paths.
5) Mark uncertainty explicitly.
6) If durable, suggest saving answer as wiki page.
```

## Lint Prompt
```text
/wiki-lint
Run a wiki lint pass.

Check for:
- stale claims vs current code
- contradictions across pages
- orphan pages/no inbound links
- missing cross-links
- repeated concepts lacking canonical pages

Requirements:
1) Create/update lint-report-<date>.md in wiki/pages.
2) Propose fixes with priority.
3) Update wiki/index.md.
4) Append lint entry to wiki/log.md.
```
