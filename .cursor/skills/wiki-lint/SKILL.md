---
name: wiki-lint
description: Run a structural and semantic lint pass on the project wiki, report issues, and maintain lint bookkeeping. Use when the user asks to audit wiki quality, detect drift, contradictions, or missing links/pages.
disable-model-invocation: true
---

# Wiki Lint

Use this skill to keep `wiki/` accurate, navigable, and policy-compliant.

## Required Checks
- Stale claims compared to current code.
- Contradictions across pages.
- Weak evidence or missing citations for behavioral claims.
- Orphan pages and weak cross-linking.
- Repeated concepts that should be merged into canonical pages.
- Required page sections exist:
  - `## Claims`
  - `## Evidence`
  - `## Open Questions`
  - `## Links`

## Required Workflow
1. Read `wiki/index.md`.
2. Inspect relevant pages and verify selected claims against code/runtime evidence.
3. Create or update `wiki/pages/lint-report-YYYY-MM-DD.md`.
4. Update `wiki/index.md` if report links or taxonomy change.
5. Append a lint entry to `wiki/log.md`.

## Report Requirements
- Prioritized findings (high/medium/low).
- Concrete fix recommendations.
- Residual uncertainty.
- Pages touched.
