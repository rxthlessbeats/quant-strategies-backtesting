---
name: wiki-ingest
description: Ingest codebase changes into the project wiki using a wiki-first workflow with evidence-backed claims and mandatory index/log updates. Use when the user asks to ingest new code context, diffs, modules, or implementation changes into wiki pages.
disable-model-invocation: true
---

# Wiki Ingest

Use this skill to persist new codebase knowledge into `wiki/pages/` while keeping `wiki/index.md` and `wiki/log.md` consistent.

## Inputs
- Scope: files, folders, diff, commit range, or module to ingest.
- Optional focus: architecture, API, frontend flow, runbook, or glossary.

## Required Workflow
1. Read `wiki/index.md` first.
2. Inspect source-of-truth evidence from code and/or runtime output.
3. Update existing pages first; create a new page only if no canonical page fits.
4. Ensure every touched page has:
   - `## Claims`
   - `## Evidence`
   - `## Open Questions`
   - `## Links`
5. Update `wiki/index.md` for any new/renamed pages.
6. Append an operation entry to `wiki/log.md`.

## Evidence Rules
- Prefer repository code paths and runtime behavior evidence.
- Treat README text as non-authoritative unless verified against code.
- State uncertainty explicitly when evidence is incomplete or conflicting.

## Required Output
- List of wiki pages changed.
- Top claims added/updated.
- Contradictions or uncertainty tracked.
- Confirmation that `wiki/index.md` and `wiki/log.md` were updated.
