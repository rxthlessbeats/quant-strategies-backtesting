---
name: wiki-query
description: Answer codebase questions using the project wiki first, then verify against current code and cite evidence. Use when the user asks for architecture, behavior, API, data flow, or operational knowledge from the repo.
disable-model-invocation: true
---

# Wiki Query

Use this skill to answer questions from the codebase wiki with code-verified citations.

## Required Workflow
1. Read `wiki/index.md` first.
2. Read the most relevant pages from `wiki/pages/`.
3. Verify important claims against current code paths and/or runtime evidence.
4. Answer with citations to:
   - wiki page paths
   - repository code paths used for verification
5. Mark uncertainty explicitly.

## Durable Knowledge Rule
If the answer reveals durable knowledge not captured in the wiki:
1. Update an existing canonical page or add a new one only when needed.
2. Update `wiki/index.md` if structure changed.
3. Append a query entry to `wiki/log.md` documenting page changes.

## Output Shape
- Direct answer.
- Evidence used (wiki pages + code paths).
- Optional follow-up wiki updates (if performed).
