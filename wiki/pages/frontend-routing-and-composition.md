# Frontend Routing and Composition

## Claims
- Root layout composes persistent nav shell and wraps app with theme/Jotai providers.
- Trading routes are grouped under `(trading)` with dashboard/chart/indicators/health pages.
- Navigation labels/routes are defined centrally in `site.tsx`.
- `/chart` route delegates to a client component via suspense boundary.

## Evidence
- Root layout and shell: `frontend/src/app/layout.tsx`.
- Providers: `frontend/src/app/providers.tsx`.
- Route group layout: `frontend/src/app/(trading)/layout.tsx`.
- Route pages: `frontend/src/app/(trading)/page.tsx`, `frontend/src/app/(trading)/chart/page.tsx`, `frontend/src/app/(trading)/indicators/page.tsx`, `frontend/src/app/(trading)/health/page.tsx`.
- Navigation config: `frontend/src/config/site.tsx`.
- Nav renderer: `frontend/src/components/nav/side-nav/components/navigation.tsx`.

## Open Questions
- Should dashboard route remain in same knowledge taxonomy as trading chart flow?
- Should route docs include SSR/client boundary notes per page?

## Links
- [frontend-data-flow-and-state](frontend-data-flow-and-state.md)
- [architecture-overview](architecture-overview.md)
