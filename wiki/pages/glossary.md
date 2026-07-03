# Glossary

## Claims
- This glossary maps core trading_rookie terms to concrete code implementations.

## Evidence
- Terms are derived from current symbols/types/modules in backend and frontend code.

## Open Questions
- Should we add a second glossary section for non-technical stakeholder wording?

## Links
- [architecture-overview](architecture-overview.md)
- [backend-api-surface](backend-api-surface.md)
- [frontend-data-flow-and-state](frontend-data-flow-and-state.md)
- [manual-ops-prompts](manual-ops-prompts.md)

## Terms
- **OHLCV bars**: Time-series candles stored in `Bar` (`backend/app/db/models.py`).
- **FetchMeta**: Per-symbol/per-interval cache freshness metadata (`backend/app/db/models.py`).
- **MarketDataModule**: Cached quote-summary module payload + refresh metadata (`backend/app/db/models.py`, `backend/app/services/market_data_service.py`).
- **ChartQuery / AnalysisChartQuery**: Request models for bar and indicator chart endpoints (`backend/app/schemas/requests.py`).
- **Indicator catalog**: Registry-backed indicator listing returned by `/api/v1/analysis/indicators` (`backend/app/services/indicator_service.py`).
- **Indicator query string**: Comma-separated specs passed to `/api/v1/analysis/chart` (e.g. `sma:20,rsi:14`), built by `buildIndicatorsQuery()` in `frontend/src/lib/indicator-utils.ts`.
- **Wiki skill (`/wiki-ingest`)**: Project skill for persisting code-backed knowledge into `wiki/pages/` (`.cursor/skills/wiki-ingest/SKILL.md`).
- **Wiki skill (`/wiki-query`)**: Project skill for wiki-first Q&A with code verification (`.cursor/skills/wiki-query/SKILL.md`).
- **Wiki skill (`/wiki-lint`)**: Project skill for wiki quality/drift audits (`.cursor/skills/wiki-lint/SKILL.md`).
- **DataSource (`cache`/`fetch`)**: Response source metadata used by chart consumers (`frontend/src/lib/types.ts`).
- **ChartWorkspace**: Main client orchestrator for chart screen (`frontend/src/components/trading/chart-workspace.tsx`).
- **IndicatorSelection**: Frontend indicator tuple (`slotId`, `id`, `params`) managed in `frontend/src/lib/indicator-utils.ts`.
- **Benchmark comparison**: Symbol-vs-benchmark return service (`backend/app/services/performance_comparison_service.py`).
