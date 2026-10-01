# RookieTrader Frontend

A custom market research interface with interactive price-history replay, a technical workspace, and a searchable indicator library. Stock candlesticks use [TradingView Lightweight Charts](https://www.tradingview.com/lightweight-charts/); financial charts use VisActor. Some UI primitives originated in [visactor-next-template](https://github.com/mengxi-ream/visactor-next-template).

## Node.js version

Next.js 15 requires **Node.js 18.18+** (recommended: **20 LTS** or **22 LTS**).

Check your version:

```cmd
node -v
```

If you see `v18.17.1` or lower, upgrade:

1. **Installer (simplest):** https://nodejs.org/ — download **20 LTS** or **22 LTS**, run the installer, then open a **new** terminal and run `node -v` again.
2. **nvm-windows:** https://github.com/coreybutler/nvm-windows — then:
   ```cmd
   nvm install 20
   nvm use 20
   ```

This repo includes `.nvmrc` set to `20` for nvm/fnm users.

## Setup

```cmd
cd frontend
npm install
```

Copy the environment file and set `API_SECRET` to the same value as the backend. `API_URL` defaults to the local API:

```cmd
copy .env.example .env.local
```

## Run

Prefer the repo-root command `uv run start`. Frontend only:

```cmd
cd frontend
npm run dev
```

Open http://localhost:3000

## Pages

| Route | Description |
|-------|-------------|
| `/` | Interactive market lens, index snapshot, and company shortcuts |
| `/chart` | Price action, market & valuation, performance comparisons, analysts, and financials & earnings |
| `/indicators` | 21 indicators grouped into Trend, Momentum, Volatility, and Volume, with search, parameters, and workspace links |
| `/health` | Backend health, connection details, and refresh |

## Environment

| Variable | Default |
|----------|---------|
| `API_URL` | FastAPI origin (`http://127.0.0.1:8000` locally). Server-only. |
| `API_SECRET` | Shared with the backend. Server-only; never `NEXT_PUBLIC_`. |

## Browser verification

Start the backend and frontend, then run from the repository root:

```sh
NODE_PATH=/path/to/browser-tools/node_modules node scripts/check-redesign.cjs
```

The browser tooling directory needs Playwright (with Chromium) and `@axe-core/playwright`. They are separate from application dependencies. Set `TRADING_URL` to test another local frontend origin; screenshots and audit results go to `TRADING_REVIEW_DIR` (default `/tmp/trading-review`).

The check exercises actual API responses, replay, search, indicator changes, presets, benchmark and period comparisons, analyst target focus, financial statement expansion, earnings views, failure recovery, light/dark themes, reduced motion, and accessibility at 320–1440px. It also checks UTC date ranges, empty and parameter-free presets, category counts, stochastic settings, and all 21 indicators together. Oscillators and volume signals have separate panes; price overlays stay on the candle chart.
