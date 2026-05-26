# RookieTrader Frontend

Next.js UI based on [visactor-next-template](https://github.com/mengxi-ream/visactor-next-template) (layout and Shadcn only). Stock K-line charts use [TradingView Lightweight Charts](https://www.tradingview.com/lightweight-charts/).

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

Copy environment (optional — default is local API):

```cmd
copy .env.example .env.local
```

## Run

Start the FastAPI backend first (see [../backend/README.md](../backend/README.md)):

```cmd
cd backend
uvicorn app.main:app --reload
```

Then the frontend:

```cmd
cd frontend
npm run dev
```

Open http://localhost:3000

## Pages

| Route | Description |
|-------|-------------|
| `/` | Home |
| `/chart` | OHLCV candlesticks, volume, indicator overlays |
| `/indicators` | Indicator catalog from API |
| `/health` | Backend health check |

## Environment

| Variable | Default |
|----------|---------|
| `NEXT_PUBLIC_API_URL` | `http://127.0.0.1:8000` |
