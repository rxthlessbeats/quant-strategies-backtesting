"use client";

import { ArrowDownRight, ArrowUpRight, ArrowRight, MoveUpRight, Play, Pause, RotateCcw, SlidersHorizontal, Activity, Layers3 } from "lucide-react";
import Link from "next/link";
import { useEffect, useMemo, useRef, useState } from "react";
import { fetchChart, fetchIndexMetrics } from "@/lib/api";
import type { AnalysisChartResponse, IndexMetricItem } from "@/lib/types";
import { latestQuoteFromBars } from "@/lib/quote-from-bars";
import { chartViewStart } from "@/lib/chart-view";

const instruments = [
  { symbol: "SPY", name: "S&P 500 ETF" },
  { symbol: "NVDA", name: "NVIDIA" },
  { symbol: "AAPL", name: "Apple" },
  { symbol: "MSFT", name: "Microsoft" },
];
const periods = { "1M": 1, "3M": 3, "6M": 6, "1Y": 12 };
const price = (value: number) => value.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
const date = (timestamp: number) => new Date(timestamp * 1000).toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric", timeZone: "UTC" });

export default function MarketOverview() {
  const [symbol, setSymbol] = useState("SPY");
  const [period, setPeriod] = useState<keyof typeof periods>("6M");
  const [data, setData] = useState<AnalysisChartResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [retry, setRetry] = useState(0);
  const [metrics, setMetrics] = useState<IndexMetricItem[]>([]);
  const [metricsError, setMetricsError] = useState(false);
  const [metricsLoading, setMetricsLoading] = useState(true);
  const [cursor, setCursor] = useState<number | null>(null);
  const [progress, setProgress] = useState(100);
  const [playing, setPlaying] = useState(false);
  const lensRef = useRef<HTMLElement>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true); setError(null); setData(null); setPlaying(false); setProgress(100); setCursor(null);
    fetchChart({ symbol, interval: "1d" }).then(response => {
      if (!cancelled) setData(response);
    }).catch(e => {
      if (!cancelled) setError(e instanceof Error ? e.message : "Market data could not be loaded.");
    }).finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [symbol, retry]);

  useEffect(() => {
    let cancelled = false;
    setMetricsLoading(true); setMetricsError(false);
    fetchIndexMetrics().then(response => { if (!cancelled) setMetrics(response.metrics); })
      .catch(() => { if (!cancelled) setMetricsError(true); })
      .finally(() => { if (!cancelled) setMetricsLoading(false); });
    return () => { cancelled = true; };
  }, [retry]);

  const bars = useMemo(() => {
    const valid = (data?.bars ?? []).filter(bar => Number.isFinite(bar.close));
    const start = chartViewStart(valid[valid.length - 1]?.timestamp ?? 0, period);
    return valid.filter(bar => bar.timestamp >= start);
  }, [data, period]);
  const points = useMemo(() => {
    if (!bars.length) return [];
    const min = Math.min(...bars.map(bar => bar.close));
    const max = Math.max(...bars.map(bar => bar.close));
    return bars.map((bar, i) => ({ x: 32 + i / Math.max(1, bars.length - 1) * 736, y: 232 - (bar.close - min) / (max - min || 1) * 166 }));
  }, [bars]);
  const visibleCount = Math.max(1, Math.round(bars.length * progress / 100));
  const currentIndex = cursor ?? Math.min(bars.length - 1, visibleCount - 1);
  const current = bars[currentIndex];
  const currentPoint = points[currentIndex];
  const path = points.slice(0, visibleCount).map((point, i) => `${i ? "L" : "M"}${point.x.toFixed(2)},${point.y.toFixed(2)}`).join(" ");
  const change = current && bars[0]?.close ? (current.close / bars[0].close - 1) * 100 : null;
  const quote = latestQuoteFromBars(data?.bars ?? []);
  const maxVolume = Math.max(1, ...bars.map(bar => bar.volume));

  useEffect(() => { setPlaying(false); setProgress(100); setCursor(null); }, [period]);
  useEffect(() => {
    if (!playing) return;
    const timer = window.setInterval(() => setProgress(value => Math.min(100, value + 0.8)), 60);
    return () => window.clearInterval(timer);
  }, [playing]);
  useEffect(() => { if (progress >= 100) setPlaying(false); }, [progress]);

  return (
    <div className="overview-page">
      <section className="perspective-hero" aria-labelledby="overview-title">
        <div className="hero-copy">
          <p className="context-label"><span className="status-dot" /> Your market, in perspective</p>
          <h1 id="overview-title">Less noise.<br />More insight.</h1>
          <p className="hero-description">Get curious. See the patterns. Explore the companies and signals behind every move.</p>
          <Link href="/chart" className="primary-link">Open workspace <ArrowUpRight size={20} /></Link>
          <div className="hero-note"><span className="note-orbit" aria-hidden="true"><Activity size={20} /></span><p>A clearer picture starts here.<br /><span>Daily prices. Deeper research.</span></p></div>
        </div>
        <section className="market-lens" ref={lensRef} aria-label="Interactive price history"
          onPointerMove={e => {
            if (e.pointerType !== "mouse" || window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
            const rect = e.currentTarget.getBoundingClientRect();
            e.currentTarget.style.setProperty("--pointer-x", `${(e.clientX - rect.left) / rect.width * 100}%`);
            e.currentTarget.style.setProperty("--pointer-y", `${(e.clientY - rect.top) / rect.height * 100}%`);
            e.currentTarget.style.setProperty("--terrain-shift", `${((e.clientX - rect.left) / rect.width - .5) * 8}px`);
          }}>
          <div className="lens-topline"><span><span className="lens-dot" /> Market lens</span><span>Daily closing prices</span></div>
          <div className="instrument-tabs" aria-label="Choose instrument">
            {instruments.map(item => <button key={item.symbol} type="button" aria-pressed={symbol === item.symbol} onClick={() => setSymbol(item.symbol)}>{item.symbol}</button>)}
          </div>
          <div className="lens-heading">
            <div><span className="lens-company">{instruments.find(item => item.symbol === symbol)?.name}</span><p className="lens-price">{current ? `$${price(current.close)}` : loading ? "Loading…" : "—"}</p></div>
            <div className="lens-return">{change !== null && <><span className={change < 0 ? "negative" : "positive"}>{change > 0 ? "+" : ""}{change.toFixed(2)}% <ArrowUpRight size={16} /></span><small>{cursor !== null || progress < 100 ? "From period start" : `${period} return`}</small></>}</div>
          </div>
          <div className="lens-visual">
            <span className="lens-watermark" aria-hidden="true">{symbol}</span>
            {points.length > 0 ? <svg key={`${symbol}-${period}`} viewBox="0 0 800 330" preserveAspectRatio="none" className="landscape-chart" role="img" aria-label={`${symbol} ${period} closing price history. Use the timeline below to inspect daily prices.`}
              onPointerMove={e => {
                if (e.pointerType !== "mouse") return;
                const rect = e.currentTarget.getBoundingClientRect();
                setCursor(Math.min(visibleCount - 1, Math.max(0, Math.round(((e.clientX - rect.left) / rect.width * 800 - 32) / 736 * (bars.length - 1)))));
              }} onPointerLeave={() => setCursor(null)}>
              <defs>
                <linearGradient id="landscape-fill" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#aa9cff" stopOpacity=".26" /><stop offset="100%" stopColor="#aa9cff" stopOpacity="0" /></linearGradient>
                <linearGradient id="trace-color"><stop stopColor="#b8a9ff" /><stop offset="100%" stopColor="#f1eeff" /></linearGradient>
              </defs>
              <g className="terrain-grid" stroke="#c9bdff" strokeOpacity=".09" fill="none">
                {[0, 1, 2, 3, 4].map(i => <path key={i} d={`M0 ${80 + i * 52}H800`} />)}
                {[0, 1, 2, 3, 4, 5, 6, 7, 8].map(i => <path key={i} d={`M${i * 100} 0V330`} />)}
              </g>
              <path d={`${path} L${points[visibleCount - 1]?.x ?? 32},330 L32,330Z`} fill="url(#landscape-fill)" />
              <g className="terrain-ridges" fill="none" stroke="#ae98ff" strokeWidth="1">
                {Array.from({ length: 12 }, (_, i) => <path key={i} d={path} transform={`translate(${i * 3},${(i + 1) * 7})`} opacity={0.16 - i * 0.009} />)}
              </g>
              <g className="volume-field" fill="#c8bbff">
                {bars.slice(0, visibleCount).map((bar, index) => index % Math.max(1, Math.floor(bars.length / 80)) === 0 ? <rect key={bar.timestamp} x={points[index].x} y={330 - bar.volume / maxVolume * 42} width="3" height={bar.volume / maxVolume * 42} opacity=".19" /> : null)}
              </g>
              <path className="price-trace" d={path} stroke="url(#trace-color)" strokeWidth="2.4" fill="none" vectorEffect="non-scaling-stroke" pathLength="1" />
              {currentPoint && <g className="chart-cursor"><line x1={currentPoint.x} y1="12" x2={currentPoint.x} y2="330" stroke="#e5dcff" strokeOpacity=".3" strokeDasharray="4 5" /><circle cx={currentPoint.x} cy={currentPoint.y} r="7" fill="#eee8ff" fillOpacity=".16" /><circle cx={currentPoint.x} cy={currentPoint.y} r="3.5" fill="#f4f0ff" /></g>}
            </svg> : <div className="lens-empty" role="status"><Activity size={32} /><p>{loading ? "Finding the bigger picture…" : "Price history is unavailable"}</p>{error && <><span>{error}</span><button type="button" onClick={() => setRetry(value => value + 1)}>Try again <RotateCcw size={14} /></button></>}</div>}
          </div>
          <div className="lens-date"><span>{current ? date(current.timestamp) : "Historical market data"}</span><span>{cursor === null ? "Explore the timeline" : "Closing price"}</span></div>
          <div className="lens-controls">
            <button type="button" className="replay-button" disabled={bars.length < 2} aria-label={playing ? "Pause price replay" : "Replay price history"} onClick={() => { setCursor(null); if (playing) setPlaying(false); else { if (progress >= 100) setProgress(1); setPlaying(true); } }}>{playing ? <Pause size={14} /> : <Play size={14} />}<span>{playing ? "Pause" : "Replay"}</span></button>
            <input type="range" min="1" max="100" value={progress} disabled={bars.length < 2} aria-label="Price history timeline" aria-valuetext={current ? `${date(current.timestamp)}, $${price(current.close)}` : "No price data"} onChange={e => { setPlaying(false); setCursor(null); setProgress(Number(e.target.value)); }} />
            <div className="period-tabs" aria-label="History period">{Object.keys(periods).map(value => <button key={value} type="button" aria-pressed={period === value} onClick={() => setPeriod(value as keyof typeof periods)}>{value}</button>)}</div>
          </div>
          <div className="lens-bottomline"><span>{quote ? `Latest daily close: ${date(quote.asOf)}` : "Prices are not real time"}</span><Link href={`/chart?symbol=${symbol}`}>Explore {symbol} <MoveUpRight size={13} /></Link></div>
        </section>
      </section>

      <section className="market-snapshot" aria-labelledby="snapshot-title">
        <div className="snapshot-heading"><h2 id="snapshot-title">The wider market</h2><span>Change from previous close</span></div>
        {metricsError || (!metricsLoading && metrics.length === 0) ? <div className="inline-empty" role="status">Market snapshot unavailable. <button type="button" onClick={() => setRetry(value => value + 1)}>Try again</button></div> : <div className="index-grid">
          {metricsLoading ? ["SPX", "NASDAQ", "Russell 2000", "SOX"].map(label => <div className="index-item" key={label}><span className="index-name">{label}</span><div className="skeleton-number" /><span className="index-asof">Loading market data…</span></div>) : metrics.map(metric => <Link key={metric.id} href={`/chart?symbol=${encodeURIComponent(metric.symbol)}`} className="index-item">
            <span className="index-name">{metric.label}<MoveUpRight size={14} /></span><div className="index-value"><strong>{price(metric.price)}</strong><span className={metric.change < 0 ? "negative" : "positive"}>{metric.change > 0 ? "+" : ""}{(metric.change * 100).toFixed(2)}%{metric.change < 0 ? <ArrowDownRight size={14} /> : <ArrowUpRight size={14} />}</span></div><span className="index-asof">As of {metric.as_of}</span>
          </Link>)}
        </div>}
      </section>

      <section className="research-section" aria-labelledby="research-title">
        <div className="section-intro"><div><p className="context-label">Make it your own</p><h2 id="research-title">Follow your curiosity.</h2></div><p>One company. A thousand perspectives.<br />Start with a name you know.</p></div>
        <div className="company-shortcuts">
          {[{ symbol: "NVDA", name: "NVIDIA", sector: "Semiconductors", initials: "nv", className: "nvidia" }, { symbol: "AAPL", name: "Apple", sector: "Consumer technology", initials: "a", className: "apple" }, { symbol: "MSFT", name: "Microsoft", sector: "Software & cloud", initials: "m", className: "microsoft" }].map(company => <Link href={`/chart?symbol=${company.symbol}`} className="company-shortcut" key={company.symbol}>
            <div className={`company-monogram ${company.className}`} aria-hidden="true">{company.initials}</div><div><h3>{company.name}</h3><span>{company.symbol} <span className="shortcut-divider">/</span> {company.sector}</span></div><ArrowUpRight size={22} />
          </Link>)}
        </div>
        <div className="research-links">
          <Link href="/chart" className="research-link"><span className="research-icon"><SlidersHorizontal size={22} /></span><div><h3>Find your signal.</h3><p>Layer indicators. Compare performance. Look closer.</p></div><ArrowRight size={24} /></Link>
          <Link href="/indicators" className="research-link"><span className="research-icon"><Layers3 size={22} /></span><div><h3>Know your toolkit.</h3><p>Explore available signals and their parameters.</p></div><ArrowRight size={24} /></Link>
        </div>
      </section>
    </div>
  );
}
