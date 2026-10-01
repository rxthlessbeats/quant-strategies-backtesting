import { Activity, ArrowUpRight, RotateCcw, Check, X } from "lucide-react";
import Link from "next/link";
import { fetchHealth, getApiBaseUrl } from "@/lib/api";

export const dynamic = "force-dynamic";

export default async function HealthPage() {
  let status: string | null = null;
  let error: string | null = null;
  try { status = (await fetchHealth()).status; }
  catch (e) { error = e instanceof Error ? e.message : "Unable to reach the market data service."; }
  const ok = status === "ok";
  const checkedAt = new Date().toLocaleString("en-US", { timeZone: "America/New_York", month: "short", day: "numeric", hour: "numeric", minute: "2-digit", timeZoneName: "short" });
  return (
    <div className="system-page">
      <div className="system-intro"><p className="context-label">Service status</p><h1>Behind the view.</h1><p>A quick check on the service that powers your research.</p></div>
      <section className={`system-status ${ok ? "is-healthy" : "is-unavailable"}`} aria-labelledby="status-title">
        <div className="status-orbit" aria-hidden="true"><span /><span /><Activity size={48} /></div>
        <div className="status-content"><span className={`service-badge ${ok ? "positive" : "negative"}`}>{ok ? <Check size={13} /> : <X size={13} />}{ok ? "Operational" : "Connection unavailable"}</span><h2 id="status-title">{ok ? "Ready when you are." : "The connection needs a moment."}</h2><p>{ok ? "The API is healthy and responding. You can explore charts, indicators, and company research." : error || `The service returned status: ${status || "unknown"}. Check again in a moment.`}</p><div className="status-actions"><Link href="/chart" className="primary-link">Open workspace <ArrowUpRight size={18} /></Link><a href="/health" className="refresh-status"><RotateCcw size={14} /> Check again</a></div></div>
      </section>
      <dl className="system-details"><div><dt>API status</dt><dd>{status || "Unavailable"}</dd></div><div><dt>Last checked</dt><dd>{checkedAt}</dd></div><div><dt>Public API path</dt><dd><code>{getApiBaseUrl()}</code></dd></div><div><dt>Health endpoint</dt><dd><code>{getApiBaseUrl()}/health</code></dd></div></dl>
      <p className="system-note">This check confirms the API connection. Individual market data providers may have their own availability or delays.</p>
    </div>
  );
}
