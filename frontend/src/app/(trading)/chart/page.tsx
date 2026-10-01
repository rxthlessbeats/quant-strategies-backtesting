import { Suspense } from "react";
import ChartPageClient from "@/components/trading/chart-page-client";

export const dynamic = "force-dynamic";

export default function ChartPage() {
  return (
    <Suspense
      fallback={
        <div className="workspace-loading" role="status">Preparing your workspace…</div>
      }
    >
      <ChartPageClient />
    </Suspense>
  );
}
