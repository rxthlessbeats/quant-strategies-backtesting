import { Suspense } from "react";
import ChartPageClient from "@/components/trading/chart-page-client";

export default function ChartPage() {
  return (
    <Suspense
      fallback={
        <div className="p-8 text-muted-foreground">Loading chart…</div>
      }
    >
      <ChartPageClient />
    </Suspense>
  );
}
