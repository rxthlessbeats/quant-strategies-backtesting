"use client";

import { TopNav } from "@/components/nav";
import Container from "@/components/container";
import ChartWorkspace from "@/components/trading/chart-workspace";

export default function ChartPageClient() {
  return (
    <>
      <TopNav title="Chart" />
      <Container className="py-6">
        <ChartWorkspace />
      </Container>
    </>
  );
}
