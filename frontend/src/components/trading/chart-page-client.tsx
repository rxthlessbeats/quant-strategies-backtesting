"use client";

import Container from "@/components/container";
import ChartWorkspace from "@/components/trading/chart-workspace";

export default function ChartPageClient() {
  return (
    <Container className="workspace-page">
      <ChartWorkspace />
    </Container>
  );
}
