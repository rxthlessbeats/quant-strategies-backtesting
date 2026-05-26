import Container from "@/components/container";
import { fetchIndexMetrics } from "@/lib/api";
import MetricCard from "./components/metric-card";

export default async function Metrics() {
  let metrics: Awaited<ReturnType<typeof fetchIndexMetrics>>["metrics"] = [];
  let error: string | null = null;

  try {
    const response = await fetchIndexMetrics();
    metrics = response.metrics;
  } catch (e) {
    error = e instanceof Error ? e.message : "Failed to load market metrics";
  }

  return (
    <Container className="grid grid-cols-1 gap-y-6 border-b border-border py-4 phone:grid-cols-2 laptop:grid-cols-4">
      {error ? (
        <div className="text-sm text-destructive">{error}</div>
      ) : (
        metrics.map((metric) => <MetricCard key={metric.id} {...metric} />)
      )}
    </Container>
  );
}
